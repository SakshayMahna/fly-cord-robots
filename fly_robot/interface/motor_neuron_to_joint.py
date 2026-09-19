"""Phase 3: map real Pugliese/MANC leg motor-neuron firing rates onto
FlyGym joint targets.

This is OUR ADDITION, not something read off the connectome — flagged per
this project's honesty rule. The real fly's motor neurons drive actual
muscles with real, specific mechanical actions; Pugliese's own annotation
only labels each motor neuron with a coarse *module* (e.g. "tibia flex"),
not a quantitative muscle model. Turning "module X's neurons are firing
at rate Y" into "joint angle should be Z" requires a real modeling choice
that isn't given by the data. What's below is the simplest defensible
choice, not a claim about fly biomechanics:

- Only 3 of Pugliese's 9 motor-neuron modules per leg have a clear
  antagonist pair with an obvious 1:1 FlyGym joint DOF:
    coxa swing / coxa stance       -> thorax-coxa pitch      (protraction/retraction)
    femur/tr extend / femur/tr flex -> coxa-trochanterfemur pitch (the "knee-adjacent" joint)
    tibia extend / tibia flex      -> trochanterfemur-tibia pitch (the "knee" joint)
  Each pair drives its joint as `neutral + gain * tanh((rate_pos - rate_neg) / rate_scale)`
  — a bounded, monotonic function of the *difference* between antagonist
  rates, which is a standard motor-control abstraction (net rate typically
  represents net muscle activation) but is our choice, not Pugliese's.
- The other 3 modules (femur reductor, substrate grip, tarsus control)
  are NOT mapped in this first version — no clear antagonist partner, and
  guessing a single-direction mapping for each felt more likely to
  mislead than help. Their FlyGym joint DOFs (coxa-trochanterfemur roll,
  tibia-tarsus1 pitch) stay at the neutral pose. This is a real
  limitation, not an oversight — noted in CHANGELOG.
- Roll/yaw DOFs at the thorax-coxa joint have no corresponding Pugliese
  motor-neuron module at all and also stay at neutral.

Leg naming: Pugliese/MANC uses (T1/T2/T3, LHS/RHS); FlyGym uses
lf/lm/lh/rf/rm/rh. T1=front, T2=middle, T3=hind; LHS=left, RHS=right.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEG_NAME_TO_FLYGYM_PREFIX = {
    ("T1", "LHS"): "lf", ("T1", "RHS"): "rf",
    ("T2", "LHS"): "lm", ("T2", "RHS"): "rm",
    ("T3", "LHS"): "lh", ("T3", "RHS"): "rh",
}

# (positive module, negative module, FlyGym DOF name suffix). The DOF name
# prefix (which leg) is filled in per-leg below.
ANTAGONIST_PAIRS = [
    ("coxa swing", "coxa stance", "coxa-pitch"),
    ("femur/tr extend", "femur/tr flex", "trochanterfemur-pitch"),
    ("tibia extend", "tibia flex", "tibia-pitch"),
]

DEFAULT_GAIN_RAD = 0.5   # ~29 degrees max deflection from neutral — arbitrary, not tuned

# tanh saturation scale. Originally 20.0 Hz (arbitrary, not tuned — this
# silently flattened the real signal into the near-linear/near-zero part
# of tanh). Measured directly against a real representative replicate
# (docs/logs/2026-09-19.md, Phase 3): actual antagonist-pair rate
# differences across all 18 mapped pairs peaked at 2.58 Hz, with several
# pairs in the 1.5-2.6 Hz range. 3.0 Hz means a typical strong (~2.5 Hz)
# asymmetry drives the joint close to full gain, while smaller real
# asymmetries still produce a proportionate response — grounded in the
# observed data range, not re-guessed.
DEFAULT_RATE_SCALE_HZ = 3.0


def _dof_name_for_pair(leg_prefix: str, suffix: str) -> str:
    """Reconstructs the exact FlyGym DOF name for a given leg + axis
    suffix. FlyGym's own naming (see tutorials/1a): `{parent}-{child}-{axis}`.
    """
    if suffix == "coxa-pitch":
        return f"c_thorax-{leg_prefix}_coxa-pitch"
    elif suffix == "trochanterfemur-pitch":
        return f"{leg_prefix}_coxa-{leg_prefix}_trochanterfemur-pitch"
    elif suffix == "tibia-pitch":
        return f"{leg_prefix}_trochanterfemur-{leg_prefix}_tibia-pitch"
    raise ValueError(f"Unknown suffix {suffix}")


@dataclass
class MotorNeuronGroups:
    """Row indices (into a Pugliese R array) of the motor neurons behind
    each (leg, side, module), and the FlyGym DOF name they map to (or
    None for unmapped modules)."""
    dof_name_by_group: dict  # (leg, side, module) -> flygym dof name or None
    indices_by_group: dict   # (leg, side, module) -> list[int] (row indices into R)


def build_motor_neuron_groups(circuit_csv: str, wtable: pd.DataFrame) -> MotorNeuronGroups:
    """circuit_csv: data/circuit_map/all_legs_circuit.csv (has manc_bodyId,
    leg, side, motor_module for role == leg_motor_neuron).
    wtable: the full-VNC wTable used for the Pugliese simulation that
    produced R — indices in the returned dict are row positions in this
    table, matching R's neuron axis.
    """
    circuit = pd.read_csv(circuit_csv)
    mn = circuit[circuit["role"] == "leg_motor_neuron"].dropna(subset=["motor_module"])

    bodyid_to_idx = {bid: i for i, bid in enumerate(wtable["bodyId"])}

    dof_name_by_group, indices_by_group = {}, {}
    for (leg, side), leg_group in mn.groupby(["leg", "side"]):
        leg_prefix = LEG_NAME_TO_FLYGYM_PREFIX[(leg, side)]
        for module, module_group in leg_group.groupby("motor_module"):
            key = (leg, side, module)
            idxs = [bodyid_to_idx[b] for b in module_group["manc_bodyId"] if b in bodyid_to_idx]
            indices_by_group[key] = idxs

            dof_name = None
            for pos, neg, suffix in ANTAGONIST_PAIRS:
                if module in (pos, neg):
                    dof_name = _dof_name_for_pair(leg_prefix, suffix)
                    break
            dof_name_by_group[key] = dof_name

    return MotorNeuronGroups(dof_name_by_group, indices_by_group)


def compute_joint_targets(
    R: np.ndarray,
    groups: MotorNeuronGroups,
    dof_order,  # list[JointDOF] from fly.get_actuated_jointdofs_order(...)
    neutral_angles: np.ndarray,
    gain_rad: float = DEFAULT_GAIN_RAD,
    rate_scale_hz: float = DEFAULT_RATE_SCALE_HZ,
) -> np.ndarray:
    """R: (n_neurons, n_timesteps) real firing rates from a Pugliese
    simulation (e.g. fly_robot.neural.run_pugliese_sim). Returns
    (n_timesteps, len(dof_order)) joint angle targets — neutral pose
    everywhere except the mapped antagonist-pair DOFs.
    """
    n_timesteps = R.shape[1]
    targets = np.tile(neutral_angles, (n_timesteps, 1)).astype(np.float64)

    dof_index_by_name = {d.name: i for i, d in enumerate(dof_order)}

    for leg, side in LEG_NAME_TO_FLYGYM_PREFIX:
        for pos_module, neg_module, suffix in ANTAGONIST_PAIRS:
            leg_prefix = LEG_NAME_TO_FLYGYM_PREFIX[(leg, side)]
            dof_name = _dof_name_for_pair(leg_prefix, suffix)
            dof_idx = dof_index_by_name.get(dof_name)
            if dof_idx is None:
                continue  # this DOF isn't actuated in the current body config

            pos_idxs = groups.indices_by_group.get((leg, side, pos_module), [])
            neg_idxs = groups.indices_by_group.get((leg, side, neg_module), [])
            if not pos_idxs and not neg_idxs:
                continue  # no motor neurons resolved for this pair — leave at neutral

            pos_rate = R[pos_idxs].mean(axis=0) if pos_idxs else np.zeros(n_timesteps)
            neg_rate = R[neg_idxs].mean(axis=0) if neg_idxs else np.zeros(n_timesteps)
            drive = np.tanh((pos_rate - neg_rate) / rate_scale_hz)
            targets[:, dof_idx] = neutral_angles[dof_idx] + gain_rad * drive

    return targets
