"""Apply trained adapter parameters to the three interface surfaces.

Kept OUT of `interface/` and `sim/` on purpose. Those modules are Phase 4's
frozen interface (config hash `04be9dec...`) and every number in
`docs/closed_loop/` was computed with them; the trainable variants live
here, clearly ours, so the frozen path stays byte-for-byte what it was and
`tests/test_frozen_motor_interface.py` keeps meaning something.

Three surfaces, and the connectome between them is never touched:

    body -> AdapterSensoryEncoder -> [connectome] -> adapter_joint_targets -> body
                                          ^
                                   command_current()
"""

from __future__ import annotations

import numpy as np

from fly_robot.adapter.parameters import PAIRS, SEGMENTS, AdapterParams
from fly_robot.interface.joint_to_sensory_neuron import (
    CHORDOTONAL, STAGE_A_POSITION_REF_RAD, STAGE_A_VELOCITY_REF_RAD_S,
    SensoryEncoder, normalised_position,
)
from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, LEG_NAME_TO_FLYGYM_PREFIX, _dof_name_for_pair,
)

# The trainable encoder runs on the Stage A rig, so it uses the scales
# measured there, not Phase 4's harness-derived ones.
VELOCITY_REF_RAD_S = STAGE_A_VELOCITY_REF_RAD_S
POSITION_REF_RAD = STAGE_A_POSITION_REF_RAD

# Verified by query against the simulated MANC table, not assumed —
# see docs/trained_adapter/DESIGN.md §2.2.
DNG100_ROWS = {"LHS": (59,), "RHS": (282,)}
MDN_ROWS = {"LHS": (3075, 4043), "RHS": (3413, 3952)}

_SEG_INDEX = {seg: i for i, seg in enumerate(SEGMENTS)}
_PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}


class AdapterSensoryEncoder(SensoryEncoder):
    """Phase 4's encoder with every constant made a trainable parameter.

    Differences from the frozen encoder, all of them parameters rather than
    new mechanisms:
      * gain, offset and saturation cap are per (class, segment) instead of
        one global gain;
      * the position/velocity mix is trainable instead of fixed at 0.5/0.5;
      * the hair-plate limit threshold is trainable instead of fixed at
        0.75;
      * chordotonal range-fractionation is always on, with trainable width
        and band-centre offset (Phase 4 had it as a separate encoder mode
        with fixed constants).

    The OFFSET is the parameter Phase 4's own RESULTS.md §5 argued for:
    it shifts a whole class's current so that *typical* drive can sit near
    threshold, instead of only transient peaks ever crossing it.

    Locality is inherited unchanged: a group keyed (segment, side, *) is
    driven only by the joints of the one leg (segment, side) maps to.
    """

    def __init__(self, groups, jointdof_order, neutral_angles_by_name,
                 params: AdapterParams, leg_permutation=None):
        super().__init__(groups, jointdof_order, neutral_angles_by_name,
                         gain=1.0, leg_permutation=leg_permutation,
                         mode="deviation")
        self.params = params
        rng = np.random.default_rng(20260920)  # same fixed draw as Phase 4
        self._preferred = {}
        for key in sorted(groups.indices_by_group):
            n = len(groups.indices_by_group[key])
            self._preferred[key] = (rng.uniform(0.0, 1.0, n)
                                    if key[2] == CHORDOTONAL else None)

    def _seg(self, key):
        return _SEG_INDEX[key[0]]

    def channel_drives(self, joint_angles, joint_velocities) -> dict:
        """Per-group drive in [0,1], with the trainable position/velocity
        mix and hair-plate threshold."""
        p = self.params
        drives = {}
        for key in self.groups.indices_by_group:
            segment, side, subclass = key
            s = _SEG_INDEX[segment]
            source = self.leg_permutation.get((segment, side), (segment, side))
            ids = self._joint_ids[source]
            if subclass == CHORDOTONAL:
                i = ids["femur_tibia"]
                if i is None:
                    drives[key] = 0.0
                    continue
                pos = normalised_position(
                    float(joint_angles[i]),
                    self._reference.get(ids["femur_tibia_name"], 0.0), self.mode, POSITION_REF_RAD)
                vel = float(np.clip(abs(float(joint_velocities[i]))
                                    / VELOCITY_REF_RAD_S, 0.0, 1.0))
                mix = float(p.chord_mix[s])
                drives[key] = float(np.clip(mix * pos + (1.0 - mix) * vel, 0.0, 1.0))
            else:
                i = ids["thorax_coxa"]
                if i is None:
                    drives[key] = 0.0
                    continue
                pos = normalised_position(
                    float(joint_angles[i]),
                    self._reference.get(ids["thorax_coxa_name"], 0.0), self.mode,
                    POSITION_REF_RAD)
                thr = float(p.hp_threshold[s])
                drives[key] = 0.0 if pos <= thr else float((pos - thr) / (1.0 - thr))
        return drives

    def _neuron_responses(self, key, drive, joint_angles, joint_velocities):
        """Chordotonal neurons are range-fractionated with trainable width
        and centre offset; hair plates respond uniformly across the pool."""
        n = len(self.groups.indices_by_group[key])
        preferred = self._preferred.get(key)
        if preferred is None:
            return np.full(n, drive, dtype=np.float64)

        p = self.params
        segment, side, _sub = key
        s = _SEG_INDEX[segment]
        source = self.leg_permutation.get((segment, side), (segment, side))
        ids = self._joint_ids[source]
        i = ids["femur_tibia"]
        if i is None:
            return np.zeros(n, dtype=np.float64)

        pos = normalised_position(
            float(joint_angles[i]),
            self._reference.get(ids["femur_tibia_name"], 0.0), self.mode,
            POSITION_REF_RAD)
        vel = float(np.clip(abs(float(joint_velocities[i])) / VELOCITY_REF_RAD_S,
                            0.0, 1.0))
        sigma = float(p.chord_sigma[s])
        centres = np.clip(preferred + float(p.chord_band[s]), 0.0, 1.0)
        tuned = np.exp(-((pos - centres) ** 2) / (2 * sigma ** 2))
        mix = float(p.chord_mix[s])
        # Movement stays un-fractionated: position and movement tuning belong
        # to different FeCO subtypes in the fly (claw vs hook), so
        # fractionating the movement signal too would assert something the
        # source does not.
        return np.clip(mix * tuned + (1.0 - mix) * vel, 0.0, 1.0)

    def input_current(self, joint_angles, joint_velocities, drives=None):
        """Per-neuron current, with per-class gain, offset and cap."""
        if drives is None:
            drives = self.channel_drives(joint_angles, joint_velocities)
        p = self.params
        current = np.zeros(self.groups.n_neurons, dtype=np.float64)
        for key, idxs in self.groups.indices_by_group.items():
            if not idxs:
                continue
            segment, side, subclass = key
            s = _SEG_INDEX[segment]
            if subclass == CHORDOTONAL:
                gain, offset, cap = p.chord_gain[s], p.chord_offset[s], p.chord_cap[s]
            else:
                gain, offset, cap = p.hp_gain[s], p.hp_offset[s], p.hp_cap[s]
            side_scale = (float(p.lr_sensory_ratio) if side == "LHS"
                          else 1.0 / float(p.lr_sensory_ratio))
            responses = self._neuron_responses(key, drives[key], joint_angles,
                                               joint_velocities)
            neuron_current = (float(gain) * responses + float(offset)) * side_scale
            neuron_current *= self.groups.scale_by_group[key]
            # Rectified: a sensory neuron's SIGN comes from the connectome,
            # never from us (PREREGISTRATION.md §B0), so the interface may
            # only ever inject non-negative current.
            current[idxs] = np.clip(neuron_current, 0.0, float(cap))
        return current


# Adhesion gating -- OUR INTERFACE ADDITION, and the signal is the
# connectome's RHYTHM, not its stance anatomy. That distinction is real and
# is recorded rather than glossed.
#
# The anatomically correct signal would be Pugliese's own motor modules,
# which literally name two pools "coxa stance" and "coxa swing". Measured on
# the untrained connectome under DNg100 drive, those pools are near-silent:
# each fires in only ~19% of (leg, side, replicate) pools, and "substrate
# grip" and "tarsus control" -- the two modules semantically closest to
# adhesion -- fire in 0%. A gate built on them sticks: three legs always-off,
# one always-on, 0-1 transitions per 2 s where stepping needs ~44.
#
# So the gate uses the per-leg SUMMED motor rate, the same signal Phase 4's
# AR(1) rhythm gate already found genuinely rhythmic in 3-5 of 6 legs,
# high-passed against its own 50 ms running mean so it oscillates about
# zero by construction:
#
#     adhere  <=>  w_seg * (rate - running_mean) > theta_seg
#
# `w_seg` may be negative: which half of the rhythm counts as stance is a
# modelling choice, and the optimiser is allowed to find the polarity
# rather than have us assert it.
#
# Adhesion is ON in stance and OFF in swing, matching what both baselines
# do. Each controller gates from its own phase signal -- see DESIGN.md
# section 10 for why that is the fair comparison.
ADHESION_TAU_S = 0.05

# The anatomical pools are still LOGGED every step, so training can be
# asked afterwards whether it ever recruited them.
ANATOMICAL_POOLS = ("coxa stance", "coxa swing", "substrate grip",
                    "tarsus control")

# FlyGym's own adhesion actuator order, from `fly.get_legs_order()`.
FLYGYM_LEG_ORDER = ("lf", "lm", "lh", "rf", "rm", "rh")
_PREFIX_TO_LEG = {v: k for k, v in LEG_NAME_TO_FLYGYM_PREFIX.items()}


def per_leg_summed_rate(rates, leg_rows):
    """(6,) summed motor-neuron rate per leg, in FlyGym's leg order."""
    out = np.zeros(len(FLYGYM_LEG_ORDER))
    for i, prefix in enumerate(FLYGYM_LEG_ORDER):
        rows = leg_rows.get(_PREFIX_TO_LEG[prefix], [])
        out[i] = float(np.sum(rates[rows])) if len(rows) else 0.0
    return out


def anatomical_pool_rates(rates, groups):
    """{module: (6,) mean rate per leg} for the anatomically named pools.

    Logged but NOT used by the gate. Keeping it lets RESULTS answer whether
    training ever recruits the pools the annotation calls stance, swing and
    grip -- which are near-silent in the untrained network.
    """
    out = {}
    for module in ANATOMICAL_POOLS:
        vals = np.zeros(len(FLYGYM_LEG_ORDER))
        for i, prefix in enumerate(FLYGYM_LEG_ORDER):
            seg, side = _PREFIX_TO_LEG[prefix]
            idx = groups.indices_by_group.get((seg, side, module), [])
            vals[i] = float(np.mean(rates[idx])) if idx else 0.0
        out[module] = vals
    return out


class AdhesionGate:
    """Stateful per-leg adhesion gate driven only by connectome output.

    Holds running statistics, so it must be constructed once per trial and
    stepped in lockstep with the neural model.

    **Scale-adaptive thresholding (2026-09-26).** The original gate compared
    `w * (rate - running_mean)` against an ABSOLUTE threshold. That was a
    real bug, and it is why no trained adapter ever produced a stepping
    gait: the per-leg signal amplitude varies by orders of magnitude between
    legs and replicates (std 0.0 to 18.6 on the same drive), so a fixed
    threshold sits either far above the signal (foot never grips) or far
    below it (foot never releases). Measured on trained candidates: duty
    cycles of 0.000 and 1.000, against 0.62-0.69 with ~12 Hz cycling for the
    controller that actually walks. Four of six legs never cycled at all.

    The threshold is now expressed in units of each leg's OWN running
    standard deviation:

        adhere  <=>  (rate - running_mean) > k * running_std

    so it cycles regardless of that leg's amplitude. `k` is the trained
    parameter (`adhesion_threshold`, reinterpreted): k=0 gives ~50% duty,
    k<0 more stance, k>0 more swing. `adhesion_weight`'s sign still allows
    the gate to invert.

    **This specifies no gait.** Each leg is thresholded against its own
    signal statistics only; nothing here says which legs step together, and
    no leg can see any other. The interleg pattern remains whatever the
    connectome produces. What it fixes is that a leg can now cycle at all.
    """

    def __init__(self, params: AdapterParams, dt: float = 0.001,
                 tau_s: float = ADHESION_TAU_S):
        self.params = params
        self.alpha = dt / tau_s
        self.baseline = None
        self.var = None
        self.w = np.array([float(params.adhesion_weight[
            _SEG_INDEX[_PREFIX_TO_LEG[p][0]]]) for p in FLYGYM_LEG_ORDER])
        self.theta = np.array([float(params.adhesion_threshold[
            _SEG_INDEX[_PREFIX_TO_LEG[p][0]]]) for p in FLYGYM_LEG_ORDER])

    def step(self, leg_rate):
        """(6,) adhesion in FlyGym leg order from (6,) per-leg summed rate."""
        leg_rate = np.asarray(leg_rate, dtype=float)
        if self.baseline is None:
            self.baseline = leg_rate.copy()
            self.var = np.zeros_like(leg_rate)
        deviation = leg_rate - self.baseline
        self.baseline = (1.0 - self.alpha) * self.baseline + self.alpha * leg_rate
        # Running variance of the same deviation the gate thresholds.
        self.var = (1.0 - self.alpha) * self.var + self.alpha * deviation ** 2
        scale = np.sqrt(self.var)
        # A leg with no rhythm has no scale to threshold against; it simply
        # does not grip, rather than latching on an arbitrary comparison.
        active = scale > 1e-9
        signal = np.where(active, self.w * deviation / np.where(active, scale, 1.0), 0.0)
        return np.where(active, signal > self.theta, 0.0).astype(float)


def command_current(params: AdapterParams, n_neurons: int, t: float,
                    pulse_start: float, pulse_end: float) -> np.ndarray:
    """Descending drive into command neurons that really exist in the net.

    Delivered as input current — exactly where Pugliese's own stimulation
    enters — so W is untouched. Ramped over `command_ramp_s` from onset.
    """
    out = np.zeros(n_neurons, dtype=np.float64)
    if not (pulse_start <= t <= pulse_end):
        return out
    ramp = min(1.0, (t - pulse_start) / max(float(params.command_ramp_s), 1e-9))
    asym = float(params.command_asymmetry)
    for side, rows in DNG100_ROWS.items():
        scale = (1.0 + asym) if side == "LHS" else (1.0 - asym)
        out[list(rows)] = float(params.dng100_level) * scale * ramp
    for _side, rows in MDN_ROWS.items():
        out[list(rows)] = float(params.mdn_level) * ramp
    return out


def adapter_joint_targets(rates: np.ndarray, groups, dof_index_by_name: dict,
                          neutral_angles: np.ndarray,
                          params: AdapterParams) -> np.ndarray:
    """Phase 4's antagonist rule with per-(pair, segment) gain, scale and
    offset:

        joint = neutral + offset + gain * tanh((rate_pos - rate_neg) / scale)

    Identical in form to the frozen rule; only the constants differ, and
    with the frozen constants it reduces to it exactly.
    """
    targets = neutral_angles.copy()
    for (segment, side), leg_prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        s = _SEG_INDEX[segment]
        for pair_name, (pos_module, neg_module, suffix) in zip(PAIRS, ANTAGONIST_PAIRS):
            k = _PAIR_INDEX[pair_name]
            dof_idx = dof_index_by_name.get(_dof_name_for_pair(leg_prefix, suffix))
            if dof_idx is None:
                continue
            pos_idxs = groups.indices_by_group.get((segment, side, pos_module), [])
            neg_idxs = groups.indices_by_group.get((segment, side, neg_module), [])
            if not pos_idxs and not neg_idxs:
                continue
            pos_rate = rates[pos_idxs].mean() if pos_idxs else 0.0
            neg_rate = rates[neg_idxs].mean() if neg_idxs else 0.0
            targets[dof_idx] = (neutral_angles[dof_idx]
                                + float(params.motor_offset[k, s])
                                + float(params.motor_gain[k, s])
                                * np.tanh((pos_rate - neg_rate)
                                          / float(params.motor_scale[k, s])))
    return targets


def motor_filter_alpha(params: AdapterParams, dof_index_by_name: dict,
                       neural_dt: float) -> np.ndarray:
    """Per-DOF first-order low-pass coefficient, from the per-segment tau.

    alpha = dt / (tau + dt); alpha -> 1 means no filtering. DOFs that no
    antagonist pair drives get alpha = 1 so they track exactly.
    """
    alpha = np.ones(len(dof_index_by_name))
    for (segment, _side), leg_prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        tau = float(params.motor_tau[_SEG_INDEX[segment]])
        a = neural_dt / (tau + neural_dt)
        for _pos, _neg, suffix in ANTAGONIST_PAIRS:
            idx = dof_index_by_name.get(_dof_name_for_pair(leg_prefix, suffix))
            if idx is not None:
                alpha[idx] = a
    return alpha
