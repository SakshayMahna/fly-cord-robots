"""Map robot joint state onto real leg proprioceptor input current — the
closed-loop sensory interface.

Mirror image of `motor_neuron_to_joint.py`: that module turns motor-neuron
firing rates into joint targets, this one turns joint state back into
current injected into the leg sensory neurons that are already present in
the simulated connectome.

**This is OUR ADDITION**, flagged per the project's honesty rule. The
connectome supplies the neurons and all their outgoing wiring; what it
does not supply is a transfer function from "the tibia is at this angle"
to "these neurons receive this much current". That is a modelling choice,
and everything below is the simplest defensible one rather than a
measured fly transfer function.

What IS grounded in primary sources
-----------------------------------
*Which* signal each class carries:

- **Chordotonal organs -> femur-tibia joint.** The femoral chordotonal
  organ monitors that joint. Mamiya, Gurung & Tuthill, *Neuron* 100(3)
  636-650 (2018), abstract, verbatim: "one group of axons encodes tibia
  position (flexion/extension), another encodes movement direction, and a
  third encodes bidirectional movement and vibration frequency", and
  "proprioceptive stimuli from a single leg joint".

- **Hair plates -> thorax-coxa joint, as LIMIT detectors.** From
  "Proprioceptive limit detectors mediate sensorimotor control of the
  Drosophila leg" (bioRxiv 2025.05.15.654260), verbatim: "three hair
  plates are located at the junction between the coxa (Cx) and thorax",
  and "CxHP8 neurons encode the anterior limits of thorax-coxa joint
  angles". Hence a rectified code that is silent through most of the
  range, not a linear one.

- **Campaniform sensilla -> NOT MODELLED.** MANC annotates 9 of them
  across all six leg nerves (MaleCNS: 12); both datasets put their
  hundreds of campaniform sensilla in the wing and haltere nerves. There
  is no leg load channel to build, and none is invented. See
  `docs/closed_loop/SENSORY_MAP.md` §1.

What is OURS, and unavoidably so
--------------------------------
Neither dataset resolves chordotonal **subtypes** — types are opaque
systematic identifiers (`SNpp39`-`SNpp60`) and `flywireType` is empty — so
the position / direction / vibration split that really exists in the fly
cannot be reconstructed. One combined position-and-movement signal
therefore drives each leg's whole chordotonal pool. Also ours: the linear
position code, the 0.5/0.5 position-movement mix, the 75% limit
threshold, and the reference scales below.

Leg naming: MANC nerves ProLN/MesoLN/MetaLN -> T1/T2/T3, matched to
FlyGym's lf/lm/lh/rf/rm/rh exactly as in `motor_neuron_to_joint.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fly_robot.interface.motor_neuron_to_joint import LEG_NAME_TO_FLYGYM_PREFIX

# MANC leg nerve -> thoracic segment. ProLN = prothoracic (front) leg
# nerve, MesoLN = mesothoracic (middle), MetaLN = metathoracic (hind).
NERVE_TO_SEGMENT = {"ProLN": "T1", "MesoLN": "T2", "MetaLN": "T3"}

# Sensory subclasses used, and the joint each drives.
CHORDOTONAL = "chordotonal organ"
HAIR_PLATE = "hair plate"
MODELLED_SUBCLASSES = (CHORDOTONAL, HAIR_PLATE)

# Reference scales, MEASURED rather than assumed. Driving the frozen motor
# interface with a real representative replicate and recording every joint
# gives 99th-percentile excursions up to 0.275 rad and angular velocities
# up to 9.8 rad/s (full per-leg table in docs/closed_loop/LOG.md).
#
# Deliberately NOT set to the motor interface's theoretical maximum
# (+-0.5 rad): most legs move far less than that, so a 0.5 rad reference
# would squash the real signal into the bottom of the encoder's range.
# That is exactly the failure that made `DEFAULT_RATE_SCALE_HZ = 20`
# wrong on the motor side (docs/logs/2026-09-19.md) — calibrate to the
# signal that actually occurs, not to the largest one imaginable.
# Excursions beyond these simply saturate the encoder, which is bounded.
POSITION_REF_RAD = 0.3
VELOCITY_REF_RAD_S = 10.0

# Position and movement contribute equally, because the real population
# encodes both and we cannot separate the subtypes. A choice, not a
# measurement.
W_POSITION = 0.5
W_VELOCITY = 0.5

# Hair plates stay silent until the joint is in the outer quarter of its
# range, then rise — the "limit detector" behaviour quoted above. The
# fraction is ours; the rectification is from the paper.
HAIR_PLATE_THRESHOLD_FRAC = 0.75

# Two encoder formulations, both pre-registered (see
# docs/closed_loop/PREREGISTRATION.md, pilot-informed amendment):
#
#   "signed"    — position is a signed monotonic code, 0.5 at rest, so
#                 flexion and extension are distinguishable. This is the
#                 original formulation. Its cost, found by the pilot, is a
#                 large CONSTANT term: every chordotonal neuron receives
#                 0.25 * g_fb at rest, and measurement showed that DC part
#                 is 88.1% of all injected drive, with real movement
#                 modulation only 11.7% of it. That tonic current, not the
#                 feedback, is what pushes the network past a bifurcation.
#
#   "deviation" — drive proportional to the RECTIFIED deviation from the
#                 settled resting pose, so there is ~0 input at rest and
#                 the injected current is movement, not a standing bias.
#                 The cost is that |deviation| cannot distinguish flexion
#                 from extension — acceptable here because the data does
#                 not resolve chordotonal subtypes anyway, so a pooled
#                 population could not signal direction either way.
ENCODER_MODES = ("signed", "deviation")


def _femur_tibia_joint(leg_prefix: str) -> str:
    return f"{leg_prefix}_trochanterfemur-{leg_prefix}_tibia-pitch"


def _thorax_coxa_joint(leg_prefix: str) -> str:
    return f"c_thorax-{leg_prefix}_coxa-pitch"


@dataclass
class SensoryGroups:
    """Row indices (into the simulation's neuron axis) of the sensory
    neurons behind each (segment, side, subclass), plus the per-group
    scale factor that equalises left/right total drive."""
    indices_by_group: dict     # (segment, side, subclass) -> list[int]
    scale_by_group: dict       # (segment, side, subclass) -> float
    n_neurons: int

    def group_keys(self):
        return sorted(self.indices_by_group)


def build_sensory_groups(wtable: pd.DataFrame, normalise_sides: bool = True) -> SensoryGroups:
    """Find each leg's proprioceptors in the simulated network.

    Leg and side come from the **nerve encoded in the `instance` string**
    (`SNpp45_MetaLN_L` -> MetaLN, left). `somaNeuromere` cannot be used:
    sensory somata sit out in the leg, so it is NaN for 5,889 of the
    5,891 sensory neurons in this table.

    `normalise_sides` equalises the TOTAL current delivered to a segment's
    left and right pools, by scaling each group by
    `mean(N_left, N_right) / N_group`. The annotation is markedly
    asymmetric — T1 has 23 left chordotonals against 57 right — and the
    primary hypothesis is about left/right coupling, so un-normalised
    feedback would build the asymmetry straight into the measured
    quantity. This touches only the interface; no connectome weight,
    sign or topology is altered. Condition C4 sets it False to measure
    what the correction does.
    """
    sensory = wtable[wtable["class"].isin(["sensory neuron", "sensory ascending"])].copy()

    def parse(instance):
        if not isinstance(instance, str):
            return None, None
        parts = instance.split("_")
        if len(parts) < 3:
            return None, None
        return NERVE_TO_SEGMENT.get(parts[1]), {"L": "LHS", "R": "RHS"}.get(parts[-1])

    parsed = sensory["instance"].apply(lambda s: pd.Series(parse(s), index=["segment", "side"]))
    sensory = pd.concat([sensory, parsed], axis=1)
    sensory = sensory[sensory["subclass"].isin(MODELLED_SUBCLASSES)
                      & sensory["segment"].notna() & sensory["side"].notna()]

    indices_by_group = {}
    for (segment, side, subclass), group in sensory.groupby(["segment", "side", "subclass"]):
        indices_by_group[(segment, side, subclass)] = sorted(int(i) for i in group.index)

    scale_by_group = {}
    for key, idxs in indices_by_group.items():
        segment, side, subclass = key
        if not normalise_sides:
            scale_by_group[key] = 1.0
            continue
        counts = [len(indices_by_group.get((segment, s, subclass), []))
                  for s in ("LHS", "RHS")]
        counts = [c for c in counts if c > 0]
        reference = float(np.mean(counts)) if counts else 0.0
        scale_by_group[key] = reference / len(idxs) if idxs else 0.0

    return SensoryGroups(indices_by_group, scale_by_group, n_neurons=len(wtable))


def chordotonal_drive(angle_rad: float, velocity_rad_s: float, reference_rad: float,
                      mode: str = "signed") -> float:
    """Combined position + movement code for the femur-tibia joint, in [0, 1].

    `signed`: position is a signed monotonic code, 0.5 at the reference
    angle, so flexion and extension are distinguishable — at the cost of a
    0.25 standing drive at rest.

    `deviation`: position is the unsigned, rectified distance from the
    settled resting pose, so rest gives 0 and the injected current tracks
    movement rather than posture.

    Movement is unsigned in both, since a pooled population cannot express
    direction.
    """
    movement = float(np.clip(abs(velocity_rad_s) / VELOCITY_REF_RAD_S, 0.0, 1.0))
    if mode == "deviation":
        position = float(np.clip(abs(angle_rad - reference_rad) / POSITION_REF_RAD, 0.0, 1.0))
    else:
        position = float(np.clip(
            0.5 + 0.5 * (angle_rad - reference_rad) / POSITION_REF_RAD, 0.0, 1.0))
    return float(np.clip(W_POSITION * position + W_VELOCITY * movement, 0.0, 1.0))


def hair_plate_drive(angle_rad: float, reference_rad: float,
                     mode: str = "signed") -> float:
    """Rectified limit-detector code for the thorax-coxa joint, in [0, 1]:
    silent until the joint passes HAIR_PLATE_THRESHOLD_FRAC of its
    reference range, then rising.

    `signed` detects one limit only (the anterior one, matching CxHP8 as
    published). `deviation` uses |deviation from rest|, so it fires at
    EITHER extreme — defensible because MANC annotates only a generic
    "hair plate" subclass with no CxHP3/4/8 identity, and the source paper
    states the three hair plates "wrap the joint along the
    anterior-posterior axis", i.e. different ones detect different limits.
    Pooling them is closer to the real population than arbitrarily picking
    one direction.

    Measured in the pilot: under `signed` these fire **never** — the
    thorax-coxa joint never reaches 75% of range in the extension
    direction, so the channel contributes exactly 0.0000 drive.
    """
    if mode == "deviation":
        position = float(np.clip(abs(angle_rad - reference_rad) / POSITION_REF_RAD, 0.0, 1.0))
    else:
        position = float(np.clip(
            0.5 + 0.5 * (angle_rad - reference_rad) / POSITION_REF_RAD, 0.0, 1.0))
    if position <= HAIR_PLATE_THRESHOLD_FRAC:
        return 0.0
    return float((position - HAIR_PLATE_THRESHOLD_FRAC) / (1.0 - HAIR_PLATE_THRESHOLD_FRAC))


class SensoryEncoder:
    """Turns the body's joint state into a per-neuron input-current vector.

    Strictly per-leg by construction: the drive for a group keyed
    (segment, side, *) is computed from the joints of exactly the one
    FlyGym leg that (segment, side) maps to. There is no code path by
    which leg i's sensors can reach leg j's neurons — enforced by the
    locality test in `tests/`, not merely asserted here.
    """

    def __init__(self, groups: SensoryGroups, jointdof_order, neutral_angles_by_name: dict,
                 gain: float = 1.0, leg_permutation: dict | None = None,
                 mode: str = "signed"):
        """`leg_permutation` maps (segment, side) -> (segment, side) and is
        used ONLY by control condition C3, which deliberately permutes
        sensory channels across legs to test whether leg-specific feedback
        matters. It is a labelled locality violation; leave it None
        everywhere else.

        `mode` selects the encoder formulation — see ENCODER_MODES. In
        "deviation" mode the reference is the SETTLED resting pose, which
        must be supplied via `set_rest_angles` after the body has settled;
        until then the neutral pose is used as the reference.
        """
        if mode not in ENCODER_MODES:
            raise ValueError(f"mode must be one of {ENCODER_MODES}, got {mode!r}")
        self.groups = groups
        self.gain = float(gain)
        self.mode = mode
        self.leg_permutation = leg_permutation or {}

        self._index_of = {d.name: i for i, d in enumerate(jointdof_order)}
        self._neutral = dict(neutral_angles_by_name)
        self._reference = dict(neutral_angles_by_name)
        self._joint_ids = {}
        for segment, side in LEG_NAME_TO_FLYGYM_PREFIX:
            prefix = LEG_NAME_TO_FLYGYM_PREFIX[(segment, side)]
            self._joint_ids[(segment, side)] = {
                "femur_tibia": self._index_of.get(_femur_tibia_joint(prefix)),
                "thorax_coxa": self._index_of.get(_thorax_coxa_joint(prefix)),
                "femur_tibia_name": _femur_tibia_joint(prefix),
                "thorax_coxa_name": _thorax_coxa_joint(prefix),
            }

    def set_rest_angles(self, joint_angles: np.ndarray) -> None:
        """Adopt the body's SETTLED pose as the reference for "deviation"
        mode. Measured after warmup, before stimulation onset, so it is a
        deterministic property of the body configuration rather than a
        tuned parameter — and it is necessarily different in the harness
        and on the ball, because the body genuinely rests differently.
        """
        for name, index in self._index_of.items():
            self._reference[name] = float(joint_angles[index])

    def channel_drives(self, joint_angles: np.ndarray, joint_velocities: np.ndarray) -> dict:
        """Per-group drive in [0, 1], before gain and normalisation.
        Exposed separately because condition C2 needs to record and then
        phase-randomise exactly these channels."""
        drives = {}
        for key in self.groups.indices_by_group:
            segment, side, subclass = key
            source = self.leg_permutation.get((segment, side), (segment, side))
            ids = self._joint_ids[source]
            if subclass == CHORDOTONAL:
                i = ids["femur_tibia"]
                drives[key] = 0.0 if i is None else chordotonal_drive(
                    float(joint_angles[i]), float(joint_velocities[i]),
                    self._reference.get(ids["femur_tibia_name"], 0.0), mode=self.mode)
            else:
                i = ids["thorax_coxa"]
                drives[key] = 0.0 if i is None else hair_plate_drive(
                    float(joint_angles[i]),
                    self._reference.get(ids["thorax_coxa_name"], 0.0), mode=self.mode)
        return drives

    def input_current(self, joint_angles: np.ndarray, joint_velocities: np.ndarray,
                      drives: dict | None = None) -> np.ndarray:
        """Current to add to the network's input vector, shape
        (n_neurons,). Zero everywhere except the leg proprioceptors.

        `drives` optionally overrides the computed per-channel drives —
        this is how C2 injects its phase-randomised surrogates through
        exactly the same path as the real signal.
        """
        if drives is None:
            drives = self.channel_drives(joint_angles, joint_velocities)
        current = np.zeros(self.groups.n_neurons, dtype=np.float64)
        if self.gain == 0.0:
            # Exactly zero, not approximately: the g_fb = 0 condition must
            # reproduce open-loop bit-for-bit.
            return current
        for key, idxs in self.groups.indices_by_group.items():
            if not idxs:
                continue
            current[idxs] = self.gain * drives[key] * self.groups.scale_by_group[key]
        return current
