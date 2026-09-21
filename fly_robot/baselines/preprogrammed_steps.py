"""Real-fly step kinematics, ported from FlyGym 1.2.1 onto our rig.

**This is NOT our work.** `PreprogrammedSteps` and the data it reads are
from FlyGym (NeLy-EPFL, Apache-2.0):
`flygym/examples/locomotion/steps.py` and
`flygym/data/behavior/single_steps_untethered.pkl`. The logic below is a
faithful port; what is ours is only the mapping onto this project's DOF
names and the fact that it drives our `build_free_fly` rig.

Why a port rather than an import: the controllers and the behavioural data
were dropped in FlyGym 2.x. Verified directly — FlyGym 2.1.0 has no
`examples` module and no `data/` directory, while 1.2.1's wheel contains
both. We run 2.1.0 for the body, so the 1.x code cannot simply be imported.

**Licence status: UNCONFIRMED and being checked with the authors.** The
1.2.1 distribution carries a single Apache-2.0 LICENSE and there is no
separate licence or notice file anywhere under `flygym/data/`, so the data
appears to fall under the same terms — but that is an inference from the
absence of a notice, not an explicit data-licence statement. Accordingly
the `.pkl` files are **NOT redistributed in this repository**. They live in
gitignored `data/flygym_behavior/`; `fetch_reference_data.py` retrieves
them from PyPI.

Provenance of the kinematics, per FlyGym's own documentation: a tethered
fly walking on an air-suspended spherical treadmill, filmed with seven
cameras, 3D keypoints via DeepFly3D, joint angles by inverse kinematics.
The file's own metadata records `timestep: 0.003` s and a source of
`data/3D_pose_alfie/clean_3d_best_ventral_best_side.csv`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

DEFAULT_DATA_PATH = Path("data/flygym_behavior/single_steps_untethered.pkl")

# FlyGym's leg identifiers, in its own order.
LEGS = [f"{side}{pos}" for side in "LR" for pos in "FMH"]

# FlyGym's 7 DOFs per leg, in its own order.
DOFS_PER_LEG = ["Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll",
                "Tibia", "Tarsus1"]

# OUR mapping: FlyGym 1.x leg id -> this project's FlyGym 2.x DOF prefix.
LEG_TO_PREFIX = {"LF": "lf", "LM": "lm", "LH": "lh",
                 "RF": "rf", "RM": "rm", "RH": "rh"}

# MIRRORING CORRECTION -- ours, and it is not optional.
#
# FlyGym 1.x's recorded kinematics use the OPPOSITE sign convention for
# right-leg roll and yaw from FlyGym 2.x's model. Mirroring left<->right
# flips rotations about the fore-aft and vertical axes but leaves the
# sagittal (pitch) ones alone, which is exactly the pattern measured:
#
#   Coxa_roll, data minus model, per leg
#     LF +0.62  LM +0.12  LH -0.04   |   RF -1.32  RM -3.44  RH -4.85
#
# Left legs agree; right legs are wildly off. Negating roll and yaw on the
# right legs drops mean |difference| across all 38 comparable DOFs from
# 0.591 to 0.315 rad, the maximum from 4.845 to 1.030, and the count of
# DOFs off by more than a radian from 6 to 2. Worked example:
#   RH Coxa_roll: data -2.402 -> negated +2.402, model +2.443.
#
# Without this the fly settles 41 degrees off vertical and cannot walk --
# a silently broken baseline rather than an obviously broken one.
MIRRORED_DOFS = {"Coxa_roll", "Coxa_yaw", "Femur_roll"}


def _mirror_sign(leg: str, dof: str) -> float:
    return -1.0 if (leg.startswith("R") and dof in MIRRORED_DOFS) else 1.0


def dof_name(leg: str, dof: str) -> str:
    """FlyGym 1.x (leg, dof) -> this project's actuated-DOF name.

    The two models describe the same seven joints per leg; only the naming
    convention changed between FlyGym 1.x and 2.x. Verified against
    `fly.get_actuated_jointdofs_order()` — exactly 7 per leg, no leftovers.
    """
    p = LEG_TO_PREFIX[leg]
    return {
        "Coxa": f"c_thorax-{p}_coxa-pitch",
        "Coxa_roll": f"c_thorax-{p}_coxa-roll",
        "Coxa_yaw": f"c_thorax-{p}_coxa-yaw",
        "Femur": f"{p}_coxa-{p}_trochanterfemur-pitch",
        "Femur_roll": f"{p}_coxa-{p}_trochanterfemur-roll",
        "Tibia": f"{p}_trochanterfemur-{p}_tibia-pitch",
        "Tarsus1": f"{p}_tibia-{p}_tarsus1-pitch",
    }[dof]


class PreprogrammedSteps:
    """Per-leg joint trajectories and swing/stance timing, by step phase.

    Ported from FlyGym 1.2.1 `examples/locomotion/steps.py`. Behaviour is
    unchanged; only the data path and the DOF-name mapping are ours.
    """

    legs = LEGS
    dofs_per_leg = DOFS_PER_LEG

    def __init__(self, path=None, neutral_pose_phases=(np.pi,) * 6):
        path = Path(path) if path is not None else DEFAULT_DATA_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. The FlyGym behavioural data is not "
                "redistributed in this repository (licence unconfirmed). "
                "Run: python -m fly_robot.baselines.fetch_reference_data")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._length = len(data["joint_LFCoxa"])
        self._timestep = data["meta"]["timestep"]
        self.duration = self._length * self._timestep
        self.meta = data["meta"]

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for leg in self.legs:
            angles = np.array([data[f"joint_{leg}{dof}"] for dof in self.dofs_per_leg])
            self._psi_funcs[leg] = CubicSpline(phase_grid, angles, axis=1,
                                               bc_type="periodic")

        self.neutral_pos = {
            leg: self._psi_funcs[leg](theta)[:, np.newaxis]
            for leg, theta in zip(self.legs, neutral_pose_phases)
        }

        stance = data["swing_stance_time"]["stance"]
        self.swing_period = {}
        for leg in self.legs:
            period = np.array([0.0, stance[leg]]) / self.duration * 2 * np.pi
            self.swing_period[leg] = period

    def get_joint_angles(self, leg, phase, magnitude=1):
        """Joint angles (7,) for `leg` at `phase`, scaled about neutral."""
        if np.ndim(phase) == 0:
            phase = np.array([phase])
        offset = self._psi_funcs[leg](phase) - self.neutral_pos[leg]
        return (self.neutral_pos[leg] + magnitude * offset).squeeze()

    def get_adhesion_onoff(self, leg, phase) -> bool:
        """Adhesion is OFF during swing, ON otherwise.

        This is FlyGym's own convention, reproduced exactly:
        `not (swing_start < phase % 2pi < swing_end)`. Matching it is the
        point — a fly that grips while swinging cannot step.
        """
        swing_start, swing_end = self.swing_period[leg]
        return not (swing_start < phase % (2 * np.pi) < swing_end)

    def default_pose_by_dof_name(self) -> dict:
        """{our DOF name: neutral angle} for all 42 leg DOFs, mirror-corrected."""
        out = {}
        for leg in self.legs:
            for i, dof in enumerate(self.dofs_per_leg):
                out[dof_name(leg, dof)] = (
                    _mirror_sign(leg, dof) * float(self.neutral_pos[leg][i, 0]))
        return out

    def targets_by_dof_name(self, phases, magnitudes) -> dict:
        """{our DOF name: angle} for all six legs at the given phases."""
        out = {}
        for i, leg in enumerate(self.legs):
            angles = self.get_joint_angles(leg, phases[i], magnitudes[i])
            for j, dof in enumerate(self.dofs_per_leg):
                out[dof_name(leg, dof)] = _mirror_sign(leg, dof) * float(angles[j])
        return out

    def adhesion_by_leg(self, phases) -> np.ndarray:
        """(6,) adhesion control in FlyGym's own leg order."""
        return np.array([float(self.get_adhesion_onoff(leg, phases[i]))
                         for i, leg in enumerate(self.legs)])
