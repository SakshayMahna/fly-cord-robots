"""Reward for the free-walking Stage A rig. Weights per REWARD.md Amendment 1.

Replaces the ball-rig reward for training. The ball is kept for Phase 4's
records and for video; it is not a sound training substrate, because its
1e-6 damping (a faithful model of a frictionless air bearing) lets one kick
coast for the whole trial. Measured coast-down time constant ~11,700 s
against a 4 s trial.

On ground the body stops when the legs stop, so progress is **thorax
displacement** and cannot be banked.

`v_ref` is MEASURED, not guessed: the CPG baseline's forward speed on this
exact rig. Every earlier attempt at this constant failed — the ball rig's
-0.95 rad/s is not reproducible from committed code, and the synthetic
tripod gait walks on no substrate at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from fly_robot.analysis.interleg_coordination import TRANSIENT_S, coordination
from fly_robot.sim.closed_loop import FLIP_UPRIGHT_THRESHOLD, NEURAL_DT

REWARD_CONFIG = {
    "version": "2026-09-21-ground",
    "rig": "free_ground",
    "weights": {
        "progress": 1.00,
        "straightness": -0.20,
        "upright": -0.50,
        "anti_flip": -1.00,
        "rhythmicity": 0.30,
        "coordination": 0.20,
        "posture": -0.10,
        "energy": -0.05,
        "saturation": -2.00,
    },
    "references": {
        # MEASURED: CPG baseline on this rig, 5 seeds, mean 14.025 mm/s,
        # sd 0.250. See media/baselines/baselines.json.
        "walking_mm_s": 14.025,
        "posture_ref_rad": 0.30,
        "energy_ref": 3.331e-05,
        "saturation_n_active": 1500.0,
        "saturation_cap": 2.0,
        "flip_upright_threshold": FLIP_UPRIGHT_THRESHOLD,
    },
    "transient_discard_s": TRANSIENT_S,
}


def reward_config_hash(config: dict | None = None) -> str:
    return hashlib.sha256(
        json.dumps(config or REWARD_CONFIG, sort_keys=True).encode()).hexdigest()


@dataclass
class RewardBreakdown:
    total: float
    terms: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        out = {"reward": self.total}
        out.update({f"term_{k}": v for k, v in self.terms.items()})
        out.update({f"raw_{k}": v for k, v in self.raw.items()})
        return out


def upright_series(quat: np.ndarray) -> np.ndarray:
    """Vertical component of the body z-axis: 1 level, 0 on its side,
    negative inverted. For a unit quaternion (w,x,y,z) that is 1-2(x^2+y^2)."""
    return 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)


def evaluate(result, config: dict | None = None) -> RewardBreakdown:
    cfg = config or REWARD_CONFIG
    w, ref = cfg["weights"], cfg["references"]
    if result.thorax_pos is None:
        raise ValueError("ground reward requires a free-ground trial (rig='ground')")

    n_planned = result.n_steps_planned
    n_done = result.terminated_at_step or n_planned
    keep = min(int(cfg["transient_discard_s"] / NEURAL_DT), max(n_done - 1, 0))

    pos = result.thorax_pos[:n_done]
    quat = result.thorax_quat[:n_done]
    angles = result.joint_angles[keep:n_done]
    full_duration = n_planned * NEURAL_DT

    # 1. progress -- net forward displacement over the FULL planned duration.
    # A terminated trial earns nothing for the time it did not survive, which
    # is what makes flipping strictly worse than walking slowly.
    dx = float(pos[-1, 0] - pos[keep, 0]) if n_done > keep else 0.0
    progress = dx / (full_duration * ref["walking_mm_s"])

    # 2. straightness -- lateral drift, a positive cost.
    dy = abs(float(pos[-1, 1] - pos[keep, 1])) if n_done > keep else 0.0
    straightness = dy / (full_duration * ref["walking_mm_s"])

    # 3/4. orientation. `upright` is graded and shapes posture; `anti_flip`
    # is a cliff that makes tipping over unprofitable outright. Time after an
    # early termination counts as flipped.
    up = upright_series(quat) if n_done else np.array([1.0])
    upright_cost = float(np.clip(1.0 - up.mean(), 0.0, 2.0))
    flipped_steps = int((up < ref["flip_upright_threshold"]).sum()) + (n_planned - n_done)
    anti_flip = flipped_steps / max(n_planned, 1)

    # A trial that terminates early may have less signal left than the
    # rhythm gate's own transient discard, which would make `coordination`
    # reduce over an empty array. A fly that flipped in the first half
    # second has no measurable rhythm, so score it as none rather than
    # crashing or, worse, silently scoring it as rhythmic.
    MIN_RHYTHM_STEPS = int(1.0 / NEURAL_DT)   # need >= 1 s of post-transient signal
    if n_done - keep >= MIN_RHYTHM_STEPS:
        rhythm, coord = coordination(result.motor_rates[:, :n_done], NEURAL_DT)
        n_rhythmic = float(rhythm.rhythmic.sum())
        tripod = float(coord.tripod_index) if (
            coord.has_rhythm and np.isfinite(coord.tripod_index)) else -1.0
    else:
        n_rhythmic, tripod = 0.0, -1.0
    rhythmicity = n_rhythmic / 6.0
    coordination_term = (tripod + 1.0) / 2.0

    if len(angles) > 1:
        posture = float(np.abs(angles - angles.mean(axis=0)).mean()) / ref["posture_ref_rad"]
        energy = float((np.diff(angles, axis=0) ** 2).sum(axis=1).mean()) / ref["energy_ref"]
    else:
        posture = energy = 0.0

    n_active = float(result.n_active_neurons)
    saturation = float(np.clip(
        (n_active - ref["saturation_n_active"]) / ref["saturation_n_active"],
        0.0, ref["saturation_cap"]))

    raw = {"progress": progress, "straightness": straightness,
           "upright": upright_cost, "anti_flip": anti_flip,
           "rhythmicity": rhythmicity, "coordination": coordination_term,
           "posture": posture, "energy": energy, "saturation": saturation}
    weighted = {k: w[k] * v for k, v in raw.items()}
    return RewardBreakdown(
        total=float(sum(weighted.values())), terms=weighted,
        raw={**raw, "dx_mm": dx, "dy_mm": dy, "speed_mm_s": dx / full_duration,
             "n_active": n_active, "n_rhythmic": n_rhythmic,
             "tripod_index": tripod, "upright_mean": float(up.mean()),
             "terminated": float(result.terminated_at_step is not None),
             "frac_completed": n_done / max(n_planned, 1)})
