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
    "version": "2026-09-25-ground-gated-energy-recalibrated",
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
        # Forward displacement (mm, over the whole trial) at which the
        # rhythmicity and coordination terms reach full credit; below it
        # they scale linearly to zero. 1 mm in 4 s is 0.25 mm/s -- 1.8% of
        # the CPG baseline, i.e. deliberately a low bar. It is meant to
        # exclude standing still, not to demand good walking; the progress
        # term is what rewards speed. See the gate's comment in evaluate().
        "rhythm_gate_dx_mm": 1.0,
        "posture_ref_rad": 0.30,
        # MEASURED from the restricted CPG gait -- the controller that
        # actually walks this body through the adapter's own 18-DOF action
        # space (baselines/restricted_action_cpg.py, 9.822 mm/s): mean
        # 1.0814e-02, sd 2.7e-06 over seeds 0-2, sampled at NEURAL_DT.
        #
        # Was 3.331e-05, carried over unchanged from the ball reward, where
        # it had been calibrated from the UNTRAINED baseline -- a fly that
        # barely moves. Against that reference a real walking gait measures
        # 325x, so the nominal -0.05 "shaping" term became -16.5 and was by
        # far the largest term in the reward. Scored end to end through
        # evaluate(), the restricted CPG gait -- 34.76 mm of genuine forward
        # walking -- totalled -15.95, while standing perfectly still totals
        # 0.00. The reward did not merely fail to reward walking; it
        # forbade it, and both R1a runs were optimising correctly when they
        # converged first on motionless rhythm and then on a silent network.
        #
        # Calibrated the same way `walking_mm_s` is (from the CPG baseline
        # on this rig), so good walking now costs ~-0.05 exactly as
        # REWARD.md intends ("to stop degenerate flailing, not to shape
        # gait"), while flailing at 10x a walking gait's energy still costs
        # -0.5. Pinned by a positive-control test in
        # tests/test_ground_training_contract.py.
        "energy_ref": 1.0814e-02,
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


class SignTestFailure(RuntimeError):
    """The ground reward's forward-axis gate failed."""


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

    # LOCOMOTION GATE on the two rhythm terms (added 2026-09-25).
    #
    # Measured hole, not a hypothetical: R1a's best candidate at generation
    # 38 scored +0.3379, of which rhythmicity contributed +0.2500 (of a
    # possible 0.30) and coordination +0.1089 -- while travelling
    # dx = -0.010 mm in 4 s. Rendered, it stands in one pose for the entire
    # trial without visible leg motion. The connectome's rhythm is real and
    # is being measured correctly at the motor-neuron level; it is simply
    # ~1/11th of the excursion needed to move the body, so the reward was
    # paying full price for rhythm that never becomes locomotion.
    #
    # Raising `progress`'s weight cannot fix this: at dx ~ 0.001 mm,
    # progress = dx / (4 s * 14.025 mm/s) ~ 1e-5, so even a 100x weight
    # stays invisible. The problem is not that progress is outweighed, it is
    # that progress is unmeasurably small while rhythm pays in full. So the
    # rhythm terms are gated on the body actually moving.
    #
    # Gated on FORWARD displacement, not |dx|: gating on absolute distance
    # would let a candidate collect the full ~0.36 of rhythm credit by
    # walking backwards, against a progress penalty of order 1e-3 -- trading
    # one hole for another. With max(0, dx), moving the wrong way earns the
    # same zero as standing still rather than being paid for it.
    gate = float(np.clip(max(0.0, dx) / ref["rhythm_gate_dx_mm"], 0.0, 1.0))
    rhythmicity *= gate
    coordination_term *= gate

    if len(angles) > 1:
        posture = float(np.abs(angles - angles.mean(axis=0)).mean()) / ref["posture_ref_rad"]
        energy = float((np.diff(angles, axis=0) ** 2).sum(axis=1).mean()) / ref["energy_ref"]
    else:
        posture = energy = 0.0

    n_active = float(result.n_active_neurons)
    peak_n_active = float(result.peak_n_active_neurons
                          if result.peak_n_active_neurons is not None
                          else result.n_active_neurons)
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
             "peak_n_active": peak_n_active,
             "fraction_saturated_steps": float(result.fraction_saturated_steps),
             "tripod_index": tripod, "upright_mean": float(up.mean()),
             "terminated": float(result.terminated_at_step is not None),
             "frac_completed": n_done / max(n_planned, 1)})


def sign_test() -> None:
    """Block training if +x thorax motion does not earn positive progress.

    The ball reward's sign hazard was unusually easy to make because its
    forward axis is negative pitch. The ground rig is simpler (+x), but this
    still pins the convention in executable form before any expensive run.
    """
    from fly_robot.sim.closed_loop import TrialResult

    n = 2000
    t = np.arange(n) * NEURAL_DT
    motor = np.tile(np.sin(2 * np.pi * 11 * t), (6, 1)).astype(np.float32)
    angles = np.zeros((n, 42), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0

    def synthetic(dx_mm: float):
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, dx_mm, n)
        return TrialResult(
            motor_rates=motor, joint_angles=angles, joint_velocities=angles,
            sensory_drive=np.zeros((0, n), dtype=np.float32), sensory_channels=[],
            ball_quat=None, ball_angvel=None, n_active_neurons=400,
            max_firing_rate=19.0, wall_clock_s=0.0, thorax_pos=pos,
            thorax_quat=quat, n_steps_planned=n,
        )

    distance = REWARD_CONFIG["references"]["walking_mm_s"] * n * NEURAL_DT
    forward, backward = evaluate(synthetic(distance)), evaluate(synthetic(-distance))
    if not (forward.raw["progress"] > 0 and backward.raw["progress"] < 0
            and forward.total > backward.total):
        raise SignTestFailure(
            "ground reward does not rank +x travel above -x travel; aborting")
