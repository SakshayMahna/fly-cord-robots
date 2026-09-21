"""The reward — frozen, hashed, and entirely OUR ADDITION.

Nothing in the fly, the connectome, or Pugliese et al. says what a good
gait is. This module is a value judgement we impose, which is why it is
written down, weighted, hashed and frozen BEFORE any run rather than tuned
until the results look right.

Approved 2026-09-21 (`docs/trained_adapter/REWARD.md`). The weights must
not be edited to fit a result, and must be identical for the real
connectome and every control — a control scored on a different objective
is not a control.

ONE NOTATION FIX relative to the approved document, which changes no
approved weight. REWARD.md's table wrote three penalty terms with a minus
already inside the definition AND a negative weight, e.g.
`straightness = -(|roll| + |yaw|)/0.95` at weight `-0.20`, which would
multiply out to a positive contribution — i.e. it would have REWARDED
veering. Here every term is defined as a positive-magnitude quantity and
the weight alone carries the sign, so the weights mean exactly what the
document's "weight" column says. Affects the notation of terms 2, 5 and 6
only. Flagged rather than silently corrected.

FORWARD IS NEGATIVE PITCH. `docs/closed_loop/PREREGISTRATION.md` §2:
"under a synthetic tripod drive the ball turns about the pitch axis
(-0.95 rad/s) ... i.e. straight-line forward walking". The progress term is
therefore `-pitch`, and `sign_test()` exists to prove it at startup. With
this backwards, a trainer learns to walk backwards while the scalar score
reads as success the whole way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from fly_robot.analysis.interleg_coordination import TRANSIENT_S, coordination
from fly_robot.sim.closed_loop import NEURAL_DT

# --- frozen configuration -------------------------------------------------
# Every number the reward depends on. Hashed; checkpoints refuse a mismatch.
REWARD_CONFIG = {
    "version": "2026-09-21",
    "weights": {
        "progress": 1.00,
        "straightness": -0.20,
        "rhythmicity": 0.30,
        "coordination": 0.20,
        "posture": -0.10,
        "energy": -0.05,
        "saturation": -2.00,
    },
    "references": {
        # Measured straight-walking rate under synthetic tripod drive,
        # PREREGISTRATION.md §2. A calibration constant, and ours.
        "walking_rad_s": 0.95,
        # The sensory encoder's own POSITION_REF_RAD.
        "posture_ref_rad": 0.30,
        # Untrained baseline, stability-filtered (DESIGN.md §5.7), so the
        # energy term is ~1.0 at the starting configuration.
        "energy_ref": 3.331e-05,
        # Pugliese's own documented oversaturation criterion.
        "saturation_n_active": 1500.0,
        "saturation_cap": 2.0,
    },
    "transient_discard_s": TRANSIENT_S,
}


def reward_config_hash(config: dict | None = None) -> str:
    """sha256 of the frozen config — same discipline as the motor
    interface's `04be9dec...`. Written into every checkpoint."""
    return hashlib.sha256(
        json.dumps(config or REWARD_CONFIG, sort_keys=True).encode()).hexdigest()


@dataclass
class RewardBreakdown:
    """Per-term, never just the scalar. A single number cannot show which
    term is being optimised, and that is what reward hacking looks like
    from the outside."""
    total: float
    terms: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        out = {"reward": self.total}
        out.update({f"term_{k}": v for k, v in self.terms.items()})
        out.update({f"raw_{k}": v for k, v in self.raw.items()})
        return out


def evaluate(result, config: dict | None = None) -> RewardBreakdown:
    """Score one `TrialResult`. Requires a trial run `on_ball=True`."""
    cfg = config or REWARD_CONFIG
    w, ref = cfg["weights"], cfg["references"]
    keep = int(cfg["transient_discard_s"] / NEURAL_DT)

    angles = result.joint_angles[keep:]
    if result.ball_angvel is None:
        raise ValueError("reward requires a tethered-ball trial (on_ball=True)")
    av = result.ball_angvel[keep:]

    # 1. progress — NEGATIVE pitch is forward (see module docstring).
    pitch = float(np.mean(av[:, 1]))
    progress = -pitch / ref["walking_rad_s"]

    # 2. straightness — a positive COST; the weight carries the minus.
    straightness = (float(np.mean(np.abs(av[:, 0])))
                    + float(np.mean(np.abs(av[:, 2])))) / ref["walking_rad_s"]

    # 3/4. rhythm and coordination, reusing Phase 4's AR(1)-gated metrics
    # unchanged — a leg's phase only counts if it is actually rhythmic.
    rhythm, coord = coordination(result.motor_rates, NEURAL_DT)
    rhythmicity = float(rhythm.rhythmic.sum()) / 6.0
    tripod = float(coord.tripod_index) if (
        coord.has_rhythm and np.isfinite(coord.tripod_index)) else -1.0
    coordination_term = (tripod + 1.0) / 2.0

    # 5. posture — deviation from the trial's own settled pose, a cost.
    posture = float(np.abs(angles - angles.mean(axis=0)).mean()) / ref["posture_ref_rad"]

    # 6. energy — squared per-step joint motion, a cost. Measured on joint
    # ANGLES rather than targets, because run_trial records angles; the two
    # differ by the actuator's own response and this is a proxy, stated.
    energy = float((np.diff(angles, axis=0) ** 2).sum(axis=1).mean()) / ref["energy_ref"]

    # 7. saturation — hinge above Pugliese's criterion, capped so one
    # catastrophic episode cannot dominate an averaged candidate score
    # without bound. This is the largest coefficient because a SEIZING
    # NETWORK SPINS THE BALL: DESIGN.md §5.7 measured a saturated replicate
    # producing 30x the ball rotation of any healthy one, so saturation is
    # instrumentally attractive to an optimiser, not merely wrong.
    n_active = float(result.n_active_neurons)
    saturation = float(np.clip(
        (n_active - ref["saturation_n_active"]) / ref["saturation_n_active"],
        0.0, ref["saturation_cap"]))

    raw_terms = {
        "progress": progress, "straightness": straightness,
        "rhythmicity": rhythmicity, "coordination": coordination_term,
        "posture": posture, "energy": energy, "saturation": saturation,
    }
    weighted = {k: w[k] * v for k, v in raw_terms.items()}
    total = float(sum(weighted.values()))

    return RewardBreakdown(
        total=total, terms=weighted,
        raw={**raw_terms, "pitch_rad_s": pitch, "n_active": n_active,
             "n_rhythmic": float(rhythm.rhythmic.sum()), "tripod_index": tripod,
             "unstable": float(bool(result.unstable))},
    )


# --- the blocking startup gate -------------------------------------------

class SignTestFailure(RuntimeError):
    """Raised when the progress term does not reward forward walking."""


def _synthetic_result(pitch_rad_s: float, n_steps: int = 4000):
    """A minimal TrialResult carrying a known ball rotation, for the sign
    gate. Everything else is held at a neutral, non-rhythmic baseline so
    only the progress term moves."""
    from fly_robot.sim.closed_loop import TrialResult

    t = np.arange(n_steps) * NEURAL_DT
    return TrialResult(
        motor_rates=np.tile(np.sin(2 * np.pi * 11.0 * t), (6, 1)).astype(np.float32),
        joint_angles=np.zeros((n_steps, 42), dtype=np.float32),
        joint_velocities=np.zeros((n_steps, 42), dtype=np.float32),
        sensory_drive=np.zeros((0, n_steps), dtype=np.float32),
        sensory_channels=[],
        ball_quat=np.zeros((n_steps, 4), dtype=np.float32),
        ball_angvel=np.column_stack([
            np.zeros(n_steps), np.full(n_steps, pitch_rad_s), np.zeros(n_steps),
        ]).astype(np.float32),
        n_active_neurons=400, max_firing_rate=19.0, wall_clock_s=0.0,
    )


def sign_test(verbose: bool = True) -> dict:
    """BLOCKING GATE: prove the progress term rewards FORWARD walking.

    Two independent checks, neither of which depends on how well any
    particular synthetic gait happens to walk:

      A. **Kinematic.** On this rig the fly stands on TOP of the ball and
         its forward axis is +x (the ball centre carries a documented
         +0.16 mm *forward* offset, `docs/closed_loop/LOG.md`). A surface
         point at the top sits at r = (0, 0, +R), so its velocity under
         angular velocity w is v = w x r = (w_y*R, -w_x*R, 0). Walking
         forward drags that surface BACKWARD, i.e. v_x < 0, which requires
         **w_y < 0**. Hence forward walking is NEGATIVE pitch, exactly as
         `PREREGISTRATION.md` §2 records (-0.95 rad/s).

      B. **End-to-end through `evaluate()`.** A trial with negative pitch
         must score progress > 0 and one with positive pitch < 0. This
         catches a sign error anywhere in the reward itself, not just in
         the convention.

    A comment asserting a sign convention is worth nothing; with this
    backwards the trainer optimises BACKWARDS walking while the scalar
    score reads as success throughout.
    """
    out = {}

    # --- A. kinematic convention, from the real model's own geometry -----
    import mujoco
    from flygym import Simulation

    from fly_robot.bodies.neuromechfly import build_ball_fly

    _fly, world, _m, _d, *_c = build_ball_fly()
    sim = Simulation(world)
    sim.reset()
    sim.warmup()
    ball_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    centre = sim.mj_data.xpos[ball_id].copy()
    radius = 3.0
    top_velocity_x = float(np.cross(np.array([0.0, -1.0, 0.0]),
                                    np.array([0.0, 0.0, radius]))[0])
    out["kinematic"] = {"ball_centre": centre.tolist(),
                        "top_surface_vx_at_negative_pitch": top_velocity_x}
    if verbose:
        print(f"  sign test [kinematic]: ball centre {centre}, "
              f"w_y=-1 drags top surface v_x={top_velocity_x:+.2f} "
              f"({'backward' if top_velocity_x < 0 else 'FORWARD'})", flush=True)
    if not top_velocity_x < 0:
        raise SignTestFailure(
            "negative pitch does NOT drag the ball's top surface backward; the "
            "assumed axis convention (x forward, z up) does not hold for this "
            "model, and the progress term's sign cannot be trusted.")

    # --- B. the reward function itself ------------------------------------
    for label, pitch in (("forward", -0.95), ("backward", +0.95)):
        score = evaluate(_synthetic_result(pitch))
        out[label] = {"pitch_rad_s": pitch,
                      "progress_term": score.terms["progress"],
                      "progress_raw": score.raw["progress"]}
        if verbose:
            print(f"  sign test [{label}]: pitch={pitch:+.2f} rad/s -> "
                  f"progress={score.raw['progress']:+.3f}", flush=True)

    if not out["forward"]["progress_raw"] > 0:
        raise SignTestFailure(
            f"a FORWARD-walking trial (pitch {-0.95:+.2f}) scored progress="
            f"{out['forward']['progress_raw']:+.3f}, expected > 0. Training "
            "would optimise BACKWARDS walking while the score read as success.")
    if not out["backward"]["progress_raw"] < out["forward"]["progress_raw"]:
        raise SignTestFailure("the progress term cannot distinguish walking direction.")
    return out
