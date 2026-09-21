"""Detector suite for the free-walking rig, before any training on it.

The ball rig was abandoned for training because a single kick banked
progress for the whole trial: the ball's 1e-6 damping models a
frictionless air bearing, coast-down time constant ~11,700 s against a
4 s trial. The first pilot's top candidate scored 94% of reference
walking speed while moving its joints 20x LESS than untrained.

These three checks are the ones that exploit had to fail, plus the
control that the rig change did not disturb the neural side:

  1. ONE-KICK-THEN-FREEZE must score ~0 progress. The exploit that worked
     on the ball must not work here.
  2. SYNTHETIC TRIPOD GAIT must walk forward. A rig where nothing can walk
     would pass check 1 trivially and be useless.
  3. UNTRAINED NEURAL METRICS must be unchanged from the ball rig. At zero
     feedback the body cannot influence the network at all, so n_active,
     max firing rate and rhythmicity must match bit-for-bit. Any
     difference means the rig change leaked into the neural path.

Check 2 also MEASURES the forward-speed reference the progress term is
normalised by -- the ground-rig equivalent of the ball's -0.95 rad/s.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.validate_ground_rig
"""

import argparse
import json

import numpy as np
from flygym import Simulation
from flygym.compose.fly.base_fly import ActuatorType

from fly_robot.analysis.interleg_coordination import coordination
from fly_robot.bodies.neuromechfly import build_free_fly
from fly_robot.experiments.harness_sine_wave_test import build_target_angle_function
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

DURATION_S = 2.0
KICK_S = 0.15          # how long the impulse lasts before freezing
SETTLE_S = 0.3


def _free_sim(adhesion=True):
    fly, world, _m, _d, *_c = build_free_fly()
    sim = Simulation(world)
    dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    sim.reset()
    sim.warmup()
    if adhesion:
        sim.set_leg_adhesion_states(fly.name, np.ones(6))
    return fly, sim, dofs


def forward_displacement(sim, fly, n_steps, target_fn, adhesion=True):
    """Step the body under `target_fn(t)` and return (dx, dy, final quat)."""
    dt = sim.timestep
    p0 = sim.get_body_positions(fly.name)[0].copy()
    for i in range(n_steps):
        t = i * dt
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, target_fn(t))
        if adhesion:
            sim.set_leg_adhesion_states(fly.name, np.ones(6))
        sim.step()
    p1 = sim.get_body_positions(fly.name)[0]
    q = sim.get_body_rotations(fly.name)[0]
    return float(p1[0] - p0[0]), float(p1[1] - p0[1]), q


def check_tripod_walks():
    """CHECK 2, and the source of the forward-speed reference."""
    fly, sim, dofs = _free_sim()
    gait = build_target_angle_function(fly, dofs)
    dt = sim.timestep
    dx, dy, q = forward_displacement(sim, fly, int(DURATION_S / dt), gait)
    speed = dx / DURATION_S
    print(f"  tripod gait: dx={dx:+.4f} mm over {DURATION_S}s "
          f"-> {speed:+.4f} mm/s   (dy={dy:+.4f}, quat_w={q[0]:.3f})")
    return {"dx_mm": dx, "dy_mm": dy, "speed_mm_s": speed, "quat_w": float(q[0])}


def check_kick_then_freeze(reference_speed):
    """CHECK 1. The ball-rig exploit, run on the new rig."""
    fly, sim, dofs = _free_sim()
    gait = build_target_angle_function(fly, dofs)
    neutral = gait(0.0)
    dt = sim.timestep

    def target(t):
        # Swing hard for KICK_S, then hold a fixed pose for the rest.
        return gait(t) if t < KICK_S else neutral

    dx, dy, q = forward_displacement(sim, fly, int(DURATION_S / dt), target)
    speed = dx / DURATION_S
    frac = speed / reference_speed if reference_speed else float("nan")
    print(f"  kick-then-freeze: dx={dx:+.4f} mm -> {speed:+.4f} mm/s "
          f"= {frac:+.1%} of tripod walking")
    return {"dx_mm": dx, "speed_mm_s": speed, "frac_of_reference": frac}


def check_neural_unchanged():
    """CHECK 3. At g_fb = 0 the body cannot reach the neurons, so the ball
    and ground rigs must produce bit-identical neural output."""
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S)
    out = {}
    for rig in ("ball", "ground"):
        res = run_trial(model, mg, sensory_groups=sg, feedback_gain=0.0,
                        rig=rig, duration_s=DURATION_S, seed=1,
                        encoder_mode="deviation")
        rhythm, _c = coordination(res.motor_rates, NEURAL_DT)
        out[rig] = {"n_active": res.n_active_neurons,
                    "max_fr": round(float(res.max_firing_rate), 6),
                    "n_rhythmic": int(rhythm.rhythmic.sum()),
                    "motor_rates": res.motor_rates}
    same = np.array_equal(out["ball"]["motor_rates"], out["ground"]["motor_rates"])
    for rig in ("ball", "ground"):
        d = out[rig]
        print(f"  {rig:>7}: n_active={d['n_active']}  max_fr={d['max_fr']:.4f}  "
              f"rhythmic={d['n_rhythmic']}/6")
    print(f"  motor-rate traces bit-identical across rigs: {same}")
    return {"ball": {k: v for k, v in out["ball"].items() if k != "motor_rates"},
            "ground": {k: v for k, v in out["ground"].items() if k != "motor_rates"},
            "bit_identical": bool(same)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="media/trained_adapter/ground_rig_validation.json")
    args = ap.parse_args()

    print("=== CHECK 2: synthetic tripod gait must WALK FORWARD ===")
    tripod = check_tripod_walks()

    print("\n=== CHECK 1: one-kick-then-freeze must score ~0 ===")
    kick = check_kick_then_freeze(tripod["speed_mm_s"])

    print("\n=== CHECK 3: untrained neural metrics unchanged by the rig ===")
    neural = check_neural_unchanged()

    verdict = {
        "tripod_walks_forward": bool(tripod["speed_mm_s"] > 0),
        "kick_scores_near_zero": bool(abs(kick["frac_of_reference"]) < 0.10),
        "neural_unchanged": neural["bit_identical"],
    }
    print("\n=== VERDICT ===")
    for k, v in verdict.items():
        print(f"  {k:<26} {'PASS' if v else 'FAIL'}")
    print(f"\n  forward-speed reference for the progress term: "
          f"{tripod['speed_mm_s']:+.4f} mm/s")

    payload = {"tripod": tripod, "kick_then_freeze": kick, "neural": neural,
               "verdict": verdict, "duration_s": DURATION_S}
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
