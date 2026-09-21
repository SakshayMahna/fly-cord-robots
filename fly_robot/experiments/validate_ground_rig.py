"""Full gate suite for the adopted Stage A rig. Run before any training.

Five checks. The first two are the ones the ball rig failed; the third is
the flip rule; the fourth proves the rig change never reached the neural
side; the fifth is the baseline comparison for the video.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.validate_ground_rig
"""

import argparse
import json

import numpy as np
from flygym import Simulation
from flygym.compose.fly.base_fly import ActuatorType
from flygym_demo.complex_terrain.common import apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import (
    CPGController, make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

from fly_robot.adapter.reward_ground import REWARD_CONFIG, evaluate, upright_series
from fly_robot.analysis.interleg_coordination import coordination
from fly_robot.baselines.run_baselines import run as run_baseline
from fly_robot.bodies.neuromechfly import build_free_fly
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

DURATION_S = 2.0
V_REF = REWARD_CONFIG["references"]["walking_mm_s"]


def check_cpg_walks(seeds=(0, 1, 2, 3, 4)):
    rows = [run_baseline("cpg", DURATION_S, s) for s in seeds]
    sp = np.array([r["speed_mm_s"] for r in rows])
    up = np.array([r["upright_min"] for r in rows])
    print(f"  CPG: mean {sp.mean():+.3f} sd {sp.std(ddof=1):.3f} mm/s, "
          f"min {sp.min():+.3f}, upright_min {up.min():.3f}")
    return {"mean_speed": float(sp.mean()), "sd_speed": float(sp.std(ddof=1)),
            "min_upright": float(up.min()),
            "pass": bool((sp > 0.5 * V_REF).all() and up.min() > 0.8)}


def check_rule_based(seeds=(0, 1, 2)):
    rows = [run_baseline("rule_based", DURATION_S, s) for s in seeds]
    sp = np.array([r["speed_mm_s"] for r in rows])
    up = np.array([r["upright_min"] for r in rows])
    print(f"  rule-based: mean {sp.mean():+.3f} sd {sp.std(ddof=1):.3f} mm/s, "
          f"upright_min {up.min():.3f}")
    return {"mean_speed": float(sp.mean()), "sd_speed": float(sp.std(ddof=1)),
            "min_upright": float(up.min()), "pass": bool((sp > 0).all())}


def check_kick_then_freeze():
    """The ball-rig exploit, on ground. Kick hard, then hold still."""
    fly, world, _m, _d, *_ = build_free_fly()
    sim = Simulation(world)
    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ctrl = CPGController(cpg_network=make_tripod_cpg_network(sim.timestep, seed=0),
                         preprogrammed_steps=PreprogrammedSteps(),
                         output_dof_order=dof_order)
    sim.reset()
    sim.warmup()
    kick_s, dt = 0.15, sim.timestep
    p0 = sim.get_body_positions(fly.name)[0].copy()
    frozen = None
    pk = None
    for i in range(int(DURATION_S / dt)):
        t = i * dt
        action = ctrl.step()
        if t < kick_s:
            frozen = action
        apply_locomotion_action(sim, fly.name, frozen if t >= kick_s else action)
        sim.step()
        if pk is None and t >= kick_s:
            pk = sim.get_body_positions(fly.name)[0].copy()
    p1 = sim.get_body_positions(fly.name)[0]
    during, after = float(pk[0] - p0[0]), float(p1[0] - pk[0])
    coast_speed = after / (DURATION_S - kick_s)
    print(f"  kick {during:+.4f} mm, then {after:+.4f} mm over "
          f"{DURATION_S - kick_s:.2f}s while frozen "
          f"({coast_speed / V_REF:+.1%} of walking)")
    return {"during_mm": during, "after_mm": after,
            "coast_frac_of_walking": coast_speed / V_REF,
            "pass": bool(abs(coast_speed / V_REF) < 0.10)}


def check_flip_terminates_and_is_never_better():
    """A flipped trial must terminate, and must score below a slow but
    upright one. Otherwise flipping becomes a strategy."""
    fly, world, _m, _d, *_ = build_free_fly()
    sim = Simulation(world)
    dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    sim.reset()
    sim.warmup()
    # Drive the legs to an extreme one-sided pose to tip the fly over.
    tgt = np.zeros(len(dofs))
    for i, d in enumerate(dofs):
        if d.name.startswith(("c_thorax-lf", "c_thorax-lm", "c_thorax-lh")):
            tgt[i] = 1.5
    flipped_at = None
    for i in range(int(2.0 / sim.timestep)):
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, tgt)
        sim.step()
        q = sim.get_body_rotations(fly.name)[0]
        if 1 - 2 * (q[1] ** 2 + q[2] ** 2) < 0.5:
            flipped_at = i
            break
    print(f"  deliberate tip-over reached upright<0.5 at step {flipped_at}"
          if flipped_at else "  could not tip the fly over with this drive")

    # Scoring: a trial that flips at 25% and stops vs one that crawls upright.
    n = 2000
    def synth(flip_at=None, speed_mm_s=0.0, upright=1.0):
        from fly_robot.sim.closed_loop import TrialResult
        done = flip_at or n
        t = np.arange(n) * NEURAL_DT
        pos = np.zeros((n, 3), np.float32)
        pos[:, 0] = speed_mm_s * t
        quat = np.zeros((n, 4), np.float32)
        quat[:, 0] = 1.0
        if flip_at:
            quat[flip_at:, 1] = 0.7   # inverted after the flip
        else:
            quat[:, 1] = np.sqrt(max(0.0, (1 - upright) / 2))
        return TrialResult(
            motor_rates=np.tile(np.sin(2 * np.pi * 11 * t), (6, 1)).astype(np.float32),
            joint_angles=np.zeros((n, 42), np.float32),
            joint_velocities=np.zeros((n, 42), np.float32),
            sensory_drive=np.zeros((0, n), np.float32), sensory_channels=[],
            ball_quat=None, ball_angvel=None, n_active_neurons=400,
            max_firing_rate=19.0, wall_clock_s=0.0, thorax_pos=pos,
            thorax_quat=quat, terminated_at_step=flip_at, n_steps_planned=n)

    flip = evaluate(synth(flip_at=n // 4, speed_mm_s=V_REF))
    crawl = evaluate(synth(speed_mm_s=0.05 * V_REF))
    print(f"  flips at 25% while moving at full speed: {flip.total:+.4f}")
    print(f"  crawls upright at 5% of walking speed:   {crawl.total:+.4f}")
    return {"flip_score": flip.total, "crawl_score": crawl.total,
            "tipped_at_step": flipped_at,
            "pass": bool(flip.total < crawl.total and flipped_at is not None)}


def check_neural_unchanged():
    """At g_fb = 0 there is no body->neuron path, so every rig must give
    identical neural output."""
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S)
    out = {}
    for rig in ("ball", "ground"):
        res = run_trial(model, mg, sensory_groups=sg, feedback_gain=0.0,
                        rig=rig, duration_s=DURATION_S, seed=1,
                        encoder_mode="deviation")
        rhythm, _ = coordination(res.motor_rates, NEURAL_DT)
        out[rig] = {"n_active": res.n_active_neurons,
                    "max_fr": round(float(res.max_firing_rate), 6),
                    "n_rhythmic": int(rhythm.rhythmic.sum()),
                    "_rates": res.motor_rates}
    same = np.array_equal(out["ball"]["_rates"], out["ground"]["_rates"])
    for rig in ("ball", "ground"):
        d = out[rig]
        print(f"  {rig:>7}: n_active={d['n_active']} max_fr={d['max_fr']:.4f} "
              f"rhythmic={d['n_rhythmic']}/6")
    print(f"  motor-rate traces bit-identical: {same}")
    return {"ball": {k: v for k, v in out["ball"].items() if k != "_rates"},
            "ground": {k: v for k, v in out["ground"].items() if k != "_rates"},
            "pass": bool(same)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="media/baselines/gate_suite.json")
    args = ap.parse_args()

    results = {}
    print("=== 1. CPG baseline walks forward consistently ===")
    results["cpg_walks"] = check_cpg_walks()
    print("\n=== 2. rule-based baseline ===")
    results["rule_based"] = check_rule_based()
    print("\n=== 3. kick-then-freeze scores ~0 progress ===")
    results["kick_then_freeze"] = check_kick_then_freeze()
    print("\n=== 4. flips terminate, and never beat walking poorly ===")
    results["flip"] = check_flip_terminates_and_is_never_better()
    print("\n=== 5. neural metrics at g_fb=0 unchanged by the rig ===")
    results["neural_unchanged"] = check_neural_unchanged()

    print("\n=== VERDICT ===")
    for k, v in results.items():
        print(f"  {k:<22} {'PASS' if v['pass'] else 'FAIL'}")
    allpass = all(v["pass"] for v in results.values())
    print(f"\n  GATE SUITE: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
