"""CPG and rule-based walking baselines on the adopted Stage A rig.

Both come from FlyGym 2.1.0's own bundled `flygym_demo.complex_terrain`
(Apache-2.0) and are used unmodified. Only the body they run on is ours —
`build_free_fly`, which now carries FlyGym's validated locomotion
configuration verbatim, so the connectome and both baselines share one
identical body.

This replaces an earlier port of the FlyGym **1.2.1** controllers. That port
is deleted: 1.x was never needed, because `flygym_demo` ships inside 2.1.0.
Its "mirroring correction" — negating right-leg roll and yaw — was not a
real convention difference at all; it was partially cancelling the
axis-order error, which is why it appeared to help (−0.78 → +2.95 mm/s)
while the fly was still face-planting.

`v_ref` for the reward's progress term is the CPG baseline's measured
forward speed here.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.baselines.run_baselines
"""

from __future__ import annotations

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
from flygym_demo.complex_terrain.rule_based_controller import RuleBasedController

from fly_robot.bodies.neuromechfly import build_free_fly

DURATION_S = 2.0
SEEDS = (0, 1, 2, 3, 4)


def _make_controller(kind, timestep, dof_order, seed):
    steps = PreprogrammedSteps()
    if kind == "cpg":
        return CPGController(
            cpg_network=make_tripod_cpg_network(timestep, seed=seed),
            preprogrammed_steps=steps, output_dof_order=dof_order)
    if kind == "rule_based":
        return RuleBasedController(
            timestep=timestep, preprogrammed_steps=steps,
            output_dof_order=dof_order, seed=seed)
    raise ValueError(kind)


def run(kind="cpg", duration_s=DURATION_S, seed=0, video_paths=None):
    fly, world, _m, _d, cam_side, cam_opp, cam_top = build_free_fly()
    sim = Simulation(world)
    if video_paths:
        by_name = {"side": cam_side, "opposite_side": cam_opp, "top_down": cam_top}
        targets = {by_name[k]: v for k, v in video_paths.items()}
        sim.set_renderer(list(targets.keys()))

    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    controller = _make_controller(kind, sim.timestep, dof_order, seed)
    sim.reset()
    sim.warmup()

    track, quats = [], []
    for i in range(int(duration_s / sim.timestep)):
        apply_locomotion_action(sim, fly.name, controller.step())
        sim.step()
        if video_paths:
            sim.render_as_needed()
        if i % 10 == 0:
            track.append(sim.get_body_positions(fly.name)[0].copy())
            quats.append(sim.get_body_rotations(fly.name)[0].copy())
    if video_paths:
        sim.renderer.save_video(targets)

    track, quats = np.array(track), np.array(quats)
    upright = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
    return {"controller": kind, "seed": seed,
            "speed_mm_s": float(track[-1, 0] - track[0, 0]) / duration_s,
            "dx_mm": float(track[-1, 0] - track[0, 0]),
            "dy_mm": float(track[-1, 1] - track[0, 1]),
            "abs_dy_mm": abs(float(track[-1, 1] - track[0, 1])),
            "upright_min": float(upright.min()),
            "upright_end": float(upright[-1]),
            "z_end": float(track[-1, 2])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out", default="media/baselines/baselines.json")
    args = ap.parse_args()

    out = {}
    for kind in ("cpg", "rule_based"):
        print(f"=== {kind} ===")
        rows = []
        for seed in args.seeds:
            r = run(kind, args.duration, seed)
            rows.append(r)
            print(f"  seed {seed}: speed={r['speed_mm_s']:+7.3f} mm/s  "
                  f"dy={r['dy_mm']:+6.2f}  upright_min={r['upright_min']:.3f}",
                  flush=True)
        sp = np.array([r["speed_mm_s"] for r in rows])
        up = np.array([r["upright_min"] for r in rows])
        dy = np.array([r["abs_dy_mm"] for r in rows])
        out[kind] = {"rows": rows, "mean_speed_mm_s": float(sp.mean()),
                     "sd_speed_mm_s": float(sp.std(ddof=1)),
                     "min_speed_mm_s": float(sp.min()),
                     "mean_abs_dy_mm": float(dy.mean()),
                     "min_upright": float(up.min()),
                     "walks": bool((sp > 0).all() and up.min() > 0.8)}
        print(f"  mean {sp.mean():+.3f} sd {sp.std(ddof=1):.3f} mm/s   "
              f"|dy| mean {dy.mean():.2f} mm   upright_min {up.min():.3f}   "
              f"WALKS: {out[kind]['walks']}\n")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"v_ref (CPG mean forward speed) = {out['cpg']['mean_speed_mm_s']:.3f} mm/s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
