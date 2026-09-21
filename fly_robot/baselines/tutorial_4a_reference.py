"""Tutorial 4a (CPG controller), reproduced STANDALONE with no project code.

Deliberately imports nothing from `fly_robot`. Its only job is to answer one
question: does the official NeuroMechFly v2 CPG walk on this machine, with
this FlyGym install? If it does, our rig is at fault and the diff against
`build_free_fly()` localises the cause. If it does not, the problem is the
environment, not our code.

Source, as installed and verified by inspection rather than assumed:
  package  flygym 2.1.0 (Apache-2.0), bundled module `flygym_demo`
  modules  flygym_demo.complex_terrain.{common, cpg_controller, preprogrammed}
  data     flygym_demo/complex_terrain/assets/single_steps_untethered.pkl

This supersedes the FlyGym 1.2.1 port in `preprogrammed_steps.py` / `cpg.py`:
the controllers were never missing from 2.x, they live in `flygym_demo`,
which was installed the whole time and which I failed to find.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.baselines.tutorial_4a_reference
"""

import argparse
import json

import numpy as np
from flygym import Simulation
from flygym.compose.world.flat_ground import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import (
    apply_locomotion_action, get_default_locomotion_dof_order,
    make_locomotion_fly,
)
from flygym_demo.complex_terrain.cpg_controller import (
    CPGController, make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

SPAWN_POS_MM = (0.0, 0.0, 0.5)


def run(duration_s=2.0, seed=0, video_path=None, spawn_pos=SPAWN_POS_MM):
    fly = make_locomotion_fly(colorize=True)
    world = FlatGroundWorld()
    world.add_fly(fly, list(spawn_pos),
                  Rotation3D(format="quat", values=[1, 0, 0, 0]))
    world.compile()

    cam_top = fly.add_tracking_camera(
        name="top_down_cam", pos_offset=(0, 0, 12),
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0)))
    cam_side = fly.add_tracking_camera()
    world.compile()

    sim = Simulation(world)
    if video_path:
        sim.set_renderer([cam_side, cam_top])

    steps = PreprogrammedSteps()
    controller = CPGController(
        cpg_network=make_tripod_cpg_network(sim.timestep, seed=seed),
        preprogrammed_steps=steps,
        output_dof_order=get_default_locomotion_dof_order(),
    )

    sim.reset()
    sim.warmup()

    track, quats = [], []
    n = int(duration_s / sim.timestep)
    for i in range(n):
        apply_locomotion_action(sim, fly.name, controller.step())
        sim.step()
        if video_path:
            sim.render_as_needed()
        if i % 10 == 0:
            track.append(sim.get_body_positions(fly.name)[0].copy())
            quats.append(sim.get_body_rotations(fly.name)[0].copy())

    if video_path:
        sim.renderer.save_video({cam_side: video_path,
                                 cam_top: video_path.replace(".mp4", "_top.mp4")})

    track, quats = np.array(track), np.array(quats)
    upright = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
    return {
        "speed_mm_s": float(track[-1, 0] - track[0, 0]) / duration_s,
        "dx_mm": float(track[-1, 0] - track[0, 0]),
        "dy_mm": float(track[-1, 1] - track[0, 1]),
        "upright_start": float(upright[0]), "upright_end": float(upright[-1]),
        "upright_min": float(upright.min()),
        "thorax_z_start": float(track[0, 2]), "thorax_z_end": float(track[-1, 2]),
        "duration_s": duration_s, "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--video", default="media/baselines/tutorial_4a.mp4")
    ap.add_argument("--out", default="media/baselines/tutorial_4a.json")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.video), exist_ok=True)

    rows = []
    for k, seed in enumerate(args.seeds):
        r = run(args.duration, seed, args.video if k == 0 else None)
        rows.append(r)
        print(f"  seed {seed}: speed={r['speed_mm_s']:+7.3f} mm/s  "
              f"dy={r['dy_mm']:+6.2f}  upright end={r['upright_end']:.3f} "
              f"min={r['upright_min']:.3f}  z={r['thorax_z_end']:.3f}", flush=True)

    speeds = np.array([r["speed_mm_s"] for r in rows])
    uprights = np.array([r["upright_min"] for r in rows])
    print(f"\n  speed: mean {speeds.mean():+.3f} sd {speeds.std():.3f} mm/s")
    print(f"  upright_min across seeds: {uprights.min():.3f}")
    walks = bool((speeds > 0).all() and uprights.min() > 0.8)
    print(f"\n  WALKS FORWARD CONSISTENTLY AND UPRIGHT: {walks}")
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "walks": walks}, f, indent=1)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
