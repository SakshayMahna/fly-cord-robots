"""Walk from the working tutorial rig toward ours, one change at a time.

Tutorial 4a walks: +13.9 mm/s, upright 0.97-1.00, across three seeds.
`build_free_fly()` face-plants. Every difference between the two is applied
here individually, against the same CPG controller, so the responsible one
is identified rather than guessed.

Differences found by reading `flygym_demo.complex_terrain.common`:

| setting            | tutorial          | ours (build_free_fly) |
|--------------------|-------------------|-----------------------|
| joint preset       | LEGS_ONLY         | ALL_BIOLOGICAL        |
| axis order         | YAW_PITCH_ROLL    | ROLL_PITCH_YAW        |
| joint stiffness    | 0.05              | 10.0                  |
| joint damping      | 0.06              | 0.5                   |
| tarsus stiffness   | 7.5 (per-joint)   | 10.0 (uniform)        |
| tarsus damping     | 1e-2              | 0.5                   |
| actuator gain      | 45                | 50                    |
| force range        | (-65, 65)         | (-30, 30)             |
| spawn height       | 0.5               | 0.7                   |

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.baselines.ablate_rig
"""

import argparse
import json

import numpy as np
from flygym import Simulation
from flygym.anatomy import (
    ActuatedDOFPreset, AxisOrder, JointPreset, PASSIVE_TARSAL_LINKS, Skeleton,
)
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly
from flygym.compose.world.flat_ground import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import (
    CPGController, make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

TUTORIAL = dict(
    joint_preset=JointPreset.LEGS_ONLY,
    axis_order=AxisOrder.YAW_PITCH_ROLL,
    joint_stiffness=0.05, joint_damping=0.06,
    tarsus_override=True,
    actuator_gain=45.0, forcerange=(-65.0, 65.0),
    spawn_z=0.5,
)

# Each entry: one setting changed from TUTORIAL to our value.
OURS = dict(
    joint_preset=JointPreset.ALL_BIOLOGICAL,
    axis_order=AxisOrder.ROLL_PITCH_YAW,
    joint_stiffness=10.0, joint_damping=0.5,
    tarsus_override=False,
    actuator_gain=50.0, forcerange=(-30.0, 30.0),
    spawn_z=0.7,
)


def build(cfg):
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(cfg["axis_order"])
    skeleton = Skeleton(axis_order=cfg["axis_order"], joint_preset=cfg["joint_preset"])
    fly = NeuroMechFly(name="nmf")
    joints = fly.add_joints(skeleton, neutral_pose=neutral_pose,
                            stiffness=cfg["joint_stiffness"],
                            damping=cfg["joint_damping"])
    if cfg["tarsus_override"]:
        for jointdof, joint in joints.items():
            if jointdof.child.link in PASSIVE_TARSAL_LINKS:
                joint.stiffness[0] = 7.5
                joint.damping[0] = 1e-2
    dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(dofs, ActuatorType.POSITION, neutral_input=neutral_pose,
                      kp=cfg["actuator_gain"], forcerange=cfg["forcerange"])
    fly.add_leg_adhesion(gain=40.0)
    world = FlatGroundWorld()
    world.add_fly(fly, [0.0, 0.0, cfg["spawn_z"]],
                  Rotation3D(format="quat", values=[1, 0, 0, 0]))
    world.compile()
    return fly, world, dofs


def trial(cfg, duration_s=2.0, seed=0):
    fly, world, dofs = build(cfg)
    sim = Simulation(world)
    controller = CPGController(
        cpg_network=make_tripod_cpg_network(sim.timestep, seed=seed),
        preprogrammed_steps=PreprogrammedSteps(), output_dof_order=dofs)
    sim.reset()
    sim.warmup()
    track, quats = [], []
    for i in range(int(duration_s / sim.timestep)):
        apply_locomotion_action(sim, fly.name, controller.step())
        sim.step()
        if i % 10 == 0:
            track.append(sim.get_body_positions(fly.name)[0].copy())
            quats.append(sim.get_body_rotations(fly.name)[0].copy())
    track, quats = np.array(track), np.array(quats)
    upright = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
    return {"speed_mm_s": float(track[-1, 0] - track[0, 0]) / duration_s,
            "dy_mm": float(track[-1, 1] - track[0, 1]),
            "upright_min": float(upright.min()),
            "upright_end": float(upright[-1]),
            "z_end": float(track[-1, 2])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="media/baselines/ablation.json")
    args = ap.parse_args()

    print(f"{'configuration':<34}{'speed':>10}{'dy':>8}{'upright_min':>13}{'z':>8}")
    base = trial(TUTORIAL, args.duration, args.seed)
    print(f"{'TUTORIAL (all upstream)':<34}{base['speed_mm_s']:>10.3f}"
          f"{base['dy_mm']:>8.2f}{base['upright_min']:>13.3f}{base['z_end']:>8.3f}")

    rows = {"tutorial": base}
    for key in OURS:
        cfg = dict(TUTORIAL)
        cfg[key] = OURS[key]
        if key == "joint_stiffness":
            cfg["joint_damping"] = OURS["joint_damping"]   # they belong together
        r = trial(cfg, args.duration, args.seed)
        rows[key] = r
        broke = "   <-- BREAKS WALKING" if (
            r["speed_mm_s"] < 0.5 * base["speed_mm_s"] or r["upright_min"] < 0.8) else ""
        label = f"+ ours: {key}"
        print(f"{label:<34}{r['speed_mm_s']:>10.3f}{r['dy_mm']:>8.2f}"
              f"{r['upright_min']:>13.3f}{r['z_end']:>8.3f}{broke}", flush=True)

    full = trial(dict(TUTORIAL, **OURS), args.duration, args.seed)
    rows["all_ours"] = full
    print(f"{'ALL ours (= build_free_fly)':<34}{full['speed_mm_s']:>10.3f}"
          f"{full['dy_mm']:>8.2f}{full['upright_min']:>13.3f}{full['z_end']:>8.3f}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
