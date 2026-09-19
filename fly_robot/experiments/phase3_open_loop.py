"""Phase 3: open loop — real Pugliese/MANC motor-neuron rates -> joint
targets, no sensory feedback yet (that's Phase 4).

Pipeline:
1. Use Pugliese's own real published full-VNC simulation output — 128
   replicates, downloaded from Zenodo in Phase 0 — rather than running
   new simulations ourselves. Analysis showed a single leg's single
   motor-neuron module often has only a handful of neurons, each
   independently recruited with modest probability per replicate; a
   handful of our own fresh replicates can't average that out the way
   128 real ones can, at zero extra simulation cost (see
   docs/logs/2026-09-19.md for the investigation this replaced a
   smaller from-scratch run with, and why).
2. Map the resulting real motor-neuron firing rates onto FlyGym joint
   targets via fly_robot.interface.motor_neuron_to_joint (our own
   interface design — see that module's docstring for exactly what's
   mapped and what isn't, and why).
3. Drive the SAME harnessed FlyGym body from Phase 2 with these targets
   instead of the Phase 2 sanity sine wave.

Timestep note: Pugliese's neural sim runs at dt=0.001s; FlyGym's physics
runs at dt=0.0001s — an exact 10x ratio. Each neural sample is held
constant for 10 physics substeps (zero-order hold). This is a real
simplification (no interpolation, no explicit modeling of how a
continuous neural signal should map to discrete physics steps) — the
project's roadmap flags proper neural/physics dt synchronization as its
own future item (`fly_robot/sim/`); this is a placeholder for it, not
that final design.

Usage:
    MUJOCO_GL=cgl python -m fly_robot.experiments.phase3_open_loop
"""

import argparse
from pathlib import Path

import numpy as np
from flygym import Simulation
from flygym.compose import KinematicPosePreset
from flygym.compose.fly.base_fly import ActuatorType
from flygym.anatomy import AxisOrder

from fly_robot.bodies.neuromechfly import build_harnessed_fly
from fly_robot.interface.motor_neuron_to_joint import (
    build_motor_neuron_groups, compute_joint_targets,
)
from fly_robot.neural.run_pugliese_sim import (  # sets up sys.path for Pugliese's src/ on import
    load_published_full_vnc_replicates, select_representative_replicate,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="media/phase3")
    parser.add_argument(
        "--circuit-csv", default="data/circuit_map/all_legs_circuit.csv",
        help="Motor-neuron-to-module mapping from Phase 1 (identify_all_legs.py)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Pugliese's real published full-VNC replicates (Zenodo, no new simulation)...")
    wtable, R_stack, neural_dt = load_published_full_vnc_replicates()
    print(f"Loaded {R_stack.shape[0]} real replicates x {R_stack.shape[1]} neurons")
    R = select_representative_replicate(R_stack)
    print(f"Got real firing rates (one representative replicate's actual time-course — "
          f"NOT an average, see module docstring for why): "
          f"{R.shape[0]} neurons x {R.shape[1]} neural timesteps (dt={neural_dt}s)")

    groups = build_motor_neuron_groups(args.circuit_csv, wtable)
    n_mapped = sum(1 for v in groups.dof_name_by_group.values() if v is not None)
    n_groups = len(groups.dof_name_by_group)
    print(f"Motor-neuron modules mapped to a joint DOF: {n_mapped}/{n_groups} "
          f"(unmapped ones, e.g. femur reductor/substrate grip/tarsus control, stay at neutral)")

    fly, world, mj_model, mj_data, camera, opposite_camera, top_down_camera = build_harnessed_fly()
    physics_sim = Simulation(world)
    renderer = physics_sim.set_renderer([camera, opposite_camera, top_down_camera])

    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    neutral_lookup = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.ROLL_PITCH_YAW
    ).joint_angles_lookup_rad
    neutral_angles = np.array([neutral_lookup.get(d.name, 0.0) for d in dof_order])

    joint_targets = compute_joint_targets(R, groups, dof_order, neutral_angles)
    n_neural_steps = joint_targets.shape[0]

    physics_dt = physics_sim.timestep
    substeps_per_neural_step = round(neural_dt / physics_dt)
    assert abs(neural_dt / physics_dt - substeps_per_neural_step) < 1e-6, (
        f"Neural dt ({neural_dt}) is not an exact multiple of physics dt ({physics_dt}) — "
        "the zero-order-hold assumption above breaks; do not proceed silently."
    )
    print(f"Neural dt={neural_dt}s, physics dt={physics_dt}s -> "
          f"{substeps_per_neural_step} physics substeps per neural sample")

    physics_sim.reset()
    physics_sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, joint_targets[0])
    physics_sim.warmup()

    for neural_step in range(n_neural_steps):
        physics_sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, joint_targets[neural_step])
        for _ in range(substeps_per_neural_step):
            physics_sim.step()
            physics_sim.render_as_needed()

    video_paths = {
        camera: out_dir / "phase3_open_loop.mp4",
        opposite_camera: out_dir / "phase3_open_loop_opposite_side.mp4",
        top_down_camera: out_dir / "phase3_open_loop_top_down.mp4",
    }
    physics_sim.renderer.save_video(video_paths)
    print(f"Simulated {n_neural_steps * substeps_per_neural_step} physics steps "
          f"({n_neural_steps * neural_dt:.2f}s). Wrote {list(video_paths.values())}")
