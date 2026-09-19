"""Harness-mode mechanical sanity check for the NeuroMechFly body.

This drives each leg's coxa-pitch and femur-tibia-pitch joints with a
plain sine wave, alternating a tripod pattern (LF+RM+LH vs RF+LM+RH) by
phase. This is NOT a biological signal, NOT derived from the connectome,
and NOT meant to look like real fly walking — it's a synthetic test
signal whose only purpose is to confirm the mechanical pipeline (body,
actuators, harness, physics stepping, rendering) works before any real
neural data (see `connectome_driven_open_loop.py`) drives it. Explicitly
flagging this per this project's honesty rule: anything not from the fly
is a labeled addition.

The fly is held fixed in space (TetheredWorld/"harness mode" — see
fly_robot/bodies/neuromechfly.py) so nothing here tests actual locomotion
or ground contact; it only tests that commanding joint angles produces
the expected mechanical motion.

Usage:
    MUJOCO_GL=cgl python -m fly_robot.experiments.harness_sine_wave_test
"""

import argparse
from pathlib import Path

import numpy as np
from flygym import Simulation
from flygym.compose import KinematicPosePreset
from flygym.compose.fly.base_fly import ActuatorType
from flygym.anatomy import AxisOrder

from fly_robot.bodies.neuromechfly import build_harnessed_fly

# Tripod grouping: real fly tripod gait alternates these two groups.
# This is a real biological pattern (not invented by us), but here it's
# only used to phase-offset a synthetic sine wave, not derived from any
# neural simulation — see module docstring.
TRIPOD_GROUP_A = {"lf", "rm", "lh"}  # phase 0
TRIPOD_GROUP_B = {"rf", "lm", "rh"}  # phase pi

SWING_AMPLITUDE_RAD = 0.35  # ~20 degrees; arbitrary, just visibly swings the leg
FREQUENCY_HZ = 2.0  # arbitrary test frequency, not derived from any fly rhythm


def build_target_angle_function(fly, dof_order):
    """Returns f(t) -> array of shape (42,), the target angle for every
    actuated DOF at time t. All non-driven DOFs stay at the neutral pose;
    only each leg's coxa-pitch and femur-tibia-pitch oscillate."""
    neutral_lookup = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.ROLL_PITCH_YAW
    ).joint_angles_lookup_rad
    neutral = np.array([neutral_lookup.get(d.name, 0.0) for d in dof_order])

    driven_mask = np.zeros(len(dof_order), dtype=bool)
    phase_offset = np.zeros(len(dof_order))
    sign = np.zeros(len(dof_order))  # coxa and tibia swing in opposite senses for a leg-like motion

    for i, dof in enumerate(dof_order):
        leg = dof.child.pos if dof.child.pos in TRIPOD_GROUP_A | TRIPOD_GROUP_B else None
        is_coxa_pitch = dof.name.endswith("coxa-pitch") and dof.name.startswith("c_thorax")
        is_tibia_pitch = dof.name.startswith(f"{leg}_trochanterfemur-") and dof.name.endswith("tibia-pitch") if leg else False
        if leg and (is_coxa_pitch or is_tibia_pitch):
            driven_mask[i] = True
            phase_offset[i] = 0.0 if leg in TRIPOD_GROUP_A else np.pi
            sign[i] = 1.0 if is_coxa_pitch else -1.0

    omega = 2 * np.pi * FREQUENCY_HZ

    def target_angles(t: float) -> np.ndarray:
        angles = neutral.copy()
        wave = SWING_AMPLITUDE_RAD * np.sin(omega * t + phase_offset) * sign
        angles[driven_mask] += wave[driven_mask]
        return angles

    return target_angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation seconds")
    parser.add_argument("--out-dir", default="media/harness_sine_wave_test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fly, world, mj_model, mj_data, camera, opposite_camera, top_down_camera = build_harnessed_fly()
    sim = Simulation(world)
    renderer = sim.set_renderer([camera, opposite_camera, top_down_camera])

    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    target_angles = build_target_angle_function(fly, dof_order)

    sim.reset()
    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, target_angles(0.0))
    sim.warmup()

    n_steps = int(args.duration / sim.timestep)
    for step in range(n_steps):
        t = step * sim.timestep
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, target_angles(t))
        sim.step()
        sim.render_as_needed()

    video_paths = {
        camera: out_dir / "harness_sine_wave_test.mp4",
        opposite_camera: out_dir / "harness_sine_wave_test_opposite_side.mp4",
        top_down_camera: out_dir / "harness_sine_wave_test_top_down.mp4",
    }
    sim.renderer.save_video(video_paths)
    print(f"Simulated {args.duration}s ({n_steps} steps). Wrote {list(video_paths.values())}")
