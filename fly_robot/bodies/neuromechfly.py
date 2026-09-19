"""Build the fly body for Phase 2+, using FlyGym/NeuroMechFly v2
(EPFL Ramdya Lab, github.com/NeLy-EPFL/flygym, Apache-2.0) — NOT a
hand-built or generic hobby-robot hexapod.

Why this instead of building our own hexapod MJCF from scratch: it's an
anatomically accurate model of the real fly (micro-CT scan derived), with
7 real degrees of freedom per leg (thorax-coxa pitch/roll/yaw, coxa-femur
pitch/roll, femur-tibia pitch, tibia-tarsus pitch — matching actual fly
joints, not a simplified 3-DOF/leg hobby-robot approximation), MuJoCo
native, peer-reviewed (Nature Methods), and — notably for us — explicitly
designed with a "VNC-level motor control" plug-in point, which is exactly
our project's architecture (real connectome → motor commands → legs).
See docs/logs/2026-09-19.md for the full evaluation and license check.

This module only composes the BODY (harnessed, no locomotion yet) — no
gait, no neural signal. That's Phase 2's whole job: confirm the mechanical
setup (joints, actuators, harness) works before any real neural signal
(Phase 3) or ground contact / walking (Phase 5) enters the picture.
"""

from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import KinematicPosePreset, NeuroMechFly, TetheredWorld
from flygym.utils.math import Rotation3D

# Position-actuator gain (torque per radian of angle error). 50 is FlyGym's
# own tutorial default (see tutorials/1a_basic_model_composition.ipynb) —
# not tuned by us yet. Revisit once we're driving real joint targets and
# can judge tracking quality against real leg kinematics.
DEFAULT_ACTUATOR_GAIN = 50


def build_harnessed_fly(name: str = "nmf", gain: float = DEFAULT_ACTUATOR_GAIN):
    """Compose a NeuroMechFly instance with only leg DOFs actuated, fixed
    in space (harness/tethered mode — `TetheredWorld` "holds the body in
    place... useful for motor control experiments without locomotion",
    per FlyGym's own docstring).

    Returns (fly, world, mj_model, mj_data, camera).
    """
    fly = NeuroMechFly(name=name)

    # Articulate every biologically real DOF (so passive joints, e.g. wings,
    # behave physically under gravity/contact) but only ACTUATE the leg
    # DOFs — matching this phase's "harness mode" scope. See
    # ActuatedDOFPreset.LEGS_ACTIVE_ONLY vs JointPreset.ALL_BIOLOGICAL:
    # articulation (can it move) is broader than actuation (do we drive it).
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.ROLL_PITCH_YAW
    )
    neutral_pose = KinematicPosePreset.NEUTRAL
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated_dofs, actuator_type="position", neutral_input=neutral_pose, kp=gain)
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    camera = fly.add_tracking_camera()

    world = TetheredWorld()
    spawn_pos = [0, 0, 0.7]  # mm; matches FlyGym tutorial's free-standing spawn height
    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])
    world.add_fly(fly, spawn_pos, spawn_rot)

    mj_model, mj_data = world.compile()
    return fly, world, mj_model, mj_data, camera
