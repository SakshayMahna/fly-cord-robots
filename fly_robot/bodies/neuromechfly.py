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
from flygym.compose.world.flat_ground import FlatGroundWorld
from flygym.utils.math import Rotation3D

# FlyGym's own default tracking camera (`add_tracking_camera()` with no
# args) sits at pos_offset=(-0.5,-7.5,5) with an xyaxes rotation of
# (1,0,0, 0,0.6,0.8) — i.e. on one side of the fly (Y is the fly's
# left-right axis; X forward, Z up, the standard body-frame convention).
# This is the same 180-degree-about-Z rotation of that setup, derived
# (not guessed — see docs/logs/2026-09-19.md) to view the OTHER side:
# position and both axis vectors get their x,y components negated
# (z unchanged), which is exactly what a rotation about the vertical
# axis does. Confirmed empirically to show real, distinct leg positions
# (pixel-diffed against a neutral-pose frame) rather than pointing at
# empty space, which a naive single-component sign flip did.
OPPOSITE_SIDE_CAMERA_POS_OFFSET = (0.5, 7.5, 5)
OPPOSITE_SIDE_CAMERA_ROTATION = Rotation3D("xyaxes", (-1, 0, 0, 0, -0.6, 0.8))

# Top-down view — standard framing in this research field (including
# NeuroMechFly's own papers) for seeing all six legs without wing
# occlusion, which both side cameras above suffer from. Derived, not
# guessed: xaxis=(1,0,0) [image-right = body forward], yaxis=(0,1,0)
# [image-up = body's +Y] => local Z = cross(X,Y) = (0,0,1), so the
# camera's view direction (-Z) points straight down. Confirmed visually
# (docs/logs/2026-09-19.md, Phase 3) to show all six legs clearly.
TOP_DOWN_CAMERA_POS_OFFSET = (0, 0, 12)
TOP_DOWN_CAMERA_ROTATION = Rotation3D("xyaxes", (1, 0, 0, 0, 1, 0))

# Position-actuator gain (torque per radian of angle error). 50 is FlyGym's
# own tutorial default (see tutorials/1a_basic_model_composition.ipynb) —
# not tuned by us yet. Revisit once we're driving real joint targets and
# can judge tracking quality against real leg kinematics.
DEFAULT_ACTUATOR_GAIN = 50


SPAWN_POS_MM = [0, 0, 0.7]  # mm; matches FlyGym tutorial's free-standing spawn height


def build_harnessed_fly(name: str = "nmf", gain: float = DEFAULT_ACTUATOR_GAIN,
                         world=None):
    """Compose a NeuroMechFly instance with only leg DOFs actuated, fixed
    in space (harness/tethered mode — `TetheredWorld` "holds the body in
    place... useful for motor control experiments without locomotion",
    per FlyGym's own docstring).

    `world` defaults to a plain `TetheredWorld` (legs in air). Pass a
    `TetheredBallWorld` instead for the ground condition — the body is
    held identically either way, so the two conditions differ in exactly
    one thing: whether the feet have a substrate. See
    `fly_robot.bodies.tethered_ball`.

    Returns (fly, world, mj_model, mj_data, camera, opposite_camera,
    top_down_camera) — three tracking cameras (two side, one top-down),
    since a single leg's real movement can be occluded from either side
    angle alone (see docs/logs/2026-09-19.md, Phase 3) — the top-down
    view is the one that reliably shows all six legs.
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
    opposite_camera = fly.add_tracking_camera(
        name="opposite_side_cam",
        pos_offset=OPPOSITE_SIDE_CAMERA_POS_OFFSET,
        rotation=OPPOSITE_SIDE_CAMERA_ROTATION,
    )
    top_down_camera = fly.add_tracking_camera(
        name="top_down_cam",
        pos_offset=TOP_DOWN_CAMERA_POS_OFFSET,
        rotation=TOP_DOWN_CAMERA_ROTATION,
    )

    if world is None:
        world = TetheredWorld()
    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])
    world.add_fly(fly, list(SPAWN_POS_MM), spawn_rot)

    mj_model, mj_data = world.compile()
    return fly, world, mj_model, mj_data, camera, opposite_camera, top_down_camera


# Leg adhesion gain. NeuroMechFly v2 walks the fly freely on ground with
# tarsal adhesion; without it the feet slip and the model cannot generate
# propulsion. The actuator is FlyGym's own (`add_leg_adhesion`), but the
# GAIN and the always-on control policy below are OUR CHOICE and are
# flagged as such — see `docs/trained_adapter/DESIGN.md`.
DEFAULT_ADHESION_GAIN = 40.0


def build_free_fly(name: str = "nmf", gain: float = DEFAULT_ACTUATOR_GAIN,
                   adhesion_gain: float = DEFAULT_ADHESION_GAIN,
                   spawn_pos_mm=None,
                   joint_stiffness: float | None = None,
                   joint_damping: float | None = None,
                   forcerange: tuple | None = None):
    """A fly free to walk on flat ground, with tarsal adhesion enabled.

    The Stage A training rig, replacing the tethered ball. The ball is kept
    for Phase 4's records and for video, but it is not a sound training
    substrate: its damping of 1e-6 models a frictionless air bearing, so an
    impulse persists for the whole trial (measured coast-down time constant
    ~11,700 s against a 4 s trial). Time-averaged ball velocity therefore
    rewards a single kick as though it were sustained walking, which is
    exactly the exploit the first pilot found. On ground the body stops
    when the legs stop, so displacement cannot be banked.

    Differences from `build_harnessed_fly`: `FlatGroundWorld` (which gives
    the fly a freejoint — "Flies are free to move", its own docstring) and
    six tarsal adhesion actuators. Body, joints, actuators and all three
    cameras are otherwise identical, so neural-side results remain
    comparable across rigs.

    Returns the same 7-tuple as the other builders.
    """
    fly = NeuroMechFly(name=name)
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.ROLL_PITCH_YAW
    )
    neutral_pose = KinematicPosePreset.NEUTRAL
    # FlyGym 2.x applies stiffness=10 / damping=0.5 to EVERY joint. FlyGym
    # 1.x -- whose CPG baseline and reference kinematics we port -- uses
    # 0.05 / 0.06 for actuated leg joints. Those passive springs are 200x
    # stiffer than the controller was designed against and fight the
    # position actuators directly. Left at the 2.x default here so nothing
    # existing changes; the baseline passes upstream's values explicitly.
    joint_kwargs = {}
    if joint_stiffness is not None:
        joint_kwargs["stiffness"] = joint_stiffness
    if joint_damping is not None:
        joint_kwargs["damping"] = joint_damping
    fly.add_joints(skeleton, neutral_pose=neutral_pose, **joint_kwargs)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    # FlyGym 2.x limits actuator force to +-30; FlyGym 1.x, which the
    # ported CPG was tuned against, allows +-65. At +-30 the legs are
    # torque-limited and cannot hold the body up under the reference gait.
    act_kwargs = {} if forcerange is None else {"forcerange": tuple(forcerange)}
    fly.add_actuators(actuated_dofs, actuator_type="position",
                      neutral_input=neutral_pose, kp=gain, **act_kwargs)
    fly.add_leg_adhesion(gain=adhesion_gain)
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    camera = fly.add_tracking_camera()
    opposite_camera = fly.add_tracking_camera(
        name="opposite_side_cam",
        pos_offset=OPPOSITE_SIDE_CAMERA_POS_OFFSET,
        rotation=OPPOSITE_SIDE_CAMERA_ROTATION,
    )
    top_down_camera = fly.add_tracking_camera(
        name="top_down_cam",
        pos_offset=TOP_DOWN_CAMERA_POS_OFFSET,
        rotation=TOP_DOWN_CAMERA_ROTATION,
    )

    world = FlatGroundWorld()
    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])
    # Spawn height is pose-dependent: the model's NEUTRAL pose rests at the
    # default, but a different standing pose (e.g. the reference-kinematics
    # neutral) needs its own height or the fly starts on tiptoe and topples.
    world.add_fly(fly, list(spawn_pos_mm or SPAWN_POS_MM), spawn_rot)
    mj_model, mj_data = world.compile()
    return fly, world, mj_model, mj_data, camera, opposite_camera, top_down_camera


def build_ball_fly(name: str = "nmf", gain: float = DEFAULT_ACTUATOR_GAIN, **ball_kwargs):
    """Same body and cameras as `build_harnessed_fly`, standing on a
    freely-rotating sphere — the standard fly-on-ball rig. See
    `fly_robot.bodies.tethered_ball` for why this is the ground condition.
    """
    from fly_robot.bodies.tethered_ball import TetheredBallWorld

    return build_harnessed_fly(name=name, gain=gain, world=TetheredBallWorld(**ball_kwargs))
