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

from flygym.anatomy import (
    ActuatedDOFPreset, AxisOrder, JointPreset, PASSIVE_TARSAL_LINKS, Skeleton,
)
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

# --- axis-order conventions -------------------------------------------------
# `AxisOrder` fixes the order in which a joint's three rotational DOFs
# compose. It is not cosmetic: feeding angles defined in one order into a
# model composed in the other inverts the fly. Measured (ablate_rig.py) --
# changing ONLY this in an otherwise-working rig takes the CPG baseline from
# +13.6 mm/s upright to +0.9 mm/s with upright -0.96, i.e. on its back.
#
# LEGACY_AXIS_ORDER is what Phases 2-4 were built and run under. Those
# results stay reproducible: the harness and ball builders keep it, and
# nothing about them changes.
#
# STAGE_A_AXIS_ORDER is what NeuroMechFly v2's own locomotion examples and
# the real-fly reference kinematics use, so it is what free walking needs.
LEGACY_AXIS_ORDER = AxisOrder.ROLL_PITCH_YAW
STAGE_A_AXIS_ORDER = AxisOrder.YAW_PITCH_ROLL

# The validated free-walking configuration, taken verbatim from
# `flygym_demo.complex_terrain.common.make_locomotion_fly` -- the setup the
# published tutorial 4a walks with. Adopted wholesale rather than
# approximated, so the connectome and both baselines share one body.
STAGE_A_FLY = dict(
    joint_preset=JointPreset.LEGS_ONLY,
    axis_order=STAGE_A_AXIS_ORDER,
    joint_stiffness=0.05,
    joint_damping=0.06,
    passive_tarsus_stiffness=7.5,
    passive_tarsus_damping=1e-2,
    actuator_gain=45.0,
    forcerange=(-65.0, 65.0),
    adhesion_gain=40.0,
    spawn_pos_mm=(0.0, 0.0, 0.5),
)


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


def build_free_fly(name: str = "nmf", colorize: bool = True, **overrides):
    """A fly free to walk on flat ground — the Stage A training rig.

    Configuration is `STAGE_A_FLY`, taken verbatim from FlyGym's own
    `flygym_demo.complex_terrain.common.make_locomotion_fly`, i.e. the setup
    tutorial 4a demonstrably walks with (+13.9 mm/s, upright >= 0.97, three
    seeds). Adopted wholesale rather than approximated so that the
    connectome, the CPG baseline and the rule-based baseline all run on one
    identical body.

    The change that mattered was `axis_order`. Everything else in our
    previous build was survivable; composing joints in ROLL_PITCH_YAW while
    driving them with YAW_PITCH_ROLL angles put the fly on its back
    (measured: +0.93 mm/s, upright -0.96). See `baselines/ablate_rig.py`.

    The ball and harness builders are untouched and keep
    `LEGACY_AXIS_ORDER`, so Phases 2-4 stay reproducible.

    Returns the same 7-tuple as the other builders.
    """
    cfg = {**STAGE_A_FLY, **overrides}

    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        cfg["axis_order"])
    skeleton = Skeleton(axis_order=cfg["axis_order"],
                        joint_preset=cfg["joint_preset"])
    fly = NeuroMechFly(name=name)
    joints = fly.add_joints(skeleton, neutral_pose=neutral_pose,
                            stiffness=cfg["joint_stiffness"],
                            damping=cfg["joint_damping"])
    # The passive tarsal links carry their own, much stiffer springs. MuJoCo
    # 3.7+ stores stiffness/damping as polynomial coefficients; the linear
    # term is index 0 (comment and approach both from upstream).
    for jointdof, joint in joints.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0] = cfg["passive_tarsus_stiffness"]
            joint.damping[0] = cfg["passive_tarsus_damping"]

    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated_dofs, actuator_type="position",
                      neutral_input=neutral_pose, kp=cfg["actuator_gain"],
                      forcerange=tuple(cfg["forcerange"]))
    fly.add_leg_adhesion(gain=cfg["adhesion_gain"])
    if colorize:
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
    world.add_fly(fly, list(cfg["spawn_pos_mm"]), spawn_rot)
    mj_model, mj_data = world.compile()
    return fly, world, mj_model, mj_data, camera, opposite_camera, top_down_camera


def build_ball_fly(name: str = "nmf", gain: float = DEFAULT_ACTUATOR_GAIN, **ball_kwargs):
    """Same body and cameras as `build_harnessed_fly`, standing on a
    freely-rotating sphere — the standard fly-on-ball rig. See
    `fly_robot.bodies.tethered_ball` for why this is the ground condition.
    """
    from fly_robot.bodies.tethered_ball import TetheredBallWorld

    return build_harnessed_fly(name=name, gain=gain, world=TetheredBallWorld(**ball_kwargs))
