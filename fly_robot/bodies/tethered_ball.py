"""Tethered-on-ball world: thorax fixed, legs on a freely-rotating sphere.

This is the standard *Drosophila* electrophysiology/imaging paradigm — the
fly is glued to a holder and walks on an air-supported foam ball, which
turns under its feet. It is the right "ground" condition for this project
because it gives the legs real ground contact, real load, and real
mechanical coupling *through the substrate*, while keeping the body fixed
so that:

  * the fly cannot fall over or wander out of frame across a long trial,
  * every trial starts from an identical body pose, and
  * the harness and ground conditions differ in exactly one thing —
    whether the feet touch a substrate — rather than also differing in
    whether the body is free.

FlyGym ships `TetheredWorld` (body rigidly fixed, legs in air) and
`FlatGroundWorld` (body completely free) with nothing in between, so the
ball is built here in MJCF directly.

Implementation: a sphere body carrying a MuJoCo **ball joint**, which has
three rotational degrees of freedom and no translational ones. The sphere
therefore spins freely about its own centre but cannot move or fall — the
same kinematics an air-bearing gives a real foam ball, without having to
model the air cushion.

Units throughout are FlyGym's: millimetres, grams, seconds (so gravity is
-9810 mm/s^2 and the fly masses 1.02e-3 g = 1.02 mg).

Nothing in this file touches the connectome or the motor interface.
"""

from __future__ import annotations

import mujoco
import numpy as np
from flygym.compose import TetheredWorld
from flygym.compose.physics import ContactParams
from flygym.compose.world.base_world import _GroundContactMixin

# A real fly-on-ball rig uses a ~6 mm expanded-polystyrene sphere for a
# ~2.5 mm fly, so radius 3.0 mm.
DEFAULT_BALL_RADIUS_MM = 3.0

# Expanded polystyrene is ~0.03 g/cm^3 = 3e-5 g/mm^3. A 3 mm sphere is
# (4/3)*pi*27 = 113 mm^3, so ~3.4e-3 g = 3.4 mg — about 3.3x the fly's own
# 1.02 mg. Real rigs deliberately use very light foam so the fly can
# actually turn the ball; this reproduces that ratio rather than inventing
# a convenient number.
DEFAULT_BALL_MASS_G = 3.4e-3

# Rotational damping at the ball joint, standing in for the residual drag
# of an air bearing. Small but nonzero: exactly zero would let numerical
# noise accumulate into unbounded spin over a long trial. Treated as an
# explicit modelling parameter, not a tuned one — held identical across
# every ground condition.
DEFAULT_BALL_DAMPING = 1e-6

# Ball centre, in world coordinates (the fly spawns at [0, 0, 0.7]).
#
# Fitted, then contact-tested -- not guessed. Three things had to be got
# right, each of which was wrong at some point (see docs/closed_loop/LOG.md):
#
# 1. Measure the tarsus tips in the SETTLED neutral pose: after
#    `Simulation.reset()`, warmup and 0.2 s of settling. A bare
#    `world.compile()` leaves a different pose, and fitting to that put
#    the ball ~2 mm out.
# 2. Fit (centre_x, centre_z) by minimising the worst
#    |distance-to-surface| over the six tips. At R = 3.0 mm the optimum is
#    (+0.161, 0, -1.742), holding all six within 0.079 mm. The small
#    FORWARD offset is what lets the front legs reach at all -- centred on
#    the body axis, only four legs ever touch.
# 3. Raise it until contact actually happens. Geometric tangency is not
#    contact: at the fitted height the four feet sitting 0.079 mm *above*
#    the surface register nothing and only the two middle legs touch. A
#    sweep in 0.04 mm steps found all six feet in contact, with zero
#    penetration, for centre_z in [-1.66, -1.62]; -1.64 is the middle of
#    that band, so it is the most tolerant choice.
DEFAULT_BALL_CENTER_MM = (0.16, 0.0, -1.64)


class TetheredBallWorld(_GroundContactMixin, TetheredWorld):
    """`TetheredWorld` plus a freely-rotating sphere under the fly.

    The fly's root segment is still held as a mocap body, so the thorax
    does not move; the only added degrees of freedom are the ball's three
    rotational ones.

    Contact is wired through FlyGym's own `_GroundContactMixin` with the
    ball registered as the "ground" geom, so foot-ball contact uses
    exactly the same `<pair>` elements, friction, solref/solimp and
    contact sensors as its flat-ground world. That matters: FlyGym does
    **not** use contype/conaffinity masks for the fly (every fly geom
    ships with `contype = conaffinity = 0`), it uses explicit contact
    pairs — so a sphere merely placed under the feet collides with
    nothing at all, silently. This was a real bug here before the mixin
    was wired in.
    """

    def __init__(self, name: str = "tethered_ball_world",
                 radius_mm: float = DEFAULT_BALL_RADIUS_MM,
                 mass_g: float = DEFAULT_BALL_MASS_G,
                 center_mm: tuple[float, float, float] = DEFAULT_BALL_CENTER_MM,
                 damping: float = DEFAULT_BALL_DAMPING,
                 rgba: tuple[float, float, float, float] = (0.85, 0.85, 0.88, 1.0)) -> None:
        super().__init__(name=name)
        self.radius_mm = float(radius_mm)
        self.mass_g = float(mass_g)
        self.center_mm = tuple(float(v) for v in center_mm)
        self.damping = float(damping)

        # FlyGym builds on MuJoCo's MjSpec API (not dm_control's mjcf), so
        # the world is assembled with add_body/add_joint/add_geom.
        body = self.mjcf_root.worldbody.add_body(name="ball", pos=list(self.center_mm))
        # A ball joint = 3 rotational DoF, 0 translational: the sphere spins
        # in place and cannot fall, which is what an air bearing achieves.
        body.add_joint(name="ball_joint", type=mujoco.mjtJoint.mjJNT_BALL,
                       damping=self.damping)
        self.ball_geom = body.add_geom(
            name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.radius_mm, 0, 0], mass=self.mass_g, rgba=list(rgba),
        )
        self.ball_body = body
        # `_GroundContactMixin` builds its contact pairs and sensors against
        # whatever is listed here; the ball plays the ground's role.
        self.ground_geoms = [self.ball_geom]

    @property
    def ball_top_z(self) -> float:
        return self.center_mm[2] + self.radius_mm

    def _attach_fly_mjcf(self, fly, spawn_position, spawn_rotation, *,
                         bodysegs_with_ground_contact=None,
                         ground_contact_params: ContactParams | None = None,
                         add_ground_contact_sensors: bool = True) -> set[str]:
        """Attach the fly tethered (mocap, no free joint) AND give its legs
        contact with the ball.

        Deliberately not `super()._attach_fly_mjcf(...)` down the MRO: the
        mixin's version adds a **free joint**, which is what makes the fly
        free to move in `FlatGroundWorld` and is exactly what a tethered
        world must not have. So the tether comes from `TetheredWorld` and
        only the contact wiring is borrowed from the mixin.
        """
        if bodysegs_with_ground_contact is None:
            # Only the distal leg segments touch the ball — the thorax and
            # head never reach it in a tethered rig, so pairing them would
            # add contact checks that can never fire.
            bodysegs_with_ground_contact = type(fly).CONTACT_BODIES_PRESET_CLASS(
                "tibia_tarsus_only"
            )
        if ground_contact_params is None:
            ground_contact_params = ContactParams()

        added = TetheredWorld._attach_fly_mjcf(self, fly, spawn_position, spawn_rotation)

        preset_cls = type(fly).CONTACT_BODIES_PRESET_CLASS
        if isinstance(bodysegs_with_ground_contact, str):
            bodysegs_with_ground_contact = preset_cls(bodysegs_with_ground_contact)
        if hasattr(bodysegs_with_ground_contact, "to_body_segments_list"):
            bodysegs_with_ground_contact = bodysegs_with_ground_contact.to_body_segments_list()

        self._set_ground_contact(fly, bodysegs_with_ground_contact, ground_contact_params)
        if add_ground_contact_sensors:
            self._add_ground_contact_sensors(fly, bodysegs_with_ground_contact)
        return added


def ball_joint_qpos_adr(mj_model) -> int:
    """Index into `mj_data.qpos` where the ball joint's orientation
    quaternion (4 values) starts."""
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    if jid < 0:
        raise ValueError("No 'ball_joint' in this model — is this a TetheredBallWorld?")
    return int(mj_model.jnt_qposadr[jid])


def ball_joint_dof_adr(mj_model) -> int:
    """Index into `mj_data.qvel` where the ball joint's angular velocity
    (3 values, rad/s about the ball's own axes) starts."""
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    if jid < 0:
        raise ValueError("No 'ball_joint' in this model — is this a TetheredBallWorld?")
    return int(mj_model.jnt_dofadr[jid])


def read_ball_state(mj_model, mj_data) -> dict:
    """Ball orientation (unit quaternion, w-first as MuJoCo stores it) and
    angular velocity (rad/s). Logged every trial as an extra output: it is
    the fly-on-ball rig's own measure of intended locomotion, so it is
    worth recording even though no hypothesis depends on it."""
    qadr = ball_joint_qpos_adr(mj_model)
    dadr = ball_joint_dof_adr(mj_model)
    return {
        "quat": np.array(mj_data.qpos[qadr:qadr + 4], dtype=float),
        "angvel_rad_s": np.array(mj_data.qvel[dadr:dadr + 3], dtype=float),
    }


def fit_ball_center_z(tarsus_tip_positions: np.ndarray, radius_mm: float,
                      center_xy: tuple[float, float] = (0.0, 0.0),
                      target_clearance_mm: float = 0.0) -> float:
    """Choose the ball's height so its surface sits against the feet.

    The neutral leg pose does NOT put all six tarsi at one height — the
    hind tarsi hang ~0.5 mm below the front ones — so there is no ball
    position that touches all six exactly. This picks the height at which
    the *median* foot sits on the surface, leaving some feet slightly
    above and some slightly into it, which the actuators then resolve.

    Returns the z of the ball centre such that the median tarsus tip lies
    `target_clearance_mm` above the sphere surface.
    """
    tips = np.asarray(tarsus_tip_positions, dtype=float)
    cx, cy = center_xy
    horizontal_sq = (tips[:, 0] - cx) ** 2 + (tips[:, 1] - cy) ** 2
    # For a tip at height z_t, the ball centre that puts it exactly on the
    # surface is z_t - sqrt(R^2 - horizontal^2).
    effective_r = np.sqrt(np.maximum((radius_mm - target_clearance_mm) ** 2 - horizontal_sq, 0.0))
    return float(np.median(tips[:, 2] - effective_r))
