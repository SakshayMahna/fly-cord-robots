"""FlyGym's CPG walking controller, ported onto our free-walking rig.

**Not our work.** `CPGNetwork` and its parameters are FlyGym 1.2.1's
(`examples/locomotion/cpg_controller.py`, Apache-2.0), reproduced faithfully.
What is ours: driving `build_free_fly` instead of FlyGym 1.x's
`SingleFlySimulation`, and the DOF-name mapping in `preprogrammed_steps`.

Its role here is threefold, which is why it was worth porting rather than
approximating:

  1. **The baseline** for the video comparison, alongside the rule-based
     controller, on the same body and terrain as the connectome.
  2. **The source of `v_ref`** — the reward's progress term is normalised
     by a real walking speed measured on this rig. Every previous attempt
     to supply that constant failed: the ball rig's -0.95 rad/s is not
     reproducible from committed code, and the synthetic tripod gait in
     `harness_sine_wave_test.py` does not walk on any substrate.
  3. **The rig's own validation.** If a published controller cannot walk
     here, the rig is wrong, not the connectome.

Adhesion follows FlyGym's convention exactly: OFF during swing, ON
otherwise, from `PreprogrammedSteps.get_adhesion_onoff`.
"""

from __future__ import annotations

import numpy as np

# Upstream's own defaults, verbatim from cpg_controller.py's __main__.
INTRINSIC_FREQS = np.ones(6) * 12.0
INTRINSIC_AMPS = np.ones(6) * 1.0
PHASE_BIASES = np.pi * np.array([
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
])
COUPLING_WEIGHTS = (PHASE_BIASES > 0) * 10.0
CONVERGENCE_COEFS = np.ones(6) * 20.0


def calculate_ddt(theta, r, w, phi, nu, R, alpha):
    """Upstream's oscillator derivatives, unchanged."""
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    return intrinsic_term + coupling_term, alpha * (R - r)


class CPGNetwork:
    """Six coupled oscillators, one per leg. Ported unchanged."""

    def __init__(self, timestep, intrinsic_freqs=None, intrinsic_amps=None,
                 coupling_weights=None, phase_biases=None,
                 convergence_coefs=None, init_phases=None,
                 init_magnitudes=None, seed=0):
        self.timestep = timestep
        self.intrinsic_freqs = (INTRINSIC_FREQS if intrinsic_freqs is None
                                else intrinsic_freqs)
        self.intrinsic_amps = (INTRINSIC_AMPS if intrinsic_amps is None
                               else intrinsic_amps)
        self.coupling_weights = (COUPLING_WEIGHTS if coupling_weights is None
                                 else coupling_weights)
        self.phase_biases = PHASE_BIASES if phase_biases is None else phase_biases
        self.convergence_coefs = (CONVERGENCE_COEFS if convergence_coefs is None
                                  else convergence_coefs)
        self.num_cpgs = self.intrinsic_freqs.size
        self.random_state = np.random.RandomState(seed)
        self.reset(init_phases, init_magnitudes)

    def step(self):
        dtheta, dr = calculate_ddt(
            theta=self.curr_phases, r=self.curr_magnitudes,
            w=self.coupling_weights, phi=self.phase_biases,
            nu=self.intrinsic_freqs, R=self.intrinsic_amps,
            alpha=self.convergence_coefs)
        self.curr_phases = self.curr_phases + dtheta * self.timestep
        self.curr_magnitudes = self.curr_magnitudes + dr * self.timestep

    def reset(self, init_phases=None, init_magnitudes=None):
        self.curr_phases = (self.random_state.random(self.num_cpgs) * 2 * np.pi
                            if init_phases is None else np.asarray(init_phases, float))
        self.curr_magnitudes = (np.zeros(self.num_cpgs) if init_magnitudes is None
                                else np.asarray(init_magnitudes, float))


# Upstream FlyGym 1.2.1 Fly() defaults, which the CPG was tuned against.
UPSTREAM_JOINT_STIFFNESS = 0.05
UPSTREAM_JOINT_DAMPING = 0.06
UPSTREAM_FORCERANGE = (-65.0, 65.0)
UPSTREAM_ACTUATOR_GAIN = 45.0


def run_cpg_walk(duration_s=2.0, seed=0, steps=None, video_paths=None,
                 record_every=10, joint_stiffness=UPSTREAM_JOINT_STIFFNESS,
                 joint_damping=UPSTREAM_JOINT_DAMPING,
                 forcerange=UPSTREAM_FORCERANGE, gain=UPSTREAM_ACTUATOR_GAIN):
    """Walk the free-ground fly under the ported CPG. Returns a dict of
    measurements: thorax track, per-leg phase, adhesion, contact."""
    from flygym import Simulation
    from flygym.compose.fly.base_fly import ActuatorType

    from fly_robot.baselines.preprogrammed_steps import PreprogrammedSteps
    from fly_robot.bodies.neuromechfly import build_free_fly

    steps = steps or PreprogrammedSteps()
    fly, world, _m, _d, cam_side, cam_opp, cam_top = build_free_fly(
        gain=gain, joint_stiffness=joint_stiffness,
        joint_damping=joint_damping, forcerange=forcerange)
    physics = Simulation(world)

    render_targets = None
    if video_paths:
        by_name = {"side": cam_side, "opposite_side": cam_opp, "top_down": cam_top}
        render_targets = {by_name[k]: v for k, v in video_paths.items()}
        physics.set_renderer(list(render_targets.keys()))

    dt = physics.timestep
    cpg = CPGNetwork(timestep=dt, seed=seed)

    actuated = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    index_of = {d.name: i for i, d in enumerate(actuated)}

    # Start from the data's own neutral pose rather than the model's, so the
    # controller is not fighting a different resting configuration.
    neutral = np.zeros(len(actuated))
    for name, angle in steps.default_pose_by_dof_name().items():
        if name in index_of:
            neutral[index_of[name]] = angle

    physics.reset()
    physics.set_actuator_inputs(fly.name, ActuatorType.POSITION, neutral)
    physics.warmup()

    n = int(duration_s / dt)
    track, quats, adhesion_log, phases_log = [], [], [], []
    targets = neutral.copy()
    for i in range(n):
        cpg.step()
        for name, angle in steps.targets_by_dof_name(
                cpg.curr_phases, cpg.curr_magnitudes).items():
            j = index_of.get(name)
            if j is not None:
                targets[j] = angle
        physics.set_actuator_inputs(fly.name, ActuatorType.POSITION, targets)
        physics.set_leg_adhesion_states(fly.name, steps.adhesion_by_leg(cpg.curr_phases))
        physics.step()
        if render_targets is not None:
            physics.render_as_needed()
        if i % record_every == 0:
            track.append(physics.get_body_positions(fly.name)[0].copy())
            quats.append(physics.get_body_rotations(fly.name)[0].copy())
            adhesion_log.append(steps.adhesion_by_leg(cpg.curr_phases))
            phases_log.append(cpg.curr_phases.copy() % (2 * np.pi))

    if render_targets is not None:
        physics.renderer.save_video(render_targets)

    return {
        "track": np.array(track), "quat": np.array(quats),
        "adhesion": np.array(adhesion_log), "phases": np.array(phases_log),
        "dt_record": dt * record_every, "duration_s": duration_s,
    }
