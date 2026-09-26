"""The coupled loop: connectome and body stepping together.

Everything before this ran in one direction. Phase 3 computed the whole
neural trace first and then replayed it into the body; nothing the legs
did could reach the neurons. Here the two advance in lockstep:

    neurons -> motor-neuron rates -> joint targets -> physics
       ^                                                 |
       |______ sensory current <- joint angle/velocity __|

One iteration is one **neural step** of `dt = 0.001 s`, inside which the
physics takes 10 substeps of `0.0001 s` (an exact integer ratio, asserted
below). Sensory current computed at the top of a neural step is held
constant across it — a zero-order hold, the same convention the motor
side already uses.

The loop is deliberately a thin composition of pieces that are each
validated on their own:

  * `SteppableRateModel` — Pugliese's rate equation, connectome and
    neuron parameters unchanged, reproduced against their published
    output to median r = 0.9993 (`validate_steppable_model.py`).
  * `motor_neuron_to_joint` — FROZEN for all of this work, config hash
    `04be9dec...`; imported and called, never modified.
  * `joint_to_sensory_neuron` — the sensory interface, with locality,
    determinism and `g_fb = 0` gates in `tests/`.
  * `TetheredWorld` / `TetheredBallWorld` — harness and ground.

Nothing here touches the connectome. Sensory feedback enters exactly
where the stimulation current already enters: as an additive term in the
input vector `I`, leaving W untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from flygym import Simulation
from flygym.compose import KinematicPosePreset
from flygym.compose.fly.base_fly import ActuatorType
from flygym.anatomy import AxisOrder

from fly_robot.bodies.neuromechfly import (
    build_ball_fly, build_free_fly, build_harnessed_fly,
)
from fly_robot.interface.joint_to_sensory_neuron import SensoryEncoder
from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ,
    LEG_NAME_TO_FLYGYM_PREFIX, MotorNeuronGroups, _dof_name_for_pair,
    leg_motor_row_indices,
)

NEURAL_DT = 0.001
TRIAL_DURATION_S = 4.0       # pre-registered

# Body z-axis vertical component below which the fly counts as flipped.
# 1 is level, 0 on its side, negative inverted. A standing fly reads 0.998
# and the walking CPG baseline never drops below 0.966, so 0.5 (60 degrees
# off vertical) is well clear of normal walking.
FLIP_UPRIGHT_THRESHOLD = 0.5

# Pulse timing (stimulation window) is NOT set here — it comes from
# `sim_params.pulse_start` / `pulse_end`, built into `neural_model` by
# `trial_setup.build_trial_components` from the actual Hydra config. Two
# stray constants duplicating those values used to live here unused;
# removed rather than left as a second, driftable source of truth.


@dataclass
class TrialResult:
    """Everything one trial produces. Arrays are (n_neural_steps, ...)."""
    motor_rates: np.ndarray        # (n_legs, n_steps) summed per-leg motor rate
    joint_angles: np.ndarray       # (n_steps, n_jointdofs)
    joint_velocities: np.ndarray   # (n_steps, n_jointdofs)
    sensory_drive: np.ndarray      # (n_channels, n_steps) per-group drive in [0,1]
    sensory_channels: list         # channel keys, matching sensory_drive rows
    ball_quat: np.ndarray | None   # (n_steps, 4) or None in the harness
    ball_angvel: np.ndarray | None # (n_steps, 3) or None in the harness
    n_active_neurons: int
    max_firing_rate: float
    wall_clock_s: float
    # `n_active_neurons` is the final state, retained for compatibility with
    # the pre-registered filter. These two fields expose transient seizures
    # that used to disappear before the final sample and evade diagnosis.
    peak_n_active_neurons: int | None = None
    fraction_saturated_steps: float = 0.0
    thorax_pos: np.ndarray | None = None   # (n_steps, 3) mm, free-ground rig only
    thorax_quat: np.ndarray | None = None  # (n_steps, 4), free-ground rig only
    terminated_at_step: int | None = None  # set when the fly flipped over
    n_steps_planned: int = 0
    adhesion: np.ndarray | None = None     # (n_steps, 6) per-leg adhesion state
    anatomical_pools: dict | None = None   # {module: (n_steps, 6)} logged, unused by the gate
    unstable: bool = False
    instability_reason: str = ""
    meta: dict = field(default_factory=dict)


def _joint_targets_from_rates(rates: np.ndarray, groups: MotorNeuronGroups,
                              dof_index_by_name: dict, neutral_angles: np.ndarray,
                              gain_rad: float, rate_scale_hz: float) -> np.ndarray:
    """Single-timestep form of the FROZEN motor interface.

    `compute_joint_targets` works on a whole precomputed (n_neurons,
    n_timesteps) trace, which a closed loop does not have — the next
    timestep does not exist yet. This applies the identical rule to one
    instantaneous rate vector: same antagonist pairs, same
    `neutral + gain * tanh((pos - neg) / scale)`, same constants imported
    from the frozen module rather than restated. Equivalence to
    `compute_joint_targets` is asserted in `tests/test_closed_loop.py`.
    """
    targets = neutral_angles.copy()
    for (segment, side), leg_prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        for pos_module, neg_module, suffix in ANTAGONIST_PAIRS:
            dof_idx = dof_index_by_name.get(_dof_name_for_pair(leg_prefix, suffix))
            if dof_idx is None:
                continue
            pos_idxs = groups.indices_by_group.get((segment, side, pos_module), [])
            neg_idxs = groups.indices_by_group.get((segment, side, neg_module), [])
            if not pos_idxs and not neg_idxs:
                continue
            pos_rate = rates[pos_idxs].mean() if pos_idxs else 0.0
            neg_rate = rates[neg_idxs].mean() if neg_idxs else 0.0
            targets[dof_idx] = neutral_angles[dof_idx] + gain_rad * np.tanh(
                (pos_rate - neg_rate) / rate_scale_hz)
    return targets


_CPG_ROWS_CACHE = None


def _cpg_rows_by_flygym_leg(
        circuit_csv: str = "data/circuit_map/all_legs_circuit.csv"):
    """{flygym leg prefix: [CPG triad row indices]}, loaded once per process.

    Needs the neuron table to map bodyIds to row indices; that load is
    cached because `run_trial` is called thousands of times per training
    run and the mapping never changes.
    """
    global _CPG_ROWS_CACHE
    if _CPG_ROWS_CACHE is None:
        from src.utils.path_utils import create_fresh_config_with_paths
        from src.utils.sim_utils import load_wTable

        from fly_robot.adapter.cpg_groups import build_cpg_groups
        from fly_robot.interface.motor_neuron_to_joint import (
            LEG_NAME_TO_FLYGYM_PREFIX)
        cfg = create_fresh_config_with_paths(
            experiment="FullVNC_DNg100_Stim", paths_template="fly_robot",
            run_id="cpg_rows_lookup")
        wtable = load_wTable(cfg.experiment.dfPath)
        by_leg = build_cpg_groups(circuit_csv, wtable)
        _CPG_ROWS_CACHE = {prefix: by_leg.get(leg, [])
                           for leg, prefix in LEG_NAME_TO_FLYGYM_PREFIX.items()}
    return _CPG_ROWS_CACHE


def run_trial(neural_model, motor_groups: MotorNeuronGroups,
              sensory_groups=None, feedback_gain: float = 0.0,
              on_ball: bool = False, duration_s: float = TRIAL_DURATION_S,
              neural_dt: float = NEURAL_DT,
              leg_permutation: dict | None = None,
              substitute_drive=None,
              encoder_mode: str = "signed",
              initial_joint_noise_rad: float = 0.0,
              seed: int = 0,
              motor_gain_rad: float = DEFAULT_GAIN_RAD,
              motor_rate_scale_hz: float = DEFAULT_RATE_SCALE_HZ,
              video_paths: dict | None = None,
              adapter=None, rig: str | None = None,
              adapter_sensory: bool = True,
              adhesion: bool = True,
              terminate_on_flip: bool = True) -> TrialResult:
    """Run one coupled trial.

    `feedback_gain = 0` makes the loop open: the sensory encoder emits
    exactly zero current (not merely small), so the neural trajectory is
    bit-identical to an open-loop run at the same seed. That equivalence
    is a pre-registered gate, tested in `tests/test_closed_loop.py`.

    `substitute_drive(step, channel_keys) -> dict` replaces the computed
    per-channel sensory drive — the hook condition C2 uses to inject
    phase-randomised surrogates through the same path as the real signal.

    `leg_permutation` is C3's deliberate locality violation; leave it None
    everywhere else.

    `adapter_sensory=False` is the R1a output-side condition. The adapter
    still provides descending drive, motor decoding and adhesion, but no body
    state enters the connectome. This must be explicit rather than achieved
    by hoping CMA-ES happens to set sensory gains near zero.

    `video_paths`: optional {camera_name: output_path} to render this
    trial, camera_name one of "side", "opposite_side", "top_down" (the
    three `build_harnessed_fly`/`build_ball_fly` already set up — see
    `bodies/neuromechfly.py`). E.g. `{"top_down": "out.mp4"}`. `run_trial`
    builds the body internally, so the camera OBJECTS don't exist until
    then — hence naming rather than passing camera instances.

    None (the default, used by every pre-registered condition and every
    gate in `tests/test_closed_loop.py`) skips rendering entirely —
    attaching a renderer is not free, so trials run for data rather than
    viewing never pay for it. Rendering does not touch the neural or
    physics path, only when frames are captured, so it cannot affect
    `TrialResult`'s numbers (the g_fb=0 gate does not need re-checking
    with video on).
    """
    rng = np.random.default_rng(seed)
    # `rig` supersedes `on_ball`; on_ball is kept so every Phase 4 call site
    # and gate keeps working unchanged.
    if rig is None:
        rig = "ball" if on_ball else "harness"
    if rig not in ("harness", "ball", "ground"):
        raise ValueError(f"rig must be harness/ball/ground, got {rig!r}")
    on_ball = rig == "ball"
    builder = {"harness": build_harnessed_fly, "ball": build_ball_fly,
               "ground": build_free_fly}[rig]
    fly, world, _mj_model, _mj_data, cam_side, cam_opposite, cam_top = builder()
    camera_by_name = {"side": cam_side, "opposite_side": cam_opposite, "top_down": cam_top}
    physics = Simulation(world)
    render_targets = None
    if video_paths is not None:
        render_targets = {camera_by_name[name]: path for name, path in video_paths.items()}
        physics.set_renderer(list(render_targets.keys()))

    physics_dt = physics.timestep
    substeps = round(neural_dt / physics_dt)
    assert abs(neural_dt / physics_dt - substeps) < 1e-9, (
        f"neural dt {neural_dt} is not an exact multiple of physics dt {physics_dt}; "
        "the zero-order-hold assumption breaks — do not proceed silently."
    )

    actuated = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    dof_index_by_name = {d.name: i for i, d in enumerate(actuated)}
    neutral_lookup = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.ROLL_PITCH_YAW).joint_angles_lookup_rad
    neutral_angles = np.array([neutral_lookup.get(d.name, 0.0) for d in actuated])

    all_jointdofs = fly.get_jointdofs_order()
    encoder = None
    if sensory_groups is not None and (adapter is None or adapter_sensory):
        if adapter is not None:
            # Phase 5 path: every encoder constant is a trained parameter.
            # `feedback_gain` and `encoder_mode` are ignored here because the
            # adapter supplies per-class gains and always uses the deviation
            # code — see fly_robot/adapter/apply.py.
            from fly_robot.adapter.apply import AdapterSensoryEncoder
            encoder = AdapterSensoryEncoder(
                sensory_groups, all_jointdofs,
                {d.name: neutral_lookup.get(d.name, 0.0) for d in all_jointdofs},
                params=adapter, leg_permutation=leg_permutation,
            )
        else:
            encoder = SensoryEncoder(
                sensory_groups, all_jointdofs,
                {d.name: neutral_lookup.get(d.name, 0.0) for d in all_jointdofs},
                gain=feedback_gain, leg_permutation=leg_permutation, mode=encoder_mode,
            )

    # The adapter delivers descending command drive through `extra_input`
    # rather than the model's pulse-gated `baseline_input`, so that its
    # onset ramp is under adapter control. Zeroing the baseline avoids
    # counting the same stimulation twice.
    adapter_alpha = saved_baseline = interleg = None
    if adapter is not None:
        from fly_robot.adapter.apply import (
            AdhesionGate, adapter_joint_targets, anatomical_pool_rates,
            command_current, motor_filter_alpha, per_leg_summed_rate,
        )
        saved_baseline = neural_model.baseline_input
        neural_model.baseline_input = np.zeros_like(saved_baseline)
        adapter_alpha = motor_filter_alpha(adapter, dof_index_by_name, neural_dt)
        adhesion_gate = AdhesionGate(adapter, dt=neural_dt)
        # Interleg phase coupling — OUR ADDITION, and OFF unless the adapter
        # carries a non-zero coupling_strength (default is exactly 0, so this
        # is the uncoupled model bit for bit). Phase is read from connectome
        # motor output only; no body sensor is involved, so the no-bypass
        # rule holds. See adapter/interleg_coupling.py.
        if hasattr(adapter, "coupling_strength"):
            from fly_robot.adapter.interleg_coupling import InterlegCoupling
            from fly_robot.adapter.apply import FLYGYM_LEG_ORDER
            interleg = InterlegCoupling(adapter, _cpg_rows_by_flygym_leg(),
                                        FLYGYM_LEG_ORDER, dt=neural_dt)

    ball_reader = None
    if on_ball:
        from fly_robot.bodies.tethered_ball import read_ball_state
        ball_reader = read_ball_state

    leg_rows = leg_motor_row_indices(motor_groups)
    leg_keys = [(seg, side) for seg in ("T1", "T2", "T3") for side in ("LHS", "RHS")]

    neural_model.reset()
    if adapter is not None:
        start_targets = adapter_joint_targets(
            neural_model.rates, motor_groups, dof_index_by_name, neutral_angles,
            adapter)
    else:
        start_targets = _joint_targets_from_rates(
            neural_model.rates, motor_groups, dof_index_by_name, neutral_angles,
            motor_gain_rad, motor_rate_scale_hz)
    if initial_joint_noise_rad > 0:
        start_targets = start_targets + rng.normal(0, initial_joint_noise_rad,
                                                    start_targets.shape)
    physics.reset()
    physics.set_actuator_inputs(fly.name, ActuatorType.POSITION, start_targets)
    physics.warmup()

    # "deviation" mode measures drive relative to the body's SETTLED pose,
    # so capture it now — after warmup, before stimulation onset. Doing it
    # here rather than hard-coding a value keeps it a measured property of
    # the body configuration (which differs between harness and ball) and
    # not a tuned parameter.
    if encoder is not None:
        encoder.set_rest_angles(physics.get_joint_angles(fly.name))

    n_steps = int(round(duration_s / neural_dt))
    channel_keys = sensory_groups.group_keys() if sensory_groups is not None else []

    motor_rates = np.zeros((len(leg_keys), n_steps), dtype=np.float32)
    joint_angles = np.zeros((n_steps, len(all_jointdofs)), dtype=np.float32)
    joint_velocities = np.zeros((n_steps, len(all_jointdofs)), dtype=np.float32)
    sensory_drive = np.zeros((len(channel_keys), n_steps), dtype=np.float32)
    ball_quat = np.zeros((n_steps, 4), dtype=np.float32) if on_ball else None
    ball_angvel = np.zeros((n_steps, 3), dtype=np.float32) if on_ball else None
    on_ground = rig == "ground"
    thorax_pos = np.zeros((n_steps, 3), dtype=np.float32) if on_ground else None
    thorax_quat = np.zeros((n_steps, 4), dtype=np.float32) if on_ground else None
    if on_ground and adhesion and adapter is None:
        # No adapter: hold adhesion on, which is only used by rig checks.
        # With an adapter it is gated per step from the connectome's own
        # stance/swing pools -- see adapter/apply.py.
        physics.set_leg_adhesion_states(fly.name, np.ones(6))
    adhesion_log = np.zeros((n_steps, 6), dtype=np.float32) if on_ground else None
    # The anatomically named pools are logged but never drive the gate, so
    # RESULTS can say whether training ever recruits them.
    anatomical_log = None
    if adapter is not None:
        from fly_robot.adapter.apply import ANATOMICAL_POOLS
        anatomical_log = {m: np.zeros((n_steps, 6), dtype=np.float32)
                          for m in ANATOMICAL_POOLS}

    max_rate = 0.0
    peak_n_active = 0
    saturated_steps = 0
    terminated_at_step = None
    unstable, reason = False, ""
    t0 = time.time()

    for step in range(n_steps):
        angles = physics.get_joint_angles(fly.name)
        velocities = physics.get_joint_velocities(fly.name)
        joint_angles[step] = angles
        joint_velocities[step] = velocities

        # --- body -> neurons -------------------------------------------------
        extra_input = None
        if encoder is not None:
            drives = encoder.channel_drives(angles, velocities)
            if substitute_drive is not None:
                drives = substitute_drive(step, channel_keys)
            for c, key in enumerate(channel_keys):
                sensory_drive[c, step] = drives[key]
            if adapter is not None or feedback_gain != 0.0:
                extra_input = encoder.input_current(angles, velocities, drives=drives)
        if adapter is not None:
            command = command_current(
                adapter, neural_model.neurons.n_neurons, neural_model.t,
                neural_model.pulse_start, neural_model.pulse_end)
            extra_input = command if extra_input is None else extra_input + command
            if interleg is not None and interleg.enabled:
                coup = interleg.step(
                    per_leg_summed_rate(neural_model.rates, leg_rows),
                    neural_model.neurons.n_neurons)
                if coup is not None:
                    extra_input = coup if extra_input is None else extra_input + coup

        # --- neurons ---------------------------------------------------------
        rates = neural_model.step(neural_dt, extra_input=extra_input)
        step_max = float(rates.max())
        max_rate = max(max_rate, step_max)
        n_active_step = int((rates > 0.01).sum())
        peak_n_active = max(peak_n_active, n_active_step)
        saturated_steps += int(n_active_step > 1500)
        if not np.isfinite(rates).all():
            unstable, reason = True, f"non-finite firing rate at step {step}"
            break
        if step_max >= 999.0:
            unstable, reason = True, f"firing rate hit the 1000 Hz clip at step {step}"

        for leg_i, leg in enumerate(leg_keys):
            rows = leg_rows.get(leg, [])
            motor_rates[leg_i, step] = rates[rows].sum() if rows else 0.0

        # --- neurons -> body -------------------------------------------------
        if adapter is not None:
            raw_targets = adapter_joint_targets(
                rates, motor_groups, dof_index_by_name, neutral_angles, adapter)
            # first-order low-pass toward the raw target, per segment
            targets = start_targets + adapter_alpha * (raw_targets - start_targets)
            start_targets = targets
        else:
            targets = _joint_targets_from_rates(
                rates, motor_groups, dof_index_by_name, neutral_angles,
                motor_gain_rad, motor_rate_scale_hz)
        physics.set_actuator_inputs(fly.name, ActuatorType.POSITION, targets)
        if on_ground and adhesion and adapter is not None:
            leg_adhesion = adhesion_gate.step(per_leg_summed_rate(rates, leg_rows))
            adhesion_log[step] = leg_adhesion
            physics.set_leg_adhesion_states(fly.name, leg_adhesion)
        if adapter is not None and anatomical_log is not None:
            for module, vals in anatomical_pool_rates(rates, motor_groups).items():
                anatomical_log[module][step] = vals
        for _ in range(substeps):
            physics.step()
            if render_targets is not None:
                physics.render_as_needed()

        if ball_reader is not None:
            state = ball_reader(physics.mj_model, physics.mj_data)
            ball_quat[step] = state["quat"]
            ball_angvel[step] = state["angvel_rad_s"]
        if on_ground:
            # Body 0 is the root segment (c_thorax) -- verified by query.
            thorax_pos[step] = physics.get_body_positions(fly.name)[0]
            q = physics.get_body_rotations(fly.name)[0]
            thorax_quat[step] = q
            if terminate_on_flip:
                upright = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
                if upright < FLIP_UPRIGHT_THRESHOLD:
                    # A flipped fly cannot walk, and letting it flail for the
                    # rest of the trial only adds noise. The reward charges
                    # the remaining time as flipped with zero further
                    # progress, so stopping early is never an advantage.
                    terminated_at_step = step + 1
                    break

        if not np.isfinite(angles).all():
            unstable, reason = True, f"non-finite joint angle at step {step}"
            break

    elapsed = time.time() - t0
    n_active = int((neural_model.rates > 0.01).sum())
    if saved_baseline is not None:
        # The model is reused across trials in a worker; leaving it zeroed
        # would silently un-stimulate every later non-adapter trial.
        neural_model.baseline_input = saved_baseline

    if render_targets is not None:
        physics.renderer.save_video(render_targets)

    return TrialResult(
        motor_rates=motor_rates, joint_angles=joint_angles,
        joint_velocities=joint_velocities, sensory_drive=sensory_drive,
        sensory_channels=channel_keys, ball_quat=ball_quat, ball_angvel=ball_angvel,
        thorax_pos=thorax_pos, thorax_quat=thorax_quat,
        terminated_at_step=terminated_at_step, n_steps_planned=n_steps,
        adhesion=adhesion_log, anatomical_pools=anatomical_log,
        n_active_neurons=n_active, max_firing_rate=max_rate,
        wall_clock_s=elapsed, peak_n_active_neurons=peak_n_active,
        fraction_saturated_steps=saturated_steps / max(n_steps, 1),
        unstable=unstable, instability_reason=reason,
        meta={"n_steps": n_steps, "substeps_per_neural_step": substeps,
              "on_ball": on_ball, "rig": rig, "adhesion": bool(on_ground and adhesion),
              "feedback_gain": feedback_gain, "seed": seed,
              "encoder_mode": encoder_mode, "adapter_sensory": adapter_sensory},
    )
