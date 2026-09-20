"""Gates for the coupled neural-physics loop.

The decisive one is `test_gain_zero_reproduces_open_loop_exactly`: with
`g_fb = 0` the closed-loop runner must produce a neural trajectory
bit-identical to an open-loop run at the same seed. If that fails, every
closed-loop result is confounded by something the loop itself is doing
rather than by feedback.

Run:
    python -m pytest tests/test_closed_loop.py -v
"""

import numpy as np
import pytest

from fly_robot.interface.motor_neuron_to_joint import (
    DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ, compute_joint_targets,
)
from fly_robot.sim.closed_loop import _joint_targets_from_rates, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

SHORT_S = 0.15  # keeps these gates to seconds while still exercising the loop


@pytest.fixture(scope="module")
def components():
    return build_trial_components(replicate=0, param_seed=PILOT_PARAM_SEED,
                                  duration_s=SHORT_S)


class _DOF:
    def __init__(self, name):
        self.name = name


def test_single_step_motor_rule_matches_frozen_batch_rule(components):
    """The loop needs a one-timestep form of the motor interface, because
    the next timestep does not exist yet. It must compute exactly what the
    frozen batch function would have."""
    _model, motor_groups, _sensory, wtable, _info = components
    rng = np.random.default_rng(0)

    dof_names = []
    for prefix in ("lf", "lm", "lh", "rf", "rm", "rh"):
        dof_names += [f"c_thorax-{prefix}_coxa-pitch",
                      f"{prefix}_coxa-{prefix}_trochanterfemur-pitch",
                      f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]
    dofs = [_DOF(n) for n in dof_names]
    neutral = rng.normal(0, 0.2, len(dofs))
    dof_index = {d.name: i for i, d in enumerate(dofs)}

    n_neurons = len(wtable)
    rates = np.zeros((n_neurons, 3))
    for module_rows in motor_groups.indices_by_group.values():
        for row in module_rows:
            rates[row] = rng.uniform(0, 12, 3)

    batch = compute_joint_targets(rates, motor_groups, dofs, neutral,
                                  gain_rad=DEFAULT_GAIN_RAD,
                                  rate_scale_hz=DEFAULT_RATE_SCALE_HZ)
    for t in range(3):
        single = _joint_targets_from_rates(
            rates[:, t], motor_groups, dof_index, neutral,
            DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ)
        np.testing.assert_allclose(single, batch[t], rtol=0, atol=1e-12)


def test_gain_zero_reproduces_open_loop_exactly(components):
    """THE gate. Closed loop at g_fb = 0 vs the same loop with no sensory
    interface at all: the neural output must match bit-for-bit."""
    model, motor_groups, sensory_groups, _wtable, _info = components

    open_loop = run_trial(model, motor_groups, sensory_groups=None,
                          feedback_gain=0.0, on_ball=False,
                          duration_s=SHORT_S, seed=0)
    closed_zero = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                            feedback_gain=0.0, on_ball=False,
                            duration_s=SHORT_S, seed=0)

    assert np.array_equal(open_loop.motor_rates, closed_zero.motor_rates), (
        "g_fb = 0 did not reproduce open loop bit-identically — the loop is "
        "perturbing the network even with feedback disabled"
    )
    assert np.array_equal(open_loop.joint_angles, closed_zero.joint_angles)


def test_gain_zero_on_ball_also_reproduces_its_own_open_loop(components):
    """Same gate in the ground condition — the ball must not leak into the
    neural path when feedback is off."""
    model, motor_groups, sensory_groups, _wtable, _info = components
    a = run_trial(model, motor_groups, sensory_groups=None, feedback_gain=0.0,
                  on_ball=True, duration_s=SHORT_S, seed=0)
    b = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                  feedback_gain=0.0, on_ball=True, duration_s=SHORT_S, seed=0)
    assert np.array_equal(a.motor_rates, b.motor_rates)


def test_trial_is_deterministic(components):
    """Same seed and config, twice, identical results."""
    model, motor_groups, sensory_groups, _wtable, _info = components
    kwargs = dict(sensory_groups=sensory_groups, feedback_gain=5.0,
                  on_ball=True, duration_s=SHORT_S, seed=3)
    first = run_trial(model, motor_groups, **kwargs)
    second = run_trial(model, motor_groups, **kwargs)
    assert np.array_equal(first.motor_rates, second.motor_rates)
    assert np.array_equal(first.joint_angles, second.joint_angles)
    assert np.array_equal(first.sensory_drive, second.sensory_drive)


def test_feedback_actually_changes_the_neural_trajectory(components):
    """A closed loop that produced the same neurons as the open one would
    mean the feedback path is not connected — the failure mode that would
    make every result a null by construction."""
    model, motor_groups, sensory_groups, _wtable, _info = components
    off = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                    feedback_gain=0.0, on_ball=True, duration_s=SHORT_S, seed=0)
    on = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                   feedback_gain=20.0, on_ball=True, duration_s=SHORT_S, seed=0)
    assert not np.array_equal(off.motor_rates, on.motor_rates)


def test_ball_state_recorded_only_on_ball(components):
    model, motor_groups, sensory_groups, _wtable, _info = components
    harness = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                        feedback_gain=1.0, on_ball=False, duration_s=SHORT_S, seed=0)
    ball = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                     feedback_gain=1.0, on_ball=True, duration_s=SHORT_S, seed=0)
    assert harness.ball_quat is None and harness.ball_angvel is None
    assert ball.ball_quat is not None and ball.ball_quat.shape[1] == 4
    assert ball.ball_angvel.shape[1] == 3


def test_no_instability_at_default_settings(components):
    model, motor_groups, sensory_groups, _wtable, _info = components
    result = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                       feedback_gain=10.0, on_ball=True, duration_s=SHORT_S, seed=0)
    assert not result.unstable, result.instability_reason
    assert np.isfinite(result.motor_rates).all()
    assert np.isfinite(result.joint_angles).all()
