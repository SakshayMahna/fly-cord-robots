"""Gates that keep the adapter trainer pointed at the WALKING task.

These exist because of a real, expensive mistake, recorded here so the
shape of it is not lost: 70 generations of CMA-ES were run, resumed five
times, and reported on, while the trainer was scoring **ball rotation on a
tethered fly** — `run_trial(..., on_ball=True)` with the ball reward
(`adapter.reward`, config hash 88f681a4...) — months after the decision to
move Stage A training to free-ground walking. The free-ground rig, the
ground reward (`adapter.reward_ground`, hash 0d648542...), the adhesion
gate and a ground-measured v_ref had all been built and documented; the
trainer simply never called any of them. Nothing failed loudly. The run
printed a reward hash that did not match the ground reward's, every
generation, and it was not checked.

The lesson is not "be careful" — it is that an objective must not be
selectable by a default that nobody re-reads. So:

  * the trainer refuses any rig that is not the ground walking rig,
  * stage and sensory-path must agree explicitly, never by coincidence,
  * "sensory off" must mean *zero* current into the connectome, verified
    numerically rather than assumed from a flag,
  * a checkpoint carries its whole training contract and refuses to resume
    under a different one, so two objectives cannot be spliced into one
    learning curve.

Run:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m pytest tests/test_ground_training_contract.py -v
"""

import numpy as np
import pytest

from fly_robot.adapter import reward_ground
from fly_robot.adapter.parameters import (
    N_PARAMS, PARAMETER_GROUPS, TRAINING_STAGES, parameter_indices_for_stage,
)
from fly_robot.training import cma_trainer
from fly_robot.training.cma_trainer import TrainConfig


# --- the objective the trainer actually optimises -------------------------

def test_trainer_uses_the_ground_reward_not_the_ball_reward():
    """The trainer's reward module must BE reward_ground.

    Checked by identity of the config hash rather than by import name, so
    that re-aliasing the import cannot make this pass while the objective
    silently changes.
    """
    assert cma_trainer.reward_mod is reward_ground
    assert (cma_trainer.reward_mod.reward_config_hash()
            == reward_ground.reward_config_hash())


def test_default_rig_is_ground():
    assert TrainConfig(run_name="_t").rig == "ground"


@pytest.mark.parametrize("bad_rig", ["ball", "tethered", "harness", ""])
def test_trainer_refuses_any_rig_but_ground(bad_rig):
    """The specific failure that happened: a ball-rig trial scored by a
    trainer whose name and docs said nothing about the ball."""
    cfg = TrainConfig(run_name="_t", rig=bad_rig)
    with pytest.raises(ValueError, match="free-ground"):
        cma_trainer.Trainer(cfg)


# --- stage / sensory-path agreement --------------------------------------

def test_output_stage_requires_sensory_off_and_vice_versa():
    """R1a's whole claim is that the connectome's OUTPUT can walk the body.
    If the sensory path were live, that claim would be untestable, so an
    inconsistent pair is refused rather than silently resolved."""
    with pytest.raises(ValueError, match="adapter_sensory"):
        cma_trainer.Trainer(TrainConfig(run_name="_t", stage="output",
                                        adapter_sensory=True))
    with pytest.raises(ValueError, match="adapter_sensory"):
        cma_trainer.Trainer(TrainConfig(run_name="_t", stage="sensory",
                                        adapter_sensory=False))


def test_stages_partition_the_parameter_vector_exactly():
    """No parameter may be missing from `full`, and output/sensory must not
    overlap — otherwise R1b would be re-training part of R1a's solution
    while being reported as a sensory-only comparison."""
    out = set(parameter_indices_for_stage("output").tolist())
    sen = set(parameter_indices_for_stage("sensory").tolist())
    assert out.isdisjoint(sen)
    assert out | sen == set(range(N_PARAMS))
    assert set(parameter_indices_for_stage("full").tolist()) == set(range(N_PARAMS))


def test_command_and_motor_and_adhesion_are_all_in_the_output_stage():
    """The output stage must contain exactly the blocks that turn connectome
    activity into body motion — and no sensory block."""
    names = set()
    for group in ("command", "motor", "adhesion"):
        names |= set(PARAMETER_GROUPS[group])
    assert TRAINING_STAGES["output"] == ("command", "motor", "adhesion")
    assert not (names & set(PARAMETER_GROUPS["sensory"]))


def test_sensory_stage_includes_no_command_or_motor_parameter():
    sensory = set(PARAMETER_GROUPS["sensory"])
    for group in ("command", "motor", "adhesion"):
        assert not (sensory & set(PARAMETER_GROUPS[group]))


# --- the numeric property, not just the flag ----------------------------

def test_sensory_off_injects_exactly_zero_current_into_the_connectome():
    """`adapter_sensory=False` must mean no body state reaches the network
    AT ALL — the no-bypass rule in its strongest form. Asserted on the
    recorded drive, because a flag that is read but not honoured is exactly
    the class of bug this file exists for.

    Runs two real 1 s coupled trials (~10 s); kept in the default suite
    deliberately, since a gate that only runs when someone remembers to ask
    for it would not have caught the mistake this file documents.
    """
    from fly_robot.adapter.parameters import default_params
    from fly_robot.sim.closed_loop import run_trial
    from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

    model, mg, sg, _wt, _info = build_trial_components(
        replicate=0, param_seed=PILOT_PARAM_SEED, duration_s=1.0,
        stim_current=382.8125)

    off = run_trial(model, mg, sensory_groups=sg, rig="ground", duration_s=1.0,
                    seed=0, adapter=default_params(), adapter_sensory=False)
    assert np.count_nonzero(off.sensory_drive) == 0, (
        "sensory_drive is non-zero with adapter_sensory=False: body state is "
        "reaching the connectome in what is supposed to be the output-only "
        "condition")
    assert off.meta["adapter_sensory"] is False

    on = run_trial(model, mg, sensory_groups=sg, rig="ground", duration_s=1.0,
                   seed=0, adapter=default_params(), adapter_sensory=True)
    assert np.count_nonzero(on.sensory_drive) > 0, (
        "sensory_drive is zero even with adapter_sensory=True — the R1a/R1b "
        "comparison would be vacuous")
    assert not np.array_equal(off.motor_rates, on.motor_rates), (
        "identical motor output with feedback on and off means the flag "
        "changes nothing measurable")


# --- the ground reward's own direction gate -----------------------------

def test_ground_sign_test_passes_and_is_a_hard_failure_type():
    """Forward (+x) travel must outrank backward travel. The ball reward's
    forward axis was negative pitch, which is how a sign error became easy
    to make in the first place."""
    reward_ground.sign_test()          # raises if the convention is wrong
    assert issubclass(reward_ground.SignTestFailure, Exception)


def test_ground_reward_ranks_forward_above_backward_end_to_end():
    """Independent of sign_test's internals: build the two synthetic cases
    and compare the full scalar the optimiser actually sees."""
    import fly_robot.adapter.reward_ground as rg
    from fly_robot.sim.closed_loop import TrialResult

    n = 2000
    t = np.arange(n) * rg.NEURAL_DT
    motor = np.tile(np.sin(2 * np.pi * 11 * t), (6, 1)).astype(np.float32)
    zeros = np.zeros((n, 42), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0

    def trial(dx_mm):
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, dx_mm, n)
        return TrialResult(
            motor_rates=motor, joint_angles=zeros, joint_velocities=zeros,
            sensory_drive=np.zeros((0, n), dtype=np.float32), sensory_channels=[],
            ball_quat=None, ball_angvel=None, n_active_neurons=400,
            max_firing_rate=19.0, wall_clock_s=0.0, thorax_pos=pos,
            thorax_quat=quat, n_steps_planned=n)

    d = rg.REWARD_CONFIG["references"]["walking_mm_s"] * n * rg.NEURAL_DT
    assert rg.evaluate(trial(d)).total > rg.evaluate(trial(-d)).total


# --- the locomotion gate on the rhythm terms ----------------------------

def _synthetic_trial(dx_mm, n=4000):
    """A strongly rhythmic, tripod-phased trial travelling `dx_mm` forward."""
    import fly_robot.adapter.reward_ground as rg
    from fly_robot.sim.closed_loop import TrialResult

    t = np.arange(n) * rg.NEURAL_DT
    motor = np.stack([
        np.sin(2 * np.pi * 11 * t + (0 if i in (0, 3, 4) else np.pi))
        for i in range(6)]).astype(np.float32)
    zeros = np.zeros((n, 42), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = np.linspace(0.0, dx_mm, n)
    return TrialResult(
        motor_rates=motor, joint_angles=zeros, joint_velocities=zeros,
        sensory_drive=np.zeros((0, n), dtype=np.float32), sensory_channels=[],
        ball_quat=None, ball_angvel=None, n_active_neurons=400,
        max_firing_rate=19.0, wall_clock_s=0.0, thorax_pos=pos,
        thorax_quat=quat, n_steps_planned=n)


def test_standing_still_earns_no_rhythm_credit():
    """The measured hole: R1a's gen-38 best scored +0.3379 while travelling
    -0.010 mm, of which rhythmicity was +0.2500 and coordination +0.1089.
    A perfectly rhythmic trial that does not move must now score ~0."""
    b = reward_ground.evaluate(_synthetic_trial(0.0))
    assert b.terms["rhythmicity"] == pytest.approx(0.0, abs=1e-9)
    assert b.terms["coordination"] == pytest.approx(0.0, abs=1e-9)
    assert b.total < 0.05, (
        f"a motionless but rhythmic trial still scores {b.total:+.4f}; the "
        "locomotion gate is not closing the hole it was added for")


def test_walking_still_earns_full_rhythm_credit():
    """The gate must not punish the case it exists to reward. At CPG-baseline
    travel both rhythm terms stay at their full weights."""
    b = reward_ground.evaluate(_synthetic_trial(56.1))
    assert b.terms["rhythmicity"] == pytest.approx(
        reward_ground.REWARD_CONFIG["weights"]["rhythmicity"], rel=1e-6)
    assert b.terms["coordination"] == pytest.approx(
        reward_ground.REWARD_CONFIG["weights"]["coordination"], rel=1e-6)
    assert b.total > 1.0


def test_walking_backwards_cannot_buy_rhythm_credit():
    """Gating on |dx| instead of forward dx would let a candidate collect
    ~0.36 of rhythm credit by walking backwards against a progress penalty
    three orders of magnitude smaller. It must earn nothing instead."""
    b = reward_ground.evaluate(_synthetic_trial(-56.1))
    assert b.terms["rhythmicity"] == pytest.approx(0.0, abs=1e-9)
    assert b.terms["coordination"] == pytest.approx(0.0, abs=1e-9)
    assert b.total < 0.0


def test_rhythm_credit_is_monotonic_in_forward_distance():
    """No cliff the optimiser could perch on: more forward travel never
    earns less rhythm credit."""
    creds = [reward_ground.evaluate(_synthetic_trial(d)).terms["rhythmicity"]
             for d in (0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 10.0)]
    assert all(b >= a - 1e-9 for a, b in zip(creds, creds[1:])), creds
    assert creds[0] == pytest.approx(0.0, abs=1e-9)
    assert creds[-1] > creds[0]


def test_gate_threshold_is_a_low_bar_not_a_performance_demand():
    """The gate is meant to exclude standing still, not to require good
    walking: its full-credit threshold must stay far below the CPG
    baseline, or it silently becomes a second progress term."""
    gate_mm = reward_ground.REWARD_CONFIG["references"]["rhythm_gate_dx_mm"]
    cpg_mm_per_trial = reward_ground.REWARD_CONFIG["references"]["walking_mm_s"] * 4.0
    assert gate_mm / cpg_mm_per_trial < 0.05
