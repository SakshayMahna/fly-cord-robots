"""Gates for the trainable adapter and the reward.

The decisive ones are `test_no_bypass_*`: every control signal must pass
sensors -> fly sensory neurons -> connectome -> fly motor neurons -> joints.
If a sensor reading can reach a joint target without going through the
connectome, the headline claim is false no matter what the learning curve
looks like.

Run:
    python -m pytest tests/test_adapter.py -v
"""

import numpy as np
import pytest

from fly_robot.adapter.apply import (
    DNG100_ROWS, MDN_ROWS, adapter_joint_targets, command_current,
    motor_filter_alpha,
)
from fly_robot.adapter.parameters import (
    N_PARAMS, PARAM_SPEC, default_params, default_z, from_z,
)
from fly_robot.adapter.reward import (
    REWARD_CONFIG, SignTestFailure, evaluate, reward_config_hash, _synthetic_result,
)
from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ,
    LEG_NAME_TO_FLYGYM_PREFIX, MotorNeuronGroups, _dof_name_for_pair,
)


class _DOF:
    def __init__(self, name):
        self.name = name


def _groups_and_dofs():
    dof_names, indices, dofmap, row = [], {}, {}, 0
    for (seg, side), prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        for pos, neg, suffix in ANTAGONIST_PAIRS:
            dof_names.append(_dof_name_for_pair(prefix, suffix))
            for module in (pos, neg):
                indices[(seg, side, module)] = [row, row + 1]
                dofmap[(seg, side, module)] = None
                row += 2
    return MotorNeuronGroups(dofmap, indices), [_DOF(n) for n in dof_names], row


# --- parameters -----------------------------------------------------------

def test_parameter_count_is_71():
    """65 interface parameters plus per-segment adhesion weight and
    threshold (2 x 3)."""
    assert N_PARAMS == 71, "DESIGN.md §2 + §10 specify 71 trainable parameters"


def test_z_zero_decodes_to_the_documented_defaults():
    """z = 0 must be the DEFAULT configuration, not the midpoint of each
    range — the search has to start somewhere known-sane."""
    p = default_params()
    for name, shape, _low, _high, default in PARAM_SPEC:
        v = np.asarray(p.values[name])
        assert np.allclose(v, default, rtol=1e-9, atol=1e-9), (
            f"{name} decoded to {v}, expected {default}")


def test_defaults_match_the_frozen_phase4_constants():
    """A failure here means the search no longer starts from Phase 4's
    configuration, which is the reference every result is read against."""
    p = default_params()
    assert np.allclose(p.motor_gain, DEFAULT_GAIN_RAD)
    assert np.allclose(p.motor_scale, DEFAULT_RATE_SCALE_HZ)
    assert np.allclose(p.chord_cap, 2.5)        # SENSORY_CURRENT_CAP
    assert np.allclose(p.hp_threshold, 0.75)    # HAIR_PLATE_THRESHOLD_FRAC
    assert np.allclose(p.chord_sigma, 0.10)     # FRACTIONATION_SIGMA


def test_dng100_default_matches_the_recorded_activity_match():
    """The default must be the REAL network's pre-registered
    activity-matched drive, not Pugliese's raw 380 -- the pre-registration
    (PREREGISTRATION.md) commits to training "relative to that start."

    Reads the recorded matching result directly rather than hardcoding its
    number a second time, so the two cannot drift apart silently: if
    match_drive.py is ever re-run and the result changes, this test fails
    instead of quietly training from a stale drive.
    """
    import json
    from pathlib import Path

    path = Path("media/trained_adapter/drive_matching.json")
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    recorded = json.loads(path.read_text())["networks"]["real"]["matched_drive"]

    p = default_params()
    assert p.dng100_level == pytest.approx(recorded), (
        f"adapter default ({p.dng100_level}) does not match the recorded "
        f"activity-matched drive ({recorded}) -- Rung 1 would launch from "
        "the wrong starting point")


def test_parameters_stay_inside_their_bounds_under_extreme_z():
    for z in (np.full(N_PARAMS, -50.0), np.full(N_PARAMS, 50.0)):
        p = from_z(z)
        for name, _shape, low, high, _default in PARAM_SPEC:
            v = np.asarray(p.values[name])
            assert (v >= low - 1e-9).all() and (v <= high + 1e-9).all(), name


def test_wrong_length_vector_is_rejected():
    with pytest.raises(ValueError):
        from_z(np.zeros(N_PARAMS - 1))


# --- motor decoder --------------------------------------------------------

def test_adapter_decoder_reduces_to_the_frozen_rule_at_defaults():
    """With Phase 4's constants the trainable decoder must BE the frozen
    rule; otherwise 'frozen behaviour' and 'adapter behaviour' are two
    different code paths and Phase 4's numbers stop being comparable."""
    groups, dofs, n_rows = _groups_and_dofs()
    rng = np.random.default_rng(0)
    neutral = rng.normal(0, 0.2, len(dofs))
    rates = rng.uniform(0, 12, n_rows)
    dof_index = {d.name: i for i, d in enumerate(dofs)}

    got = adapter_joint_targets(rates, groups, dof_index, neutral, default_params())

    from fly_robot.sim.closed_loop import _joint_targets_from_rates
    expected = _joint_targets_from_rates(rates, groups, dof_index, neutral,
                                         DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_decoder_parameters_actually_change_the_output():
    """Guards the silent-no-op failure: a 'trained' decoder whose
    parameters were ignored would train to nothing and read as a null
    result about the connectome."""
    groups, dofs, n_rows = _groups_and_dofs()
    rng = np.random.default_rng(1)
    neutral = np.zeros(len(dofs))
    rates = rng.uniform(2, 10, n_rows)
    dof_index = {d.name: i for i, d in enumerate(dofs)}

    base = adapter_joint_targets(rates, groups, dof_index, neutral, default_params())
    z = default_z().copy()
    z[:] = 0.0
    idx = sum(int(np.prod(s)) if s else 1
              for n, s, *_ in PARAM_SPEC[:next(i for i, (n, *_) in enumerate(PARAM_SPEC)
                                               if n == "motor_gain")])
    z[idx] = 2.0     # bump one motor_gain entry
    assert not np.allclose(base, adapter_joint_targets(
        rates, groups, dof_index, neutral, from_z(z)))


# --- no bypass ------------------------------------------------------------

def test_no_bypass_targets_depend_only_on_rates():
    """THE structural gate. Joint targets are a pure function of the
    connectome's motor-neuron rates and the decoder parameters. No sensor
    reading may reach a joint target except through the connectome."""
    groups, dofs, n_rows = _groups_and_dofs()
    rng = np.random.default_rng(2)
    neutral = rng.normal(0, 0.1, len(dofs))
    dof_index = {d.name: i for i, d in enumerate(dofs)}
    params = default_params()

    rates = rng.uniform(0, 10, n_rows)
    first = adapter_joint_targets(rates, groups, dof_index, neutral, params)
    for _ in range(5):   # same rates, repeatedly: identical targets
        np.testing.assert_array_equal(
            adapter_joint_targets(rates, groups, dof_index, neutral, params), first)

    other = adapter_joint_targets(rng.uniform(0, 10, n_rows), groups, dof_index,
                                  neutral, params)
    assert not np.allclose(first, other), "targets ignore the connectome's rates"


def test_adhesion_depends_only_on_connectome_output():
    """NO-BYPASS for the adhesion gate. Adhesion is a pure function of the
    per-leg motor rates fed to it and the trainable parameters. No sensor
    reading or body state may reach it, or adhesion becomes a control
    channel that skips the connectome entirely."""
    from fly_robot.adapter.apply import AdhesionGate

    rng = np.random.default_rng(11)
    series = [rng.uniform(0, 30, 6) for _ in range(200)]

    def run(params):
        gate = AdhesionGate(params, dt=0.001)
        return np.array([gate.step(x) for x in series])

    first = run(default_params())
    # Deterministic: identical input sequence -> identical output, always.
    for _ in range(3):
        np.testing.assert_array_equal(run(default_params()), first)

    assert first.shape == (200, 6)
    assert set(np.unique(first)) <= {0.0, 1.0}, "adhesion must be on or off"


def test_adhesion_is_not_stuck_and_responds_to_the_rhythm():
    """A gate that never switches is useless, and one that ignores its
    input would make the connectome irrelevant to adhesion."""
    from fly_robot.adapter.apply import AdhesionGate

    t = np.arange(400) * 0.001
    rhythm = 10.0 + 5.0 * np.sin(2 * np.pi * 11.0 * t)   # ~11 Hz, as measured
    gate = AdhesionGate(default_params(), dt=0.001)
    out = np.array([gate.step(np.full(6, v)) for v in rhythm])

    duty = out.mean()
    transitions = np.abs(np.diff(out[:, 0])).sum()
    assert 0.05 < duty < 0.95, f"gate is effectively stuck (duty {duty:.2f})"
    assert transitions > 4, f"gate barely switches ({transitions} transitions)"

    flat = AdhesionGate(default_params(), dt=0.001)
    flat_out = np.array([flat.step(np.full(6, 10.0)) for _ in range(400)])
    assert np.abs(np.diff(flat_out[:, 0])).sum() == 0, (
        "a constant input must not produce switching")


def test_adhesion_polarity_is_trainable():
    """`adhesion_weight` may go negative so the optimiser can choose which
    half of the rhythm counts as stance, rather than us asserting it."""
    from fly_robot.adapter.apply import AdhesionGate
    from fly_robot.adapter.parameters import AdapterParams

    base = default_params()
    def with_w(v):
        vals = dict(base.values)
        vals["adhesion_weight"] = np.full(3, v)
        return AdapterParams(vals)

    t = np.arange(400) * 0.001
    rhythm = 10.0 + 5.0 * np.sin(2 * np.pi * 11.0 * t)

    def run(params):
        g = AdhesionGate(params, dt=0.001)
        return np.array([g.step(np.full(6, v)) for v in rhythm])

    pos, neg = run(with_w(1.0)), run(with_w(-1.0))
    assert not np.array_equal(pos, neg), "polarity has no effect"
    # Opposite polarity should gate the opposite half of the cycle.
    assert 0.05 < neg.mean() < 0.95


def test_adhesion_weight_can_be_negative_and_starts_positive():
    spec = {n: (lo, hi, d) for n, _s, lo, hi, d in PARAM_SPEC}
    lo, hi, default = spec["adhesion_weight"]
    assert lo < 0 < hi, "weight must be able to change sign"
    assert default > 0, "weight is initialised positive"


def test_sensory_encoder_only_ever_injects_non_negative_current():
    """A sensory neuron's SIGN comes from the connectome, never from us
    (PREREGISTRATION.md §B0). A negative injected current would be the
    interface overriding Dale's law with a modelling choice."""
    from fly_robot.adapter.apply import AdapterSensoryEncoder
    from fly_robot.interface.joint_to_sensory_neuron import SensoryGroups

    groups = SensoryGroups(
        indices_by_group={("T1", "LHS", "chordotonal organ"): [0, 1, 2],
                          ("T1", "LHS", "hair plate"): [3, 4]},
        scale_by_group={("T1", "LHS", "chordotonal organ"): 1.0,
                        ("T1", "LHS", "hair plate"): 1.0},
        n_neurons=10)
    dofs = [_DOF("c_thorax-lf_coxa-pitch"),
            _DOF("lf_trochanterfemur-lf_tibia-pitch")]
    enc = AdapterSensoryEncoder(groups, dofs, {d.name: 0.0 for d in dofs},
                                params=from_z(np.full(N_PARAMS, -3.0)))
    cur = enc.input_current(np.array([0.4, -0.4]), np.array([5.0, -5.0]))
    assert (cur >= 0).all(), "interface injected NEGATIVE current into a sensory neuron"


# --- command drive --------------------------------------------------------

def test_command_drive_hits_only_verified_command_rows():
    p = default_params()
    cur = command_current(p, 23532, t=1.0, pulse_start=0.02, pulse_end=3.999)
    driven = set(np.flatnonzero(cur).tolist())
    expected = set(DNG100_ROWS["LHS"] + DNG100_ROWS["RHS"]
                   + MDN_ROWS["LHS"] + MDN_ROWS["RHS"])
    assert driven == expected, f"drive reached unexpected rows: {driven ^ expected}"


def test_command_drive_is_zero_outside_the_pulse_window():
    p = default_params()
    assert not command_current(p, 23532, 0.0, 0.02, 3.999).any()
    assert not command_current(p, 23532, 5.0, 0.02, 3.999).any()


def test_command_asymmetry_splits_left_and_right():
    z = default_z().copy()
    i = sum(int(np.prod(s)) if s else 1
            for n, s, *_ in PARAM_SPEC[:next(k for k, (n, *_) in enumerate(PARAM_SPEC)
                                             if n == "command_asymmetry")])
    z[i] = 2.0
    cur = command_current(from_z(z), 23532, 1.0, 0.02, 3.999)
    assert cur[DNG100_ROWS["LHS"][0]] > cur[DNG100_ROWS["RHS"][0]]


# --- reward ---------------------------------------------------------------

def test_forward_walking_scores_positive_progress():
    """Forward is NEGATIVE pitch (PREREGISTRATION.md §2). Backwards here
    means training a backwards-walking fly while the score reads as
    success."""
    fwd = evaluate(_synthetic_result(-0.95))
    back = evaluate(_synthetic_result(+0.95))
    assert fwd.raw["progress"] > 0 > back.raw["progress"]
    assert fwd.total > back.total


def test_saturation_penalty_outweighs_the_progress_a_seizure_can_earn():
    """DESIGN.md §5.7 measured a saturated replicate (n_active 3,794)
    spinning the ball at 30x any healthy one. A seizing network is
    INSTRUMENTALLY ATTRACTIVE, so the penalty must dominate."""
    ref = REWARD_CONFIG["references"]
    hinge = min((3794.0 - ref["saturation_n_active"]) / ref["saturation_n_active"],
                ref["saturation_cap"])
    penalty = abs(REWARD_CONFIG["weights"]["saturation"] * hinge)
    progress = abs(0.471 / ref["walking_rad_s"] * REWARD_CONFIG["weights"]["progress"])
    assert penalty > 2 * progress, (
        f"saturation penalty {penalty:.2f} does not clearly beat the "
        f"{progress:.2f} of progress a seizure earns")


def test_penalty_terms_reduce_the_score():
    """Catches the sign-notation bug the approved REWARD.md table had:
    penalties written negative AND weighted negative would REWARD veering."""
    clean = _synthetic_result(-0.95)
    veering = _synthetic_result(-0.95)
    veering.ball_angvel = veering.ball_angvel.copy()
    veering.ball_angvel[:, 0] = 0.5     # add roll
    veering.ball_angvel[:, 2] = 0.5     # add yaw
    assert evaluate(veering).total < evaluate(clean).total


def test_reward_hash_is_stable_and_changes_with_the_config():
    assert reward_config_hash() == reward_config_hash()
    tweaked = {**REWARD_CONFIG,
               "weights": {**REWARD_CONFIG["weights"], "progress": 0.9}}
    assert reward_config_hash(tweaked) != reward_config_hash()


def test_motor_filter_alpha_is_bounded():
    a = motor_filter_alpha(default_params(), {"c_thorax-lf_coxa-pitch": 0}, 0.001)
    assert (a > 0).all() and (a <= 1).all()
