"""Regression gate: the Phase 4 motor interface still behaves exactly as
it did when it was frozen.

Phase 5 makes the motor decoder trainable — the adapter supplies its
constants at call time instead of the module-level defaults being the only
option. That is approved, but it creates a specific hazard: a later change
to the defaults, or to the rule itself, would silently invalidate every
Phase 4 number, since `docs/closed_loop/` reports results computed under
the frozen configuration.

So this pins both halves:

  * the CONFIG HASH recorded in `docs/closed_loop/AUDIT.md` §5, recomputed
    from the live module constants — catches anyone editing a gain, a
    module name, or the leg map;
  * the BEHAVIOUR of the rule at those defaults, computed independently
    here from the documented formula rather than from a stored golden
    array, so the test states what it is checking instead of just
    asserting bytes match.

Run:
    python -m pytest tests/test_frozen_motor_interface.py -v
"""

import hashlib
import json

import numpy as np
import pytest

from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ,
    LEG_NAME_TO_FLYGYM_PREFIX, compute_joint_targets,
)
from fly_robot.sim.closed_loop import _joint_targets_from_rates

# docs/closed_loop/AUDIT.md §5, and PREREGISTRATION.md §0b. Frozen at git
# commit c2c6409afd1be573cb22285e68925353945e0a8c.
FROZEN_CONFIG_SHA256 = (
    "04be9dec181ec2e20fad91ba07cfe018d0a413720b0dd441a2dd8aab7294fc1d")


def frozen_config_hash() -> str:
    """Recompute the audit's config hash from the live module constants.

    The serialisation is the one that reproduces the recorded digest:
    `json.dumps(cfg, sort_keys=True)` over exactly the four documented
    fields, in the shapes AUDIT.md §5 prints them.
    """
    config = {
        "ANTAGONIST_PAIRS": [list(pair) for pair in ANTAGONIST_PAIRS],
        "DEFAULT_GAIN_RAD": DEFAULT_GAIN_RAD,
        "DEFAULT_RATE_SCALE_HZ": DEFAULT_RATE_SCALE_HZ,
        "LEG_MAP": [[list(k), v] for k, v in LEG_NAME_TO_FLYGYM_PREFIX.items()],
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class _DOF:
    def __init__(self, name):
        self.name = name


def test_config_hash_matches_the_audit():
    """If this fails, a Phase 4 constant moved and every number in
    docs/closed_loop/ was computed under different settings than the code
    now implements. Do not update the constant to make it pass — record an
    amendment, as the project's correction convention requires."""
    assert frozen_config_hash() == FROZEN_CONFIG_SHA256, (
        "motor interface config changed; docs/closed_loop/ results no longer "
        f"describe this code.\n  expected {FROZEN_CONFIG_SHA256}\n"
        f"  actual   {frozen_config_hash()}"
    )


def test_frozen_defaults_are_the_documented_values():
    """Stated separately so a failure says WHICH constant moved."""
    assert DEFAULT_GAIN_RAD == 0.5
    assert DEFAULT_RATE_SCALE_HZ == 3.0
    assert ANTAGONIST_PAIRS == [
        ("coxa swing", "coxa stance", "coxa-pitch"),
        ("femur/tr extend", "femur/tr flex", "trochanterfemur-pitch"),
        ("tibia extend", "tibia flex", "tibia-pitch"),
    ]


def _fake_groups(dof_names):
    """A minimal MotorNeuronGroups-alike with two neurons per module."""
    from fly_robot.interface.motor_neuron_to_joint import MotorNeuronGroups
    indices, dofs, row = {}, {}, 0
    for leg, side in LEG_NAME_TO_FLYGYM_PREFIX:
        for pos, neg, _suffix in ANTAGONIST_PAIRS:
            for module in (pos, neg):
                indices[(leg, side, module)] = [row, row + 1]
                dofs[(leg, side, module)] = None
                row += 2
    return MotorNeuronGroups(dofs, indices), row


def test_rule_is_neutral_plus_gain_times_tanh_of_rate_difference():
    """The documented rule, verbatim from AUDIT.md §5:
        joint = neutral + 0.5 rad * tanh((rate_pos - rate_neg) / 3.0 Hz)
    computed here independently and compared against the real function."""
    dof_names = []
    for prefix in ("lf", "lm", "lh", "rf", "rm", "rh"):
        dof_names += [f"c_thorax-{prefix}_coxa-pitch",
                      f"{prefix}_coxa-{prefix}_trochanterfemur-pitch",
                      f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]
    dofs = [_DOF(n) for n in dof_names]
    groups, n_rows = _fake_groups(dof_names)

    rng = np.random.default_rng(20260921)
    neutral = rng.normal(0, 0.2, len(dofs))
    rates = rng.uniform(0, 12, (n_rows, 4))

    got = compute_joint_targets(rates, groups, dofs, neutral)

    dof_index = {d.name: i for i, d in enumerate(dofs)}
    expected = np.tile(neutral, (4, 1))
    for (leg, side), prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        for pos, neg, suffix in ANTAGONIST_PAIRS:
            name = {"coxa-pitch": f"c_thorax-{prefix}_coxa-pitch",
                    "trochanterfemur-pitch": f"{prefix}_coxa-{prefix}_trochanterfemur-pitch",
                    "tibia-pitch": f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"}[suffix]
            i = dof_index[name]
            pos_rate = rates[groups.indices_by_group[(leg, side, pos)]].mean(axis=0)
            neg_rate = rates[groups.indices_by_group[(leg, side, neg)]].mean(axis=0)
            expected[:, i] = neutral[i] + 0.5 * np.tanh((pos_rate - neg_rate) / 3.0)

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)


def test_explicit_defaults_equal_implicit_defaults():
    """The Phase 5 change: the adapter passes the constants in rather than
    relying on module defaults. Passing the frozen values explicitly must
    be indistinguishable from passing nothing — otherwise 'frozen
    behaviour' and 'adapter behaviour' are two different code paths."""
    dofs = [_DOF(f"c_thorax-{p}_coxa-pitch") for p in ("lf", "rf")]
    groups, n_rows = _fake_groups([d.name for d in dofs])
    rng = np.random.default_rng(1)
    neutral = rng.normal(0, 0.1, len(dofs))
    rates = rng.uniform(0, 10, (n_rows, 3))

    implicit = compute_joint_targets(rates, groups, dofs, neutral)
    explicit = compute_joint_targets(rates, groups, dofs, neutral,
                                     gain_rad=DEFAULT_GAIN_RAD,
                                     rate_scale_hz=DEFAULT_RATE_SCALE_HZ)
    assert np.array_equal(implicit, explicit)

    dof_index = {d.name: i for i, d in enumerate(dofs)}
    for t in range(3):
        single = _joint_targets_from_rates(rates[:, t], groups, dof_index, neutral,
                                           DEFAULT_GAIN_RAD, DEFAULT_RATE_SCALE_HZ)
        np.testing.assert_allclose(single, implicit[t], rtol=0, atol=1e-12)


def test_adapter_constants_actually_change_the_output():
    """Guards the opposite failure: if the call-time constants were ignored,
    a 'trained' decoder would silently be the frozen one."""
    dofs = [_DOF("c_thorax-lf_coxa-pitch")]
    groups, n_rows = _fake_groups([d.name for d in dofs])
    rng = np.random.default_rng(2)
    neutral = np.zeros(1)
    rates = rng.uniform(2, 10, (n_rows, 2))

    base = compute_joint_targets(rates, groups, dofs, neutral)
    wider = compute_joint_targets(rates, groups, dofs, neutral, gain_rad=0.9)
    slower = compute_joint_targets(rates, groups, dofs, neutral, rate_scale_hz=12.0)
    assert not np.array_equal(base, wider)
    assert not np.array_equal(base, slower)
