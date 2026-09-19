"""Validation gates for the closed-loop sensory interface.

These are the checks `docs/closed_loop/PREREGISTRATION.md` §6 requires to
pass before any main experiment trial runs. They are tests, not
assertions in a notebook, so a later change that breaks locality or
determinism fails the build instead of quietly corrupting a result.

Run:
    python -m pytest tests/test_sensory_interface.py -v
"""

import numpy as np
import pandas as pd
import pytest

from fly_robot.interface.joint_to_sensory_neuron import (
    CHORDOTONAL, HAIR_PLATE, SensoryEncoder, build_sensory_groups,
    chordotonal_drive, hair_plate_drive,
)
from fly_robot.interface.motor_neuron_to_joint import LEG_NAME_TO_FLYGYM_PREFIX
from fly_robot.neural.replicate_ensemble import load_published_full_vnc_replicates


class _DOF:
    def __init__(self, name):
        self.name = name


@pytest.fixture(scope="module")
def wtable():
    wt, _, _ = load_published_full_vnc_replicates()
    return wt


@pytest.fixture(scope="module")
def jointdofs():
    names = []
    for (segment, side), prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        names += [f"c_thorax-{prefix}_coxa-pitch",
                  f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]
    return [_DOF(n) for n in names]


@pytest.fixture(scope="module")
def groups(wtable):
    return build_sensory_groups(wtable)


def _encoder(groups, jointdofs, **kw):
    neutral = {d.name: 0.0 for d in jointdofs}
    return SensoryEncoder(groups, jointdofs, neutral, **kw)


# --------------------------------------------------------------------------
# Group construction
# --------------------------------------------------------------------------

def test_all_six_legs_have_sensory_neurons(groups):
    for segment in ("T1", "T2", "T3"):
        for side in ("LHS", "RHS"):
            total = sum(len(groups.indices_by_group.get((segment, side, sc), []))
                        for sc in (CHORDOTONAL, HAIR_PLATE))
            assert total > 0, f"{segment}-{side} has no proprioceptors"


def test_no_neuron_belongs_to_two_legs(groups):
    seen = {}
    for key, idxs in groups.indices_by_group.items():
        for i in idxs:
            assert i not in seen, f"neuron {i} in both {seen[i]} and {key}"
            seen[i] = key


def test_side_normalisation_equalises_total_drive(groups):
    """Left and right pools of a segment must receive the same TOTAL
    current for the same drive — that is the whole point of the
    normalisation."""
    for segment in ("T1", "T2", "T3"):
        for subclass in (CHORDOTONAL, HAIR_PLATE):
            totals = []
            for side in ("LHS", "RHS"):
                key = (segment, side, subclass)
                idxs = groups.indices_by_group.get(key, [])
                if idxs:
                    totals.append(len(idxs) * groups.scale_by_group[key])
            if len(totals) == 2:
                assert totals[0] == pytest.approx(totals[1], rel=1e-9), \
                    f"{segment} {subclass} totals {totals}"


def test_unnormalised_mode_is_raw_counts(wtable):
    """C4's as-annotated control must NOT equalise the sides."""
    g = build_sensory_groups(wtable, normalise_sides=False)
    assert all(s == 1.0 for s in g.scale_by_group.values())


# --------------------------------------------------------------------------
# Encoder shape
# --------------------------------------------------------------------------

def test_chordotonal_drive_is_bounded_and_monotonic_in_position():
    vals = [chordotonal_drive(a, 0.0, 0.0) for a in np.linspace(-1.0, 1.0, 50)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_chordotonal_drive_responds_to_movement():
    still = chordotonal_drive(0.0, 0.0, 0.0)
    moving = chordotonal_drive(0.0, 5.0, 0.0)
    assert moving > still
    assert chordotonal_drive(0.0, -5.0, 0.0) == pytest.approx(moving), \
        "movement code must be unsigned — a pooled population cannot signal direction"


def test_hair_plate_is_silent_through_most_of_range():
    assert hair_plate_drive(0.0, 0.0) == 0.0          # neutral
    assert hair_plate_drive(-0.3, 0.0) == 0.0         # far flexion
    assert hair_plate_drive(0.1, 0.0) == 0.0          # still below threshold
    assert hair_plate_drive(0.3, 0.0) == pytest.approx(1.0)   # at the limit
    assert 0.0 < hair_plate_drive(0.2, 0.0) < 1.0


# --------------------------------------------------------------------------
# THE LOCALITY GATE — sensors on leg i may only drive leg i's neurons
# --------------------------------------------------------------------------

def test_locality_perturbing_one_leg_changes_only_that_legs_neurons(groups, jointdofs):
    """Pre-registration hard rule. Perturb each leg's joints alone and
    assert that no other leg's sensory neurons see any change."""
    enc = _encoder(groups, jointdofs, gain=10.0)
    n = len(jointdofs)
    baseline = enc.input_current(np.zeros(n), np.zeros(n))

    name_to_i = {d.name: i for i, d in enumerate(jointdofs)}
    for (segment, side), prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        angles = np.zeros(n)
        velocities = np.zeros(n)
        angles[name_to_i[f"c_thorax-{prefix}_coxa-pitch"]] = 0.29
        angles[name_to_i[f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]] = 0.25
        velocities[name_to_i[f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]] = 8.0

        changed = np.where(enc.input_current(angles, velocities) != baseline)[0]
        assert len(changed) > 0, f"perturbing {segment}-{side} changed nothing"

        own = set()
        for subclass in (CHORDOTONAL, HAIR_PLATE):
            own.update(groups.indices_by_group.get((segment, side, subclass), []))
        leaked = set(changed.tolist()) - own
        assert not leaked, (
            f"perturbing {segment}-{side} changed {len(leaked)} neurons belonging to "
            f"other legs — cross-leg path in our code, forbidden by the pre-registration"
        )


def test_c3_permutation_is_the_only_cross_leg_path(groups, jointdofs):
    """C3 deliberately violates locality. Confirm it actually does — a
    control that silently behaved like the real condition would be
    worthless."""
    swap = {("T1", "LHS"): ("T3", "RHS"), ("T3", "RHS"): ("T1", "LHS")}
    enc = _encoder(groups, jointdofs, gain=10.0, leg_permutation=swap)
    n = len(jointdofs)
    name_to_i = {d.name: i for i, d in enumerate(jointdofs)}

    baseline = enc.input_current(np.zeros(n), np.zeros(n))
    angles = np.zeros(n)
    prefix = LEG_NAME_TO_FLYGYM_PREFIX[("T1", "LHS")]
    angles[name_to_i[f"{prefix}_trochanterfemur-{prefix}_tibia-pitch"]] = 0.25
    changed = set(np.where(enc.input_current(angles, np.zeros(n)) != baseline)[0].tolist())

    t3r = set(groups.indices_by_group.get(("T3", "RHS", CHORDOTONAL), []))
    assert changed & t3r, "C3 permutation did not route T1-LHS sensors to T3-RHS neurons"


# --------------------------------------------------------------------------
# Determinism and the g_fb = 0 gate
# --------------------------------------------------------------------------

def test_gain_zero_gives_exactly_zero_current(groups, jointdofs):
    """`g_fb = 0` must reproduce open-loop BIT-identically, so the added
    current has to be exactly 0.0 — not merely small."""
    enc = _encoder(groups, jointdofs, gain=0.0)
    n = len(jointdofs)
    current = enc.input_current(np.full(n, 0.27), np.full(n, 9.0))
    assert np.count_nonzero(current) == 0
    assert current.dtype == np.float64


def test_deterministic_for_identical_input(groups, jointdofs):
    enc = _encoder(groups, jointdofs, gain=7.5)
    n = len(jointdofs)
    rng = np.random.default_rng(0)
    angles, velocities = rng.normal(0, 0.1, n), rng.normal(0, 3.0, n)
    first = enc.input_current(angles, velocities)
    for _ in range(3):
        assert np.array_equal(enc.input_current(angles, velocities), first)


def test_current_scales_linearly_with_gain(groups, jointdofs):
    n = len(jointdofs)
    rng = np.random.default_rng(1)
    angles, velocities = rng.normal(0, 0.1, n), rng.normal(0, 3.0, n)
    a = _encoder(groups, jointdofs, gain=2.0).input_current(angles, velocities)
    b = _encoder(groups, jointdofs, gain=8.0).input_current(angles, velocities)
    np.testing.assert_allclose(b, 4.0 * a, rtol=1e-12, atol=0)


def test_current_is_finite_and_non_negative_over_extreme_input(groups, jointdofs):
    """Bounded-rate gate: the encoder must not emit inf/NaN or negative
    current even when the body does something absurd."""
    enc = _encoder(groups, jointdofs, gain=40.0)
    n = len(jointdofs)
    for angles, velocities in [
        (np.full(n, 1e3), np.full(n, 1e4)),
        (np.full(n, -1e3), np.full(n, -1e4)),
        (np.zeros(n), np.zeros(n)),
    ]:
        current = enc.input_current(angles, velocities)
        assert np.isfinite(current).all()
        assert (current >= 0).all()
        assert current.max() <= 40.0 * max(groups.scale_by_group.values()) + 1e-9
