"""Gates for the steppable rate model's active-column matvec.

The optimisation skips columns of W whose rate is exactly zero. That is
only legitimate if it changes nothing, so these tests assert equality to
the LAST BIT rather than to a tolerance — `np.array_equal`, not
`assert_allclose`. A tolerance-based test here would pass even if the
optimisation were quietly perturbing the trajectory, which is exactly the
failure worth catching in a system the audit already showed to be
phase-sensitive (`docs/closed_loop/AUDIT.md` §4).

The saturation test exists because the speedup INVERTS when the network
saturates (measured 0.74x, i.e. slower, at full density). The guard is
supposed to fall back to the dense path there; these check it actually
fires, and that the answer stays identical when it does.

Run:
    python -m pytest tests/test_steppable_model.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from fly_robot.neural.steppable_rate_model import (
    ACTIVE_COLUMN_NNZ_FRACTION, SteppableRateModel, build_neuron_set,
)


def _toy_network(n=400, density=0.05, seed=0):
    """A small network with the same structure as the real one: mixed
    signs, a stimulation current into two 'command' rows."""
    rng = np.random.default_rng(seed)
    w = sp.random(n, n, density=density, format="csr", random_state=seed,
                  data_rvs=lambda k: rng.normal(0, 30, k))
    neurons = build_neuron_set(
        w,
        tau=rng.normal(0.02, 0.002, n),
        a=rng.normal(1.0, 0.1, n),
        threshold=rng.normal(7.5, 0.6, n),
        fr_cap=rng.normal(200.0, 10.0, n),
        exc_mult=0.03, inh_mult=0.03,
    )
    baseline = np.zeros(n)
    baseline[[1, 2]] = 380.0
    return neurons, baseline


def _run(neurons, baseline, steps, active_column_matvec, extra=None):
    m = SteppableRateModel(neurons, baseline, pulse_start=0.02, pulse_end=9.9,
                           active_column_matvec=active_column_matvec)
    trace = np.empty((steps, neurons.n_neurons))
    for i in range(steps):
        trace[i] = m.step(0.001, extra_input=extra)
    return trace, m


def test_active_column_matvec_is_bit_identical_open_loop():
    """THE gate for this optimisation: 500 open-loop steps, both paths,
    equal to the last bit."""
    neurons, baseline = _toy_network()
    fast, mf = _run(neurons, baseline, 500, True)
    dense, md = _run(neurons, baseline, 500, False)

    assert np.array_equal(fast, dense), (
        "active-column matvec changed the trajectory; max|diff| = "
        f"{np.abs(fast - dense).max():.3e}"
    )
    assert mf.n_matvec_fast > 0, "the fast path never ran — test proves nothing"
    assert md.n_matvec_fast == 0 and md.n_matvec_dense > 0


def test_active_column_matvec_is_bit_identical_with_extra_input():
    """Same, with a sensory-style extra current applied — the closed-loop
    case, where `total` is not just W@r."""
    neurons, baseline = _toy_network(seed=3)
    rng = np.random.default_rng(7)
    extra = np.zeros(neurons.n_neurons)
    extra[rng.choice(neurons.n_neurons, 40, replace=False)] = rng.uniform(0, 3, 40)

    fast, _ = _run(neurons, baseline, 300, True, extra=extra)
    dense, _ = _run(neurons, baseline, 300, False, extra=extra)
    assert np.array_equal(fast, dense)


def test_synaptic_input_matches_dense_elementwise():
    """Directly on the matvec, at a realistic sparse operating point."""
    neurons, baseline = _toy_network(seed=11)
    m = SteppableRateModel(neurons, baseline, active_column_matvec=True)
    rng = np.random.default_rng(1)
    r = np.zeros(neurons.n_neurons)
    idx = rng.choice(neurons.n_neurons, 25, replace=False)
    r[idx] = rng.uniform(0.1, 20.0, 25)

    assert np.array_equal(m._synaptic_input(r), neurons.w_eff @ r)


def test_all_zero_rates_give_exactly_zero_input():
    """The degenerate case the whole argument rests on: no active columns
    at all must still produce the right vector, not an empty-slice error."""
    neurons, baseline = _toy_network(seed=5)
    m = SteppableRateModel(neurons, baseline, active_column_matvec=True)
    r = np.zeros(neurons.n_neurons)
    out = m._synaptic_input(r)
    assert out.shape == (neurons.n_neurons,)
    assert np.array_equal(out, np.zeros(neurons.n_neurons))


def test_saturated_network_falls_back_to_dense_path():
    """REGRESSION GUARD. With every neuron active the column gather is
    measurably slower than the dense matvec, so the guard must switch. If
    this fails, saturated trials silently cost ~35% more than before."""
    neurons, baseline = _toy_network(seed=13)
    m = SteppableRateModel(neurons, baseline, active_column_matvec=True)
    rng = np.random.default_rng(2)

    dense_r = rng.uniform(0.1, 20.0, neurons.n_neurons)   # 100% active
    m._synaptic_input(dense_r)
    assert m.n_matvec_dense == 1 and m.n_matvec_fast == 0, (
        "guard did not fall back on a fully-saturated rate vector"
    )

    sparse_r = np.zeros(neurons.n_neurons)
    sparse_r[rng.choice(neurons.n_neurons, 5, replace=False)] = 1.0
    m._synaptic_input(sparse_r)
    assert m.n_matvec_fast == 1, "guard refused the fast path when it should apply"


def test_saturated_fallback_is_still_bit_identical():
    """The fallback must not be a different answer, only a different route
    to it — including when a trial crosses the threshold mid-run."""
    neurons, baseline = _toy_network(seed=17)
    m_fast = SteppableRateModel(neurons, baseline, active_column_matvec=True)
    m_dense = SteppableRateModel(neurons, baseline, active_column_matvec=False)
    rng = np.random.default_rng(4)

    for frac in (0.0, 0.01, 0.2, 0.5, 1.0):
        r = np.zeros(neurons.n_neurons)
        k = int(frac * neurons.n_neurons)
        if k:
            r[rng.choice(neurons.n_neurons, k, replace=False)] = rng.uniform(0.1, 20.0, k)
        assert np.array_equal(m_fast._synaptic_input(r), m_dense._synaptic_input(r)), (
            f"paths diverged at {frac:.0%} active")
    assert m_fast.n_matvec_fast > 0 and m_fast.n_matvec_dense > 0, (
        "this sweep should have exercised BOTH paths")


def test_guard_threshold_is_a_fraction_of_nnz():
    """The guard is a measured crossover, not a magic number; if someone
    raises it past the measured inversion point the optimisation becomes a
    pessimisation."""
    assert 0.0 < ACTIVE_COLUMN_NNZ_FRACTION <= 0.5, (
        "measured crossover is near 50% of nnz (1.03x at 4,629 active, "
        "0.74x at 23,532); a threshold above it makes saturated trials slower"
    )
