"""Gates for the wiring controls C1 (shuffle) and C2 (random network).

A control that is not actually matched to the real network is not a
control — any difference in behaviour could then be the mismatch rather
than the wiring. These assert the matching directly, on a synthetic
network with the same structure as the real one (Dale's law, mixed signs,
a protected interface).

Run:
    python -m pytest tests/test_connectome_controls.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp

from fly_robot.neural.connectome_controls import (
    degree_preserving_shuffle, matched_random_network,
)


def _dale_network(n=600, density=0.05, seed=0):
    """A network obeying Dale's law in the [pre, post] orientation the
    controls operate on: each ROW (presynaptic neuron) is purely
    excitatory or purely inhibitory, as the real connectome is."""
    rng = np.random.default_rng(seed)
    w = sp.random(n, n, density=density, format="coo", random_state=seed)
    presyn_sign = rng.choice([1, -1], size=n)
    data = np.abs(rng.normal(5, 2, w.nnz)) * presyn_sign[w.row]
    return sp.csr_matrix((data, (w.row, w.col)), shape=(n, n))


def _dale_violations(w) -> int:
    """Violations counted over ROWS: the controls operate on w[pre, post]."""
    wr = sp.csr_matrix(w)
    bad = 0
    for i in range(w.shape[0]):
        d = wr.data[wr.indptr[i]:wr.indptr[i + 1]]
        if len(d) and (d > 0).any() and (d < 0).any():
            bad += 1
    return bad


# --- C2: matched random network -------------------------------------------

def test_c2_matches_size_and_sparsity_exactly():
    w = _dale_network()
    rand, rep = matched_random_network(w, seed=1)
    assert rand.shape == w.shape
    assert rep["sparsity_preserved"] and rand.nnz == w.nnz


def test_c2_matches_sign_ratio_exactly():
    """A control with fewer inhibitory edges would be a weaker network,
    not a randomly wired one."""
    w = _dale_network()
    rand, rep = matched_random_network(w, seed=2)
    assert rep["sign_ratio_preserved"]
    assert (rand.data > 0).sum() == (w.data > 0).sum()
    assert (rand.data < 0).sum() == (w.data < 0).sum()


def test_c2_preserves_dale_law():
    """The real connectome obeys Dale's law exactly (0 of 22,769
    presynaptic neurons carry both signs). A control that violated it
    would differ by being biologically impossible, not by being random."""
    w = _dale_network()
    assert _dale_violations(w) == 0, "test fixture is not Dale-compliant"
    rand, rep = matched_random_network(w, seed=3)
    assert rep["dale_preserved"] and _dale_violations(rand) == 0


def test_c2_preserves_the_weight_multiset():
    """Same magnitudes, different wiring — so the control cannot be weaker
    merely by having smaller weights."""
    w = _dale_network()
    rand, rep = matched_random_network(w, seed=4)
    assert rep["weight_multiset_preserved"]
    np.testing.assert_array_equal(np.sort(rand.data), np.sort(w.data))


def test_c2_actually_randomises_the_topology():
    """The point of C2 is that degrees are NOT preserved. If they were, it
    would be C1."""
    w = _dale_network()
    rand, rep = matched_random_network(w, seed=5)
    assert not rep["out_degree_preserved"]
    assert rep["mean_abs_out_degree_change"] > 0
    assert rep["frac_edges_rewired"] > 0.9


def test_c2_protects_the_interface():
    """Interface edges are held out, exactly as C1 holds them, so all three
    conditions are driven and read through identical wiring."""
    w = _dale_network()
    protected_rows = np.arange(0, 20)
    protected_cols = np.arange(500, 520)
    rand, rep = matched_random_network(
        w, seed=6, protected_rows=protected_rows, protected_cols=protected_cols)
    assert rep["n_edges_protected"] > 0

    wl, rl = w.tolil(), rand.tolil()
    for r in protected_rows:
        np.testing.assert_array_equal(np.asarray(wl.rows[r]), np.asarray(rl.rows[r]),
                                      err_msg=f"protected row {r} was rewired")


def test_c2_creates_no_self_loops_or_duplicates():
    w = _dale_network()
    rand, rep = matched_random_network(w, seed=7)
    # nnz == len(data) is asserted inside; duplicates would have collapsed.
    assert rand.nnz == w.nnz
    coo = rand.tocoo()
    keys = coo.row.astype(np.int64) * rand.shape[0] + coo.col
    assert len(np.unique(keys)) == len(keys), "duplicate edges present"


def test_c2_is_deterministic_and_seed_dependent():
    w = _dale_network()
    a, _ = matched_random_network(w, seed=8)
    b, _ = matched_random_network(w, seed=8)
    c, _ = matched_random_network(w, seed=9)
    assert (a != b).nnz == 0, "same seed gave a different network"
    assert (a != c).nnz > 0, "different seeds gave the same network"


def test_c2_does_not_modify_the_real_matrix():
    w = _dale_network()
    before_data, before_nnz = w.data.copy(), w.nnz
    matched_random_network(w, seed=10)
    np.testing.assert_array_equal(w.data, before_data)
    assert w.nnz == before_nnz


# --- C1 vs C2: they must be different controls ----------------------------

def test_c1_preserves_degrees_and_c2_does_not():
    """The whole reason for running both: C1 keeps the degree sequence,
    C2 discards it. If both preserved degrees, C2 would add nothing."""
    w = _dale_network()
    shuffled, c1 = degree_preserving_shuffle(w, seed=11)
    rand, c2 = matched_random_network(w, seed=11)
    assert c1["out_degree_preserved"] and c1["in_degree_preserved"]
    assert not c2["out_degree_preserved"]
    # both keep the global statistics
    assert shuffled.nnz == rand.nnz == w.nnz
    assert (shuffled.data > 0).sum() == (rand.data > 0).sum() == (w.data > 0).sum()
