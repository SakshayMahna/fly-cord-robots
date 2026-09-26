"""Gates for the electrical-coupling addition (`neural/gap_junctions.py`).

Electrical synapses are below EM resolution and absent from every published
fly connectome, so adding them back is OUR addition and has to be held to
the same standard as every other addition in this project:

  * it must never touch `W_eff` — the chemical synapse graph, its weights
    and its neurotransmitter signs stay exactly as the connectome has them;
  * conductance 0 must reproduce the uncoupled model BIT FOR BIT, because
    that ablation is what the whole claim rests on;
  * the coupling matrix must be symmetric with a zero diagonal, which is
    what makes it electrical rather than a second chemical synapse.
"""

import numpy as np
import pytest

from fly_robot.neural.gap_junctions import (
    coupling_report, homologous_cpg_coupling, ipsilateral_cpg_coupling,
)
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

CSV = "data/circuit_map/all_legs_circuit.csv"


@pytest.fixture(scope="module")
def components():
    return build_trial_components(replicate=0, param_seed=PILOT_PARAM_SEED,
                                  duration_s=0.2, stim_current=380.0)


def test_coupling_matrix_is_symmetric_with_zero_diagonal(components):
    _m, _mg, _sg, wtable, _info = components
    for build in (homologous_cpg_coupling, ipsilateral_cpg_coupling):
        G = build(CSV, wtable, conductance=3.0)
        assert (abs(G - G.T) > 1e-12).nnz == 0, f"{build.__name__} not symmetric"
        assert G.diagonal().sum() == 0.0
        assert G.nnz > 0


def test_ipsilateral_coupling_never_crosses_the_midline(components):
    """The ipsilateral variant exists precisely so the left-right
    relationship is left to the connectome's own chemical synapses."""
    import pandas as pd
    _m, _mg, _sg, wtable, _info = components
    circuit = pd.read_csv(CSV)
    idx_to_side = {}
    bodyid_to_idx = {b: i for i, b in enumerate(wtable["bodyId"])}
    for _, row in circuit.iterrows():
        i = bodyid_to_idx.get(row["manc_bodyId"])
        if i is not None:
            idx_to_side[i] = row["side"]
    G = ipsilateral_cpg_coupling(CSV, wtable, conductance=3.0)
    r, c = G.nonzero()
    for a, b in zip(r, c):
        assert idx_to_side[a] == idx_to_side[b], (
            "ipsilateral coupling crossed the midline")


def test_gap_junctions_do_not_modify_the_connectome(components):
    """The hard rule. `W_eff` must be untouched by attaching coupling."""
    model, _mg, _sg, wtable, _info = components
    before = model.neurons.w_eff.copy()
    G = homologous_cpg_coupling(CSV, wtable, conductance=25.0)
    model.gap_junctions = G
    model._gj_leak = np.asarray(G.sum(axis=1)).ravel()
    model.reset()
    for _ in range(50):
        model.step(0.001)
    after = model.neurons.w_eff
    assert (abs(before - after) > 0).nnz == 0, (
        "the connectome's weight matrix changed while gap junctions were active")


def test_zero_conductance_is_bit_identical_to_no_coupling(components):
    """The ablation the result rests on: g=0 must be the published model,
    exactly — not approximately."""
    model, _mg, _sg, wtable, _info = components

    model.gap_junctions = None
    model.reset()
    baseline = np.stack([model.step(0.001) for _ in range(200)])

    G = homologous_cpg_coupling(CSV, wtable, conductance=0.0)
    model.gap_junctions = G if G.nnz else None
    model._gj_leak = (np.asarray(G.sum(axis=1)).ravel() if G.nnz else None)
    model.reset()
    zero = np.stack([model.step(0.001) for _ in range(200)])

    assert np.array_equal(baseline, zero), (
        "conductance 0 did not reproduce the uncoupled model bit for bit; "
        "the K=0 ablation would not be a valid control")


def test_nonzero_conductance_actually_changes_the_dynamics(components):
    """The positive half: a gate that changes nothing proves nothing."""
    model, _mg, _sg, wtable, _info = components
    model.gap_junctions = None
    model.reset()
    baseline = np.stack([model.step(0.001) for _ in range(200)])

    G = homologous_cpg_coupling(CSV, wtable, conductance=25.0)
    model.gap_junctions = G
    model._gj_leak = np.asarray(G.sum(axis=1)).ravel()
    model.reset()
    coupled = np.stack([model.step(0.001) for _ in range(200)])
    assert not np.array_equal(baseline, coupled)


def test_coupling_report_counts_what_was_added(components):
    _m, _mg, _sg, wtable, _info = components
    rep = coupling_report(homologous_cpg_coupling(CSV, wtable, conductance=7.0))
    assert rep["conductance"] == pytest.approx(7.0)
    assert rep["n_coupled_neurons"] == 18      # 3 roles x 6 legs
    assert rep["n_electrical_connections"] == 3 * 6 * 5   # all-to-all within role
