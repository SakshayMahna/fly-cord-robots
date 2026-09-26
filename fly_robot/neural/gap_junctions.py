"""Electrical coupling (gap junctions) between leg CPGs — OUR ADDITION.

**This is not in the connectome, and it cannot be.** Electrical synapses are
10-20 nm, below the resolution of the electron microscopy used to build
every published Drosophila connectome, so they are absent from MANC,
MaleCNS, FANC and BANC alike. The wiring we have is a map of *chemical*
synapses only. Adding them back is therefore restoring a known biological
mechanism the dataset physically cannot contain — not inventing one.

Why this specific mechanism, and why now
----------------------------------------
Four measurements, in order, put us here:

1. Real flies coordinate their legs through **central** circuits, not
   sensory feedback: air-stepping flies (no ground contact at all) show the
   *highest* tripod coordination, and silencing proprioceptors preserves
   left-right coupling (Chen et al. 2026, "Central versus peripheral neural
   control of a coordinated walking pattern in Drosophila", bioRxiv
   2026.04.29.721658). So the body is not the missing ingredient.
2. The published simulation of those same central circuits produces **no**
   interleg coordination. Pugliese et al. report phase coupling absent
   across the six leg CPGs, and that including the VNC neurons which link
   left and right CPGs disynaptically "was insufficient to couple the
   phase".
3. Our own measurement: the six legs are not merely mis-phased, they run at
   **different frequencies** (median spread 3.03 Hz across published
   replicates; `GROUND_BRIDGE.md` §13). Phase coupling is impossible
   without frequency locking.
4. That spread is **not** an artifact of Pugliese's random per-neuron
   parameters. Setting every neuron's tau, gain and threshold spread to
   ~zero leaves a 1.5-2.8 Hz spread: the six leg circuits have structurally
   different natural frequencies, set by their own wiring.

Gap junctions are the canonical mechanism for synchronising biological
oscillators, and are directly implicated in insect motor coordination
(e.g. Nature 2023, "Gap junctions desynchronize a neural circuit to
stabilize insect flight"). They are exactly the thing that is both (a)
known to exist, (b) known to do this job, and (c) provably invisible to the
method that produced our data.

What this module does and does not touch
----------------------------------------
* It **never modifies `W_eff`**. The connectome's chemical synapse graph,
  its weights and its neurotransmitter signs are untouched, exactly as
  `CLAUDE.md` requires. Electrical coupling is a *separate*, symmetric,
  explicitly-labelled matrix applied alongside it.
* Chemical synapses act through the threshold nonlinearity; a gap junction
  does not. So it enters the derivative directly as a diffusive term,

      dr_i/dt = (activation_i - r_i)/tau_i + sum_j g_ij (r_j - r_i)

  which is symmetric, proportional to the state difference, and vanishes
  once cells are synchronised — the defining properties of electrical
  coupling, and the reason it locks oscillators.
* **The rate-model caveat, stated plainly:** a real gap junction couples
  *membrane potential*, while this model's state variable is firing rate.
  The diffusive form above is the standard rate-model approximation of
  electrical coupling, not a literal simulation of it. That is an
  approximation we are making, and it is ours to own.
* `conductance = 0` must reproduce the uncoupled model **bit for bit**.
  That is the ablation the whole result rests on, and it is asserted in
  the tests, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fly_robot.adapter.cpg_groups import CPG_ROLES


def homologous_cpg_coupling(circuit_csv: str, wtable: pd.DataFrame,
                            conductance: float,
                            roles: tuple = CPG_ROLES) -> sp.csr_matrix:
    """All-to-all electrical coupling between HOMOLOGOUS CPG neurons.

    Each of the three CPG-triad roles (`CPG_excit_hub`, `CPG_excit2`,
    `CPG_inhib`) has one neuron per leg. This couples the six copies of
    each role to one another — that is, the same cell type across
    segments and sides — and never couples different roles together.

    Why homologous rather than all-to-all across the whole triad: serially
    homologous neurons in different segments are the natural candidates for
    electrical coupling in a segmented nervous system, and it keeps the
    addition minimal and describable — three coupled groups of six, one
    scalar conductance, rather than an arbitrary dense block.

    Returns a symmetric CSR matrix `G` with zero diagonal, in the same row
    indexing as `wtable` (and therefore as `W_eff`).
    """
    circuit = pd.read_csv(circuit_csv)
    bodyid_to_idx = {b: i for i, b in enumerate(wtable["bodyId"])}
    n = len(wtable)

    rows, cols, vals = [], [], []
    for role in roles:
        idx = sorted({bodyid_to_idx[b] for b in
                      circuit.loc[circuit["role"] == role, "manc_bodyId"]
                      if b in bodyid_to_idx})
        for a in idx:
            for b in idx:
                if a != b:
                    rows.append(a)
                    cols.append(b)
                    vals.append(float(conductance))
    if not rows:
        return sp.csr_matrix((n, n))
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # Symmetry is a defining property of an electrical synapse; assert it
    # rather than trusting the construction above.
    assert (abs(G - G.T) > 1e-12).nnz == 0, "gap-junction matrix is not symmetric"
    assert G.diagonal().sum() == 0.0, "gap junction placed on the diagonal"
    return G


def ipsilateral_cpg_coupling(circuit_csv: str, wtable: pd.DataFrame,
                             conductance: float,
                             roles: tuple = CPG_ROLES) -> sp.csr_matrix:
    """Electrical coupling between homologous CPG neurons **on the same side
    only** — left legs to left legs, right to right, never across the midline.

    Why this variant exists. Diffusive coupling minimises differences, so
    coupling all six legs together drives them *in phase* — a pronk, not a
    tripod (measured: tripod index ~0 at the locking conductance). A tripod
    needs the two sides in ANTIphase, and electrical coupling cannot
    produce antiphase.

    Rather than hand the model the tripod pattern — which would be encoding
    the answer we are trying to test for — this couples only within a side
    and leaves the left-right relationship to the connectome's **own**
    contralateral chemical synapses. "Same side couples electrically" is an
    anatomical assumption; "these three legs step together" would be the
    result. We are only allowed the former.

    Returns a symmetric CSR matrix in `wtable` row indexing.
    """
    circuit = pd.read_csv(circuit_csv)
    bodyid_to_idx = {b: i for i, b in enumerate(wtable["bodyId"])}
    n = len(wtable)

    rows, cols, vals = [], [], []
    for role in roles:
        sub = circuit[circuit["role"] == role]
        for side in sorted(sub["side"].dropna().unique()):
            idx = sorted({bodyid_to_idx[b] for b in
                          sub.loc[sub["side"] == side, "manc_bodyId"]
                          if b in bodyid_to_idx})
            for a in idx:
                for b in idx:
                    if a != b:
                        rows.append(a)
                        cols.append(b)
                        vals.append(float(conductance))
    if not rows:
        return sp.csr_matrix((n, n))
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    assert (abs(G - G.T) > 1e-12).nnz == 0, "gap-junction matrix is not symmetric"
    assert G.diagonal().sum() == 0.0, "gap junction placed on the diagonal"
    return G


def coupling_report(G: sp.csr_matrix) -> dict:
    """Summary for the log: how much was added, and where."""
    coupled = sorted(set(G.nonzero()[0].tolist()))
    return {"n_coupled_neurons": len(coupled),
            "n_electrical_connections": int(G.nnz),
            "conductance": float(G.data.max()) if G.nnz else 0.0,
            "coupled_rows": coupled}
