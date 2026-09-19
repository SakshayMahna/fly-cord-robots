"""Control manipulations for the closed-loop experiments (C1 and C2).

These exist to answer "is the *real wiring* doing the work?" and "is
*structured* feedback doing the work?" — the two ways a coordination
result could be trivially explained away.

Neither of these is used in any experimental condition. The real
connectome is never modified: `degree_preserving_shuffle` returns a NEW
matrix and the original is untouched, exactly as this project's honesty
rule requires. A shuffled connectome is a labelled control, never a
substitute for the real one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _sorted_contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Vectorised membership test against a sorted int64 array."""
    pos = np.searchsorted(sorted_keys, query)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    return sorted_keys[pos] == query


def _edge_list(w: sp.spmatrix):
    coo = sp.coo_matrix(w)
    return coo.row.copy(), coo.col.copy(), coo.data.copy()


def degree_preserving_shuffle(w, seed: int = 0, rounds: int = 10,
                              protected_rows: np.ndarray | None = None,
                              protected_cols: np.ndarray | None = None):
    """C1: randomise who connects to whom, preserving every degree.

    Implements the classic double-edge swap with rejection: taking two
    edges a->b and c->d and rewiring them to a->d and c->b leaves the
    out-degree of a and c and the in-degree of b and d all unchanged. A
    swap is **rejected** if it would create a self-loop or duplicate an
    edge that already exists, because a duplicate is not representable in
    a weight matrix — `csr_matrix` silently *sums* two entries at the same
    position, which destroys edges and breaks the degree preservation
    this control depends on. (That is not hypothetical: a first version
    of this function permuted targets wholesale and repaired collisions
    afterwards, and lost 15 edges with out-degree errors up to 11.
    Rejection makes the guarantee structural rather than something to
    check for afterwards.)

    Each round draws a random perfect matching over the eligible edges —
    `rng.permutation` then consecutive pairs — so every proposed swap
    within a round touches disjoint edges and they cannot interfere.
    `rounds` is how many times each edge is offered a swap on average.

    **Within sign class.** Excitatory edges are only ever swapped with
    excitatory ones and inhibitory with inhibitory, so the
    excitatory/inhibitory balance of the network and of every individual
    neuron is untouched. This is well-defined here because the connectome
    obeys Dale's law exactly — verified directly, 0 of 22,769
    presynaptic neurons have outgoing edges of both signs — so each
    source's edges all live in one class and a source can never change
    sign as a side effect.

    **Weights travel with their edge.** The (source, weight) pair stays
    together while the target moves, matching the a->d gets w1, c->b gets
    w2 convention of the double-edge swap. The multiset of weights is
    therefore identical to the original.

    **Identities are never relabelled.** Row and column i mean the same
    neuron before and after, so sensory current still goes into the
    neurons annotated as that leg's sensory neurons, and the motor
    readout still reads the neurons annotated as that leg's motor
    neurons. Only the wiring between them is randomised.

    `protected_rows` / `protected_cols` optionally hold edges out of the
    shuffle entirely — an edge whose source is in `protected_rows` or
    whose target is in `protected_cols` is left exactly as it was. See
    the note in `docs/closed_loop/PREREGISTRATION.md` about why one might
    want to protect the motor neurons' *incoming* edges: with them
    shuffled, each leg's motor pool is driven by arbitrary interneurons,
    so a loss of coordination could come from the readout becoming
    meaningless rather than from coupling genuinely failing.

    Returns `(shuffled_matrix, report_dict)`.
    """
    w = sp.csr_matrix(w)
    rows, cols, data = _edge_list(w)
    rng = np.random.default_rng(seed)

    protected = np.zeros(len(rows), dtype=bool)
    if protected_rows is not None:
        protected |= np.isin(rows, np.asarray(protected_rows))
    if protected_cols is not None:
        protected |= np.isin(cols, np.asarray(protected_cols))

    n_cols = w.shape[1]
    new_cols = cols.copy()
    n_accepted = 0
    n_proposed = 0

    for mask_sign in (data > 0, data < 0):
        idx = np.where(mask_sign & ~protected)[0]
        if len(idx) < 2:
            continue
        for _round in range(rounds):
            shuffled_idx = rng.permutation(idx)
            if len(shuffled_idx) % 2:
                shuffled_idx = shuffled_idx[:-1]
            i, j = shuffled_idx[0::2], shuffled_idx[1::2]
            n_proposed += len(i)

            # Proposed rewiring: i keeps its source, takes j's target.
            new_i_col, new_j_col = new_cols[j], new_cols[i]
            no_self_loop = (rows[i] != new_i_col) & (rows[j] != new_j_col)

            existing = np.sort(rows.astype(np.int64) * n_cols + new_cols.astype(np.int64))
            key_i = rows[i].astype(np.int64) * n_cols + new_i_col.astype(np.int64)
            key_j = rows[j].astype(np.int64) * n_cols + new_j_col.astype(np.int64)
            # A swap is fine if neither new edge already exists, and the
            # two new edges are not identical to each other.
            free = ~_sorted_contains(existing, key_i) & ~_sorted_contains(existing, key_j)
            accept = no_self_loop & free & (key_i != key_j)

            # Two swaps in the SAME round can each be individually valid
            # yet independently create the same new edge — neither sees
            # the other, because `existing` was snapshotted before the
            # round. Drop any swap whose new edges collide with another
            # accepted swap's. (Without this, 9,014 edges silently
            # collapsed on the full matrix.)
            acc_idx = np.where(accept)[0]
            if len(acc_idx):
                proposed = np.concatenate([key_i[acc_idx], key_j[acc_idx]])
                uniq, counts = np.unique(proposed, return_counts=True)
                clashing = uniq[counts > 1]
                if len(clashing):
                    bad = (_sorted_contains(clashing, key_i[acc_idx])
                           | _sorted_contains(clashing, key_j[acc_idx]))
                    accept[acc_idx[bad]] = False

            new_cols[i[accept]], new_cols[j[accept]] = new_i_col[accept], new_j_col[accept]
            n_accepted += int(accept.sum())

    shuffled = sp.csr_matrix((data, (rows, new_cols)), shape=w.shape)
    assert shuffled.nnz == len(data), (
        f"{len(data) - shuffled.nnz} edges collapsed — duplicates were created and "
        "summed. Degree preservation is broken; do not use this matrix."
    )

    out_before = np.asarray((w != 0).sum(axis=1)).ravel()
    out_after = np.asarray((shuffled != 0).sum(axis=1)).ravel()
    in_before = np.asarray((w != 0).sum(axis=0)).ravel()
    in_after = np.asarray((shuffled != 0).sum(axis=0)).ravel()

    report = {
        "n_edges": int(len(data)),
        "n_swaps_proposed": int(n_proposed),
        "n_swaps_accepted": int(n_accepted),
        "swap_acceptance_rate": float(n_accepted / max(n_proposed, 1)),
        "n_edges_protected": int(protected.sum()),
        "frac_targets_changed": float((new_cols != cols).mean()),
        "out_degree_preserved": bool(np.array_equal(out_before, out_after)),
        "in_degree_preserved": bool(np.array_equal(in_before, in_after)),
        "weight_multiset_preserved": bool(
            np.array_equal(np.sort(shuffled.data), np.sort(w.data))),
        "n_positive_before": int((w.data > 0).sum()),
        "n_positive_after": int((shuffled.data > 0).sum()),
        "max_in_degree_error": int(np.abs(in_before - in_after).max()),
        "max_out_degree_error": int(np.abs(out_before - out_after).max()),
        # Existing self-loops can be swapped away (new ones are never
        # created), so this count only ever falls. 16 of 1.37M edges in
        # the real matrix — reported rather than hidden.
        "n_self_loops_before": int((rows == cols).sum()),
        "n_self_loops_after": int((rows == new_cols).sum()),
    }
    return shuffled, report


def phase_randomised_surrogate(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """C2: a surrogate with the SAME amplitude spectrum and randomised
    phases — identical mean, variance and autocorrelation to the real
    signal, but no timing relationship to the body that produced it.

    This is the rate-matched-noise control. It is deliberately built from
    **each channel's own recorded feedback from E3**, not from synthetic
    noise: matching the real drive's spectrum means the closed loop
    receives the same amount and the same kind of input, and the only
    thing removed is whether that input arrives at the right moment.
    Anything that survives it was caused by extra drive, not by feedback
    being informative.
    """
    x = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, spectrum.shape)
    phases[0] = 0.0
    if x.shape[-1] % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=x.shape[-1])
    # Preserve the mean exactly; irfft of a randomised spectrum is
    # zero-mean up to floating point.
    return surrogate - surrogate.mean() + x.mean()


def rate_matched_noise_channels(recorded_drive: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply `phase_randomised_surrogate` independently to each recorded
    sensory channel.

    `recorded_drive` is (n_channels, n_timesteps) — the actual sensory
    input each channel delivered during a matched E3 trial. Channels are
    randomised independently, so any cross-channel timing structure is
    destroyed along with the timing relation to the body.
    """
    rng = np.random.default_rng(seed)
    drive = np.atleast_2d(np.asarray(recorded_drive, dtype=float))
    return np.stack([phase_randomised_surrogate(ch, rng) for ch in drive])
