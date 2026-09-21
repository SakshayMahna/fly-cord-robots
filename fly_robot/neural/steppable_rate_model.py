"""A steppable version of Pugliese et al.'s firing-rate VNC model.

WHY THIS EXISTS (and what is / isn't changed from their code)
------------------------------------------------------------
Their `run_single_simulation` solves the whole 2 s window in one
`diffrax.diffeqsolve` call with an adaptive Dopri5 controller. That is
ideal for open-loop runs and useless for closed-loop ones: to feed
sensory input back from a physics body, the integration has to be
advanced in small increments with new input injected between them.

What is IDENTICAL to their model:
  * The rate equation itself. We call their own
    `rate_equation_half_tanh` semantics verbatim:
        total = I(t) + W_eff @ R
        act   = max(fr_cap * tanh((a / fr_cap) * (total - threshold)), 0)
        dR/dt = (act - R) / tau
  * The connectivity `W_eff = reweight_connectivity(W, exc_mult, inh_mult)`
    — transpose, then scale positive and negative entries separately.
    **No weight, sign, or topology is altered** (project honesty rule).
  * Per-neuron parameters (tau / a / threshold / fr_cap), which we take
    straight from their `prepare_neuron_params`, including their
    `set_sizes` size-scaling.

What is DIFFERENT, and therefore has to be validated (see
`validate_against_published.py`):
  1. **Integrator**: fixed-step RK4 instead of adaptive Dopri5. Adaptive
     control cannot be carried across the tiny segments a closed loop
     needs — restarting it every 1 ms costs ~90 RHS evaluations per
     millisecond (measured), i.e. ~80 min per 2 s trial, versus 4 for
     fixed-step RK4.
  2. **Sparse matrix-vector product** instead of dense. W is 0.248%
     dense (1,372,404 nonzeros of 553.7M), so a CSR matvec is ~32x
     faster than the dense one (0.82 ms vs 26 ms measured) and drops
     the matrix from 2.06 GiB to 11 MiB. This is a pure representation
     change — the same numbers, just not storing the zeros.
     Floating-point summation order differs, so results are close but
     not bit-identical; that is what the validation quantifies.
  3. **Input may vary over time.** Their `I` is a constant vector gated
     by a pulse window. Here the same pulse-gated constant is the
     `baseline_input`, and an optional `extra_input` (the sensory
     feedback) is added on top, held constant across each step. With
     `extra_input = 0` the model is open-loop and must reproduce their
     result — that is exactly the g_fb = 0 test.

What is NOT a difference: the active-column matvec
----------------------------------------------------
`W @ R` is evaluated over only the **nonzero entries of R**. This is not
an approximation, a reduced model, or a topology change — a neuron whose
rate is exactly 0 contributes exactly `w * 0.0 = 0.0` to every
postsynaptic sum, and dropping terms that are exactly zero cannot change
a floating-point result. The summation ORDER is preserved too: scipy's
CSR matvec accumulates each row in increasing column order, and a CSC
matvec accumulates each row in increasing column order as well, so the
surviving addends arrive in the same sequence. Verified empirically
bit-identical, not merely argued — see `tests/test_steppable_model.py`.

It matters because the network is *dynamically* sparse in a way the
matrix is not: in a stable trial only ~471 of 23,532 neurons are ever
nonzero (measured), touching 2.4% of W's 1,372,404 stored entries, while
scipy's CSR matvec touches all of them regardless. Measured end-to-end
through `run_trial`: 33.1 s -> 9.75 s per 4 s trial, outputs identical to
the last bit.

The win shrinks as the network saturates and **inverts** once gathering
columns costs more than it saves (measured: 0.74x, i.e. slower, with all
23,532 active). Hence `ACTIVE_COLUMN_NNZ_FRACTION` — above that share of
W's nonzeros the dense path is used instead, so the model is never slower
than before. Both paths are the same arithmetic; the guard only picks
which one runs.

Nothing here reads or writes the connectome; it only integrates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


# Above this share of W's stored entries falling in active columns, the
# column-gather costs more than it saves and the dense path is used.
# MEASURED, not chosen: the crossover sits near 50% of nnz (speedup 1.03x
# at 4,629 active neurons, 0.74x at 23,532), so 0.35 leaves margin and
# switches over before the fast path can ever be the slower one.
ACTIVE_COLUMN_NNZ_FRACTION = 0.35


@dataclass(frozen=True)
class NeuronSet:
    """Per-neuron biophysical parameters for ONE replicate, plus the
    effective (already reweighted) connectivity they share.

    `w_eff` is indexed [post, pre] — the transpose convention Pugliese's
    `reweight_connectivity` produces — so `w_eff @ R` is the synaptic
    input each neuron receives.
    """
    w_eff: sp.csr_matrix   # (n, n) float32, [post, pre]
    tau: np.ndarray        # (n,)
    a: np.ndarray          # (n,)
    threshold: np.ndarray  # (n,)
    fr_cap: np.ndarray     # (n,)

    @property
    def n_neurons(self) -> int:
        return self.w_eff.shape[0]


def reweight_connectivity_sparse(w: sp.spmatrix, exc_mult: float, inh_mult: float) -> sp.csr_matrix:
    """Sparse equivalent of Pugliese's `reweight_connectivity`: transpose,
    then scale positive and negative entries by separate multipliers.

    Their version is `exc*maximum(W.T, 0) + inh*minimum(W.T, 0)`. On a
    sparse matrix the same thing is done by scaling the stored values in
    place — structural zeros are unaffected either way, since both
    `maximum(0, 0)` and `minimum(0, 0)` are 0.
    """
    wt = sp.csr_matrix(w.T, dtype=np.float32, copy=True)
    data = wt.data
    wt.data = np.where(data > 0, exc_mult * data, inh_mult * data).astype(np.float32)
    return wt


def build_neuron_set(w_dense_or_sparse, tau, a, threshold, fr_cap,
                      exc_mult: float, inh_mult: float) -> NeuronSet:
    """Assembles a NeuronSet from a [pre, post] weight matrix and the
    per-neuron parameter vectors for one replicate.
    """
    w = sp.csr_matrix(w_dense_or_sparse)
    return NeuronSet(
        w_eff=reweight_connectivity_sparse(w, exc_mult, inh_mult),
        tau=np.asarray(tau, dtype=np.float64),
        a=np.asarray(a, dtype=np.float64),
        threshold=np.asarray(threshold, dtype=np.float64),
        fr_cap=np.asarray(fr_cap, dtype=np.float64),
    )


class SteppableRateModel:
    """Integrates Pugliese's rate equation one small step at a time.

    Usage (open loop, equivalent to their single run):
        m = SteppableRateModel(neuron_set, baseline_input, pulse_start, pulse_end)
        for _ in range(2000):
            m.step(0.001)

    Usage (closed loop): pass `extra_input` — the sensory drive computed
    from the body's current state — to each `step`.
    """

    def __init__(self, neurons: NeuronSet, baseline_input: np.ndarray,
                 pulse_start: float = 0.02, pulse_end: float = 1.999,
                 substeps: int = 1, active_column_matvec: bool = True):
        """`substeps` splits each `step(dt)` into `substeps` RK4 stages of
        dt/substeps. The sensory `extra_input` is held constant across
        the whole `dt` regardless (zero-order hold) — substeps only
        refine the integration, not the input.

        `active_column_matvec` selects the optimised matvec described in
        the module docstring. It is a pure speed switch: both settings
        compute the same numbers to the last bit, and the tests assert
        that rather than assuming it. Set False to force the original
        dense-vector path (used by the equivalence test, and available if
        the optimisation is ever suspected).
        """
        self.neurons = neurons
        self.baseline_input = np.asarray(baseline_input, dtype=np.float64)
        self.pulse_start = float(pulse_start)
        self.pulse_end = float(pulse_end)
        self.substeps = int(substeps)
        self.active_column_matvec = bool(active_column_matvec)
        # CSC view of the SAME matrix — a representation change, no copy of
        # any weight is altered. Built once; the transpose-free column
        # slicing below needs column-major storage.
        self._w_csc = sp.csc_matrix(neurons.w_eff) if self.active_column_matvec else None
        self._out_degree = (np.diff(self._w_csc.indptr)
                            if self.active_column_matvec else None)
        self._nnz_budget = (ACTIVE_COLUMN_NNZ_FRACTION * neurons.w_eff.nnz
                            if self.active_column_matvec else 0.0)
        # Diagnostics: how often the guard sent us down the dense path.
        self.n_matvec_fast = 0
        self.n_matvec_dense = 0
        self.reset()

    def _synaptic_input(self, r: np.ndarray) -> np.ndarray:
        """`W_eff @ r`, skipping columns whose rate is exactly zero.

        Identical arithmetic to `self.neurons.w_eff @ r` — see the module
        docstring for why dropping exactly-zero terms is exact and why the
        summation order is preserved.
        """
        if not self.active_column_matvec:
            self.n_matvec_dense += 1
            return self.neurons.w_eff @ r
        active = np.flatnonzero(r)
        if self._out_degree[active].sum() >= self._nnz_budget:
            # Saturated: gathering columns would cost more than it saves.
            self.n_matvec_dense += 1
            return self.neurons.w_eff @ r
        self.n_matvec_fast += 1
        return self._w_csc[:, active] @ r[active]

    def reset(self) -> None:
        """R0 = 0 and t = 0, matching their `run_single_simulation`."""
        self.t = 0.0
        self.rates = np.zeros(self.neurons.n_neurons, dtype=np.float64)
        self.n_matvec_fast = 0
        self.n_matvec_dense = 0

    def _derivative(self, t: float, r: np.ndarray, extra_input: np.ndarray | None) -> np.ndarray:
        n = self.neurons
        pulse_active = (t >= self.pulse_start) & (t <= self.pulse_end)
        total = self._synaptic_input(r)
        if pulse_active:
            total = total + self.baseline_input
        if extra_input is not None:
            total = total + extra_input
        activation = np.maximum(n.fr_cap * np.tanh((n.a / n.fr_cap) * (total - n.threshold)), 0.0)
        return (activation - r) / n.tau

    def step(self, dt: float, extra_input: np.ndarray | None = None) -> np.ndarray:
        """Advances by `dt` (classical RK4) and returns the new rates.

        `extra_input` is the sensory feedback current, in the same units
        as the stimulation current, held constant across the step.
        """
        h = dt / self.substeps
        r, t = self.rates, self.t
        for _ in range(self.substeps):
            k1 = self._derivative(t, r, extra_input)
            k2 = self._derivative(t + h / 2, r + h / 2 * k1, extra_input)
            k3 = self._derivative(t + h / 2, r + h / 2 * k2, extra_input)
            k4 = self._derivative(t + h, r + h * k3, extra_input)
            r = r + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + h
        # Their solver clips the saved output to [0, 1000] Hz; do the same
        # here so a divergence shows up identically rather than as a NaN.
        self.rates = np.clip(r, 0.0, 1000.0)
        self.t = t
        return self.rates

    def run_open_loop(self, n_steps: int, dt: float = 0.001) -> np.ndarray:
        """Convenience: step `n_steps` times with no sensory input and
        return the (n_neurons, n_steps + 1) trace, including the t=0
        initial state — the same layout their `saveat` produces.
        """
        out = np.empty((self.neurons.n_neurons, n_steps + 1), dtype=np.float32)
        out[:, 0] = self.rates
        for i in range(n_steps):
            out[:, i + 1] = self.step(dt)
        return out
