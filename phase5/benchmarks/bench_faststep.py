"""Full-step cost with the active-column matvec, and the runaway worst case.

The matvec is 11.8x faster, but _derivative also does dense tanh/maximum
over all 23,532 neurons four times per RK4 step. This measures the whole
step, which is the number that actually matters, and checks the result is
still bit-identical to the current model.
"""
import time
import numpy as np
import scipy.sparse as sp

from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
from fly_robot.neural.steppable_rate_model import SteppableRateModel

model, mg, sg, wtable, info = build_trial_components(
    replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=4.0)
W = model.neurons.w_eff
Wc = sp.csc_matrix(W)
n = W.shape[0]
nm = model.neurons


class FastModel(SteppableRateModel):
    """Identical arithmetic; skips the provably-zero columns of W."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._Wc = sp.csc_matrix(self.neurons.w_eff)

    def _derivative(self, t, r, extra_input):
        nn = self.neurons
        idx = np.flatnonzero(r)
        total = self._Wc[:, idx] @ r[idx]
        if (t >= self.pulse_start) & (t <= self.pulse_end):
            total = total + self.baseline_input
        if extra_input is not None:
            total = total + extra_input
        act = np.maximum(nn.fr_cap * np.tanh((nn.a / nn.fr_cap) * (total - nn.threshold)), 0.0)
        return (act - r) / nn.tau


def timeit(fn, k):
    fn()
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    return (time.perf_counter() - t0) / k


# --- correctness: run both 1000 steps open loop, compare -----------------
fast = FastModel(nm, model.baseline_input, model.pulse_start, model.pulse_end)
model.reset(); fast.reset()
for _ in range(1000):
    a = model.step(0.001)
    b = fast.step(0.001)
print(f"after 1000 open-loop steps: max|ref - fast| = {np.abs(a-b).max():.3e}")
print(f"  active neurons: ref {int((a>0).sum())}, fast {int((b>0).sum())}")

# --- speed at the realistic (stable) operating point ---------------------
model.rates = a.copy(); fast.rates = b.copy()
model.t = fast.t = 1.0
ref_s = timeit(lambda: model.step(0.001), 100)
fast_s = timeit(lambda: fast.step(0.001), 200)
print(f"\nSTABLE regime ({int((a>0).sum())} active):")
print(f"  reference step {ref_s*1e3:7.3f} ms -> {ref_s*4000:6.2f} s / 4 s trial")
print(f"  fast step      {fast_s*1e3:7.3f} ms -> {fast_s*4000:6.2f} s / 4 s trial"
      f"   ({ref_s/fast_s:.1f}x)")

# --- worst case: how does it scale with the number of active neurons? ----
print("\nscaling with active-neuron count (the runaway regime reaches ~4,600):")
outdeg = np.diff(Wc.indptr)
order = np.argsort(-outdeg)          # highest out-degree first = worst case
rng = np.random.default_rng(0)
for k in (471, 1000, 2000, 4629, 10000, 23532):
    # worst case: the k busiest columns are the active ones
    r = np.zeros(n)
    r[order[:k]] = rng.uniform(0.1, 20, k)
    nnz_touched = outdeg[order[:k]].sum()
    fast.rates = r.copy(); fast.t = 1.0
    model.rates = r.copy(); model.t = 1.0
    f = timeit(lambda: fast.step(0.001), 30)
    rf = timeit(lambda: model.step(0.001), 30)
    print(f"  active={k:6d} (worst-case cols, {nnz_touched/W.nnz*100:5.1f}% of nnz)  "
          f"fast {f*1e3:7.3f} ms  ref {rf*1e3:7.3f} ms  speedup {rf/f:5.2f}x"
          f"   -> {f*4000:6.1f} s/trial")
