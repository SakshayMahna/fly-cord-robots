"""End-to-end: one real 4 s closed-loop trial, reference vs active-column step.

Monkeypatches ONLY the _derivative implementation (same arithmetic) so the
comparison is like-for-like through run_trial itself. Verifies the trial
outputs are identical before reporting any speedup.
"""
import time
import numpy as np
import scipy.sparse as sp

from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
from fly_robot.sim.closed_loop import run_trial
from fly_robot.neural.steppable_rate_model import SteppableRateModel

G_FB, ENC = 3.5, "deviation"


def make_fast(model):
    Wc = sp.csc_matrix(model.neurons.w_eff)
    outdeg = np.diff(Wc.indptr)
    total_nnz = model.neurons.w_eff.nnz

    def _derivative(self, t, r, extra_input):
        nn = self.neurons
        idx = np.flatnonzero(r)
        # guard: fall back to the dense path once the active set is big enough
        # that gathering columns costs more than it saves (measured crossover
        # is around 50% of nnz)
        if outdeg[idx].sum() < 0.35 * total_nnz:
            total = Wc[:, idx] @ r[idx]
        else:
            total = nn.w_eff @ r
        if (t >= self.pulse_start) & (t <= self.pulse_end):
            total = total + self.baseline_input
        if extra_input is not None:
            total = total + extra_input
        act = np.maximum(nn.fr_cap * np.tanh((nn.a / nn.fr_cap) * (total - nn.threshold)), 0.0)
        return (act - r) / nn.tau
    return _derivative


results = {}
for label in ("reference", "fast"):
    model, mg, sg, wt, info = build_trial_components(
        replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=4.0)
    orig = SteppableRateModel._derivative
    if label == "fast":
        SteppableRateModel._derivative = make_fast(model)
    t0 = time.time()
    res = run_trial(model, mg, sensory_groups=sg, feedback_gain=G_FB, on_ball=True,
                    duration_s=4.0, seed=1, encoder_mode=ENC)
    wall = time.time() - t0
    SteppableRateModel._derivative = orig
    results[label] = (res, wall)
    print(f"{label:10s} wall={wall:6.2f}s  (run_trial internal {res.wall_clock_s:6.2f}s)  "
          f"n_active={res.n_active_neurons}  max_fr={res.max_firing_rate:.3f}")

ref, fast = results["reference"][0], results["fast"][0]
print("\nidentical outputs?")
for name in ("motor_rates", "joint_angles", "sensory_drive"):
    a, b = getattr(ref, name), getattr(fast, name)
    print(f"  {name:15s} max|diff| = {np.abs(a - b).max():.3e}")
print(f"  n_active  {ref.n_active_neurons} vs {fast.n_active_neurons}")
print(f"\nspeedup: {results['reference'][1]/results['fast'][1]:.2f}x end-to-end")
