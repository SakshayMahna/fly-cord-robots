"""Parallel scaling of the FAST (active-column) neural step across CPU cores.

The sparse matvec is memory-bandwidth bound, so cores do not scale for free.
This measures it on the real W_eff, using the fast path we would actually
run. Stage 1 dumps the matrix once; workers memory-map it back.
"""
import multiprocessing as mp
import os
import tempfile
import time
import numpy as np
import scipy.sparse as sp

CACHE_DIR = os.environ.get("PHASE5_BENCH_CACHE", tempfile.gettempdir())
CACHE = os.path.join(CACHE_DIR, "phase5_weff.npz")
PARAMS = os.path.join(CACHE_DIR, "phase5_params.npz")
STEPS = 400


def build_cache():
    from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
    model, *_ = build_trial_components(replicate=1, param_seed=PILOT_PARAM_SEED,
                                       duration_s=4.0)
    sp.save_npz(CACHE, sp.csc_matrix(model.neurons.w_eff))
    model.reset()
    for _ in range(1000):           # settle into the real active state
        r = model.step(0.001)
    n = model.neurons
    np.savez(PARAMS, tau=n.tau, a=n.a, threshold=n.threshold, fr_cap=n.fr_cap,
             baseline=model.baseline_input, r0=r)
    print(f"cached: {(r>0).sum()} active neurons at t=1.0s")


def worker(args):
    seed, steps = args
    Wc = sp.load_npz(CACHE)
    P = np.load(PARAMS)
    tau, a, thr, cap = P["tau"], P["a"], P["threshold"], P["fr_cap"]
    base, r = P["baseline"], P["r0"].copy()

    def deriv(rr):
        idx = np.flatnonzero(rr)
        total = Wc[:, idx] @ rr[idx] + base
        return (np.maximum(cap * np.tanh((a / cap) * (total - thr)), 0.0) - rr) / tau

    t0 = time.perf_counter()
    h = 0.001
    for _ in range(steps):
        k1 = deriv(r); k2 = deriv(r + h/2*k1); k3 = deriv(r + h/2*k2); k4 = deriv(r + h*k3)
        r = np.clip(r + (h/6)*(k1 + 2*k2 + 2*k3 + k4), 0, 1000)
    return time.perf_counter() - t0


if __name__ == "__main__":
    if not os.path.exists(CACHE):
        build_cache()
    print(f"cores: {mp.cpu_count()}   steps per worker: {STEPS}")
    base_tp = None
    for nproc in (1, 2, 4, 6, 8, 10):
        with mp.Pool(nproc) as pool:
            t0 = time.perf_counter()
            pool.map(worker, [(i, STEPS) for i in range(nproc)])
        wall = time.perf_counter() - t0
        tp = nproc * STEPS / wall                      # neural steps / second, aggregate
        trials_hr = tp / 4000 * 3600                   # 4000 neural steps = one 4 s trial
        if base_tp is None:
            base_tp = tp
            print(f"  nproc={nproc:2d}  wall={wall:6.2f}s  {tp:7.1f} steps/s  "
                  f"~{trials_hr:6.1f} neural-only trials/hr  (baseline)")
        else:
            print(f"  nproc={nproc:2d}  wall={wall:6.2f}s  {tp:7.1f} steps/s  "
                  f"~{trials_hr:6.1f} neural-only trials/hr  "
                  f"scaling {tp/base_tp:4.2f}x ({tp/base_tp/nproc*100:3.0f}% eff)")
