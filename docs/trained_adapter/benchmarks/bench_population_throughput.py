"""Population-parallel throughput: real 4 s closed-loop trials, N workers.

This is the number an ES trainer's wall time is actually made of — a
generation is P independent trials, so throughput is trials/hour with P
spread across workers. Unlike `bench_parallel2.py` (neural step only), every
trial here is the full loop: connectome, body, sensory encode, motor decode.

Two things are measured besides speed, because both turned out to constrain
the plan more than cores do:

  * peak RSS while BUILDING a worker's components (the dense weight matrix
    is materialised transiently), and
  * steady-state RSS per worker after `jax.clear_caches()`.

Builds are serialised behind a lock so the transient peaks do not overlap;
that is also how a real trainer should start its pool.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        docs/trained_adapter/benchmarks/bench_population_throughput.py [--trials K]
"""

import argparse
import gc
import multiprocessing as mp
import os
import time

import numpy as np
import psutil

TRIAL_S = 4.0
G_FB = 3.5
ENCODER = "deviation"

_build_lock = None
_start_barrier = None


def _init(lock, barrier):
    global _build_lock, _start_barrier
    _build_lock = lock
    _start_barrier = barrier


def worker(args):
    """Build once (under the lock), then run K trials and time them."""
    wid, n_trials = args
    import jax
    from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
    from fly_robot.sim.closed_loop import run_trial

    proc = psutil.Process(os.getpid())
    rss = lambda: proc.memory_info().rss / 1e9

    with _build_lock:
        before = rss()
        model, motor_groups, sensory_groups, _wt, _info = build_trial_components(
            replicate=wid % 4, param_seed=PILOT_PARAM_SEED, duration_s=TRIAL_S)
        peak_build = rss()
        # The dense W is held only by JAX's caches; the model itself needs
        # just the 5.5 MB sparse copy. Dropping it takes a worker from
        # ~3.3 GB to ~0.55 GB, which is what makes many workers affordable.
        for arr in jax.live_arrays():
            try:
                arr.delete()
            except Exception:
                pass
        jax.clear_caches()
        gc.collect()
        steady = rss()

    # Every worker waits here, so the timed phase is genuinely concurrent.
    # Without this, workers that built early run their trials while others
    # are still building, and the "contention" being measured is partly
    # just skew in the build queue.
    # Timeout so a synchronisation bug fails loudly instead of hanging (an
    # earlier version deadlocked silently for 43 minutes).
    _start_barrier.wait(timeout=900)

    times = []
    phase0 = time.perf_counter()
    for _ in range(n_trials):
        t0 = time.perf_counter()
        res = run_trial(model, motor_groups, sensory_groups=sensory_groups,
                        feedback_gain=G_FB, on_ball=True, duration_s=TRIAL_S,
                        seed=wid, encoder_mode=ENCODER)
        times.append(time.perf_counter() - t0)
    return dict(wid=wid, times=times, peak_build=peak_build, steady=steady,
                before=before, n_active=res.n_active_neurons,
                phase_s=time.perf_counter() - phase0, rss_end=rss())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3, help="trials per worker")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    args = ap.parse_args()

    total_ram = psutil.virtual_memory().total / 1e9
    print(f"cores={mp.cpu_count()}  RAM={total_ram:.1f} GB  "
          f"trial={TRIAL_S}s  trials/worker={args.trials}\n", flush=True)
    print(f"{'procs':>5} {'wall':>8} {'median trial':>13} {'trials/hr':>10} "
          f"{'scaling':>8} {'eff':>6} {'peak RSS/w':>11} {'steady/w':>9}", flush=True)

    base = None
    rows = []
    for nproc in args.procs:
        # Manager-backed, NOT mp.Barrier(): macOS spawns workers rather than
        # forking, and a raw Barrier passed through Pool initargs deadlocks
        # there (a raw Lock survives, which is why this only showed up once
        # the barrier was added). Manager proxies pickle correctly.
        manager = mp.Manager()
        lock, barrier = manager.Lock(), manager.Barrier(nproc)
        with mp.Pool(nproc, initializer=_init, initargs=(lock, barrier)) as pool:
            t0 = time.perf_counter()
            out = pool.map(worker, [(i, args.trials) for i in range(nproc)])
            wall = time.perf_counter() - t0

        all_times = np.concatenate([o["times"] for o in out])
        # Throughput over the barrier-synchronised phase only: the pool's
        # one-off build cost is not part of a generation's wall time.
        trial_wall = max(o["phase_s"] for o in out)
        tph = nproc * args.trials / trial_wall * 3600
        if base is None:
            base, scaling, eff = tph, 1.0, 1.0
        else:
            scaling, eff = tph / base, tph / base / nproc
        peak = max(o["peak_build"] for o in out)
        steady = max(o["steady"] for o in out)
        rows.append((nproc, tph, scaling))
        print(f"{nproc:5d} {wall:7.1f}s {np.median(all_times):12.2f}s "
              f"{tph:10.0f} {scaling:7.2f}x {eff*100:5.0f}% "
              f"{peak:10.2f}G {steady:8.2f}G", flush=True)

    print(f"\nmedian trial time, 1 worker: "
          f"{np.median(np.concatenate([o['times'] for o in out])):.2f}s")
    return rows


if __name__ == "__main__":
    main()
