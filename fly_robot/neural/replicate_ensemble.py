"""Run, load, and combine multiple simulation replicates.

Pugliese's model resamples each neuron's biophysical parameters
(tau/threshold/gain/fr_cap) per replicate — a single run is one
stochastic sample, not a deterministic "the" answer. This module handles
the two real complications that come with that (see docstrings below):
some replicates go unstable and must be excluded, and raw time-series
must never be averaged across replicates (it cancels real rhythmic
signal) — only scalar summary statistics may be.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import sparse

from fly_robot.neural.pugliese_paths import PROJECT_ROOT, PUGLIESE_REPO
from fly_robot.neural.single_simulation import run_single_simulation_helper

import jax
from src.simulation.vnc_sim import prepare_neuron_params, prepare_sim_params
from src.utils.sim_utils import load_wTable
from src.utils.path_utils import create_fresh_config_with_paths

N_ACTIVE_UPPER = 1500  # Pugliese's own documented threshold ("Descending
                        # neuron activation screen" Methods): oversaturated /
                        # generally unstable simulations recruit more than
                        # this. Confirmed empirically against 8 real
                        # replicates here too: 6/8 landed in 313-728 active
                        # neurons (normal), 2/8 in 5718-8348 (a completely
                        # different regime) — 1500 sits cleanly in the gap.
                        # An earlier version of this code used 500 (a
                        # different config default, for a different
                        # experiment) and wrongly excluded every normal
                        # replicate — don't reintroduce that.


def run_experiment_multi_replicate(experiment_name: str, n_replicates: int,
                                    rtol=1e-4, atol=1e-7, base_seed=0,
                                    cache_path: Path | None = None):
    """Runs several replicates (each with its own randomly-sampled neuron
    parameters) and returns the RAW stack (not yet filtered or combined —
    see `filter_and_average_replicates` / `select_representative_replicate`).

    Why this exists (see docs/logs/2026-09-19.md, Phase 3): a single
    replicate only recruits a handful of the ~730 motor neurons in the
    full-VNC network (matches the paper's own Fig. 4b: "2-10 leg MNs"
    recruited per replicate) — using one arbitrary replicate as if it
    were a deterministic "the" signal is misleading, since even
    Pugliese's own paper never reports single-replicate output as
    representative, always aggregate statistics over 128 replicates.

    If cache_path is given, R_stack (and dt) are cached there (and loaded
    from there instead of re-simulating, if the file already exists) —
    each replicate costs several minutes, and re-running the same
    experiment repeatedly while iterating on the interface/joint mapping
    downstream would otherwise be wasteful. wTable is NOT cached (pandas
    DataFrames don't round-trip through np.savez the way a plain array
    does — an earlier version of this tried and silently corrupted it);
    it's cheap to reload from config.experiment.dfPath directly every
    time, cache hit or not.
    """
    config = create_fresh_config_with_paths(
        experiment=experiment_name, paths_template="fly_robot",
        run_id=f"phase0_{experiment_name}_multi",
    )
    config.experiment.n_replicates = n_replicates
    config.sim.rtol = rtol
    config.sim.atol = atol

    wTable = load_wTable(config.experiment.dfPath)

    if cache_path is not None and cache_path.exists():
        print(f"Loading cached replicates from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return wTable, cached["R_stack"], cached["dt"].item()

    neuronParams = prepare_neuron_params(config, wTable)
    nNeurons = len(neuronParams.W)
    simParams = prepare_sim_params(config, 1, nNeurons)

    t0 = time.time()
    R_list = []
    for i in range(n_replicates):
        neuronParams_i = jax.tree.map(lambda x: x[i], neuronParams)
        key = jax.random.PRNGKey(base_seed + i)
        R_i = run_single_simulation_helper(neuronParams.W, neuronParams_i, simParams, key)
        R_i.block_until_ready()
        R_list.append(np.array(R_i))
        n_active = (np.array(R_i).max(axis=1) > 0.01).sum()
        print(f"  replicate {i + 1}/{n_replicates} done ({n_active} neurons active)")
    elapsed = time.time() - t0

    R_stack = np.stack(R_list, axis=0)  # (n_replicates, n_neurons, n_timesteps)
    print(f"[{experiment_name}] {n_replicates} replicates of {nNeurons} neurons in "
          f"{elapsed:.1f}s wall-clock")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, R_stack=R_stack, dt=float(simParams.dt))
        print(f"Cached to {cache_path}")

    return wTable, R_stack, float(simParams.dt)


def load_published_full_vnc_replicates(
    rs_path: Path = PROJECT_ROOT / "data" / "pugliese_published_sim" / "simulations"
    / "DNg100_Stim_fullManc" / "hyak" / "30662613-33195857-33195876-33195882"
    / "DNg100_Stim_fullManc_Rs.npz",
    dfpath: Path = PUGLIESE_REPO / "data" / "manc full vnc data" / "wTable_20251006.feather",
):
    """Loads Pugliese's own real published full-VNC simulation output
    (128 replicates, downloaded from their Zenodo record 22260924 during
    Phase 0 — see docs/logs/2026-09-19.md) instead of running new
    simulations ourselves.

    Why this is preferable to running our own replicates: a single leg's
    single motor-neuron module often has only a handful of neurons, each
    independently recruited with modest (~5-25%, see per-module analysis
    in docs/logs) probability per replicate — the expected active count
    per (leg, module, replicate) is frequently well under 1. A handful of
    our own fresh replicates (at ~4 minutes each) can't average that out
    the way 128 real, already-computed ones can, at zero additional
    simulation cost. dt confirmed as 0.001s directly from their actual
    run_config.yaml (not assumed).
    """
    wtable = pd.read_feather(dfpath).reset_index(drop=True)  # load_wTable only handles .pkl/.csv, not .feather
    Rs = sparse.load_npz(rs_path)
    n_reps = Rs.shape[0]
    R_stack = np.stack([np.asarray(Rs[i].todense()) for i in range(n_reps)], axis=0)
    return wtable, R_stack, 0.001


def filter_and_average_replicates(R_stack: np.ndarray, n_active_upper: int = N_ACTIVE_UPPER):
    """Excludes replicates whose peak network-wide activity exceeds
    n_active_upper (Pugliese's own "oversaturated/unstable" criterion —
    see run_experiment_multi_replicate docstring), then averages the rest.

    IMPORTANT — do not use this to build a time-varying drive signal for
    the robot. Verified directly (docs/logs/2026-09-19.md, Phase 3): a
    given motor-neuron module can show real, substantial oscillatory
    activity (1-6 Hz) in every stable replicate individually, but each
    replicate's oscillation peaks at an essentially random time within
    the 2s window (no shared phase across replicates, since neuron
    parameters are resampled per replicate). Averaging raw time-series
    across replicates at matching time indices cancels this out — it
    does not reveal a "typical" rhythm, it destroys it. Pugliese's own
    paper never averages raw time-domain traces across replicates either,
    only scalar summary statistics (e.g. the oscillation score — see
    oscillation_scoring.py). This function is for computing statistics
    like that, NOT for producing a signal meant to preserve real timing —
    use `select_representative_replicate` for that instead.
    """
    n_active_per_rep = (R_stack.max(axis=2) > 0.01).sum(axis=1)  # (n_replicates,)
    stable = n_active_per_rep <= n_active_upper
    print(f"Replicates: {stable.sum()}/{len(stable)} stable (n_active <= {n_active_upper}), "
          f"excluded {(~stable).sum()} as oversaturated/unstable "
          f"(n_active={n_active_per_rep[~stable].tolist()})")
    if stable.sum() == 0:
        raise RuntimeError("All replicates were unstable — nothing to average. "
                            "Re-run with different seeds or check the stimulation protocol.")
    return R_stack[stable].mean(axis=0)


def select_representative_replicate(R_stack: np.ndarray, n_active_upper: int = N_ACTIVE_UPPER):
    """Returns ONE stable replicate's actual time-course (not an average —
    see filter_and_average_replicates docstring for why averaging destroys
    real phase/timing information). Picks the first stable replicate by
    index — not cherry-picked for a "good-looking" result, just the first
    one that passes the same stability filter used everywhere else.
    """
    n_active_per_rep = (R_stack.max(axis=2) > 0.01).sum(axis=1)
    stable_idxs = np.where(n_active_per_rep <= n_active_upper)[0]
    if len(stable_idxs) == 0:
        raise RuntimeError("All replicates were unstable — none to select from.")
    chosen = int(stable_idxs[0])
    print(f"Selected replicate {chosen} (n_active={n_active_per_rep[chosen]}) as the "
          f"representative time-course, out of {len(stable_idxs)} stable replicates "
          f"(first by index, not cherry-picked)")
    return R_stack[chosen]
