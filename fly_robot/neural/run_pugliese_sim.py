"""Run Pugliese et al.'s own (unmodified) firing-rate simulation code in
two modes:

1. `t1` — reproduces their published result exactly: their T1 (front leg)
   network, their T1 config, stimulating DNg100. This is Phase 0's
   original goal (a validation that our environment/pipeline reproduces
   their published rhythmic motor output), not something we invented.

2. `full_vnc` — our own addition, not in the paper: the SAME dynamics
   equations and the SAME unmodified DNg100 stimulation, but run on the
   true whole-VNC network (all six legs, 23628 neurons) instead of their
   T1-restricted network. This directly tests whether the hypothesis
   from `identify_all_legs.py` (that T2/T3 have their own DNg100-driven
   CPGs) produces actual rhythmic output, not just matching connectivity
   weights — the test we didn't yet have. Nothing about their model
   equations, connectome weights, or neurotransmitter signs is modified;
   we only feed the simulator a larger real slice of the same connectome.

Both modes use their exact `compute_oscillation_score` (autocorrelation
peak-based, normalized against a reference sine wave) — we do not invent
our own rhythm metric.

Usage:
    python -m fly_robot.neural.run_pugliese_sim --mode t1
    python -m fly_robot.neural.run_pugliese_sim --mode full_vnc
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("FLY_ROBOT_REPO", str(PROJECT_ROOT))

import jax
jax.config.update("jax_default_device", jax.devices("cpu")[0])
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PUGLIESE_REPO = PROJECT_ROOT / "external" / "Pugliese_cpg_2025"
sys.path.insert(0, str(PUGLIESE_REPO))

from src.simulation.vnc_sim import prepare_neuron_params, prepare_sim_params, run_single_simulation
from src.utils.sim_utils import load_wTable, compute_oscillation_score
from src.utils.path_utils import create_fresh_config_with_paths, register_custom_resolvers

register_custom_resolvers()


def run_single_simulation_helper(W, neuronParams, simParams, key):
    """Verbatim from Pugliese's Tutorial 1 notebook — not our method."""
    from src.simulation.vnc_sim import reweight_connectivity
    W_masked = W * neuronParams.W_mask
    W_reweighted = reweight_connectivity(W_masked, simParams.exc_multiplier, simParams.inh_multiplier)
    return run_single_simulation(
        W_reweighted, neuronParams.tau, neuronParams.a, neuronParams.threshold,
        neuronParams.fr_cap, neuronParams.input_currents[0], simParams.noise_stdv,
        simParams.t_axis, simParams.T, simParams.dt, simParams.pulse_start,
        simParams.pulse_end, simParams.r_tol, simParams.a_tol, key,
    )


def adaptive_stim_single(W, neuronParams_single, simParams, key, initial_stim_mask,
                          n_active_lower=5, n_active_upper=500,
                          n_high_fr_upper=100, high_fr_threshold=100.0,
                          max_iters=10):
    """Faithful re-implementation, for a single simulation, of the doubling
    /halving stimulus-search procedure documented in Pugliese et al.'s
    Methods ("Descending neuron activation screen": "If the simulation was
    underactive, Istim was doubled... If it was oversaturated, Istim was
    halved... repeated up to a maximum of 10 times") and implemented for
    their batched runner in `vnc_sim._adjust_stimulation_for_batch`
    (src/simulation/vnc_sim.py:1143). We can't reuse that function directly
    — it's written for their pmap/batch/checkpoint runner — so this mirrors
    its exact thresholds and bound-narrowing logic for the n_replicates=1
    case we need. `n_active` here uses their exact criterion from that
    function: sum(R, axis=time) > 0, i.e. any nonzero activity at all, not
    the stricter >0.01 threshold used elsewhere for motor-neuron scoring.
    """
    input_currents = neuronParams_single.input_currents
    stim_value = float(input_currents[initial_stim_mask][0])
    next_highest, next_lowest = None, None

    for it in range(max_iters):
        params_iter = neuronParams_single._replace(input_currents=input_currents)
        R = run_single_simulation_helper(W, params_iter, simParams, key)
        R = np.array(R)

        n_active = int((R.sum(axis=1) > 0).sum())
        n_high_fr = int((R.max(axis=1) > high_fr_threshold).sum())
        print(f"  [adaptive] iter {it + 1}/{max_iters}: stimI={stim_value:.2f}, "
              f"n_active={n_active}, n_high_fr={n_high_fr}")

        oversaturated = (n_active > n_active_upper) or (n_high_fr > n_high_fr_upper)
        underactive = n_active < n_active_lower

        if not oversaturated and not underactive:
            print(f"  [adaptive] converged: stimI={stim_value:.2f}")
            return R, stim_value, True

        if oversaturated:
            next_highest = stim_value
            stim_value = stim_value / 2 if next_lowest is None else (stim_value + next_lowest) / 2
        else:  # underactive
            next_lowest = stim_value
            stim_value = stim_value * 2 if next_highest is None else (stim_value + next_highest) / 2

        input_currents = jnp.where(initial_stim_mask, stim_value, input_currents)

    print(f"  [adaptive] did not converge after {max_iters} iterations (last stimI={stim_value:.2f})")
    return R, stim_value, False


def run_experiment(experiment_name: str, rtol=1e-4, atol=1e-7, seed=0, adaptive=False):
    # config_dir left at its default ("../../configs", relative to
    # path_utils.py's own location inside the Pugliese repo) — Hydra's
    # initialize() requires a relative path, so we can't point it at our
    # absolute PUGLIESE_REPO path directly.
    config = create_fresh_config_with_paths(
        experiment=experiment_name, paths_template="fly_robot",
        run_id=f"phase0_{experiment_name}",
    )
    config.experiment.n_replicates = 1
    config.sim.rtol = rtol
    config.sim.atol = atol

    wTable = load_wTable(config.experiment.dfPath)
    neuronParams = prepare_neuron_params(config, wTable)
    nNeurons = len(neuronParams.W)
    simParams = prepare_sim_params(config, 1, nNeurons)

    neuronParams_single = jax.tree.map(lambda x: x[0], neuronParams)
    key = jax.random.PRNGKey(seed)

    t0 = time.time()
    if adaptive:
        stim_mask = neuronParams_single.input_currents != 0
        R, final_stim, converged = adaptive_stim_single(
            neuronParams.W, neuronParams_single, simParams, key, stim_mask,
        )
        R = jnp.asarray(R)
        print(f"[{experiment_name}] adaptive stim search {'converged' if converged else 'did NOT converge'} "
              f"at stimI={final_stim:.2f}")
    else:
        R = run_single_simulation_helper(neuronParams.W, neuronParams_single, simParams, key)
        R.block_until_ready()
    elapsed = time.time() - t0
    print(f"[{experiment_name}] simulated {nNeurons} neurons, {simParams.T}s @ dt={simParams.dt} "
          f"in {elapsed:.1f}s wall-clock")

    return wTable, np.array(R), simParams


def score_population(R: np.ndarray, wtable: pd.DataFrame, mask: np.ndarray, label: str, dt: float):
    """Reuses Pugliese's exact compute_oscillation_score. Returns
    (mean_score, n_active, n_total) for the given boolean mask over rows.

    compute_oscillation_score's frequency is in cycles/sample (it's
    1/lag_in_samples of the autocorrelation peak) — their own analysis
    notebooks (e.g. Figure 2.ipynb: `mnFreq/overallParams.sim.dt`) divide
    by dt to convert to Hz. We do the same here; the raw value alone is
    not a physically meaningful frequency."""
    if mask.sum() == 0:
        print(f"  {label}: 0 neurons in this group — skipping")
        return None
    activity = jnp.asarray(R[mask])
    max_frs = activity.max(axis=1)
    active = np.array(max_frs) > 0.01
    active_mask = jnp.asarray(active)
    score, mean_freq_per_sample = compute_oscillation_score(activity, active_mask)
    mean_freq_hz = float(mean_freq_per_sample) / dt
    print(f"  {label}: {int(active.sum())}/{mask.sum()} active, "
          f"mean oscillation score (active only) = {float(score):.3f}, "
          f"mean freq = {mean_freq_hz:.2f} Hz")
    return {
        "label": label, "n_total": int(mask.sum()), "n_active": int(active.sum()),
        "mean_oscillation_score": float(score), "mean_frequency_hz": mean_freq_hz,
    }


def analyze(wTable: pd.DataFrame, R: np.ndarray, out_prefix: str, out_dir: Path, dt: float):
    results = []
    is_mn = (wTable["class"] == "motor neuron").to_numpy()
    print(f"Total motor neurons in network: {is_mn.sum()}")
    results.append(score_population(R, wTable, is_mn, "all motor neurons", dt))

    if "somaNeuromere" in wTable.columns and wTable["somaNeuromere"].notna().any():
        for seg in ["T1", "T2", "T3"]:
            seg_mn = is_mn & (wTable["somaNeuromere"] == seg).to_numpy()
            results.append(score_population(R, wTable, seg_mn, f"{seg} motor neurons", dt))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{out_prefix}_oscillation_scores.json", "w") as f:
        json.dump([r for r in results if r], f, indent=2)

    # A quick plot: a handful of active MN traces, for a visual sanity check.
    active_mn_idx = np.where(is_mn & (R.max(axis=1) > 0.01))[0]
    if len(active_mn_idx) > 0:
        plot_idx = active_mn_idx[:12]
        fig, ax = plt.subplots(figsize=(8, 4))
        t = np.arange(R.shape[1]) * 0.001
        for i, idx in enumerate(plot_idx):
            row = wTable.iloc[idx]
            ax.plot(t, R[idx] + i * 5, label=f"{row.get('somaNeuromere', '?')} {row['type']}", linewidth=0.8)
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.set_title(f"{out_prefix}: active motor neuron traces (stacked, offset for visibility)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{out_prefix}_mn_traces.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / f'{out_prefix}_mn_traces.png'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["t1", "full_vnc"], required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--out-dir", default="media/simulation")
    parser.add_argument("--adaptive", action="store_true",
                         help="Use Pugliese et al.'s documented doubling/halving stimulus search "
                              "instead of a fixed stimI (see Methods, 'Descending neuron activation screen').")
    args = parser.parse_args()

    experiment = "DNg100_Stim" if args.mode == "t1" else "FullVNC_DNg100_Stim"
    wTable, R, simParams = run_experiment(experiment, rtol=args.rtol, atol=args.atol, adaptive=args.adaptive)
    analyze(wTable, R, out_prefix=args.mode, out_dir=Path(args.out_dir), dt=float(simParams.dt))
