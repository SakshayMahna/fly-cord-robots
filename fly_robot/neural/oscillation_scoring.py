"""Score and plot rhythmic motor-neuron output using Pugliese's own exact
`compute_oscillation_score` (autocorrelation peak-based, normalized
against a reference sine wave) — we do not invent our own rhythm metric.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fly_robot.neural import pugliese_paths  # noqa: F401 — sets sys.path/env before the import below
from src.utils.sim_utils import compute_oscillation_score


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


def score_and_plot_by_segment(wTable: pd.DataFrame, R: np.ndarray, out_prefix: str, out_dir: Path, dt: float):
    """Scores all motor neurons (and each T1/T2/T3 segment, if present),
    saves the scores as JSON, and plots a handful of active traces as a
    visual sanity check."""
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
        t = np.arange(R.shape[1]) * dt
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
