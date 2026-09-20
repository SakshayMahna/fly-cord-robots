"""Pilot run for the closed-loop experiments.

PILOT SEEDS ONLY (parameter seed 641, replicate indices 0-7) — the draws
already used throughout the audit. Main-experiment seeds
(parameter seed 20260919) are never touched here, so nothing learned from
this pilot can leak into the main results through a tuning decision.

The pilot exists to answer three questions before committing six hours of
compute, not to test any hypothesis:

  1. How long does a trial actually take, end to end?
  2. Does anything go unstable across the pre-registered `g_fb` sweep —
     firing rates exploding, physics diverging, or the network recruiting
     so many neurons that it meets Pugliese's own oversaturation
     criterion?
  3. Do rhythms survive closing the loop at all? (Rhythmicity is
     evaluated first, per PREREGISTRATION.md §4a — if the rhythm is gone
     there is nothing to measure coupling on.)

Usage:
    MUJOCO_GL=cgl python -m fly_robot.experiments.closed_loop_pilot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fly_robot.analysis.interleg_coordination import LEGS, coordination
from fly_robot.neural.replicate_ensemble import N_ACTIVE_UPPER
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

# Pre-registered sweep (PREREGISTRATION.md §3).
G_FB_LEVELS = (0.0, 2.5, 5.0, 10.0, 20.0, 40.0)


def pilot_conditions(g_fb_levels):
    """(condition id, on_ball, uses_sensory, g_fb) for each pilot trial."""
    conditions = [("E0", False, False, 0.0), ("E2", True, False, 0.0),
                  ("E1", False, True, 10.0)]
    conditions += [("E3", True, True, g) for g in g_fb_levels]
    return conditions


def run(replicates, g_fb_levels, duration_s, out_dir: Path):
    rows = []
    total_start = time.time()

    for replicate in replicates:
        setup_start = time.time()
        model, motor_groups, sensory_groups, _wtable, info = build_trial_components(
            replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=duration_s)
        setup_s = time.time() - setup_start
        print(f"\n=== pilot replicate {replicate} (param seed {PILOT_PARAM_SEED}), "
              f"setup {setup_s:.1f}s ===", flush=True)

        for condition, on_ball, uses_sensory, g_fb in pilot_conditions(g_fb_levels):
            result = run_trial(
                model, motor_groups,
                sensory_groups=sensory_groups if uses_sensory else None,
                feedback_gain=g_fb, on_ball=on_ball,
                duration_s=duration_s, seed=replicate)

            rhythm, coord = coordination(result.motor_rates, NEURAL_DT,
                                          n_surrogates=100, seed=replicate)
            row = dict(
                replicate=replicate, condition=condition, on_ball=on_ball, g_fb=g_fb,
                wall_s=round(result.wall_clock_s, 1),
                n_active=result.n_active_neurons,
                oversaturated=result.n_active_neurons > N_ACTIVE_UPPER,
                max_fr=round(result.max_firing_rate, 1),
                unstable=result.unstable, reason=result.instability_reason,
                n_rhythmic=int(rhythm.rhythmic.sum()),
                has_rhythm=bool(coord.has_rhythm),
                median_dom_hz=round(float(np.median(rhythm.dominant_hz[rhythm.rhythmic]))
                                    , 2) if rhythm.rhythmic.any() else np.nan,
                tripod=round(float(coord.tripod_index), 3)
                if coord.has_rhythm and np.isfinite(coord.tripod_index) else np.nan,
                n_valid_pairs=int(coord.valid.sum()),
                n_sig_pairs=int(coord.significant.sum()),
                ball_pitch_rad_s=round(float(np.mean(result.ball_angvel[:, 1])), 3)
                if result.ball_angvel is not None else np.nan,
            )
            rows.append(row)
            print(f"  {condition:3s} ball={int(on_ball)} g_fb={g_fb:5.1f}  "
                  f"{result.wall_clock_s:5.1f}s  n_active={result.n_active_neurons:5d}"
                  f"{'  OVERSATURATED' if row['oversaturated'] else ''}"
                  f"  max_fr={result.max_firing_rate:7.1f}  "
                  f"rhythmic={row['n_rhythmic']}/6"
                  f"{'  UNSTABLE: ' + result.instability_reason if result.unstable else ''}",
                  flush=True)

    elapsed = time.time() - total_start
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "pilot_trials.csv", index=False)

    print("\n" + "=" * 74)
    print("PILOT SUMMARY")
    print("=" * 74)
    print(f"\n{len(frame)} trials in {elapsed / 60:.1f} min "
          f"({frame.wall_s.mean():.1f}s per trial, "
          f"{frame.wall_s.min():.1f}-{frame.wall_s.max():.1f}s)")

    print("\n--- stability and rhythm, by condition ---")
    summary = frame.groupby(["condition", "g_fb"]).agg(
        n=("replicate", "size"),
        median_n_active=("n_active", "median"),
        frac_oversaturated=("oversaturated", "mean"),
        max_fr=("max_fr", "max"),
        frac_unstable=("unstable", "mean"),
        median_rhythmic_legs=("n_rhythmic", "median"),
        frac_has_rhythm=("has_rhythm", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    projected = frame.wall_s.mean() * 940 / 3600
    print(f"\nProjected full matrix (940 trials): {projected:.1f} hours")

    (out_dir / "pilot_summary.json").write_text(json.dumps({
        "n_trials": len(frame), "elapsed_min": round(elapsed / 60, 1),
        "mean_trial_s": round(float(frame.wall_s.mean()), 1),
        "projected_full_matrix_hours": round(float(projected), 1),
        "any_unstable": bool(frame.unstable.any()),
        "frac_oversaturated": round(float(frame.oversaturated.mean()), 3),
        "param_seed": PILOT_PARAM_SEED,
        "replicates": list(map(int, replicates)),
    }, indent=2))
    print(f"\nWrote {out_dir}/pilot_trials.csv and pilot_summary.json")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="pilot replicate indices (must be within 0-7)")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=Path("media/closed_loop/pilot"))
    args = parser.parse_args()

    assert all(0 <= r <= 7 for r in args.replicates), (
        "Pilot must use pilot seeds only (replicate indices 0-7 of parameter "
        "seed 641). Main-experiment seeds must never be used for tuning."
    )
    run(args.replicates, G_FB_LEVELS, args.duration, args.out_dir)
