"""Validate the causal phase estimator against the offline Hilbert phase.

Required before the Rung 2 conductor uses `CausalPhaseEstimator` for
anything (your instruction). Recorded on the UNTRAINED connectome, at the
real network's pre-registered activity-matched drive (382.8125), on the
eligible pilot replicates — no adapter changes, no Rung 2 code path
exercised, just the signal the estimator would have to work with.

Both estimators run on the same signal — per-leg summed motor rate
(`TrialResult.motor_rates`), the established, already-validated readout
`analysis/interleg_coordination.py` uses for every rhythm/tripod
measurement in this project — so the comparison is like-for-like, not
apples to oranges.

Reports, per leg and pooled: circular correlation between the two phase
series, and the lag (in samples and ms) that maximises it — the causal
estimator's window necessarily introduces some delay, and this measures
exactly how much.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.validate_causal_phase
"""

import argparse
import json

import numpy as np

from fly_robot.adapter.causal_phase import STRIDE_STEPS, WINDOW_S, CausalPhaseEstimator
from fly_robot.adapter.parameters import default_params
from fly_robot.analysis.interleg_coordination import LEGS, TRANSIENT_S, leg_rhythm
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

MATCHED_DRIVE = 382.8125     # PREREGISTRATION.md / RESULTS.md
REPLICATES = (0, 1, 3, 4, 6, 7)   # the real network's eligible pool
DURATION_S = 4.0


def circular_correlation(a, b):
    """Standard circular-circular correlation (Fisher & Lee 1983), for
    comparing two phase series without the wraparound artefacts a linear
    correlation would introduce at +-pi."""
    a, b = np.asarray(a), np.asarray(b)
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    num = np.sum((ca - ca.mean()) * (cb - cb.mean())) * np.sum((sa - sa.mean()) * (sb - sb.mean()))
    num -= np.sum((ca - ca.mean()) * (sb - sb.mean())) * np.sum((sa - sa.mean()) * (cb - cb.mean()))
    den1 = np.sum((ca - ca.mean()) ** 2) * np.sum((sa - sa.mean()) ** 2) \
        - np.sum((ca - ca.mean()) * (sa - sa.mean())) ** 2
    den2 = np.sum((cb - cb.mean()) ** 2) * np.sum((sb - sb.mean()) ** 2) \
        - np.sum((cb - cb.mean()) * (sb - sb.mean())) ** 2
    if den1 <= 0 or den2 <= 0:
        return float("nan")
    return float(num / np.sqrt(den1 * den2))


def best_lag(a, b, max_lag):
    """Lag (in samples, can be negative) of `b` relative to `a` that
    maximises circular correlation, searched over [-max_lag, max_lag]."""
    best_l, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[:len(b) - lag] if lag else b
        else:
            aa, bb = a[:len(a) + lag], b[-lag:]
        n = min(len(aa), len(bb))
        if n < 50:
            continue
        r = circular_correlation(aa[:n], bb[:n])
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, lag
    return best_l, best_r


def run_one(replicate):
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
        stim_current=MATCHED_DRIVE)
    res = run_trial(model, mg, sensory_groups=sg, rig="ground", duration_s=DURATION_S,
                    seed=replicate, adapter=default_params())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="media/trained_adapter/causal_phase_validation.json")
    args = ap.parse_args()

    keep = int(TRANSIENT_S / NEURAL_DT)
    stride_s = STRIDE_STEPS * NEURAL_DT
    # A bandpass filter's own group delay is typically comparable to its
    # window length, not to the oscillation period -- the first attempt
    # searched +/-60 ms (half a cycle at 11 Hz) and pinned at that boundary
    # on most legs, meaning the true optimum was outside the search range.
    # Search out to the full window length either way instead of guessing
    # a tighter bound a second time.
    max_lag_samples = int(round(WINDOW_S / stride_s))

    print(f"window={WINDOW_S*1000:.0f}ms  stride={STRIDE_STEPS} steps "
          f"({stride_s*1000:.0f}ms)  max lag searched=+/-{max_lag_samples} strides "
          f"(+/-{max_lag_samples*stride_s*1000:.0f}ms)\n")

    per_leg_corr = {leg: [] for leg in LEGS}
    per_leg_lag_ms = {leg: [] for leg in LEGS}
    rows = []

    for rep in REPLICATES:
        res = run_one(rep)
        n = res.terminated_at_step or res.n_steps_planned
        if n <= keep + 200:   # not enough post-transient signal to analyse
            print(f"replicate {rep}: terminated at step {n} "
                  f"(< transient + 200 steps) -- skipped")
            continue
        motor_rates = res.motor_rates[:, :n]

        # Offline: whole-trace bandpass + Hilbert (the established method).
        rhythm = leg_rhythm(motor_rates, NEURAL_DT)
        offline_phase = rhythm.phase   # (6, n-transient)

        # Causal: step through the SAME trial's motor_rates, one step at a
        # time, exactly as it would run inside a real trial.
        est = CausalPhaseEstimator(dt=NEURAL_DT)
        causal_phase = np.zeros((6, n))
        for t in range(n):
            causal_phase[:, t] = est.update(motor_rates[:, t])
        causal_phase = causal_phase[:, keep:]   # align to the offline window

        print(f"replicate {rep}: n_rhythmic={int(rhythm.rhythmic.sum())}/6")
        for i, leg in enumerate(LEGS):
            if not rhythm.rhythmic[i]:
                print(f"   {leg}: not rhythmic, skipped")
                continue
            n_common = min(offline_phase.shape[1], causal_phase.shape[1])
            a, b = offline_phase[i, :n_common], causal_phase[i, :n_common]
            lag, corr = best_lag(a, b, max_lag_samples)
            lag_ms = lag * stride_s * 1000
            per_leg_corr[leg].append(corr)
            per_leg_lag_ms[leg].append(lag_ms)
            print(f"   {leg}: circular corr={corr:+.3f} at lag={lag_ms:+.0f} ms")
            rows.append({"replicate": rep, "leg": f"{leg[0]}-{leg[1]}",
                        "corr": corr, "lag_ms": lag_ms})

    print("\n=== per-leg summary (pooled across replicates) ===")
    summary = {}
    for leg in LEGS:
        c, l = per_leg_corr[leg], per_leg_lag_ms[leg]
        leg_str = f"{leg[0]}-{leg[1]}"
        if not c:
            print(f"  {leg_str}: no rhythmic instances observed")
            continue
        summary[leg_str] = {"mean_corr": float(np.mean(c)), "min_corr": float(np.min(c)),
                            "mean_lag_ms": float(np.mean(l)), "n": len(c)}
        print(f"  {leg_str}: mean corr={np.mean(c):+.3f} (min {np.min(c):+.3f}, n={len(c)})  "
              f"mean lag={np.mean(l):+.1f} ms")

    all_corr = [c for v in per_leg_corr.values() for c in v]
    verdict = bool(all_corr) and float(np.mean(all_corr)) > 0.8 and float(np.min(all_corr)) > 0.5
    print(f"\noverall: mean corr={np.mean(all_corr):+.3f}  min corr={np.min(all_corr):+.3f}  "
          f"n={len(all_corr)}")
    print(f"VERDICT (mean>0.8 and min>0.5): {'GOOD ENOUGH TO USE' if verdict else 'NOT YET GOOD ENOUGH'}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"window_s": WINDOW_S, "stride_steps": STRIDE_STEPS,
                   "per_leg_summary": summary, "rows": rows,
                   "overall_mean_corr": float(np.mean(all_corr)) if all_corr else None,
                   "overall_min_corr": float(np.min(all_corr)) if all_corr else None,
                   "verdict": verdict}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
