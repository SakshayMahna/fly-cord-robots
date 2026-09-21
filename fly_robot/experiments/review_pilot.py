"""Generation-120 review, with the early-stop rule committed BEFORE the data.

The rule below was agreed at generation ~58 and written down as executable
code, not prose, so that applying it later is a computation rather than a
judgement call made while looking at the answer. That is the same
discipline `docs/closed_loop/PREREGISTRATION.md` §B2 used for the Phase 4
stopping rule, and the reason Phase 4's result is reportable at all.

THE RULE
--------
Stop the pilot if BOTH hold at generation 120:

  A. `best` has not improved beyond noise:
        runmax(MA10(best))[120]  <=  runmax(MA10(best))[60]
     where MA10 is a 10-generation trailing mean and runmax is the running
     maximum of it. Using a smoothed running max rather than raw `best`
     matters because raw `best` is the maximum over 16 candidates of a
     mean of only 2 episodes -- it is upward-biased by construction and
     drifts up on noise alone.

  B. median gains are still mostly reduced saturation, not better gait:
        delta(mean weighted saturation term) / delta(median score)  >  0.5
     comparing the last 10 generations against the first 10.

Otherwise continue to 250.

KNOWN LIMITATION of criterion B, stated now rather than discovered later:
the trainer logs per-term values as MEANS over the whole population, while
`median` is the median of candidate scores. Comparing them is an
approximation. The clean comparison -- median score with and without the
saturation term -- needs per-CANDIDATE term logging, which this run does
not have. That gap is recorded in the log; B is therefore a guide with a
stated error bar, not a precise accounting.

Usage:
    PYTHONPATH=$PWD .venv/bin/python -m fly_robot.experiments.review_pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np

REVIEW_GENERATION = 120
BASELINE_GENERATION = 60
MA_WINDOW = 10
SATURATION_SHARE_THRESHOLD = 0.5


def moving_average(x, window):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


def running_max_ma(best, window=MA_WINDOW):
    """Running max of the trailing moving average, indexed by generation.

    Returns an array aligned so that entry g is the running max of MA10
    over all generations <= g (NaN before the window fills).
    """
    ma = moving_average(best, window)
    out = np.full(len(best), np.nan)
    if len(ma) == 0:
        return out
    out[window - 1:] = np.maximum.accumulate(ma)
    return out


def evaluate_rule(history) -> dict:
    gens = np.array([r["generation"] for r in history])
    best = np.array([r["best"] for r in history])
    median = np.array([r["median"] for r in history])
    sat_term = np.array([r.get("term_saturation", np.nan) for r in history])

    rm = running_max_ma(best)

    def at(g):
        idx = np.where(gens <= g)[0]
        return float(rm[idx[-1]]) if len(idx) and np.isfinite(rm[idx[-1]]) else float("nan")

    rm_baseline, rm_review = at(BASELINE_GENERATION), at(REVIEW_GENERATION)

    first, last = history[:MA_WINDOW], history[-MA_WINDOW:]
    d_median = float(np.mean([r["median"] for r in last])
                     - np.mean([r["median"] for r in first]))
    d_sat = float(np.mean([r["term_saturation"] for r in last])
                  - np.mean([r["term_saturation"] for r in first]))
    share = d_sat / d_median if d_median > 0 else float("nan")

    crit_a = bool(np.isfinite(rm_review) and np.isfinite(rm_baseline)
                  and rm_review <= rm_baseline)
    crit_b = bool(np.isfinite(share) and share > SATURATION_SHARE_THRESHOLD)

    return {
        "n_generations": len(history),
        "runmax_ma10_at_60": rm_baseline,
        "runmax_ma10_at_120": rm_review,
        "criterion_A_no_improvement": crit_a,
        "delta_median": d_median,
        "delta_saturation_term": d_sat,
        "saturation_share_of_median_gain": share,
        "criterion_B_gains_are_saturation": crit_b,
        "STOP": bool(crit_a and crit_b),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="pilot")
    ap.add_argument("--out-dir", default="media/trained_adapter")
    args = ap.parse_args()

    path = Path(args.out_dir) / args.run_name / "history.json"
    history = json.loads(path.read_text())
    gens = np.array([r["generation"] for r in history])
    best = np.array([r["best"] for r in history])
    median = np.array([r["median"] for r in history])
    sat = np.array([r["frac_saturated"] for r in history])
    sigma = np.array([r["sigma"] for r in history])

    print(f"=== pilot review: {len(history)} generations ===\n")
    print(f"{'metric':<12}{'first 10':>11}{'last 10':>11}{'slope/gen':>12}{'r':>8}")
    for y, name in ((best, "best"), (median, "median"),
                    (sat, "saturated"), (sigma, "sigma")):
        sl = np.polyfit(gens, y, 1)[0]
        r = np.corrcoef(gens, y)[0, 1]
        print(f"{name:<12}{y[:10].mean():>+11.3f}{y[-10:].mean():>+11.3f}"
              f"{sl:>+12.5f}{r:>+8.3f}")

    keys = [k for k in history[0] if k.startswith("term_")]
    print(f"\nper-term population MEAN (first 10 -> last 10 generations):")
    print(f"{'term':<16}{'first 10':>11}{'last 10':>11}{'delta':>11}")
    for k in keys:
        a = np.mean([r[k] for r in history[:MA_WINDOW]])
        b = np.mean([r[k] for r in history[-MA_WINDOW:]])
        print(f"{k.replace('term_',''):<16}{a:>+11.4f}{b:>+11.4f}{b-a:>+11.4f}")

    verdict = evaluate_rule(history)
    print("\n=== early-stop rule (committed before the data) ===")
    print(f"  A. runmax(MA10 best): gen60 {verdict['runmax_ma10_at_60']:+.4f} -> "
          f"gen120 {verdict['runmax_ma10_at_120']:+.4f}   "
          f"-> no improvement: {verdict['criterion_A_no_improvement']}")
    print(f"  B. delta median {verdict['delta_median']:+.4f}, of which saturation "
          f"{verdict['delta_saturation_term']:+.4f} "
          f"({verdict['saturation_share_of_median_gain']:.0%})   "
          f"-> mostly saturation: {verdict['criterion_B_gains_are_saturation']}")
    print(f"\n  VERDICT: {'STOP the pilot' if verdict['STOP'] else 'CONTINUE to 250'}")

    out = Path(args.out_dir) / args.run_name / "review_gen120.json"
    out.write_text(json.dumps(verdict, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
