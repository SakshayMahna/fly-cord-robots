"""Secondary fallback: highest stable sub-cliff drive for unmatched controls.

Procedure pre-registered in `docs/trained_adapter/PREREGISTRATION.md`
Amendment 2, committed before C2's primary search began and before C1's
verdict was known.

This does NOT replace the primary result. A control reaching this point has
already been reported as "cannot be activity-matched"; this finds a drive
at which it can at least be trained, for a clearly labelled secondary
comparison:

    "not activity-matched; secondary comparison"

That label is mandatory wherever the control appears, because it runs at
lower total activity than the real network and a weaker result from it
could be the activity difference rather than the wiring.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.fallback_drive
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fly_robot.experiments.match_drive import (
    DURATION_S, MAX_ITERS, MIN_BRACKET, MIN_MEDIAN_RHYTHMIC, NETWORKS,
    REPLICATES, baseline, median_activity, rhythmicity,
)
from fly_robot.training.replicate_filter import STABILITY_MAX_N_ACTIVE

LABEL = "not activity-matched; secondary comparison"


def stable_and_rhythmic(drive, **kw):
    """Both pre-registered conditions at this drive, plus the evidence."""
    per_rep = {r: baseline(drive, r, **kw) for r in REPLICATES}
    median_n = float(np.median(list(per_rep.values())))
    stable = median_n <= STABILITY_MAX_N_ACTIVE
    eligible = [r for r, n in per_rep.items() if n <= STABILITY_MAX_N_ACTIVE]
    if not stable or not eligible:
        return False, {"median_n_active": median_n, "eligible": eligible,
                       "stable": stable, "median_n_rhythmic": None}
    rh = rhythmicity(drive, eligible, **kw)
    med_rh = float(np.median(list(rh.values())))
    return (med_rh >= MIN_MEDIAN_RHYTHMIC), {
        "median_n_active": median_n, "eligible": eligible, "stable": stable,
        "n_rhythmic": rh, "median_n_rhythmic": med_rh}


def highest_stable_drive(upper, **kw):
    """Largest drive in [0, upper] meeting both conditions."""
    lo, hi = 0.0, float(upper)
    best = None
    for it in range(1, MAX_ITERS + 1):
        mid = 0.5 * (lo + hi)
        ok, ev = stable_and_rhythmic(mid, **kw)
        print(f"    iter {it:2d}: drive={mid:8.2f}  median n_active="
              f"{ev['median_n_active']:8.1f}  median n_rhythmic="
              f"{ev['median_n_rhythmic']}  passes={ok}", flush=True)
        if ok:
            best = (mid, ev)
            lo = mid            # try higher
        else:
            hi = mid            # too high, come down
        if hi - lo < MIN_BRACKET:
            break
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", default="media/trained_adapter/drive_matching.json")
    ap.add_argument("--out", default="media/trained_adapter/drive_fallback.json")
    args = ap.parse_args()

    primary = json.loads(Path(args.primary).read_text())
    out = {}
    for name, entry in primary["networks"].items():
        if "cannot be activity-matched" not in entry.get("verdict", ""):
            continue
        print(f"=== {name}: {entry['verdict']} ===")
        # Upper bound: the drive the primary search settled on, which sits
        # at the lower edge of the discontinuity it could not cross.
        upper = entry["matched_drive"]
        print(f"  searching [0, {upper:.2f}] for the highest stable, "
              f"rhythmic drive", flush=True)
        best = highest_stable_drive(upper, **NETWORKS[name])
        if best is None:
            out[name] = {"verdict": "no stable rhythmic regime at any drive",
                         "label": None}
            print(f"  VERDICT: no stable rhythmic regime at any drive — "
                  f"do not train\n")
        else:
            drive, ev = best
            out[name] = {"fallback_drive": drive, "label": LABEL,
                         "verdict": f"train at {drive:.2f}, labelled '{LABEL}'",
                         **ev}
            print(f"  VERDICT: train at drive {drive:.2f}, labelled "
                  f"'{LABEL}'\n")

    if not out:
        print("no control needed the fallback")
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
