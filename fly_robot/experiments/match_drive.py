"""Activity-match DNg100 drive across networks. Procedure pre-registered.

See `docs/trained_adapter/PREREGISTRATION.md`, committed before this was
run. Nothing here is tuned per condition except the drive itself, which is
the quantity being matched.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.match_drive
"""

import argparse
import json

import numpy as np

from fly_robot.analysis.interleg_coordination import coordination
from fly_robot.sim.closed_loop import NEURAL_DT
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
from fly_robot.training.replicate_filter import STABILITY_MAX_N_ACTIVE

REPLICATES = tuple(range(8))
DURATION_S = 4.0
NATIVE_DRIVE = 380.0
DRIVE_LO, DRIVE_HI = 0.0, 2000.0
TOLERANCE = 0.05          # fractional
MIN_BRACKET = 1.0         # drive units
MAX_ITERS = 30
MIN_MEDIAN_RHYTHMIC = 2   # of 6 legs

NETWORKS = {
    "real": {},
    "C1": {"shuffle_seed": 20260921},
    "C2": {"random_seed": 20260921},
}


def baseline(drive, replicate, **kw):
    model, _mg, _sg, _wt, _i = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
        stim_current=drive, **kw)
    model.reset()
    for _ in range(int(DURATION_S / 0.001)):
        model.step(0.001)
    return int((model.rates > 0.01).sum())


def median_activity(drive, **kw):
    return float(np.median([baseline(drive, r, **kw) for r in REPLICATES]))


def match_drive(target, **kw):
    """Bisection on drive. Returns (drive, achieved_median, converged, iters)."""
    lo, hi = DRIVE_LO, DRIVE_HI
    best = (None, None, np.inf)
    for it in range(1, MAX_ITERS + 1):
        mid = 0.5 * (lo + hi)
        med = median_activity(mid, **kw)
        err = abs(med - target) / target
        if err < best[2]:
            best = (mid, med, err)
        print(f"    iter {it:2d}: drive={mid:8.2f}  median n_active={med:9.1f}  "
              f"err={err:6.1%}", flush=True)
        if err <= TOLERANCE:
            return mid, med, True, it
        if hi - lo < MIN_BRACKET:
            return best[0], best[1], best[2] <= TOLERANCE, it
        if med > target:
            hi = mid
        else:
            lo = mid
    return best[0], best[1], best[2] <= TOLERANCE, MAX_ITERS


def rhythmicity(drive, replicates, **kw):
    """AR(1)-gated n_rhythmic per replicate, at the matched drive."""
    from fly_robot.sim.closed_loop import run_trial
    out = {}
    for r in replicates:
        model, mg, sg, _wt, _i = build_trial_components(
            replicate=r, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
            stim_current=drive, **kw)
        res = run_trial(model, mg, sensory_groups=sg, feedback_gain=0.0,
                        rig="harness", duration_s=DURATION_S, seed=r,
                        encoder_mode="deviation")
        rhythm, _c = coordination(res.motor_rates, NEURAL_DT)
        out[r] = int(rhythm.rhythmic.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="media/trained_adapter/drive_matching.json")
    args = ap.parse_args()

    print("=== target: real network at its native drive ===")
    native = {r: baseline(NATIVE_DRIVE, r) for r in REPLICATES}
    target = float(np.median(list(native.values())))
    print(f"  per-replicate n_active: {native}")
    print(f"  TARGET (median) = {target}\n")

    results = {"target": target, "native_real": native, "networks": {}}
    for name, kw in NETWORKS.items():
        print(f"=== {name} ===")
        drive, med, converged, iters = match_drive(target, **kw)
        entry = {"matched_drive": drive, "achieved_median": med,
                 "converged": bool(converged), "iterations": iters,
                 "kwargs": {k: v for k, v in kw.items()}}
        print(f"  matched drive = {drive:.2f}  median n_active = {med:.1f}  "
              f"converged = {converged}")

        if not converged:
            entry["verdict"] = "FAILS reachability — cannot be activity-matched"
            results["networks"][name] = entry
            print(f"  VERDICT: {entry['verdict']}\n")
            continue

        per_rep = {r: baseline(drive, r, **kw) for r in REPLICATES}
        eligible = [r for r, n in per_rep.items() if n <= STABILITY_MAX_N_ACTIVE]
        entry["per_replicate_n_active"] = per_rep
        entry["eligible"] = eligible
        print(f"  per-replicate n_active at matched drive: {per_rep}")
        print(f"  eligible after the unchanged stability filter: "
              f"{len(eligible)}/8 -> {eligible}")

        if not eligible:
            entry["verdict"] = "FAILS stability — no eligible replicate at matched drive"
            results["networks"][name] = entry
            print(f"  VERDICT: {entry['verdict']}\n")
            continue

        rh = rhythmicity(drive, eligible, **kw)
        med_rh = float(np.median(list(rh.values())))
        entry["n_rhythmic"] = rh
        entry["median_n_rhythmic"] = med_rh
        print(f"  n_rhythmic per eligible replicate: {rh}")
        print(f"  median n_rhythmic = {med_rh} (need >= {MIN_MEDIAN_RHYTHMIC})")

        if med_rh < MIN_MEDIAN_RHYTHMIC:
            entry["verdict"] = "FAILS rhythmicity — no significant rhythm at matched drive"
        else:
            entry["verdict"] = "PASSES — train from this matched drive"
        results["networks"][name] = entry
        print(f"  VERDICT: {entry['verdict']}\n")

    print("=== SUMMARY ===")
    for name, e in results["networks"].items():
        print(f"  {name:<5} drive={e['matched_drive']:8.2f}  {e['verdict']}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
