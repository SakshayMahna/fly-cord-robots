"""Does restoring electrical coupling frequency-lock the six leg CPGs?

The decisive experiment for the gap-junction hypothesis
(`neural/gap_junctions.py` documents why this mechanism and not another).

Measured facts this is responding to:
  * the six legs run at DIFFERENT frequencies (median spread 3.03 Hz across
    published replicates), so they are independent oscillators, not
    mis-phased ones;
  * that spread survives removing all per-neuron parameter heterogeneity
    (1.5-2.8 Hz with every stdv set to ~0), so it is structural;
  * phase coupling is impossible without frequency locking first.

Coupled-oscillator theory predicts locking once coupling exceeds the spread
of natural frequencies. This sweeps the conductance and measures, per
replicate:

  * **frequency spread** across the six legs (the thing that must collapse)
  * **n_active** (must not run away into the saturation regime)
  * **tripod index** (does the locked state resemble a fly's gait?)
  * **n_rhythmic** (coupling must not simply flatten the rhythm)

`conductance = 0` is included as the ablation and must reproduce the
uncoupled model exactly.

Usage:
    FLY_ROBOT_REPO=$PWD MUJOCO_GL=cgl PYTHONPATH=$PWD \
        .venv/bin/python -m fly_robot.experiments.gap_junction_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_robot.analysis.interleg_coordination import coordination
from fly_robot.interface.motor_neuron_to_joint import leg_motor_row_indices
from fly_robot.neural.gap_junctions import coupling_report, homologous_cpg_coupling
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

CIRCUIT_CSV = "data/circuit_map/all_legs_circuit.csv"
DT = 0.001


def fine_freq(x: np.ndarray, dt: float = DT, pad: int = 16) -> float:
    """Spectral peak in the 2-20 Hz walking band, zero-padded so the
    estimate is not limited to the 0.5 Hz bins of a 2 s window — the
    resolution error that originally hid this whole problem."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0):
        return np.nan
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n), n=n * pad))
    f = np.fft.rfftfreq(n * pad, dt)
    band = (f >= 2.0) & (f <= 20.0)
    return float(f[band][np.argmax(X[band])])


def run(replicate: int, conductance: float, duration_s: float, drive: float):
    model, mg, _sg, wtable, _info = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED,
        duration_s=duration_s, stim_current=drive)
    report = None
    if conductance > 0:
        G = homologous_cpg_coupling(CIRCUIT_CSV, wtable, conductance)
        model.gap_junctions = G
        model._gj_leak = np.asarray(G.sum(axis=1)).ravel()
        report = coupling_report(G)

    model.reset()
    R = np.stack([model.step(DT) for _ in range(int(duration_s / DT))], axis=1)

    rows = leg_motor_row_indices(mg)
    legs = sorted(rows)
    S = np.stack([R[rows[l], :].sum(axis=0) for l in legs])
    freqs = np.array([fine_freq(S[i]) for i in range(6)])
    rhythm, coord = coordination(S, DT)
    return {
        "replicate": replicate, "conductance": conductance,
        "freqs": [None if np.isnan(f) else round(float(f), 3) for f in freqs],
        "freq_spread": (float(np.nanmax(freqs) - np.nanmin(freqs))
                        if np.isfinite(freqs).any() else None),
        "n_active": int((R[:, -1] > 0.01).sum()),
        "peak_n_active": int((R > 0.01).sum(axis=0).max()),
        "n_rhythmic": int(rhythm.rhythmic.sum()),
        "tripod_index": (float(coord.tripod_index)
                         if coord.has_rhythm and np.isfinite(coord.tripod_index)
                         else None),
        "report": report,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conductances", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 4])
    ap.add_argument("--duration-s", type=float, default=3.0)
    ap.add_argument("--drive", type=float, default=380.0)
    ap.add_argument("--out", default="media/gap_junctions/sweep.json")
    args = ap.parse_args()

    rows = []
    print(f"{'g':>7}{'rep':>5}{'spread':>9}{'n_rhy':>7}{'tripod':>9}"
          f"{'n_act':>7}{'peak':>7}   per-leg Hz")
    for g in args.conductances:
        for rep in args.replicates:
            r = run(rep, g, args.duration_s, args.drive)
            rows.append(r)
            sp = "  n/a" if r["freq_spread"] is None else f"{r['freq_spread']:6.2f}"
            tp = "   n/a" if r["tripod_index"] is None else f"{r['tripod_index']:+6.3f}"
            print(f"{g:>7.2f}{rep:>5}{sp:>9}{r['n_rhythmic']:>7}{tp:>9}"
                  f"{r['n_active']:>7}{r['peak_n_active']:>7}   {r['freqs']}",
                  flush=True)

    print("\n=== summary by conductance ===")
    print(f"{'g':>7}{'median spread':>15}{'median n_rhythmic':>19}"
          f"{'median tripod':>15}{'max peak_n_active':>19}")
    summary = {}
    for g in args.conductances:
        sel = [r for r in rows if r["conductance"] == g]
        spreads = [r["freq_spread"] for r in sel if r["freq_spread"] is not None]
        trip = [r["tripod_index"] for r in sel if r["tripod_index"] is not None]
        s = {"median_spread": float(np.median(spreads)) if spreads else None,
             "median_n_rhythmic": float(np.median([r["n_rhythmic"] for r in sel])),
             "median_tripod": float(np.median(trip)) if trip else None,
             "max_peak_n_active": int(max(r["peak_n_active"] for r in sel))}
        summary[str(g)] = s
        ms = "n/a" if s["median_spread"] is None else f"{s['median_spread']:.2f}"
        mt = "n/a" if s["median_tripod"] is None else f"{s['median_tripod']:+.3f}"
        print(f"{g:>7.2f}{ms:>15}{s['median_n_rhythmic']:>19}"
              f"{mt:>15}{s['max_peak_n_active']:>19}")

    base = summary[str(args.conductances[0])]["median_spread"]
    best = min((v for v in summary.values() if v["median_spread"] is not None),
               key=lambda v: v["median_spread"], default=None)
    if base and best:
        print(f"\nuncoupled spread {base:.2f} Hz -> best coupled "
              f"{best['median_spread']:.2f} Hz")
        if best["median_spread"] < 0.5 and best["median_n_rhythmic"] >= 4:
            print("LOCKED: electrical coupling collapses the frequency spread "
                  "while the rhythm survives.")
        elif best["median_spread"] < base * 0.5:
            print("PARTIAL: spread substantially reduced but not locked.")
        else:
            print("NO LOCKING at any conductance tested.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"drive": args.drive, "duration_s": args.duration_s,
                   "rows": rows, "summary": summary}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
