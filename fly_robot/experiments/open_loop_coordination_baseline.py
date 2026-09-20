"""E0 — the open-loop interleg-coordination baseline.

Measures how coordinated the six legs already are with NO sensory
feedback, using Pugliese et al.'s own published 128-replicate full-VNC
output rather than new simulation. This is the number every closed-loop
condition is compared against, and it is also a direct check on our
pipeline: if it disagrees with the paper's own reported findings, something
differs from their setup and nothing downstream can be trusted.

Their two tested claims (v2 full text) are that left/right phase coupling
is absent and that the tripod pattern does not emerge. Both are checked
here explicitly, alongside the same-side (ipsilateral) pairs the paper does
not report either way.

Usage:
    python -m fly_robot.experiments.open_loop_coordination_baseline
    python -m fly_robot.experiments.open_loop_coordination_baseline --out-json media/closed_loop/e0_baseline.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fly_robot.analysis.interleg_coordination import (
    BAND_HZ, LEG_LABELS, LEGS, coordination, leg_motor_readout, pair_indices,
)
from fly_robot.interface.motor_neuron_to_joint import (
    build_motor_neuron_groups, leg_motor_row_indices,
)
from fly_robot.neural.replicate_ensemble import (
    N_ACTIVE_UPPER, load_published_full_vnc_replicates,
)

# Measured, not assumed: the false-positive rate of the surrogate test on
# phase-randomised data with no true phase relationship (see AUDIT.md §6).
CALIBRATED_CHANCE = 0.076


def build_motor_indices(circuit_csv: str, wtable: pd.DataFrame) -> dict:
    """{(segment, side): [row indices into R]} for each leg's motor pool.

    Thin wrapper around the shared `leg_motor_row_indices` (previously an
    independent, near-duplicate implementation lived here — reading the
    CSV directly and asserting no dupes — see `docs/logs/` reorg entries
    for the pattern of centralizing these). Equivalent in practice: every
    `leg_motor_neuron` row in the circuit map carries a `motor_module`
    label (verified — 0 of 330 are NaN), which is the only case where
    `build_motor_neuron_groups`'s `dropna` could have excluded a row this
    used to include.
    """
    wt_indexed = wtable.reset_index(drop=True)
    groups = build_motor_neuron_groups(circuit_csv, wt_indexed)
    return leg_motor_row_indices(groups)


def pair_class(i: int, j: int) -> str:
    """How a leg pair relates anatomically — the axis along which the
    paper's claims are framed."""
    if LEGS[i][0] == LEGS[j][0]:
        return "contralateral"          # left vs right of the same segment
    if LEGS[i][1] == LEGS[j][1]:
        return "ipsilateral"            # same side, front/middle/hind
    return "diagonal"


def run(circuit_csv: str, n_surrogates: int = 100) -> dict:
    wtable, R_stack, dt = load_published_full_vnc_replicates()
    motor_indices = build_motor_indices(circuit_csv, wtable)
    print(f"Motor neurons per leg: "
          f"{ {f'{a}-{b}': len(v) for (a, b), v in motor_indices.items()} }")

    n_active = np.array([(R_stack[i].max(axis=1) > 0.01).sum() for i in range(len(R_stack))])
    stable = np.where(n_active <= N_ACTIVE_UPPER)[0]
    print(f"Stable replicates: {len(stable)}/{len(R_stack)} "
          f"(n_active <= {N_ACTIVE_UPPER}, Pugliese's own criterion)")

    plv, sig, valid, tripod, dom, active = [], [], [], [], [], []
    for r in stable:
        rhythm, coord = coordination(
            leg_motor_readout(R_stack[r], motor_indices), dt,
            n_surrogates=n_surrogates, seed=int(r),
        )
        plv.append(coord.plv); sig.append(coord.significant); valid.append(coord.valid)
        tripod.append(coord.tripod_index)
        dom.append(rhythm.dominant_hz); active.append(rhythm.active)

    plv = np.array(plv); sig = np.array(sig); valid = np.array(valid)
    dom = np.array(dom); active = np.array(active); tripod = np.array(tripod)

    classes = np.array([pair_class(i, j) for i, j in pair_indices()])

    print(f"\nPer-leg rhythm (band {BAND_HZ[0]}-{BAND_HZ[1]} Hz):")
    print(pd.DataFrame({
        "leg": LEG_LABELS,
        "frac_active": active.mean(0).round(3),
        "median_dom_Hz": [round(float(np.median(dom[active[:, j], j])), 2)
                          if active[:, j].any() else np.nan for j in range(6)],
    }).to_string(index=False))

    results = {"n_replicates_total": int(len(R_stack)),
               "n_replicates_stable": int(len(stable)),
               "calibrated_chance": CALIBRATED_CHANCE,
               "tripod_index_median": float(np.nanmedian(tripod)),
               "tripod_index_frac_positive": float((tripod > 0).mean()),
               "by_class": {}}

    print(f"\nInterleg coupling (chance = {CALIBRATED_CHANCE * 100:.1f}%):")
    rows = []
    for cls in ("ipsilateral", "contralateral", "diagonal"):
        m = classes == cls
        v, s, p = valid[:, m], sig[:, m], plv[:, m]
        entry = {"n_pairs": int(m.sum()), "n_valid_trials": int(v.sum()),
                 "frac_significant": float(s.sum() / max(v.sum(), 1)),
                 "median_plv": float(np.median(p[v])) if v.any() else float("nan")}
        results["by_class"][cls] = entry
        rows.append({"pair class": cls, "pairs": entry["n_pairs"],
                     "valid trials": entry["n_valid_trials"],
                     "significant": f"{entry['frac_significant'] * 100:.1f}%",
                     "median PLV": round(entry["median_plv"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTripod index: median {results['tripod_index_median']:+.3f}, "
          f"{results['tripod_index_frac_positive'] * 100:.1f}% of replicates positive")

    print("\nPer-pair detail:")
    for p_i, (i, j) in enumerate(pair_indices()):
        if valid[:, p_i].sum() == 0:
            continue
        print(f"  {LEG_LABELS[i]:7s}-{LEG_LABELS[j]:7s} [{pair_class(i, j)[:5]}] "
              f"n={valid[:, p_i].sum():3d}  sig={sig[:, p_i].sum() / valid[:, p_i].sum() * 100:5.1f}%  "
              f"medPLV={np.median(plv[valid[:, p_i], p_i]):.3f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit-csv", default="data/circuit_map/all_legs_circuit.csv")
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--out-json", default="media/closed_loop/e0_baseline.json")
    args = parser.parse_args()

    results = run(args.circuit_csv, args.n_surrogates)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
