"""Identify T1/T2/T3 leg circuits (DNg100 + 3-neuron CPG + leg motor
neurons) and resolve them into MaleCNS.

Status (2026-09-19, see CHANGELOG): what this script identifies from
static connectivity alone was originally reported as a hypothesis. It has
since been CONFIRMED — the exact T2/T3 neurons this script picks out were
checked directly against Pugliese et al.'s own real published full-VNC
simulation output (Zenodo record 22260924) and shown to be genuinely
rhythmically active (104-124/128 replicates, scores 0.37-0.71, same range
as the confirmed T1 circuit). The method/reasoning below (written when
this was still a hypothesis) is left as-is since it's still the correct
description of HOW these candidates were identified — only the
confidence level at the end has changed.

Background: Pugliese et al. published and validated this 3-neuron CPG
circuit (1 inhibitory, 2 excitatory) for T1 (front leg), driven by
DNg100, and (per their v2 preprint, which we initially missed — see
CHANGELOG 2026-09-19) also directly confirmed via full-VNC simulation
that a functioning version exists in all six legs. This script identifies
*which specific neurons* comprise the T2/T3 versions from connectivity
alone — useful because the paper's text doesn't name them, only shows
they exist.

Method (all from MANC — Pugliese's own full-VNC files, not modified):
1. Each of the 3 CPG cell types (IN17A001, INXXX466, IN16B036) has
   exactly 6 instances in MANC: one per side x one per thoracic segment.
   This by itself is just a naming/typing fact from Janelia's annotation,
   not evidence of shared function — same `type` string is only a
   morphological/lineage claim.
2. To test function, we look at real synaptic weights (MANC's own
   `W_20260522_allSynapses.npz` matrix, row/column order given by
   `wTable_20260522_allSynapses.feather`): does DNg100 drive each
   segment's triad, and does each triad show the same mutual
   excitation/inhibition topology as the published T1 circuit? Side
   pairing (which DNg100 copy drives which side's triad) is inferred
   from the weights themselves, not assumed.
3. This connectivity match is what let us identify the specific
   candidates that were later confirmed dynamically (see Status above) —
   at the time this script was written, it was reported as a
   *data-derived hypothesis* only.

Every leg motor neuron already annotated with a `motor module` label in
Pugliese's full-VNC table (T1/T2/T3, 142/95/93 neurons) is included too,
independent of the CPG confirmation — those are a much more
solid, directly-read fact from the data (class == motor neuron + a
joint-module label), not an inference.

Usage:
    python -m fly_robot.connectome.identify_all_leg_circuits \
        --pugliese-repo /path/to/Pugliese_cpg_2025 \
        --out data/circuit_map/all_legs_circuit.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fly_robot.connectome.client import (
    collapse_malecns_matches, resolve_manc_bodyids_to_malecns,
)

# The 3 CPG cell types from Pugliese's T1 circuit (see identify_t1_circuit.py
# for how these were decoded from their experiment config row-indices).
CPG_TYPES = {
    "IN17A001": "excit_hub",   # strong DNg100 target, drives excit2 + inhib
    "INXXX466": "excit2",
    "IN16B036": "inhib",
}
DN_TYPE = "DNg100"


def _load_manc_full_vnc(pugliese_repo: Path):
    data_dir = pugliese_repo / "data" / "manc full vnc data"
    wtable = pd.read_feather(data_dir / "wTable_20260522_allSynapses.feather").reset_index(drop=True)
    weights = np.load(data_dir / "W_20260522_allSynapses.npz", allow_pickle=True)["arr_0"]
    return wtable, weights


def _pair_dn_to_side(wtable, weights, dn_bodyids, hub_rows):
    """For each hub (excit_hub) neuron, find which DNg100 copy drives it
    (by comparing actual synaptic weight, not assuming L drives L)."""
    idx_of = {bid: i for i, bid in enumerate(wtable["bodyId"])}
    pairing = {}
    for _, hub in hub_rows.iterrows():
        hub_idx = idx_of[hub["bodyId"]]
        best_dn, best_w = None, 0.0
        for dn_bid in dn_bodyids:
            w = weights[idx_of[dn_bid], hub_idx]
            if w > best_w:
                best_dn, best_w = dn_bid, w
        pairing[hub["bodyId"]] = (best_dn, best_w)
    return pairing


def find_cpg_candidates(wtable, weights) -> pd.DataFrame:
    idx_of = {bid: i for i, bid in enumerate(wtable["bodyId"])}
    dn_rows = wtable[wtable["type"] == DN_TYPE]
    dn_bodyids = dn_rows["bodyId"].tolist()

    hub_rows = wtable[wtable["type"] == "IN17A001"]
    pairing = _pair_dn_to_side(wtable, weights, dn_bodyids, hub_rows)

    records = []
    for _, hub in hub_rows.iterrows():
        seg = hub["somaNeuromere"]
        side = hub["somaSide"]
        dn_bid, dn_w = pairing[hub["bodyId"]]
        if dn_bid is None or dn_w < 10:
            continue  # no meaningfully-driven hub found on this side/segment

        # Same-segment, same-side excit2 / inhib partners.
        same = wtable[(wtable["somaNeuromere"] == seg) & (wtable["somaSide"] == side)]
        excit2 = same[same["type"] == "INXXX466"]
        inhib = same[same["type"] == "IN16B036"]
        if len(excit2) != 1 or len(inhib) != 1:
            continue  # ambiguous or missing partner on this side — skip rather than guess
        excit2_bid = excit2.iloc[0]["bodyId"]
        inhib_bid = inhib.iloc[0]["bodyId"]

        w = lambda a, b: weights[idx_of[a], idx_of[b]]
        records.append({
            "leg": seg, "side": side,
            "dn_bodyId": dn_bid,
            "excit_hub_bodyId": hub["bodyId"],
            "excit2_bodyId": excit2_bid,
            "inhib_bodyId": inhib_bid,
            "w_dn_to_hub": dn_w,
            "w_hub_to_excit2": w(hub["bodyId"], excit2_bid),
            "w_hub_to_inhib": w(hub["bodyId"], inhib_bid),
            "w_excit2_to_hub": w(excit2_bid, hub["bodyId"]),
            "w_excit2_to_inhib": w(excit2_bid, inhib_bid),
            "w_inhib_to_hub": w(inhib_bid, hub["bodyId"]),
            "w_inhib_to_excit2": w(inhib_bid, excit2_bid),
        })
    return pd.DataFrame(records)


def collect_motor_neurons(wtable) -> pd.DataFrame:
    mn = wtable[wtable["class"].astype(str).str.contains("motor", case=False, na=False)].copy()
    mn["has_module"] = mn["motor module"].astype(str).str.strip().replace("nan", "").ne("")
    leg_mn = mn[mn["has_module"] & mn["somaNeuromere"].isin(["T1", "T2", "T3"])].copy()
    return leg_mn[["bodyId", "type", "somaNeuromere", "somaSide", "motor module", "predictedNt"]]


def resolve_to_malecns(manc_bodyids: list[int]) -> pd.DataFrame:
    """Thin wrapper: this script only needs a subset of the shared
    lookup's columns, renamed for this CSV's existing schema."""
    result = resolve_manc_bodyids_to_malecns(manc_bodyids)
    return result.rename(columns={"somaNeuromere": "malecns_somaNeuromere"})[
        ["malecns_bodyId", "type", "instance", "manc_bodyId", "status",
         "malecns_somaNeuromere", "malecns_class", "malecns_superclass"]
    ]


def _expected_malecns_superclass(row) -> str | None:
    """Which MaleCNS `superclass` a given circuit role should land on,
    used to resolve one-to-many `mancBodyid` matches (see
    `client.collapse_malecns_matches`).

    A MANC leg motor neuron should match a MaleCNS neuron with
    `superclass` containing "motor" (`vnc_motor`). The CPG interneurons
    and the DN have no single positive substring covering both
    `vnc_intrinsic` and `descending_neuron`, so they are left to the
    deterministic tie-break rather than given a rule that would be
    half-right.

    Note `class` cannot be used for this: it is NaN for every MaleCNS
    motor neuron. Only `superclass` carries the distinction."""
    return "motor" if row["role"] == "leg_motor_neuron" else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pugliese-repo", required=True, type=Path)
    parser.add_argument("--out", default="data/circuit_map/all_legs_circuit.csv")
    args = parser.parse_args()

    wtable, weights = _load_manc_full_vnc(args.pugliese_repo)

    cpg_candidates = find_cpg_candidates(wtable, weights)
    print("CPG candidate triads found (leg x side):")
    print(cpg_candidates[["leg", "side", "w_dn_to_hub", "w_hub_to_excit2",
                           "w_inhib_to_hub", "w_inhib_to_excit2"]].to_string(index=False))

    cpg_bodyids = pd.unique(cpg_candidates[
        ["dn_bodyId", "excit_hub_bodyId", "excit2_bodyId", "inhib_bodyId"]
    ].values.ravel())

    leg_mn = collect_motor_neurons(wtable)
    print(f"\nLeg motor neurons with joint-module annotation: {len(leg_mn)} "
          f"({leg_mn['somaNeuromere'].value_counts().to_dict()})")

    all_manc_ids = list(cpg_bodyids) + leg_mn["bodyId"].tolist()
    malecns = resolve_to_malecns(all_manc_ids)
    print(f"\nResolved {malecns['malecns_bodyId'].notna().sum()}/{len(all_manc_ids)} "
          f"total neurons to MaleCNS.")

    # Assemble final long-form table.
    rows = []
    for _, r in cpg_candidates.iterrows():
        for role, bid_col in [("command_DN", "dn_bodyId"), ("CPG_excit_hub", "excit_hub_bodyId"),
                               ("CPG_excit2", "excit2_bodyId"), ("CPG_inhib", "inhib_bodyId")]:
            rows.append({"leg": r["leg"], "side": r["side"], "role": role,
                         "manc_bodyId": r[bid_col], "motor_module": None})
    for _, r in leg_mn.iterrows():
        rows.append({"leg": r["somaNeuromere"], "side": r["somaSide"],
                     "role": "leg_motor_neuron", "manc_bodyId": r["bodyId"],
                     "motor_module": r["motor module"]})

    long_df = pd.DataFrame(rows).drop_duplicates(subset=["manc_bodyId", "role"])
    merged = long_df.merge(malecns, on="manc_bodyId", how="left")

    # `mancBodyid` is not 1:1 — a plain merge fans one MANC neuron out into
    # several rows, which double-counts it in every downstream per-leg
    # readout. Collapse back to one row per (MANC neuron, role).
    n_before = len(merged)
    merged = collapse_malecns_matches(
        merged, key_cols=["manc_bodyId", "role"],
        expected_class_substring=_expected_malecns_superclass,
    )
    n_ambiguous = int(merged["malecns_ambiguous"].sum())
    print(f"\nCollapsed {n_before} joined rows -> {len(merged)} unique (MANC neuron, role) "
          f"pairs; {int((merged['malecns_n_matches'] > 1).sum())} had multiple MaleCNS "
          f"matches, of which {n_ambiguous} could not be resolved by cell class "
          f"(tie-broken deterministically, flagged as malecns_ambiguous).")

    assert not merged.duplicated(["manc_bodyId", "role"]).any(), \
        "Duplicate (manc_bodyId, role) survived collapsing — do not proceed."

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nWrote {len(merged)} rows to {out_path}")
