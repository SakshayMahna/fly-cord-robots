"""Identify the Pugliese et al. DNg100 T1 walking circuit inside MaleCNS.

Pugliese's model (github.com/smpuglie/Pugliese_cpg_2025) was built on MANC,
not MaleCNS, and only ever covers the T1 (front leg) DN-to-motor-neuron
circuit. This script reads their MANC-derived T1 connectivity table and
resolves every neuron in it to its MaleCNS body ID via the `mancBodyid`
field that Janelia's MaleCNS annotation ships (a per-neuron cross-reference
to the MANC specimen, not something we compute ourselves).

Output is a mapping table only — no connectome data (wiring or synapse
signs) is modified or re-derived; we are relocating known neurons in a
different specimen's coordinate/ID space.

Usage:
    python -m fly_robot.connectome.identify_t1_circuit \
        --pugliese-table "/path/to/wTable_20250813_DNtoMN_unsorted_withModules.csv" \
        --out data/circuit_map/t1_front_leg_circuit.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from fly_robot.connectome.client import resolve_manc_bodyids_to_malecns

# CPG neurons, identified from Pugliese's `DNg100_Stim_CoreCPG.yaml` config
# (`keepOnly: [31, 277, 617, 1167]`), resolved to MANC bodyIds via row index
# in their T1 DNtoMN table. Kept here as named constants since the row-index
# lookup is otherwise silent and easy to lose track of.
CPG_MANC_BODY_IDS = {
    10093: "DNg100",     # descending neuron, walking command (2 excitatory + 1 inhibitory drive it)
    10707: "IN17A001",   # excitatory (acetylcholine)
    11751: "INXXX466",   # excitatory (acetylcholine)
    13905: "IN16B036",   # inhibitory (glutamate)
}


def build_circuit_map(pugliese_table_path: str) -> pd.DataFrame:
    manc_df = pd.read_csv(pugliese_table_path)

    cpg_rows = manc_df[manc_df["bodyId"].isin(CPG_MANC_BODY_IDS)].copy()
    cpg_rows["role"] = "CPG_or_command"

    mn_rows = manc_df[
        manc_df["class"].astype(str).str.contains("motor", case=False, na=False)
    ].copy()
    mn_rows["role"] = "leg_motor_neuron"

    manc_side = pd.concat([cpg_rows, mn_rows], ignore_index=True)
    manc_ids = manc_side["bodyId"].tolist()

    malecns_side = resolve_manc_bodyids_to_malecns(manc_ids)

    merged = manc_side.merge(
        malecns_side, left_on="bodyId", right_on="manc_bodyId", how="left"
    )

    n_total = len(manc_side)
    n_resolved = merged["malecns_bodyId"].notna().sum()
    print(
        f"Resolved {n_resolved}/{n_total} MANC-side neurons to MaleCNS "
        f"via mancBodyid ({n_total - n_resolved} unmatched — see unmatched rows)."
    )

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pugliese-table",
        required=True,
        help="Path to Pugliese's wTable_*_DNtoMN_unsorted_withModules.csv",
    )
    parser.add_argument(
        "--out",
        default="data/circuit_map/t1_front_leg_circuit.csv",
        help="Output CSV path for the resolved circuit map",
    )
    args = parser.parse_args()

    result = build_circuit_map(args.pugliese_table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")
