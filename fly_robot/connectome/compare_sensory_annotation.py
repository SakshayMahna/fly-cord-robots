"""Compare leg proprioceptive-sensory annotation between MANC and MaleCNS.

Asked before committing to a sensory interface: is MaleCNS's sensory
annotation more complete, and more left/right symmetric, than the MANC
annotation the simulation currently runs on? If it is, switching would be
worth proposing.

The two datasets do NOT annotate sensory neurons the same way, so this
script discovers each one's scheme rather than assuming a shared field:

  MANC (Pugliese's `wTable_20251006.feather`)
    class    = 'sensory neuron' / 'sensory ascending'
    subclass = 'chordotonal organ' | 'campaniform sensilla' | 'hair plate' | ...
    leg/side = parsed from `instance`, e.g. 'SNpp45_MetaLN_L'
               (sensory somata sit in the leg, so `somaNeuromere` is NaN
               for 5,889 of 5,891 sensory neurons — it cannot be used)

  MaleCNS (neuprint `male-cns:v1.0`)
    class      = 'mechanosensory_proprioceptive'
    subclass   = 'chordotonal organ' | 'campaniform sensilla' | 'hair plate'
                 | 'leg' (a generic proprioceptor label MANC has no
                 equivalent of)
    entryNerve = explicit field — no string parsing needed
    rootSide   = 'L' / 'R'

Nerve-to-leg mapping (same in both): ProLN -> T1 (front),
MesoLN -> T2 (middle), MetaLN -> T3 (hind).

The decisive column is REACHABILITY. The dynamics run on the MANC matrix,
so a MaleCNS neuron is only usable if its `mancBodyid` cross-reference
lands on a row that exists in the simulated network. Identifying a neuron
we cannot drive is worth nothing.

Usage:
    python -m fly_robot.connectome.compare_sensory_annotation
"""

import argparse
from pathlib import Path

import pandas as pd

from fly_robot.connectome.client import get_client

NERVE_TO_LEG = {"ProLN": "T1", "MesoLN": "T2", "MetaLN": "T3"}
LEG_NERVES = list(NERVE_TO_LEG)

# MANC subclasses that are proprioceptive/load-sensing rather than
# tactile-bristle or gustatory.
MANC_PROPRIOCEPTIVE = ["chordotonal organ", "campaniform sensilla", "hair plate"]

DEFAULT_MANC_WTABLE = (
    Path("external/Pugliese_cpg_2025/data/manc full vnc data/wTable_20251006.feather")
)


def manc_leg_proprioceptors(wtable_path: Path = DEFAULT_MANC_WTABLE) -> pd.DataFrame:
    """Leg-nerve proprioceptors as annotated in the MANC table the
    simulation actually uses. Row position in this table IS the neuron's
    index in the simulation, so it is returned as `sim_row`."""
    wt = pd.read_feather(wtable_path).reset_index(drop=True)
    sn = wt[wt["class"].isin(["sensory neuron", "sensory ascending"])].copy()

    def parse(instance):
        if not isinstance(instance, str):
            return None, None
        parts = instance.split("_")
        if len(parts) < 3:
            return None, None
        side = {"L": "L", "R": "R"}.get(parts[-1])
        return NERVE_TO_LEG.get(parts[1]), side

    parsed = sn["instance"].apply(lambda s: pd.Series(parse(s), index=["leg", "side"]))
    sn = pd.concat([sn, parsed], axis=1)
    sn["sim_row"] = sn.index

    return sn[sn["subclass"].isin(MANC_PROPRIOCEPTIVE)
              & sn["leg"].notna() & sn["side"].notna()].copy()


def malecns_leg_proprioceptors() -> pd.DataFrame:
    """Leg-nerve proprioceptors as annotated in MaleCNS, with each one's
    MANC cross-reference (where Janelia computed one)."""
    client = get_client()
    return client.fetch_custom(f"""
    MATCH (n:Neuron)
    WHERE n.class = 'mechanosensory_proprioceptive'
      AND n.entryNerve IN {LEG_NERVES}
    RETURN n.bodyId AS malecns_bodyId, n.subclass AS subclass,
           n.entryNerve AS entryNerve, n.rootSide AS side,
           n.mancBodyid AS manc_bodyId, n.instance AS instance
    """).assign(leg=lambda d: d["entryNerve"].map(NERVE_TO_LEG))


def asymmetry(counts: pd.Series) -> float:
    """|L - R| / (L + R). 0 = perfectly symmetric, 1 = one side only."""
    left = counts.get("L", 0)
    right = counts.get("R", 0)
    total = left + right
    return abs(left - right) / total if total else float("nan")


def main(wtable_path: Path = DEFAULT_MANC_WTABLE) -> None:
    manc = manc_leg_proprioceptors(wtable_path)
    mcns = malecns_leg_proprioceptors()

    wt = pd.read_feather(wtable_path).reset_index(drop=True)
    simulated_ids = set(wt["bodyId"])
    mcns["in_sim"] = mcns["manc_bodyId"].map(
        lambda b: bool(pd.notna(b) and b in simulated_ids))

    print("=" * 72)
    print("LEG PROPRIOCEPTIVE SENSORY ANNOTATION: MANC vs MaleCNS")
    print("=" * 72)

    print(f"\nMANC    (what the simulation runs on): {len(manc)} leg proprioceptors")
    print(f"MaleCNS (neuprint male-cns:v1.0)     : {len(mcns)} leg proprioceptors")
    print(f"  of which cross-referenced to MANC  : {mcns['manc_bodyId'].notna().sum()} "
          f"({mcns['manc_bodyId'].notna().mean() * 100:.1f}%)")
    print(f"  of which REACHABLE in the sim      : {mcns['in_sim'].sum()} "
          f"({mcns['in_sim'].mean() * 100:.1f}%), "
          f"{mcns.loc[mcns['in_sim'], 'manc_bodyId'].nunique()} unique rows")

    print("\n--- MANC: leg x side x subclass ---")
    print(pd.crosstab([manc["leg"], manc["side"]], manc["subclass"]).to_string())

    print("\n--- MaleCNS: leg x side x subclass (all, native annotation) ---")
    print(pd.crosstab([mcns["leg"], mcns["side"]], mcns["subclass"]).to_string())

    print("\n--- MaleCNS: leg x side x subclass (only those REACHABLE in the sim) ---")
    reach = mcns[mcns["in_sim"]]
    print(pd.crosstab([reach["leg"], reach["side"]], reach["subclass"]).to_string())

    print("\n--- Totals and left/right asymmetry per leg ---")
    rows = []
    for leg in ("T1", "T2", "T3"):
        m = manc[manc["leg"] == leg]["side"].value_counts()
        c = mcns[mcns["leg"] == leg]["side"].value_counts()
        r = reach[reach["leg"] == leg]["side"].value_counts()
        rows.append(dict(
            leg=leg,
            manc_L=m.get("L", 0), manc_R=m.get("R", 0), manc_asym=round(asymmetry(m), 3),
            mcns_L=c.get("L", 0), mcns_R=c.get("R", 0), mcns_asym=round(asymmetry(c), 3),
            reach_L=r.get("L", 0), reach_R=r.get("R", 0), reach_asym=round(asymmetry(r), 3),
        ))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Load sensing (campaniform sensilla) in leg nerves ---")
    print(f"  MANC   : {(manc['subclass'] == 'campaniform sensilla').sum()}")
    print(f"  MaleCNS: {(mcns['subclass'] == 'campaniform sensilla').sum()}")
    print("  For scale, campaniform sensilla outside the leg nerves "
          "(wing/haltere fields) number in the hundreds in both datasets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wtable", type=Path, default=DEFAULT_MANC_WTABLE)
    args = parser.parse_args()
    main(args.wtable)
