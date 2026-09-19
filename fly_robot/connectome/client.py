"""neuprint client setup for the MaleCNS connectome (v1.0).

Dataset choice: MaleCNS, not MANC. MaleCNS is a distinct, newer specimen
(brain + VNC imaged as one continuous volume, incl. the neck connective)
released 2026; MANC (2023) is nerve-cord-only and is what the Pugliese et
al. CPG model was built on. We use MaleCNS going forward for our own
neuron identification and circuit extraction; MANC is only relevant if we
later re-run Pugliese's exact pipeline for validation.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client

NEUPRINT_SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

_client = None


def get_client() -> Client:
    """Return a cached, authenticated neuprint Client for MaleCNS v1.0."""
    global _client
    if _client is None:
        load_dotenv()
        token = os.environ.get("NEUPRINT_TOKEN")
        if not token:
            raise RuntimeError(
                "NEUPRINT_TOKEN not set. Add it to a .env file in the repo root "
                "(see README) — get a token from https://neuprint.janelia.org "
                "under your account profile."
            )
        _client = Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    return _client


def resolve_manc_bodyids_to_malecns(manc_bodyids: list[int]) -> pd.DataFrame:
    """Look up MaleCNS neurons by their `mancBodyid` cross-reference field
    (computed by Janelia's own annotation team — not something we derive).

    Shared by `identify_t1_circuit.py` (T1-only) and `identify_all_leg_circuits.py`
    (all six legs), which both need the same MANC-bodyId -> MaleCNS lookup;
    previously each had its own near-identical copy of this query.
    """
    client = get_client()
    query = f"""
    MATCH (n:Neuron)
    WHERE n.mancBodyid IN {list(int(b) for b in manc_bodyids)}
    RETURN n.bodyId AS malecns_bodyId, n.type AS type, n.instance AS instance,
           n.mancBodyid AS manc_bodyId, n.mancType AS mancType,
           n.status AS status, n.somaNeuromere AS somaNeuromere,
           n.predictedNt AS predictedNt, n.class AS malecns_class,
           n.superclass AS malecns_superclass
    """
    return client.fetch_custom(query)


def collapse_malecns_matches(merged: pd.DataFrame, key_cols: list[str],
                             expected_class_substring) -> pd.DataFrame:
    """Reduce a MANC->MaleCNS join to ONE MaleCNS row per key.

    `mancBodyid` is **not** a 1:1 cross-reference: some MANC neurons match
    several MaleCNS neurons, and some of those matches are plainly wrong —
    e.g. MANC leg motor neurons matching MaleCNS sensory neurons
    (`SNta02`, `SNpp45`, `SNta29`) or interneurons (`IN13A030`,
    `INXXX471`). A plain `merge` therefore silently multiplies rows, which
    double-counts the same MANC neuron downstream. (This was a real bug:
    it inflated `all_legs_circuit.csv` from 330 motor neurons to 346 rows
    and double-weighted 16 of them in every per-leg readout.)

    The MANC bodyId is the canonical identity here — it is what indexes
    the simulated network — so the MaleCNS columns are annotation, and a
    duplicate must be resolved rather than propagated. Resolution order:

    1. Prefer a match whose MaleCNS `superclass` contains the substring
       `expected_class_substring(row)` returns for that row (e.g. "motor"
       for a motor neuron), since a cross-reference landing on a wildly
       different cell class is almost certainly spurious. `superclass` is
       the field that carries this (`vnc_motor` / `vnc_intrinsic` /
       `vnc_sensory`); `class` is NaN for every MaleCNS motor neuron and
       is useless for the purpose.
    2. Break any remaining tie by lowest `malecns_bodyId` — arbitrary but
       deterministic, so reruns are reproducible.

    Two columns are added so the ambiguity is visible rather than hidden:
    `malecns_n_matches` (how many MaleCNS neurons matched) and
    `malecns_ambiguous` (True where step 1 did not pick a unique winner).
    """
    out = []
    for key, group in merged.groupby(key_cols, dropna=False, sort=False):
        n_matches = int(group["malecns_bodyId"].notna().sum())
        if len(group) == 1:
            row = group.iloc[0].copy()
            row["malecns_n_matches"] = n_matches
            row["malecns_ambiguous"] = False
            out.append(row)
            continue

        wanted = expected_class_substring(group.iloc[0])
        cls = group["malecns_superclass"].astype(str).str.lower()
        preferred = group[cls.str.contains(str(wanted).lower(), na=False)] if wanted else group
        resolved_by_class = len(preferred) == 1
        candidates = preferred if len(preferred) >= 1 else group

        row = candidates.sort_values("malecns_bodyId").iloc[0].copy()
        row["malecns_n_matches"] = n_matches
        row["malecns_ambiguous"] = not resolved_by_class
        out.append(row)

    return pd.DataFrame(out).reset_index(drop=True)
