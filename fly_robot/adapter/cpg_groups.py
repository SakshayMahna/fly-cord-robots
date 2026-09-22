"""Per-leg CPG triad neuron groups — the Rung 2 conductor's injection point.

Proposal only (`docs/trained_adapter/RUNG2_DESIGN.md` Option A). This
module is scaffolding for the pre-registered coupling-bound sweep, not
part of the trainer or the 71-parameter adapter — Rung 2 is not built
until the sweep's bound is committed and approved.

The three CPG-triad roles (`CPG_excit_hub`, `CPG_excit2`, `CPG_inhib`) were
identified from wiring and independently confirmed against Pugliese's own
published dynamics in Phase 1 (`docs/logs/2026-09-19.md`) — not new work,
just read here rather than re-derived.
"""

from __future__ import annotations

import pandas as pd

from fly_robot.interface.motor_neuron_to_joint import LEG_NAME_TO_FLYGYM_PREFIX

CPG_ROLES = ("CPG_excit_hub", "CPG_excit2", "CPG_inhib")


def build_cpg_groups(circuit_csv: str, wtable: pd.DataFrame) -> dict:
    """{(segment, side): [row indices]} — all 3 CPG-triad neurons per leg,
    row indices into the simulation's neuron axis (matching `wtable`'s
    row order, the same convention `build_motor_neuron_groups` uses).
    """
    circuit = pd.read_csv(circuit_csv)
    cpg = circuit[circuit["role"].isin(CPG_ROLES)]
    bodyid_to_idx = {bid: i for i, bid in enumerate(wtable["bodyId"])}

    out = {}
    for (leg, side), group in cpg.groupby(["leg", "side"]):
        idxs = [bodyid_to_idx[b] for b in group["manc_bodyId"] if b in bodyid_to_idx]
        out[(leg, side)] = sorted(idxs)
    return out


def flat_cpg_rows(groups: dict) -> list:
    """All CPG-triad row indices across every leg, deduplicated."""
    return sorted({i for idxs in groups.values() for i in idxs})
