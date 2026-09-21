"""Which replicates a network is allowed to be trained or scored on.

The pre-registered stability filter (`docs/closed_loop/PREREGISTRATION.md`
§3) excludes replicates whose baseline activity meets Pugliese's own
oversaturation criterion, `n_active > 1500`. Phase 4 applied it to
replicate 2 by hand, because that was the only oversaturated draw among
the four it used.

**That hand-applied list is what went wrong.** The first Phase 5 pilot
inherited "exclude 2" as a hard-coded tuple and ran on replicates
(0,1,3,4,5,6,7) — but replicate 5 has a baseline `n_active` of **3,794**,
so it should have been excluded by the same rule. Three of 22 generations
drew it and collapsed to a median of -1.70 against -0.12 elsewhere,
regardless of candidate quality. Worse, the episode count had been sized
against the stability-FILTERED noise (SD 0.0066) while the trainer sampled
the unfiltered pool (SD 0.176) — a **27x** mismatch between the analysis
and the thing it was meant to size.

So the pool is no longer written down anywhere. It is COMPUTED, per
network, from that network's own adapter-off baseline, and
`assert_pool_eligible` is called wherever a pool is used.

Controls get the same treatment against their OWN baselines. If a shuffled
or random network fails the filter on many replicates, that is a result to
report, not a reason to loosen the threshold for it.

Measuring the baseline needs no physics: at zero feedback there is no path
from body to neurons, so the neural trajectory cannot depend on the body.
Verified directly — a full ball trial and a bare neural run give the same
`n_active` to the unit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Pugliese's own documented oversaturation criterion, pre-registered.
STABILITY_MAX_N_ACTIVE = 1500

CACHE_DIR = Path("data/replicate_eligibility")


class IneligibleReplicate(RuntimeError):
    """A pool contained a replicate that fails the stability filter."""


def _cache_path(param_seed: int, shuffle_seed: int | None, duration_s: float) -> Path:
    tag = f"seed{param_seed}_shuffle{shuffle_seed}_T{duration_s:g}"
    return CACHE_DIR / f"{tag}.json"


def baseline_n_active(replicate: int, param_seed: int,
                      shuffle_seed: int | None = None,
                      duration_s: float = 4.0,
                      random_seed: int | None = None) -> int:
    """`n_active` for this replicate with the adapter off and no feedback.

    Neural-only: at zero feedback the body cannot influence the network, so
    this is exactly what a full trial would report, at a fraction of the
    cost.
    """
    from fly_robot.sim.trial_setup import build_trial_components

    model, _mg, _sg, _wt, _info = build_trial_components(
        replicate=replicate, param_seed=param_seed, duration_s=duration_s,
        shuffle_seed=shuffle_seed, random_seed=random_seed)
    model.reset()
    for _ in range(int(round(duration_s / 0.001))):
        model.step(0.001)
    return int((model.rates > 0.01).sum())


def measure_baselines(candidates, param_seed: int, shuffle_seed: int | None = None,
                      duration_s: float = 4.0, use_cache: bool = True,
                      verbose: bool = True, random_seed: int | None = None) -> dict:
    """{replicate: baseline n_active} for `candidates`, cached on disk.

    The cache is keyed by (param_seed, shuffle_seed, duration) so a control
    network never reads the real network's baselines.
    """
    path = _cache_path(param_seed, shuffle_seed if shuffle_seed is not None
                       else (f"rand{random_seed}" if random_seed is not None else None),
                       duration_s)
    cached = {}
    if use_cache and path.exists():
        cached = {int(k): int(v) for k, v in json.loads(path.read_text()).items()}

    out = {}
    for rep in candidates:
        if rep in cached:
            out[rep] = cached[rep]
            continue
        n = baseline_n_active(rep, param_seed, shuffle_seed, duration_s, random_seed)
        out[rep] = n
        cached[rep] = n
        if verbose:
            flag = "  EXCLUDED" if n > STABILITY_MAX_N_ACTIVE else ""
            print(f"  replicate {rep}: baseline n_active = {n}{flag}", flush=True)

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in sorted(cached.items())},
                                   indent=1))
    return out


def eligible_replicates(baselines: dict,
                        threshold: int = STABILITY_MAX_N_ACTIVE) -> list:
    """Pure function: the replicates that pass the filter, sorted.

    Split out from the measurement so the rule itself is testable without
    running any simulation.
    """
    return sorted(r for r, n in baselines.items() if n <= threshold)


def assert_pool_eligible(pool, baselines: dict,
                         threshold: int = STABILITY_MAX_N_ACTIVE,
                         where: str = "pool") -> None:
    """Raise if `pool` contains a replicate that fails the filter.

    Called wherever a pool is used — training, evaluation, inspection — so
    the failure that produced the contaminated first pilot cannot recur
    silently in any of them.
    """
    unknown = [r for r in pool if r not in baselines]
    if unknown:
        raise IneligibleReplicate(
            f"{where} contains replicate(s) {unknown} with no measured baseline; "
            "eligibility cannot be asserted. Measure the baseline rather than "
            "assuming the replicate is fine.")
    bad = {r: baselines[r] for r in pool if baselines[r] > threshold}
    if bad:
        raise IneligibleReplicate(
            f"{where} contains replicate(s) that FAIL the pre-registered "
            f"stability filter (n_active > {threshold}): "
            + ", ".join(f"replicate {r} (n_active={n})" for r, n in sorted(bad.items()))
            + ". This is the defect that contaminated the first Phase 5 pilot; "
              "exclude them rather than loosening the threshold.")


def resolve_pool(candidates, param_seed: int, shuffle_seed: int | None = None,
                 duration_s: float = 4.0, verbose: bool = True,
                 random_seed: int | None = None) -> tuple:
    """Measure, filter, and assert in one call. Returns (pool, baselines).

    Refuses to return an empty pool: a network on which every replicate
    oversaturates is a RESULT about that network — most plausibly a control
    — and must be reported, not worked around.
    """
    baselines = measure_baselines(candidates, param_seed, shuffle_seed,
                                  duration_s, verbose=verbose,
                                  random_seed=random_seed)
    pool = eligible_replicates(baselines)
    excluded = {r: n for r, n in baselines.items() if n > STABILITY_MAX_N_ACTIVE}
    if verbose and excluded:
        print(f"  filter excluded {len(excluded)} of {len(baselines)}: "
              + ", ".join(f"rep {r} (n_active={n})" for r, n in sorted(excluded.items())),
              flush=True)
    if not pool:
        raise IneligibleReplicate(
            f"EVERY candidate replicate fails the stability filter for this "
            f"network (baselines: {baselines}). For a control network this is "
            "a finding to report — the network oversaturates without any "
            "manipulation — not something to fix by relaxing the threshold.")
    assert_pool_eligible(pool, baselines, where="resolved pool")
    return pool, baselines
