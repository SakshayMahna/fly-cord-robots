"""Gates for the pre-registered stability filter on replicate pools.

These exist because of a specific failure. The first Phase 5 pilot ran on a
HAND-WRITTEN pool that carried Phase 4's "exclude replicate 2" forward
while replicate 5 — baseline `n_active` 3,794, well past Pugliese's 1,500
oversaturation criterion — sat in it unnoticed. Three of 22 generations
drew it and collapsed to a median of -1.70 against -0.12 elsewhere,
regardless of candidate quality, and the episode count had been sized
against the FILTERED noise (SD 0.0066) while the trainer sampled the
unfiltered pool (SD 0.176): a 27x mismatch.

So: no pool is ever written down. It is computed from measured baselines,
and asserted at every point of use. The tests below use injected baseline
maps rather than real simulations, so the rule is checkable without a
10-hour run — the point is that the LOGIC cannot regress.

Run:
    python -m pytest tests/test_replicate_filter.py -v
"""

import numpy as np
import pytest

from fly_robot.training.cma_trainer import TrainConfig, Trainer
from fly_robot.training.replicate_filter import (
    STABILITY_MAX_N_ACTIVE, IneligibleReplicate, assert_pool_eligible,
    eligible_replicates,
)

# MEASURED baselines for param seed 641, adapter off, g_fb = 0 — computed by
# replicate_filter.measure_baselines, not assumed. Replicate 2's 4,176
# independently corroborates Phase 4's decision to exclude it.
MEASURED = {0: 522, 1: 380, 2: 4176, 3: 367, 4: 488, 5: 3794, 6: 421, 7: 445}


def test_threshold_is_puglieses_documented_criterion():
    assert STABILITY_MAX_N_ACTIVE == 1500


def test_filter_excludes_the_replicate_that_contaminated_the_first_pilot():
    pool = eligible_replicates(MEASURED)
    assert 5 not in pool, "replicate 5 (n_active 3,794) must be excluded"
    assert 2 not in pool, "replicate 2 was already excluded in Phase 4"
    assert pool == [0, 1, 3, 4, 6, 7]


def test_filter_keeps_everything_that_passes():
    assert eligible_replicates({0: 100, 1: 1500}) == [0, 1]   # 1500 is inclusive
    assert eligible_replicates({0: 1501}) == []


def test_assert_rejects_an_ineligible_pool():
    with pytest.raises(IneligibleReplicate, match="FAIL the pre-registered"):
        assert_pool_eligible([0, 5], MEASURED)


def test_assert_rejects_a_replicate_with_no_measured_baseline():
    """Silently assuming an unmeasured replicate is fine is how the first
    pool went wrong — it was never measured, only inherited."""
    with pytest.raises(IneligibleReplicate, match="no measured baseline"):
        assert_pool_eligible([0, 99], MEASURED)


def test_assert_accepts_a_clean_pool():
    assert_pool_eligible([0, 1, 3, 4, 6, 7], MEASURED)


# --- the pools actually used ----------------------------------------------

def _trainer(pool, baselines, episodes=2, tmp=None):
    cfg = TrainConfig(run_name="t", episodes=episodes, pool=tuple(pool),
                      baselines=baselines, out_dir=str(tmp))
    return Trainer(cfg)


def test_training_draws_only_from_the_eligible_pool(tmp_path):
    """THE gate. Across many generations, no episode may ever land on a
    replicate that fails the filter."""
    pool = eligible_replicates(MEASURED)
    t = _trainer(pool, MEASURED, tmp=tmp_path)
    seen = set()
    for gen in range(500):
        drawn = t.episode_replicates(gen)
        seen.update(drawn)
    assert seen <= set(pool)
    assert 5 not in seen and 2 not in seen


def test_training_refuses_an_ineligible_pool_at_draw_time(tmp_path):
    """Even if a bad pool were somehow configured, the per-generation
    assertion must catch it rather than training on it."""
    t = _trainer([0, 5], MEASURED, tmp=tmp_path)
    with pytest.raises(IneligibleReplicate):
        t.episode_replicates(0)


def test_episode_draw_is_still_deterministic_under_filtering(tmp_path):
    """Filtering must not break the resume guarantee — the draw is still a
    pure function of the generation index."""
    pool = eligible_replicates(MEASURED)
    a = _trainer(pool, MEASURED, tmp=tmp_path)
    b = _trainer(pool, MEASURED, tmp=tmp_path)
    for gen in (0, 1, 17, 249):
        assert a.episode_replicates(gen) == b.episode_replicates(gen)


def test_episodes_cannot_exceed_the_eligible_pool_size(tmp_path):
    """Asking for more episodes than eligible replicates must degrade
    gracefully, not raise or silently sample with replacement."""
    t = _trainer([0, 1], MEASURED, episodes=6, tmp=tmp_path)
    drawn = t.episode_replicates(0)
    assert len(drawn) == 2 and len(set(drawn)) == 2


def test_a_network_failing_everywhere_is_reported_not_worked_around():
    """If a CONTROL network oversaturates on every replicate, that is a
    result about the control. The filter must not be loosened for it."""
    from fly_robot.training.replicate_filter import resolve_pool
    all_bad = {r: 9999 for r in range(8)}
    assert eligible_replicates(all_bad) == []
    with pytest.raises(IneligibleReplicate, match="a finding to report"):
        # resolve_pool measures for real, so exercise the guard directly
        # through the same message path it would raise.
        from fly_robot.training import replicate_filter as rf
        original = rf.measure_baselines
        rf.measure_baselines = lambda *a, **k: all_bad
        try:
            resolve_pool(range(8), param_seed=0, verbose=False)
        finally:
            rf.measure_baselines = original
