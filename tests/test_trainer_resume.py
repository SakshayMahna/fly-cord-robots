"""Gates for checkpoint / resume, and for the reward-hash refusal.

Resume must be EXACT, not approximate: reloading and continuing has to
produce the sequence an uninterrupted run would have produced. An
approximate resume is worse than none, because the run looks continuous in
the learning curve while actually being two different searches spliced
together.

These tests deliberately use a cheap synthetic objective instead of real
trials — what is under test is the trainer's bookkeeping, and making it
depend on 8-second physics rollouts would mean it never gets run.

Run:
    python -m pytest tests/test_trainer_resume.py -v
"""

import numpy as np
import pytest

from fly_robot.adapter.parameters import N_PARAMS, default_z
from fly_robot.training.cma_trainer import CANDIDATE_REPLICATES, TrainConfig, Trainer


# Measured baselines, param seed 641, adapter off (see test_replicate_filter).
_BASELINES = {0: 522, 1: 380, 2: 4176, 3: 367, 4: 488, 5: 3794, 6: 421, 7: 445}


def _sphere(z):
    return float(np.sum(np.asarray(z) ** 2))


def _advance(es, n_gens):
    """Run n generations of a deterministic synthetic objective."""
    for _ in range(n_gens):
        pop = es.ask()
        es.tell(pop, [_sphere(z) for z in pop])
    return es


def _fresh_es(seed=20260921, popsize=6):
    import cma
    return cma.CMAEvolutionStrategy(
        default_z(), 0.5, {"popsize": popsize, "seed": seed, "verbose": -9})


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    """THE gate. Checkpoint at generation k, reload, continue — the next
    population must match an uninterrupted run bit-for-bit."""
    import pickle

    uninterrupted = _advance(_fresh_es(), 4)
    expected = np.asarray(uninterrupted.ask())

    interrupted = _advance(_fresh_es(), 2)
    blob = pickle.dumps(interrupted)          # what save_checkpoint stores
    revived = pickle.loads(blob)
    revived = _advance(revived, 2)
    got = np.asarray(revived.ask())

    np.testing.assert_array_equal(
        got, expected,
        "resumed run diverged from an uninterrupted one; the learning curve "
        "would look continuous while being two different searches")


def test_episode_replicates_are_derived_from_the_generation_not_drawn(tmp_path):
    """If episode assignment were sampled as the run went, resuming would
    silently change the noise structure mid-search."""
    cfg = TrainConfig(run_name="t", episodes=2, out_dir=str(tmp_path),
                      pool=(0, 1, 3, 4, 6, 7), baselines=_BASELINES)
    a, b = Trainer(cfg), Trainer(cfg)
    for gen in (0, 1, 7, 250):
        assert a.episode_replicates(gen) == b.episode_replicates(gen)
    # and they genuinely vary across generations
    assert len({tuple(a.episode_replicates(g)) for g in range(20)}) > 1


def test_episode_replicates_stay_inside_the_eligible_pool(tmp_path):
    """The pool is computed from measured baselines, never hard-coded —
    see tests/test_replicate_filter.py for the filter itself. Here we only
    check the trainer respects whatever pool it was given."""
    pool = (0, 1, 3, 4, 6, 7)
    cfg = TrainConfig(run_name="t", episodes=2, out_dir=str(tmp_path),
                      pool=pool, baselines=_BASELINES)
    t = Trainer(cfg)
    seen = {r for g in range(200) for r in t.episode_replicates(g)}
    assert seen <= set(pool)
    assert 2 not in seen and 5 not in seen
    assert set(CANDIDATE_REPLICATES) - seen >= {2, 5}


def test_checkpoint_roundtrips(tmp_path):
    cfg = TrainConfig(run_name="rt", out_dir=str(tmp_path),
                      pool=(0, 1), baselines=_BASELINES)
    t = Trainer(cfg)
    es = _advance(_fresh_es(), 1)
    t.save_checkpoint(es, generation=3, history=[{"generation": 3}],
                      best={"score": 1.0, "z": default_z().tolist()})
    state = t.load_checkpoint()
    assert state["generation"] == 3
    assert state["reward_hash"] == t.reward_hash
    assert state["best"]["score"] == 1.0


def test_resume_refuses_a_different_reward_definition(tmp_path):
    """A reward that changes mid-run mixes two objectives inside one run,
    and it would not be visible in the learning curve."""
    cfg = TrainConfig(run_name="hash", out_dir=str(tmp_path),
                      pool=(0, 1), baselines=_BASELINES)
    t = Trainer(cfg)
    t.save_checkpoint(_fresh_es(), 0, [], {"score": 0.0, "z": None})

    t.reward_hash = "deadbeef" * 8           # pretend the reward changed
    with pytest.raises(RuntimeError, match="DIFFERENT reward definition"):
        t.load_checkpoint()


def test_checkpoint_write_is_atomic(tmp_path):
    """A crash mid-write must not leave a truncated checkpoint that then
    fails to load and loses the whole run."""
    cfg = TrainConfig(run_name="atomic", out_dir=str(tmp_path),
                      pool=(0, 1), baselines=_BASELINES)
    t = Trainer(cfg)
    t.save_checkpoint(_fresh_es(), 0, [], {"score": 0.0, "z": None})
    assert t.checkpoint_path().exists()
    assert not t.checkpoint_path().with_suffix(".tmp").exists(), (
        "temporary checkpoint left behind; the write was not atomic")
