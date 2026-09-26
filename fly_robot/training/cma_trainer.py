"""CMA-ES over a documented subset of the 71-parameter adapter.

Design in `docs/trained_adapter/DESIGN.md` §5b; reward in `REWARD.md`.

Three properties this file exists to guarantee, beyond running a search:

  * **The connectome is not touched.** `W_eff` is hashed at worker startup
    and re-hashed after every trial; a mismatch aborts the run rather than
    warning. No gradient is ever taken through it.
  * **The sign gate blocks startup.** `reward_ground.sign_test()` runs before the
    first generation. With the progress sign inverted the search optimises
    BACKWARDS walking while the score reads as success throughout, so this
    is a hard gate, not a log line.
  * **Resume is exact.** A checkpoint restores the CMA-ES state, the RNG,
    and the episode replicate assignment, so a resumed run continues the
    sequence an uninterrupted one would have produced. Episode replicates
    are derived from the generation index rather than drawn as the run
    goes, precisely so that resuming cannot quietly change the noise
    structure mid-search.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fly_robot.adapter import reward_ground as reward_mod
from fly_robot.adapter.parameters import (
    N_PARAMS, TRAINING_STAGES, default_z, from_z, parameter_indices_for_stage,
    parameter_names_for_groups,
)
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED
from fly_robot.training.replicate_filter import assert_pool_eligible, resolve_pool

# Candidate replicates. Which of these are actually USED is not decided
# here — `replicate_filter.resolve_pool` measures each one's adapter-off
# baseline for THIS network and drops any that fail the pre-registered
# stability filter. A hard-coded pool is exactly what contaminated the
# first pilot: it carried Phase 4's "exclude replicate 2" forward while
# replicate 5 (baseline n_active 3,794) sat in the pool unnoticed.
CANDIDATE_REPLICATES = tuple(range(8))
TRIAL_S = 4.0


@dataclass
class TrainConfig:
    run_name: str = "pilot"
    population: int = 16          # lean row
    episodes: int = 2             # lean row — PILOT ONLY, see DESIGN.md §5.4
    generations: int = 250
    sigma0: float = 0.5
    seed: int = 20260921
    param_seed: int = PILOT_PARAM_SEED
    trial_s: float = TRIAL_S
    shuffle_seed: int | None = None   # C1 control: degree-preserving shuffle
    random_seed_net: int | None = None  # C2 control: matched random network
    condition: str = "real"           # "real" | "C1" | "C2"
    # R1a is intentionally output-only. R1b (`stage="sensory"`) takes an
    # R1a best vector as `initial_z`, then unlocks the sensory block only.
    # Electrical coupling between leg CPGs -- OUR ADDITION, not connectome.
    # 0.0 is the published model exactly. See GAP_JUNCTIONS.md: without it
    # the six leg CPGs run at different frequencies and cannot phase-couple.
    gap_conductance: float = 0.0
    stage: str = "output"          # output | sensory | full
    rig: str = "ground"             # ground is the walking task; never ball
    adapter_sensory: bool = False
    initial_z: list[float] = field(default_factory=list)
    workers: int = 6                # measured local throughput saturates near 6
    out_dir: str = "media/trained_adapter"
    # Eligible replicates, COMPUTED from this network's own adapter-off
    # baselines by replicate_filter.resolve_pool -- never written by hand.
    pool: tuple = ()
    baselines: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- worker ---------------------------------------------------------------

_W = {}


def _init_worker(lock, cfg_dict):
    """Build the connectome once per worker, serialised behind a lock.

    Serialisation is not tidiness: building materialises the dense weight
    matrix transiently (~7.4 GB peak, DESIGN.md §5.6), so concurrent builds
    on many workers will OOM. After building, JAX's caches hold ~2.7 GB that
    nothing needs; dropping them takes a worker to ~0.55 GB.
    """
    import hashlib

    import jax

    from fly_robot.sim.trial_setup import build_trial_components

    cfg = TrainConfig(**cfg_dict)
    first = cfg.pool[0]
    with lock:
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=first, param_seed=cfg.param_seed,
            duration_s=cfg.trial_s, shuffle_seed=cfg.shuffle_seed,
            random_seed=cfg.random_seed_net)
        for arr in jax.live_arrays():
            try:
                arr.delete()
            except Exception:
                pass
        jax.clear_caches()

    _W["cfg"] = cfg
    _W["lock"] = lock
    _W["components"] = {first: (model, mg, sg)}
    _W["order"] = [first]
    _W["w_hash"] = hashlib.sha256(model.neurons.w_eff.data.tobytes()).hexdigest()


def _components_for(replicate: int):
    """Per-replicate components, built on demand and cached in the worker.

    Two constraints, both learned the hard way (DESIGN.md §5.6):

      * **Builds must hold the lock.** Building materialises the dense
        weight matrix transiently at ~7.4 GB. Six workers building at once
        would want ~44 GB on a 19 GB machine. An earlier version of this
        function built unguarded and drove the machine into swap within two
        minutes of launch.
      * **The cache must be bounded.** A generation touches only
        `cfg.episodes` replicates, but over a run it would otherwise
        accumulate all seven, at ~0.55 GB each, in every worker.
    """
    import jax

    from fly_robot.sim.trial_setup import build_trial_components

    if replicate in _W["components"]:
        return _W["components"][replicate]

    cfg = _W["cfg"]
    with _W["lock"]:
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=replicate, param_seed=cfg.param_seed,
            duration_s=cfg.trial_s, shuffle_seed=cfg.shuffle_seed,
            random_seed=cfg.random_seed_net,
            gap_conductance=cfg.gap_conductance)
        for arr in jax.live_arrays():
            try:
                arr.delete()
            except Exception:
                pass
        jax.clear_caches()

    _W["components"][replicate] = (model, mg, sg)
    _W["order"].append(replicate)
    while len(_W["order"]) > max(2, cfg.episodes):
        _W["components"].pop(_W["order"].pop(0), None)
    return _W["components"][replicate]


def _evaluate_one(task):
    """Score one (candidate, episode) pair. Runs in a worker process."""
    import hashlib

    from fly_robot.sim.closed_loop import run_trial

    z, replicate, trial_seed, cand_index = task
    cfg = _W["cfg"]
    model, mg, sg = _components_for(replicate)
    params = from_z(np.asarray(z))

    result = run_trial(model, mg, sensory_groups=sg, rig=cfg.rig,
                       duration_s=cfg.trial_s, seed=trial_seed, adapter=params,
                       adapter_sensory=cfg.adapter_sensory)

    # No-bypass / no-modification gate, every single trial.
    w_hash = hashlib.sha256(model.neurons.w_eff.data.tobytes()).hexdigest()
    if w_hash != _W["w_hash"]:
        raise RuntimeError(
            "the connectome changed during a trial — W_eff hash mismatch. "
            "Training is invalid; aborting rather than continuing.")

    breakdown = reward_mod.evaluate(result)
    row = breakdown.as_row()
    row.update(candidate=cand_index, replicate=replicate, seed=trial_seed,
               wall_s=round(result.wall_clock_s, 2),
               unstable=bool(result.unstable),
               peak_n_active=int(result.peak_n_active_neurons
                                 if result.peak_n_active_neurons is not None
                                 else result.n_active_neurons),
               fraction_saturated_steps=float(result.fraction_saturated_steps),
               terminated=bool(result.terminated_at_step is not None))
    if result.adhesion is not None:
        row["adhesion_duty"] = result.adhesion.mean(axis=0).tolist()
    if result.anatomical_pools is not None:
        for module, values in result.anatomical_pools.items():
            row[f"anatomical_{module}"] = values.mean(axis=0).tolist()
    return row


# --- trainer --------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: TrainConfig):
        if cfg.stage not in TRAINING_STAGES:
            raise ValueError(f"stage must be one of {sorted(TRAINING_STAGES)}, "
                             f"got {cfg.stage!r}")
        if cfg.rig != "ground":
            raise ValueError(
                "the adapter trainer is a free-ground walking experiment; "
                "use a separately labelled experiment for the legacy ball rig")
        expected_sensory = cfg.stage != "output"
        if cfg.adapter_sensory != expected_sensory:
            raise ValueError(
                f"stage={cfg.stage!r} requires adapter_sensory={expected_sensory}; "
                "do not create an ambiguous sensory condition")
        self.cfg = cfg
        self.out = Path(cfg.out_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.reward_hash = reward_mod.reward_config_hash()
        self.active_indices = parameter_indices_for_stage(cfg.stage)
        self.active_names = parameter_names_for_groups(TRAINING_STAGES[cfg.stage])
        self.initial_z = (default_z() if not cfg.initial_z
                          else np.asarray(cfg.initial_z, dtype=float).ravel())
        if self.initial_z.shape != (N_PARAMS,):
            raise ValueError(
                f"initial_z must have {N_PARAMS} entries, got {self.initial_z.shape}")
        self.contract = {
            "rig": cfg.rig,
            "stage": cfg.stage,
            "adapter_sensory": cfg.adapter_sensory,
            "active_parameter_names": self.active_names,
            "active_parameter_count": int(len(self.active_indices)),
            "initial_z_sha256": hashlib.sha256(self.initial_z.tobytes()).hexdigest(),
            "gap_conductance": float(cfg.gap_conductance),
        }

    def expand_z(self, active_z: np.ndarray) -> np.ndarray:
        """Put a stage's CMA coordinates back into the full adapter vector."""
        active_z = np.asarray(active_z, dtype=float).ravel()
        if active_z.shape != (len(self.active_indices),):
            raise ValueError(f"expected {len(self.active_indices)} active parameters, "
                             f"got {active_z.shape}")
        full = self.initial_z.copy()
        full[self.active_indices] = active_z
        return full

    # -- episode assignment, derived not drawn -----------------------------
    def episode_replicates(self, generation: int) -> list[int]:
        """Which neuron-parameter draws this generation's episodes use.

        A deterministic function of the generation index, so a resumed run
        scores candidates on exactly the draws the original would have. If
        these were sampled as the run went, resuming would silently change
        the noise structure mid-search.
        """
        rng = np.random.default_rng((self.cfg.seed, generation))
        pool = list(self.cfg.pool)
        size = min(self.cfg.episodes, len(pool))
        drawn = [int(r) for r in rng.choice(pool, size=size, replace=False)]
        # Belt and braces: the pool was filtered at construction, but this
        # is the exact point where the contaminated pilot went wrong, so it
        # is asserted every generation rather than trusted.
        assert_pool_eligible(drawn, self.cfg.baselines,
                             where=f"generation {generation} episode draw")
        return drawn

    # -- checkpointing -----------------------------------------------------
    def checkpoint_path(self) -> Path:
        return self.out / "checkpoint.pkl"

    def save_checkpoint(self, es, generation: int, history: list, best: dict):
        tmp = self.checkpoint_path().with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({
                "es": es, "generation": generation, "history": history,
                "best": best, "config": self.cfg.as_dict(),
                "reward_hash": self.reward_hash,
                "reward_config": reward_mod.REWARD_CONFIG,
                "trainer_contract": self.contract,
            }, f)
        os.replace(tmp, self.checkpoint_path())   # atomic

    def load_checkpoint(self):
        path = self.checkpoint_path()
        if not path.exists():
            return None
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["reward_hash"] != self.reward_hash:
            raise RuntimeError(
                "checkpoint was written under a DIFFERENT reward definition\n"
                f"  checkpoint: {state['reward_hash'][:16]}...\n"
                f"  current:    {self.reward_hash[:16]}...\n"
                "Resuming would mix two objectives inside one run, and it "
                "would not be visible in the learning curve. Refusing.")
        if state.get("trainer_contract") != self.contract:
            raise RuntimeError(
                "checkpoint was written for a DIFFERENT training contract "
                "(rig, stage, active parameter mask, or warm start).\n"
                "Start a new run name; do not resume one objective under another.")
        return state

    def _write_generation_records(self, generation: int, rows: list[dict],
                                  candidate_records: list[dict]) -> None:
        """Persist candidate/episode evidence without duplicate rows on resume."""
        directory = self.out / "episodes"
        directory.mkdir(exist_ok=True)
        path = directory / f"generation_{generation:04d}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"generation": generation, "stage": self.cfg.stage,
                       "rig": self.cfg.rig, "episodes": rows,
                       "candidates": candidate_records}, f, indent=1)
        os.replace(tmp, path)

    # -- the loop ----------------------------------------------------------
    def run(self, resume: bool = True):
        import cma

        print(f"=== {self.cfg.condition} / {self.cfg.run_name} ===", flush=True)
        print(f"reward config hash: {self.reward_hash}", flush=True)
        print(f"ground stage: {self.cfg.stage} ({len(self.active_indices)} active "
              f"parameters; sensory={'on' if self.cfg.adapter_sensory else 'off'})",
              flush=True)

        # BLOCKING GATE — never a warning.
        print("running sign test (blocking gate)...", flush=True)
        reward_mod.sign_test()
        print("  sign test PASSED\n", flush=True)

        # Eligible replicates, measured from THIS network's own adapter-off
        # baselines. For a control this may exclude more than for the real
        # connectome — that is a result about the control, and it is logged
        # rather than worked around.
        if not self.cfg.pool:
            print("resolving eligible replicates from adapter-off baselines "
                  f"(filter: n_active <= 1500, condition={self.cfg.condition})...",
                  flush=True)
            pool, baselines = resolve_pool(
                CANDIDATE_REPLICATES, self.cfg.param_seed,
                shuffle_seed=self.cfg.shuffle_seed, duration_s=self.cfg.trial_s,
                random_seed=self.cfg.random_seed_net)
            self.cfg.pool = tuple(pool)
            self.cfg.baselines = baselines
            print(f"  eligible pool: {list(pool)}\n", flush=True)
        with open(self.out / "replicate_eligibility.json", "w") as f:
            json.dump({"condition": self.cfg.condition,
                       "param_seed": self.cfg.param_seed,
                       "shuffle_seed": self.cfg.shuffle_seed,
                       "threshold_n_active": 1500,
                       "baselines": {str(k): v for k, v in self.cfg.baselines.items()},
                       "pool": list(self.cfg.pool)}, f, indent=1)

        state = self.load_checkpoint() if resume else None
        if state is not None:
            es, start_gen = state["es"], state["generation"] + 1
            history, best = state["history"], state["best"]
            print(f"resuming from generation {start_gen}", flush=True)
        else:
            es = cma.CMAEvolutionStrategy(
                self.initial_z[self.active_indices], self.cfg.sigma0,
                {"popsize": self.cfg.population, "seed": self.cfg.seed,
                 "verbose": -9})
            start_gen, history, best = 0, [], {"score": -np.inf, "z": None}

        lock = mp.Manager().Lock()
        with mp.Pool(self.cfg.workers, initializer=_init_worker,
                     initargs=(lock, self.cfg.as_dict())) as pool:
            for gen in range(start_gen, self.cfg.generations):
                t0 = time.time()
                active_population = es.ask()
                population = [self.expand_z(z) for z in active_population]
                replicates = self.episode_replicates(gen)

                tasks = [(z, rep, gen * 1000 + i, i)
                         for i, z in enumerate(population)
                         for rep in replicates]
                rows = pool.map(_evaluate_one, tasks)

                # CMA-ES minimises, so negate. Average over episodes.
                scores = np.full(len(population), np.nan)
                for i in range(len(population)):
                    vals = [r["reward"] for r in rows if r["candidate"] == i]
                    scores[i] = float(np.mean(vals))
                es.tell(list(active_population), [-s for s in scores])

                gen_best = int(np.argmax(scores))
                if scores[gen_best] > best["score"]:
                    best = {"score": float(scores[gen_best]),
                            "z": np.asarray(population[gen_best]).tolist(),
                            "generation": gen}

                candidate_records = []
                for i, score in enumerate(scores):
                    candidate_rows = [r for r in rows if r["candidate"] == i]
                    terms = {k: float(np.mean([r[k] for r in candidate_rows]))
                             for k in candidate_rows[0] if k.startswith("term_")}
                    candidate_records.append({
                        "candidate": i, "score": float(score),
                        "replicates": [r["replicate"] for r in candidate_rows],
                        "fraction_peak_saturated": float(np.mean([
                            r["peak_n_active"] > 1500 for r in candidate_rows])),
                        "mean_saturated_step_fraction": float(np.mean([
                            r["fraction_saturated_steps"] for r in candidate_rows])),
                        "terminated_fraction": float(np.mean([
                            r["terminated"] for r in candidate_rows])),
                        **terms,
                    })

                # Per-term logging: a single scalar cannot show WHICH term is
                # being optimised, and that is what reward hacking looks like.
                term_means = {k: float(np.mean([r[k] for r in rows]))
                              for k in rows[0] if k.startswith("term_")}
                record = {
                    "generation": gen, "best": float(scores.max()),
                    "median": float(np.median(scores)), "worst": float(scores.min()),
                    "sigma": float(es.sigma), "wall_s": round(time.time() - t0, 1),
                    "frac_saturated": float(np.mean([r["raw_saturation"] > 0
                                                     for r in rows])),
                    "frac_peak_saturated": float(np.mean([
                        r["peak_n_active"] > 1500 for r in rows])),
                    "mean_saturated_step_fraction": float(np.mean([
                        r["fraction_saturated_steps"] for r in rows])),
                    "frac_unstable": float(np.mean([r["unstable"] for r in rows])),
                    "frac_terminated": float(np.mean([r["terminated"] for r in rows])),
                    "replicates": replicates, "stage": self.cfg.stage,
                    "rig": self.cfg.rig, **term_means,
                }
                if "adhesion_duty" in rows[0]:
                    record["adhesion_duty"] = np.mean(
                        np.asarray([r["adhesion_duty"] for r in rows]), axis=0).tolist()
                history.append(record)
                self._write_generation_records(gen, rows, candidate_records)
                self.save_checkpoint(es, gen, history, best)
                with open(self.out / "history.json", "w") as f:
                    json.dump(history, f, indent=1)

                print(f"gen {gen:4d}  best={record['best']:+.4f} "
                      f"med={record['median']:+.4f}  sigma={record['sigma']:.3f}  "
                      f"sat={record['frac_saturated']:.2f}  "
                      f"{record['wall_s']:.0f}s", flush=True)

        with open(self.out / "best.json", "w") as f:
            json.dump({"best": best, "reward_hash": self.reward_hash,
                       "config": self.cfg.as_dict()}, f, indent=1)
        return best, history
