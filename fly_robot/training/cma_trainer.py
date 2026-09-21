"""CMA-ES over the 65-parameter adapter. The connectome is never trained.

Design in `docs/trained_adapter/DESIGN.md` §5b; reward in `REWARD.md`.

Three properties this file exists to guarantee, beyond running a search:

  * **The connectome is not touched.** `W_eff` is hashed at worker startup
    and re-hashed after every trial; a mismatch aborts the run rather than
    warning. No gradient is ever taken through it.
  * **The sign gate blocks startup.** `reward.sign_test()` runs before the
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

import json
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fly_robot.adapter import reward as reward_mod
from fly_robot.adapter.parameters import N_PARAMS, default_z, from_z
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED

# Replicate 2 is excluded by the pre-registered stability filter: it is
# oversaturated at baseline with no feedback at all.
EPISODE_REPLICATES = (0, 1, 3, 4, 5, 6, 7)
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
    condition: str = "real"           # "real" | "C1" | "C2"
    workers: int = 6
    out_dir: str = "media/trained_adapter"

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
    with lock:
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=EPISODE_REPLICATES[0], param_seed=cfg.param_seed,
            duration_s=cfg.trial_s, shuffle_seed=cfg.shuffle_seed)
        for arr in jax.live_arrays():
            try:
                arr.delete()
            except Exception:
                pass
        jax.clear_caches()

    _W["cfg"] = cfg
    _W["components"] = {EPISODE_REPLICATES[0]: (model, mg, sg)}
    _W["w_hash"] = hashlib.sha256(model.neurons.w_eff.data.tobytes()).hexdigest()


def _components_for(replicate: int):
    """Per-replicate components, built on demand and cached in the worker."""
    import jax

    from fly_robot.sim.trial_setup import build_trial_components

    if replicate not in _W["components"]:
        cfg = _W["cfg"]
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=replicate, param_seed=cfg.param_seed,
            duration_s=cfg.trial_s, shuffle_seed=cfg.shuffle_seed)
        for arr in jax.live_arrays():
            try:
                arr.delete()
            except Exception:
                pass
        jax.clear_caches()
        _W["components"][replicate] = (model, mg, sg)
    return _W["components"][replicate]


def _evaluate_one(task):
    """Score one (candidate, episode) pair. Runs in a worker process."""
    import hashlib

    from fly_robot.sim.closed_loop import run_trial

    z, replicate, trial_seed, cand_index = task
    cfg = _W["cfg"]
    model, mg, sg = _components_for(replicate)
    params = from_z(np.asarray(z))

    result = run_trial(model, mg, sensory_groups=sg, on_ball=True,
                       duration_s=cfg.trial_s, seed=trial_seed, adapter=params)

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
               unstable=bool(result.unstable))
    return row


# --- trainer --------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.out = Path(cfg.out_dir) / cfg.run_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.reward_hash = reward_mod.reward_config_hash()

    # -- episode assignment, derived not drawn -----------------------------
    def episode_replicates(self, generation: int) -> list[int]:
        """Which neuron-parameter draws this generation's episodes use.

        A deterministic function of the generation index, so a resumed run
        scores candidates on exactly the draws the original would have. If
        these were sampled as the run went, resuming would silently change
        the noise structure mid-search.
        """
        rng = np.random.default_rng((self.cfg.seed, generation))
        return [int(r) for r in rng.choice(EPISODE_REPLICATES,
                                           size=self.cfg.episodes, replace=False)]

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
        return state

    # -- the loop ----------------------------------------------------------
    def run(self, resume: bool = True):
        import cma

        print(f"=== {self.cfg.condition} / {self.cfg.run_name} ===", flush=True)
        print(f"reward config hash: {self.reward_hash}", flush=True)

        # BLOCKING GATE — never a warning.
        print("running sign test (blocking gate)...", flush=True)
        reward_mod.sign_test()
        print("  sign test PASSED\n", flush=True)

        state = self.load_checkpoint() if resume else None
        if state is not None:
            es, start_gen = state["es"], state["generation"] + 1
            history, best = state["history"], state["best"]
            print(f"resuming from generation {start_gen}", flush=True)
        else:
            es = cma.CMAEvolutionStrategy(
                default_z(), self.cfg.sigma0,
                {"popsize": self.cfg.population, "seed": self.cfg.seed,
                 "verbose": -9})
            start_gen, history, best = 0, [], {"score": -np.inf, "z": None}

        lock = mp.Manager().Lock()
        with mp.Pool(self.cfg.workers, initializer=_init_worker,
                     initargs=(lock, self.cfg.as_dict())) as pool:
            for gen in range(start_gen, self.cfg.generations):
                t0 = time.time()
                population = es.ask()
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
                es.tell(list(population), [-s for s in scores])

                gen_best = int(np.argmax(scores))
                if scores[gen_best] > best["score"]:
                    best = {"score": float(scores[gen_best]),
                            "z": np.asarray(population[gen_best]).tolist(),
                            "generation": gen}

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
                    "frac_unstable": float(np.mean([r["unstable"] for r in rows])),
                    "replicates": replicates, **term_means,
                }
                history.append(record)
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
