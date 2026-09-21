# Preserved artifacts — what we captured, and what can be regenerated

*2026-09-21. Inventory of everything the trained-adapter work produced,
with provenance and reproducibility stated per item.*

This file exists because `.gitignore` treats `media/` and `data/` as
regenerable output, which is the right default and **wrong for some of
what is here**. The exceptions are listed in `.gitignore` with the reason;
this is the inventory behind them.

## Reproducibility, stated honestly

| class | items | can it be regenerated? |
|---|---|---|
| **Irreplaceable** | `history_contaminated_run.json` | **No.** The code that produced it has been fixed (`replicate_filter.py`). Re-running the current code cannot reproduce it, by design. |
| **Expensive** | `pilot/history.json`, `pilot/checkpoint.pkl` | In principle yes — deterministic given seed 20260921 and the frozen reward — but it is **~5 hours** of compute for 121 generations. |
| **Cheap** | every `.mp4`, `inspection.json`, `score_noise.json` | Yes, in minutes, from the committed `z` vectors plus the scripts. Kept because they are the video deliverable and because regenerating them requires the environment to still work. |

Everything that is cheap to regenerate has its generating command recorded
below, so this archive is checkable rather than merely stored.

---

## 1. The trained-adapter pilot (`media/trained_adapter/pilot/`)

The run that was **stopped at generation 120 of 250** by the pre-committed
rule. Full analysis in `../LOG.md`.

| file | what it is |
|---|---|
| `history.json` | 121 generations: best / median / worst, sigma, per-term population means, saturation and instability fractions, and the episode replicate draw per generation. **The authoritative record** — see the warning below. |
| `checkpoint.pkl` | CMA-ES state at generation 120: distribution mean, covariance, step size, RNG, plus the reward config hash. Resumable. Pickle, so it is tied to this `cma` version (4.5.0). |
| `best.json` | The all-time best candidate — **generation 15, score +1.1053**. Recovered from the checkpoint after the early stop, since `best.json` is normally only written at a run's natural end. **This candidate is a reward hack** (§4). |
| `search_mean.json` | CMA-ES distribution mean at generation 120 — what the search actually converged toward, and a better answer to "what did it learn" than `best.json`. |
| `inspection.json` | Behavioural diagnostics for untrained / best / search-mean. |
| `review_gen120.json` | The stopping rule's verdict, as computed. |
| `replicate_eligibility.json` | Measured adapter-off baselines and the eligible pool for this run. |
| `clips/{untrained,best,searchmean}_{side,opposite_side,top_down}.mp4` | 9 videos, 3 conditions × 3 cameras, 4 s each. |

> **`pilot_stdout.log` is an UNDERCOUNT and must not be used for analysis.**
> Six worker processes interleave writes into one stream, and 8 of 116
> generation lines were mangled. It is kept as a run record only.
> `history.json` is complete and contiguous (generations 0–120, no gaps).

Regenerate the clips with:
```bash
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
    -m fly_robot.experiments.inspect_trained_adapter --run-name pilot
```

## 2. The contaminated first pilot (`history_contaminated_run.json`)

24 generations, **discarded**. Trained on a pool that included replicate 5,
whose adapter-off baseline `n_active` is 3,794 against the pre-registered
limit of 1,500.

**Kept deliberately, and it is the one file here that cannot be
regenerated.** It is the evidence for the contamination analysis in
`../LOG.md` — generations drawing replicate 5 collapsed to a median of
−1.699 against −0.122 otherwise. It was rescued from `/tmp`, where it would
have been deleted.

## 3. Noise and eligibility measurements

| file | what it is |
|---|---|
| `media/trained_adapter/score_noise.json` | Across- and within-replicate spread of every observable the reward is built from, on the **filtered** pool. The basis for the E ≥ 6 episode count. |
| `replicate_baselines_seed641.json` | Adapter-off `n_active` for all 8 replicates at parameter seed 641: `{0:522, 1:380, 2:4176, 3:367, 4:488, 5:3794, 6:421, 7:445}`. Copied out of `data/`, which is entirely gitignored (6.6 GB). |

```bash
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
    docs/trained_adapter/benchmarks/measure_score_noise.py
```

## 4. Reading the pilot clips without being misled

`best_*.mp4` looks like the success case and **is not**. That candidate
scores 94% of reference walking speed while moving its joints **20× less
than untrained** (3.17 rad of total joint travel against 61.94), with one
leg producing 100% of the motion. It imparts a single impulse and the ball
— damped at 1e-6 to model a frictionless air bearing, coast-down time
constant ~11,700 s against a 4 s trial — simply keeps spinning.

`searchmean_*.mp4` is the more honest picture of what 120 generations
produced: no net locomotion, but tripod index +0.312 against the untrained
−0.406 and 4/6 rhythmic legs against 3/6. One trial, one replicate — a
lead, not a result.

## 5. Phase 4 closed-loop clips (`media/closed_loop/`)

The `no_drive` / `stable` / `transition` / `seizure` clips behind
`docs/closed_loop/RESULTS.md` §8, plus `e0_baseline.json` and the pilot
summaries. `pilot_trials.csv` files were already tracked (`.csv` slips past
the media ignore rules).

Note `ball_check_*.mp4` is **not** preserved — it is a rig sanity check
superseded by the kinematic sign test in `fly_robot/adapter/reward.py`.

## 6. What is deliberately NOT kept

- **`data/`** (6.6 GB) — connectome dumps, caches, Pugliese simulation
  output. Regenerable and far too large. Includes
  `data/replicate_eligibility/`, a cache whose contents are copied here.
- **`external/`** — cloned reference repos, not ours.
- Intermediate scratch benchmarks, which live in
  `docs/trained_adapter/benchmarks/` as scripts rather than outputs.
