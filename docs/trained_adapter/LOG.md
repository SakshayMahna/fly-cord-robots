# Log — trained adapter (Phase 5)

Chronological, newest last. Kept separate from `docs/logs/` for the same
reason `docs/closed_loop/LOG.md` is: this is one continuous arc, not daily
findings. `DESIGN.md` is the current state; this is how it got there.

---

## 2026-09-20 — compute gate

Measured the gate before designing around it, and the headline inverted the
assumption the brief was built on: **the connectome is 88.7% of a trial's
cost and the body is 9.0%.** Any effort aimed at accelerating MuJoCo is
aimed at a tenth of the problem. The component timings sum to 24.7 s against
a recorded pilot median of 25.2 s across 172 trials, which is what makes the
breakdown trustworthy rather than indicative.

Three things followed from looking at that number properly.

**A lossless speedup, found by noticing a mismatch between two kinds of
sparsity.** W is 0.248% dense and stored sparsely, so that was already
exploited. What was not is that the *rate vector* is sparse too, and
dynamically so: only 471 of 23,532 neurons are ever nonzero in a stable
trial. scipy's CSR matvec touches all 1,372,404 stored entries regardless of
how many entries of R are zero. Gathering only the active columns touches
33,514 — 2.44% — and, because a term `w × 0.0` is exactly zero and the
surviving terms are visited in the same order, the result is identical to
the last bit. Measured end-to-end through the real `run_trial`: 33.1 s →
9.75 s, `max|diff| = 0.000e+00` on motor rates, joint angles and sensory
drive.

**But it inverts, which is the part worth remembering.** Speedup falls to
1.03× at 4,629 active neurons (the runaway regime) and to **0.74× — slower
than before** with all 23,532 active, because gathering columns costs more
than it saves once most columns are active. An optimisation that becomes a
pessimisation in exactly the regime the pilot already identified as
dangerous is a trap; hence the density guard, and hence two of the seven
tests are specifically about the fallback firing.

**No useful crop exists, and this was worth testing rather than assuming.**
A neuron that is not forward-reachable from the driven set can never leave
zero (at R=0 with no input, `act` is 0 and so is dR/dt), so deleting it
would be exactly lossless. That is only **846 of 23,532 — 3.6%**. The
network reaches 13,541 neurons within two hops. So "crop the network" is off
the table as a cheap win, and any *larger* crop would be a topology change,
which the honesty rules forbid.

**On the GPU question**, the brief asked for a Colab measurement that this
machine cannot produce, and rather than estimate one, recorded the
structural reasons a GPU looks wrong here: the loop is sequential in time,
batching the population would require batching physics, FlyGym 2.1.0 has
**zero** references to mjx, and 69 of the body's 70 geoms are meshes — MJX's
most limited path. Reported as reasons, with the missing measurement named
as missing.

---

## 2026-09-21 — decisions applied

Folder moved `phase5/` → `docs/trained_adapter/`; the first draft had used
the brief's path in violation of the project's own "no phase-numbered
names" rule, flagged at the time rather than silently chosen.

**Landed the active-column matvec** behind `active_column_matvec=True` with
`ACTIVE_COLUMN_NNZ_FRACTION = 0.35` as the guard. Tests assert
`np.array_equal`, not `assert_allclose` — a tolerance would pass even if the
optimisation were perturbing a trajectory, and `AUDIT.md` §4 already
established this system is phase-sensitive between any two numerically
distinct runs, so "close" would be exactly the wrong bar. Full suite 32
passed, including Phase 4's `g_fb = 0` bit-identity gate.

**Recovered the frozen motor interface's config hash.** Phase 4 recorded
`04be9dec181ec2e20…` in `AUDIT.md` §5 but the code that computed it was
never committed, so the hash was quotable but not checkable. Recovered the
serialisation by trying candidates against the recorded digest —
`json.dumps(cfg, sort_keys=True)` over the four documented fields — and the
regression test now recomputes it from the live module constants. A test
that restated the constant would have been worthless; this one actually
fails if someone edits a gain.

**Found a per-worker memory problem, and then that it was mostly
illusory.** Building trial components peaks at **7.39 GB** RSS and settles
at **3.29 GB**, because `neuron_params.W` is materialised dense (23,532² ×
4 B = 2.22 GB) before being converted to the 5.5 MB sparse copy the model
actually keeps. At 3.3 GB/worker, a 32-worker VM would need ~105 GB. But the
retained memory is held by JAX's caches, not by anything needed:
`jax.clear_caches()` plus deleting live arrays takes a worker to **0.55 GB**
and the model still steps correctly (441 active neurons after 200 steps, as
expected). That is a 6× reduction in the number that drives VM sizing. The
transient 7.4 GB build peak remains, so pool startup must be serialised —
or, better, workers should load a cached sparse matrix rather than each
rebuilding from the dense one.

**A debugging fix worth recording because it cost 43 minutes.** The
population-throughput benchmark hung: 43 minutes elapsed against an expected
~11, with the parent Python idle at 10 MB RSS and no output past the point
the first version had reached in under a minute.

**Suspected cause, not isolated:** `mp.Barrier` passed through
`Pool(initargs=...)`. macOS spawns rather than forks, and a raw `Barrier`
plausibly does not survive that path, whereas a raw `Lock` does — which
would explain why the hang appeared only once the barrier was added. Fixed
by switching to a `Manager()`-backed barrier and adding `timeout=900` to the
wait, after which it ran normally. That is consistent with the diagnosis but
does not prove it; the mechanism was never isolated, because the fix was
cheap and the measurement was the point.

One piece of the original reasoning was **wrong and is corrected here**:
"no worker processes" was cited as evidence, but it was an artifact of
`pgrep -f bench_population_throughput` — spawned workers run under a
`multiprocessing.spawn` command line and never match that pattern, so they
are invisible to it whether or not they exist. The real evidence was the
elapsed time and the silent output.

The
barrier was added in the first place because without it the measurement was
wrong, not just slow: builds are serialised behind a lock, so early workers
were running trials while later ones were still building, and the reported
"contention" was partly queue skew. The first, unsynchronised run showed
nproc=4 as *slower* than nproc=2, which is what prompted the re-measure.

**Throughput measured, and the metric choice matters.** 1,331 trials/hour at
10 workers; single trial ~8.2–8.7 s uncontended, against 25.2 s as the pilot
ran it. Throughput is reported **gated by the slowest worker**, not the
mean, because an ES generation is synchronous — the update needs every
candidate's fitness, so a generation costs what its slowest rollout costs.

Two artefacts are recorded in `DESIGN.md` §5.6 rather than smoothed away.
The 1-worker baseline was noisy (other work on the machine), which produces
an impossible 132% efficiency at 2 workers and a dip below it at 4; the
scaling column should not be read as a curve. And the M3 Pro is
**heterogeneous** — 6 performance cores, 5 efficiency — so workers on
E-cores set the pace under max-gating. That means the extrapolation to
homogeneous server cores is, unusually, more likely to be pessimistic than
optimistic. The 16- and 32-core figures still carry about a factor-of-two
error bar and want a calibration run on the real VM before any long booking.

Useful consequence of the memory finding: RAM sizing is set by the 0.57 GB
steady-state figure, not the 7.4 GB build peak — **provided** pool startup
is serialised. Thirty-two workers each rebuilding from the dense matrix
concurrently would want ~235 GB and would OOM. That is the most likely way
to waste a rented VM, so it is written into the design rather than left to
be discovered.

**Score noise measured, and it reframed the reward.** Ran the untrained
frozen defaults across the 7 pilot replicates to get the spread of every
observable the reward will be built from — needed both for scaling the
reward terms and for sizing the episode count.

The dominant result was not the spread itself but **where it comes from**.
One replicate (5) is saturated at `n_active = 3,794`, and it alone
contributes ~97% of the variance in the progress term: SD across all seven
replicates is 0.176, against **0.0066** once the pre-registered stability
filter (`n_active ≤ 1500`) is applied. **The filter is worth 27× in noise.**

And the mechanism matters more than the number: rep 5 produces **ball pitch
+0.471 rad/s, 30× any healthy replicate.** A seizing network thrashes and
spins the ball. So saturation is not merely scientifically wrong here — it
is *instrumentally attractive* to an optimiser looking for ball rotation.
That converted the saturation penalty from a precaution into the
largest-weighted term in `REWARD.md`, sized (−2.00, hinged at 1,500) to beat
the progress available that way by a measured ~6×.

**Episode count: the binding term is coordination, not progress.** Using
SE = σ/√E with Δ ≥ 2·SE on the filtered pool, progress needs only E ≥ 1 to
resolve 2% of walking, while the tripod index needs **E ≥ 6** to resolve a
+0.2 change and rhythmicity E ≥ 4 for one leg. Coordination is the noisy
quantity — which is unsurprising, since it is the thing the project is
actually asking about. Powered-run recommendation is therefore E = 6, with
the explicit caveat that this sizes *terminal-effect detection*, not
CMA-ES's ability to rank nearby candidates generation over generation; that
needs the within-generation spread, which does not exist until a search has
run, and is a specific deliverable of the lean pilot.

**A caveat recorded rather than glossed:** the within-replicate arm found
almost no variance (pitch varying in the 4th decimal, `n_active` identical
to the unit across 6 perturbed starts). That is substantially **by
construction** — the measurement ran at `g_fb = 0`, where there is no path
from body to neurons at all, so the neural trajectory *cannot* depend on
initial pose. It supports sampling the replicate rather than the initial
condition, but must be re-measured once the adapter runs a nonzero feedback
gain.

**Caught a sign hazard that would have been very hard to see later — see
below for the one I did NOT catch.**
`PREREGISTRATION.md` §2 records forward walking as **−0.95 rad/s** on the
pitch axis. So the reward's progress term has to be `−pitch`. With the sign
flipped, the trainer would learn to walk backwards and the scalar score
would read as success throughout. `REWARD.md` §2 makes this the first thing
in the document and requires a sign test — assert the synthetic tripod gait
scores positive and its time-reversal negative — rather than a comment.
Worth noting the untrained connectome sits at **+0.005 rad/s**, i.e.
essentially stationary and marginally *backward*.

---

## 2026-09-21 (later) — the first pilot was contaminated, and why

**Stopped at generation 24 and discarded.** The run was training on a
replicate pool that violates the project's own pre-registered stability
filter, and the contamination was large enough to swamp the fitness signal.

### What happened

`EPISODE_REPLICATES` was a hand-written tuple, `(0, 1, 3, 4, 5, 6, 7)`. The
"exclude 2" came from Phase 4, where replicate 2 was the only oversaturated
draw among the four that phase used. Phase 5 widened the pool to eight
replicates and carried the same hand-written exclusion forward **without
re-applying the rule to the new members**. Measured baselines (adapter off,
`g_fb = 0`, so the body cannot influence the network at all):

| replicate | 0 | 1 | **2** | 3 | 4 | **5** | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| baseline `n_active` | 522 | 380 | **4,176** | 367 | 488 | **3,794** | 421 | 445 |

Replicate 5 fails the pre-registered filter (`n_active > 1500`, Pugliese's
own oversaturation criterion) by a factor of 2.5, and had been sitting in
the training pool from the start.

### Why it mattered more than it looks

Generations that drew replicate 5 collapsed **regardless of candidate
quality**: median score **−1.699** with it (3 of 22 generations) against
**−0.122** without, and correlation(saturation fraction, median) = −0.744.
The saturation penalty is −2.00 by design, so a pre-saturated replicate
applies close to the full penalty to every candidate in that generation —
pure noise injected into the comparison the search is trying to make.

**The deeper error is an inconsistency between my own analysis and my own
trainer.** The episode count was sized in DESIGN.md §5.7 against the
stability-**filtered** pool (progress SD **0.0066**). The trainer sampled
the **unfiltered** pool (SD **0.176**). So the run operated at **27× the
noise the episode count was chosen for** — and 27× is the exact figure I
had reported as the argument for filtering, two sections earlier, before
failing to apply it in the code.

Best-score trend over 24 generations was flat (slope −0.00012/gen,
r = −0.011). That is consistent with the signal being swamped, but it is
**not** evidence the approach fails: 24 generations is very early for
CMA-ES on 65 parameters, and no claim is made either way.

### The fix

The pool is no longer written down anywhere. `training/replicate_filter.py`
**computes** it per network from that network's own adapter-off baseline,
and `assert_pool_eligible` is called at every point of use — including once
per generation inside the trainer, because that is precisely where this
went wrong.

Measuring a baseline needs no physics: at zero feedback there is no path
from body to neurons, so a bare neural run gives the same `n_active` as a
full ball trial. Verified directly (replicate 5: 3,794 both ways), which
makes the filter cheap enough to run unconditionally.

Controls get the same treatment against **their own** baselines. If a
shuffled or random network oversaturates on most replicates, `resolve_pool`
refuses to return an empty pool and says so — that is a finding about the
control, not a reason to loosen the threshold for it.

### Audit: where the filter was and was not applied

| site | before | now |
|---|---|---|
| `training/cma_trainer.py` | hand-written pool, **replicate 5 included** | computed + asserted per generation |
| `benchmarks/measure_score_noise.py` | hand-written, **rep 5 included** — this is what produced the two SDs (0.176 unfiltered vs 0.0066 filtered) | computed via `resolve_pool` |
| `experiments/inspect_trained_adapter.py` | `--replicate 1`, unchecked | baseline measured and asserted eligible |
| `experiments/closed_loop_pilot.py` (Phase 4) | `--replicates 0 1 2 3`, filter applied **post hoc** as an `oversaturated` column and a reported fraction | unchanged — Phase 4's published numbers were computed that way and excluding replicate 2 is already recorded in `RESULTS.md` §2 |
| `sim/trial_setup.py` | `PILOT_REPLICATE_INDICES = range(8)`, `MAIN_REPLICATE_INDICES = range(20)` | unchanged — these are candidate *index sets*, not pools; nothing samples from them directly |
| `benchmarks/bench_*.py`, `reachability.py` | `replicate=1` fixed | unchanged — timing/structure measurements, and replicate 1 is eligible (380) |
| `experiments/render_closed_loop_clips.py` | replicates 0 and 1 | unchanged — both eligible; the clips deliberately show a saturated *state*, reached by feedback gain, not by a pre-saturated draw |

Phase 4's own analyses are **not** retrospectively affected: they reported
oversaturation as an outcome per trial rather than sampling a pool to train
on, and replicate 2's exclusion is already in the record. The 4,176 measured
here independently corroborates that decision.

### Noted, not acted on

Whether a trained adapter could *survive* a pre-saturated replicate — by
lowering `dng100_level`, say — is a legitimate robustness question. It is
recorded as a possible later test and is **not** part of training.

---

## 2026-09-21 (later still) — generation-120 review point, rule committed in advance

Agreed at generation ~58, **before the generation-120 data existed**, and
written as executable code (`fly_robot/experiments/review_pilot.py`) rather
than prose so that applying it is a computation, not a judgement made while
looking at the answer. Same discipline as `docs/closed_loop/`
PREREGISTRATION §B2, which is the only reason Phase 4's result is
reportable.

**Stop the pilot at generation 120 if BOTH:**

  **A.** `runmax(MA10(best))[120] <= runmax(MA10(best))[60]` — a
  10-generation trailing mean, then its running maximum. Raw `best` is not
  used because it is the max over 16 candidates of a mean of only **2**
  episodes, so it is upward-biased by construction and drifts upward on
  noise alone. The gen-15 spike to +1.105, never reproduced in the 43
  generations since, is the worked example of why.

  **B.** `delta(mean weighted saturation term) / delta(median) > 0.5`
  between the first and last 10 generations — i.e. the median's improvement
  is still mostly the population learning not to seize, rather than
  learning to walk.

Otherwise continue to 250.

**Stated limitation of criterion B, recorded now rather than discovered
during the review:** the trainer logs per-term values as MEANS over the
population while `median` is a median of candidate scores, so comparing
them is an approximation. The clean comparison — median score with and
without the saturation term — needs **per-candidate** term logging, which
this run does not have.

That gap is worth naming plainly: per-term logging was introduced
specifically to catch reward hacking, and as built it averages over the
whole population, which hides the single candidate one would actually want
to inspect. It could not explain the gen-15 spike for exactly that reason.
Fixing it requires a restart, so it is deferred to the powered run rather
than applied mid-pilot.
