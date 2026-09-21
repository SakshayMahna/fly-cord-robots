# Trained adapter, frozen connectome: design and compute gate

*Written 2026-09-20; **decisions applied 2026-09-21**. The trainer is still
not built — that remains the stop line. What has landed since the first
draft is the compute work the gate justified: the active-column speedup and
its tests, the frozen-interface regression test, and the throughput
measurements behind §5.6.*

**Audience:** the project owner, as the decision-maker on the compute gate,
and whoever implements this afterwards.

---

## 0. Decisions taken, and corrections to the original brief

**0.1 `PHASE4.md` does not exist.** Phase 4's record is `docs/closed_loop/`
— `AUDIT.md`, `SENSORY_MAP.md`, `PREREGISTRATION.md` (+ two dated
amendments), `RESULTS.md`, `LOG.md`. That is what was read for this design.
Nothing is missing; the file is just named differently.

**0.2 Folder name — resolved.** The first draft lived in `phase5/`, which
`CLAUDE.md` forbids ("no phase-numbered file/folder names — use
content-derived names"). Moved to `docs/trained_adapter/`, alongside
`docs/closed_loop/`. **The convention applies to future file names in this
work too**, so nothing here is phase-numbered.

**0.3 Compute route — resolved: many-core CPU, not GPU.** §5.5 gives the
measured reasons. The Colab GPU figure the original brief asked for was
never measured (no GPU on this machine) and is not invented anywhere in
this document. That route is closed by decision, not by assumption.

**0.4 Decisions recorded** (2026-09-21), each expanded in the section named:

| decision | where |
|---|---|
| Land the active-column speedup with a density guard, bit-identity tests and a saturation regression test | §5.2 — **done** |
| Motor decoder: adapter supplies constants at call time; regression-test the frozen behaviour | §2.3 — **done** |
| Stage A1 runs the **lean** budget; the default row needs explicit approval | §5.4 |
| A3 turning uses **asymmetric DNg100 drive**; MDN is exploratory-only | §2.2 |
| Plan for a rented many-core CPU VM | §5.6 |

---

## 1. What Phase 4 constrains

The pilot's finding (`docs/closed_loop/RESULTS.md`) is that across four
hand-designed encoders, proprioceptive feedback is either negligible
(effect 0.0000–0.0010) or destroys the rhythm (≈1.01, total decorrelation),
with nothing in between. The measured mechanism: sensory neurons sit ~30×
below their own firing threshold in the stable regime, and thresholds are
narrowly distributed, so the pool crosses **together** rather than
gradually.

Two consequences for Phase 5's design, both load-bearing:

1. **The usable region, if one exists, is narrow and the trainer has to
   find it rather than be handed it.** This is why the encoder's
   saturation cap, offset and per-class gain are all trainable, and why the
   reward carries an explicit saturation penalty (§3) instead of a
   hard-coded gain ceiling. If the trainer's solution is "turn the sensory
   gain to zero," that is a real and reportable result — it would say the
   adapter drives the body *open-loop* and Phase 4's wall is not an artefact
   of hand design.

2. **RESULTS.md §5 already names the specific hypothesis worth testing**,
   and it is the one thing the trained encoder can do that a hand-designed
   one could not: "An encoder whose *typical* rather than peak current sat
   near threshold might recruit gradually." A trainable per-class offset is
   exactly that knob. This is flagged as a **hypothesis**, not a prediction.

---

## 2. The adapter — exact parameter set

Everything in this table is **our own addition**, per the honesty rule.
None of it is in the fly; it is the interface, and the interface is the only
thing that is ever trained.

Parameters are shared **per thoracic segment (T1/T2/T3), not per leg**, with
a single global left–right asymmetry scalar. Rationale: six independent
per-leg parameter sets would let the trainer hand-build a gait leg by leg,
which is precisely the outcome that would make the result uninteresting —
we want coordination to come from the connectome, not from the adapter
having enough freedom to impose it. Left/right sharing also avoids baking in
the annotation asymmetry the sensory interface already has to correct for
(T1: 23 left chordotonals vs 57 right).

### 2.1 Sensory encoder — 31 parameters

| block | parameters | count |
|---|---|---:|
| chordotonal (×3 segments) | gain, offset, saturation cap, position/velocity mix | 12 |
| hair plate (×3 segments) | gain, offset, saturation cap, limit-threshold fraction | 12 |
| chordotonal tuning (×3 segments) | fractionation spread σ, band-centre offset | 6 |
| global | left/right gain ratio | 1 |

The offset is the new knob relative to Phase 4: it shifts the *whole* class's
current, so the trainer can park typical drive near threshold instead of
having peaks be the only thing that ever crosses.

### 2.2 Command drive — 4 parameters

Input channels, not free weights: a level injected into command neurons that
**are present in the simulated network**, verified by query, not assumed:

| neuron | rows in the simulated table | note |
|---|---|---|
| DNg100 | 59, 282 | the walking command Pugliese et al. use; 380 is their verified value |
| MDN | 3075, 3413, 3952, 4043 | present |
| DNa02 | 82, 90 | present |

Parameters: DNg100 level, MDN level, left/right drive asymmetry, DNg100
onset ramp time constant.

> **Verified vs. not.** That these four types exist at these row indices in
> the simulated MANC table is **confirmed** (queried directly, not assumed).
> That MDN drives *backward* walking and DNa02 drives *turning* is a claim
> from the literature that **this project has not checked against primary
> text**. A citation has been offered for MDN (Bidaye et al., the
> "moonwalker" descending neuron); per the project's own rule a citation
> supplied second-hand is a lead to verify, not a source, so it stays
> marked unverified here until someone reads the paper.
>
> **Decided (2026-09-21):**
> - **A3 turning uses asymmetric DNg100 drive.** It needs no new functional
>   claim — it is the same command neuron Pugliese et al. already
>   characterise, driven unequally on the two sides.
> - **MDN backward-walking runs as a clearly-labelled exploratory test
>   only.** No claim, in RESULTS or the video script, rests on it. If it
>   produces something striking, that is a reason to verify the primary
>   source and pre-register a proper test — not a reason to report it.

### 2.3 Motor decoder — 30 parameters

| block | parameters | count |
|---|---|---:|
| per antagonist pair × segment (3×3) | gain_rad, rate_scale_hz, offset | 27 |
| per segment | output low-pass time constant | 3 |

**This required explicitly un-freezing the motor interface.** `CLAUDE.md`
and `sim/closed_loop.py` both record `motor_neuron_to_joint.py` as FROZEN
for the duration of the closed-loop work, config hash `04be9dec…`.

**Approved and implemented (2026-09-21):** the module keeps its frozen
defaults and the adapter supplies constants **at call time**, so the frozen
path stays the default and every Phase 4 number stays reproducible.
`tests/test_frozen_motor_interface.py` pins this from both directions:

- the config hash is **recomputed from the live module constants** and
  compared to `04be9dec181ec2e20…` as recorded in `AUDIT.md` §5 — so the
  test fails if anyone edits a gain, a module name or the leg map. (The
  hashing code was never committed in Phase 4; the serialisation that
  reproduces the recorded digest — `json.dumps(cfg, sort_keys=True)` over
  the four documented fields — was recovered and is now in the test, so the
  hash is checkable rather than merely quoted.)
- the **rule itself** is re-derived independently from AUDIT.md's
  `joint = neutral + 0.5 rad · tanh((rate_pos − rate_neg) / 3.0 Hz)` and
  compared against `compute_joint_targets`, rather than against a stored
  golden array — so a failure says what broke.
- passing the frozen constants explicitly is asserted **bit-identical** to
  passing nothing, and passing different ones is asserted to actually
  change the output. The second guard matters more than it looks: without
  it, a decoder whose parameters were silently ignored would train to
  nothing and look like a null result about the connectome.

**Open decision:** 3 of 9 per-leg motor modules (femur reductor, substrate
grip, tarsus control) have no antagonist mapping and are currently unused.
Phase 5 could map them, adding ~9 parameters and, more importantly, giving
the fly tarsal grip on the ball. Recommend **not** in A1 — one change at a
time — and revisiting if A1 plateaus.

### 2.4 Optional, trained only if the above fails — 3 parameters

Interface-level interleg phase coupling (strength, preferred phase offset,
time constant), **explicitly ours and not in the fly**. Per the brief: train
without it first, and report both outcomes either way.

**Total: 65 parameters** (68 with coupling). Comfortably inside the "low
hundreds" target, and small enough that CMA-ES is the right tool.

### 2.5 The no-bypass requirement

Every control signal must pass sensors → fly sensory neurons → connectome →
fly motor neurons → joints. The automated test, mirroring Phase 4's locality
and `g_fb = 0` gates:

- **Structural:** the joint-target vector is a pure function of
  `neural_model.rates` and the decoder parameters; assert by passing a
  sentinel rate vector and checking targets change, then freezing rates and
  perturbing every sensor reading and asserting targets do **not** change.
- **Numerical:** with sensory gain 0, the trial is bit-identical to
  open-loop at the same seed (Phase 4's existing gate, reused unchanged).
- **Topological:** assert `W_eff` is unchanged from the loaded connectome at
  the end of every trial — hash it at load and at teardown.

---

## 3. Reward

Measured on the tethered-ball rig, over 3.5 s after a fixed 0.5 s transient
discard (identical to Phase 4's convention).

| term | definition | sign |
|---|---|---|
| progress | ball pitch rotation (the rig's own forward-walking axis; verified −0.95 rad/s under synthetic tripod drive) | + |
| straightness | penalty on roll and yaw magnitude | − |
| posture | penalty on joint excursions beyond the measured working range | − |
| energy | sum of squared joint-target *changes* | − |
| **saturation** | penalty if `n_active > 1500` — **Pugliese's own documented oversaturation criterion**, already used as Phase 4's stability filter | − |
| rhythmicity | the AR(1)-gated rhythm score already built in `analysis/interleg_coordination.py` | + |

The saturation term is what lets the trainer discover Phase 4's constraint
instead of being told it. Weights are themselves a design choice and will be
fixed **before** the first real run and recorded here, not tuned per
condition (the brief's "no per-condition hand tuning" rule).

**Known gap, carried from Phase 4 and not fixable here:** there is no leg
load / campaniform channel in either MANC (9 leg sensilla) or MaleCNS (12),
so ground contact cannot be fed back through its real channel. A posture and
progress reward is therefore doing work that, in a real fly, load sensing
would partly do. This is a real limitation, stated up front.

---

## 4. Curriculum and controls

**A1 tethered-on-ball** (rig already built and verified) → **A2 free walking
on flat ground** → **A3 command test with the adapter frozen**.

A2 is the first time the body can fall over. Expect to need an early-
termination condition and a survival term; that is a change to the reward,
so it gets recorded as an amendment rather than slipped in.

Controls, reusing Phase 4's existing implementations
(`neural/connectome_controls.py`) so they are not re-derived:

| id | what | already exists? |
|---|---|---|
| C1 | degree- and sign-preserving shuffle, interface edges protected | yes — `degree_preserving_shuffle`, with a degree-preservation test |
| C2 | random recurrent network, matched size and sparsity | **no** — needs building |
| C3 | sine/CPG baseline controller, same body and reward | **no** — needs building (cheap; no connectome) |

Each control gets the **same adapter, the same trainer, the same budget.** A
control that is given less search is not a control. If C1–C3 match the real
connectome, `RESULTS.md` says so plainly, in those words.

---

## 5. Compute gate — measured

All figures measured on this machine (Apple M3 Pro, 11 cores, 19 GB), on the
real 23,532-neuron network, 4 s trial, tethered-on-ball, stable regime.

### 5.1 Where the time actually goes

Component timings, summed, reproduce the recorded pilot median (24.7 s
estimated vs 25.2 s actual across 172 recorded trials) — so the breakdown is
trustworthy:

| component | s per 4 s trial | share |
|---|---:|---:|
| **neural (sparse matvec, RK4 ×4/step)** | **21.9** | **88.7%** |
| physics (10 MuJoCo substeps/step) | 2.2 | 9.0% |
| sensory encode | 0.3 | 1.2% |
| motor decode | 0.3 | 1.1% |
| physics I/O | 0.0 | 0.0% |

**This inverts the usual assumption.** The body is not the bottleneck; the
connectome is. Any acceleration effort aimed at MuJoCo is aimed at 9% of the
cost.

### 5.2 A lossless 3.4× — approved and landed 2026-09-21

In a stable trial only **471 of 23,532** neurons are ever nonzero (measured:
369 at 0.1 s rising to 471 at 1.0 s). A neuron at exactly zero contributes
exactly zero to `W @ R`, but scipy's CSR matvec touches all 1,372,404 stored
entries regardless. Gathering only the active **columns** (CSC) touches
33,514 — 2.44%.

Measured, end-to-end, through the real `run_trial`:

| | wall | outputs |
|---|---:|---|
| current code | 33.1 s | — |
| active-column step | **9.75 s** | `max|diff| = 0.000e+00` on motor rates, joint angles, sensory drive; identical `n_active` (388) and `max_fr` (19.137) |

**3.40× end-to-end, bit-identical.** This is not an approximation, a
reduced model, or a topology change — it is declining to multiply by zeros.
It changes no weight, sign, or neuron.

Caveat, measured rather than assumed: the speedup depends on how sparse the
active set is, and **inverts in the saturated regime**.

| active neurons | fraction of nnz touched | speedup |
|---:|---:|---:|
| 471 (stable) | 12.3% | 2.1× |
| 2,000 | 31.8% | 1.4× |
| 4,629 (runaway) | 51.3% | 1.03× |
| 23,532 (all) | 100% | **0.74× — slower** |

(worst case: the busiest columns active). So it ships with a density guard
that falls back to the dense path above **35% of nnz**
(`ACTIVE_COLUMN_NNZ_FRACTION` in `neural/steppable_rate_model.py`), chosen
below the measured inversion point so the model is never slower than
before. Runaway trials — which training will actively be penalised for —
cost full price.

**As landed**, in `SteppableRateModel`:

- `active_column_matvec=True` by default; `False` forces the original
  dense path, which is what the equivalence tests compare against.
- per-trial counters (`n_matvec_fast` / `n_matvec_dense`) so it is
  visible how often a trial fell back, rather than inferred.
- the argument for why this is exact is written into the module docstring
  under a heading that says it is **not** a difference from Pugliese's
  model: dropping terms that are exactly `w × 0.0` cannot change a
  floating-point sum, and the order of the surviving terms is preserved
  because scipy's CSR and CSC matvecs both accumulate each row in
  increasing column order.

**Tests** (`tests/test_steppable_model.py`, 7 gates). They assert
`np.array_equal`, not `assert_allclose` — a tolerance would pass even if
the optimisation were quietly perturbing a trajectory, which is the exact
failure worth catching in a system the audit already showed to be
phase-sensitive (`AUDIT.md` §4):

| gate | what it pins |
|---|---|
| bit-identity, open loop | 500 steps, both paths, equal to the last bit |
| bit-identity, with `extra_input` | the closed-loop case, where `total` is not just `W@r` |
| matvec elementwise | the operation itself at a realistic sparse operating point |
| all-zero rates | the degenerate case the whole argument rests on |
| **saturation fallback** | the guard actually fires at 100% active — if it stops firing, saturated trials silently cost ~35% more |
| **saturated fallback still exact** | a run crossing the threshold mid-trial is identical either side of it |
| guard threshold sanity | the constant stays below the measured inversion point |

Two of these are the saturation regression guard, because the speedup
**inverts** rather than merely fading — a guard that silently stopped
working would be a pessimisation, not a missing optimisation.

Full suite after landing: **32 passed**, including Phase 4's `g_fb = 0`
bit-identity gate, which is the one that would have caught any change to
the neural trajectory.

### 5.3 Parallel scaling of the neural step, measured

*Superseded for planning purposes by §5.6, which measures whole trials in
parallel worker processes. Kept because it isolates the neural step, and so
shows **why** the ceiling in §5.6 is where it is.*

The sparse matvec is memory-bandwidth bound, so cores do not come free:

| processes | aggregate neural steps/s | scaling | efficiency |
|---:|---:|---:|---:|
| 1 | 442 | 1.00× | — |
| 2 | 963 | 2.18× | 109% |
| 4 | 1,319 | 2.98× | 75% |
| **6** | **1,687** | **3.82×** | **64%** |
| 8 | 1,434 | 3.25× | 41% |
| 10 | 1,341 | 3.03× | 30% |

**Saturates at ~6 workers, ~3.8×**, then degrades. Effective throughput on
this Mac with both improvements: ~2.6 s/trial, **≈1,200–1,400 trials/hour**.

### 5.4 Budget

At 65 parameters, CMA-ES's default population is λ = 4 + ⌊3 ln 65⌋ = 16;
noisy fitness argues for more. Episodes per candidate matter because Phase 4
showed per-replicate behaviour varies *enormously* — replicate 1 stays stable
through the entire gain sweep while replicate 0 bifurcates — so a candidate
scored on one neuron-parameter draw is being scored on luck.

| scenario | λ | episodes | gens | trials | status |
|---|---:|---:|---:|---:|---|
| default | 32 | 3 | 300 | 28,800 | **needs explicit approval — do not run** |
| reduced-λ | 16 | 3 | 300 | 14,400 | not approved |
| **lean** | 16 | 2 | 250 | **8,000** | **approved for Stage A1** |

**Decided 2026-09-21: Stage A1 runs the lean row.** The default row is not
to be run without explicit approval.

Stage A1 needs the real run **plus C1 and C2 at equal budget** (C3 is much
cheaper — no connectome), so A1 is **~24,000 trials** in total. Wall-clock
in §5.6.

What the lean row costs, stated so it is not discovered later: 2 episodes
per candidate instead of 3 is the thinnest defensible averaging over
per-replicate variance, and Phase 4 showed that variance is **large** —
replicate 1 stayed stable across the entire gain sweep while replicate 0
bifurcated. A candidate scored on 2 draws is still substantially scored on
luck. The practical consequence is that a *negative* A1 result at lean
budget is weak evidence: it would not distinguish "the adapter cannot do
this" from "the search was too noisy." A *positive* result is unaffected by
this concern. If A1 plateaus, the first thing to question is the budget,
not the hypothesis — and that is the moment to ask about the default row.

### 5.5 Why a GPU is the wrong answer here — decided 2026-09-21

I could not measure Colab GPU throughput from this machine; there is no GPU
attached. I am not going to invent the number. What I *can* report is the
structural reasons to expect a GPU to disappoint, each measured:

1. **The loop is sequential in time.** 4,000 neural steps, each depending on
   the physics state the previous one produced. Time cannot be parallelised;
   only the ES population can.
2. **Batching the population requires batching the physics too**, or the GPU
   idles every step waiting on CPU MuJoCo. That means MJX — and **FlyGym
   2.1.0 contains zero references to mjx** (checked), and `mujoco.mjx` is
   not installed. So this is a port, not a flag.
3. **The body is a hard MJX target**: 69 of its 70 geoms are **meshes**.
   Mesh collision is MJX's expensive and most feature-limited path. This is
   a measured property of the model, and the single largest risk in any GPU
   plan.
4. **The neural workload is small in the regime we want.** 33,514 nonzeros
   per matvec is a tiny kernel; 4 launches/step × 4,000 steps = 16,000
   launches per trial. GPUs win here only when the population batch is large
   enough to amortise that — which returns us to (2).

**Decided: many CPU cores, not a GPU.** Sizing in §5.6.

Should the question ever be reopened, the measurement that would settle it
is not a microbenchmark of the matvec but a batched-rollout test: port
`SteppableRateModel` to JAX with `jax.experimental.sparse.BCOO`, `vmap` over
a population of 32, and drive it with *recorded* joint trajectories (no
MuJoCo in the loop). That is an upper bound on the GPU path — if it does not
win there, it will not win with physics attached. Not written.

### 5.6 Population-parallel throughput and VM sizing — measured 2026-09-21

Real 4 s closed-loop trials in N worker processes, post-speedup, barrier-
synchronised so the timed phase is genuinely concurrent
(`benchmarks/bench_population_throughput.py`).

| workers | median trial | trials/hr | peak RSS/worker | steady RSS/worker |
|---:|---:|---:|---:|---:|
| 1 | 8.65 s | 326 | 2.6 GB | 0.45 GB |
| 2 | 8.44 s | 861 | 3.3 GB | 0.56 GB |
| 4 | 8.18 s | 710 | 3.4 GB | 0.61 GB |
| 6 | 10.15 s | 979 | 3.3 GB | 0.58 GB |
| 8 | 12.16 s | **1,174** | 3.3 GB | 0.57 GB |
| 10 | 14.73 s | **1,331** | 3.3 GB | 0.57 GB |

**Single-trial time after the speedup: ~8.2–8.7 s uncontended** (was 25.2 s
as the pilot ran it). The controlled A/B in §5.2 — same session, same
replicate, only the matvec changed — measured 33.1 s → 9.75 s; the spread
between these is machine state and replicate, so treat ~8.5 s as the
planning figure and 3.4× as the attributable speedup.

**`trials/hr` here is gated by the slowest worker, and that is deliberate.**
An ES generation is synchronous: the update needs every candidate's fitness,
so a generation costs what its slowest rollout costs, not the mean. The
column is therefore the number that actually converts to training wall time.

**Two artefacts to read past, rather than around:**

- The 1-worker baseline (326/hr) is noisy — other work was running on this
  machine — which is why 2 workers show an impossible 132% efficiency and 4
  workers dip *below* 2. Do not read the scaling column as a clean curve;
  the trustworthy quantities are the median trial time and the top-end
  trials/hr.
- **This is a heterogeneous CPU: the M3 Pro has 6 performance and 5
  efficiency cores.** Workers landed on E-cores run several times slower,
  and because throughput is max-gated, they set the pace. That is why the
  median trial time (8.18 s at 4 workers) is so much better than the implied
  per-trial rate. A server with homogeneous cores should do materially
  better than this curve suggests — which cuts the other way from most
  benchmark optimism, so it is worth stating explicitly.

#### Extrapolation to 8 / 16 / 32 cores

Ideal single-core rate is 3600 / 8.5 s ≈ **424 trials/hr/core**. Measured
efficiency at 8 workers here is 1,174 / (8 × 424) = **0.35**, depressed by
the E-core gating above. A homogeneous server should land higher; memory
bandwidth is the binding constraint (§5.3) and server parts have more
channels, but that is an argument, not a measurement.

| cores | η = 0.35 (this Mac's measured efficiency) | η = 0.6 (homogeneous cores) |
|---:|---:|---:|
| 8 | **1,174 — measured** | — |
| 16 | ~2,400 (extrapolated) | ~4,100 (extrapolated) |
| 32 | ~4,700 (extrapolated) | ~8,100 (extrapolated) |

**Everything except the 8-core row is extrapolation, not measurement.**

#### Stage A1 wall time and cost

Stage A1 at the approved lean budget is 8,000 trials × 3 runs (real, C1, C2)
= **24,000 trials**, plus C3 which is much cheaper. That is
24,000 × 8.5 s ≈ **57 core-hours** of actual compute.

| cores | η = 0.35 | η = 0.6 |
|---:|---:|---:|
| 16 | ~10 h wall / ~162 core-h billed | ~6 h / ~95 core-h |
| 32 | ~5 h wall / ~162 core-h billed | ~3 h / ~95 core-h |

**Cost.** Billed core-hours are 57/η regardless of instance size, so the
decision is wall time, not money. At an indicative **$0.03–0.05 per
vCPU-hour** for general-purpose on-demand x86 — **this is a rough figure
from memory, not a quote; verify current pricing before committing spend**
— 95–162 core-hours is roughly **$3–8** per Stage A1, less on spot. Even a
4× pricing error leaves this inexpensive; the reason to care about η is
iteration speed.

**Recommended shape: 16–32 vCPU, ≥ 32 GB RAM, x86 or ARM.** RAM is set by
steady-state 0.57 GB/worker (32 workers ≈ 18 GB + OS), *not* by the 7.4 GB
transient build peak — provided pool startup is serialised, as the benchmark
does, or workers load a cached sparse matrix instead of each rebuilding from
the dense one. Rebuilding concurrently on 32 workers would need ~235 GB and
will OOM; this is the single most likely way to waste a VM rental.

**Before committing to a long run, do one calibration run on the actual
VM** — the 16/32-core numbers above are extrapolated from a heterogeneous
11-core laptop, and the honest error bar on them is roughly a factor of two.
`bench_population_throughput.py` runs as-is and takes about ten minutes.

---

## 6. Cost-reduction options — status

1. ~~**Land the lossless 3.4× step**~~ — **done** (§5.2). Bit-identical,
   tested, 32/32 suite green.
2. ~~**Run on many-core CPU rather than GPU**~~ — **decided** (§5.5, §5.6).
3. ~~**Lean budget**~~ — **approved for Stage A1** (§5.4).
4. **Shorten training trials to 2.5 s, validate finalists at 4 s** — still
   available, not taken. ~37% saving; 2.5 s still gives ~22 cycles at the
   measured ~11 Hz rhythm. Worth taking only if §5.6's wall time is still
   uncomfortable; it costs some rhythm statistics per trial.
5. **RK4 → RK2** — still available, **not taken and not recommended
   without seeing the validation number first**. It would halve matvecs per
   step, but it changes the neuron model's numerics, which the brief's own
   rule puts behind explicit approval. The audit found the system is
   genuinely phase-sensitive between any two numerically distinct runs
   (`AUDIT.md` §4), so I expect this to degrade agreement with Pugliese's
   published output. `validate_steppable_model.py` can measure exactly how
   much, and that measurement should come before any decision.

**Not recommended: a cropped subnetwork.** I tested it. Neurons not
forward-reachable from the driven set are provably always zero and could be
deleted losslessly — but that is only **846 of 23,532 (3.6%)**; the network
reaches 13,541 neurons within two hops. There is no meaningful free crop,
and any *larger* crop would be a topology change.

---

## 7. What I am not doing without your approval

- **Short-term adaptation (SFA / synaptic depression)** in the neuron model.
  The brief flags it as needing approval and I agree; it is also the single
  most plausible candidate for making feedback usable, because it would
  break exactly the mechanism Phase 4 measured (a narrow threshold
  distribution crossed all at once). If you want it: config-flagged off by
  default, with a test asserting the disabled path reproduces current
  behaviour bit-identically, and a re-run of the Phase 4 bifurcation to
  report the effect. **Not started.**
- **§6 items 4 and 5** (shorter training trials; RK4 → RK2).
- **The default budget row** (§5.4). Stage A1 is approved at lean only.
- **The trainer itself.** This is the stop line and it has not been
  crossed: there is no ES loop, no reward implementation, no adapter
  parameter vector, and no C2/C3 control in the codebase.

## 8. What has and has not been built

**Built** (2026-09-21), all of it compute-gate or regression work:

| | |
|---|---|
| `neural/steppable_rate_model.py` | active-column matvec + density guard + fallback counters |
| `tests/test_steppable_model.py` | 7 gates: bit-identity ×3, degenerate case, saturation fallback ×2, guard sanity |
| `tests/test_frozen_motor_interface.py` | 5 gates: config hash vs `04be9dec…`, the documented rule re-derived, call-time constants |
| `docs/trained_adapter/benchmarks/` | the scripts behind every number in §5 |

Suite: **32 passed**.

**Not built:** the trainer, the adapter parameter vector, the reward, the
curriculum, C2, C3, and any Stage A/B run.

## 9. Open questions for you

1. Is the lean-budget caveat in §5.4 acceptable — specifically that a
   *negative* A1 result at 2 episodes/candidate will be weak evidence?
2. VM shape: §5.6 recommends **16–32 vCPU, ≥ 32 GB**, ~$3–8 per Stage A1 at
   indicative pricing that still needs verifying. Confirm before any spend,
   and expect the first ten minutes of the rental to go on a calibration
   run rather than training.
3. The reward weights (§3) are a design choice that must be fixed *before*
   the first real run and recorded, per the no-per-condition-tuning rule.
   Do you want to see them proposed as a separate short decision, or should
   I fix them when the trainer is approved?
