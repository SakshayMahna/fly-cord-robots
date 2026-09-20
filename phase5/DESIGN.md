# Phase 5 — Trained adapter, frozen connectome: design and compute gate

*Written 2026-09-20. **Nothing in this document has been built.** It is the
pre-implementation proposal the Phase 5 brief asks for: exact parameter set,
reward, curriculum, controls, and a compute estimate, to be approved (or
changed) before a trainer exists.*

**Audience:** the project owner, as the decision-maker on the compute gate,
and whoever implements this afterwards.

---

## 0. Three corrections to the brief, before anything else

**0.1 `PHASE4.md` does not exist.** Phase 4's record is `docs/closed_loop/`
— `AUDIT.md`, `SENSORY_MAP.md`, `PREREGISTRATION.md` (+ two dated
amendments), `RESULTS.md`, `LOG.md`. That is what was read for this design.
Nothing is missing; the file is just named differently.

**0.2 This folder is called `phase5/`, which the project's own convention
forbids.** `CLAUDE.md` says: "no phase-numbered file/folder names — use
content-derived names," and Phase 4 accordingly lives in
`docs/closed_loop/`. The brief asked for `phase5/DESIGN.md` explicitly, so
that is where this is. The content-derived equivalent would be
`docs/trained_adapter/`. Flagging rather than silently choosing either way.

**0.3 The compute gate cannot be run as specified from this machine, and
the measurements suggest it is asking about the wrong hardware.** The brief
says "measure batched rollout throughput on one Colab GPU." There is no GPU
here (`jax.devices()` → `[CpuDevice(id=0)]`). More importantly, §5 below
measures where the time actually goes, and the answer makes a GPU a poor
fit for this workload. Everything below is measured on the local M3 Pro,
with estimates labelled as estimates. A ready-to-run Colab script is
specified in §5.5 so the GPU number can be obtained if you still want it.

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
> the simulated MANC table is **confirmed** (queried directly). That MDN
> drives *backward* walking and DNa02 drives *turning* is a claim from the
> literature that this project has **not yet checked against primary
> sources**, and must be before it is written into any result or the video
> script. For Stage A3 the safest turning manipulation is **asymmetric
> DNg100 drive**, which needs no new functional claim at all.

### 2.3 Motor decoder — 30 parameters

| block | parameters | count |
|---|---|---:|
| per antagonist pair × segment (3×3) | gain_rad, rate_scale_hz, offset | 27 |
| per segment | output low-pass time constant | 3 |

**This requires explicitly un-freezing the motor interface.** `CLAUDE.md`
and `sim/closed_loop.py` both record `motor_neuron_to_joint.py` as FROZEN
for the duration of the closed-loop work, config hash `04be9dec…`. Phase 5
trains it by definition. Proposal: leave the module frozen and have the
adapter supply its constants at call time, so the Phase 4 numbers stay
exactly reproducible and the frozen path remains the default.

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

### 5.2 A lossless 3.4× that needs no approval on scientific grounds

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
that falls back to the dense path above ~35% of nnz, making it never slower.
Runaway trials — which training will actively be penalised for — cost full
price.

### 5.3 Parallel scaling, measured

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

| scenario | λ | episodes | gens | trials | hours (this Mac) |
|---|---:|---:|---:|---:|---:|
| **default** | 32 | 3 | 300 | 28,800 | **~21** |
| reduced-λ | 16 | 3 | 300 | 14,400 | ~11 |
| **lean** | 16 | 2 | 250 | 8,000 | **~6** |

Stage A1 alone needs the real run **plus C1 and C2 at equal budget** (C3 is
much cheaper — no connectome). So **A1 costs 3× the per-run figure**: ~63 h
at default, ~18 h lean. A2 and Stage B are further multiples.

**The default scenario exceeds the brief's ~10 GPU-hour gate. Per the brief,
this is where I stop for your decision.**

### 5.5 Why a GPU is probably the wrong answer here — and what I could not measure

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

**Recommendation: this workload wants many CPU cores, not a GPU.** Extrapolating
the measured scaling curve (labelled: **extrapolation, not measurement**) a
64-vCPU cloud instance at ~50% efficiency would give roughly 0.3 s/trial,
putting the *default* 28,800-trial run at ~2.5 h and Stage A1 with controls
comfortably inside a working day — without porting anything.

To get the GPU number anyway, the measurement needed is not a microbenchmark
of the matvec but an honest batched-rollout test: port `SteppableRateModel`
to JAX with `jax.experimental.sparse.BCOO`, `vmap` over a population of 32,
and drive it with *recorded* joint trajectories (so no MuJoCo is in the
loop). That gives an upper bound on the GPU path — if it doesn't win there,
it won't win with physics attached. That is ~100 lines and I can write it on
request; it is not written yet because the brief says build nothing first.

---

## 6. Options, if you want to stay inside the 10-hour gate

Ranked by how much I'd recommend them:

1. **Land the lossless 3.4× step** (§5.2). Bit-identical, already measured.
   Costs nothing scientifically. Needs your approval only because it is a
   code change.
2. **Run on a many-core CPU VM** rather than a GPU (§5.5). No port, no new
   risk, biggest single win.
3. **Go to the "lean" budget** (λ=16, 2 episodes, 250 gens). The cost is
   statistical power against per-replicate variance — and Phase 4 showed
   that variance is large, so this is a real trade, not a free one.
4. **Shorten training trials to 2.5 s, validate finalists at 4 s.** ~37%
   saving. 2.5 s still gives ~22 cycles at the measured ~11 Hz rhythm.
5. **RK4 → RK2**, halving matvecs per step for ~2×. **This changes the
   neuron model's numerics and therefore needs explicit approval under the
   brief's own rule.** It is checkable — `validate_steppable_model.py`
   already compares against Pugliese's published output — but the audit
   found the system is genuinely phase-sensitive between any two numerically
   distinct runs, so I would expect this to degrade agreement and would want
   to see the number before relying on it.

**Not recommended: a cropped subnetwork.** I tested it. Neurons not
forward-reachable from the driven set are provably always zero and could be
deleted losslessly — but that is only **846 of 23,532 (3.6%)**; the network
reaches 13,541 neurons within two hops. There is no meaningful free crop,
and any *larger* crop would be a topology change.

---

## 7. What I am not proposing without your approval

- **Short-term adaptation (SFA / synaptic depression)** in the neuron model.
  The brief flags it as needing approval and I agree; it is also the single
  most plausible candidate for making feedback usable, because it would
  break exactly the mechanism Phase 4 measured (a narrow threshold
  distribution crossed all at once). If you want it: config-flagged off by
  default, with a test asserting the disabled path reproduces current
  behaviour bit-identically, and a re-run of the Phase 4 bifurcation to
  report the effect. **Not started.**
- Anything in §6 items 1, 4, 5.
- The trainer itself.

## 8. Open questions for you

1. `phase5/` or the convention-compliant `docs/trained_adapter/`? (§0.2)
2. Which compute route — many-core CPU (recommended), or should I write the
   JAX/GPU upper-bound benchmark first? (§5.5)
3. Which budget row from §5.4?
4. Approve the lossless step optimisation (§5.2)?
