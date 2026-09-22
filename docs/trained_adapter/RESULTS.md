# Results — trained adapter

*No training run has happened. This file records findings established
while building and validating the Stage A rig, so they are not lost in the
log.*

## The anatomically named motor pools are near-silent

Measured on the untrained connectome under DNg100 drive, across 6 legs ×
6 replicates:

| motor module | mean rate | fraction of leg-pools active |
|---|---:|---:|
| tibia extend | 0.198 | 0.25 |
| femur/tr extend | 0.196 | 0.17 |
| coxa stance | 0.191 | 0.19 |
| femur/tr flex | 0.068 | 0.36 |
| femur reductor | 0.051 | 0.08 |
| coxa swing | 0.025 | 0.19 |
| tibia flex | 0.001 | 0.03 |
| **substrate grip** | **0.000** | **0.00** |
| **tarsus control** | **0.000** | **0.00** |

The two modules semantically closest to adhesion — `substrate grip`
(47 neurons, 6–9 per leg) and `tarsus control` (15) — **never fire at
all**. The pair the annotation names `coxa stance` / `coxa swing` fires in
roughly one leg-pool in five.

This sharpens Phase 3's recorded limitation that 3 of 9 motor modules are
unmapped: two of those three are not merely unmapped by us, they are
**silent in the model**. Nothing downstream could have used them.

**Consequence for the adhesion interface:** the gate uses the connectome's
**rhythm** (per-leg summed motor rate, high-passed against its own 50 ms
running mean) and **not its stance anatomy**, because the anatomical signal
does not exist to be used. A gate built on `coxa stance` − `coxa swing`
sticks: three legs always-off, one always-on, 0–1 transitions per 2 s where
stepping needs ~44.

**Open question for training to answer.** All four anatomical pools are
logged every step and every generation. Whether a trained adapter ever
recruits them — particularly `substrate grip`, which is what a real fly
would use to hold the substrate — is a stronger and more interesting result
than the adhesion gate itself, and it will be reported either way.

## Baselines on the adopted rig

| controller | mean speed | sd | seeds | min upright |
|---|---:|---:|---:|---:|
| CPG (FlyGym demo) | 14.025 mm/s | 0.250 | 5 | 0.966 |
| rule-based (FlyGym demo) | 7.396 mm/s | 0.392 | 3 | 0.873 |
| **untrained connectome** | **+0.011 mm total** | — | 1 | 0.901 |

The untrained connectome moves its legs energetically — 160 rad of total
joint travel over 4 s — and travels 0.011 mm. It stays upright and does not
flip. That is the honest starting point training has to improve on.

## Both wiring controls oversaturate before training can start

C2 (matched random network) is built and passes every matching test. On
the real connectome it preserves size, sparsity to the edge, sign ratio
exactly (735,914 / 636,490), Dale's law (0 violations), the weight
multiset, and the interface — while rewiring 76.6% of edges and changing
out-degree by 31.1 on average. It is a correctly matched control.

It cannot be trained. Neither can C1.

**Baseline `n_active` under DNg100 drive, adapter off** (stability
threshold 1,500, Pugliese's own oversaturation criterion):

| replicate | real | C1 shuffled | C2 random |
|---:|---:|---:|---:|
| 0 | 513 | 6,726 | 13,532 |
| 1 | 392 | 6,638 | 13,590 |
| 2 | 4,733 | 6,713 | 13,479 |
| 3 | 360 | 6,766 | 13,659 |
| 4 | 505 | 6,908 | 13,751 |
| 5 | 4,481 | 6,808 | 13,679 |
| 6 | 417 | 6,766 | 13,714 |
| 7 | 438 | 6,794 | 13,557 |
| **median** | **471** | **6,766** | **13,624** |
| **eligible replicates** | **6/8** | **0/8** | **0/8** |

**The finding, stated carefully.** Randomising the wiring while holding
size, sparsity, sign ratio, Dale's law and the weight distribution fixed
makes the network explode: 14× the real network's activity for C1 and 29×
for C2, against a drive the real connectome handles stably. The ordering
real ≪ C1 ≪ C2 is itself informative — preserving each neuron's degree
sequence recovers some stability, but nowhere near enough. Whatever keeps
this network bounded is in the specific topology, not in its summary
statistics.

That is a real result about the connectome and it was not designed for;
it fell out of applying the pre-registered filter to the controls with
their own baselines, as the rule requires.

**But it blocks the controls as training conditions**, and a headline
claim with no trainable control is much weaker. `resolve_pool` refuses to
return a pool rather than quietly training on saturated networks — which
is the behaviour it was built for, but it means Stage A cannot proceed as
planned without a decision. Options are set out in `RUN_PLAN.md`.

## Activity matching (pre-registered) — neither control can be trained

Full procedure: `PREREGISTRATION.md`. Target = median adapter-off baseline
`n_active` of the **real** network at its native DNg100 drive of 380,
across replicates 0–7, parameter seed 641, 4.0 s trials.

**Native per-replicate `n_active` (real network, drive 380):**

| replicate | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n_active | 522 | 380 | 4,176 | 367 | 488 | 3,794 | 421 | 445 |

**Target (median) = 466.5.**

### Primary matching — bisection on drive

| network | matched drive | achieved median | converged | verdict |
|---|---:|---:|---|---|
| **real** | 382.81 | 469.0 | **yes** (8 iters) | **PASSES** |
| C1 (shuffled) | 311.52 | 255.5 | no (12 iters, exhausted bracket) | **cannot be activity-matched** |
| C2 (random) | 213.38 | 16.0 | no (12 iters, exhausted bracket) | **cannot be activity-matched** |

The real network's matched drive (382.81) sits almost exactly on its native
drive (380) — the intended no-op check on the procedure passes.

**Real network at its matched drive**, eligible replicates and rhythm:

| replicate | 0 | 1 | 3 | 4 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|
| n_active | 515 | 399 | 367 | 460 | 426 | 478 |
| n_rhythmic (of 6) | 6 | 3 | 4 | 3 | 4 | 6 |

6/8 replicates eligible (2, 5 excluded by the unchanged stability filter,
consistent with their native-drive values), **median n_rhythmic = 4.0**.
**Real network trains from drive 382.81.**

**Both controls fail because their target lies inside a near-discontinuity,
not because the search failed to converge on a real intermediate value:**

| network | just below the jump | just above |
|---|---|---|
| C1 | drive 311.52 → 255.5 active | drive 312.01 → 3,495 active |
| C2 | drive 213.38 → 16 active | drive 214.84 → 6,822 active |

A ~0.5-drive-unit change multiplies activity by **13–430×**. The real
network's own cliff sits well *above* its matched drive (437 vs. 382.81),
giving it a genuine intermediate regime; neither control has one at any
drive near its target.

### Secondary fallback (pre-registered, Amendment 2) — also fails

Highest drive with median `n_active ≤ 1500` **and** median
AR(1)-gated `n_rhythmic ≥ 2` of 6, searched over `[0, matched_drive]`:

**C1** (bracket [0, 311.52], 9 iterations to exhaust to <1.0 width):

| drive | 155.76 | 77.88 | 38.94 | 19.47 | 9.74 | 4.87 | 2.43 | 1.22 | 0.61 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median n_active | 2.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**C2** (bracket [0, 213.38], 8 iterations):

| drive | 106.69 | 53.34 | 26.67 | 13.34 | 6.67 | 3.33 | 1.67 | 0.83 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| median n_active | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Every drive tested for both controls, down to near zero, gives
essentially no activity at all** — not "some activity without rhythm," but
no activity to be rhythmic. Neither network has an intermediate regime
between silent and exploded.

**Verdicts, both networks: "no stable rhythmic regime at any drive."
Neither is trained, at any drive, under any label.**

### What this means for Stage A1

The pre-registered design planned three trained conditions (real, C1, C2)
at identical budget. **That plan cannot be executed as specified**: the
gates that were committed *before* this data existed — unchanged
throughout, applied identically to all three networks — permit training
only the real connectome.

This is not early abandonment of the controls. Both went through the full
sequence in `PREREGISTRATION.md` (primary matching → stability →
rhythmicity, then the Amendment 2 fallback) and failed at the first,
most permissive gate in that sequence. Reporting "cannot be trained" is the
designed outcome of a rule set before either control's data existed, not a
judgement call made after seeing it.

**The finding is itself the headline result of the controls arm**:
preserving size, sparsity, sign ratio, Dale's law and the weight
distribution — while destroying the specific wiring — is sufficient to
eliminate the graded intermediate activity regime the real connectome has.
Real ≪ C1 ≪ C2 in native-drive activity (`RESULTS.md`, above); at matched
target activity, only the real network has *any* drive that reaches it
without either falling silent or exploding. Whatever keeps this network in
a usable dynamic range is a property of the specific topology, not
recoverable from degree sequence, sign balance, or edge weights alone.

**Revised plan: one trained run (real connectome, drive 382.81), not
three.** `RUN_PLAN.md` updated accordingly.
