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

## Comparison basis for every trained rung — stated once, applies throughout

**No connectome control is trainable** (above: both C1 and C2 failed the
complete pre-registered matching sequence, primary and fallback). There is
no trained-C1 or trained-C2 number for Rung 1 or Rung 2 to be compared
against. **The learning comparison for both rungs is against the C3
baselines only** — the FlyGym CPG (14.025 ± 0.250 mm/s) and rule-based
(7.396 ± 0.392 mm/s) controllers, measured on the identical body and
terrain (`baselines/run_baselines.py`).

This limits what any trained result can claim: it can show the adapter (or
the adapter plus a conductor) learned to move the body, and how that
compares to a hand-designed controller on the same body — it cannot show
that the specific wiring was *necessary* for that, since no wiring control
could be trained to test against. Recorded here so it is read alongside
the numbers, not discovered as a footnote later.

---

# Rung 1 (R1a, output-stage) — result

**The trained adapter does not walk.** Reported as the outcome of the
pre-registered stopping rule (`RUNBOOK.md` §6: flat `term_progress` by
generation 60–80 is a result, not a reason to keep waiting), not as a run
that was abandoned early.

## What was run

Three ground runs, **14,784 episodes** in total. The first two are recorded
because each ended for a reason that is itself a finding, not because they
are comparable attempts:

| run | gens | episodes | outcome |
|---|---:|---:|---|
| `r1a_ground` | 41 | 4,032 | invalid objective — the reward forbade walking (`GROUND_BRIDGE.md` §9) |
| `r1a_ground_v2` | 38 | 3,648 | same, plus a drive bound that put 1σ past the saturation cliff |
| **`r1a_ground_v3`** | **74** | **7,104** | **the valid run** — corrected reward, 3.43 h |

Only v3 is evidence about the connectome. In v1 and v2 the objective was
broken: the gait known to walk scored −15.95 against 0.00 for standing
still, so both runs converged on inaction by playing correctly.

## The result (v3)

| | value | reference |
|---|---:|---|
| best score | +0.0501 (gen 49) | — |
| median displacement | +0.009 mm / 4 s | — |
| **fastest episode ever** | **0.627 mm/s** | **6.4% of the restricted CPG's 9.822 mm/s** |
| episodes exceeding 1 mm | 29 of 7,104 (**0.41%**) | — |
| `term_progress`, gens 0–20 / 20–40 / 40–60 / 60–74 | −0.00001 / +0.00039 / −0.00003 / +0.00045 | no trend |

Progress oscillates around zero across the whole run. The best candidate
moves its legs visibly (unlike v1's motionless one) and travels 0.11–0.19 mm
in 4 s — real motion, but ~0.3–0.5% of a walking gait.

## Why — the mechanism, measured

Diagnosed in `GROUND_BRIDGE.md` §10. Three facts, in order of how binding
they are:

1. **Never six live legs.** `lm` and `rf` carry *identically zero* motor
   rate in the best candidate, and both belong to the same tripod group, so
   tripod alternation is structurally impossible for it. Sweeping drive
   343.5 → 392.0 never recovers all six (4/6, 5/6, 5/6, 5/6).
2. **No stance/swing cycling, so no traction.** The working CPG grips at a
   uniform ~65% duty, cycling 12 Hz (95–96 transitions per leg per 4 s).
   Trained candidates either barely grip (0.008–0.207) or pin legs
   permanently down (duty 1.000). **The bound is not the obstacle** — the θ
   giving 65% duty (−2.45, −1.91, −1.25, −0.04) lies well inside the
   searchable [−9, 9]. The search could reach it and did not, because
   gripping only pays once the rest of the gait is coordinated.
3. **What it does instead.** With traction unavailable, the only strategy
   that scores is asymmetric drive that pivots the body for a small net
   +x. `command_asymmetry` is pinned at 95% of its range, and **62% of
   episodes that move travel further sideways than forward** (median
   |dy|/|dx| = 1.35).

21 of 40 trained parameters sit at or within 8% of a bound, coherently:
`motor_scale` maxed on middle/hind entries (flattening `tanh`, suppressing
those legs), `motor_offset` pinned at ±0.29 of ±0.30 on five entries. The
search switches legs off and braces rather than tuning a gait.

## What this does and does not establish

**Does.** Through this interface — a frozen connectome, DNg100 tonic drive,
an antagonist-pair motor decoder over 18 DOFs, and a motor-derived adhesion
gate — the decoded output does not animate six legs simultaneously and does
not produce stance/swing cycling, so it cannot generate traction. That is
consistent with the literature the project is built on: Pugliese et al.
report **no consistent left/right phase coupling** under DNg100 drive
(Fig. 4d–e, Ext. Data Fig. 9b) and T3 as least robust.

**Does not.** It does not show the connectome *cannot* drive walking. The
interface is narrow by construction (18 of 42 DOFs; 3 of 9 motor modules
mapped, two of which are silent in the model — see the top of this file),
the sensory path was off by design in R1a, and no interleg coordination was
supplied. It also says nothing about necessity of the wiring: no connectome
control is trainable, so the only comparison is against the C3 baselines.

**Not attempted: rescuing this by reward design.** Forcing adhesion duty
toward 65%, prescribing per-leg phase, penalising asymmetry, or rewarding
tripod index directly would each hand the animal the answer being tested
for (`GROUND_BRIDGE.md` §11). Coordination, if supplied, belongs in **Rung
2** where it is labelled as our engineering addition with a K=0 ablation —
not hidden in the objective.

## Artefacts

* `media/trained_adapter/r1a_ground_v3/` — checkpoint (gen 73), history,
  per-generation episode records
* `media/trained_adapter/r1a_v3_best_r{0,1,3}_{top,side}.mp4` — the best
  candidate, rendered
* `media/trained_adapter/restricted_action_cpg.json` — the feasibility
  reference (18-DOF CPG at 9.822 mm/s)
