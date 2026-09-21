# Stage A1 run plan — for approval before launch

*2026-09-21. Nothing launched. Every number below is measured on this
project's own code unless marked as an extrapolation.*

## What runs

Three training runs, **identical in every respect except the connectome**:

| run | network | purpose |
|---|---|---|
| `real` | the MANC connectome | the treatment |
| `C1` | degree- and sign-preserving shuffle, interface edges protected | is it the wiring? |
| `C2` | random recurrent network, matched size and sparsity | **needs building** |

Plus **C3**, a sine/CPG floor controller on the same body and reward —
much cheaper, no connectome. The FlyGym CPG and rule-based controllers are
already ported and measured, and serve as the published-baseline
comparison (14.025 and 7.396 mm/s).

**Controls get the same budget, the same episodes, the same generations
and the same reward as the treatment.** A control given a weaker test is
not a control.

## Budget

| setting | value | why |
|---|---|---|
| population λ | 16 | CMA-ES default for 71 parameters is 4 + ⌊3 ln 71⌋ = 16 |
| episodes E | **6** | binding term is the tripod index (DESIGN.md §5.7); progress alone needs only 1 |
| generations | 250 | |
| trial length | 4 s | pre-registered |
| **trials per run** | **24,000** | 16 × 6 × 250 |
| **total** | **72,000** | three runs; C3 adds ~2% |

## Wall time and cost

Per-trial cost is measured: **~8.5 s** single-core after the lossless
active-column speedup, so 24,000 trials is **~57 core-hours per run** and
**~170 core-hours for all three**.

| cores | η = 0.35 (measured on this Mac) | η = 0.6 (homogeneous server) |
|---:|---|---|
| 16 | ~30 h wall, ~486 core-h billed | ~18 h, ~283 core-h |
| 32 | ~15 h wall, ~486 core-h billed | ~9 h, ~283 core-h |

**Cost: roughly $9–25 for the whole of Stage A1** at an indicative
$0.03–0.05 per vCPU-hour. That figure is from memory and **must be checked
against current pricing before committing spend** — but the decision here
is wall time, not money. Even a 4× pricing error leaves this inexpensive.

Billed core-hours are 170/η regardless of instance size, so a larger
instance buys wall time at roughly constant cost.

**Recommended: 32 vCPU, ≥ 32 GB RAM.** RAM is set by the measured 0.57 GB
steady-state per worker (32 workers ≈ 18 GB + OS), **not** by the 7.4 GB
transient build peak — provided pool startup is serialised, which the
trainer does. Thirty-two workers building concurrently would want ~235 GB
and will OOM; that is the single most likely way to waste the rental.

**First ten minutes go to a calibration run, not training.** The 16/32-core
figures are extrapolated from a heterogeneous 11-core laptop and carry
about a factor-of-two error bar. `bench_population_throughput.py` settles
it in ~10 minutes on the actual VM.

## What is logged

Per generation: best / median / worst, **per-candidate term breakdown**
(not just population means — the gap that stopped me explaining the gen-15
spike in the first pilot), sigma, saturation fraction, dense-matvec
fallback fraction, flip/termination fraction, adhesion duty per leg, and
the four **anatomical pool rates** (`coxa stance`, `coxa swing`,
`substrate grip`, `tarsus control`) so RESULTS can say whether training
ever recruits them.

Checkpoint every generation, atomically, carrying the reward config hash
`0d64854215…`; resume refuses a hash mismatch.

## Before believing any score

**Inspection video of the top candidates, every time.** The first pilot's
best candidate scored 94% of reference walking speed by kicking the ball
once and holding still. The detectors and the rendered clips exist
precisely because a scalar cannot tell walking from an exploit. Nothing
gets reported as a result until it has been watched.

## Gates already passing

| | |
|---|---|
| CPG baseline walks | 14.025 ± 0.250 mm/s, upright ≥ 0.966 |
| rule-based walks | 7.396 ± 0.392 mm/s |
| kick-then-freeze | coasts +0.0% of walking |
| flips terminate, never pay | −0.7500 vs +0.0375 |
| neural at g_fb=0 unchanged by rig | motor traces bit-identical |
| adhesion not stuck, either polarity | duty 0.287 / 0.295, 0 legs stuck |
| no bypass | targets and adhesion depend only on connectome output |
| suite | **70 tests pass** |

## Blocking items before launch

1. **C2 does not exist.** A random recurrent network with matched size and
   sparsity still needs building and testing. C1's shuffle already exists.
2. **VM not rented.** Needs your approval, then a calibration run.
3. **Lean-pilot caveat still stands.** The first pilot was stopped at
   generation 120 by its own pre-committed rule. This run is the powered
   one; a negative result from it *is* reportable, unlike the pilot's.

## Honest expectations

The untrained connectome travels **0.011 mm** in 4 s while moving its legs
160 rad. The gap to the CPG baseline's 14 mm/s is three orders of
magnitude. Phase 4 established that proprioceptive feedback through this
connectome is either negligible or destructive across four hand-designed
encoders; Stage A asks whether a *trained* interface finds the narrow
regime those missed. It may not. A clean negative — with controls at equal
budget and the inspection video to prove the top candidate was not
exploiting the reward — is a publishable result for the video and is the
outcome the design is built to support.
