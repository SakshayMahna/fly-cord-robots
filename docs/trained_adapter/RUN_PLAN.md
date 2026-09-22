# Stage A1 run plan — for approval before launch

*2026-09-21, revised 2026-09-22. Nothing launched. Every number below is
measured on this project's own code unless marked as an extrapolation.*

> **Revised after activity matching.** The plan below originally trained
> three conditions (real, C1, C2) at identical budget. Both C1 and C2
> failed the full pre-registered matching sequence — primary matching,
> then the Amendment 2 fallback — and neither has a stable rhythmic regime
> at *any* drive (`RESULTS.md`). Per the pre-registered decision rule, they
> are **not trained**, at any drive, under any label. This is the
> designed outcome of gates committed before the data existed, not a
> decision made after seeing it. **Only the real connectome trains.**

## What runs

**One training run: the real MANC connectome**, at its activity-matched
drive of 382.81 (`PREREGISTRATION.md`, `RESULTS.md`).

| run | network | status |
|---|---|---|
| `real` | the MANC connectome | **trains**, from matched drive 382.81 |
| `C1` | degree- and sign-preserving shuffle | **does not train** — no stable rhythmic regime at any drive |
| `C2` | random recurrent network, matched size/sparsity/signs | **does not train** — no stable rhythmic regime at any drive |

C1 and C2's absence from training **is** their result, not a gap in the
plan — see `RESULTS.md` for the full matching and fallback data. Neither
is a weaker test that was skipped; both went through the complete
sequence and failed the first, most permissive gate in it.

**C3**, the FlyGym CPG and rule-based controllers, is unaffected — it uses
no connectome and is already measured (14.025 and 7.396 mm/s), serving as
the published-baseline comparison for the video regardless of the trained
run's outcome.

## Launch order — committed 2026-09-21, controls resolved 2026-09-22

The original order was real, then C1, then C2, at identical budget
regardless of outcome — precisely so the decision to run all three was
made before any of their results existed, not while reading the first
one. That commitment is what makes it legitimate that C1 and C2 are now
excluded: the exclusion was decided by a rule fixed in advance
(`PREREGISTRATION.md`), applied identically to both, and reached without
either control's activity-matching data existing yet when the rule was
written.

**What actually runs now: the real connectome only.**

## Budget

| setting | value | why |
|---|---|---|
| population λ | 16 | CMA-ES default for 71 parameters is 4 + ⌊3 ln 71⌋ = 16 |
| episodes E | **6** | binding term is the tripod index (DESIGN.md §5.7); progress alone needs only 1 |
| generations | 250 | |
| trial length | 4 s | pre-registered |
| **trials** | **24,000** | 16 × 6 × 250, one run (real connectome only) |

## Wall time and cost

Per-trial cost is measured: **~8.5 s** single-core after the lossless
active-column speedup, so 24,000 trials is **~57 core-hours**.

| cores | η = 0.35 (measured on this Mac) | η = 0.6 (homogeneous server) |
|---:|---|---|
| 16 | ~10 h wall, ~163 core-h billed | ~6 h, ~95 core-h |
| 32 | ~5 h wall, ~163 core-h billed | ~3 h, ~95 core-h |

**Verified pricing (`artifacts/vm_pricing_2026-09-22.md`): DigitalOcean
32 vCPU / 64 GB CPU-Optimized, $1.00/hr**, confirmed against
digitalocean.com directly. At that rate:

| efficiency | wall time (32 cores) | cost |
|---|---:|---:|
| η = 0.35 (this Mac's measured rate) | ~5 h | **~$5** |
| η = 0.6 (homogeneous server, extrapolated) | ~3 h | **~$3** |

Billed core-hours are ~95–163/η regardless of instance size, so a larger
instance buys wall time at roughly constant cost. This is 2.5× smaller
than the three-run budget the plan originally priced, purely because two
of the three planned runs are not happening.

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

1. **VM not rented.** Recommendation: DigitalOcean 32 vCPU / 64 GB,
   $1.00/hr, verified directly against digitalocean.com
   (`artifacts/vm_pricing_2026-09-22.md`). Needs your approval, then a
   calibration run.
2. **Lean-pilot caveat still stands.** The first pilot was stopped at
   generation 120 by its own pre-committed rule. This run is the powered
   one; a negative result from it *is* reportable, unlike the pilot's.
3. **No trainable control remains.** A negative result for the real
   connectome cannot be checked against a trained C1 or C2, because
   neither can be trained at all — that absence is itself the controls'
   result (`RESULTS.md`) and is reported alongside the real run's, not
   silently missing.

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
