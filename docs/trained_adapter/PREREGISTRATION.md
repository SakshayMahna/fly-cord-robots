# Pre-registration — activity matching for the wiring controls

*Written 2026-09-21, **before the matching search was run**. Committed
first so that the procedure, the tolerance, the seeds and every decision
rule are fixed in advance and applying them later is a computation, not a
judgement made while looking at the answer. Same discipline as
`docs/closed_loop/PREREGISTRATION.md` §B2, which is the only reason Phase
4's stopping rule is reportable.*

## Why this exists

At the native DNg100 drive of 380, the two wiring controls oversaturate
and cannot be trained at all (`RESULTS.md`): median baseline `n_active` is
471 for the real connectome, **6,766** for C1 and **13,624** for C2,
against a stability threshold of 1,500. That ordering is reported as a
finding in its own right and the filter is **not** relaxed.

Comparing a stable network against two exploded ones would not be a
controlled comparison, so drive is matched first. The matching is
automatic, identical for all three networks, and applied to the real
network too — where it should be close to a no-op, which is itself a check
that the procedure is not doing something strange.

## The procedure

**Target.** `T` = median adapter-off baseline `n_active` of the **real**
network at its native drive of 380, over all eight candidate replicates at
parameter seed 641, 4.0 s. Computed once and recorded.

**Search.** Bisection on DNg100 drive over `[0, 2000]`, minimising
`|median n_active(drive) − T|`, where the median is taken over the same
eight replicates with the same parameter seed.

Bisection is valid because baseline `n_active` is monotone non-decreasing
in drive — verified before this was written, for all three networks:

| drive | 0 | 25 | 50 | 100 | 200 | 380 |
|---|---|---|---|---|---|---|
| real | 0 | 0 | 0 | 0 | 4 | 392 |
| C1 | 0 | 0 | 0 | 0 | 4 | 6,638 |
| C2 | 0 | 0 | 0 | 0 | 4 | 13,590 |

**Tolerance.** Converged when `|median − T| / T ≤ 0.05`, or when the
bracket narrows below 1.0 drive unit — whichever comes first. Maximum 30
iterations.

**Seeds.** Replicates 0–7 at parameter seed **641** (the pilot seed set).
The main-experiment seed 20260919 is not used here and stays unspent.

**Recorded per network:** the matched drive, the achieved median, the
per-replicate `n_active`, the iteration count, and whether it converged.

**No other per-condition settings.** Drive is the only thing matched. The
reward, the adapter, the budget, the trial length, the stability filter
and the rhythmicity criterion are identical across networks.

## Decision rules — committed now

Applied in order to each network after matching:

1. **Reachability.** If no drive in `[0, 2000]` brings the median within
   tolerance, the network **cannot be activity-matched**. Report that as
   its result. Do not train it.
2. **Stability.** Apply the unchanged pre-registered filter
   (`n_active ≤ 1500`) per replicate at the matched drive. If no replicate
   is eligible, report that. Do not train it.
3. **Rhythmicity.** At the matched drive, compute AR(1)-gated `n_rhythmic`
   per eligible replicate (`analysis/interleg_coordination.py`, unchanged).
   Require **median `n_rhythmic` ≥ 2 of 6**.

   Two legs is the threshold because interleg coordination is the question
   being asked, and with fewer than two rhythmic legs there is no interleg
   phase relationship to measure at all. The number is set here, before the
   controls' values are known, and deliberately not calibrated against
   them.

   A network below it has **no significant rhythm at its matched drive**.
   Report that as its result. Do not train it.
4. **Training.** Every network passing 1–3 is trained from its own matched
   drive, at the same budget, with the same reward. The adapter's
   `dng100_level` parameter is trainable **relative to that start**: its
   default becomes the matched drive rather than 380, so each network
   begins at its own activity-matched operating point and the optimiser
   can move from there.

## What a failure at each step means, stated in advance

- **Fails reachability or stability:** the wiring cannot be driven into a
  usable regime at any drive. That is a strong statement about the wiring
  and is reported as such.
- **Fails rhythmicity:** the wiring can be driven to the right *amount* of
  activity but produces no rhythm. This is the more interesting failure,
  because it separates "how much activity" from "what structure the
  activity has" — and the latter is what the project is actually about.
- **Passes everything:** the control is a fair comparison and is trained.
  A control that then matches the real connectome's trained performance
  would be a serious negative result for the headline claim, and would be
  reported plainly, in those words.
