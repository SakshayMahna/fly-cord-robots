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

---

# Amendment 1 — monotonicity claim corrected (2026-09-21)

**Written while C1's search was running, before C2's began.**

The procedure above justifies bisection by asserting that baseline
`n_active` is "monotone non-decreasing in drive — verified before this was
written, for all three networks". **That claim is broader than what was
actually tested and is wrong as stated.**

What was tested: drives {0, 25, 50, 100, 200, 380} on **replicate 1 only**.
Within that range it is monotone for all three networks.

What the search then measured, above that range:

| network | drive → median `n_active` |
|---|---|
| real | 406 → 582, but **437 → 3,991** |
| real | **500 → 4,015**, but **1000 → 2,886** |
| C1 | 310.55 → 248, but **312.50 → 3,496** |

So activity is **monotone below the bifurcation and not above it**, and the
transition is a near-discontinuity rather than a steep slope. That is the
same regime structure Phase 4 established — a cliff with no graded middle —
showing up in a different measurement.

**Does this invalidate the search?** No, and the reason is specific rather
than reassuring: the implementation keeps the best-error candidate seen at
any iteration rather than trusting the bracket, and it reports
`converged` from that error against the tolerance. A network whose target
lies below its cliff converges correctly (the real network did, to 0.5%).
A network whose target lies *inside* its discontinuity exhausts the bracket
and is reported as unconverged — which is the honest answer, not a search
failure to be worked around.

The corrected claim, for the record: **bisection is valid on the
sub-bifurcation branch, which is where any usable operating point lies.**

---

# Amendment 2 — secondary fallback for unmatched controls (2026-09-21)

**Written before C2's search began, and before C1's verdict was known.**

A control that cannot be activity-matched would otherwise be reported and
never trained, leaving the headline claim with fewer trained controls than
planned. That is a real cost, so a secondary comparison is added.

**It does not replace the primary result.** "Cannot be activity-matched"
remains the primary finding for any control that fails reachability, and is
reported first and as such.

## The fallback

For any control failing reachability, also train it at its **highest stable
sub-cliff drive**, defined as the largest drive satisfying **both**:

1. median adapter-off baseline `n_active` ≤ **1,500** (the unchanged
   pre-registered stability threshold), and
2. median AR(1)-gated `n_rhythmic` ≥ **2 of 6** across eligible replicates
   — the same rhythmicity criterion, unchanged.

**Search:** bisection on drive over `[0, D]`, where `D` is the lower edge of
the bracket the primary search exhausted, using the same seeds (replicates
0–7, parameter seed 641), the same 4.0 s duration, the same 1.0-drive-unit
minimum bracket and the same 30-iteration cap. The objective is the largest
drive meeting both conditions, not a target value, so tolerance does not
apply; the search returns the largest passing drive found.

**Labelling, mandatory and verbatim wherever it appears:**
**"not activity-matched; secondary comparison"**. Any figure, table or
video caption showing this control carries that label. It is not to be
described as a matched control, because it is not one — it runs at lower
total activity than the real network, and a weaker result from it could be
the activity difference rather than the wiring.

**If no drive satisfies both conditions**, the control has no stable
rhythmic regime at any drive. Report that; do not train it. That is a
stronger statement about the wiring than the matching failure alone.
