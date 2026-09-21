# Reward — proposal for approval

*2026-09-21. **Nothing here is implemented.** This is the decision doc the
trainer is gated on: no training run happens until these weights are
approved and frozen. Numbers come from `DESIGN.md` §5.7, measured at the
frozen Phase 4 defaults on the tethered-ball rig.*

**Audience:** the project owner, as the approver.

---

## 0. What this is, in the project's own terms

**The reward is entirely OUR ADDITION.** There is nothing in the fly, in the
connectome, or in Pugliese et al. that says what a good gait is. The
connectome supplies dynamics; the body supplies mechanics; the reward is a
value judgement we are imposing, and it is the single most editorialising
component in the whole project. It is therefore written down, weighted, and
frozen *before* any run, rather than tuned until the results look good.

The honesty rule that matters most here: **the reward must never be adjusted
after seeing a result, per condition, or between the real connectome and its
controls.** C1 (shuffled) and C2 (random) are scored with the identical
frozen reward. A control given a different objective is not a control.

---

## 1. Why every term needs a measured scale

Weighting terms with different natural spreads is how a reward secretly
optimises one term. Measured at the untrained defaults
(`DESIGN.md` §5.7, stability-filtered pool of 6 replicates):

| observable | mean | SD |
|---|---:|---:|
| ball pitch (progress) | +0.0049 rad/s | 0.0066 |
| ball roll | 0.060 rad/s | 0.035 |
| ball yaw | 0.085 rad/s | 0.019 |
| rhythmic legs | 4.00 of 6 | 0.89 |
| tripod index | −0.403 | 0.238 |
| joint excursion | 0.0018 rad | 0.0003 |
| n_active | 917 (367–3,794) | — |

Each term below is normalised by a **stated reference**, so that a term of
1.0 means something specific and the weights are dimensionless.

---

## 2. THE SIGN PROBLEM — read this before anything else

> **Forward walking is NEGATIVE pitch.** `PREREGISTRATION.md` §2, verbatim:
> "under a synthetic tripod drive the ball turns about the **pitch** axis
> **(−0.95 rad/s)** with roll and yaw near zero — i.e. straight-line forward
> walking."
>
> So the progress term must reward **−pitch**. Get this backwards and the
> trainer will learn to walk **backwards**, and the scalar score will look
> like success the whole time. This is the single most likely way for this
> reward to be silently wrong.

Two things follow, both required before the first run:

1. **A sign test, not a comment.** Drive the body with the same synthetic
   tripod gait the ball rig was verified with, assert the progress term
   comes out **positive**, and assert a time-reversed drive comes out
   negative. This is cheap and it is the only thing standing between us and
   a confidently-reported backwards-walking fly.
2. **The −0.95 reference is from a synthetic gait, not the connectome.** It
   is a calibration constant, and it is ours. The untrained connectome sits
   at +0.005 rad/s — i.e. essentially stationary, marginally backward.

---

## 3. The terms

Evaluated over the trial after a fixed **0.5 s transient discard**, matching
Phase 4's convention exactly.

| # | term | definition | reference | weight |
|---|---|---|---|---:|
| 1 | progress | `−mean(ball_pitch) / 0.95` | measured straight-walking rate | **+1.00** |
| 2 | straightness | `−(mean|roll| + mean|yaw|) / 0.95` | same | **−0.20** |
| 3 | rhythmicity | `n_rhythmic / 6` | all six legs rhythmic | **+0.30** |
| 4 | coordination | `(tripod_index + 1) / 2` | mapped from [−1,1] to [0,1] | **+0.20** |
| 5 | posture | `−mean|joint − rest| / 0.30 rad` | the encoder's own `POSITION_REF_RAD` | **−0.10** |
| 6 | energy | `−mean(Σ Δtarget²) / E_ref` | untrained baseline | **−0.05** |
| 7 | **saturation** | hinge, see §4 | Pugliese's 1,500 criterion | **−2.00** |

### Rationale for the relative weights

**Progress dominates (1.00)** because the claim being tested is "drives a
body," and walking is the thing. Everything else is a shaping or safety
term and is deliberately smaller, so that no combination of them can
outweigh actually moving.

**Rhythmicity (0.30) and coordination (0.20) are shaping terms, not the
objective.** They matter because Phase 4 established the rhythm is the
connectome's real output and a gait that loses it has lost the thing we are
driving the body with. They are kept well below progress on purpose: if
coordination were weighted comparably, a candidate could score well by
producing a beautifully coordinated gait that goes nowhere, and we would
have rewarded ourselves for the connectome's *existing* behaviour rather
than for it driving a body.

**Straightness (−0.20) is small** because the ball rig is forgiving and
punishing yaw too hard would suppress turning, which Stage A3 needs.

**Posture and energy are small (−0.10, −0.05).** They exist to stop
degenerate flailing, not to shape gait. The measured baseline excursion is
0.0018 rad against a 0.30 rad reference, so this term is ~0.006 at rest —
it only bites on genuinely extreme postures.

**Saturation is the largest single coefficient (−2.00)** and §4 explains
why it has to be.

---

## 4. The saturation penalty

```
saturation_penalty = clip((n_active − 1500) / 1500, 0, 2)
```

Zero below Pugliese's own documented oversaturation criterion; rising
linearly above it; capped at 2 (i.e. n_active ≥ 4,500) so a single
catastrophic trial cannot dominate a candidate's averaged score by an
unbounded amount.

**Why it must outweigh progress, with a measurement rather than an
intuition.** `DESIGN.md` §5.7 measured replicate 5, which is saturated at
`n_active = 3,794`, producing **ball pitch +0.471 rad/s — 30× any healthy
replicate.** A seizing network thrashes the legs and spins the ball. Under
the progress term alone that is worth `0.471/0.95 ≈ 0.50` if the sign
favoured it.

So: a saturated network is not merely scientifically wrong, it is
**instrumentally attractive** to an optimiser. At `n_active = 3,794` the
penalty is `−2.00 × 1.53 = −3.06`, which swamps any progress score
reachable that way. That margin is the point of the weight, not a
round number.

Three further reasons this term carries the most weight:

- It encodes Phase 4's entire finding. The pilot's result is that feedback
  either does nothing or drives the network into runaway; the trainer must
  be able to *discover* that constraint rather than be handed a gain
  ceiling. The penalty is how it feels the wall.
- It is the difference between a 27×-noisier and a 27×-cleaner search
  (§5.7). Saturated trials dominate score variance.
- Saturated trials also cost ~2.5× more wall time, because they fall back to
  the dense matvec path (§5.2).

**Stated plainly as a limitation:** this term is a strong prior that
saturation is bad. It is justified by Pugliese's own criterion and by
Phase 4's results, but if the trained adapter's *only* route to locomotion
turned out to run through high `n_active`, this reward would forbid it and
we would never see it. That is an accepted, deliberate trade, and it is
recorded here so it cannot later be discovered as a surprise.

---

## 5. How this could be gamed, and what stops it

Written before the run, because reward hacking found afterwards is
indistinguishable from a result.

| exploit | what stops it | is that sufficient? |
|---|---|---|
| spin the ball by seizing | saturation penalty (§4) | yes, by ~6× margin — measured |
| walk backwards | the sign test (§2) | yes, if the test exists |
| one leg paddling the ball | rhythmicity + coordination terms | **partly** — worth watching |
| freeze and score 0 | progress is the dominant positive term | yes |
| buzz the joints for "rhythm" | rhythm gate is AR(1)-tested, in-band 2–20 Hz | yes — reuses Phase 4's gate |
| exploit ball inertia with one shove | mean over 3.5 s, not net displacement | yes |

**Per-term logging is mandatory** (`DESIGN.md` §5b): every generation logs
each term separately, not just the scalar. A single number cannot show which
term is being optimised, and that is exactly what reward hacking looks like
from the outside.

---

## 6. Freezing

The reward's full definition — every weight, reference and threshold — is
serialised and **sha256-hashed**, the same discipline as the frozen motor
interface's `04be9dec…`. The hash is written into every checkpoint, and the
resume path refuses a mismatch rather than warning
(`DESIGN.md` §5b). Rationale: a reward that changes mid-run silently mixes
two objectives, and it would not be visible in the learning curve.

If a weight must change after approval, that is an **amendment** — dated,
with the old values kept — per the convention `docs/closed_loop/` already
uses. Never a silent edit.

---

## 7. What I need from you

1. **Approve or adjust the seven weights** in §3.
2. **Confirm the saturation cap** at 2.0 (n_active ≥ 4,500). It bounds how
   much one catastrophic episode can hurt a candidate averaged over E
   episodes.
3. **Stage A2 will need a survival/termination term** once the body can
   fall over. Not proposed here — A1 is on the ball and cannot fall. Flagging
   so it is an amendment later rather than a surprise.

Open question I could not settle alone: whether **E = 6** (§5.7) or the
lean pilot's **E = 2** should be used for the *controls*. Matching the real
run is the clean answer, but C1/C2 are where a false negative is cheapest
to make and most damaging to the headline claim.
