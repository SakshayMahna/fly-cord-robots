# Rung 2 — trainable interleg conductor: design proposal

*2026-09-22. **Proposal only. Nothing built, nothing launched.** Answers
the "DESIGN FIRST" requirement before any code exists. Matches
`DESIGN.md` §2.4 ("interleg phase coupling — strength, preferred phase
offset, time constant"), which is what "Rung 2" turned out to name.*

**Audience:** the project owner, as approver before any of this is built.

---

## 1. What the conductor has to do

Phase 4 and the untrained-connectome measurements agree: individual legs
can be rhythmic (median n_rhythmic 4/6 at the real network's matched
drive, `RESULTS.md`) without being **coordinated** with each other — the
paper's own finding (Fig. 4d-e, no consistent left/right phase coupling)
reproduced here. Rung 1 asks whether a trained interface alone fixes that.
Rung 2 asks the next question: with a small, labelled, bounded nudge
toward a tripod relationship, does frozen connectome dynamics do the rest?

Two sub-problems, and they are genuinely separate:

1. **Reading** — get a usable estimate of each leg's rhythm phase, online,
   from data available so far in the trial (no look-ahead).
2. **Writing** — turn a phase *error* (how far this leg is from where the
   tripod pattern wants it) into something that changes the connectome's
   behaviour, without violating no-bypass and without triggering the
   seizure Phase 4 already characterised.

---

## 2. Reading: an online phase estimate

The existing rhythm/phase code (`analysis/interleg_coordination.py`) is
**offline** — bandpass-filters and Hilbert-transforms the *whole*
post-transient window at once. It cannot run inside the step loop; a
sensory encoder or a conductor cannot see the future.

**Proposed:** reuse the same high-pass construction already built and
tested for the adhesion gate (`AdhesionGate`, `apply.py`) — a per-leg
signal minus its own exponential running mean (time constant τ, one of
the three trainable parameters) — as the input to a simple **quadrature
phase estimator**: a second running signal, low-pass filtered with a
90°-lag relationship to the first at the measured dominant frequency
(~11 Hz, `AUDIT.md` §4), giving an analytic-signal pair
`(x, x_quad)` whose `arctan2` is an online phase estimate. This is the
standard causal approximation to the Hilbert transform (a fixed
approximate-quadrature filter rather than a whole-signal FFT) and needs no
future data.

**This needs validation before it is trusted**, and that validation is
part of what building Rung 2 means: compare the online estimate against
the offline Hilbert phase on recorded traces from Rung 1, on the same
signal, and report the agreement (e.g. circular correlation) before using
it to drive anything. Flagged now rather than assumed.

---

## 3. Writing: three placements, as you named them

### Option A — small current into the CPG triad neurons

**Where:** the 18 `CPG_excit_hub` / `CPG_excit2` / `CPG_inhib` neurons (3
per leg), already identified from wiring and independently confirmed
against real published dynamics in Phase 1 — not a guess about which
neurons matter.

**Mechanism:** the same functional form already ported and validated for
the CPG baseline's own leg-to-leg coordination
(`baselines/cpg.py: calculate_ddt`, itself FlyGym's own Kuramoto-style
oscillator coupling) — a correction proportional to
`sin(phase_j − phase_i − target_offset)`, summed over the other five legs
weighted by whether they are in the same tripod group, converted to a
small current added to that leg's CPG triad.

**Pros:** genuinely intervenes on the circuit that generates the rhythm,
not just its readout — the closer of the three to "does the connectome's
own dynamics coordinate, given a nudge." Uses a real, previously-verified
part of the connectome as the target, not an arbitrary one.

**Cons — and this is the one that matters.** Phase 4's entire pilot result
is that *any* added current here is dangerous: sensory current thirty
times below threshold on average still found a bifurcation at the tail.
Injecting current into three-neuron CPG triads that are themselves the
core oscillator is closer to the failure mode than the sensory pathway
was. Measured now, to bound it rather than guess: CPG-triad firing rates
at the real network's matched drive sit at **mean 1.9–2.3 Hz, threshold
median 35.6** — i.e. large headroom in isolation, but Phase 4's lesson was
that the danger is in *near-simultaneous* crossing across a pool with a
narrow threshold spread, not the average distance to threshold. Requires
the mandatory bound-verification in Rule 4 to be taken seriously, not
pro forma.

### Option B — per-leg drive modulation

**Not structurally available as a distinct option.** DNg100 is a bilateral
pair (rows 59, 282 — left/right, not per-leg); there is no per-leg
"master drive" neuron to modulate independently of the CPG triad itself.
A "per-leg drive" that actually reaches only one leg's rhythm generator
**is** Option A under a different name — it would have to act through the
same CPG triad or invent a channel the connectome doesn't have. Not
proposed separately.

### Option C — timing adjustment in the motor decoder

**Where:** downstream of the connectome entirely, in
`adapter_joint_targets` — a phase-dependent delay or advance applied to
each leg's joint-target output, computed from the same online phase
estimate, nudging when a leg's *output* lands rather than when its
*neurons* fire.

**Pros:** by construction, **cannot** change `n_active`, firing rates, or
anything about the connectome's dynamics — it only shifts a readout that
was already going to happen. The seizure risk Rule 4 exists to bound is
structurally absent, not merely tested for. Simpler no-bypass argument:
reads motor-neuron rates (already permitted), writes only to joint
targets (already the decoder's whole job).

**Cons:** weaker as a test of the actual hypothesis. If this works, the
honest conclusion is "the *output mapping* can be timed into a tripod
pattern," not "the connectome's own dynamics coordinate legs given a
nudge" — the neurons never received any input that could make them
genuinely synchronise. It is closer in spirit to hand-building a gait in
the interface, the thing §2's per-segment-not-per-leg sharing rule exists
to avoid, though it is milder since it does not touch amplitude, only
timing, and is still bounded, trainable and small.

## 4. Recommendation

**Option A, strictly bounded, with Option C recorded as the fallback if A
cannot be built without triggering saturation even at small coupling.**

Reasoning: Option A is the one that actually tests the question Rung 2
exists to ask. Option C would answer a real but different question, and
answering it instead of A without saying so would misdescribe the result.
The risk Option A carries is exactly the risk Rule 4 is written to bound
and verify — not a reason to avoid the honest version of the test, but the
reason the bound-verification step is mandatory rather than a formality.

**Concrete proposal if approved:**

```
phase_error(leg) = sin(mean_j w(leg,j) * (phase_j - phase_leg - target(leg,j)))
current(leg, neuron) = clip(coupling_strength * phase_error(leg), -cap, +cap)
```

applied additively to the three CPG-triad rows for that leg, where
`w(leg, j)` is +1 for legs in the same tripod group as `leg`, −1 for the
other group (mirroring the CPG baseline's own `phase_biases` construction,
`baselines/cpg.py`), and `target(leg, j)` is 0 within a group and the
trainable **preferred phase offset** (defaulting to π, the tripod
relationship) between groups.

## 5. Parameters — 3, matching the original scope

| parameter | role | bounds (proposed) | default |
|---|---|---|---:|
| `coupling_strength` | global K in the phase-error equation | 0 to a measured-safe max | **0** |
| `preferred_phase_offset` | target phase relationship between tripod groups | 0 to 2π | π (tripod) |
| `phase_tau` | time constant of the online phase estimator | bounded by the measured ~11 Hz rhythm period | tied to the measured period |

`coupling_strength` starting at exactly 0 is what makes Rung 2 contain
Rung 1 as a special case (Rule 2). The upper bound on `coupling_strength`
and the current cap are **not yet set** — they need the same
measure-first treatment as every other bound in this project (e.g.
`SENSORY_CURRENT_CAP` in Phase 4), which means running the untrained
connectome with a swept fixed coupling and finding where `n_active`
starts to move, before picking a number. Not done; flagged as the first
thing to do if this design is approved.

## 6. Tests required (not written)

Per your four rules, in the order they'd be written:

1. **Coupling = 0 reproduces Rung 1 bit-identically** — same discipline as
   every other zero-effect gate in this project (`g_fb = 0`,
   `active_column_matvec`): `np.array_equal`, not a tolerance.
2. **Nonzero coupling measurably changes interleg phase** — a positive
   control. Run at a deliberately large (test-only, not the trained
   default) coupling and confirm the tripod index moves; a conductor that
   does nothing at any coupling would otherwise pass gate 1 by being
   inert rather than correct.
3. **Bounded, verified at the bound** — run at `coupling_strength` and the
   current cap both at their maximum permitted values and assert
   `n_active` stays under Pugliese's 1,500 criterion across the eligible
   replicate pool. This is the test Rule 4 asks for; it is a measurement,
   not a proof, and the bound should be set low enough that this test has
   real margin, not just barely passes.
4. **No-bypass** — the conductor's current depends only on motor-neuron
   rates (already-computed, already-permitted signal) and the trainable
   parameters; never on `thorax_pos`, `thorax_quat`, joint angles, or
   anything body-derived. Same structural-gate pattern as
   `test_adhesion_depends_only_on_connectome_output`.

## 7. What is identical to Rung 1

Per Rule 5: adhesion gate, reward, `v_ref`, saturation penalty, stability
filter, eligible replicate pool, episode count, population, generations,
seeds. Only the adapter parameter count changes (71 → 74) and the neural
input gains one more additive term, in the same place `command_current`
already adds one (an additive term in `total`, before the tanh
nonlinearity — `steppable_rate_model.py`'s `_derivative`).

---

## 8. Causal phase estimator — validation FAILS, blocking regardless of option

*2026-09-22. Measured against your instruction to validate before the
conductor uses it.*

`fly_robot/adapter/causal_phase.py` implements the estimator proposed in
§2: trailing-window (300 ms) bandpass + Hilbert, recomputed every 10 ms,
using the same transform as the established offline method
(`analysis/interleg_coordination.py`) so the comparison is like-for-like.

**Result** (`validate_causal_phase.py`, real network at matched drive
382.8125, pilot replicates {0,1,3,4,6,7}, 17 rhythmic-leg instances):

| | value |
|---|---:|
| mean circular correlation | **+0.301** |
| minimum circular correlation | **−0.251** (anti-correlated) |
| lag range at "best fit" | −300 ms to +300 ms, incoherent across legs |

**Verdict: NOT YET GOOD ENOUGH.** Two legs' best-fit lag pinned at the
search boundary even after the range was widened from ±60 ms to ±300 ms,
meaning the correlation doesn't have a clean peak at any reasonable delay
— the two phase series are not simply a delayed copy of each other, they
diverge.

**Diagnosed, not just observed.** Two quick variants were tried on the
cleanest replicate (0, 6/6 rhythmic, dominant frequency 11.4–12.0 Hz,
matching `AUDIT.md`'s ~11 Hz prediction almost exactly, so band-centering
is not the problem):

| variant | correlation |
|---|---:|
| 300 ms window, 2–20 Hz (as built) | +0.732 (this leg's best) |
| 1.0 s window, 2–20 Hz | +0.099 — worse |
| 300 ms window, 8–14 Hz narrow | −0.331 — worse |
| 1.0 s window, 8–14 Hz narrow | −0.024 — worse |

Neither a longer window nor a narrower band helped — both made it worse.
**Suspected cause: `scipy.signal.filtfilt` (zero-phase, forward-backward)
inside a sliding window.** It needs symmetric padding at both edges of the
window, and the edge that matters — the right edge, "now," the exact
sample the estimate is read from — is where a forward-backward filter's
own boundary handling is least reliable. Re-computing it on a fresh,
disjoint window every stride, rather than running a genuinely one-sided
continuously-updated filter, is likely compounding this with real block
discontinuities. **Not confirmed** — this is a diagnosis from the pattern
of the failure, not a proven root cause.

### This blocks Rung 2 regardless of Option A vs Option C

Both placements need a phase estimate to compute anything — Option A
needs it to know what current to inject, Option C needs it to know how to
shift joint-target timing. **The estimator failure is common to both**;
only the bound-sweep/seizure-risk question in §9 is specific to Option A.
Fixing this is on the critical path for Rung 2 no matter which write-side
option is ultimately used.

**Proposed fix, not yet built:** replace the sliding-`filtfilt` approach
with a genuinely causal filter — a one-sided IIR bandpass
(`scipy.signal.lfilter`/`sosfilt`, not `sosfiltfilt`) combined with a
causal analytic-signal approximation (a fixed all-pass Hilbert-approximator
filter, or a resonator/PLL-style running quadrature pair), rather than
re-filtering a sliding block from scratch every stride. This needs its own
build-and-revalidate cycle before Rung 2 can proceed; not started.

## 9. Coupling-strength bound sweep — write-side safety only

Because the estimator above is not trustworthy, the bound sweep
(`sweep_coupling_bound.py`) does **not** use it. It replays a **precomputed,
offline-derived reference phase trace, open loop** — fixed ahead of time
from one real untrained trial (replicate 0), not recomputed from the
trial being tested — to shape the injected current. This isolates one
question cleanly: is a current of this shape and magnitude, injected into
the CPG triad, safe? It says nothing about whether such a correction would
be *accurate* if computed live (the estimator's job, which failed above).

See the log for results, once the sweep completes.

### Sweep results

`media/trained_adapter/coupling_bound_sweep.json`. K ∈ {0, 1, 2, 4, 8, 16,
32, 64, 128}, pilot replicates {0, 1, 4, 6}, cap fixed at Phase 4's own
`SENSORY_CURRENT_CAP = 2.5` (reused, not re-derived — see caveat below).

**No saturation, no instability, at any K tested.** `n_active` stays
within roughly 2–9% of the K=0 baseline at every K (e.g. replicate 0:
515 → 515–537, never approaching the 1,500 stability threshold).
`n_rhythmic` drifts mildly downward at higher K — replicate 0 goes 6→5
from K=8 onward, replicate 1 touches its floor of exactly 2 (the
pre-registered minimum, not below it) only at K=128. Neither the
saturation nor the rhythmicity gate was violated anywhere in the tested
range.

**A methodological correction, made before reporting this as clean.** My
first read of this table was that K ≥ 8 all tested the same thing, since
`inj = clip(K·error, ±2.5)` should saturate to the cap almost every step
once K is large. Checked directly: **false** — the per-replicate sequences
are not identical across K (e.g. rep 0: 515, 515, 527, 528, 522, 537, 536,
522, 532 — real, if noisy, variation, not a step function). `sin(phase
error)` crosses zero every half-cycle regardless of K, so injected current
is not pinned at the cap continuously even at K=128. The sweep is
genuinely informative about K, not accidentally a repeated cap-only test.

**What this bound does and does not establish.** It is a real safety
result for correction current **up to magnitude 2.5** (the injected
current's actual ceiling, set by the cap, not by K) — K itself was pushed
to 128 without finding a ceiling in K. It says nothing about whether a
*larger* cap would also be safe; that is a different, untested axis, and
2.5 is carried over from Phase 4's sensory-current work rather than
independently re-derived here for the CPG-triad target specifically.

### Proposed bound, for approval

| parameter | proposed trainable range | basis |
|---|---|---|
| `coupling_strength` (K) | 0 to **64** | half the tested-safe ceiling (128), for margin against pilot-only sampling |
| current cap | fixed at **2.5** | Phase 4's `SENSORY_CURRENT_CAP`, reused not re-derived; not itself trainable in this proposal |

**Not committing K's upper bound at the full tested 128** — 64 leaves a
2× margin given this was measured on 4 pilot replicates, not the full
eligible pool, and using an open-loop reference signal rather than a real
closed-loop one.
