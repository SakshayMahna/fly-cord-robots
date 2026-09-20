# Pre-registration: does closing the loop produce interleg coordination?

**Committed before any main run. Nothing below may be changed after the
first main-experiment trial executes.** Deviations, if any become
necessary, get appended to `LOG.md` with a reason and a timestamp — never
edited into this file silently.

*Written 2026-09-19, after the audit (`AUDIT.md`) and the sensory map
(`SENSORY_MAP.md`). Read both first: the hypotheses below are framed around
findings from those documents, not around the original task sketch.*

---

## 0. What changed from the original plan, and why

Three reframings, all forced by audit findings rather than chosen for
convenience:

1. **H1 is now specifically about left↔right coupling.** The open-loop
   baseline is not uncoupled. Same-side (front↔middle↔hind) pairs show
   substantial coupling already — 52% significant, median PLV 0.535 — while
   left↔right pairs do not (16.9%, PLV 0.155; chance 7.6%). See `AUDIT.md`
   §6. The original "feedback creates coupling that is absent open-loop"
   only holds for the left↔right case, so that is the primary metric, and
   same-side coupling is tested baseline-relatively instead.

2. **H2 no longer involves load sensing.** MANC annotates 9 leg campaniform
   sensilla across all six legs (MaleCNS: 12); both datasets put their
   hundreds of campaniform sensilla in the wing and haltere nerves. No load
   channel can be built from real annotated neurons, and none will be
   invented. H2 is reframed around ground contact acting through
   joint-angle feedback. See `SENSORY_MAP.md` §1.

3. **Sensory drive is normalised per side, with a control.** The raw
   annotation gives one side up to 2.4× the other's sensory input, which
   would build an asymmetry directly into the primary measure. Normalisation
   is interface-level only; condition **C4** runs the un-normalised mapping
   so the choice is measured. See `SENSORY_MAP.md` §4.

Two things did **not** change: the connectome is never modified, and the
motor interface stays frozen at config hash
`04be9dec181ec2e20fad91ba07cfe018d0a413720b0dd441a2dd8aab7294fc1d`.

---

## 0b. What body this runs on — stated explicitly

**The body is NeuroMechFly v2's anatomically accurate *Drosophila* body,
not a hexapod robot.** It is a micro-CT-derived fly model with 7 real
degrees of freedom per leg (42 actuated), from FlyGym (EPFL
Neuroengineering Lab, Nature Methods 2024). See
`fly_robot/bodies/neuromechfly.py`.

This is deliberate for the closed-loop work, and it is a **narrowing** of
the project's eventual scope, which is a fly connectome driving a body it
did not evolve for. The hypotheses here are about whether connectome plus
body produces coordination that connectome alone does not. Asking that on
a fly-shaped body first removes a whole class of confound: if coordination
fails on a robot body, it is ambiguous whether the wiring cannot do it or
the body was simply wrong for it. On the fly's own body, a negative result
means something. Transfer to a non-fly body is later work and a separate
claim.

Consequence to state in any write-up: a positive result here shows the
connectome coordinates **its own body**, not that it generalises.

---

## 1. Hypotheses

**H1 — Feedback induces left↔right coupling.**
Closing the sensory loop increases phase locking between contralateral leg
pairs, relative to the open-loop baseline in the same body condition.

- *Primary metric:* PLV between left and right legs of the same segment
  (T1-L↔T1-R, T2-L↔T2-R, T3-L↔T3-R), neural readout.
- *Key secondary:* tripod index.
- *Secondary:* change in same-side (ipsilateral) PLV relative to the
  matched open-loop baseline — a **baseline-relative** test, because the
  open-loop value is already high and an absolute threshold would be
  meaningless.

**H2 — Ground contact changes coupling, through joint-angle feedback.**
Allowing the legs to contact ground and load the body changes interleg
coupling relative to the harness condition, even though the only sensory
channels are chordotonal (femur–tibia) and hair plate (thorax–coxa).

*Explicitly not claimed:* that this is how the real fly does it. Real
ground contact is sensed substantially by leg campaniform sensilla, which
we cannot model (§0.2). H2 asks whether the mechanical consequences of
ground contact — altered joint trajectories — are enough on their own.

**H3 — Any coupling depends on the real wiring.**
Coupling produced under H1/H2 is reduced or abolished when the connectome
is replaced by a degree- and sign-preserving shuffle (C1), when sensory
signals are replaced by rate-matched noise (C2), or when sensory channels
are permuted across legs (C3).

---

## 2. Conditions

Body support: **harness** = thorax rigidly fixed, legs in air
(`TetheredWorld`). **ground** = thorax fixed identically, feet on a
freely-rotating sphere — the standard **tethered-on-ball** rig used in
real *Drosophila* labs (`fly_robot/bodies/tethered_ball.py`).

The ball is what makes the two conditions differ in **exactly one thing**:
whether the feet have a substrate. A free-standing body would differ in
two (substrate *and* freedom to move), and would also fall over, since
raw connectome drive is not a balance controller. The ball gives real
contact, real load, and real mechanical coupling through a shared
substrate — the "biomechanical coupling" Pugliese et al. name as a
candidate mechanism — with the body pose identical to the harness.

Fixed parameters, held identical across every ground condition:

| | value | basis |
|---|---|---|
| radius | 3.0 mm | real rigs use a ~6 mm foam ball for a ~2.5 mm fly |
| mass | 3.4e-3 g | expanded polystyrene at ~0.03 g/cm³; 3.3× the fly's 1.02 mg |
| centre | (0.16, 0, −1.64) mm | fitted to the settled neutral stance, then raised until all six feet contact (§ LOG) |
| joint | MuJoCo `ball` (3 rotational DoF, 0 translational) | an air bearing's kinematics without modelling the air |
| damping | 1e-6 | residual air-bearing drag; nonzero only to stop numerical drift accumulating |
| contact | FlyGym's own `_GroundContactMixin`, `tibia_tarsus_only` | identical friction/solref/solimp to its flat-ground world |

Verified: all six tarsi contact at rest and throughout a driven gait, and
under a synthetic tripod drive the ball turns about the **pitch** axis
(−0.95 rad/s) with roll and yaw near zero — i.e. straight-line forward
walking.

**Ball rotation (quaternion + angular velocity) is logged every trial** as
an extra output. No hypothesis depends on it; it is the rig's own measure
of intended locomotion and is cheap to record.

| ID | loop | body | connectome | sensory | tests |
|---|---|---|---|---|---|
| **E0** | open | harness | real | none | baseline; replicates the paper |
| **E1** | closed | harness | real | normalised | feedback alone |
| **E2** | open | ground | real | none | body mechanics alone |
| **E3** | closed | ground | real | normalised | feedback + body |
| **C1** | closed | ground | **shuffled, protected** (degree- and sign-preserving; interface edges held out) | normalised | is it the wiring? — **primary wiring control** |
| **C1b** | closed | ground | **shuffled, everything** | normalised | secondary wiring control |
| **C2** | closed | ground | real | **rate-matched noise** | structured feedback vs extra drive |
| **C3** | closed | ground | real | **permuted across legs** | does leg-specificity matter? *(deliberate locality violation — control only)* |
| **C4** | closed | ground | real | **un-normalised, as-annotated** | what does the side normalisation do? |

E0 ≡ E1 at `g_fb = 0`, and E2 ≡ E3 at `g_fb = 0`. Both are run separately
anyway and asserted **bit-identical** to the corresponding sweep point.
That is the `g_fb = 0` reproduction test, and a failure invalidates the run.

---

## 3. Sweeps, seeds, duration

**Feedback gain `g_fb` ∈ {0, 2.5, 5, 10, 20, 40}** — six levels, on E1 and
E3 only. Chosen from measured sensory-neuron thresholds (median 3.10, IQR
1.97–4.82) and the resulting steady activation: `I=5` activates 76% of the
pool at ~4 Hz, `I=10` 95% at ~16 Hz, `I=20` 99.6% at ~38 Hz, `I=40` ~80 Hz
and saturating. The range therefore spans sub-threshold to saturating,
bracketing the interesting region rather than guessing at it.

**DNg100 drive ∈ {285, 380, 475}** — the published value (380) and ±25%.

**Controls C1, C1b, C2–C4** run at `g_fb = 10`, drive = 380.

**Seeds: 20 per condition.** A seed selects one stochastic draw of
per-neuron parameters plus a randomised initial joint state. Main runs use
parameter seed **20260919**, replicate indices 0–19 — a draw never used for
any tuning or exploratory decision. **Pilot seed set: parameter seed 641,
indices 0–7**, the draws already used throughout the audit. Pilot seeds are
never used in main analysis; main seeds are never used for tuning.

**Trial duration 4.0 s**, neural dt 0.001 s, physics dt 0.0001 s, tonic
stimulation 0.02–3.999 s. Longer than Pugliese's 2 s because the audit
showed the cost is 11.2 s per 2 s trial, and 3.5 s of post-transient signal
gives ~38 cycles at the measured ~11 Hz rhythm instead of ~16 — materially
better phase statistics for the same afternoon of compute.

**Transient discard: first 0.5 s**, fixed, all conditions.

**Stability filter:** replicates with `n_active > 1500` are excluded, using
Pugliese's own documented oversaturation criterion. The excluded count is
reported per condition; it is not a quality filter and is never applied
after seeing a result.

**Budget:** E1 and E3 at 6 × 3 × 20 = 360 trials each; E0 and E2 at 3 × 20
= 60 each; C1, C1b, C2, C3, C4 at 20 each = 100. **940 trials**, ~24 s each
(22.4 s neural + 0.8 s physics, measured) ≈ **6.3 hours**. No reduced
matrix needed.

---

## 4b. Control definitions (C1, C2)

Both implemented in `fly_robot/neural/connectome_controls.py`. Neither
modifies the real connectome — the shuffle returns a new matrix and the
original is untouched.

**C1 — degree-preserving edge shuffle.** Classic double-edge swap: edges
a→b and c→d become a→d and c→b, which leaves the out-degree of a and c
and the in-degree of b and d unchanged. Weights travel with their source,
so the weight multiset is identical. Swaps happen **within sign class
only** — well-defined here because the connectome obeys **Dale's law
exactly** (verified: 0 of 22,769 presynaptic neurons have outgoing edges
of both signs), so a source can never change sign as a side effect.
Neurons are never relabelled, so sensory injection and motor readout
still address the same cells.

A swap is **rejected** if it would create a self-loop or duplicate an
existing edge, because `csr_matrix` silently *sums* duplicates and
destroys edges. Verified on the real matrix (seed 0, 10 rounds):

| | |
|---|---|
| swaps proposed / accepted | 6,862,020 / 6,491,786 (94.6%) |
| targets changed | 99.99% |
| out-degree preserved | **yes, max error 0** |
| in-degree preserved | **yes, max error 0** |
| weight multiset preserved | **yes** |
| per-neuron (excitatory, inhibitory) out-counts | **identical** |
| self-loops | 16 → 0 (existing ones can be swapped away; new ones are never created) |
| deterministic for a fixed seed | yes |

**C1 is the PROTECTED variant** (decided before any run). Three edge sets
are held out of the shuffle entirely and verified to come through
untouched:

| held out | edges |
|---|---:|
| incoming to motor neurons | 87,534 |
| outgoing from sensory neurons | 236,224 |
| outgoing from DNg100 (rows 59, 282) | 1,043 |
| **total protected** | **320,551 of 1,372,404 (23.4%)** |

Why: shuffling these would randomise the *interface*, not the circuit.
With motor neurons' inputs shuffled, each leg's motor pool is driven by
arbitrary interneurons and the per-leg readout stops meaning "this leg" —
so a loss of coordination could come from the readout becoming
meaningless rather than from coupling genuinely failing. Likewise,
shuffling sensory outputs or DNg100's axon would change where feedback and
command drive enter, which is a different manipulation from changing the
wiring they enter *into*. C1 therefore keeps the interface fixed and
randomises only the internal circuit — the thing H3 is actually about.

Verified with the protected sets (seed 0, 10 rounds): 4,945,235 of
5,259,260 swaps accepted (94.0%), **76.6%** of targets changed,
out- and in-degree preserved with **max error 0**, weight multiset
identical, and every protected edge present unchanged in the output.

**C1b is the shuffle-everything variant**, run as a secondary control at
the same settings, so the effect of the protection itself is measured
rather than assumed. If C1 and C1b disagree, that difference is reported;
it is not grounds for preferring whichever looks better.

**C2 — rate-matched noise.** Phase-randomised surrogates of **each
channel's actual recorded sensory drive from the matched E3 trial**, not
synthetic noise. The surrogate preserves the amplitude spectrum exactly,
hence the mean, variance and autocorrelation, and destroys only the timing
relationship to the body. Channels are randomised independently, so
cross-channel timing structure goes too. Surrogates are injected through
exactly the same code path as the real drive
(`SensoryEncoder.input_current(..., drives=...)`).

Verified over 200 draws × 6 channels: mean correlation with the real
signal **−0.011**, mean preserved to 4e-17, standard-deviation ratio
1.000, amplitude spectrum to 1e-16. Individual draws vary widely
(|r| median 0.61) because a narrowband signal keeps its frequency under
phase randomisation — which is exactly why the null needs many draws
rather than one.

---

## 4a. Rhythmicity is evaluated FIRST

**A condition must have a rhythm before any coupling metric is computed
for it.** Phase is meaningless without one: the Hilbert transform returns
a phase for pure noise perfectly happily, and two such phases will
sometimes look locked. Gating first is what stops that becoming a
spurious coupling result.

**Per leg-trial.** A leg is *rhythmic* if it (a) exceeds the minimum
amplitude `MIN_PEAK_ACTIVITY`, and (b) its in-band spectral peak strength
(peak power / total 2–20 Hz power) exceeds the **95th percentile of an
AR(1) null** matched to that leg's own variance and lag-1
autocorrelation, over 200 surrogates.

The null is deliberately **red noise, not the phase-randomised surrogate**
used for the coupling test. A phase-randomised surrogate preserves the
amplitude spectrum exactly, so it has the same spectral peak as the real
signal and cannot test whether that peak is real. AR(1) noise reproduces
the smooth, autocorrelated background a non-oscillating rate signal has,
and asks whether the peak stands out above it.

Calibration, measured not assumed:

| | |
|---|---|
| false positives, white noise | **5.0%** (nominal 5%) |
| false positives, red noise (φ = 0.9) | **3.3%** |
| detection, real trace + noise at 1× its in-band amplitude | **100%** |
| detection, at 2× | **100%** |
| detection, at 4× | **94%** |
| detection, at 8× | 22% |
| on real published data | 66.7% of active leg-trials called rhythmic |

On the real data the rejected legs are visibly weak (peak strength
0.18–0.43) against a stable threshold near 0.49, while accepted ones sit
at 0.70–0.97.

**Per trial.** Coupling is computed only for pairs where **both** legs are
rhythmic. A trial with fewer than 2 rhythmic legs yields no coupling
numbers at all (NaN, `has_rhythm = False`), not a zero.

**Per condition.** If fewer than 50% of a condition's trials have ≥ 2
rhythmic legs, the condition is reported as **"no rhythm"** and **no
coupling metric is reported for it** — not a null coupling result, which
would wrongly imply coupling was measured and found absent. The number of
rhythmic legs per trial is reported for every condition regardless.

A condition showing "no rhythm" is a real and reportable outcome: it says
the manipulation abolished the oscillation itself, which is a different
finding from abolishing coordination.

---

## 4. Metrics

Computed identically on both readouts, so neural and mechanical
coordination are comparable:

- **Neural readout (pre-declared):** per leg, the **sum** of firing rates
  over that leg's motor neurons, de-duplicated (the audit found 16 repeated
  rows; see §8). Sum rather than mean because only 2–10 of each leg's ~50–70
  motor neurons are recruited per replicate.
- **Kinematic readout:** per leg, the femur–tibia joint angle; stance/swing
  from ground contact where a ground condition applies.

Per leg: dominant frequency and spectral peak strength in the **2–20 Hz**
band. (The band was re-derived from data after an error — an earlier
version used a firing-*rate* range as a frequency band and capped below the
real rhythm. See `AUDIT.md` §6.)

Per pair (15 unordered pairs): phase difference via Hilbert transform of
the band-passed signal; **phase-locking value**; circular mean phase
difference; Rayleigh test.

**Tripod index:** PLV-weighted mean cosine of (observed − expected) phase
difference, where expected is 0 within a tripod group and π between.
Groups: {T1-L, T2-R, T3-L} against {T1-R, T2-L, T3-R}.

**Null:** phase-randomised surrogates preserving each signal's amplitude
spectrum, 200 per trial. Measured false-positive rate of this procedure is
**7.6%** against a nominal 5% (`AUDIT.md` §6); **7.6% is used as the chance
baseline**, not 5%.

**Across conditions:** bootstrap confidence intervals over seeds; effect
sizes reported alongside p-values; **all conditions reported, including
failures and null results.**

---

## 5. What counts as "coordination emerged"

Declared in advance so the answer cannot be chosen after seeing the data.

**H1 is supported** if, in E3 (or E1) at some `g_fb > 0`, compared to the
matched `g_fb = 0` baseline at the same drive level:

1. the contralateral significance rate exceeds the baseline rate with a
   bootstrap 95% CI on the difference excluding zero, **and**
2. median contralateral PLV increases by ≥ 0.15 absolute, **and**
3. the effect is present at ≥ 2 adjacent `g_fb` levels (not a single point).

**H1 is refuted** if no `g_fb` level meets (1). A refutation is a real
result and will be reported as prominently as a confirmation.

**Tripod specifically** counts as emerged only if the tripod index rises
above **+0.3** and its bootstrap 95% CI excludes 0. The open-loop baseline
is **−0.22**, so this requires a genuine reversal, not a drift toward zero.

**H2 is supported** if E3 differs from E1 on the primary metric with a
bootstrap CI excluding zero, at matched `g_fb` and drive.

**H3 is supported** if C1, C2 and C3 each show a lower primary metric than
E3 with CIs excluding zero. Each control is reported separately — partial
support (e.g. C1 yes, C2 no) is reported as partial, not rounded up. C1b
is reported alongside C1; a disagreement between them is reported as such
and is never grounds for preferring whichever looks better.

**A control that abolishes the rhythm** (§4a) supports H3 only in the
weaker sense that the manipulation mattered. It is reported as "no
rhythm", explicitly distinguished from "rhythm present, coupling lost",
because the two mean different things.

**Same-side coupling** is tested only as a *change* from the matched
baseline, with the same three criteria. Its open-loop level is reported in
`RESULTS.md` as an observation about the published model, clearly separated
from any closed-loop claim.

---

## 6. Validation gates before main runs

Each must pass and be recorded in `LOG.md`:

- [x] Steppable model reproduces published output — recruitment Jaccard
      1.000/0.997/1.000 on three replicates, median r ≥ 0.9988 (`AUDIT.md` §4).
- [x] Coordination metric is dt-robust — PLV correlation 1.0000, 100%
      significance agreement at 4× finer steps (`AUDIT.md` §6).
- [x] Significance test calibrated — 7.6% false-positive rate measured.
- [x] `g_fb = 0` emits exactly zero current (not merely small), so the
      closed loop reduces to open loop bit-for-bit —
      `tests/test_sensory_interface.py::test_gain_zero_gives_exactly_zero_current`.
- [x] Determinism: identical input → identical current, repeated —
      `::test_deterministic_for_identical_input`. Shuffle determinism for
      a fixed seed verified separately (§4b).
- [x] **Locality**: perturbing leg *i*'s joints alone changes only leg
      *i*'s sensory neurons, for all six legs —
      `::test_locality_perturbing_one_leg_changes_only_that_legs_neurons`.
      The C3 permutation is confirmed to be the *only* cross-leg path —
      `::test_c3_permutation_is_the_only_cross_leg_path`.
- [x] Encoder output bounded, finite and non-negative under extreme input
      (±1e3 rad, ±1e4 rad/s) — `::test_current_is_finite_and_non_negative_over_extreme_input`.
- [x] Ball world: all six feet contact at rest and under drive; ball
      rotates about pitch with roll/yaw ≈ 0; no NaN over 4 s.
- [ ] **End-to-end `g_fb = 0` ≡ E0 on a full trial** (needs the coupled
      loop runner, not yet built).
- [ ] dt-robustness re-checked at the 4 s duration.
- [ ] Firing rates bounded and physics stable across the full sweep range;
      any failing condition is **reported, not silently dropped**.

All passing gates run as tests, not notebook assertions:
`python -m pytest tests/test_sensory_interface.py` — 13 passed.

---

## 7. Stop conditions (still live)

Work stops and the user is consulted if any of these occur:

- a step appears to require modifying the connectome or the frozen motor
  interface;
- coordination appears to require a cross-leg path in our code (forbidden
  here; that belongs to later work and must be labelled as an addition);
- results depend strongly on dt at the 4 s duration;
- compute cost exceeds what is reasonable — a reduced matrix is proposed
  rather than conditions being silently cut.

*(The "open-loop baseline shows coupling" condition already triggered, was
reported, and is resolved by the H1 reframing in §0.1.)*

---

## 8. Open implementation items

These must be closed before the first main trial, and none of them may
change anything in §1–§5:

1. ~~De-duplicate the 16 repeated motor-neuron rows~~ — **done**; root
   cause was a one-to-many MaleCNS join, fixed at source. Baseline
   recomputed; every figure moved by <1 percentage point (`AUDIT.md` §3, §6).
2. ~~Choose the body-support mechanism~~ — **done**: tethered-on-ball,
   built in `fly_robot/bodies/tethered_ball.py`, parameters fixed in §2.
3. ~~Define the degree- and sign-preserving shuffle for C1~~ — **done**, §4b.
4. ~~Define "rate-matched" for C2~~ — **done**, §4b.
5. **Build the coupled neural↔physics loop runner**, then run a PILOT on
   the pilot seed set only (parameter seed 641, indices 0–7). Gates before
   any main run: (a) `g_fb = 0` in the full runner reproduces open-loop
   output exactly at the same seed; (b) pilot uses pilot seeds only;
   (c) timing and any instability reported. **Stop after the pilot.**
6. ~~Decide C1's protected-edge variant~~ — **done**: C1 is protected,
   C1b is shuffle-everything (§4b).

---

## 9. Inherited limitations (stated, not fixable here)

- Dynamics run on **MANC**, not MaleCNS (`AUDIT.md` §1) — the project's
  nominal dataset has never been simulated.
- **No leg load sensing** (`SENSORY_MAP.md` §1).
- **No chordotonal subtype resolution** — one combined signal drives a
  population that is really heterogeneous (`SENSORY_MAP.md` §3).
- **Encoder form is our design**, consistent with published encoding
  properties but not a measured fly transfer function.
- The body is **NeuroMechFly's fly body**, driven by a connectome from a
  *different specimen* than the body was built from.

---

# AMENDMENT 1 — pilot-informed, 2026-09-20

**Added after the pilot, before any main-experiment trial.** Pilot used
pilot seeds only (parameter seed 641, replicates 0–3); no main seed
(20260919) has been touched. Nothing above §0–§9 has been edited; this
amendment is additive and timestamped, per the rule at the top of this
file.

## A1. The `g_fb` sweep was wrong, and is re-derived

**What the first pilot found.** The pre-registered sweep
{0, 2.5, 5, 10, 20, 40} sits almost entirely past a bifurcation:
`g_fb = 2.5` changed `n_active` by ~3% (a near no-op), and every level
≥ 5 pushed the network into a self-sustaining high-activity state
(`n_active` ≈ 3,600–4,900 against Pugliese's oversaturation criterion of
1,500) with rhythmicity collapsing from a median of 3.5 rhythmic legs
to 1.

**Why it was wrong.** `g_fb` was calibrated from *single-neuron*
steady-state activation — threshold median 3.10, so `I = 5` activates 76%
of the sensory pool. Correct arithmetic, wrong level of description: it
ignores that 472 sensory neurons with out-degree 22–44 deliver an
aggregate downstream drive roughly an order of magnitude larger. The same
class of error as `rate_scale_hz = 20` and the 0.5–10 Hz band.

**Replacement sweep: {0, 1, 2, 2.5, 3, 3.5, 4, 4.5}**, concentrated in
the window the first pilot left unsampled.

## A2. A second encoder formulation is added as a pre-registered factor

**Why.** Decomposing the recorded sensory drive showed the feedback was
overwhelmingly a standing bias rather than feedback:

| | chordotonal DC | modulation | modulation/DC |
|---|---:|---:|---:|
| `signed` (original) | 0.2961 | 0.0347 | **11.7%** |

**88.1% of all injected current was DC** — every chordotonal neuron
receives 0.25 × `g_fb` at rest, because the signed position code sits at
0.5 when the joint is at its reference angle. And the **hair plate
channel was identically silent** (mean and std exactly 0.0000): the
thorax–coxa joint never reaches 75% of range in the extension direction,
so the limit detectors never fire at all.

**`deviation` encoder.** Drive proportional to the rectified deviation
from the **settled resting pose** (measured after warmup, before
stimulation onset — a deterministic property of the body configuration,
not a tuned parameter, and necessarily different in harness and on the
ball). Rest gives ≈ 0 input.

Measured effect: chordotonal DC 0.3085 → **0.0789**, modulation 0.1024 →
**0.1335** — modulation now *exceeds* DC — and hair plates become
functional (DC 0.0184, modulation 0.0860) instead of identically zero.

Cost, stated: `|deviation|` cannot distinguish flexion from extension.
Acceptable because the data does not resolve chordotonal subtypes anyway,
so a pooled population could not signal direction either way. For hair
plates, firing at *either* extreme is arguably closer to the real
population than picking one direction: MANC annotates only a generic
"hair plate" subclass with no CxHP3/4/8 identity, and the source paper
states the three hair plates "wrap the joint along the
anterior-posterior axis".

**Encoder is now a pre-registered factor**, not a fixed choice. Both are
run; neither is selected after seeing results.

## A3. Baseline rhythmicity confirmed

The pilot's `g_fb = 0` median of **3.5 rhythmic legs** was checked against
the audit baseline rather than assumed to match:

| | median rhythmic legs |
|---|---:|
| published data, 2 s, all 118 stable replicates | **3.0** (mean 3.25) |
| published data, 2 s, the 4 pilot replicates | 3.5 |
| our model, 2 s, same 4 replicates | 3.0 |
| our model, 4 s, same 4 replicates | **3.5** |

Consistent. Our model matches the published data on 3 of 4 replicates at
matched duration (the one that differs by 1 is replicate 2, the
oversaturated draw). The +0.5 is the **longer analysis window**: 3.5 s
gives more cycles and therefore more power to detect a rhythm, adding one
leg on 2 of 4 replicates. Published distribution has 3 (n=36) and 4
(n=30) as its two commonest values, so the pilot's four draws are typical.

## A4. The bifurcation persists across BOTH encoders

Per the instruction not to treat it as a result until tested against both
formulations. Fine sweep, 3 usable replicates (replicate 2 excluded by
the pre-registered stability filter — oversaturated at baseline, with no
feedback at all):

| `g_fb` | signed: n_active | rhythmic | deviation: n_active | rhythmic |
|---:|---:|---:|---:|---:|
| 0 | 380 | 4 | 380 | 4 |
| 1 | 384 | 4 | 380 | 4 |
| 2 | 389 | 4 | 384 | 4 |
| 2.5 | 385 | 4 | 384 | 4 |
| 3 | 396 | 4 | 384 | 4 |
| 3.5 | **4135** | **1** | 388 | 4 |
| 4 | **4216** | **1** | 388 | 3 |
| 4.5 | **4710** | **1** | **3794** | **1** |

The `deviation` encoder moves the cliff from ~3.25 to ~4.25 but does not
remove it. **More importantly, neither encoder has a graded regime**:
below the cliff, feedback changes `n_active` by 2–4% and leaves the median
rhythmic-leg count unchanged at 4; above it, the rhythm is destroyed.
Feedback is either negligible or catastrophic, with nothing in between.

Note the deviation encoder bifurcates at roughly **one third** of the
total DC current that destabilises the signed one, so DC alone does not
explain the cliff. The mechanism is consistent with positive feedback:
movement → sensory drive → more motor drive → more movement. The
deviation encoder, by design, responds to movement, which makes its peak
drive higher even though its mean is far lower.

Zero numerical instability in any of the 68 trials; peak firing rate
235.6 Hz against a 1000 Hz clip. This is a property of the modelled
circuit, not of the integrator.

## A5. What is NOT decided here

Whether "proprioceptive feedback of any effective strength destabilises
the unmodified connectome" is reportable as a *finding* is deferred to
the user. It now satisfies the stated bar (persists across both
encoders), but alternatives remain untested — for example capping
per-neuron current, driving a random fixed subset of proprioceptors, or
introducing the inhibitory component a real sensory pathway would have.
No main run is started until that is decided.

---

# AMENDMENT 2 — encoder variants and stopping rule, 2026-09-20

**Written and committed BEFORE the runs it governs.** Pilot seeds only
(parameter seed 641, replicates 0–3). Additive; nothing above is edited.

## B0. Encoder-side inhibition is rejected

Adding an inhibitory component to the sensory encoder was considered and
is **rejected on principle**: the sign of every sensory neuron's output is
already given by the connectome (via `predictedNt`, and the connectome
obeys Dale's law exactly — 0 of 22,769 presynaptic neurons carry both
signs). Introducing encoder-side inhibition would override the data with a
modelling choice, which this project's first constraint forbids.

`RESULTS.md` must record the corollary explicitly: **the network's own
inhibitory interneurons — 636,490 of its 1,372,404 edges, 46% — do not
prevent the runaway.** Whatever destabilises the loop does so in the
presence of the real inhibitory circuitry.

## B1. Two further encoder variants, both built on `deviation`

**No further encoder variants after this round**, regardless of outcome.

### Variant A — `deviation_capped`: per-neuron saturation cap

A ceiling on the current any single sensory neuron may receive.

**Cap = 2.5, measured not chosen.** Per-neuron sensory current was
recorded across every stable-regime pilot trial (deviation encoder,
`g_fb ≤ 3.5`, 6.6M neuron-timesteps): p95 = 1.062, p99 = 1.589,
p99.9 = 2.251, **max = 2.386**. The runaway regime (`g_fb ≥ 4.0`) instead
runs at median 2.0–2.1, p99 6.5–8.5, max 8.5.

A cap of 2.5 therefore sits just above everything the stable regime ever
produces — so it **cannot** alter behaviour below the cliff — and bites
only on the excursions that characterise runaway. The variant tests one
specific hypothesis: that the bifurcation is driven by transient peaks in
per-neuron drive.

Verified: at extreme input the per-neuron maximum falls 7.556 → 2.500 and
total injected current falls 32%.

### Variant B — `deviation_fractionated`: range-fractionated tuning

Each chordotonal neuron gets a fixed preferred joint-angle band, drawn
once from seed 20260920, with Gaussian tuning of width σ = 0.1 over a
normalised range of [0, 1]. Only the subset tuned near the current angle
responds.

**Motivation, from primary text** — "Biomechanical origins of
proprioceptor feature selectivity and topographic maps in the Drosophila
leg" (bioRxiv 2022.08.08.503192), verbatim: *"Fractionation of the tibia
joint angle range across position-tuned proprioceptors has been
previously described in the grasshopper FeCO"*, and *"The cell bodies of
position-tuned proprioceptors form a goniotopic map of joint angle"*.

**Labelled as OUR DESIGN CHOICE.** The real map is orderly and
anatomically arranged; ours assigns bands at random, because the
connectome carries no per-neuron tuning annotation (types are opaque
`SNppNN` identifiers). What is reproduced is the population property that
matters here — only a subset responds at any one angle — not the spatial
map. No claim is made about arrangement in the VNC; the same paper reports
that *"calcium imaging from position-tuned axons failed to resolve any
topographic organization"*.

The shared movement term is left un-fractionated: position tuning and
movement tuning belong to different FeCO subtypes (claw vs hook), so
fractionating movement too would assert more than the source supports.

Verified: 22.0% of each chordotonal pool responds above half-maximum at
mid-range (23.5% theoretical), 14–15% at the range extremes; total
injected current falls 39% at extreme input.

**Honest cost, recorded in advance:** fractionation *reintroduces* drive
at rest — neurons tuned to the resting angle fire there, which is correct
for a position code. Total rest current is 101.7 against 0.0 for plain
`deviation`, though still roughly a quarter of the `signed` encoder's.
If the bifurcation returns for this variant, that is a likely reason.

## B2. Decision rule — committed before running

The sweep is {0, 1, 2, 2.5, 3, 3.5, 4, 4.5}; replicates oversaturated at
baseline are excluded by the existing stability filter.

A variant is **USABLE** if BOTH hold:

1. **Rhythm preserved** — at least **3 sweep levels above 0** have a median
   rhythmic-leg count within **1** of that variant's own `g_fb = 0`
   baseline.

2. **Measurable feedback effect** — at those same levels, feedback
   demonstrably changes the neural trajectory. Effect measure, declared
   now: `E(seed, g) = 1 − (mean over legs of the Pearson correlation
   between that leg's motor readout at gain g and at gain 0, same seed,
   post-transient window)`. The runner is deterministic, so `E = 0`
   exactly when feedback does nothing. Required: the **bootstrap 95% CI of
   `E` across seeds excludes zero** (which demands consistency across
   seeds, not merely a large effect in one), **and** median `E > 0.05`
   — a floor, so an effect that is reproducible but negligible does not
   count as usable.

**Outcomes:**

- **Either variant usable** → it is used for the main runs. If both
  qualify, the one with more usable levels is chosen; on a tie, variant A
  (fewer added assumptions).
- **Neither usable** → the bifurcation across **all four encoders**
  (`signed`, `deviation`, `deviation_capped`, `deviation_fractionated`)
  becomes the reported result, written up in `RESULTS.md` with the
  inhibition corollary from §B0.

Both outcomes are reported in full regardless of which occurs, including
the per-level `E` values so the dose-response shape is visible rather than
summarised into a verdict.
