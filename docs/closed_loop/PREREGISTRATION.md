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

Body support: **harness** = base rigidly fixed, legs in air (`TetheredWorld`).
**ground** = feet contacting ground with partial body-weight support, held
**identical across every ground condition**. The support mechanism is an
open implementation item (see §8); whatever is chosen is documented in
`LOG.md` and used unchanged everywhere.

| ID | loop | body | connectome | sensory | tests |
|---|---|---|---|---|---|
| **E0** | open | harness | real | none | baseline; replicates the paper |
| **E1** | closed | harness | real | normalised | feedback alone |
| **E2** | open | ground | real | none | body mechanics alone |
| **E3** | closed | ground | real | normalised | feedback + body |
| **C1** | closed | ground | **shuffled** (degree- and sign-preserving) | normalised | is it the wiring? |
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

**Controls C1–C4** run at `g_fb = 10`, drive = 380.

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
= 60 each; C1–C4 at 20 each = 80. **920 trials**, ~24 s each (22.4 s neural
+ 0.8 s physics, measured) ≈ **6.1 hours**. No reduced matrix needed.

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
support (e.g. C1 yes, C2 no) is reported as partial, not rounded up.

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
- [ ] `g_fb = 0` reproduces open-loop **bit-identically**, same seed.
- [ ] Determinism: same seed + config → identical results, twice.
- [ ] Locality test passes: perturbing leg *i*'s sensors changes only leg
      *i*'s sensory input (all conditions except C3).
- [ ] dt-robustness re-checked at the 4 s duration.
- [ ] Firing rates bounded and physics stable across the full sweep range;
      any failing condition is **reported, not silently dropped**.

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

1. **De-duplicate the 16 repeated motor-neuron rows** in
   `all_legs_circuit.csv` (T1-L 2, T1-R 5, T2-L 1, T2-R 1, T3-L 4, T3-R 3;
   346 rows → 330 unique). They currently double-count neurons in the
   per-leg sum.
2. **Choose the partial body-weight support mechanism.** FlyGym ships
   `TetheredWorld` (rigid) and `FlatGroundWorld` (free) with nothing in
   between. Candidates: reduced gravity, or a vertical spring between
   thorax and a fixed site. Whichever is chosen is documented and held
   identical across E2, E3 and C1–C4.
3. **Define the degree- and sign-preserving shuffle for C1**, reusing
   Pugliese's own `shuffle_utils` where it fits.
4. **Define "rate-matched" for C2** — noise matched to each leg's real
   sensory drive in mean and variance, uncorrelated across legs.

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
