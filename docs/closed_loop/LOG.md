# Closed-loop work log

Chronological. Every run, pilot decision, bug and change. Newest entries
appended at the bottom. This is the record that `PREREGISTRATION.md`
deviations must be written into — never edit the pre-registration silently.

---

## 2026-09-19 — Audit (step 1)

Full detail in `AUDIT.md`. Summary of what happened, in order:

1. **Inventoried the simulated network.** Found 5,891 sensory neurons
   already present in the 23,532-neuron network Pugliese simulate,
   including 394 leg chordotonal organs and 78 hair plates. All have
   outgoing synapses; each leg's group reaches its own leg's CPG and motor
   neurons within two hops. No network change is needed to close the loop.

2. **Discovered the solver could not be stepped.** Their
   `run_single_simulation` hands the whole 2 s window to one adaptive
   `diffrax` Dopri5 call. Measured the cost of restarting it every 1 ms:
   **2.37 s of compute per 1 ms simulated — 79 min per 2 s trial.**
   Infeasible for a ~900-trial matrix.

3. **Built `steppable_rate_model.py`.** Same rate equation, same
   connectome, same neuron parameters; fixed-step RK4 instead of adaptive
   Dopri5, and CSR sparse instead of dense (W is 0.248% dense — 2.06 GiB →
   11 MiB). **11.2 s per 2 s trial**, a 420× speedup.

4. **Validated it against their published output.** Rebuilt three published
   replicates' exact neuron parameters using their own
   `prepare_neuron_params` with the seeds from their `run_config.yaml`
   (641, 98, 445, 424 — the seed→replicate mapping was confirmed
   empirically, not assumed). Recruitment Jaccard 1.000 / 0.997 / 1.000;
   median correlation 0.9993 / 0.9988 / 0.9997.

5. **Chased down the residual disagreement rather than accepting it.**
   Replicate 40 showed only 69% of neurons above r = 0.99. Refining the
   step size made it *worse*, not better — and two of our own runs at 4×
   different step sizes agreed with each other no better (r = 0.9969) than
   ours agreed with theirs (0.9988). Window-by-window, the first 1.5 s is
   perfect (r ≥ 0.9995, 100% above 0.99) and only the last 0.5 s degrades.
   **Conclusion: intrinsic phase sensitivity of the oscillators, not
   integrator error.** Recorded rather than glossed, because Phase 4
   measures phase.

6. **Bug found and fixed in our own analysis.** The rhythm band was set to
   0.5–10 Hz, citing "0.9–6.5 Hz" from `docs/logs/2026-09-19.md`. That
   figure is a firing **rate** range, not an oscillation **frequency** —
   two different quantities both denominated in Hz. The cap sat below the
   real rhythm, so every "dominant frequency" reported was meaningless
   leftover power. Caught by inspecting raw traces after the summary
   statistics looked wrong (mean-upcrossing intervals of 82–86 ms — clearly
   ~12 Hz — against a reported 0.67 Hz).

   Re-derived from data over a deliberately over-wide 0.5–60 Hz search
   across all 581 active leg-trials: median **10.7 Hz**, 5th–95th
   percentile 4.7–13.3 Hz. Consistent with our own Phase 0 numbers from
   Pugliese's `compute_oscillation_score` (5.7–11.4 Hz) and with real
   *Drosophila* stepping (~5–15 Hz). **Band corrected to 2–20 Hz.**

7. **Calibrated the significance test** on phase-randomised data with no
   true phase relationship: 7.6% false positives against a nominal 5%.
   7.6% adopted as the chance baseline.

8. **Ran the E0 open-loop baseline** on Pugliese's 128 published
   replicates (118 stable). Result: ipsilateral 52% significant / PLV
   0.535; contralateral 16.9% / 0.155; diagonal 16.7% / 0.147; tripod index
   −0.221.

   Ruled out before reporting: shared motor neurons between legs (checked —
   no overlap on any of the 15 pairs), and test miscalibration (measured,
   step 7).

   Checked against the paper's **primary text** rather than memory or a
   search summary. Their two tested claims are about left↔right coupling
   and the tripod pattern; both are reproduced. The words "ipsilateral" and
   "contralateral" appear **zero times** in the paper.

   **Stop condition triggered and reported to the user rather than resolved
   unilaterally.**

9. **dt-robustness of the metric** (the other stop condition): six
   replicates at h = 1 ms vs h = 0.25 ms. Dominant frequency difference
   0.000 Hz median, PLV correlation 1.0000, significance agreement 100%,
   tripod index identical. **Not triggered** — the raw traces drift in
   phase but the pairwise phase *relationships* do not.

**Commit:** `7d7c995`.

---

## 2026-09-19 — User decisions on the stop condition

Received and recorded verbatim in effect:

1. Reframe H1: primary = left↔right PLV; key secondary = tripod index;
   secondary = change in same-side PLV vs open-loop baseline, by a
   baseline-relative test. Report open-loop same-side coupling as an
   observation in `RESULTS.md`.
2. No load sensors to be invented. Reframe H2 as: does ground contact
   change coupling via joint-angle (chordotonal / hair plate) feedback.
   Record missing leg campaniform annotation as a limitation.
3. Left/right asymmetry: normalise total sensory drive per side so both
   sides get equal input, interface-level only. Add a control condition
   using the un-normalised, as-annotated mapping.
4. Check leg sensory annotation counts per leg and side in MaleCNS vs
   MANC and report; propose switching only if MaleCNS is more complete and
   symmetric; do not switch without approval.
5. Update `PREREGISTRATION.md` with all of the above and commit before any
   main runs.

---

## 2026-09-19 — MaleCNS vs MANC sensory annotation (decision 4)

New script: `fly_robot/connectome/compare_sensory_annotation.py`. Full
numbers in `SENSORY_MAP.md` §1.

Discovered that the two datasets use **different annotation schemes** —
MaleCNS has `class = mechanosensory_proprioceptive` with an explicit
`entryNerve` field, where MANC has `subclass` values and requires parsing
the `instance` string. Neither scheme was assumed; both were queried.

Findings:

- MaleCNS annotates **645** leg proprioceptors to MANC's **481**, and is
  dramatically more left/right symmetric for T2 (asymmetry 0.018 vs 0.244)
  and T3 (0.044 vs 0.125).
- **But only 40.6% have a `mancBodyid` cross-reference, and only 248
  unique rows are reachable in the simulated MANC network** — roughly half
  of MANC's native 481. The dynamics run on MANC, so unreachable neurons
  are unusable.
- T1 stays badly asymmetric in both (0.385 vs 0.419), and *in opposite
  directions* — MANC right-biased 27:66, MaleCNS left-biased 45:20.
- Leg campaniform sensilla are effectively absent in **both**: MANC 9,
  MaleCNS 12. Switching does not rescue load sensing.

**Recommendation to the user: do not switch.** Not switched. The only real
route to MaleCNS's better annotation is simulating MaleCNS-derived weights
— the long-standing open gap in `AUDIT.md` §1, and a much larger piece of
work that would break comparability with Pugliese's published baseline.

---

## 2026-09-19 — Sensory map and pre-registration (decisions 1–3, 5)

Encoding properties verified against **primary text**, not search
summaries, per the project rule:

- **Chordotonal → femur–tibia.** Mamiya, Gurung & Tuthill, *Neuron* 100(3)
  636–650 (2018), fetched via NCBI eutils: "one group of axons encodes
  tibia position (flexion/extension), another encodes movement direction,
  and a third encodes bidirectional movement and vibration frequency."
- **Hair plate → thorax–coxa, limit detection.** bioRxiv
  2025.05.15.654260 full text: "CxHP8 neurons encode the anterior limits of
  thorax-coxa joint angles"; hair plates "located at the junction between
  the coxa (Cx) and thorax".

**Constraint found:** neither dataset resolves chordotonal subtypes. Types
are opaque systematic identifiers (`SNpp39`–`SNpp60`), `flywireType` is
empty, and no field names the organ or joint. A claw/hook/club split would
have to be invented, so it is not built — one combined position+movement
signal drives each leg's whole chordotonal pool, labelled as our choice.

**`g_fb` range set from measurement, not guessed.** Computed the actual
per-neuron thresholds of the 481 leg proprioceptors under Pugliese's own
`prepare_neuron_params`: median 3.10 (IQR 1.97–4.82), well below DNg100's
115 because of `set_sizes` size-scaling. Resulting activation: `I=5` →
76% of the pool active at ~4 Hz; `I=10` → 95% at ~16 Hz; `I=20` → 99.6% at
~38 Hz; `I=40` → ~80 Hz, saturating. Sweep set to {0, 2.5, 5, 10, 20, 40}
to bracket that range.

**Compute budget measured, not estimated.** Physics stepping costs 18.6 µs
per step — 0.4 s per 2 s trial, negligible beside the 11.2 s neural cost.
Trial duration therefore raised from Pugliese's 2 s to **4 s**, giving ~38
rhythm cycles post-transient instead of ~16. Full 920-trial matrix ≈ **6.1
hours**. No reduced matrix needed.

**Side normalisation** defined as `s(g) = N_ref(segment, subclass) / N(g)`,
equalising total drive between a segment's left and right pools while
preserving real front/middle/hind differences and keeping per-neuron
currents in a sensible range (worked example in `SENSORY_MAP.md` §4). C4
added as the un-normalised control.

Written: `SENSORY_MAP.md`, `PREREGISTRATION.md`. Four open implementation
items recorded in `PREREGISTRATION.md` §8 — none of them may change §1–§5.

**No main-experiment trial has been run.**

---

## 2026-09-19 — De-duplication, ball world, controls, sensory interface

### De-duplicating the motor neurons (and finding why they duplicated)

The 16 repeated rows were not a data-entry quirk. `mancBodyid` is **not a
1:1 cross-reference**, so `long_df.merge(malecns, on="manc_bodyId")` in
`identify_all_leg_circuits.py` fanned single MANC neurons out into several
rows — a classic one-to-many join blow-up — each then counted again in
every per-leg sum.

Several of the extra matches were plainly wrong: MANC *motor* neurons
matching MaleCNS **sensory** neurons (`SNta02,SNta09`, `SNpp45`, `SNta29`)
or interneurons (`IN13A030`, `INXXX471`, `IN20A.22A009`).

Fixed at source with `client.collapse_malecns_matches`: prefer the match
whose MaleCNS **`superclass`** fits the role (`vnc_motor` for a motor
neuron), else tie-break deterministically on lowest bodyId, and record
`malecns_n_matches` / `malecns_ambiguous` so ambiguity stays visible.

First attempt used `class` and resolved **zero** of the 17 multi-matches —
`class` is NaN for every MaleCNS motor neuron. Only `superclass` carries
the distinction. With `superclass`, 5 resolve by cell class and 12 remain
genuine ties (both candidates equally plausible, e.g. two different
`Ti flexor MN`s).

Result: 367 → 350 rows, motor neurons 346 → **330, zero duplicates**. Also
surfaced **9 MANC motor neurons whose only MaleCNS match is non-motor** —
no effect on dynamics (the sim indexes by MANC bodyId) but their MaleCNS
identity should not be asserted.

### E0 baseline recomputed

`fly_robot/experiments/open_loop_coordination_baseline.py`, now a proper
script rather than ad-hoc analysis, with an assertion that the circuit map
contains no duplicate bodyIds.

| | before de-dup | after |
|---|---|---|
| ipsilateral significant | 52.0% | **51.8%** |
| ipsilateral median PLV | 0.535 | **0.535** |
| contralateral significant | 16.9% | **17.3%** |
| contralateral median PLV | 0.155 | **0.156** |
| tripod index (median) | −0.221 | **−0.229** |

Everything moved by less than one percentage point. **The finding was never
an artifact of the duplicates** — worth knowing, since it was the obvious
thing to suspect.

### Tethered-on-ball world

Built `fly_robot/bodies/tethered_ball.py`. Three real bugs on the way,
each caught by checking rather than assuming:

1. **Nothing collided.** A sphere placed under the feet contacted nothing
   at all, silently. FlyGym gives every fly geom `contype = conaffinity =
   0` and wires contact through explicit `<pair>` elements instead, so
   contype/conaffinity masks do nothing. Fixed by inheriting FlyGym's own
   `_GroundContactMixin` with the ball registered as the ground geom, so
   foot-ball physics is identical to its flat-ground world. Had to override
   `_attach_fly_mjcf` rather than call `super()`, because the mixin's
   version adds a **free joint** — precisely what a tethered world must not
   have.
2. **Fitted the ball to the wrong pose.** Tarsus positions were measured
   straight from `world.compile()`, which is not the settled neutral
   stance. Real tips are at z ≈ +0.65…+0.91, not −0.38…−0.91 — a ~2 mm
   error. Re-measured after `reset()` + `warmup()` + 0.2 s of settling.
3. **Geometric tangency is not contact.** At the best-fit height, the four
   feet sitting 0.079 mm *above* the surface register nothing and only the
   two middle legs touch. Swept centre height in 0.04 mm steps: all six
   feet contact, with zero penetration, for centre_z ∈ [−1.66, −1.62].
   Took −1.64, the middle of that band.

Also found the centre needs a small **forward** offset (+0.16 mm): centred
on the body axis, the front legs never reach the ball at all. Final
parameters: R = 3.0 mm, 3.4 mg, centre (0.16, 0, −1.64), MuJoCo `ball`
joint, damping 1e-6.

Verified: 6/6 feet contact at rest and throughout a 4 s driven gait; under
a synthetic tripod drive the ball turns about **pitch** with roll and yaw
≈ 0 (straight-line forward walking); no NaN. Visual check rendered and
inspected — the fly sits on top of the sphere with legs wrapping down
around it, body clear of the surface.

### C1 — degree-preserving shuffle

Pugliese's own `shuffle_utils` permutes whole columns within a class,
which is not the degree-preserving edge swap specified, so this is a new
implementation.

Two bugs, both caught by asserting rather than trusting:

1. First version permuted targets wholesale and repaired collisions
   afterwards. `csr_matrix` **silently sums** duplicate entries, so 15
   edges vanished and out-degree errors reached 11. Replaced with
   rejection: a swap that would create a self-loop or duplicate is simply
   not taken, making the guarantee structural.
2. Even then, **9,014 edges collapsed** — two swaps *within the same round*
   can each be individually valid yet independently create the same new
   edge, since both are checked against a snapshot taken before the round.
   Fixed by also rejecting swaps whose new edges collide with another
   accepted swap's in the same round.

Dale's law verified first, since sign-class swapping depends on it: **0 of
22,769** presynaptic neurons have outgoing edges of both signs.

Final: 6,491,786 of 6,862,020 swaps accepted (94.6%), 99.99% of targets
changed, out- and in-degree preserved with **max error 0**, weight multiset
identical, per-neuron (excitatory, inhibitory) counts identical,
deterministic per seed, 1.6 s. Self-loops 16 → 0 (existing ones can be
swapped away; new ones are never created) — reported, not hidden.

### C2 — rate-matched noise

Phase-randomised surrogates of each channel's own recorded E3 drive.
Over 200 draws × 6 channels: mean correlation with the real signal
**−0.011**, mean preserved to 4e-17, std ratio 1.000, amplitude spectrum
to 1e-16. Single draws vary widely (|r| median 0.61) because a narrowband
signal keeps its frequency under phase randomisation — which is why the
null needs many draws, not one.

### Sensory interface

`fly_robot/interface/joint_to_sensory_neuron.py` — the mirror of
`motor_neuron_to_joint.py`. 472 neurons driven (394 chordotonal, 78 hair
plate), matching `SENSORY_MAP.md` exactly.

**Reference scales measured, not assumed.** Drove the frozen motor
interface with a real representative replicate and recorded every joint:
99th-percentile excursions reach 0.275 rad and velocities 9.8 rad/s, with
most legs far smaller (right-hind is driven ~10× harder than the rest,
consistent with its 14 active motor neurons against 1–2 elsewhere). Set
`POSITION_REF_RAD = 0.3`, `VELOCITY_REF_RAD_S = 10.0`.

Deliberately **not** the motor interface's theoretical ±0.5 rad maximum:
most legs move far less, so a 0.5 rad reference would squash the real
signal into the bottom of the encoder's range — the identical failure mode
that made `DEFAULT_RATE_SCALE_HZ = 20` wrong on the motor side.

Side normalisation verified: left and right pools of each segment receive
equal **total** drive (T1 40.0/40.0, T2 72.0/72.0, T3 85.0/85.0) while
per-neuron currents stay in a firing range.

### Validation gates

`tests/test_sensory_interface.py`, **13 passed**. Locality, determinism,
`g_fb = 0` → exactly zero, linearity in gain, boundedness under absurd
input, and a positive check that C3's permutation really is the only
cross-leg path (a control that silently behaved like the real condition
would be worthless).

**No main-experiment trial has been run.** The one piece still missing is
the coupled neural↔physics loop runner.

---

## 2026-09-19 — Rhythm-first gating, C1/C1b, and the coupled loop runner

### User decisions recorded

1. Build the coupled loop runner. Gates before main runs: (a) `g_fb = 0`
   in the full runner reproduces open-loop output exactly at the same
   seed; (b) pilot on pilot seeds only; (c) report timing and any
   instability. **Stop after the pilot.**
2. C1 becomes the **protected** variant — hold out incoming edges to motor
   neurons, outgoing edges from sensory neurons, and DNg100's outgoing
   edges; shuffle the rest with the same verified procedure. Add **C1b** =
   shuffle-everything as a secondary control.
3. Pre-register that **rhythmicity is evaluated first**; conditions without
   a significant rhythm are reported as "no rhythm" and coupling metrics
   are not computed for them.
4. Update `PREREGISTRATION.md` and commit **before** the pilot.

### Rhythmicity gate

Added to `interleg_coordination.py`. A leg is rhythmic only if its in-band
spectral peak strength beats the 95th percentile of an **AR(1) null**
matched to its own variance and lag-1 autocorrelation.

The choice of null mattered and is not the obvious one: the
phase-randomised surrogate used for the *coupling* test preserves the
amplitude spectrum **exactly**, so it carries the same spectral peak as
the real signal and can never test whether that peak is real. Red noise
reproduces the autocorrelated background a non-oscillating rate signal
has, and asks whether the peak stands above it.

**A first calibration attempt looked catastrophic** — an 11 Hz rhythm
detected only 10% of the time at "SNR 2". That turned out to be a broken
*test*, not a broken gate: the synthetic signal added an oscillation to
AR(1) noise in a way that corrupted the AR(1) refit, and its nominal SNR
bore little relation to in-band SNR. Re-run properly, by burying a **real**
strongly-rhythmic trace in controlled noise:

| | |
|---|---|
| detection at 1x / 2x / 4x / 8x the signal's in-band amplitude | 100% / 100% / 94% / 22% |
| false positives, white noise | 5.0% (nominal 5%) |
| false positives, red noise (φ = 0.9) | 3.3% |
| on real published data | 66.7% of active leg-trials pass |

On real data the rejected legs have peak strength 0.18-0.43 against a
stable threshold near 0.49; accepted ones sit at 0.70-0.97. Worth
recording that the first calibration was wrong and was caught by checking
against real data rather than trusting the synthetic number.

### C1 protected / C1b

Protected sets: 87,534 edges into motor neurons + 236,224 out of sensory
neurons + 1,043 out of DNg100 = **320,551 of 1,372,404 (23.4%)**. With
them held out the shuffle still changes 76.6% of targets, preserves in-
and out-degree with **max error 0**, preserves the weight multiset, and
leaves every protected edge present and unchanged (verified by set
comparison, not assumed).

Rationale recorded in `PREREGISTRATION.md` §4b: shuffling those edges
would randomise the *interface* rather than the circuit, so a coordination
loss could come from the per-leg readout ceasing to mean "this leg"
instead of from coupling genuinely failing.

### The coupled loop runner

`fly_robot/sim/closed_loop.py` plus `fly_robot/sim/trial_setup.py`. One
iteration is one neural step of 1 ms containing 10 physics substeps (exact
integer ratio, asserted). Sensory current is held constant across a neural
step — the same zero-order hold the motor side already uses.

One piece needed writing rather than reusing: the frozen motor interface's
`compute_joint_targets` operates on a whole precomputed
(n_neurons, n_timesteps) trace, which a closed loop does not have because
the next timestep does not exist yet. `_joint_targets_from_rates` applies
the identical rule to one instantaneous rate vector, importing the frozen
module's constants rather than restating them, and a test asserts it
reproduces the batch function to 1e-12. The frozen module itself is
untouched.

Sensory feedback enters exactly where the stimulation current already
enters — an additive term in the input vector `I`. **W is never touched.**

### Gates (`tests/test_closed_loop.py`, 7 passed)

- **`g_fb = 0` reproduces open loop bit-identically**, in both the harness
  and the ball condition — the decisive gate. Arrays compared with
  `array_equal`, not `allclose`.
- Single-step motor rule matches the frozen batch rule to 1e-12.
- Determinism: same seed and config, twice, identical.
- **Feedback actually changes the neural trajectory** at `g_fb = 20` — a
  positive check, because a disconnected feedback path would otherwise
  produce a null result by construction and look like a finding.
- Ball state recorded on the ball, absent in the harness.
- No instability at default settings.

### Timing

`build_trial_components` 6.1 s; a full 4 s trial 23-24 s, matching the
audit's projection. Full 940-trial matrix projects to ~6.3 hours.

---

## 2026-09-20 — PILOT RESULTS (pilot seeds only; stopped after, as instructed)

36 trials: parameter seed 641, replicates 0–3, conditions E0 / E1 / E2 and
E3 across the full pre-registered `g_fb` sweep. **No main-experiment seed
was touched.** Raw output in `media/closed_loop/pilot/`.

### (c) Timing

| | |
|---|---|
| per trial (4 s simulated) | **24.9 s** (range 23.5–26.0) |
| `build_trial_components` | 6.1–7.0 s per replicate |
| 36 trials | 16.1 min |
| **projected full matrix (940 trials)** | **6.5 hours** |

Matches the audit's 6.3 h projection. Compute is not a constraint.

### (a) `g_fb = 0` gate — passes at condition level too

E0 (harness), E2 (ball) and E3 (ball, feedback disabled) produce
**identical** `n_active` and `n_rhythmic` for every replicate:

| replicate | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| n_active (all three conditions) | 522 | 380 | 4176 | 367 |
| rhythmic legs (all three) | 5 | 3 | 3 | 4 |

The body differs between these conditions; the neural path does not. This
is the same gate `tests/test_closed_loop.py` asserts bit-identically, now
confirmed end-to-end on full 4 s trials.

Replicate 2 is oversaturated (4176) **at baseline, with no feedback at
all** — a pre-existing unstable draw, not something the loop caused. Under
the pre-registered filter (`n_active ≤ 1500`) it would be excluded. 1 of 4
here against a known base rate of 10/128; small-sample noise.

### (b) Stability — and a hard finding that blocks the main runs

**Nothing went numerically unstable.** 0 of 36 trials flagged; no NaN, no
divergence, no physics blow-up; peak firing rate 235.6 Hz against a
1000 Hz clip.

But the network does something else, and it invalidates the
pre-registered sweep:

| `g_fb` | median n_active | fraction oversaturated | median rhythmic legs |
|---:|---:|---:|---:|
| 0 | 451 | 0.25 | 3.5 |
| **2.5** | **462** | **0.25** | **3.5** |
| **5** | **3983** | **1.00** | **1.0** |
| 10 | 4095 | 1.00 | 1.0 |
| 20 | 3708 | 1.00 | 1.5 |
| 40 | 4846 | 1.00 | 1.5 |

Per replicate, E3:

| replicate | 0 | 2.5 | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 522 | 539 | 4318 | 4650 | 3790 | 4058 |
| 1 | 380 | 386 | 3578 | 4051 | 3626 | 4804 |
| 3 | 367 | 379 | 4492 | 3569 | 4109 | 4888 |

**This is a bifurcation, not a dose-response.** `g_fb = 2.5` changes
`n_active` by about 3% (522→539, 380→386, 367→379) — barely distinguishable
from no feedback. `g_fb = 5` jumps it roughly tenfold and the rhythm
collapses from a median of 3.5 rhythmic legs to 1. Every level at or above
5 sits in that saturated regime. The network flips into a self-sustaining
high-activity state: runaway excitation, not instability in the numerical
sense.

**So the pre-registered sweep {0, 2.5, 5, 10, 20, 40} has effectively two
usable points, one of which is nearly a no-op.** Four of six levels
destroy the very rhythm the experiment measures.

### Why the earlier calibration missed this

`g_fb` was set from **single-neuron steady-state activation**: with
threshold median 3.10, `I = 5` activates 76% of the sensory pool at ~4 Hz,
`I = 10` 95% at ~16 Hz. That arithmetic is correct and irrelevant — it
ignores what those neurons then do downstream. 472 sensory neurons with
median out-degree 22–44, injecting into a network already driven by
DNg100, deliver an aggregate drive roughly an order of magnitude beyond
what the single-neuron figure suggests.

The same class of error as the `rate_scale_hz = 20` and 0.5–10 Hz band
mistakes: a number derived from the right formula applied at the wrong
level of description. Caught here by the pilot, which is what it was for.

**Stopped and reported rather than re-deriving the sweep unilaterally** —
changing a pre-registered parameter is the user's call.

---

## 2026-09-20 — Encoder variants A and B: both NOT usable; rule applied

Amendment 2 committed at `809639d` **before** these runs. Pilot seeds only.

### Verification done before writing the amendment

- **Range fractionation checked against primary text**, not a search
  summary: bioRxiv 2022.08.08.503192, verbatim — *"Fractionation of the
  tibia joint angle range across position-tuned proprioceptors has been
  previously described in the grasshopper FeCO"* and *"The cell bodies of
  position-tuned proprioceptors form a goniotopic map of joint angle"*.
  Also noted from the same paper that *"calcium imaging from position-tuned
  axons failed to resolve any topographic organization"*, so no claim is
  made about VNC arrangement.
- **Cap value measured, not chosen**: per-neuron current across 6.6M
  stable-regime neuron-timesteps — p95 1.062, p99 1.589, p99.9 2.251,
  max 2.386; runaway regime median 2.0–2.1, max 8.5. Cap 2.5 sits above
  everything stable, so it cannot change sub-cliff behaviour.
- **Fractionation width verified**: 22.0% of each chordotonal pool
  responds above half-maximum at mid-range against 23.5% theoretical,
  14–15% at the range extremes (Gaussian truncated at the boundary).
- **Recorded in advance** that fractionation reintroduces rest drive
  (101.7 vs 0.0), since position-tuned neurons tuned to the resting angle
  correctly fire there. Stated before the run rather than after.

### Result — both variants NOT USABLE, 0 of 7 levels each

| | rhythm-preserving levels | levels also passing the effect floor |
|---|---:|---:|
| `deviation_capped` | 6 | **0** |
| `deviation_fractionated` | 5 | **0** |

Where the rhythm survives, the effect size is 0.0000–0.0010. Where the
effect is large (≈1.01, total decorrelation) the rhythm is gone. The
bootstrap CI criterion was met at some mid levels, but the median-effect
floor of 0.05 failed everywhere the rhythm survived — by two orders of
magnitude.

### Mechanism, measured rather than inferred

**The sensory neurons essentially never fire.** The fraction of the 472
proprioceptors whose input exceeds its own threshold is 0.0% at g_fb = 1
and only 1.1% at g_fb = 4.5. Median threshold 3.06; typical per-neuron
current ~0.1, peak 2.4 — about **30x below threshold on average and still
below it at the peak**.

So sub-cliff feedback is injected and recruits nobody; the 0.1%
trajectory change is sub-threshold current nudging the dynamics without
producing activity. The cliff is where transient excursions finally cross
threshold — near-simultaneously, because the thresholds are narrowly
distributed — and 472 neurons with out-degree 22–44 firing at once
saturates the network.

The mismatch is between the *distribution* of drive over time (near zero
mostly, occasionally saturating) and a narrow threshold distribution.
Recorded as a hypothesis for future work only — **no further encoder
variants**, per the pre-registration.

### Written up

`docs/closed_loop/RESULTS.md` — pilot results, the decision rule applied
verbatim, the mechanism, the inhibition corollary (46% of edges are
inhibitory and present throughout; real inhibition does not prevent the
runaway), and an explicit list of what the result does and does not mean.
H1/H2/H3 remain **untested**; the pilot establishes they are not testable
with this interface.

---

## 2026-09-20 — Code dedup + a stale-numbers bug found while verifying it

### Dedup (clean-up pass, no behavior change)

Two independent implementations of "per-leg motor-neuron row indices,
de-duplicated" existed: `_leg_motor_row_indices` in `sim/closed_loop.py`
(derived from `MotorNeuronGroups`) and `build_motor_indices` in
`experiments/open_loop_coordination_baseline.py` (read the circuit CSV
directly). Centralized as `leg_motor_row_indices` in
`interface/motor_neuron_to_joint.py`, next to `build_motor_neuron_groups`
which is the actual source of truth both call sites build from. Verified
the two old implementations returned bit-identical index sets before
removing either (not just matching counts — checked set equality per
leg). Also removed two other pieces of dead code found while at it:
`PULSE_START_S`/`PULSE_END_MARGIN_S` constants in `closed_loop.py` that
were defined and never read (real pulse timing comes from
`sim_params.pulse_start/pulse_end`, built by `trial_setup.py` from the
actual Hydra config), and an unused `record_every` parameter on
`run_trial`.

### While verifying the dedup: found AUDIT.md §6 was stale

Re-ran `open_loop_coordination_baseline.py` (unchanged by the dedup — the
index sets are identical) to confirm the E0 numbers still matched
`AUDIT.md` before committing. **They didn't**: current output gives
ipsilateral n=227/54.6%/PLV=0.906 against the committed 521/51.8%/0.535 —
not a small drift, a large one, in both trial count and PLV.

Isolated with `git stash`: the *original, unmodified, committed* script
gives the same 227/54.6%/0.906 as the current one, reproducibly across
two independent runs — so this was never caused by today's dedup. Traced
via `git log` on `interleg_coordination.py` (only two commits): `AUDIT.md`
§6's table was written from commits `7d7c995`/`de3a231`, **before**
`83ec15b` changed `coordination()`'s definition of `valid` from
`active[i] and active[j]` to the stricter, pre-registered
`rhythmic[i] and rhythmic[j]` (AR(1)-gated). The audit table was never
recomputed after that methodology change and silently drifted out of
sync with every other analysis in this project, which already uses the
rhythm-gated version (`PREREGISTRATION.md` §4a, every pilot run).

**The finding survives, more strongly.** Recomputed on the same 118
stable replicates with current code:

| | was (stale) | now (current methodology) |
|---|---:|---:|
| ipsilateral significant / PLV | 51.8% / 0.535 | **54.6% / 0.906** |
| contralateral significant / PLV | 17.3% / 0.156 | **16.8% / 0.130** |
| diagonal significant / PLV | 16.5% / 0.146 | **17.5% / 0.115** |
| tripod index (median) | −0.229 | **−0.280** |

Fewer pairs qualify as "valid" under the stricter gate (ambiguous,
weakly-rhythmic legs are excluded entirely rather than diluting the
median), but among survivors ipsilateral phase-locking is *more*
pronounced, not less. Contralateral/diagonal and the tripod index barely
move. The stop condition, the H1 reframing, and everything decided on the
strength of this finding stand.

**Fixed, not silently**: `AUDIT.md` §6 keeps its original table with a
dated, clearly-marked superseding note underneath (per this project's
rule — an amendment is additive and dated, never an edit that erases what
was there); `PREREGISTRATION.md` §0.1 now points to `AUDIT.md` rather
than repeating figures that could drift again. Caught by verifying a
refactor's output against the committed record before trusting either —
exactly the discipline that surfaced the v1/v2 paper-version mixup and
the wrong replicate-instability threshold earlier in this project.

All 20 existing gates (`pytest tests/`) still pass after the dedup.
