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
