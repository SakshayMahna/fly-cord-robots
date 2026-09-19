# Closed-loop audit: what we actually have before adding sensory feedback

*Written 2026-09-19. This is step 1 of the closed-loop work — a factual
inventory of the existing pipeline, with every number queried from the data
or measured from a run, not recalled. Nothing new was built for the
experiments here; the one new module (`steppable_rate_model.py`) exists
because the audit found the old one could not be stepped at all, and is
itself audited below.*

> **A note on the folder name.** The task description for this work called
> for a `phase4/` folder. That collides with a standing project decision to
> drop phase-numbered file and folder names in favour of names derived from
> the work itself (see `docs/logs/2026-09-19.md`, "Second reorganization").
> This folder is `docs/closed_loop/` for that reason. The mapping is:
> `phase4/AUDIT.md` → `docs/closed_loop/AUDIT.md`, and likewise
> `SENSORY_MAP.md`, `PREREGISTRATION.md`, `RESULTS.md`, `LOG.md`.

---

## 1. Dataset and version

**There are two connectomes in this project and they are used for different
things.** This is the single most important thing to get straight before
reading anything below.

| | Used for | Version | Where |
|---|---|---|---|
| **MANC** | The **simulation** — every firing rate we have ever computed | `wTable_20251006.feather` / `W_20251006.feather` | `external/Pugliese_cpg_2025/data/manc full vnc data/` |
| **MaleCNS** | **Neuron identification only** — cross-referencing, 3D renders | `male-cns:v1.0` via neuprint | `data/circuit_map/*.csv` (`malecns_bodyId` column) |

The project's standing decision (`CLAUDE.md`, 2026-09-18) is to use MaleCNS
going forward. That decision has **not** yet reached the dynamics: every
simulated result to date, including everything in this audit, runs on
Pugliese's MANC matrix. `CLAUDE.md` already flags this as a known open gap
("a from-scratch run on MaleCNS-derived weights specifically hasn't been
done"). The closed-loop experiments inherit it. It is a limitation to state
in any write-up, not a bug.

### Network actually simulated

Queried from `wTable_20251006.feather`:

- **23,532 neurons** — matches the count the paper states for its full-VNC run.
- **W is 0.248% dense**: 1,372,404 nonzero entries out of 553,754,  224.
- Class composition:

| class | n |
|---|---:|
| intrinsic neuron | 13,044 |
| **sensory neuron** | **5,891** |
| ascending neuron | 1,849 |
| descending neuron | 1,322 |
| **motor neuron** | **732** |
| sensory ascending | 534 |
| efferent neuron | 92 |
| (other / NaN) | 68 |

**The sensory neurons are already in the simulated network.** They are
currently silent only because nothing drives them — no code change to the
network is needed to give them input. This is the finding that makes the
closed loop possible at all.

---

## 2. DNg100 and the CPG neurons, as found in the data

Stimulation targets are rows **59** and **282** of the wTable. Verified
directly rather than assumed:

| wTable row | MANC bodyId | type | instance | class | predicted NT |
|---:|---:|---|---|---|---|
| 59 | 10093 | DNg100 | DNxl058_CvC_L | descending neuron | acetylcholine |
| 282 | 10339 | DNg100 | DNxl058_CvC_R | descending neuron | acetylcholine |

These row indices and the stimulus magnitude come from Pugliese's own
`run_config.yaml` files inside their Zenodo archive
(`data/pugliese_published_sim/.../logs/run_config.yaml`), not from the paper
text — the paper does not state them.

CPG triads, from `data/circuit_map/all_legs_circuit.csv`:

| leg | side | hub (IN17A001) | excit2 (INXXX466) | inhib (IN16B036) |
|---|---|---:|---:|---:|
| T1 | LHS | 10707 | 11751 | 13905 |
| T1 | RHS | 10690 | 13698 | 14096 |
| T2 | LHS | 10559 | 11767 | 13186 |
| T2 | RHS | 10072 | 12107 | 17322 |
| T3 | LHS | 10498 | 12315 | 12953 |
| T3 | RHS | 10558 | 152696 | 156245 |

All 21 rows are `status == Traced`.

### Two labelling quirks found during the audit

1. **DNg100 is tagged with a leg in the circuit map** (10339 as `T2/RHS`,
   10093 as `T3/LHS`). DNg100 is not leg-specific — this is an artifact of
   `identify_all_leg_circuits.py` recording which leg's triad each DN copy
   was found driving most strongly. Harmless, but do not read it as a claim
   about DNg100's anatomy.
2. **One MANC neuron resolves to two different MaleCNS neurons with
   different type strings.** T3-LHS `CPG_excit2`, MANC bodyId 12315, maps to
   both MaleCNS 801483 (`IN03A031`) and 801149 (`INXXX466`). Since the
   simulation is driven by MANC row indices, this does not affect any
   dynamics — but it means the MaleCNS identity of that one neuron is
   **ambiguous** and should not be asserted.

---

## 3. Leg motor neurons: grouping and where it comes from

The joint/flexor-extensor grouping is **not ours**. It is read directly
from the `motor module` column of Pugliese's own `wTable_20251006.feather`
— a pre-existing MANC annotation. We did not infer, cluster, or assign it.

Counts per leg (rows in `all_legs_circuit.csv`). `*` marks the modules the
frozen motor interface actually maps to a joint:

| motor module | T1-L | T1-R | T2-L | T2-R | T3-L | T3-R |
|---|---:|---:|---:|---:|---:|---:|
| `*` coxa stance | 6 | 7 | 7 | 7 | 8 | 6 |
| `*` coxa swing | 7 | 8 | 4 | 3 | 3 | 3 |
| `*` femur/tr extend | 8 | 8 | 5 | 5 | 6 | 7 |
| `*` femur/tr flex | 11 | 13 | 9 | 9 | 9 | 11 |
| `*` tibia extend | 2 | 2 | 2 | 2 | 2 | 2 |
| `*` tibia flex | 17 | 16 | 12 | 11 | 12 | 11 |
| femur reductor | 6 | 6 | 3 | 3 | 2 | 2 |
| substrate grip | 9 | 8 | 6 | 9 | 8 | 8 |
| tarsus control | 7 | 8 | **0** | **0** | **0** | **0** |

### Gaps and uncertainties, stated plainly

- **6 of 9 modules are mapped to a joint; 3 are not** (femur reductor,
  substrate grip, tarsus control). Pre-existing, documented limitation of
  the frozen interface — see `motor_neuron_to_joint.py`.
- **`tarsus control` is annotated for T1 only.** Zero neurons for T2 and T3.
  This is an annotation gap in the source data, not a filtering bug on our
  side.
- **`tibia extend` has exactly 2 neurons per leg**, against 11–17 for
  `tibia flex`. Any antagonist-difference signal on the tibia joint is
  therefore built from a very small extensor pool.
- **16 duplicate rows.** The file has 346 motor-neuron rows but only **330
  unique** MANC bodyIds (T1-L 2, T1-R 5, T2-L 1, T2-R 1, T3-L 4, T3-R 3).
  A neuron listed twice is summed twice in a per-leg readout. Small, but it
  must be de-duplicated before the main runs — **open item**.
- **No motor neuron is shared between legs.** Checked explicitly: all 15 leg
  pairs have empty intersection. This matters because it rules out the most
  obvious artifact behind the coupling result in §6.

---

## 4. Neural model, timesteps, and synchronisation

### The model

Pugliese's rate equation, unchanged:

```
total       = I(t) + W_eff · R
activation  = max( fr_cap · tanh( (a / fr_cap) · (total − threshold) ), 0 )
dR/dt       = (activation − R) / tau
```

`W_eff = reweight_connectivity(W, exc_mult, inh_mult)` — W transposed, with
positive and negative entries scaled separately. **No weight, sign, or
topology is altered anywhere in this project.**

Parameters, from their `run_config.yaml` (all four published runs agree):

| | value |
|---|---|
| tau | 0.02 ± 0.002 s |
| a | 1 ± 0.1 |
| threshold | 7.5 ± 0.6 |
| fr_cap | 200 ± 10 Hz |
| excitatory / inhibitory multiplier | 0.03 / 0.03 |
| stimulus current | 380, into rows 59 and 282 |
| pulse window | 0.02 – 1.999 s |
| T, neural dt | 2.0 s, 0.001 s |
| published solver tolerances | rtol 2e-6, atol 5e-9 |
| seeds | 641, 98, 445, 424 (32 replicates each = 128) |

Per-neuron parameters are **resampled per replicate**, then size-scaled by
`set_sizes` (`a /= size`, `threshold *= size`). A replicate is a stochastic
sample, not a deterministic answer.

### Timesteps

| | value |
|---|---|
| neural dt | 0.001 s |
| physics dt | 0.0001 s (measured from `Simulation.timestep`) |
| ratio | exactly 10 |

Synchronisation is **zero-order hold**: each neural sample is held constant
across 10 physics substeps. The existing open-loop script asserts the ratio
is an exact integer and refuses to run otherwise.

### The solver problem — and what was done about it

**This is the main thing the audit changed.** Their `run_single_simulation`
hands the entire 2 s window to one adaptive `diffrax` Dopri5 call. A closed
loop has to inject new input between steps, which that structure cannot do.

Measured on this machine (M3, CPU):

| approach | cost per 2 s trial |
|---|---|
| their adaptive solver, restarted every 1 ms | **79 min** |
| fixed-step RK4, dense matvec | 3.5 min |
| **fixed-step RK4, sparse matvec** | **11.2 s** |

`fly_robot/neural/steppable_rate_model.py` takes the third option. It
changes exactly three things — integrator (fixed-step RK4), matrix
representation (CSR sparse; 2.06 GiB → 11 MiB), and allowing `I` to vary
over time. The equation, the connectome and every neuron parameter are
untouched.

### Validation against their published output

`fly_robot/neural/validate_steppable_model.py` rebuilds a published
replicate's exact neuron parameters using **their own**
`prepare_neuron_params` with the seed from their `run_config.yaml`, runs it
open-loop, and compares to their published trace. This also confirmed the
seed→replicate mapping empirically.

| replicate (seed) | recruited: ours / theirs / shared | Jaccard | median r | median peak err |
|---|---|---:|---:|---:|
| 0 (641) | 608 / 608 / 608 | 1.000 | 0.9993 | 0.0007 Hz |
| 40 (98) | 690 / 690 / 689 | 0.997 | 0.9988 | 0.0016 Hz |
| 100 (424) | 303 / 303 / 303 | 1.000 | 0.9997 | 0.0003 Hz |

**Recruitment is reproduced exactly or near-exactly.** Where traces differ,
they differ in *late-window phase*:

| window | median r | frac r > 0.99 |
|---|---:|---:|
| 0.0–0.5 s | 0.9997 | 1.00 |
| 0.5–1.0 s | 0.9995 | 1.00 |
| 1.0–1.5 s | 0.9995 | 1.00 |
| 1.5–2.0 s | 0.9911 | 0.51 |

**This drift is not our integrator being sloppy.** Two of *our own* runs at
4× different step sizes agree with each other (median r = 0.9969) no better
than ours agrees with theirs (0.9988), and refining the step made agreement
slightly *worse*, not better. The system is genuinely sensitive: oscillators
accumulate phase difference between any two numerically-distinct runs.

Consequences, stated rather than glossed:
- Absolute phase at t = 2 s is **not** reproducible across integrators.
- Our own runs are bit-deterministic for a given seed and config.
- All conditions being compared use the same integrator, so this is a shared
  systematic property rather than a confound between conditions.
- Whether it moves the *coordination metric* is checked in §6.

---

## 5. Frozen motor interface

Frozen for the whole of the closed-loop work, per the task's hard rules.

| | |
|---|---|
| module | `fly_robot/interface/motor_neuron_to_joint.py` |
| source sha256 | `8319c2a78d2a67a9d68a9f5f2ff8d71ccce9b243aaf62f49ac2d6ec0a41e5823` |
| **config sha256** | **`04be9dec181ec2e20fad91ba07cfe018d0a413720b0dd441a2dd8aab7294fc1d`** |
| git commit | `c2c6409afd1be573cb22285e68925353945e0a8c` |

The config hash covers the antagonist pairs, the leg-name map, and both
gains:

```json
{"ANTAGONIST_PAIRS": [["coxa swing","coxa stance","coxa-pitch"],
                      ["femur/tr extend","femur/tr flex","trochanterfemur-pitch"],
                      ["tibia extend","tibia flex","tibia-pitch"]],
 "DEFAULT_GAIN_RAD": 0.5, "DEFAULT_RATE_SCALE_HZ": 3.0,
 "LEG_MAP": [[["T1","LHS"],"lf"],[["T1","RHS"],"rf"],[["T2","LHS"],"lm"],
             [["T2","RHS"],"rm"],[["T3","LHS"],"lh"],[["T3","RHS"],"rh"]]}
```

Rule: `joint = neutral + 0.5 rad · tanh( (rate_pos − rate_neg) / 3.0 Hz )`.

Body: FlyGym/NeuroMechFly v2, **42 actuated position DOFs, 7 per leg**,
actuator gain 50. Of the 7 per leg, 3 are driven (coxa pitch,
trochanterfemur pitch, tibia pitch); the other 4 (coxa roll, coxa yaw,
trochanterfemur roll, tarsus pitch) hold neutral.

---

## 6. Open-loop baseline (E0) — and a result that trips a stop condition

**Method.** Pugliese's own 128 published full-VNC replicates; 118 pass the
stability filter (`n_active ≤ 1500`, their documented criterion). Per-leg
readout = **summed** firing rate over that leg's motor neurons (sum, not
mean: only 2–10 of each leg's ~50–70 motor neurons are recruited per
replicate, so a mean mostly measures how many are silent). First 0.5 s
discarded. Bandpass, Hilbert phase, pairwise phase-locking value, tested
against **phase-randomised surrogates** of the same signals.

### A measurement error found and fixed

The analysis band was first set to 0.5–10 Hz, citing "0.9–6.5 Hz" from
`docs/logs/2026-09-19.md`. **That figure is a firing *rate* range, not an
oscillation frequency** — two different quantities that are both in Hz. The
band cap sat below the real rhythm, so every "dominant frequency" it
reported was meaningless leftover power.

Measured properly over all 581 active leg-trials, searching a deliberately
over-wide 0.5–60 Hz so the band could not bias the answer:

- **median 10.7 Hz**, 5th–95th percentile 4.7–13.3 Hz, 97.6% between 2–20 Hz.
- Directly visible in raw traces: mean-upcrossing intervals of **82–86 ms**
  (≈12 Hz), and every active T3-RHS motor neuron peaking at 11.99 Hz with
  92–97% of its in-band power there.
- Consistent with our own Phase 0 numbers from Pugliese's own
  `compute_oscillation_score` (5.7–11.4 Hz per segment,
  `media/simulation/full_vnc_oscillation_scores.json`) and with real
  *Drosophila* stepping (~5–15 Hz).

Band corrected to **2–20 Hz**, documented in the module.

### Test calibration

Run on phase-randomised data with the same spectra and no true phase
relationship, the test reports **7.6% significant** against a nominal 5%.
Slightly liberal; used as the chance baseline below rather than 5%.

### Result

Rhythm is present — all six legs, median dominant frequency 8.7–11.3 Hz,
active in 69–97% of replicates.

| pair class | pairs | valid trials | significant | median PLV |
|---|---:|---:|---:|---:|
| **ipsilateral** (same side, front↔mid↔hind) | 6 | 521 | **52.0%** | **0.535** |
| **contralateral** (left↔right, same segment) | 3 | 243 | 16.9% | 0.155 |
| **diagonal** (opposite side, different segment) | 6 | 498 | 16.7% | 0.147 |

*(chance = 7.6%)*

**Tripod index: median −0.221**; only 14.4% of replicates positive.

Strongest pairs are all same-side: T2-RHS↔T3-RHS (PLV 0.935, 77.6%
significant), T1-RHS↔T3-RHS (0.557), T1-LHS↔T2-LHS (0.559),
T1-RHS↔T2-RHS (0.543). Weakest are the three left↔right pairs
(0.143–0.168).

### What the paper actually claims

From the v2 full text (`biorxiv.org/content/10.1101/2025.09.12.675944v2`),
quoted rather than paraphrased:

> "despite robust rhythmic activity in both legs, no consistent phase
> relationship emerged between **the left and right** coxa promotor motor
> neurons (Fig. 4d,e), and phase coupling was absent across the six leg CPGs
> in the full connectome simulation (Extended Data Fig. 9b)."

> "Our DNg100 simulations did not produce the **tripod** interleg
> coordination pattern characteristic of hexapod walking. Several VNC
> neurons connect **the left and right** CPG circuits disynaptically, but
> their inclusion was insufficient to couple the phase of the left and right
> legs."

The words "ipsilateral" and "contralateral" appear **zero times** in the
paper.

### Reading this honestly

Our baseline **agrees with both specific claims the paper tests**: left↔right
coupling is weak (16.9% vs 7.6% chance, PLV 0.155), and the tripod pattern is
absent (index −0.22).

It **adds a distinction the paper does not report**: same-side
front↔middle↔hind coupling is substantial (52%, PLV 0.535). Their broad
summary sentence — "phase coupling was absent across the six leg CPGs" —
is not what we measure, if that sentence is read as covering all 15 pairs.

Ruled out as explanations: shared motor neurons between legs (no overlap
exists), and test miscalibration (7.6% false-positive rate measured). Not
yet ruled out: whether ipsilateral coupling reflects genuine same-side
connectome structure or common drive from the shared bilateral DNg100
stimulus — though common drive alone would be expected to couple left↔right
pairs just as strongly, and it does not.

**This trips the pre-agreed stop condition** ("open-loop baseline shows
coupling → STOP and report"). Reported here; not resolved unilaterally.

### dt robustness of the metric

The concern raised in §4 — that late-window phase drift might corrupt a
phase-based measure — was tested rather than assumed. Six replicates were
run through our own model at **h = 1 ms and h = 0.25 ms** and the full
coordination analysis applied to both:

| | |
|---|---|
| median \|dominant frequency difference\| | **0.000 Hz** |
| median PLV correlation across pairs | **1.0000** |
| median max \|PLV difference\| | **0.001** |
| significance agreement | **100%** (6/6 replicates, all pairs) |
| tripod index, median | −0.281 vs −0.281 |

One replicate (4) moved: one leg's dominant frequency shifted 1.33 Hz and
its worst PLV changed by 0.08, tripod index −0.363 → −0.462. Every
significance call still agreed.

**The coordination metric is dt-robust even though individual traces are
not.** Phase drift accumulates in the raw traces but does not propagate
into the pairwise phase *relationships*, which is the quantity the
experiments actually depend on. The second stop condition ("results depend
strongly on dt") is **not** triggered.

---

## 7. Sensory neurons available for the closed loop

Preliminary — full detail belongs in `SENSORY_MAP.md`. Recorded here because
it determines whether the closed loop is possible at all.

Leg assignment comes from the **nerve** encoded in the `instance` string
(`SNpp45_MetaLN_L` → MetaLN, left), since sensory somata sit in the leg and
`somaNeuromere` is NaN for 5,889 of 5,891 sensory neurons. Nerve→leg mapping
used: ProLN→T1, MesoLN→T2, MetaLN→T3.

| leg | side | chordotonal | hair plate | campaniform |
|---|---|---:|---:|---:|
| T1 | LHS | 23 | 4 | 0 |
| T1 | RHS | 57 | 8 | 1 |
| T2 | LHS | 54 | 12 | 2 |
| T2 | RHS | 90 | 20 | 2 |
| T3 | LHS | 80 | 9 | 2 |
| T3 | RHS | 90 | 25 | 2 |

**Every one of these neurons has outgoing synapses** in the simulated
matrix (median out-degree 22–44), and each leg's group reaches its own
leg's CPG+motor circuit within two hops via 126–622 intermediate neurons.
So injecting current into them will propagate.

Two problems that need a decision before building the encoders:

1. **Leg campaniform sensilla are effectively absent** — 0–2 per leg,
   against the 163 in the whole dataset, which sit overwhelmingly in the
   ADMN (217) and DMetaN (337) nerves, i.e. wing and haltere fields. The
   planned "foot contact load → campaniform" channel has almost nothing to
   connect to. Proprioception via chordotonal organs is well covered;
   **load sensing is not.**
2. **Strong left/right asymmetry** in sensory annotation — T1 has 23 left
   vs 57 right chordotonals, T2 54 vs 90. Feedback built on these will be
   asymmetric by construction, which is awkward for an experiment whose
   primary question is left↔right coupling.

Both are annotation-coverage properties of MANC, not choices of ours.

---

## 8. Open items

- [ ] De-duplicate the 16 repeated motor-neuron rows before main runs.
- [ ] Decide how to handle near-absent leg campaniform sensilla (§7.1).
- [ ] Decide whether sensory left/right asymmetry (§7.2) needs mitigating.
- [ ] Resolve the E0 ipsilateral-coupling finding (§6) before pre-registering.
- [x] dt-robustness of the coordination metric — **passed** (§6).
- [ ] MaleCNS-derived weights still never simulated (§1) — inherited gap.

## 9. Summary of the two stop conditions

| condition | status |
|---|---|
| "open-loop baseline shows coupling" | **TRIGGERED** — ipsilateral only; left↔right and tripod match the paper (§6) |
| "results depend strongly on dt" | not triggered — metric is dt-robust (§6) |
