# Findings changelog

Chronological log of what we learned/decided each session. This is the
source material for the video script — write it so a future read explains
*why*, not just *what*.

## 2026-09-18 — Dataset choice: MaleCNS over MANC, and locating the DNg100 circuit in it

**Decision:** use MaleCNS (`male-cns:v1.0`, neuprint server
`neuprint.janelia.org`) as our connectome going forward, not MANC.

**Why MaleCNS, and what changes:**
- MANC (2023) is nerve-cord-only, ~23k neurons. Pugliese et al.'s CPG model
  (github.com/smpuglie/Pugliese_cpg_2025) was built on MANC and only ever
  covers the T1 (front-leg) DN→motor-neuron circuit — it does not solve
  our Phase 1 need for all six legs either way.
- MaleCNS (released June 2026, Cell paper Sept 2026) is a different,
  newer specimen: brain + VNC imaged as one continuous volume (including
  the neck connective MANC lacks), ~166,700 neurons. This matters for us
  because it's a superset — same VNC circuitry, plus the brain, which
  keeps later phases (real descending drive instead of synthetic tonic
  stimulation) open without a second dataset migration.
- Since we have to build our own multi-leg circuit extraction regardless
  of dataset, MaleCNS costs us nothing extra structurally and gets us the
  more complete dataset now.

**Caveat (still open):** Phase 0's original "done when" criterion was to
qualitatively match Pugliese's *published rhythm figures*. Those figures
were generated from MANC-derived weights. Running our model on
MaleCNS-derived weights is not the same numerical circuit, even though
the neuron identities match — connectivity/synapse counts between MANC
and MaleCNS for the same neuron pair can differ (different specimen, own
proofreading pass). We have not yet decided whether Phase 0's validation
step should also include a literal MANC run for direct comparison.

**How we identified the circuit in MaleCNS:**
MaleCNS neuron records carry a `mancBodyid` field — a per-neuron
cross-reference to the corresponding MANC specimen body, computed by
Janelia's own annotation team, not by us. We used this instead of
matching on the `type` string, because it's a hard per-neuron reference
rather than a name that could coincidentally collide.

Pugliese's own MANC body IDs for the core circuit are only recoverable
by reading their experiment configs, not published names:
`configs/experiment/DNg100_Stim_CoreCPG.yaml` sets
`keepOnly: [31, 277, 617, 1167]` — these are *row indices* into
`data/manc t1 connectome data/wTable_20250813_DNtoMN_unsorted_withModules.csv`,
which resolve to:

| MANC bodyId | type      | role         | predicted NT   |
|-------------|-----------|--------------|----------------|
| 10093       | DNg100    | command (DN) | acetylcholine  |
| 10707       | IN17A001  | CPG, excit.  | acetylcholine  |
| 11751       | INXXX466  | CPG, excit.  | acetylcholine  |
| 13905       | IN16B036  | CPG, inhib.  | glutamate      |

This matches the paper's description of a minimal 3-neuron CPG (2
excitatory + 1 inhibitory) driven by DNg100.

Querying MaleCNS by `mancBodyid` for these 4 plus all 148 rows tagged
`class == "motor neuron"` in the same table (`fly_robot/connectome/identify_circuit.py`):

- **4/4 CPG+command neurons resolved 1:1**, same `type` string on both
  sides, both `status == Traced`.
- **138/148 T1 leg motor neurons resolved** (see script output; unresolved
  ones span several joint modules — coxa stance, tarsus control, femur
  reductor — not concentrated in one type, so likely a proofreading/ID
  churn gap rather than a systematic mismatch, but not yet investigated).

Output: `data/circuit_map/t1_front_leg_circuit.csv` (gitignored,
regenerate via `python -m fly_robot.connectome.identify_circuit`).

**Not yet done:** T2/T3 (mid/hind leg) circuits — Pugliese's table is
T1-only, so those motor neurons and any analogous CPG will need to be
identified independently (Phase 1 work), not carried over from this table.

## 2026-09-18 — First 3D render of the T1 walking circuit (MaleCNS)

Built `fly_robot/analysis/visualize_circuit.py`: fetches real skeletons
(from neuprint, via `navis`) for the 142 resolved neurons in
`t1_front_leg_circuit.csv` and renders them in 3D, colored by role
(DNg100 = red, CPG excitatory = blue, CPG inhibitory = orange, T1 leg
motor neurons = gray). Dark/light variants, plus a version cropped to
just the VNC leg neuropil with the CPG neurons drawn thicker (otherwise
invisible next to the much larger motor neuron dendritic fields) — see
`media/t1_walking_circuit_*.png`.

This was a real check, not just a picture: if our neuron identification
via `mancBodyid` were wrong, we'd expect to see anatomically nonsensical
results (e.g. neurons scattered outside neuropil, or motor neurons in the
brain). Instead the render shows exactly what's biologically expected:

- DNg100's dendrites arborize in the brain; a single axon descends
  through the neck connective into the VNC (visible as the long thin red
  line spanning the whole image) — correct for a descending neuron.
- The 138 T1 leg motor neurons cluster bilaterally in the two T1 leg
  neuropils, with thin processes exiting toward the leg nerve.
- The 3 CPG interneurons cluster tightly inside one T1 neuropil, right
  where DNg100's axon terminates — consistent with DNg100 driving this
  circuit directly, as the paper describes.
- DNg100 also has a second, denser local arbor inside that same T1
  neuropil beyond just the CPG contact point — plausible for a command
  neuron with direct as well as CPG-mediated motor drive, but this is our
  visual read, not something we've independently confirmed against the
  paper's text.

One implementation note for future reruns: `navis.plot2d`'s
`view=("x", "-z")` inverts the *displayed* axis, it does not negate the
underlying coordinate data — cropping/zooming code must compute bounds
from raw `z`, not `-z`, or the crop silently points at empty space.

## 2026-09-18 — Phase 1: evidence the T1 CPG motif repeats in T2/T3 (hypothesis, not yet confirmed)

Pugliese's model and published circuit cover **T1 only**. Before doing our
own from-scratch circuit discovery for T2/T3, we checked a cheaper
question first: does the *exact same 3-neuron CPG motif* they found in T1
simply recur, unremarked, in the other two leg segments? Built
`fly_robot/connectome/identify_all_legs.py` to test this directly against
real MANC connectivity (Pugliese's own `wTable_20260522_allSynapses.feather`
+ `W_20260522_allSynapses.npz` full-VNC data — not modified, just read).

**Step 1 — naming fact (weak evidence on its own):** each of the 3 CPG
cell types (`IN17A001`, `INXXX466`, `IN16B036`) has exactly 6 instances in
MANC: one per body side × one per thoracic segment (T1/T2/T3). Same
`type` string across segments is a lineage/morphology claim from Janelia's
annotators, not proof they're wired the same way — so we went further.

**Step 2 — actual synaptic weights (real evidence):** for every
leg segment × side (6 total), we automatically paired each hub neuron
with whichever of the two DNg100 copies actually drives it most strongly
(inferred from the weight matrix — not assumed from L/R symmetry), then
read off the full triad's internal weights. All 6 came back structurally
identical to the published T1 circuit:

| leg | side | DN→hub | hub→excit2 | inhib→hub | inhib→excit2 |
|-----|------|-------:|-----------:|----------:|-------------:|
| T1  | LHS  | 187 | 539 | -531 | -172 |
| T1  | RHS  | 217 | 456 | -580 |  -79 |
| T2  | LHS  | 185 | 521 | -526 | -241 |
| T2  | RHS  | 192 | 605 | -558 | -303 |
| T3  | LHS  | 130 | 509 | -588 | -191 |
| T3  | RHS  | 153 | 468 | -561 | -173 |

Every segment/side: DNg100 strongly excites an "excit_hub" neuron, which
strongly excites a second excitatory neuron and weakly excites an
inhibitory one; the inhibitory neuron strongly inhibits back onto both —
the same qualitative circuit motif Pugliese describe for T1, at similar
weight magnitudes. T1-RHS's `inhib→excit2` (-79) is noticeably weaker than
the other five (-172 to -303) — a real asymmetry worth remembering if T1
LHS/RHS behave differently in simulation later.

**What this is and isn't:** this is a data-derived *hypothesis* that T2
and T3 have their own local DNg100-driven CPGs, structurally homologous
to the published T1 one. It is **not** confirmed by reading whether
Pugliese's paper discusses T2/T3 at all (we haven't read the full text
closely for this), and **not** validated by simulation — connectivity
topology alone doesn't guarantee the same rhythmic dynamics. Treat as a
strong lead for Phase 1, not an established fact, until we either find
it addressed in the paper or reproduce rhythmic output from these
candidate circuits ourselves.

**Motor neurons:** separately, and on much firmer footing (a direct
annotation, not an inference), Pugliese's full-VNC table already labels
330 leg motor neurons by joint/module across all three segments (T1: 142,
T2: 95, T3: 93) — tibia flex/extend, coxa swing/stance, femur reductor,
tarsus control, substrate grip, etc. No equivalent CPG discovery was
needed for these; we just read the existing annotation.

**Resolution into MaleCNS:** of all 350 unique MANC bodyIds involved (2
DNg100 + 24 CPG-candidate neurons + 330 motor neurons — some overlap
removed), **333 resolved (95.1%)** via `mancBodyid`. Output:
`data/circuit_map/all_legs_circuit.csv` (gitignored; regenerate via
`python -m fly_robot.connectome.identify_all_legs --pugliese-repo <path>`).

**Visual spot-check:** rendered all three segments together
(`fly_robot/analysis/visualize_all_legs.py` →
`media/all_legs_circuit_{dark,light}.png`, T1 in saturated colors since
it's confirmed, T2/T3 in cooler/muted colors since they're still a
hypothesis). Result: three separate bilateral motor-neuron neuropil
clusters in the expected anterior-to-posterior order, DNg100 descending
through and reaching into each one, and each segment's candidate CPG
triad sitting compactly inside its own segment's cluster — no neurons
turned up in anatomically nonsensical places.

**Second, independent check — does each triad actually output onto its
own segment's motor neurons** (not just onto each other)? Internal
mutual excitation/inhibition alone doesn't make something a CPG if it
doesn't drive any motor output. Checked direct synaptic weight from each
triad member onto that segment's same-side annotated leg motor neurons:

| segment | hub → local MNs | excit2 → local MNs | inhib → local MNs |
|---|---|---|---|
| T1 (confirmed) | 24/71, Σw=+99 | 31/71, Σw=+357 | 24/71, Σw=-54 |
| T2 (candidate) | 20/47, Σw=+116 | 15/47, Σw=+143 | 12/47, Σw=-51 |
| T3 (candidate) | 11/46, Σw=+63 | 16/46, Σw=+65 | 5/46, Σw=-11 |

All three segments show the same qualitative pattern (both excitatory
neurons broadly excite the local motor pool, the inhibitory neuron
broadly inhibits it) — a second, independent line of evidence beyond the
internal triad topology. T3's connections are consistently weaker and
sparser than T1/T2 (e.g. inhib→MN summed weight -11 vs -54/-51) — real,
worth remembering as a caveat rather than treating all three segments as
equally strong evidence.

**Still not enough to call this confirmed (as of this point in the
session):** matching static connectivity (internal topology + motor
output) is real evidence but not proof of function — it doesn't establish
that stimulating these circuits actually produces a sustained oscillation
the way Pugliese demonstrated for T1. **Not yet done:** (1) reading
Pugliese's paper text directly for any T2/T3 discussion that would
confirm or contradict this independently of our own analysis; (2) running
the actual rate-model dynamics on all three candidate circuits and
checking for rhythmic output — the real test, and still owed for T1
itself (Phase 0's original goal).

*(Superseded a few hours later the same day — see the 2026-09-19 entry
below: the paper's own v2 already answers this directly.)*

## 2026-09-19 — Correction: the paper's v2 already confirms all-six-legs dynamically (we read the wrong version)

Following the plan above, we went to read Pugliese et al.'s actual paper
text before running our own simulation — and caught a real mistake in the
process, which is exactly why that step mattered.

**What went wrong first:** a WebSearch summary claimed "four connectome
datasets"; a WebFetch summary of the same paper claimed "two: MANC and
FANC." We downloaded the bioRxiv PDF ourselves to settle it — but
initially fetched **v1** (Sept 2025, `.../675944v1.full.pdf`, 31 pages).
v1 does say exactly two datasets (MANC, FANC) and restricts all dynamical
simulation to the T1 front-leg network — confirmed by direct quotes, e.g.
*"To select a subset of cells to simulate from the entire MANC dataset,
we first selected all front leg motor neurons."* Based on v1, we wrote up
yesterday's T2/T3 finding as "a hypothesis, not confirmed by the paper."

**The mistake:** the repo's own README links to **v2**
(`.../675944v2`), posted April 30, 2026 — 45 pages, substantially
expanded. We only noticed because leftover commented-out code in their
`Figure 2.ipynb` notebook referenced run folders named
`DNg100_Stim_IMAC_vncOnly` and `DNg100_Stim_BANC_vncOnly`, which don't
exist in what v1 describes. That was the signal to go back and read v2
directly (`/tmp/pugliese_v2_text.txt` in this session), rather than trust
our own earlier reading as final.

**What v2 actually says (quotes, not paraphrase):**

- **Four datasets, not two:** MANC (male VNC), FANC (female VNC), **mCNS**
  ("male central nervous system" — this is the dataset we've been calling
  MaleCNS; confirmed by bibliography ref [31], Berg et al., the Janelia
  FlyEM MaleCNS paper), and **BANC** — in v2's usage, the *female*
  brain-and-cord connectome (ref [30], Bates et al., "Distributed control
  circuits across a brain-and-cord connectome," bioRxiv July 2025 —
  Seung/Murthy/Lee labs). Note: BANC here is NOT the same thing v1's text
  vaguely gestured at as unused future work — v2 actually uses it as one
  of the four datasets.
- **They already ran the full-VNC, six-leg, dynamical simulation we were
  about to build ourselves.** Section "Motor rhythms in all six legs" /
  Figure 4: *"we scaled up our simulations to the entire MANC connectome
  (23,532 neurons, including all six leg neuropils...). Bilateral
  activation of the two DNg100 axons produced oscillatory activity in the
  core CPG neurons and a subset of motor neurons of all six legs... This
  demonstrat[es] that a single descending pathway is sufficient to
  recruit rhythm-generating circuits throughout the nerve cord."`
  **This is the direct dynamical confirmation of our T2/T3 hypothesis —
  from the paper's own primary text, not our inference.**
- **Segment asymmetry, matching our own finding:** *"Motor rhythms were
  somewhat less robust in the hind legs and overall rhythmicity was
  reduced compared to the front leg subnetwork."* This independently
  agrees with what we found ourselves via pure connectivity weights
  yesterday (T3's triad→motor-neuron connections were the weakest of the
  three segments: 5-16/46 MNs connected vs T1's 24-31/71). Two
  independent methods — their dynamics, our static connectivity — point
  the same way. Worth remembering as a real, repeated signal, not
  coincidence.
- **No interleg phase coupling, even with the full simulation:**
  *"despite robust rhythmic activity in both legs, no consistent phase
  relationship emerged between the left and right coxa promotor motor
  neurons... and phase coupling was absent across the six leg CPGs in the
  full connectome simulation."* This is the exact "known open issue"
  already in `CLAUDE.md`'s background section — now with a precise
  citation (Fig. 4d-e, Extended Data Fig. 9b) instead of a general
  recollection. It's also the central premise of our entire project: this
  is the gap a real physical body might close.

**What we're changing because of this:** `CLAUDE.md`'s dataset section
and scientific-background section need the v1→v2 correction (two vs four
datasets; T1-only vs full-VNC dynamics already published). Our T2/T3
finding from 2026-09-18 is now labeled **confirmed by the paper** (not
just "our hypothesis"), with our own connectivity analysis kept as
independent corroborating evidence, not the sole basis.

**Lesson for how we work, not just what we found:** this is the second
time in two days a secondary summary (search snippet, fetched-page
summary) was wrong or incomplete in a way that mattered, and the fix both
times was going to the primary source ourselves. Added this as an
explicit rule in `CLAUDE.md`.

**Still to do:** we're proceeding to run our own version of this full-VNC
simulation anyway (same experiment, our own MaleCNS-derived candidate
circuits + Pugliese's own unmodified simulation code on the full MANC
connectivity matrix they provide), for two reasons: (1) it validates our
own simulation pipeline against a known published result, which was
Phase 0's original goal regardless; (2) their result is reported
qualitatively ("somewhat less robust") — running it ourselves gets us
exact numbers and the raw activity traces, which we need later anyway for
tuning the interface. Results in the next entry.

## 2026-09-19 (cont.) — Setting up and running Pugliese's actual simulation code

**Environment:** cloned Pugliese's repo properly into `external/Pugliese_cpg_2025`
(gitignored — 375MB including their data, not ours to version; MIT
licensed, so reuse is fine). Installed their exact dependency pins
(JAX 0.9.2, diffrax 0.7.0) into our venv and `pip install -e .`'d their
package. Added `fly_robot/neural/run_pugliese_sim.py`, which imports and
calls their unmodified simulation functions directly — we do not
reimplement or alter their neuron dynamics equations, connectivity
handling, or oscillation-scoring method anywhere.

**T1 reproduction — succeeded, matches their tutorial exactly.** Running
their own `DNg100_Stim` config on their own T1 (4,604-neuron) network:
3/144 motor neurons active, mean oscillation score 0.797, ~9.5 Hz —
matches their Tutorial 1 notebook's stated result ("these 3 motor neurons
all have a rhythmic output") almost exactly. Sustained, clean oscillation
confirmed visually (`media/simulation/t1_mn_traces.png`). This is Phase
0's original "done when" criterion, now actually met, not just inferred.
(Along the way: `compute_oscillation_score`'s frequency output is in
cycles/sample, not Hz — their own `Figure 2.ipynb` divides by `dt` to
convert; we do the same, and caught this only because the first printed
value, 0.01 Hz, was obviously not a plausible fly stepping frequency and
had to be traced down rather than reported as-is.)

**Full-VNC attempt — built the pipeline, but it does not yet reproduce
their six-leg result, and we have not fully root-caused why.** To run
their code on our full 23,628-neuron network instead of their 4,604-neuron
one: converted `wTable_20260522_allSynapses.feather` / `W_...npz` to the
`.csv`/`.npy` formats their loaders accept
(`fly_robot/neural/prepare_full_vnc_data.py`; pure format conversion, same
values). Debugging path, each step a real finding, not a guess accepted
without checking:

1. First attempt (unilateral DNg100, stimI=250, same as T1's default):
   **zero** motor neurons active anywhere, in any leg.
2. Found the source matrix (`W_20260522_allSynapses.npz`) has no minimum
   synapse-count filter — 74.1% of nonzero entries are below the 5-synapse
   floor Pugliese's own Methods say they impose for their T1 network.
   Applied the same floor. **Still zero.**
3. Checked whether the paper's Fig. 4 experiment even matches our
   stimulation protocol — it explicitly says "**bilateral** activation of
   the two DNg100 axons," but we'd only stimulated one. Fixed the config
   to stimulate both (bodyIds 10093 and 10339, same stimI=250 each).
   **Still zero** — not even `IN17A001` (the T1 hub neuron, which reliably
   fires from a documented 187-217 weight direct DNg100 connection) shows
   *any* activity one hop downstream, even though DNg100 itself is
   confirmed tonically active (rate ~9.4, ~7.6) in this run.
4. Found a real, mechanistic candidate cause: their model scales each
   neuron's threshold/gain by cell size *relative to the network's own
   median size* (`set_sizes`, `src/utils/sim_utils.py:73` — confirmed by
   reading the function). Our full network's median cell size is ~40%
   smaller than the T1 network's (more small sensory/intrinsic neurons
   included), which systematically raises every neuron's relative-size
   ratio, raising thresholds and lowering gains network-wide relative to
   what was tuned for the smaller, curated T1 network.
5. The paper's Methods do document an automated adaptive stimulus-search
   procedure for exactly this kind of scale mismatch (doubling/halving
   `Istim` based on how many neurons are recruited, "Descending neuron
   activation screen" section) and it exists in their code
   (`_adjust_stimulation_for_batch`, `vnc_sim.py:1143`) — but written for
   their batched/pmap runner, not directly callable for a single
   simulation. We re-implemented the same documented algorithm (same
   thresholds: `n_active_lower=5`, `n_active_upper=500`,
   `n_high_fr_upper=100` @ 100Hz, ≤10 iterations) faithfully for the
   single-simulation case (`adaptive_stim_single` in
   `run_pugliese_sim.py`) rather than hand-picking a stimulus value.
   Result: it "converged" immediately at stimI=250, because 23 neurons
   were already active network-wide (DNg100 + a couple of near-neighbors)
   — clearing their `n_active_lower=5` bar even though **none of the 23
   were motor neurons**. Their adaptive procedure guards against numerical
   instability, not specifically against "activity dies before reaching
   the legs" — so it doesn't fix this on its own.

**Decision at this point, per explicit instruction not to keep
parameter-guessing:** rather than continue tuning our own re-derived
matrix by trial and error, we're downloading Pugliese's own actual saved
simulation output for this exact experiment from their Zenodo archive
(`DNg100_Stim_fullManc.zip`, 2.0GB, "Full MANC DNg100 bilateral
activation, 32 replicates per run, 128 replicates in total," explicitly
used for Fig. 4a-b) — their repo's own `Figure 4.ipynb` loads this same
file rather than regenerating it. This lets us verify the all-six-legs
result against their real output directly, and is more reliable than
continuing to debug why our independently-reconstructed 23,628-neuron
matrix (built from a newer, slightly different-sized MANC snapshot —
23,628 vs. their reported 23,532 neurons, a ~100-neuron difference we
have not explained) behaves differently from their exact 23,532-neuron
network under the same nominal protocol. Continued in the next entry once
the download and analysis are done.

## 2026-09-19 (cont.) — Resolution: their real data confirms our T2/T3 candidates, and explains why our own attempt failed

**Why our own full-VNC simulation attempt was failing (found from their
real config, not guessed):** their actual run config
(`run_id=33195857/logs/run_config.yaml`, extracted from the Zenodo
archive) shows they used:
- **A different, older data file than the one we built our own attempt
  on**: `wTable_20251006.feather` / `W_20251006.feather` (23,532 neurons —
  exactly matching the paper's stated count), not the
  `..._20260522_allSynapses...` file we used (23,628 neurons, a newer
  MANC proofreading snapshot). This fully explains the neuron-count
  mismatch we'd flagged as unexplained — they are literally different
  dated exports of MANC, not the same data. Both files were already
  sitting in Pugliese's own repo the whole time; we had used the wrong
  one.
- **stimI = 380** (fixed, manually set — `adjustStimI: false` in their
  config, so it was *not* found by any automated search either). Not 250
  (T1's value), not 400 (the "two CNS datasets" value from the Methods
  text, which we'd reasonably but wrongly guessed might transfer here).
  This was simply not derivable from the paper text alone — only from
  their actual config file.

This means our earlier debugging (synapse floor, bilateral stimulation,
identifying the size-normalization effect, implementing their adaptive
stimulus algorithm) was genuine, correct reasoning about real effects —
but the root cause was simpler than any of that: wrong input file, wrong
stimulus magnitude, both undiscoverable without the actual config. Lesson
for next time: when trying to reproduce a specific published figure, look
for their exact run config before re-deriving parameters from the paper
text — the text alone under-specifies the experiment.

**Verification against their real output (not our simulation — their
actual computed data)**, loaded from `DNg100_Stim_fullManc_Rs.npz`
(merged, 128 replicates × 23,532 neurons × 2001 timesteps, run ids
30662613+33195857+33195876+33195882 from Zenodo record 22260924):

Per-segment motor neuron oscillation scores, computed with their own
unmodified `compute_oscillation_score`, averaged over all 128 replicates:

| segment | mean oscillation score | mean # active MNs | replicates with any MN activity |
|---|---|---|---|
| T1 | 0.650 | 8.5 / 179 | 95% |
| T2 | 0.607 | 10.4 / 177 | 95% |
| T3 | 0.381 | 26.2 / 153 | 97% |

T2 is nearly as robust as T1. T3 has *more* nominally-active motor neurons
on average but a *lower* oscillation score — i.e. more neurons cross the
activity threshold but fire less cleanly/periodically. This matches the
paper's own qualitative statement ("somewhat less robust in the hind
legs") with an actual number attached, and matches our independent
connectivity finding from 2026-09-18 that T3's candidate triad has the
weakest output onto its local motor neurons of the three segments.

**Direct check of the exact 9 neurons we independently identified**
(3 confirmed T1 + 6 candidate T2/T3, by MANC bodyId, cross-referenced into
this real wTable): every single one shows substantial rhythmic activity
across the 128 replicates —

| neuron | role | active reps | mean oscillation score |
|---|---|---|---|
| 10707 (IN17A001) | T1 hub, confirmed | 124/128 | 0.613 |
| 11751 (INXXX466) | T1 excit2, confirmed | 120/128 | 0.675 |
| 13905 (IN16B036) | T1 inhib, confirmed | 107/128 | 0.784 |
| 10559 | T2 hub, candidate | 124/128 | 0.483 |
| 11767 | T2 excit2, candidate | 122/128 | 0.565 |
| 13186 | T2 inhib, candidate | 116/128 | 0.672 |
| 10498 | T3 hub, candidate | 124/128 | 0.708 |
| 12315 | T3 excit2, candidate | 124/128 | 0.685 |
| 12953 | T3 inhib, candidate | 104/128 | 0.373 |

**This is the direct confirmation we were after**: neurons we identified
purely from static wiring (no access to any dynamics, done before this
data was ever loaded) are, in Pugliese's own real published simulation
output, genuinely rhythmically active — not marginally, not
coincidentally, at the same general strength as the confirmed T1 circuit.
Visual confirmation saved to
`media/simulation/published_data_t1_t2_t3_cpg_traces.png` (replicate 45,
the same example replicate their own `Figure 4.ipynb` uses): T1 and T2
show clean sustained oscillation; T3 shows the same three neurons
oscillating but visibly less cleanly (irregular amplitude/timing around
t≈0.6-0.75s and t≈1.75-2.0s) — consistent with its lower quantitative
score.

**Where this leaves Phase 0 and Phase 1:**
- Phase 0's original goal (reproduce Pugliese's rhythmic result) is now
  met two ways: our own from-scratch T1 simulation run
  (`media/simulation/t1_mn_traces.png`), and direct analysis of their real
  published full-VNC data.
- Phase 1's T2/T3 hypothesis from 2026-09-18 is upgraded from "hypothesis,
  supported only by our own connectivity analysis" to **confirmed by
  actual published simulation dynamics**, cross-referenced against the
  exact neurons we identified independently.
- We have NOT gotten our own from-scratch full-VNC simulation (on our
  MaleCNS-resolved circuit, or even on MANC with the correct file/stimI)
  producing this result ourselves yet — what we've verified is that
  *their* real computed output confirms our candidates. Re-running it
  ourselves with the corrected file (`wTable_20251006.feather`) and
  stimI=380 is a cheap, worthwhile follow-up to actually close that gap,
  but is no longer blocking — the scientific question is answered.

**Follow-up, done same day:** generalized `prepare_full_vnc_data.py` to
accept either data-file pair (`--source 20251006` or
`--source 20260522_allSynapses`) and to handle the `.feather`-format W
matrix (a wide DataFrame indexed by `bodyId_pre`/`bodyId_post`, reindexed
explicitly to `wTable`'s bodyId order rather than assumed) — their real
production wPath was `.feather`, which their current `load_W` doesn't
actually support, so we still convert to `.npy`, just from the *correct*
source file this time. Updated the experiment config with their verified
stimI=380 and bilateral `[59, 282]`. Result:

```
Total motor neurons in network: 732
  all motor neurons: 30/732 active, score=0.469, freq=6.98 Hz
  T1: 3/179 active, score=0.779, freq=10.54 Hz
  T2: 3/177 active, score=0.772, freq=11.36 Hz
  T3: 24/153 active, score=0.392, freq=5.73 Hz
```

**Our own from-scratch simulation now reproduces the six-leg rhythmic
result**, with the same qualitative pattern seen everywhere else in this
investigation (T1/T2 clean and strong, T3 weaker score despite more
neurons technically active). This is a single replicate (n=1, one random
seed), not averaged over 128 like the published analysis, so don't expect
these exact numbers to match the published table above — the pattern
matching is what matters. This closes the "not yet done" gap noted above;
Phase 0 is now complete on every front we set out to check.

Also moved our own two Hydra config additions (`paths/fly_robot.yaml`,
`experiment/FullVNC_DNg100_Stim.yaml`) into a tracked location
(`fly_robot/neural/pugliese_extra_configs/`) with a small
`setup_pugliese_configs.py` to install them after cloning — they'd
previously been sitting only inside the gitignored `external/` clone and
would have been silently lost on a fresh checkout.

## 2026-09-19 (cont.) — Closing Phase 1: motor neurons grouped by leg × joint

The remaining piece of Phase 1 ("group leg motor neurons by leg, joint,
flexor/extensor") turned out to already be sitting in data we pulled
yesterday — `identify_all_legs.py`'s `collect_motor_neurons()` already
read Pugliese's own `motor module` annotation (a direct label from their
table, not something we inferred) for every leg motor neuron in T1/T2/T3.
Today's work was presenting it properly rather than new extraction:
grouped counts saved to
`data/circuit_map/motor_neuron_groups_by_leg_and_joint.csv`.

| joint module | T1 | T2 | T3 |
|---|---|---|---|
| coxa swing | 15 | 7 | 6 |
| coxa stance | 13 | 14 | 14 |
| femur/tr extend | 16 | 10 | 13 |
| femur/tr flex | 24 | 18 | 20 |
| femur reductor | 12 | 6 | 4 |
| tibia extend | 4 | 4 | 4 |
| tibia flex | 33 | 23 | 23 |
| substrate grip | 17 | 15 | 16 |
| tarsus control | 15 | 0 | 0 |
| **total** | **149** | **97** | **100** |

**One thing we noticed and are flagging rather than explaining away:**
"tarsus control" has 15 motor neurons in T1 and **zero** in T2 or T3. We
don't know yet whether this is a real biological difference (front legs
doing something with the tarsus — the leg's foot/claw segment — that mid
and hind legs don't) or a labeling gap specific to Pugliese's T1-focused
annotation effort (recall their per-leg "motor module" labels were
originally built out for the T1 circuit specifically; T2/T3 rows may
simply not have been assigned this particular category during their
annotation pass, even if the corresponding neurons exist and are just
unlabeled). Not resolved — worth checking against MANC/MaleCNS's own
independent motor neuron annotations directly before relying on this
absence for any real interface design.

**MaleCNS resolution:** 312/346 leg motor neurons (all three segments
combined) resolved to MaleCNS via `mancBodyid` — 17 unresolved in T1, 11
in T2, 6 in T3.

**Phase 1 status:** the neuron-identification side (which neurons drive
which leg, which joint, confirmed rhythmically active — see the entry
above) is done for the CPG side and now for the motor-neuron side too.
What Phase 1 has NOT done: identify any DN/CPG-to-motor-neuron mapping
beyond what Pugliese's own data already labeled — i.e. we have not
independently derived which specific interneurons drive which specific
joints, only inherited that grouping from their annotation.
