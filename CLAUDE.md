# Fly Connectome → Robot Body (Video 1: Hexapod → Quadruped)

## What this project is
We drive a simulated legged robot using a simulation of the real fruit fly
(Drosophila) ventral nerve cord (VNC) connectome — the fly's equivalent of a
spinal cord. The result becomes a YouTube video.

Video 1 story:
1. Real fly nerve-cord wiring generates a walking rhythm.
2. That rhythm drives a six-legged robot (hexapod) in MuJoCo.
3. We "amputate" the middle legs and test whether the same nervous system
   can drive a four-legged body (quadruped).

## Honesty constraints (non-negotiable)
Every claim in code, logs, plots, and docs must be precise:
- The connectome wiring (synapse graph + neurotransmitter signs) is NEVER
  modified. We only design/tune the INTERFACE (neurons → joints, sensors → neurons)
  and a few global gains.
- Anything we add that is not in the fly (e.g. interleg coupling, filters,
  oscillator priors) must be explicit, config-flagged, and named as an addition.
- This is NOT a fly mind, NOT consciousness, NOT a brain "learning" a new body.
- Never invent neuron IDs, cell-type names, or dataset facts. Query them from the
  data, and flag anything uncertain.
- **The user does not have a biology background and is relying on us to be the
  fact-check, not the source of new errors.** Precision matters more than
  fluency here:
  - When a claim comes from a paper, quote or closely paraphrase the exact
    sentence and say which section/figure it's from — don't rely on a
    secondary summary (including a web search snippet or an AI-generated
    summary of a fetched page) as if it were the primary source. We hit this
    directly on 2026-09-18: a WebSearch summary claimed "four connectome
    datasets," a WebFetch summary of the same paper claimed "two: MANC and
    FANC," and only reading the actual PDF text resolved which was right.
    Treat any tool-generated summary of a paper as a lead to verify, not
    an answer to report.
  - Label every finding as **confirmed** (directly stated in a primary
    source, or directly computed from real data) vs. **hypothesis**
    (a pattern we noticed that's plausible but not independently verified)
    vs. **our own addition** (something we're choosing to build that has
    no fly-biology basis at all, e.g. an interleg coupling term). Never let
    these blend together in logs, code comments, or the video script.
  - If something is ambiguous, contradictory, or you're not sure — stop and
    say so explicitly, in plain language, rather than picking the more
    confident-sounding option.

## Scientific background (short)
- Fly walking is largely hardwired. The brain sends high-level commands via
  descending neurons (DNs); the VNC contains central pattern generators (CPGs)
  that produce leg rhythms; leg sensors (proprioception) adjust in real time.
- Pugliese et al., "Connectome simulations identify a central pattern
  generator circuit for fly walking," bioRxiv. **Read the paper directly, and
  check which version** — v1 (Sept 2025, 2 datasets, T1-only dynamics) and v2
  (April 2026, 4 datasets, full-VNC six-leg dynamics) report meaningfully
  different scope; see CHANGELOG 2026-09-19 for how we caught this
  (a WebSearch summary and a WebFetch summary of the same paper disagreed with
  each other; only the primary PDF text resolved it). As of v2:
  - A firing-rate model of VNC connectomes. Tonic stimulation of DNg100
    (walking command neuron) drives rhythmic leg motor neuron activity.
  - Core 3-neuron CPG (1 inhibitory, 2 excitatory), found via a T1
    front-leg-restricted network: DNg100 → E1/IN17A001 (excit.) +
    E2/INXXX466 (excit.) + I1/IN16B036 (inhib.). DNb08 recruits an
    overlapping 5-neuron circuit (shares E1, E2) and drives a different,
    non-walking rhythmic behavior.
  - **v2 confirms this same circuit (same 4 cell types) recurs functionally,
    not just anatomically, in all six leg neuropils**: they ran the full
    23,532-neuron MANC connectome, bilaterally activated DNg100, and got
    rhythmic CPG + motor neuron activity in all six legs (Fig. 4). Hind legs
    (T3) were "somewhat less robust" than front legs — matches our own
    independent connectivity-based finding (`CHANGELOG.md` 2026-09-18) that
    T3's triad→motor-neuron weights are the weakest of the three segments.
  - Four connectome datasets in v2: MANC (male VNC), FANC (female VNC),
    **mCNS** (male whole CNS — this is the dataset we call MaleCNS
    throughout this project), and **BANC** (female whole CNS, Bates et al.
    2025 — a different dataset from MANC/FANC/mCNS, not to be confused with
    "the brain and nerve cord" generically).
- **Known open issue, now precisely sourced:** even in the full six-leg
  simulation, DNg100 activation alone produces NO consistent left/right phase
  coupling (Fig. 4d-e, Extended Data Fig. 9b: "phase coupling was absent
  across the six leg CPGs in the full connectome simulation"). The paper's own
  conclusion is that additional mechanisms — phasic sensory feedback or
  biomechanical coupling — may be needed for real interleg coordination.
  **Closing that loop with an actual physical body is the central bet of this
  entire project**, not a side detail.
- **This is a known open question in the field, not something we invented
  — but the specific experiment hasn't been published (checked 2026-09-19,
  see docs/logs).** Pugliese et al.'s own Discussion section proposes
  exactly this as unfinished future work, in almost these words: *"In the
  future, it may be possible to test hypotheses by coupling VNC
  connectome simulations to control and receive feedback from
  biomechanical models of the fly body interacting with a simulated
  physical environment, which would require adding biologically
  realistic interfaces of proprioceptive sensors and muscle actuators."*
  A closely related but methodologically different effort exists
  ("FlyGM," NeurIPS 2025): it couples the connectome's wiring *topology*
  to a biomechanical body, but as an architectural prior for a graph
  neural network **trained by reinforcement learning** to solve
  locomotion tasks — a different question ("does connectome-shaped
  architecture help an RL policy learn to walk") from ours ("does the
  literal, unmodified, untrained real synaptic wiring plus a real body
  produce coordination"). We never train or optimize the connectome
  itself — matching this project's core constraint (never modify the
  wiring, only the interface) — which is precisely what distinguishes
  our approach from FlyGM's. As far as we've found, nobody has published
  the literal closed-loop result Phase 4 is attempting either way it
  comes out.

## Dataset: MaleCNS, not MANC (decided 2026-09-18)
We use **MaleCNS v1.0** (`male-cns:v1.0` on neuprint, server
`https://neuprint.janelia.org`) as our connectome going forward, not MANC.

- MANC (2023) is nerve-cord-only, ~23k neurons. Pugliese's *v1* preprint
  built their model on MANC restricted to T1 (front-leg) only; **v2 (April
  2026) extended this to a full 23,532-neuron whole-MANC six-leg simulation**
  — see the scientific-background section above. So "Pugliese only did T1"
  is no longer accurate as a reason to prefer MaleCNS; it was true of v1,
  not v2. Our actual reasons to use MaleCNS instead of MANC still hold on
  their own merits (below).
- MaleCNS (June 2026, Cell paper Sept 2026) is a different, newer specimen:
  brain + VNC imaged as one continuous volume (including the neck connective
  MANC lacks), ~166,700 neurons. It's a superset for our purposes and keeps
  later phases (real descending drive from the brain, not synthetic tonic
  stimulation) open without a second dataset migration. Pugliese et al.
  themselves now also use this same dataset (their "mCNS") as one of their
  four — independent validation that it's a reasonable choice.
- MaleCNS neuron records carry a `mancBodyid` field — a per-neuron
  cross-reference to the corresponding MANC specimen body, computed by
  Janelia's annotation team. We use this (not `type`-string matching) to
  relocate neurons Pugliese identified in MANC into MaleCNS's ID space.
  See `fly_robot/connectome/identify_t1_circuit.py` and
  `docs/logs/2026-09-18.md` for the worked example (T1 circuit, 142/152
  neurons resolved).
- Open caveat: Phase 0's original goal (match Pugliese's published rhythm
  figures) assumed MANC-derived weights. Running on MaleCNS-derived weights
  is not numerically the same circuit even with matching neuron identities.
  For direct pipeline validation we run Pugliese's own unmodified code on
  their own MANC data first (see `fly_robot/neural/reproduce_rhythmic_output.py`,
  `--mode t1` and `--mode full_vnc`), then treat MaleCNS-based results as a
  separate, not-numerically-identical reproduction.

## Working principle: verify against primary sources, every time (added 2026-09-19)
Twice in two days, a secondhand summary (a WebSearch snippet, a WebFetch
summary of a fetched page) was wrong or materially incomplete in a way that
would have produced a wrong claim if trusted directly — see CHANGELOG
2026-09-19. Rule going forward: **any factual claim about a paper, dataset,
or repo that will be written into code, comments, CHANGELOG, or the video
script must be checked against the actual primary text/data, not a summary
of it.** A summary is a lead to verify, never a citation. This matters more
here than in typical coding work because the user does not have a biology
background and is relying on us to be the fact-check, not a source of new,
harder-to-catch errors.

## Key resources
- Pugliese repo (JAX/JIT VNC rate model, notebooks, Hydra configs):
  https://github.com/smpuglie/Pugliese_cpg_2025 (work in progress — read the
  code; do not assume APIs). Their neuron identities are only recoverable from
  experiment config row-indices into their own CSVs, not published names —
  see the CHANGELOG entry for how we decoded them.
- MaleCNS v1.0 (brain + VNC, CC-BY): https://male-cns.janelia.org/download/
  Bulk files under gs://flyem-male-cns/v1.0/; programmatic access via
  neuprint-python (`fly_robot/connectome/client.py`).
- MANC connectome (male adult nerve cord, 2023) — only relevant now for
  cross-referencing MaleCNS neurons back to Pugliese's exact model.
- Leg motor neurons have annotated/predicted muscle targets (partial coverage).
- Reference controllers/robots: Drosophibot (fly-like hexapod, neural activity
  → actuator conversion), Walknet (stick-insect hexapod controller, handles leg loss),
  Massi et al. 2019 (CPG + cerebellar learning + CMA-ES on a quadruped).
- Simulator: MuJoCo (CPU for dev), MJX (JAX) for batched rollouts later.
- Evolution (later phase): CMA-ES, preferably evosax (JAX).

## Compute environment
- Development: local M3 Mac (CPU) for Phases 0–4; GPU (Kaggle/Colab) only
  needed once MJX batched rollouts + CMA-ES arrive (Phase 5).
- Python: project venv is 3.12 (`.venv/`), not the system's 3.14 — better
  wheel compatibility with navis/neuprint-python's scientific-stack deps.
- `jax-metal` (Apple Silicon JAX/GPU) is experimental/frequently broken —
  plan on CPU JAX locally.
- Sessions can disconnect: checkpoint state, configs, and logs to Google Drive
  frequently once we move to Colab/Kaggle; all runs must be resumable.
- Headless rendering: `MUJOCO_GL=cgl` on macOS (Apple's Core OpenGL — verified
  2026-09-19; EGL is Linux-only and does not work here), `MUJOCO_GL=egl` on
  Linux/Colab.
- Fixed random seeds; every run is reproducible from its config.
- neuprint auth token lives in `.env` (gitignored, never commit).

## Architecture (current — items marked *not yet created* are still ahead)
```
fly_robot/
  connectome/         # neuprint client, MANC<->MaleCNS neuron identification,
                      # circuit extraction (identify_t1_circuit.py,
                      # identify_all_leg_circuits.py), MANC/MaleCNS annotation
                      # comparison (compare_sensory_annotation.py)
  neural/             # Pugliese's own JAX rate model, run unmodified
                      # (single_simulation.py, replicate_ensemble.py,
                      # reproduce_rhythmic_output.py) plus our OWN steppable
                      # reimplementation for closed-loop use — same equation,
                      # connectome and neuron params, fixed-step RK4 + sparse
                      # matvec instead of their adaptive/dense solver
                      # (steppable_rate_model.py, validated against their
                      # published output in validate_steppable_model.py).
                      # connectome_controls.py: degree-preserving shuffle and
                      # phase-randomised surrogates, for the shuffled-wiring
                      # and rate-matched-noise controls (never modifies the
                      # real connectome in place).
    pugliese_extra_configs/  # our Hydra config additions, tracked (external/ is gitignored)
  interface/          # motor-neuron rates -> joint targets (motor_neuron_to_joint.py,
                      # FROZEN once closed-loop work started, config hash tracked) and
                      # joint state -> sensory-neuron input current
                      # (joint_to_sensory_neuron.py, four encoder variants —
                      # see docs/closed_loop/)
  bodies/             # NeuroMechFly/FlyGym v2 body composition (not a hand-built hexapod — see
                      # docs/logs/2026-09-19.md). 7 real DOF/leg, not the originally-sketched 3.
                      # tethered_ball.py: the standard fly-lab rig (thorax fixed,
                      # legs on a freely-rotating sphere) as the "ground" condition.
  sim/                # the coupled loop: neural dt (1ms) and physics dt (0.1ms)
                      # stepped together, sensory feedback entering as an
                      # additive term in the stimulation current
                      # (closed_loop.py, trial_setup.py)
  experiments/        # runnable scripts: harness_sine_wave_test.py (mechanical
                      # sanity), connectome_driven_open_loop.py (Phase 3 open
                      # loop), open_loop_coordination_baseline.py (E0 baseline),
                      # closed_loop_pilot.py (pre-registered pilot sweeps),
                      # render_closed_loop_clips.py (video capture of specific
                      # pilot states)
  analysis/           # circuit visualizations (visualize_t1_circuit.py,
                      # visualize_all_leg_circuits.py); interleg_coordination.py —
                      # the rhythm/phase-locking/tripod-index metrics, with a
                      # rhythm-first gate (AR(1) null) before any coupling
                      # number is computed
tests/                # pytest gates for the interface and closed loop —
                      # locality, determinism, g_fb=0 bit-identity, degree
                      # preservation. Run with `pytest tests/`.
docs/
  logs/               # dated findings, one file per day — Phases 0-3
  closed_loop/        # the closed-loop work's own audit trail — AUDIT.md,
                      # SENSORY_MAP.md, PREREGISTRATION.md (pre-registration +
                      # amendments), RESULTS.md, LOG.md (chronological). Kept
                      # separate from docs/logs/ because it's one continuous
                      # pre-registered arc, not daily findings.
CHANGELOG.md          # short index into docs/logs/ and docs/closed_loop/, one line per entry
external/             # cloned reference repos (e.g. Pugliese_cpg_2025) — gitignored, MIT-licensed reuse
data/                 # connectome dumps/caches/simulation output — regenerable, gitignored
media/                # renders, neuron-activity overlays for the video — gitignored
                       # (regenerate via analysis/ or experiments/ scripts rather than versioning binaries)
configs/              # *not yet created* — our own experiment configs (Hydra or plain YAML), once we have our own model
notebooks/            # *not yet created* — thin wrappers only; logic lives in the package
```

## Phase roadmap (Video 1)
Track detailed progress as GitHub issues (one per phase); this is the
short version for orientation.

- **Phase 0 — Reproduce the science. DONE (2026-09-19).** Queried MaleCNS
  for DNg100, the CPG neurons, and the T1 leg motor neurons (resolved
  correctly, anatomically sane render). Ran Pugliese's own simulation code
  on their T1 network and reproduced their published rhythmic motor output
  exactly (`media/simulation/t1_mn_traces.png`). Separately, downloaded
  and analyzed their actual real published full-VNC simulation output
  (Zenodo record 22260924, `DNg100_Stim_fullManc.zip`) and confirmed
  rhythmic CPG + motor neuron activity in T1/T2/T3, matching Fig. 4's
  qualitative claims with real numbers attached — see CHANGELOG
  2026-09-19. Also got our own from-scratch full-VNC simulation working
  (their verified data file + stimI=380, both found only by extracting
  their real run config, not derivable from the paper text) and it
  independently reproduces the same six-leg rhythmic pattern. Remaining
  gap, not blocking: everything dynamically verified so far used MANC
  data (ours or theirs) — a from-scratch run on MaleCNS-derived weights
  specifically hasn't been done.
- **Phase 1 — All six legs.** Extract T2/T3 (mid/hind leg) DN/CPG/motor-neuron
  circuits in MaleCNS. **CPG identification: DONE, confirmed by real
  dynamics, not just connectivity** — the 6 candidate T2/T3 neurons we
  identified from wiring alone (`identify_all_leg_circuits.py`) were checked
  directly against Pugliese's real published simulation output and all 6
  show substantial rhythmic activity (104-124/128 replicates active,
  scores 0.37-0.71 — see CHANGELOG 2026-09-19). **Not yet done:** the
  motor-neuron side (which specific leg joints/muscles get driven, per
  segment) still needs the same level of scrutiny; grouping by
  joint/flexor-extensor for T2/T3 motor neurons is the remaining Phase 1
  work.
- **Phase 2 — Hexapod body in MuJoCo.** Body: FlyGym/NeuroMechFly v2
  (real fly anatomy, 7 DOF/leg — not the originally-sketched simplified
  3 DOF/leg; see `fly_robot/bodies/neuromechfly.py` and
  `docs/logs/2026-09-19.md`). **Harness mode (fixed base) + sanity sine
  gait: DONE** (`fly_robot/experiments/harness_sine_wave_test.py`,
  `media/harness_sine_wave_test/harness_sine_wave_test.mp4`) — confirms the mechanical
  pipeline (body, actuators, harness, physics, rendering) works. This
  sine wave is a synthetic test signal only, not connectome-derived.
- **Phase 3 — Open loop. DONE.** Real motor-neuron rates → joint targets
  (`fly_robot/interface/motor_neuron_to_joint.py`,
  `fly_robot/experiments/connectome_driven_open_loop.py`). Uses Pugliese's own
  real published 128-replicate full-VNC data (not new simulation — see
  `load_published_full_vnc_replicates`), one representative stable
  replicate (never an average across replicates — averaging cancels real
  rhythmic signal since each replicate's oscillation has an independent
  random phase; see docs/logs/2026-09-19.md). Only 3 of 9 per-leg
  motor-neuron modules (coxa swing/stance, femur extend/flex, tibia
  extend/flex) have a clear antagonist-pair mapping to a joint; the
  other 3 (femur reductor, substrate grip, tarsus control) are left
  unmapped — a real limitation. Verified both numerically (~20°
  deviation on the most-driven joints) and visually (three tracking
  cameras — side, opposite-side, top-down — built into
  `build_harnessed_fly()`; top-down is the one that clearly shows leg
  movement without wing occlusion).
- **Phase 4 — Closed loop. Pre-registered pilot run; main experiment
  BLOCKED pending a usable sensory encoder.** Full pre-registration and
  audit trail in `docs/closed_loop/` (`AUDIT.md`, `SENSORY_MAP.md`,
  `PREREGISTRATION.md` + two dated amendments, `RESULTS.md`, `LOG.md`).
  Built a steppable reimplementation of Pugliese's rate model (theirs is
  adaptive/dense and cannot be interrupted for feedback; ours is
  fixed-step + sparse, validated to median r=0.9993 against their
  published output, ~400x faster), the sensory interface (joint angle
  → chordotonal/hair-plate input current — **no leg load/campaniform
  channel exists in either MANC or MaleCNS annotation, so foot load
  feedback is not modelled**, a real limitation not an oversight), the
  tethered-on-ball ground condition, and rhythm-first coupling metrics
  (AR(1)-gated — a leg's phase only counts if it's actually rhythmic).
  **Pilot result: across four encoder formulations (signed position code,
  deviation-from-rest, a per-neuron current cap, and range-fractionated
  tuning — the last motivated by real FeCO range fractionation, labelled
  as our design choice), feedback is either negligible (effect
  ~0.0000-0.0010) or destroys the rhythm entirely (~1.01, i.e. total
  decorrelation) — no gain produces a measurable, non-destructive effect.**
  This holds with the network's own real inhibitory wiring fully active
  (46% of edges); encoder-side inhibition was considered and rejected
  since sensory neuron signs come from the connectome, not from us.
  H1/H2/H3 (interleg coupling from feedback) remain **untested** — the
  pilot establishes they aren't testable with this interface, not that
  they're false. No further encoder variants after this round, per the
  pre-registered stopping rule; next step is a design decision, not more
  tuning.
- **Phase 5 — Ground walking.** Tune interface with CMA-ES (brain weights
  frozen). GPU compute (Kaggle/Colab) likely needed here.
- **Phase 6 — Amputate middle legs → quadruped.** Remap and re-tune
  interface only; connectome/brain weights stay frozen.
- **Phase 7 — Controls.** Shuffled connectome, degree-matched random graph,
  plain oscillator — to show the real wiring matters.

## Working conventions
- **Keep the findings log updated as you go** (added to, 2026-09-19: split
  from one growing `CHANGELOG.md` into `docs/logs/<date>.md`, one file per
  day, with `CHANGELOG.md` kept as a short index of one-line summaries +
  links — mirrors this agent's own memory-index pattern). Every session
  that produces a finding, decision, or non-obvious debugging fix gets an
  entry in that day's log file (with *why*, not just *what*), plus a
  one-line pointer added to the `CHANGELOG.md` index. This doubles as the
  video script source; don't let it fall behind the code.
- Explain non-obvious modelling choices in comments; note the paper section
  they come from when relevant.
- When uncertain about biology, data fields, or repo behaviour: stop and flag
  it rather than guess.
- Save figures at video quality (dark and light variants when useful).
- Every experiment is config-driven and reproducible from its config + seed.
