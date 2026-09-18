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
  See `fly_robot/connectome/identify_circuit.py` and `CHANGELOG.md` for the
  worked example (T1 circuit, 142/152 neurons resolved).
- Open caveat: Phase 0's original goal (match Pugliese's published rhythm
  figures) assumed MANC-derived weights. Running on MaleCNS-derived weights
  is not numerically the same circuit even with matching neuron identities.
  For direct pipeline validation we run Pugliese's own unmodified code on
  their own MANC data first (see `fly_robot/neural/run_pugliese_sim.py`,
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
- Headless rendering: MUJOCO_GL=egl.
- Fixed random seeds; every run is reproducible from its config.
- neuprint auth token lives in `.env` (gitignored, never commit).

## Architecture (target)
```
fly_robot/
  connectome/         # neuprint client, MANC<->MaleCNS neuron identification
  neural/             # JAX rate model (reuse/adapt Pugliese), stimulation protocols
  interface/          # motor-neuron → joint mapping; sensors → sensory-neuron inputs
  bodies/             # MuJoCo MJCF: hexapod (3 DOF/leg), quadruped (amputated variant)
  sim/                # coupled loop: neural dt vs physics dt synchronisation
  experiments/        # phase scripts
  analysis/           # rhythm, phase-coupling, gait plots, circuit visualizations
configs/              # every experiment config-driven (Hydra or plain YAML)
data/                 # connectome dumps/caches — regenerable, gitignored
media/                # renders, neuron-activity overlays for the video — gitignored
                       # (regenerate via analysis/ scripts rather than versioning binaries)
notebooks/            # thin wrappers only; logic lives in the package
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
  identified from wiring alone (`identify_all_legs.py`) were checked
  directly against Pugliese's real published simulation output and all 6
  show substantial rhythmic activity (104-124/128 replicates active,
  scores 0.37-0.71 — see CHANGELOG 2026-09-19). **Not yet done:** the
  motor-neuron side (which specific leg joints/muscles get driven, per
  segment) still needs the same level of scrutiny; grouping by
  joint/flexor-extensor for T2/T3 motor neurons is the remaining Phase 1
  work.
- **Phase 2 — Hexapod body in MuJoCo.** Harness mode (fixed base), sanity
  sine gait, 3 DOF/leg.
- **Phase 3 — Open loop.** Motor neuron rates → joint targets; legs step in
  harness.
- **Phase 4 — Closed loop.** Joint angle → chordotonal-like inputs; foot
  load → campaniform-like inputs. Measure interleg phase coupling.
- **Phase 5 — Ground walking.** Tune interface with CMA-ES (brain weights
  frozen). GPU compute (Kaggle/Colab) likely needed here.
- **Phase 6 — Amputate middle legs → quadruped.** Remap and re-tune
  interface only; connectome/brain weights stay frozen.
- **Phase 7 — Controls.** Shuffled connectome, degree-matched random graph,
  plain oscillator — to show the real wiring matters.

## Working conventions
- **Keep `CHANGELOG.md` updated as you go** — every session that produces a
  finding, decision, or non-obvious debugging fix gets an entry (dated,
  with *why*, not just *what*). This doubles as the video script source;
  don't let it fall behind the code.
- Explain non-obvious modelling choices in comments; note the paper section
  they come from when relevant.
- When uncertain about biology, data fields, or repo behaviour: stop and flag
  it rather than guess.
- Save figures at video quality (dark and light variants when useful).
- Every experiment is config-driven and reproducible from its config + seed.
