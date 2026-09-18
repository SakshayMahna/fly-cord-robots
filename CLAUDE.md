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

## Scientific background (short)
- Fly walking is largely hardwired. The brain sends high-level commands via
  descending neurons (DNs); the VNC contains central pattern generators (CPGs)
  that produce leg rhythms; leg sensors (proprioception) adjust in real time.
- Pugliese et al. (2025/2026), "Connectome simulations identify a central
  pattern generator circuit for fly walking": a firing-rate model of VNC
  connectomes. Tonic stimulation of DNg100 (walking command neuron) drives
  rhythmic leg motor neuron activity. A minimal 3-neuron CPG (1 inhibitory,
  2 excitatory) was found: DNg100 → IN17A001 (excit.) + INXXX466 (excit.)
  + IN16B036 (inhib.). DNb08 also drives rhythm.
- Known open issue: in their open-loop simulations, legs were rhythmic but NOT
  coordinated (no tripod gait; left/right not phase-locked). They propose that
  proprioceptive feedback / body mechanics may be needed. Closing the loop with
  a physical body is exactly what we test.

## Dataset: MaleCNS, not MANC (decided 2026-09-18)
We use **MaleCNS v1.0** (`male-cns:v1.0` on neuprint, server
`https://neuprint.janelia.org`) as our connectome going forward, not MANC.

- MANC (2023) is nerve-cord-only, ~23k neurons. Pugliese's model was built on
  MANC, and only ever covers the T1 (front-leg) DN→motor-neuron circuit.
- MaleCNS (June 2026, Cell paper Sept 2026) is a different, newer specimen:
  brain + VNC imaged as one continuous volume (including the neck connective
  MANC lacks), ~166,700 neurons. It's a superset for our purposes and keeps
  later phases (real descending drive from the brain, not synthetic tonic
  stimulation) open without a second dataset migration.
- We have to build our own six-leg circuit extraction regardless of dataset
  (Pugliese only did T1), so MaleCNS costs nothing extra structurally.
- MaleCNS neuron records carry a `mancBodyid` field — a per-neuron
  cross-reference to the corresponding MANC specimen body, computed by
  Janelia's annotation team. We use this (not `type`-string matching) to
  relocate neurons Pugliese identified in MANC into MaleCNS's ID space.
  See `fly_robot/connectome/identify_circuit.py` and `CHANGELOG.md` for the
  worked example (T1 circuit, 142/152 neurons resolved).
- Open caveat: Phase 0's original goal (match Pugliese's published rhythm
  figures) assumed MANC-derived weights. Running on MaleCNS-derived weights
  is not numerically the same circuit even with matching neuron identities —
  not yet decided whether to also run a literal MANC reproduction for
  comparison.

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

- **Phase 0 — Reproduce the science.** Query MaleCNS (not MANC) for DNg100,
  the CPG neurons, and the T1 leg motor neurons; confirm they resolve
  correctly (anatomically sane render). **Done:** neuron identification +
  first 3D circuit render. **Not done:** running the actual rate-model
  simulation and comparing rhythms to the paper.
- **Phase 1 — All six legs.** Extract T2/T3 (mid/hind leg) DN/CPG/motor-neuron
  circuits ourselves in MaleCNS — Pugliese only ever did T1, so there's no
  existing table to borrow. *(In progress.)*
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
