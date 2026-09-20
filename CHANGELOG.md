# Findings log — index

Dated, detailed findings live in `docs/logs/`, one file per day (this
mirrors how the project's own memory/index system works: a short pointer
here, the actual content there). Each entry below is one line — read the
linked file for the full reasoning, quotes, and numbers. This is the
source material for the video script, so the linked files are written to
explain *why*, not just *what*.

- **[2026-09-20 — Phase 5 design + compute gate](phase5/DESIGN.md)** —
  Proposal only; nothing built. Specifies the 65-parameter trainable
  adapter (sensory encoder / command drive / motor decoder, all explicitly
  ours), the reward, the curriculum, and the controls, then measures the
  compute gate instead of estimating it
  ([benchmarks](phase5/benchmarks/)). Main findings: the **connectome, not
  the body, is 88.7% of a trial's cost** — so accelerating MuJoCo targets
  9% of the problem; only 471 of 23,532 neurons are ever nonzero in a
  stable trial, and skipping the provably-zero columns gives a
  **bit-identical 3.40× end-to-end** speedup (33.1 s → 9.75 s per trial),
  which inverts to 0.74× under saturation and so needs a density guard;
  CPU parallelism saturates at ~6 workers (~3.8×); and there is **no
  useful lossless subnetwork crop** (only 3.6% of neurons are provably
  inert — the network reaches 13,541 within two hops). Argues, with the
  measurements, that this workload wants **many CPU cores rather than a
  GPU** — the loop is sequential in time, batching the population would
  require batching physics, FlyGym 2.1.0 has no MJX support at all, and 69
  of the body's 70 geoms are meshes. The Colab GPU number the brief asked
  for was **not measured** (no GPU on this machine) and deliberately not
  invented. Default training budget exceeds the brief's 10-hour gate, so
  the work stops there for a decision.
- **[2026-09-19 to 2026-09-20 — closed-loop work](docs/closed_loop/)** —
  A continuous pre-registered arc, kept in its own subfolder rather than
  daily logs (`AUDIT.md`, `SENSORY_MAP.md`, `PREREGISTRATION.md` + two
  dated amendments, `RESULTS.md`, `LOG.md` for the full chronology).
  Built a steppable reimplementation of Pugliese's rate model (their
  adaptive solver can't be interrupted for feedback; ours is validated to
  median r=0.9993 against their published output, ~400x faster), the
  sensory interface, the tethered-on-ball ground rig, and rhythm-gated
  coupling metrics. Ran a pre-registered pilot rather than the full
  experiment: found the open-loop baseline itself already shows same-side
  leg coupling the paper doesn't report (triggered a stop condition,
  resolved by reframing the primary hypothesis to left/right coupling
  specifically); found the first gain sweep sat almost entirely past a
  bifurcation; tried four sensory encoder formulations in response,
  including one grounded in real chordotonal range-fractionation; all
  four land in the same place — feedback is either negligible or destroys
  the rhythm entirely, with the network's own real inhibitory wiring
  active throughout. Reported as the result rather than chased further,
  per a stopping rule committed before the runs that produced it.
- **[2026-09-19](docs/logs/2026-09-19.md)** — Corrected a stale paper
  version we'd been citing (v1 vs v2); v2 already confirms rhythmic
  activity in all six legs by simulation. Got Pugliese's real simulation
  code running ourselves, reproduced their T1 result, then debugged our
  own full-VNC attempt through several real issues (missing synapse
  filter, wrong stimulation protocol, a network-size threshold-scaling
  effect) down to the actual root cause (wrong data file, wrong stimulus
  magnitude — both only found by extracting their real run config).
  Verified our independently-identified T2/T3 candidate neurons directly
  against their real published output: all 6 genuinely rhythmic. Closed
  Phase 1's remaining motor-neuron grouping. Reorganized the repo
  (this file included), then started Phase 2: chose FlyGym/NeuroMechFly
  (real fly anatomy, MuJoCo-native, 7 DOF/leg) as the robot body instead
  of a generic hexapod, and got the harness-mode sanity gait working.
  Phase 3 (open loop): built the motor-neuron-rate-to-joint interface,
  caught and fixed a replicate-instability bug and a subtler
  averaging-cancels-rhythm bug (switched to one real representative
  replicate instead), recalibrated the rate-to-angle scale against
  measured data, and got real connectome-driven leg movement both
  numerically and visually confirmed (top-down camera, derived not
  guessed). Recorded a teleoperation design (not built) and confirmed via
  the literature that Phase 4's closed-loop experiment is a real open
  question — proposed as future work by Pugliese et al. themselves, not
  yet published by anyone in this literal (untrained, unmodified
  connectome) form. Reorganized again: dropped phase-numbered file/folder
  names in favor of content-derived ones, split an overgrown 408-line
  module into four focused ones, verified every renamed script still
  behaves identically, and added a fully-verified references section to
  `README.md`.
- **[2026-09-18](docs/logs/2026-09-18.md)** — Decided to use MaleCNS
  instead of MANC going forward; identified and resolved Pugliese et
  al.'s published T1 walking circuit into MaleCNS; first 3D circuit
  render; found connectivity evidence (later confirmed the next day) that
  the same CPG motif repeats in T2/T3.
