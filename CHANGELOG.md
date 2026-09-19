# Findings log — index

Dated, detailed findings live in `docs/logs/`, one file per day (this
mirrors how the project's own memory/index system works: a short pointer
here, the actual content there). Each entry below is one line — read the
linked file for the full reasoning, quotes, and numbers. This is the
source material for the video script, so the linked files are written to
explain *why*, not just *what*.

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
  connectome) form.
- **[2026-09-18](docs/logs/2026-09-18.md)** — Decided to use MaleCNS
  instead of MANC going forward; identified and resolved Pugliese et
  al.'s published T1 walking circuit into MaleCNS; first 3D circuit
  render; found connectivity evidence (later confirmed the next day) that
  the same CPG motif repeats in T2/T3.
