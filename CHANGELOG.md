# Findings log — index

Dated, detailed findings live in `docs/logs/`, one file per day (this
mirrors how the project's own memory/index system works: a short pointer
here, the actual content there). Each entry below is one line — read the
linked file for the full reasoning, quotes, and numbers. This is the
source material for the video script, so the linked files are written to
explain *why*, not just *what*.

- **[2026-09-25 — the reward forbade walking: a positive control was
  missing](docs/trained_adapter/GROUND_BRIDGE.md#9-the-actual-blocker-the-reward-forbade-walking)**
  — After R1a v2 also converged on inaction (this time a **silent
  connectome**: drive 51.8, `peak_n_active` 0), the reward itself was
  tested rather than patched a third time. Across **7,488 episodes in two
  runs nothing ever walked** (max 2.156 mm in 4 s = 0.54 mm/s vs the
  restricted CPG's proven 9.822), and trials that *did* move were punished
  (mean reward −5.01 / −3.99 vs ~0 for standing still). Scoring the
  known-good gait end to end through `evaluate()` settled it: the
  restricted CPG walking 34.76 mm scores **−15.95**, against **0.00** for
  standing perfectly still — the reward did not fail to reward walking, it
  forbade it. Cause: `energy_ref = 3.331e-05`, carried over unchanged from
  the ball reward where it was calibrated from the **untrained,
  near-motionless** baseline; real gaits measure **325–539×** that, so a
  −0.05 "shaping" term evaluated to −16.5 and dominated the objective.
  Both runs were therefore *playing correctly* — the earlier "holes"
  (motionless rhythm, then silence) were the best available strategies, and
  neither result says anything about the connectome. Recalibrated to
  **1.0814e-02**, the restricted CPG's own measured energy (same method as
  `walking_mm_s`): the gait now scores **+0.5323** with energy at −0.0509.
  The real lesson is methodological — there was a `sign_test` but **no
  positive control**, and a sign test cannot catch a mis-scaled magnitude
  because both directions were scaled wrongly. Three tests now score a
  known-good gait through the reward; they would have caught this before
  either run started.
- **[2026-09-25 — R1a run 1: the optimiser stood still and got paid for
  it](docs/trained_adapter/GROUND_BRIDGE.md#8-r1a-run-1-the-optimiser-found-a-hole-and-what-it-tells-us)**
  — First real ground-walking run. Stability was solved fast (flipping 20%
  → 3% in 25 generations) but forward progress never left zero. Rendered
  the best candidate rather than trusting its +0.3379: it travels
  **−0.010 mm in 4 s** and is **visually motionless across the whole
  trial**, scoring almost entirely on **rhythmicity +0.2500 and
  coordination +0.1089** while `progress` contributes −0.0002. The
  connectome's rhythm is real (5/6 legs, 12.00 Hz — §7a) but at ~1/11th
  the excursion needed to move the body, so the reward was paying full
  price for rhythm that never becomes locomotion. A first reading blamed
  the rhythm terms for the whole search and **was wrong**: the per-term log
  shows the population is dominated by **saturation at −0.5585** (mean
  n_active ≈ 1,900), so most of parameter space explodes the network and
  one safe corner pays 0.36 for standing still. Raising `progress`'s weight
  cannot fix that (at dx ≈ 0.001 mm it is ~1e-5; even 100× stays
  invisible), so the rhythm terms are now **gated on forward displacement**
  — full credit at 1 mm/trial (0.25 mm/s, 1.8% of the CPG baseline), zero
  at rest, and zero for walking backwards so the gate cannot be bought in
  reverse. The exploiting candidate drops +0.3379 → −0.0210; genuine
  walking keeps full credit. New reward hash `707b7c68…`, so the contract
  guard refuses to resume run 1 under it.
- **[2026-09-25 — the ground bridge: a wrong objective caught, and the
  feasibility gate that should have come first](docs/trained_adapter/GROUND_BRIDGE.md)**
  — 70 generations of CMA-ES were run, resumed five times and reported on
  while scoring **ball rotation on a tethered fly**, not walking: the
  trainer passed `on_ball=True` with the ball reward (hash `88f681a4...`)
  months after the move to free ground, and the mismatch against
  `reward_ground`'s `0d648542...` was printed at every launch and never
  checked. The flat learning curve (no improvement after generation 2) and
  negative mean progress were the signal and were read as noise. Fixed
  structurally, not by care: the trainer now **refuses any rig but ground**,
  stores a training contract in every checkpoint and refuses cross-objective
  resumes, and five gates in `tests/test_ground_training_contract.py` make
  the specific mistake unrepeatable — including one that asserts
  `sensory_drive` is *numerically* zero when the sensory path is off. Then
  the cheap gate that should have preceded any long search: FlyGym's own CPG
  controller, restricted to the **18 of 42 DOFs the adapter can actually
  move**, walks at **9.822 mm/s against 13.978 unrestricted (70%)** — so the
  action space is *not* the blocker and R1a is worth running. That gate's
  first version reported the opposite, an artifact of `JointDOF` objects not
  matching string names, caught only because it printed how many DOFs it
  drove; it now raises instead of reporting a confident wrong verdict.
  Then two diagnostics before launching: the connectome's decoded joints
  already oscillate at **12.00 Hz, matching the working gait DOF for DOF**,
  but at **1/11th the excursion** (0.062 vs 0.707 rad) — and turning the
  amplitude up does not walk, it **flips the fly** (11 of 12 sweep
  configurations terminated; uprightness falls monotonically with
  amplitude; the only survivor is the default). Coherent with Pugliese's
  own finding that DNg100 drive yields no left/right phase coupling:
  scaling an uncoordinated rhythm scales the incoordination too. R1a
  therefore launches from defaults, and must solve excursion and stability
  jointly — which uniform scaling provably cannot and per-leg
  differentiated control might.
- **[2026-09-20 to 2026-09-21 — trained adapter: design, compute gate, and
  a lossless 3.4×](docs/trained_adapter/)** — Design and measurement only;
  **the trainer is deliberately not built** (`DESIGN.md` for the current
  state, `LOG.md` for the chronology, `benchmarks/` for every number).
  Specifies the 65-parameter trainable adapter (sensory encoder / command
  drive / motor decoder, all explicitly ours), reward, curriculum and
  controls, then measured the compute gate rather than estimating it. The
  headline inverted the assumption the work was planned on: **the
  connectome is 88.7% of a trial's cost and the body is 9.0%**, so
  accelerating MuJoCo targets a tenth of the problem. Noticing that the
  *rate vector* is sparse where only the *matrix* had been exploited gave a
  **bit-identical 3.40× end-to-end** speedup (33.1 s → 9.75 s per trial) —
  which **inverts to 0.74×, i.e. slower, under saturation**, so it ships
  behind a density guard with two regression tests aimed squarely at the
  fallback. Also measured: **no useful lossless crop exists** (only 3.6% of
  neurons are provably inert; the network reaches 13,541 within two hops),
  CPU parallelism saturates around 6 workers, and per-worker memory drops
  **3.29 GB → 0.55 GB** with `jax.clear_caches()` — the number that decides
  VM sizing. Concluded, with the measurements behind it, that the workload
  wants **many CPU cores rather than a GPU**; the Colab GPU figure the
  brief asked for was **not measured** (no GPU here) and deliberately not
  invented. Recovered Phase 4's frozen-interface config hash
  (`04be9dec…`), whose generating code was never committed, so it is now
  checkable rather than merely quoted. Then measured the **score noise**
  before proposing a reward, and it reframed the design: one saturated
  replicate contributes ~97% of the variance in the progress term (the
  pre-registered stability filter is worth **27×**), and that replicate
  spins the ball at **30× any healthy one** — a seizing network is
  *instrumentally attractive* to an optimiser, which makes the saturation
  penalty the largest-weighted reward term rather than a precaution. The
  binding episode count is set by **coordination, not progress** (tripod
  index needs E ≥ 6; progress needs E ≥ 1). Also caught a sign hazard:
  forward walking is **negative** pitch, so a flipped sign would train a
  backwards-walking fly while the score read as success throughout.
  `REWARD.md` is the approval gate; the trainer is still not built.
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
