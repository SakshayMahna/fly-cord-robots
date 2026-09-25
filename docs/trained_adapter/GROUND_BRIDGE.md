# The ground bridge — what is actually being tested, and what happened first

*Written 2026-09-25, after a significant and expensive error. The error is
recorded in full here because the fix is procedural, not just a code
change.*

## 1. What went wrong

Seventy generations of CMA-ES were run, stopped and resumed five times, and
reported on — **optimising the wrong objective the entire time.**

The trainer called `run_trial(..., on_ball=True)` and scored trials with the
**ball reward** (`fly_robot/adapter/reward.py`, config hash `88f681a4…`).
That is the tethered-ball rig: thorax fixed, legs on a freely rotating
sphere, progress measured as ball rotation. It is not walking.

The decision to move Stage A training to **free-ground walking** had been
made much earlier and acted on everywhere else: the free-ground rig was
built, `reward_ground.py` was written with nine terms, the adhesion gate was
derived from motor output, `v_ref = 14.025 mm/s` was measured on ground, the
detector suite was re-run on the new rig, and all of it was documented. The
trainer simply never called any of it. `run_trial` grew a `rig=` parameter;
the trainer kept passing the older `on_ball=True`.

**Nothing failed loudly.** Worse, the evidence was on screen every single
launch: the trainer printed `reward config hash: 88f681a4…` at startup,
while `reward_ground.reward_config_hash()` is `0d648542…`. Those two strings
were both recorded in this project's own notes. The mismatch was visible
roughly a dozen times and was not checked.

**Consequence, stated plainly.** The generation-70 run is a valid
*ball-rig adapter experiment*. It is **not** evidence about walking, and its
best score (+0.2928) means nothing about locomotion. It must never be
presented as "the fly learned to walk." Its honest use is as a documented
rejected objective — including in the video, where it is a real story beat:
the loophole was found and the experiment was corrected.

Two independent things also deserve noting, because they were the actual
signal and were dismissed as noise: the run's best score never improved
after generation 2 (68 generations flat), and mean progress stayed
*negative*. A run that is not learning is a reason to audit the objective,
not to keep resuming it.

### Why it was possible, and what now prevents it

The objective was selectable by a **default nobody re-read**. `on_ball=True`
was a plausible-looking argument in a call that already had six others.

`tests/test_ground_training_contract.py` now makes that specific mistake
impossible to repeat quietly:

| gate | what it blocks |
|---|---|
| `test_trainer_uses_the_ground_reward_not_the_ball_reward` | the trainer's reward module must *be* `reward_ground`, checked by config-hash identity so re-aliasing the import cannot hide a swap |
| `test_trainer_refuses_any_rig_but_ground` | `Trainer` raises on `rig` in {ball, tethered, harness, ""} — the objective is no longer reachable by default |
| `test_output_stage_requires_sensory_off_and_vice_versa` | an inconsistent stage/sensory pair is refused, not silently resolved |
| `test_sensory_off_injects_exactly_zero_current_into_the_connectome` | runs two real trials and asserts `sensory_drive` is *numerically* zero — a flag that is read but not honoured is exactly this class of bug |
| `test_stages_partition_the_parameter_vector_exactly` | R1b cannot secretly re-train part of R1a's solution |

The trainer additionally stores a **training contract** (rig, stage, active
parameter mask, warm-start hash) in every checkpoint and refuses to resume
across a different one. Two objectives can no longer be spliced into one
learning curve — which is precisely what five casual resumes did.

## 2. What the experiment actually is

The honest statement of the question:

> Can a **frozen** fly VNC connectome, with a **small, constrained,
> trainable body interface**, produce walking on a fly body?

The connectome's wiring, weights and neurotransmitter signs are never
trained — that constraint is unchanged and is what separates this from
FlyGM-style work where connectome topology seeds a network that is then
trained by RL (CLAUDE.md, scientific background). Only the interface is
learned.

### The staged ladder

Training all 71 parameters at once against a distant reward conflates
"can the connectome's output drive this body" with "can a learned sensory
encoder rescue it." The stages separate those, and are enforced in code
(`PARAMETER_GROUPS` / `TRAINING_STAGES` in `adapter/parameters.py`):

| stage | parameters | sensory path | question |
|---|---:|---|---|
| **R1a** (`--stage output`) | **40** — command drive, motor decoder, adhesion | **off** (zero current, asserted) | Can the connectome's *own output*, decoded properly, walk the body? |
| **R1b** (`--stage sensory`) | **31** — chordotonal + hair-plate encoders | on | Starting from R1a's solution, does proprioceptive feedback *improve* it? |
| `full` | 71 | on | exploratory only; not a clean test of either question |

40 + 31 = 71 exactly, with no overlap (`test_stages_partition…`). R1b takes
R1a's best vector via `--warm-start`, so its comparison is against a real
baseline rather than a fresh search.

**Why R1b is a separate rung rather than the default.** Phase 4 established
that this sensory interface has a cliff: below it, sensory current is too
weak to recruit neurons; above it, many cross threshold together and the
rhythm is destroyed (`docs/closed_loop/RESULTS.md`). There is also **no
leg-load / campaniform channel in either MANC or MaleCNS annotation**, so
foot contact is not sensed at all (`docs/closed_loop/SENSORY_MAP.md`) — a
real limitation, not an oversight. R1b must therefore be judged against
**R1a**, and against **R1b with sensory gains forced to zero**, or extra
parameters will look like extra capability.

## 3. The feasibility gate that runs *before* any long search

`fly_robot/baselines/restricted_action_cpg.py` — our own diagnostic
addition, not fly biology, and never part of a reported walking result.

The adapter can only move **18 of the body's 42 actuated leg DOFs**: the
three pitch axes per leg that `ANTAGONIST_PAIRS` maps a motor-neuron
antagonist pair onto (coxa-pitch, trochanterfemur-pitch, tibia-pitch). The
other 24 are held at neutral — because their motor modules are unmapped, or
because they are **silent in the model**: `substrate grip` and
`tarsus control` never fire at all (`RESULTS.md`).

So before spending 24,000 trials: take FlyGym's own CPG controller, which
walks this body, and give it **only those same 18 DOFs**, masking joint
angles only and leaving its adhesion untouched, so the DOF restriction is
the single variable.

**Result (2026-09-25, 2 s trials, seeds 0–2):**

| condition | DOFs driven | mean speed | sd | min upright |
|---|---:|---:|---:|---:|
| unrestricted | 42 | 13.978 mm/s | 0.156 | 0.966 |
| **restricted** | **18** | **9.822 mm/s** | **0.079** | 0.959 |

The restricted controller reaches **70% of full-DOF speed**, forward, upright,
with low variance across seeds, and with lateral drift *smaller* than the
unrestricted gait's (|dy| 0.33–1.27 mm vs 1.39–7.17 mm). Full record:
`media/trained_adapter/restricted_action_cpg.json`.

**The 18-DOF action space is sufficient for walking.** Therefore a walking
failure in R1a is attributable to the connectome's output or the decoder,
**not** to a crippled action space. That is what makes R1a worth running.

> A methodological note kept deliberately: the first version of this gate
> reported "CANNOT support walking" — an artifact. `dof_order` holds
> `JointDOF` objects, not strings, so zero of the 18 names matched and the
> restricted run froze *every* DOF. It was caught only because the script
> prints how many DOFs it actually drove. The script now raises if the
> matched count is not exactly 18, rather than reporting a confident wrong
> verdict. A gate that can fail silently in the pessimistic direction would
> have sent this project off to redesign an interface that was fine.

## 4. Claims this work can and cannot support

Kept separate on purpose, because they are easy to blur in a video script:

**Strict (what R1a/R1b test).** The runtime path is
`DNg100 → frozen connectome → motor neurons → trained decoder/adhesion →
body`. No hand-designed oscillator, no sine wave, no body-state shortcut
around the connectome (`adapter_sensory=False` verified numerically). If
this walks, the claim is: *a real fly VNC circuit's own rhythmic output,
translated through a small trained interface, walks a fly body.*

**Assisted (Rung 2, if built).** The connectome is known to produce **no
consistent left/right phase coupling** under DNg100 drive alone — that is
Pugliese et al.'s own finding (Fig. 4d-e; Extended Data Fig. 9b), and
closing that loop with a body is this project's central bet. If R1a moves
but does not coordinate, a **bounded interleg conductor** is a legitimate
next step — but it is **our engineering addition**, must start at zero
coupling, must have a K=0 ablation, and any result using it is
"connectome-guided controller," not literal connectome locomotion.
Rung 2 is **not** buildable yet regardless: its causal phase estimator
fails its own pre-set validation bar on one leg (`RUNG2_DESIGN.md` §10).

**Not supported by anything here.** That the wiring was *necessary* — no
connectome control is trainable (C1 and C2 both fail the pre-registered
matching sequence), so the only comparison is against the C3 hand-designed
baselines (`RESULTS.md`). And nothing here is about a physical robot or a
generic quadruped; a middle-leg lesion of NeuroMechFly is not a quadruped
robot, and would need its own interface and validation.

## 5. Calibration by a conventional controller — allowed, with a rule

Using the working CPG controller's joint trajectories to *initialise or
sanity-check* the decoder's scales and adhesion timing is legitimate and far
cheaper than asking CMA-ES to discover every joint scale from zero. The rule
is that it must be **absent at evaluation time**: no CPG output may enter
the runtime path of a reported result, and any such use must be stated
outright — "a conventional controller calibrated the body interface; the
reported behaviour is driven only by the connectome." Not yet done; recorded
here as approved-in-principle, not as something already used.

## 6. Run commands

```bash
# Feasibility gate first — seconds, not hours.
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
    -m fly_robot.baselines.restricted_action_cpg --seeds 0 1 2

# R1a: ground, output-side only, sensory path off.
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
    -m fly_robot.experiments.train_adapter \
    --run-name r1a_ground --stage output --episodes 6 \
    --generations 250 --workers 6 --out-dir media/trained_adapter

# R1b: warm-start from R1a, unlock the sensory block only.
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
    -m fly_robot.experiments.train_adapter \
    --run-name r1b_sensory --stage sensory \
    --warm-start media/trained_adapter/r1a_ground/best.json \
    --episodes 6 --generations 250 --workers 6
```

`--workers 6` is the measured local throughput knee on the M3 Pro
(efficiency falls off past 6; 8 workers gave ~1,140–1,200 trials/hr with
~20 h for 250 generations). Both runs checkpoint every generation and resume
automatically; a resume under a changed contract is refused rather than
silently mixed.

**Never resume the generation-70 `rung1` checkpoint under the ground
objective.** It was written under a different reward hash and a different
rig; the contract check will refuse it, and it should be kept intact as the
ball-rig record.

---

## 7. Two diagnostics run before launching R1a, and what they found

*2026-09-25. Both cheap, both run because a 20-hour search deserves a
sanity check first, and both changed what we expect.*

### 7a. The connectome's frequency is already right; its amplitude is not

`fly_robot/experiments/compare_decoded_to_working_gait.py` compares the
untrained decoder's joint motion against the restricted CPG gait that we
now know walks, on the same 18 DOFs.

| | connectome (untrained) | working CPG | ratio |
|---|---:|---:|---:|
| mean peak-to-peak | **0.062 rad** | 0.707 rad | **0.09×** |
| median dominant frequency | **12.00 Hz** | 12.00 Hz | **1.00×** |

**The stepping frequency matches almost exactly, DOF for DOF.** The
connectome's rhythm is reaching the joints, in band, at a plausible walking
rate — that is a real positive result and it is worth stating plainly.

What is wrong is the **excursion**: about 1/11th of a gait that walks. Also
recorded: for this replicate the **entire left hind leg is effectively
still** (three DOFs at 0.0002–0.0007 rad), as is the left-middle tibia —
4 of 18 DOFs are not meaningfully driven. That asymmetry is in the
connectome's own output, not something the decoder chose.

### 7b. Raising the amplitude does not produce walking — it flips the fly

`fly_robot/experiments/calibrate_decoder_amplitude.py` sweeps the two
constants that set excursion (`motor_gain` ∈ [0.05, 1.20], `motor_scale` ∈
[0.50, 20.0] — scale divides the tanh argument, so smaller = larger swing)
and measures **forward speed**, not just amplitude.

**11 of 12 configurations flipped the fly.** The only survivor is the
default (gain 0.50, scale 3.00), at −0.140 mm/s and amplitude 0.09×. Mean
uprightness falls monotonically as amplitude rises (0.909 → 0.803), and no
setting reaches even 0.3× the working gait's excursion before falling over.

So the naive reading of 7a — "just turn up the gain" — is **wrong**, and it
is much better to have learned that in four minutes than after a 20-hour
search warm-started into it.

**Why this is coherent rather than discouraging.** Bigger strides from an
*uncoordinated* rhythm are worse than small ones. The connectome under
DNg100 drive produces **no consistent left/right phase coupling** — that is
Pugliese et al.'s own published finding (Fig. 4d-e; Extended Data Fig. 9b),
and the reason this project exists. Uniform amplitude scaling cannot fix
that, because it scales the uncoordinated pattern up along with everything
else. Per-leg, per-segment differentiated control plausibly can, and that is
exactly what R1a's 40 parameters are: `motor_gain`, `motor_scale`,
`motor_offset` and `motor_tau` per (antagonist pair × segment), plus
per-segment adhesion weight and threshold.

A methodological note, since it nearly produced a bad launch: the sweep's
first selection picked "fastest = gain 0.50 / scale 1.50 at +0.000 mm/s"
— from configurations that had all **flipped**, because a terminated trial
reports 0.000 mm/s and a max-by-speed happily selects one. Warm-starting a
20-hour run there would have been worse than not calibrating at all. The
script now selects among survivors only and raises if none survive.

### What this means for the launch

**R1a starts from the defaults, with no warm start** — the calibration
found nothing better, which is itself the finding. The search now has a
well-posed job and a strong gradient to work with:

* the action space is sufficient (§3: 70% of full-DOF speed is reachable),
* the frequency is already correct (7a),
* flipping is heavily penalised (reward −0.92 to −1.01) versus merely not
  moving (−0.14), so CMA-ES has a clear signal away from the failure mode
  that dominates the naive fix,
* and it must solve excursion **and** stability jointly, which uniform
  scaling provably cannot and differentiated per-leg control might.

This is a genuine experiment with a real chance of either outcome. If R1a
plateaus near "upright but barely moving", the diagnosis is already in
place: coordination, not amplitude, and Rung 2's bounded conductor is the
pre-registered next step — still blocked on its phase estimator
(`RUNG2_DESIGN.md` §10).

---

## 8. R1a, run 1: the optimiser found a hole, and what it tells us

*2026-09-25. Stopped cleanly at generation 41 of 250 (~2.5 h). Checkpoint
intact at `media/trained_adapter/r1a_ground/`; videos at
`media/trained_adapter/r1a_best_r{0,1,3}_{top,side}.mp4`.*

### What happened

Termination (flipping) fell from ~20% of trials to ~3% within 25
generations — the search solved *stability* quickly and convincingly. But
**forward progress never moved off zero**:

| | gens 10–20 | gens 20–39 |
|---|---:|---:|
| `term_progress` (population mean) | −0.00010 | −0.00012 |

Best score plateaued at +0.3154 (gen 18) → +0.3178 (gen 38): twenty
generations for +0.0024, with sigma flat at 0.50–0.54 (still exploring, not
numerically stalled).

### What the best candidate was actually doing

Rendered and inspected, per this project's standing rule. Across three
replicates it travels **−0.010 mm, +0.001 mm and +0.000 mm in 4 s**, stays
essentially perfectly upright (0.998–0.999), never flips, never saturates
(peak `n_active` 269–406 against a 1,500 threshold). Three frames sampled
across the whole trial are **visually identical** — a fixed splayed pose,
no perceptible leg motion.

Its measured reward breakdown:

| term | weighted |
|---|---:|
| **rhythmicity** | **+0.2500** (of a possible 0.30) |
| **coordination** | **+0.1089** |
| energy | −0.0192 |
| progress | **−0.0002** |
| saturation | −0.0000 |
| **total** | **+0.3379** |

**The entire score is rhythm credit.** The connectome genuinely is producing
rhythmic motor-neuron output — `n_rhythmic` 5/6 is real, and §7a measured
its frequency at exactly the right 12.00 Hz — but at ~1/11th the excursion
needed to move the body, so none of it becomes locomotion. The reward was
paying full price for that.

### A correction to a claim made in the moment

An initial read attributed the plateau to the rhythm terms dominating the
*whole search*. The per-generation term log does not support that. Across
the **population**, the dominant term by far is **saturation, at −0.5585**
(mean raw ≈ 0.28, i.e. mean `n_active` ≈ 1,900). The true picture is:

* most of the parameter space drives the network past 1,500 active neurons
  and eats a −2.0-weighted penalty;
* one safe corner avoids that entirely and collects ~0.36 of rhythm credit
  for standing still;
* CMA-ES, correctly, went to the corner.

The distinction matters for the fix. **Raising `progress`'s weight cannot
work**: at dx ≈ 0.001 mm, `progress = dx / (4 s × 14.025 mm/s) ≈ 1e-5`, so
even a 100× weight stays invisible. Progress is not being outweighed — it
is unmeasurably small while rhythm pays in full.

### The fix: a locomotion gate on the rhythm terms

Rhythm now earns credit only in proportion to the body actually moving:

```
gate = clip(max(0, dx) / rhythm_gate_dx_mm, 0, 1)
rhythmicity  *= gate
coordination *= gate
```

`rhythm_gate_dx_mm = 1.0` — 1 mm over a 4 s trial is 0.25 mm/s, **1.8% of
the CPG baseline**. It is deliberately a low bar: it exists to exclude
standing still, not to demand good walking. The `progress` term remains
what rewards speed. (`dx` here is post-transient displacement, the same
quantity `progress` uses, so at exactly 1 mm of total travel the gate sits
at 0.875 rather than 1.0.)

**Gated on forward `dx`, not `|dx|`.** Gating on absolute distance would let
a candidate collect the full ~0.36 by walking *backwards* against a progress
penalty three orders of magnitude smaller — trading one hole for another.
With `max(0, dx)`, going the wrong way earns the same zero as standing
still.

Verified on the exact candidate that exploited it:

| replicate | dx | score before | score after |
|---|---:|---:|---:|
| 0 | −0.010 mm | +0.3379 | **−0.0210** |
| 1 | +0.001 mm | +0.3252 | **−0.0023** |
| 3 | +0.000 mm | +0.3560 | **−0.0041** |

And confirmed not to punish the case it exists to reward — a synthetic
tripod-rhythmic trial at CPG-baseline travel keeps **full** credit
(rhythmicity +0.3000, coordination +0.2000, total +1.3750); credit is
monotonic in forward distance with no cliff; backward travel earns zero.
Five tests in `tests/test_ground_training_contract.py` pin all of this.

Reward version `2026-09-25-ground-locomotion-gated`, hash `707b7c68…`
(was `0d648542…`). The contract guard therefore **refuses** to resume
generation 41 under it — correctly, since that run optimised a different
objective. Run 1 is kept as the record of the finding.

### What this run does and does not tell us

It does **not** show the connectome cannot walk. It shows that under a
reward which pays for rhythm unconditionally, the cheapest way to score is
to stand still and hum — and the optimiser found that in under 20
generations. That is a statement about the reward, not the biology.

What it does establish, and is worth keeping: the search **can** solve
stability fast (20% → 3% flipping), the connectome's rhythm reaches the
motor neurons at a correct walking frequency, and the binding constraint on
the population is **neural saturation**, not stability or coordination.

### 8b. The second fix: the drive bound was feeding the saturation penalty

The term log said the population was dominated by **saturation (−0.5585,
mean `n_active` ≈ 1,900)**, so that was measured rather than reasoned about.

Drive sweep on replicate 0, adapter otherwise at defaults:

| drive | 350 | 382.8 | 385 | 388 | 390 | 392 | **395** | 400 | 440 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `n_active` | 336 | **518** | 514 | 550 | 574 | 577 | **1,153** | **3,919** | 4,057 |

**The pre-registered matched drive sits ~10 units (2.7%) below a cliff.**
Past ~393 the network goes from ~600 active neurons to ~4,000.

Now the search geometry. With the old bound `dng100_level ∈ [0, 600]`, the
sigmoid mapping put z = 0 at 382.8 but **z = +0.5 — exactly one `sigma0` —
at drive 446**, deep in saturation:

| z | −2 | −1 | 0 | **+0.5** | +1 | +2 |
|---|---:|---:|---:|---:|---:|---:|
| old bound (600) | 115.6 | 236.0 | 382.8 | **446.4** | 496.4 | 557.2 |
| **new bound (400)** | 300.4 | 356.5 | 382.8 | **389.4** | 393.5 | 397.6 |

So roughly **47% of every generation's samples on this axis** were being
spent on trials that saturate and collect a −2.0-weighted penalty. That is
the −0.5585, and it is a property of the *search bound*, not of the
connectome.

Narrowing the upper bound to **400** leaves one sigma at 389.4 (safe),
puts the cliff at about +1σ instead of +0.13σ, and drops the wasted
fraction to ~16%. Preserved deliberately:

* **the starting drive is unchanged** — z = 0 still decodes to 382.8125
  exactly, so the pre-registered matched value is untouched and its
  regression test still passes;
* **downward range is essentially intact** (z = −2 → 300, z = −3 → 210),
  which matters because a *lower* drive paired with a higher motor gain is
  a plausible solution the search should still be able to reach.

This changes a **search bound only** — not the connectome, not the reward,
not the drive the run starts from. And raising drive was never a route to
the excursion the decoder actually lacks: more drive buys saturation, not
amplitude (§7b).

---

## 9. The actual blocker: the reward forbade walking

*2026-09-25. Found after R1a v2 also converged on inaction — this time on a
**silent connectome** (`dng100_level` 51.8, `peak_n_active` 0,
`n_rhythmic` 0/6). Two different "do nothing" solutions in two runs is a
pattern, so the reward itself was tested instead of patched a third time.*

### The measurement that settles it

Across **7,488 episodes** in both runs, **nothing ever walked**. Maximum
forward travel ever recorded: **2.156 mm in 4 s = 0.54 mm/s**, against the
restricted CPG's proven 9.822 mm/s. Fewer than 0.5% of episodes exceeded
1 mm.

And the trials that *did* move were punished:

| | movers (dx > 0.5 mm) | near-still |
|---|---:|---:|
| run 1 mean reward | **−5.01** | +0.08 |
| run 2 mean reward | **−3.99** | −0.05 |

Movement earned ~+0.07 and cost ~−4 to −5. So the decisive test: **score
the gait that is known to walk.** Scored end to end through
`reward_ground.evaluate()` (sampling verified exact — physics 0.1 ms,
sampled every 10 steps = `NEURAL_DT`):

| term | restricted CPG (34.76 mm, 8.69 mm/s) |
|---|---:|
| progress | +0.6195 |
| **energy** | **−16.5320** |
| posture | −0.0264 |
| **TOTAL** | **−15.9488** |

**Standing perfectly still scores 0.00.** The reward did not merely fail to
reward walking — it forbade it, by about 16 points.

**Both runs were therefore playing correctly.** What §8 called a "hole"
(motionless rhythm, +0.3379) and what v2 found next (a silent network,
0.0000) were simply the best available strategies. Neither run could ever
have found locomotion, and neither result says anything about the
connectome.

### Root cause: one constant, calibrated against the wrong thing

`energy_ref` was **3.331e-05**, carried into the ground reward unchanged
from the ball reward (REWARD.md: "terms 5–9 unchanged from the approved
reward"), where it had been calibrated from the **untrained baseline** — a
fly that barely moves. Measured against gaits that do:

| gait | raw energy | × old reference |
|---|---:|---:|
| restricted CPG (18 DOF) | 1.0814e-02 ± 2.7e-06 | **325×** |
| unrestricted CPG (42 DOF) | 1.7945e-02 ± 5.1e-06 | 539× |

So a term whose weight is −0.05, and whose stated purpose in REWARD.md is
"to stop degenerate flailing, **not to shape gait**", evaluated to −16.5
for a normal gait and became the largest term in the objective.

**Fix:** `energy_ref = 1.0814e-02`, the restricted CPG's measured energy —
calibrated exactly the way `walking_mm_s = 14.025` already is, from the CPG
baseline on this rig. Result:

| | before | after |
|---|---:|---:|
| CPG gait energy term | −16.53 | **−0.0509** |
| **CPG gait total** | **−15.95** | **+0.5323** |

Good walking now costs ~−0.05 as intended; flailing at ~10× a walking
gait's energy still costs ~−0.5. Note the +0.5323 is earned with **zero**
rhythm credit (this controller is not connectome-driven, so `motor_rates`
are zeros) — a connectome-driven walker can add up to +0.50 on top.

### The methodological gap, which is the real lesson

The reward had a `sign_test()` (forward must outrank backward) but **no
positive control**: nothing asserted that a known-good gait scores well. A
sign test cannot catch a mis-scaled magnitude — both directions were
scaled wrongly and it passed happily throughout.

`tests/test_ground_training_contract.py` now runs the restricted CPG gait
through the reward and asserts it beats standing still, that the energy
term stays a shaping term for real walking (|term| < 0.15), and that it
still has teeth against flailing. **That test would have caught this
before either run started**, and it is the same discipline this project
already applies everywhere else: validate the instrument against a
reference before trusting it.

Reward version `2026-09-25-ground-gated-energy-recalibrated`, hash
`6260cec8…`. 104 tests pass.

### What still stands, and what to watch next

The energy miscalibration was dominant but is not the only obstacle. Under
the corrected reference, run-1 movers' energy penalty falls from −1.32 to
about −0.004, but their **saturation** penalty (−3.32 mean; 96% of run-1
movers saturated) remains. That is a legitimate constraint — Pugliese's own
1,500-neuron criterion — rather than a miscalibration, and the narrowed
drive bound already halved it in run 2 (−1.82). Whether motion in this
system *requires* driving the network into saturation is now the open
question, and it is exactly what R1a v3 will answer.

---

## 10. Why v3 stalls: no traction, and never six live legs

*2026-09-25, at generation ~70 of R1a v3 — the pre-committed 60–80
checkpoint. With the reward corrected, the run no longer collapses to
inaction, but progress plateaued at ~0.03–0.05 mm/s (0.3–0.5% of the
restricted CPG's 9.822). This is the mechanical diagnosis.*

### Finding 1 — the adhesion gate never cycles

A gait needs each foot to grip during stance and release during swing.
Measured on the working restricted-CPG gait:

| | per-leg duty | stance↔swing transitions / 4 s |
|---|---|---|
| **working CPG** | 0.62, 0.65, 0.69, 0.63, 0.64, 0.69 (uniform ~65%) | 95–96 → **12 Hz cycling** |
| **trained best (gen 49)** | 0.008, 0.098, 0.207, 0.000, 0.000, 0.014 | feet essentially never grip |
| **another candidate (gen 68)** | 0.34, **1.00**, 0.43, **0.00**, **1.00**, 0.03 | 4 of 6 legs never cycle |

Different candidates fail in opposite directions — barely gripping, or
permanently pinned — but none cycles. **Without stance/swing alternation
there is no traction**, so leg motion cannot become forward thrust.

**The bound is not the obstacle.** For each leg with real rhythmic output,
the threshold that would produce a 65 %-duty cycle is comfortably inside
the searchable range:

| leg | θ for 65 % duty | θ the search chose | bounds |
|---|---:|---:|---|
| lf | −2.446 | +7.237 | [−9, 9] |
| lh | −1.908 | +4.491 | [−9, 9] |
| rm | −1.249 | +2.464 | [−9, 9] |
| rh | −0.043 | +4.491 | [−9, 9] |

The search could reach a working duty and chose not to — because gripping
only pays off *if the rest of the gait is already coordinated*. Alone, it
just anchors the animal. That is a credit-assignment problem, not a
parameterisation problem.

### Finding 2 — there are never six live legs

`lm` and `rf` have **identically zero** motor rate in the best candidate:
no θ can gate a signal that does not exist. Worse, the two dead legs are
both members of the same tripod group ({rf, lm, rh}), so tripod
alternation is structurally impossible for that candidate.

It is not simply a drive artefact — sweeping drive with the trained
parameters never recovers all six:

| drive | 343.5 | 360.0 | 382.8 | 392.0 |
|---|---|---|---|---|
| live legs | 4/6 | 5/6 | 5/6 | 5/6 |

**A hypothesis tested and rejected.** I suspected `command_asymmetry`
(pinned at 95 % of its range) was starving one side. It is not: forcing it
to zero gives **3/6** live legs — a *different* three (lm, rf, rh) — and
less forward travel (−0.026 mm vs +0.216). Asymmetry shifts *which* legs
animate and genuinely produces the most displacement available, which is
why the search maximised it. The lateral drift it causes is nearly free:
`straightness` averages only 0.30× the magnitude of `progress`, and among
episodes that move at all, **62 % travel further sideways than forward**
(median |dy|/|dx| = 1.35).

### Finding 3 — the search is pressed against its walls

21 of 40 trained parameters sit at or within 8 % of a bound, and the
pattern is coherent: `motor_scale` maxed (≈19 of 20) on most middle/hind
entries, which flattens `tanh(Δrate/scale)` and suppresses those legs'
excursion; `motor_gain` at minimum on two hind entries; `motor_offset`
pinned at ±0.29 of ±0.30 on five entries, i.e. holding joints at extreme
static postures. The search is not tuning a gait — it is switching legs
off and bracing.

### Summary of the mechanism

The connectome's rhythm is real and at the right frequency (§7a). But the
decoded output animates at most five of six legs, the adhesion gate never
cycles, and so no traction is produced. With walking unreachable, the only
strategy that scores is asymmetric drive that pivots the body for a small
net +x — which is exactly what the search found.

## 11. The line we must not cross

*Raised by the user at this point, and it is the right question to ask
before touching anything else.*

Each obstacle above has an obvious "fix" available in the reward or the
bounds. Taking them all would produce a walking fly and destroy the
result. The distinction that matters:

**Legitimate — correcting our own measurement errors.**
* `energy_ref` (§9) was calibrated against a motionless baseline and was
  wrong by 325×, making the reward forbid walking. Fixing it removes an
  error; it does not encode an answer. Mandatory.
* The `dng100_level` bound (§8b) was mis-scaled so one sigma of sampling
  landed past a measured cliff. Recalibrating a search bound to the
  system's measured dynamics is instrument calibration.

**Borderline — shaping that closes a measured exploit.**
* The locomotion gate (§8) stops rhythm being paid for when the body does
  not move. Defensible, but it *is* designed-in structure and must be
  declared as such.

**Not legitimate — designing the gait and calling it a finding.**
* Forcing the adhesion duty toward the CPG's 65 %, prescribing per-leg
  phase offsets, adding an anti-asymmetry penalty, or rewarding tripod
  index directly. **Every one of these would hand the animal the answer we
  claim to be testing for.** If we must specify the duty cycle, the phase
  relationships and the symmetry, then we have built the walker and the
  connectome is decoration.

The project's own pre-registration already anticipated this: coordination,
if it has to be supplied, is **Rung 2** — a bounded interleg conductor that
is *explicitly labelled as our engineering addition*, with a coupling
strength that starts at zero and a K=0 ablation. That is the honest place
to put it. Hiding the same intervention inside the reward function would
disguise engineering as biology.

**Therefore: no further reward changes are proposed.** The measured
result — the decoded output does not animate six legs and produces no
stance/swing cycling, so this interface cannot generate traction — is a
legitimate finding and consistent with the literature this project is
built on (Pugliese et al.: DNg100 drive yields no consistent left/right
phase coupling; T3 least robust).

---

## 12. The silent legs are in the PUBLISHED model, not in our pipeline

*2026-09-25. The decisive check on whether Rung 1's failure mode is our
artifact: apply our exact per-leg motor analysis to **Pugliese's own
published 128-replicate full-VNC output** (Zenodo 22260924, downloaded in
Phase 0), rather than to our reimplementation.*

Same motor mapping (`build_motor_neuron_groups` on `all_legs_circuit.csv`),
same readout (per-leg summed motor rate), same stability filter
(peak `n_active` ≤ 1500 → 118 of 128 replicates stable).

| | all 128 | stable (118) |
|---|---:|---:|
| replicates with 6/6 legs showing **any** activity | 56.2% | 52.5% |
| replicates with 6/6 legs **substantially rhythmic** (std > 0.5) | 16.4% | **9.3%** |

**Only ~9% of the authors' own stable replicates animate all six legs.**
The mode is 5/6; 3/6 or fewer is common. Our own runs showed 2 of 6
eligible replicates at 6/6 (33%), well within sampling noise at n = 6.

**Conclusion: our pipeline is not diverging. It faithfully reproduces the
published model's behaviour.** The silent legs that block Rung 2 are a
property of the published rate model under DNg100 tonic drive, not of our
decoder, our reimplementation, or our training.

Note the level at which this sits: replicates share **identical wiring**
and differ only in per-neuron parameter draws
(`neuron_params[replicate]`, `trial_setup.py`). So this is a statement
about the **rate model plus its parameter sampling**, not about the
connectome's synaptic data.

### Per-leg breakdown, and a claim that needs checking

Across the 118 stable published replicates:

| leg | % any activity | % std > 0.5 | median std |
|---|---:|---:|---:|
| T1-LHS | 86.4% | 70.3% | 0.818 |
| **T1-RHS** | 71.2% | **22.0%** | **0.224** |
| T2-LHS | 78.0% | 59.3% | 0.661 |
| T2-RHS | 92.4% | 83.9% | 2.213 |
| T3-LHS | 78.8% | 60.2% | 0.839 |
| **T3-RHS** | 96.6% | **96.6%** | **6.329** |

By segment: **T1 46.2%** (median std 0.440), T2 71.6% (1.428),
**T3 78.4%** (2.639).

Two observations, both flagged rather than asserted:

1. **Measured this way, T3 (hind) is the most robust segment and T1
   (front) the least** — the opposite ordering to the "hind legs somewhat
   less robust than front legs" that `CLAUDE.md` records from the paper's
   Fig. 4. **This is not yet a contradiction.** Our metric is the standard
   deviation of summed *motor-neuron pool* rate (an amplitude measure);
   the paper's claim may rest on CPG-neuron activity, on rhythm quality
   rather than amplitude, or on a different readout entirely. **Before
   this is repeated anywhere, it must be checked against the actual v2
   text and Fig. 4** — per this project's standing rule that a claim about
   a paper is verified against primary text, never against our own notes.
2. **Strong left/right asymmetry** (T1-LHS 70.3% vs T1-RHS 22.0%; T3-RHS
   96.6% vs T3-LHS 60.2%) in a run where DNg100 is understood to be
   activated bilaterally. Also worth verifying against the run config
   before being interpreted.

### What this settles

* **Rung 2 is not viable as designed**, and the reason is upstream of us:
  the conductor needs a readable phase on each leg, and the published
  model supplies six readable legs in ~9% of stable replicates.
* **Rung 1's negative result is confirmed against primary data.** The
  interface did not fail because of a decoding mistake; it was given a
  motor output that does not animate a hexapod.
