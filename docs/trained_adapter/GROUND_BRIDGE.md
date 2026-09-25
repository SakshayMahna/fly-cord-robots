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
