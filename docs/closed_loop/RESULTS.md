# Results — closed-loop pilot

*2026-09-20. **Pilot only.** Every number here comes from pilot seeds
(parameter seed 641, replicates 0–3). No main-experiment seed (20260919)
has ever been run. The hypotheses in `PREREGISTRATION.md` §1 are therefore
**not tested** — this document reports what the pilot established about
whether they are testable at all.*

---

## Headline

The pre-registered decision rule (`PREREGISTRATION.md` §B2) was applied
verbatim and **neither encoder variant is usable**. Zero sweep levels out
of seven met both criteria, for either variant.

By the rule committed before those runs, that makes the finding below the
reported result.

> **Across all four sensory encoder formulations, proprioceptive feedback
> into the unmodified VNC connectome is either negligible or destroys the
> rhythm. There is no intermediate regime in which feedback measurably
> influences the network while leaving it rhythmic.**

---

## 1. What was tried

Four encoders, each a strictly more careful attempt than the last:

| encoder | position code | rest drive | outcome |
|---|---|---:|---|
| `signed` | signed, 0.5 at reference | 88.1% of all current | cliff at g ≈ 3.25 |
| `deviation` | rectified deviation from settled rest | 0.0 | cliff at g ≈ 4.25 |
| `deviation_capped` | + per-neuron current cap 2.5 | 0.0 | cliff at g ≈ 4.25 |
| `deviation_fractionated` | + range-fractionated tuning | 101.7 | cliff at g ≈ 3.75 |

Each fixed a real, identified defect in its predecessor — an 88% DC bias,
then unbounded per-neuron peaks, then all-neurons-respond-at-once — and
each shifted the cliff by at most one gain unit without removing it.

## 2. The decision rule, applied

Sweep {0, 1, 2, 2.5, 3, 3.5, 4, 4.5}; 3 usable replicates (replicate 2
excluded by the pre-registered stability filter, being oversaturated at
baseline with no feedback at all).

**`deviation_capped`**

| g_fb | median rhythmic legs | rhythm OK | median effect | effect > 0.05? |
|---:|---:|---|---:|---|
| 1.0 | 4 | yes | 0.00000 | no |
| 2.0 | 4 | yes | 0.00017 | no |
| 2.5 | 4 | yes | 0.00043 | no |
| 3.0 | 4 | yes | 0.00068 | no |
| 3.5 | 4 | yes | 0.00081 | no |
| 4.0 | 3 | yes | 0.00102 | no |
| 4.5 | 1 | **no** | 1.01537 | yes |

**`deviation_fractionated`**

| g_fb | median rhythmic legs | rhythm OK | median effect | effect > 0.05? |
|---:|---:|---|---:|---|
| 1.0 | 4 | yes | 0.00000 | no |
| 2.0 | 4 | yes | 0.00009 | no |
| 2.5 | 4 | yes | 0.00018 | no |
| 3.0 | 4 | yes | 0.00024 | no |
| 3.5 | 4 | yes | 0.00025 | no |
| 4.0 | 2 | **no** | 1.01303 | yes |
| 4.5 | 1 | **no** | 1.01213 | yes |

**Levels meeting both criteria: 0 of 7, for both variants** (threshold: 3).

The structure is stark. Where the rhythm survives, the effect size is
**0.0000–0.0010** — feedback changes the neural trajectory by about a
tenth of one percent. Where the effect is large it is **≈ 1.01**, i.e.
total decorrelation, and the rhythm is gone. Nothing lies between.

## 3. Why — the mechanism

Not a tuning failure. Measured directly:

**The sensory neurons essentially never fire.** Across the whole stable
regime, the fraction of the 472 leg proprioceptors whose input exceeds
their own firing threshold is **0.0% at g_fb = 1 rising only to 1.1% at
g_fb = 4.5**. Their median threshold is 3.06, while typical per-neuron
sensory current is ~0.1 and peaks at 2.4 — roughly **thirty times below
threshold on average, and still below it at the peak**.

So below the cliff, feedback is injected and simply fails to recruit
anybody. The 0.1% trajectory change is the sub-threshold current nudging
membrane dynamics without producing spikes.

The cliff is reached when transient excursions — one leg swinging far
enough that its drive saturates — push current past threshold. Because
the thresholds are narrowly distributed, crossing is near-simultaneous across
much of the pool, and 472 neurons with out-degree 22–44 firing together
overwhelms the network. `n_active` jumps roughly tenfold, to 3,600–4,900
against Pugliese's own oversaturation criterion of 1,500.

**This is a property of the modelled circuit, not of our integrator.**
Zero of 68 + 36 trials showed numerical instability; peak firing rate was
235.6 Hz against a 1000 Hz clip; and the coordination metric was
separately shown to be dt-robust (`AUDIT.md` §6).

## 4. The inhibition corollary

Encoder-side inhibition was rejected on principle (`PREREGISTRATION.md`
§B0): the sign of every sensory neuron's output is given by the
connectome, and the connectome obeys Dale's law exactly — **0 of 22,769
presynaptic neurons carry both signs**. Adding inhibition at the interface
would override data with a modelling choice.

The consequence must be stated plainly, because it is the strongest form
of this result:

> **The runaway happens in the presence of the network's own inhibitory
> circuitry.** 636,490 of the connectome's 1,372,404 edges — **46%** — are
> inhibitory, and they are all active and unmodified throughout. Real
> inhibition, as wired in the fly, does not prevent it.

## 5. What this does and does not mean

**Does:** within this model — Pugliese et al.'s firing-rate dynamics, the
MANC connectome, our motor and sensory interfaces, a NeuroMechFly body on
a ball — there is no feedback gain at which proprioceptive feedback both
matters and leaves the rhythm intact, across four encoder formulations.

**Does not:**

- It is **not** a claim about the fly. Real proprioceptive feedback plainly
  works in real flies.
- It is **not** a test of H1/H2/H3. Those remain untested; the pilot
  establishes that they cannot be tested with this interface.
- It is **not** a claim that no encoder could work. Four were tried and a
  fifth was ruled out in advance by the pre-registration. One specific
  observation is worth recording for anyone continuing: the mismatch is
  between the *distribution* of sensory drive over time — near zero most
  of the time, occasionally saturating — and a narrow threshold
  distribution. An encoder whose *typical* rather than peak current sat
  near threshold might recruit gradually. That is a hypothesis for future
  work, not something this pilot tested.

## 6. Limitations carried forward

Everything in `PREREGISTRATION.md` §9 still applies, in particular:

- Dynamics run on **MANC**, not MaleCNS — the project's nominal dataset
  has still never been simulated.
- **No leg load sensing**: 9 leg campaniform sensilla in MANC, 12 in
  MaleCNS. Ground contact cannot be fed back through its real channel, and
  `signed` left the hair-plate channel identically silent as well.
- Encoder form is **ours**, consistent with published encoding properties
  but not a measured fly transfer function.
- Rate model, not spiking; one representative body; pilot-scale n (3–4
  usable replicates per condition).

## 7. Reproducing

```bash
MUJOCO_GL=cgl python -m fly_robot.experiments.closed_loop_pilot \
    --replicates 0 1 2 3 --encoders signed deviation
MUJOCO_GL=cgl python -m fly_robot.experiments.closed_loop_pilot \
    --replicates 0 1 2 3 --encoders deviation_capped deviation_fractionated \
    --out-dir media/closed_loop/pilot_variants
```

Raw per-trial output: `media/closed_loop/pilot*/pilot_trials.csv`.

## 8. Video reference clips

Four short (4 s) rendered clips of specific real pilot states — for
illustration/video use, not additional data:

```bash
MUJOCO_GL=cgl python -m fly_robot.experiments.render_closed_loop_clips
```

Writes `media/closed_loop/clips/{name}_{side,opposite_side,top_down}.mp4`:

| clip | condition | n_active | what it shows |
|---|---|---:|---|
| `no_drive` | no DNg100 stimulation at all | 0 | true non-movement — nothing is driving the animal |
| `stable` | deviation encoder, g_fb=3.5 (pilot replicate 1) | 388 | normal connectome-driven leg motion; feedback active but measurably not doing anything (§3) |
| `transition` | deviation encoder, g_fb=4.0 (pilot replicate 0) | 4040 | one sweep step past this replicate's cliff (was 543 at g_fb=3.5) — already the runaway state, since the "transition" between stable and unstable is itself the discontinuity §2 describes, not a gradual ramp |
| `seizure` | deviation encoder, g_fb=4.5 (pilot replicate 0) | 4629 | fully saturated, rhythm gone |

`no_drive`/`stable` and `transition`/`seizure` deliberately use different
pilot replicates (1 and 0) — replicate 1 stays stable through the entire
sweep including g_fb=4.5, so it cannot show the cliff at all. An earlier
version of this script rendered all four clips from replicate 1 and
`transition`/`seizure` silently came out identical to `stable`; caught by
checking `n_active` against the pilot's own recorded sweep table
(`docs/closed_loop/LOG.md`) rather than assuming the render matched the
label.
