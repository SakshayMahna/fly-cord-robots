# Electrical coupling: the missing mechanism

*Started 2026-09-26. This is the first intervention in the project that
made the six leg CPGs behave as one system.*

## Why this mechanism, in four measurements

1. **Coordination in real flies is central, not sensory.** Air-stepping
   flies — no ground contact at all — show the *highest* tripod
   coordination, and silencing proprioceptors preserves left-right coupling
   (bioRxiv 2026.04.29.721658). So the body is not the missing ingredient,
   and our Path A (biomechanical coupling) was aimed at the wrong thing.
2. **The published simulation of those same central circuits does not
   coordinate.** Pugliese et al. report phase coupling absent across the six
   leg CPGs, and that including the VNC neurons linking left and right CPGs
   disynaptically "was insufficient to couple the phase."
3. **The legs are not mis-phased — they run at different frequencies.**
   Median spread 3.03 Hz across published replicates (`GROUND_BRIDGE.md`
   §13), against 0.00 Hz for the CPG controller that walks. Phase coupling
   is impossible without frequency locking first.
4. **That spread is structural, not a parameter artifact.** Setting every
   neuron's tau, gain and threshold spread to ~zero leaves 1.5–2.8 Hz. The
   six leg circuits have genuinely different natural frequencies, set by
   their own wiring.

And the decisive fact: **electrical synapses are 10–20 nm, below the
resolution of the EM used to build every published Drosophila connectome.**
They are absent from MANC, MaleCNS, FANC and BANC alike. The wiring we have
is a map of *chemical* synapses only. Gap junctions are the canonical
mechanism for synchronising biological oscillators and are directly
implicated in insect motor coordination (*Nature* 2023, "Gap junctions
desynchronize a neural circuit to stabilize insect flight").

So this is **restoring a known biological mechanism the dataset physically
cannot contain**, not inventing one to get a result. That distinction is the
whole justification, and it is the sentence that has to survive scrutiny.

## What was added, precisely

`neural/gap_junctions.py`. A **separate symmetric conductance matrix**,
never `W_eff`:

    dr_i/dt = (activation_i − r_i)/tau_i  +  Σ_j g_ij (r_j − r_i)

Applied to the **derivative**, not to the synaptic input: a gap junction
conducts directly between cells and does not pass through the synaptic
threshold nonlinearity. The term is symmetric, proportional to the state
difference, and vanishes once cells synchronise — the defining properties
of electrical coupling.

**Caveat we own:** a real gap junction couples membrane *potential*; this
model's state variable is firing *rate*. The diffusive form is the standard
rate-model approximation of electrical coupling, not a literal simulation.

**Guarantees, asserted in `tests/test_gap_junctions.py` (6 tests):** the
connectome's weight matrix is unchanged while coupling is active;
conductance 0 reproduces the uncoupled model **bit for bit** (the ablation
the claim rests on); nonzero conductance provably changes the dynamics; the
matrix is symmetric with zero diagonal; ipsilateral coupling never crosses
the midline.

## Result: frequency locking, achieved

Replicate 0, per-leg dominant frequency as conductance rises:

| g | per-leg Hz | spread |
|---:|---|---:|
| 0 | 11.40, 11.96, 11.44, 11.96, 9.38, 11.96 | 2.58 Hz |
| 1 | 11.98, 12.00, 11.96, 11.98, **8.56**, 11.98 | 3.44 Hz |
| 10 | 12.19 ×5, **6.08** | 6.10 Hz |
| **20** | **11.958 × 6** | **0.00 Hz** |
| 50 | 11.56 × 6 | 0.02 Hz |

**All six legs lock to a single frequency, 6/6 rhythmic, peak `n_active`
536 — no saturation.** The route there is textbook coupled-oscillator
behaviour: the weak leg (T3-LHS) first entrains at a **subharmonic** (~6 Hz,
half the group frequency) before pulling into 1:1 as coupling rises.

**Which cells need coupling:** the excitatory hub alone is sufficient
(spread 0.02 at g=20, 6/6 rhythmic) — six neurons. Both excitatory roles
together is the most robust (spread 0.00 across g=20–100). **Inhibitory
coupling alone does not lock** (spread 5.6–5.9 at every conductance).

## What is NOT solved: phase

Every configuration locks **in phase** — all six legs together, a pronk,
not a tripod. Tripod index stays ≈ −0.08 to −0.24 regardless of
conductance, which cell type is coupled, or whether coupling is all-to-all
or ipsilateral-only.

This is expected: diffusive coupling minimises differences, so in-phase is
its stable state. Antiphase between tripod groups requires something else —
inhibition between groups, or delays.

**Deliberately not done:** coupling legs according to their tripod group, or
adding a phase-offset term that specifies the gait. Either would hand the
model the pattern we are claiming to discover (`GROUND_BRIDGE.md` §11). The
ipsilateral variant was written specifically to avoid this: "same side
couples electrically" is an anatomical assumption; "these three legs step
together" would be the answer.

## Where this leaves the walking question

With the untrained (default) adapter the body still does not move — dx
between −0.49 and +0.18 mm, essentially unchanged. That is expected and is
a *separate*, already-diagnosed problem: the default decoder's joint
excursion is ~1/11th of a walking gait (§7a), which only training fixes.
Coupling did improve leg recruitment on replicate 4 (4 → 6 rhythmic).

So for the first time every identified blocker has been addressed:

| blocker | status |
|---|---|
| reward forbade walking | fixed (§9) |
| drive bound put 1σ past the saturation cliff | fixed (§8b) |
| legs at different frequencies | **fixed — this document** |
| joint excursion 11× too small | training's job, never yet attempted with the above fixed |

The next run is the first legitimate test of the walking question.
