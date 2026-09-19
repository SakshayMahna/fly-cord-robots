# Sensory map: which robot signal drives which neurons, and why

*Step 2 of the closed-loop work. Every neuron count here was queried from
the data; every encoding claim is either quoted from a primary source or
explicitly labelled as our own design choice. Written 2026-09-19.*

---

## 1. Dataset decision: stay on MANC (checked, not assumed)

MaleCNS was checked against MANC before building anything, because MaleCNS
is this project's nominal dataset and a better sensory annotation would
have been worth switching for. Reproduce with:

```
python -m fly_robot.connectome.compare_sensory_annotation
```

The two datasets annotate sensory neurons **differently**, so the
comparison had to discover each scheme rather than share one query:

| | MANC (what the simulation runs on) | MaleCNS (`male-cns:v1.0`) |
|---|---|---|
| class | `sensory neuron` / `sensory ascending` | `mechanosensory_proprioceptive` |
| subclass | `chordotonal organ`, `campaniform sensilla`, `hair plate` | same three, **plus** a generic `leg` |
| leg assignment | parsed from `instance` (`SNpp45_MetaLN_L`) | explicit **`entryNerve`** field |
| side | parsed from `instance` | explicit `rootSide` |

`somaNeuromere` is unusable for this in MANC — it is NaN for 5,889 of
5,891 sensory neurons, because sensory somata sit out in the leg, not in
the VNC.

### MaleCNS's annotation is genuinely better — and still unusable

| leg | MANC L / R | asym | MaleCNS L / R | asym | MaleCNS **reachable** L / R | asym |
|---|---|---:|---|---:|---|---:|
| T1 | 27 / 66 | 0.419 | 45 / 20 | 0.385 | 20 / 12 | 0.250 |
| T2 | 68 / 112 | 0.244 | 139 / 144 | **0.018** | 41 / 54 | 0.137 |
| T3 | 91 / 117 | 0.125 | 142 / 155 | **0.044** | 58 / 66 | 0.065 |

*(asymmetry = |L−R| / (L+R); 0 is perfect)*

MaleCNS finds **645** leg proprioceptors to MANC's **481**, and for T2 and
T3 it is dramatically more symmetric (0.018 and 0.044, versus 0.244 and
0.125). That is a real improvement.

**But the dynamics run on the MANC matrix**, so a MaleCNS neuron is only
usable if its `mancBodyid` cross-reference lands on a row that exists in
the simulated network. It usually does not:

- only **40.6%** (262/645) have a `mancBodyid` at all;
- only **248 unique rows** are reachable in the simulated network.

So adopting MaleCNS's sensory annotation would **roughly halve** the usable
sensory population, from 481 to 248. It buys better symmetry at the cost of
half the neurons — and the T1 asymmetry, the worst case, survives the
switch anyway (0.385 vs 0.419), *in the opposite direction* (MANC is
right-biased 27:66, MaleCNS is left-biased 45:20).

**Recommendation: do not switch.** The only way to actually get MaleCNS's
better annotation is to run the dynamics on MaleCNS-derived weights — the
long-standing open gap recorded in `AUDIT.md` §1, a much larger piece of
work that would also break direct comparability with Pugliese's published
baseline. Left/right asymmetry is instead handled at the interface, by
normalisation (§4), with the un-normalised mapping kept as a control.

### Load sensing is absent in both datasets

| | leg-nerve campaniform sensilla |
|---|---:|
| MANC | **9** |
| MaleCNS | **12** |

Both datasets put the hundreds of campaniform sensilla they *do* contain in
the wing and haltere nerves (ADMN 217/218, DMetaN 337/195), not the legs.
There is no version of this project in which leg load sensing is well
covered by the available annotation. This is why the ground-contact
hypothesis is framed around joint-angle feedback rather than load feedback
(see `PREREGISTRATION.md`, H2).

---

## 2. Neurons used

MANC, leg nerves only (ProLN→T1 front, MesoLN→T2 middle, MetaLN→T3 hind),
subclass `chordotonal organ` and `hair plate`. Campaniform excluded — 9
neurons across six legs cannot support a channel.

| leg | side | chordotonal | hair plate | total |
|---|---|---:|---:|---:|
| T1 | L | 23 | 4 | 27 |
| T1 | R | 57 | 8 | 65 |
| T2 | L | 54 | 12 | 66 |
| T2 | R | 90 | 20 | 110 |
| T3 | L | 80 | 9 | 89 |
| T3 | R | 90 | 25 | 115 |
| | | **394** | **78** | **472** |

Verified in `AUDIT.md` §7: every one of these has outgoing synapses in the
simulated matrix (median out-degree 22–44), and each leg's group reaches
its own leg's CPG and motor neurons within two hops. Injecting current into
them will propagate.

---

## 3. What each class encodes

### Chordotonal organs → femur–tibia joint angle and movement

The femoral chordotonal organ (FeCO) is the large leg chordotonal organ and
monitors the femur–tibia joint. Mamiya, Gurung & Tuthill, *Neuron* 100(3)
636–650 (2018), "Neural Coding of Leg Proprioception in Drosophila",
abstract, verbatim:

> "Using subclass-specific genetic driver lines, we show that one group of
> axons encodes **tibia position (flexion/extension)**, another encodes
> **movement direction**, and a third encodes **bidirectional movement and
> vibration frequency**."

> "…how proprioceptive stimuli from **a single leg joint** are encoded by a
> diverse population of sensory neurons"

**Constraint we cannot work around:** neither dataset says which neuron is
which subtype. MANC/MaleCNS types are opaque systematic identifiers
(`SNpp39`–`SNpp60`), `flywireType` is empty, and there is no field naming
the organ or the joint. We therefore **cannot** build a claw/hook/club
split, and we cannot assign different chordotonal neurons to different
joints. Any such split would be invented.

**Our design choice (labelled as such):** drive each leg's entire
chordotonal pool with a single combined position-and-movement signal from
that leg's femur–tibia joint — a deliberately coarse stand-in for a
population that really is heterogeneous.

### Hair plates → thorax–coxa joint, limit detection

Hair plates sit at the thorax–coxa joint and fire near the joint's
*extremes*, not across its range. From the primary text of "Proprioceptive
limit detectors mediate sensorimotor control of the *Drosophila* leg"
(bioRxiv 2025.05.15.654260), verbatim:

> "On the fly's front leg, three hair plates are located at the junction
> between the coxa (Cx) and thorax… These hair plates (CxHP8, CxHP3, CxHP4)
> wrap the joint along the anterior-posterior axis."

> "Figure 1. **CxHP8 neurons encode the anterior limits of thorax-coxa
> joint angles.**"

> "a hair plate on the fly coxa (CxHP8) **detects the limits of anterior leg
> movement**."

**Our design choice:** a half-wave rectified function of the thorax–coxa
angle that is silent through most of the range and rises only in the outer
portion of it. The rectification is grounded in the quotes above; the
specific threshold is ours.

### Campaniform sensilla → not used

9 neurons in MANC leg nerves (§1). Recorded as a limitation, not modelled.

---

## 4. Encoders

Identical in form across all six legs — no per-leg or per-condition
parameters, per the project's hard rules. All configuration lives in one
place so a run is reproducible from its config.

Inputs come from the physics body, for leg *i*:

- `θ_FTi` — femur–tibia joint angle (`{leg}_trochanterfemur-{leg}_tibia-pitch`)
- `θ_TC` — thorax–coxa joint angle (`c_thorax-{leg}_coxa-pitch`)
- `θ̇_FTi` — femur–tibia angular velocity

Both encoders output a **unitless drive in [0, 1]**, scaled by the single
global feedback gain `g_fb`:

```
chordotonal:   u_ch = clip( w_pos · p + w_vel · v , 0, 1 )
                 p = (θ_FTi − θ_min) / (θ_max − θ_min)      position, normalised
                 v = clip( |θ̇_FTi| / v_ref , 0, 1 )          movement, unsigned
                 w_pos = w_vel = 0.5

hair plate:    u_hp = clip( (θ_TC − θ_lim) / (θ_max − θ_lim) , 0, 1 )
                 θ_lim = θ_min + 0.75·(θ_max − θ_min)        silent below 75% of range
```

`p` and `v` are mixed because the real population encodes both and we
cannot separate the subtypes (§3). `w_pos = w_vel = 0.5` is a choice, not a
measurement.

### Per-neuron current, and the side normalisation

```
I_sensory[neuron in group g] = g_fb · u(g) · s(g)
```

where the group `g` is a (segment, side, subclass) pool and `s(g)` is the
normalisation factor:

```
s(g) = N_ref(segment, subclass) / N(g)        [normalised — default]
s(g) = 1                                       [as-annotated — control C4]
```

`N_ref(segment, subclass)` is the mean of that segment's left and right
counts. This makes **total** drive into the left and right pools of a
segment equal, while leaving genuine front/middle/hind differences intact.

Worked example, T1 chordotonal (L = 23, R = 57, mean = 40):
`s(L) = 40/23 = 1.74`, `s(R) = 40/57 = 0.70`. At `g_fb = 10` and `u = 1`,
per-neuron current is 17.4 on the left and 7.0 on the right — both above the
median sensory threshold of 3.10, neither saturating — and total drive is
400 on each side.

**Why normalise at all:** the primary hypothesis is about left↔right
coupling, and the raw annotation gives one side up to 2.4× the sensory
input of the other. Un-normalised feedback would build an asymmetry into
the very quantity being measured. The normalisation touches **only the
interface** — no connectome weight, sign, or topology is altered — and
condition **C4** runs the un-normalised, as-annotated mapping so the effect
of this choice is measured rather than assumed.

### Latency

One configurable delay applied to all sensory drive, identical across legs.
Default **0 ms**, so the baseline has no unmodelled dynamics; non-zero
values are available for a robustness check and are not part of the main
sweep.

---

## 5. Locality — enforced, not asserted

Sensors on robot leg *i* may drive only sensory neurons annotated to fly
leg *i*. There are no cross-leg paths in our code. This is enforced by an
automated test that perturbs each leg's sensors alone and asserts that only
that leg's sensory neurons see changed input; it fails the build otherwise.

The **only** exception is control condition **C3**, which permutes sensory
channels across legs deliberately, to test whether leg-specific feedback
matters. It is a labelled violation, used as a control and nowhere else.

Note that the *connectome* contains whatever cross-leg pathways it
contains, and those are free to act — the rule constrains our interface,
not the fly's wiring.

---

## 6. Limitations to carry into RESULTS

1. **No leg load sensing.** 9 leg campaniform sensilla in MANC (12 in
   MaleCNS). Ground-contact forces cannot be fed back through their real
   channel.
2. **No chordotonal subtype resolution.** Position, direction and vibration
   subtypes exist in the fly and are documented, but are not separable in
   the data, so one combined signal drives the whole pool.
3. **One joint per class.** Chordotonal → femur–tibia, hair plate →
   thorax–coxa. The real leg has proprioceptors at joints we do not drive.
4. **Left/right asymmetry is corrected, not absent.** Normalisation
   equalises total drive; it cannot create neurons that were never
   annotated. C4 measures what this correction does.
5. **Encoder form is ours.** Linear position, rectified limit detection,
   the 0.5/0.5 mix, the 75% threshold — all design choices consistent with
   the cited encoding properties, none of them measured fly transfer
   functions.
6. **MANC, not MaleCNS** (§1), inheriting `AUDIT.md` §1's open gap.
