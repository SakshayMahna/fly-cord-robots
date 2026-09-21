# Results — trained adapter

*No training run has happened. This file records findings established
while building and validating the Stage A rig, so they are not lost in the
log.*

## The anatomically named motor pools are near-silent

Measured on the untrained connectome under DNg100 drive, across 6 legs ×
6 replicates:

| motor module | mean rate | fraction of leg-pools active |
|---|---:|---:|
| tibia extend | 0.198 | 0.25 |
| femur/tr extend | 0.196 | 0.17 |
| coxa stance | 0.191 | 0.19 |
| femur/tr flex | 0.068 | 0.36 |
| femur reductor | 0.051 | 0.08 |
| coxa swing | 0.025 | 0.19 |
| tibia flex | 0.001 | 0.03 |
| **substrate grip** | **0.000** | **0.00** |
| **tarsus control** | **0.000** | **0.00** |

The two modules semantically closest to adhesion — `substrate grip`
(47 neurons, 6–9 per leg) and `tarsus control` (15) — **never fire at
all**. The pair the annotation names `coxa stance` / `coxa swing` fires in
roughly one leg-pool in five.

This sharpens Phase 3's recorded limitation that 3 of 9 motor modules are
unmapped: two of those three are not merely unmapped by us, they are
**silent in the model**. Nothing downstream could have used them.

**Consequence for the adhesion interface:** the gate uses the connectome's
**rhythm** (per-leg summed motor rate, high-passed against its own 50 ms
running mean) and **not its stance anatomy**, because the anatomical signal
does not exist to be used. A gate built on `coxa stance` − `coxa swing`
sticks: three legs always-off, one always-on, 0–1 transitions per 2 s where
stepping needs ~44.

**Open question for training to answer.** All four anatomical pools are
logged every step and every generation. Whether a trained adapter ever
recruits them — particularly `substrate grip`, which is what a real fly
would use to hold the substrate — is a stronger and more interesting result
than the adhesion gate itself, and it will be reported either way.

## Baselines on the adopted rig

| controller | mean speed | sd | seeds | min upright |
|---|---:|---:|---:|---:|
| CPG (FlyGym demo) | 14.025 mm/s | 0.250 | 5 | 0.966 |
| rule-based (FlyGym demo) | 7.396 mm/s | 0.392 | 3 | 0.873 |
| **untrained connectome** | **+0.011 mm total** | — | 1 | 0.901 |

The untrained connectome moves its legs energetically — 160 rad of total
joint travel over 4 s — and travels 0.011 mm. It stays upright and does not
flip. That is the honest starting point training has to improve on.
