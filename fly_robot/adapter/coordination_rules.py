"""Cruse-style interleg coordination rules — OUR ADDITION, and a real one.

Read `docs/trained_adapter/GROUND_BRIDGE.md` §11 before changing anything
here: this module sits on the wrong side of a line this project has been
careful about, and it is only acceptable because it is declared.

Why it exists
-------------
Four mechanisms failed to coordinate the connectome's six leg CPGs, each
for a measured reason (GROUND_BRIDGE §14, GAP_JUNCTIONS): raising stride
amplitude flips the fly; electrical coupling locks the legs *in phase*, the
one gait that cannot support a body; a scale-adaptive adhesion gate makes
legs cycle but at independent phases; and trainable phase coupling cannot
shift CPG phase at all below the strength that destroys the rhythm.
Anatomical conduction delay was then ruled out on physics: the longest
inter-CPG distance in MANC is 432 um, antiphase at 12 Hz needs 41.7 ms, and
that would require a conduction velocity of 0.0104 m/s — ten times slower
than the slowest plausible insect axon.

So the timing has to come from us. This is the least-bad way to supply it.

What these rules are, and why they are not a stored gait
--------------------------------------------------------
Cruse's ipsilateral coordination rules, from the stick insect literature:

  R1  an ongoing SWING in a leg INHIBITS swing onset in its next ROSTRAL
      neighbour                                    ("not while I'm up")
  R2  the onset of STANCE FACILITATES swing onset in the next ROSTRAL
      neighbour                                    ("your turn, I'm down")
  R3  posterior movement during STANCE increasingly FACILITATES swing
      onset in the next CAUDAL neighbour           ("I'm nearly done")

Every rule is local, between ADJACENT legs, and phrased as a constraint on
*when a leg may lift* — never as a phase to hold. Tripod is not stored
anywhere in this file. It emerges because "alternate with all your
neighbours" has exactly one consistent solution on two rows of three.

**That is also why it survives amputation.** Remove the middle legs and
front/hind become each other's neighbours; the same rules then have
diagonal alternation as their consistent solution, i.e. a trot. The
adjacency graph changes; the rules do not.

Where this sits relative to the connectome — stated plainly
-----------------------------------------------------------
In Walknet each leg's own controller IS the oscillator, so the rules
modulate it directly. Here the oscillator is the connectome, and we
measured that its phase cannot be shifted by injected current (unmoved
below K≈5, rhythm destroyed by K=10). So the rules cannot act on the CPGs.
They act DOWNSTREAM instead, by delaying each leg's decoded output.

The division of labour is therefore:

    connectome : the shape of each leg's step, its intrinsic frequency,
                 and which motor neurons fire when within the cycle
    these rules: WHEN each leg takes its turn, relative to its neighbours

That is a real concession and the honest caption for any result using this
module is "the fly's wiring shapes each leg's step; our rules decide when
the legs take turns." It is not "the connectome coordinated the legs."
"""

from __future__ import annotations

import numpy as np

# FlyGym leg order: left-front, left-mid, left-hind, right-front, ...
LEG_ORDER = ("lf", "lm", "lh", "rf", "rm", "rh")

# Adjacency, derived from body layout alone — NOT from any gait.
# Rostral = toward the head. Each leg's rostral and caudal ipsilateral
# neighbour, and its contralateral partner. Absent entries are edges of the
# body. Removing a leg from LEG_ORDER and rebuilding this map is all that a
# quadruped needs; nothing else in this file changes.
def build_adjacency(legs: tuple = LEG_ORDER) -> dict:
    """{leg: {'rostral': leg|None, 'caudal': leg|None, 'contra': leg|None}}"""
    order = {"f": 0, "m": 1, "h": 2}          # front, middle, hind
    by_side = {"l": [], "r": []}
    for leg in legs:
        by_side[leg[0]].append(leg)
    for side in by_side:
        by_side[side].sort(key=lambda l: order[l[1]])
    adj = {}
    for side, chain in by_side.items():
        other = "r" if side == "l" else "l"
        for i, leg in enumerate(chain):
            contra = next((l for l in by_side[other] if l[1] == leg[1]), None)
            adj[leg] = {
                "rostral": chain[i - 1] if i > 0 else None,
                "caudal": chain[i + 1] if i + 1 < len(chain) else None,
                "contra": contra,
            }
    return adj


class CoordinationRules:
    """Applies R1-R3 (plus a contralateral variant of R1) to produce a
    per-leg output delay, in neural steps.

    Each leg has an intrinsic phase supplied by the connectome. When that
    phase reaches swing onset the leg *wants* to lift; the rules may hold
    it, which is implemented as an increase in that leg's playback delay.
    A held leg's decoded trajectory is simply replayed later — the
    connectome's waveform is never altered, only when it is delivered.
    """

    def __init__(self, legs: tuple = LEG_ORDER, dt: float = 0.001,
                 hold_rate_s: float = 0.25, release_rate_s: float = 0.5,
                 max_delay_s: float = 0.12, swing_fraction: float = 0.4,
                 contra_strength: float = 1.0):
        self.legs = tuple(legs)
        self.adj = build_adjacency(self.legs)
        self.idx = {l: i for i, l in enumerate(self.legs)}
        self.dt = dt
        # How fast a held leg accumulates delay, and sheds it when released.
        self.hold_step = dt / hold_rate_s
        self.release_step = dt / release_rate_s
        self.max_delay = max_delay_s
        self.swing_fraction = swing_fraction
        self.contra_strength = contra_strength
        self.delay_s = np.zeros(len(self.legs))
        self.prev_stance = np.zeros(len(self.legs), dtype=bool)
        self.just_touched = np.zeros(len(self.legs), dtype=bool)

    def swing_state(self, phase: np.ndarray) -> np.ndarray:
        """True where a leg is in its swing portion of the cycle.

        Phase in (-pi, pi]; swing occupies the leading `swing_fraction`.
        Which part of the cycle counts as swing is our modelling choice,
        the same choice the adhesion gate's polarity already makes.
        """
        frac = (np.asarray(phase) + np.pi) / (2 * np.pi)
        return frac < self.swing_fraction

    def step(self, phase: np.ndarray, freq_hz: float = 12.0) -> np.ndarray:
        """One update. `phase` is per-leg INTRINSIC phase (from connectome
        motor output). Returns per-leg delay in SECONDS.

        The rules observe the EFFECTIVE phase — intrinsic phase shifted back
        by that leg's current delay — because the delayed output is what the
        body and the neighbours actually see. Observing intrinsic phase here
        would leave the loop open: delays would accumulate while the states
        the rules read never changed, and nothing would ever coordinate.
        """
        phase = np.asarray(phase, dtype=float)
        phase = np.angle(np.exp(1j * (phase - 2 * np.pi * freq_hz * self.delay_s)))
        swing = self.swing_state(phase)
        stance = ~swing
        self.just_touched = stance & ~self.prev_stance      # stance onset
        # Progress through stance, 0 at touchdown -> 1 at lift-off. Used by
        # R3 as "how far through my push-back am I".
        frac = (phase + np.pi) / (2 * np.pi)
        stance_progress = np.clip(
            (frac - self.swing_fraction) / max(1e-9, 1 - self.swing_fraction),
            0.0, 1.0)

        hold = np.zeros(len(self.legs))
        for leg in self.legs:
            i = self.idx[leg]
            a = self.adj[leg]
            inhibit = 0.0
            facilitate = 0.0
            # R1: my ROSTRAL neighbour swinging inhibits my swing onset.
            r = a["rostral"]
            if r is not None and swing[self.idx[r]]:
                inhibit += 1.0
            # R1-contra: the same, across the midline. Cruse's contralateral
            # influence is weaker than the ipsilateral one; `contra_strength`
            # carries that and can be set to 0 to test ipsilateral-only.
            cl = a["contra"]
            if cl is not None and swing[self.idx[cl]]:
                inhibit += self.contra_strength
            # R2: my ROSTRAL neighbour just touched down -> go.
            if r is not None and self.just_touched[self.idx[r]]:
                facilitate += 1.0
            # R3: my CAUDAL neighbour late in stance -> increasingly go.
            c = a["caudal"]
            if c is not None and stance[self.idx[c]]:
                facilitate += stance_progress[self.idx[c]]
            # A leg is only held at the moment it is trying to lift.
            wants_to_lift = swing[i]
            hold[i] = 1.0 if (wants_to_lift and inhibit > facilitate) else 0.0

        self.delay_s = np.clip(
            self.delay_s + np.where(hold > 0, self.hold_step, -self.release_step),
            0.0, self.max_delay)
        self.prev_stance = stance
        return self.delay_s.copy()

    def report(self) -> dict:
        return {"legs": list(self.legs),
                "adjacency": self.adj,
                "delay_s": self.delay_s.tolist()}
