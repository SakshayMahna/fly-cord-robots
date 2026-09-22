"""Causal, online per-leg phase estimate for the Rung 2 conductor.

**Second attempt.** The first (trailing-window `filtfilt` + Hilbert,
recomputed from a fresh block every 10 ms) failed validation: mean
circular correlation 0.301 against the offline whole-trace method,
minimum -0.251 (anti-correlated), lag incoherent and sign-flipping across
legs. Diagnosed as likely `filtfilt`'s edge handling landing on the exact
sample being read ("now"), compounded by block-boundary discontinuities
every time the window was recomputed from scratch. See
`docs/trained_adapter/RUNG2_DESIGN.md` §8 for the full first-attempt
record — kept there rather than deleted.

**This implementation: a single-pole complex resonator**, the standard
causal construction for narrowband phase tracking of a roughly-known
oscillation frequency (used in phase-locked-loop demodulation). No
window, no block recomputation, no `filtfilt` — pure streaming IIR state
updated every step:

    z[n] = r * exp(j*w0) * z[n-1] + (1-r) * x[n]
    phase[n] = angle(z[n])

where `w0 = 2*pi*f0*dt` (center frequency) and `r = exp(-dt/tau)` (pole
radius, from a settling time constant `tau`). This is mathematically a
real signal frequency-shifted down by `f0`, low-pass filtered, then
shifted back — a complex bandpass concentrated near `f0`. Because the
SAME filter (same `r`, same `f0`) runs on every leg, a fixed group delay
is common to all six, which is exactly what this project's coupling needs
(only relative phase between legs matters for the Kuramoto-style
correction, `RUNG2_DESIGN.md` §3) — unlike the offline method's
deliberate zero-phase `filtfilt`, whose own docstring explains it
specifically to avoid rotating relative phase for a WHOLE-signal, non-
causal analysis; that reasoning does not carry over to an online filter,
which necessarily has some delay.

Validated in `validate_causal_phase.py` against the offline method before
anything downstream uses it.
"""

from __future__ import annotations

import numpy as np

# Measured dominant rhythm frequency, mean of six legs' `dominant_hz` on
# one clean untrained replicate at the matched drive (AUDIT.md's ~11 Hz
# prediction, confirmed directly: 11.4-12.0 Hz range).
CENTER_HZ = 11.7

# Settling time constant. Swept {50, 100, 150, 300} ms on one replicate:
# correlation rises and lag grows together as tau increases; 100 ms gave
# correlations 0.836-0.993 across all six legs with lag -60 to -160 ms,
# a good balance validated more fully in validate_causal_phase.py rather
# than picked by this single-replicate sweep alone.
TAU_S = 0.100


class CausalPhaseEstimator:
    """Per-leg complex-resonator phase, one IIR pole per leg, updated
    every neural step (no windowing, no stride, no block recomputation).

    Call `update(leg_signals)` once per neural step with the current
    per-leg readout (shape (6,)); `phase` holds the running estimate.
    """

    def __init__(self, dt: float, center_hz: float = CENTER_HZ,
                 tau_s: float = TAU_S, n_legs: int = 6):
        self.dt = dt
        w0 = 2 * np.pi * center_hz * dt
        self.r = float(np.exp(-dt / tau_s))
        self.pole = self.r * np.exp(1j * w0)
        self.z = np.zeros(n_legs, dtype=complex)
        self.phase = np.zeros(n_legs)

    def update(self, leg_signals: np.ndarray) -> np.ndarray:
        x = np.asarray(leg_signals, dtype=np.float64)
        self.z = self.pole * self.z + (1.0 - self.r) * x
        self.phase = np.angle(self.z)
        return self.phase

    def reset(self):
        self.z[:] = 0.0
        self.phase[:] = 0.0
