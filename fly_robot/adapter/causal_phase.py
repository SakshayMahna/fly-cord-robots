"""Causal, online per-leg phase estimate for the Rung 2 conductor.

Proposal only (`docs/trained_adapter/RUNG2_DESIGN.md`) — validated in
`fly_robot/experiments/validate_causal_phase.py` against the offline
method before it is trusted by anything, per your instruction.

**Method: trailing-window bandpass + Hilbert, not a novel quadrature
filter.** The existing offline method (`analysis/interleg_coordination.py:
leg_rhythm`) bandpass-filters and Hilbert-transforms the *whole*
post-transient trace, which needs the future and cannot run online. This
uses the identical bandpass + Hilbert transform on a trailing window of
the last `WINDOW_S` seconds only, recomputed every `STRIDE_STEPS` steps —
strictly causal (the window never includes the current or a future
sample), and directly comparable to the offline method since it is the
same transform, not a different approximation. That comparability is the
point: it makes the validation in item 3 a fair like-for-like check, not
an apples-to-oranges one.

Cost: `WINDOW_S / (STRIDE_STEPS * dt)` bandpass+Hilbert calls per second
of trial, each over a short window rather than a whole trial — measured in
`validate_causal_phase.py`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import hilbert

from fly_robot.analysis.interleg_coordination import _bandpass

# Measured dominant rhythm frequency is ~11 Hz (AUDIT.md §4); a window
# needs several cycles for the bandpass filter's edge effects to settle
# before the phase at its rightmost (most recent) sample is trustworthy.
# 300 ms is ~3.3 cycles at 11 Hz -- short enough to be genuinely online,
# long enough to filter meaningfully. Not yet validated; that is exactly
# what validate_causal_phase.py checks, including whether this specific
# choice is adequate.
WINDOW_S = 0.300
# Recomputing every step is unnecessary (phase changes slowly relative to
# neural dt) and costly at 4000 steps/trial; every 10 ms (10 neural steps
# at dt=0.001) resolves the ~11 Hz rhythm's phase to within ~4% of a cycle.
STRIDE_STEPS = 10


class CausalPhaseEstimator:
    """Per-leg trailing-window phase, recomputed every `stride` steps.

    Call `update(leg_signals)` once per neural step with the current
    per-leg readout (e.g. summed CPG-triad or motor rate, shape (6,)).
    `phase` holds the most recent estimate (radians, updated only on
    stride steps; held constant between updates -- a zero-order hold,
    the same convention this project already uses for sensory drive).
    """

    def __init__(self, dt: float, window_s: float = WINDOW_S,
                 stride_steps: int = STRIDE_STEPS, n_legs: int = 6):
        self.dt = dt
        self.stride = stride_steps
        self.maxlen = int(round(window_s / dt))
        self.buffers = [deque(maxlen=self.maxlen) for _ in range(n_legs)]
        self.phase = np.zeros(n_legs)
        self._step = 0

    def update(self, leg_signals: np.ndarray) -> np.ndarray:
        for i, v in enumerate(leg_signals):
            self.buffers[i].append(float(v))
        self._step += 1
        if self._step % self.stride != 0 or len(self.buffers[0]) < self.maxlen:
            return self.phase
        fs = 1.0 / self.dt
        for i, buf in enumerate(self.buffers):
            x = np.asarray(buf, dtype=np.float64)
            x = x - x.mean()
            xb = _bandpass(x[None, :], fs)
            # Rightmost sample = most recent = "now", the only causal choice.
            self.phase[i] = float(np.angle(hilbert(xb, axis=1))[0, -1])
        return self.phase
