"""Measure whether the six legs are rhythmically coordinated with each other.

This is the primary outcome measure for the closed-loop experiments. It is
deliberately written to work identically on two very different kinds of
signal, so neural and mechanical coordination are compared on equal terms:

  * a NEURAL readout — per-leg summed motor-neuron firing rate, and
  * a KINEMATIC readout — a per-leg joint angle from the physics body.

Pugliese et al. report (Fig. 4d-e, Extended Data Fig. 9b) that "phase
coupling was absent across the six leg CPGs in the full connectome
simulation". Replicating that absence in our own open-loop baseline is a
precondition for trusting anything we measure after closing the loop — if
coupling shows up open-loop, our pipeline differs from theirs and the
whole experiment is invalid (see the project's Phase 4 stop conditions).

Why a surrogate null rather than a fixed PLV threshold: the analysis
window is short (~1.5 s) and the rhythms are slow (~1-6 Hz), so only a
handful of cycles are observed. Two *independent* oscillators of similar
frequency will show a high phase-locking value over so few cycles purely
by chance. Phase-randomised surrogates preserve each signal's amplitude
spectrum — and therefore its frequency content and cycle count — while
destroying any true phase relationship, which makes them the right null
for exactly this confound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# Pugliese/MANC leg naming, in a fixed order used by every array here.
LEGS = [(seg, side) for seg in ("T1", "T2", "T3") for side in ("LHS", "RHS")]
LEG_LABELS = [f"{seg}-{side}" for seg, side in LEGS]

# The two tripod groups of the canonical insect alternating-tripod gait:
# front-left + middle-right + hind-left step together, against
# front-right + middle-left + hind-right. Within a group the expected
# phase difference is 0; between groups it is pi.
TRIPOD_A = {("T1", "LHS"), ("T2", "RHS"), ("T3", "LHS")}

# Rhythm band, set from the data rather than assumed.
#
# CAUTION, because we got this wrong once: two different quantities in
# this project are both measured in Hz. A neuron's FIRING RATE (how hard
# it fires) and the OSCILLATION FREQUENCY of that rate (how fast it rises
# and falls) are not the same thing. The "0.9-6.5 Hz" figure in
# docs/logs/2026-09-19.md is a firing RATE range, and an earlier version
# of this constant mistakenly used it as a frequency band — which capped
# the band below the real rhythm and made every reported "dominant
# frequency" meaningless.
#
# Measured directly over all 581 active leg-trials in Pugliese's 128
# published full-VNC replicates (per-leg summed motor-neuron rate,
# post-transient window, searched over a deliberately over-wide
# 0.5-60 Hz so the band itself could not bias the answer): median
# 10.7 Hz, 5th-95th percentile 4.7-13.3 Hz, 97.6% of leg-trials between
# 2 and 20 Hz. Independently consistent with our own Phase 0 numbers
# from Pugliese's own `compute_oscillation_score` (5.7-11.4 Hz per
# segment, media/simulation/full_vnc_oscillation_scores.json), and with
# real Drosophila stepping frequencies (~5-15 Hz).
#
# 2-20 Hz therefore covers the real rhythm with margin on both sides,
# while excluding DC/envelope drift below and integration noise above.
BAND_HZ = (2.0, 20.0)

# Discarded from the start of every trial before analysis: the stimulus
# turns on at t=0.02 s and the network needs time to settle into whatever
# rhythm it is going to have. Pre-registered, not tuned per condition.
TRANSIENT_S = 0.5

# A leg must exceed this peak rate (Hz, summed over its motor neurons) in
# the analysis window to have a meaningful phase at all.
MIN_PEAK_ACTIVITY = 0.05


@dataclass
class LegRhythm:
    """Per-leg rhythmicity for one trial."""
    active: np.ndarray        # (6,) bool — had enough activity to phase-analyse
    dominant_hz: np.ndarray   # (6,) float — spectral peak frequency in BAND_HZ
    peak_strength: np.ndarray # (6,) float — peak power / total in-band power
    phase: np.ndarray         # (6, n_t) float — instantaneous phase (radians)


@dataclass
class Coordination:
    """Pairwise coordination for one trial. Pairs are in the fixed order
    produced by `pair_indices()` — 15 unordered pairs of 6 legs."""
    pairs: list[tuple[int, int]]
    plv: np.ndarray            # (15,) phase-locking value
    mean_phase_diff: np.ndarray  # (15,) circular mean of phase difference
    plv_null_p95: np.ndarray   # (15,) 95th percentile of the surrogate null
    significant: np.ndarray    # (15,) bool — plv exceeds its own null
    valid: np.ndarray          # (15,) bool — both legs were active
    tripod_index: float        # agreement with the alternating-tripod pattern


def pair_indices() -> list[tuple[int, int]]:
    return [(i, j) for i in range(len(LEGS)) for j in range(i + 1, len(LEGS))]


def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. filtfilt is used specifically
    because it introduces NO phase shift — a causal filter would rotate
    every leg's phase and corrupt the quantity we are measuring."""
    nyq = fs / 2
    low, high = BAND_HZ[0] / nyq, min(BAND_HZ[1] / nyq, 0.99)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, x, axis=-1)


def leg_rhythm(signals: np.ndarray, dt: float,
               transient_s: float = TRANSIENT_S,
               min_peak: float = MIN_PEAK_ACTIVITY) -> LegRhythm:
    """signals: (6, n_timesteps) per-leg readout (neural or kinematic).
    Returns per-leg rhythmicity and instantaneous phase over the
    post-transient window."""
    fs = 1.0 / dt
    start = int(round(transient_s / dt))
    x = np.asarray(signals, dtype=np.float64)[:, start:]

    peak = x.max(axis=1) - x.min(axis=1)
    active = peak > min_peak

    xd = x - x.mean(axis=1, keepdims=True)
    xb = _bandpass(xd, fs)

    # Spectral peak inside the band
    freqs = np.fft.rfftfreq(xb.shape[1], dt)
    power = np.abs(np.fft.rfft(xb, axis=1)) ** 2
    in_band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    band_power = power[:, in_band]
    band_freqs = freqs[in_band]
    peak_bin = band_power.argmax(axis=1)
    dominant = band_freqs[peak_bin]
    total = band_power.sum(axis=1)
    strength = np.where(total > 0, band_power.max(axis=1) / np.maximum(total, 1e-30), 0.0)

    phase = np.angle(hilbert(xb, axis=1))
    return LegRhythm(active=active, dominant_hz=dominant,
                     peak_strength=strength, phase=phase)


def _plv(phase_i: np.ndarray, phase_j: np.ndarray) -> complex:
    return np.mean(np.exp(1j * (phase_i - phase_j)))


def _phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the SAME amplitude spectrum but randomised phases —
    preserves frequency content and autocorrelation, destroys any true
    inter-signal phase relationship."""
    spec = np.fft.rfft(x)
    random_phase = rng.uniform(0, 2 * np.pi, spec.shape)
    random_phase[0] = 0.0
    if x.shape[-1] % 2 == 0:
        random_phase[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * random_phase), n=x.shape[-1])


def coordination(signals: np.ndarray, dt: float, n_surrogates: int = 200,
                 seed: int = 0, transient_s: float = TRANSIENT_S) -> tuple[LegRhythm, Coordination]:
    """Full coordination analysis for one trial.

    signals: (6, n_timesteps), legs in `LEGS` order.
    """
    rhythm = leg_rhythm(signals, dt, transient_s=transient_s)
    pairs = pair_indices()
    rng = np.random.default_rng(seed)

    start = int(round(transient_s / dt))
    x = np.asarray(signals, dtype=np.float64)[:, start:]
    xb = _bandpass(x - x.mean(axis=1, keepdims=True), 1.0 / dt)

    # Null: surrogate phases, same pipeline
    null_plv = np.zeros((n_surrogates, len(pairs)))
    for s in range(n_surrogates):
        sur_phase = np.angle(hilbert(
            np.stack([_phase_randomise(xb[i], rng) for i in range(len(LEGS))]), axis=1))
        for p, (i, j) in enumerate(pairs):
            null_plv[s, p] = np.abs(_plv(sur_phase[i], sur_phase[j]))

    plv = np.zeros(len(pairs))
    mean_diff = np.zeros(len(pairs))
    valid = np.zeros(len(pairs), dtype=bool)
    for p, (i, j) in enumerate(pairs):
        z = _plv(rhythm.phase[i], rhythm.phase[j])
        plv[p] = np.abs(z)
        mean_diff[p] = np.angle(z)
        valid[p] = rhythm.active[i] and rhythm.active[j]

    null_p95 = np.percentile(null_plv, 95, axis=0)
    significant = valid & (plv > null_p95)

    # Tripod index: +1 if every pair sits exactly at its expected tripod
    # phase, 0 if phases are unrelated to the tripod pattern. Computed as
    # the mean cosine of (observed - expected) phase difference over valid
    # pairs, weighted by each pair's PLV so unlocked pairs don't count as
    # evidence for the pattern.
    cosines, weights = [], []
    for p, (i, j) in enumerate(pairs):
        if not valid[p]:
            continue
        same_group = (LEGS[i] in TRIPOD_A) == (LEGS[j] in TRIPOD_A)
        expected = 0.0 if same_group else np.pi
        cosines.append(np.cos(mean_diff[p] - expected))
        weights.append(plv[p])
    tripod = float(np.average(cosines, weights=weights)) if cosines and np.sum(weights) > 0 else np.nan

    return rhythm, Coordination(pairs=pairs, plv=plv, mean_phase_diff=mean_diff,
                                 plv_null_p95=null_p95, significant=significant,
                                 valid=valid, tripod_index=tripod)


def leg_motor_readout(R: np.ndarray, motor_indices: dict) -> np.ndarray:
    """Pre-declared NEURAL readout: per leg, the SUM of firing rates over
    that leg's motor neurons — i.e. total motor drive to the leg.

    Sum rather than mean because only a handful of each leg's 48-74 motor
    neurons are recruited in any replicate (Pugliese's Fig. 4b: "2-10 leg
    MNs"); a mean over the full pool mostly measures how many neurons are
    silent, which is not what we want the phase of.

    R: (n_neurons, n_timesteps). motor_indices: {(seg, side): [row, ...]}.
    """
    return np.stack([
        R[motor_indices[leg]].sum(axis=0) if motor_indices.get(leg) else np.zeros(R.shape[1])
        for leg in LEGS
    ])
