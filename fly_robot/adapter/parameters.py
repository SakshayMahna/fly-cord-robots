"""The trainable adapter — 71 parameters, and nothing else.

**EVERYTHING IN THIS MODULE IS OUR OWN ADDITION**, per the project's
honesty rule. The connectome supplies neurons, weights, signs and
topology; none of that appears here and none of it is trainable. What is
trainable is the interface on either side of it:

    body sensors -> [sensory encoder] -> fly sensory neurons
                                              |
                                        (the connectome,
                                         never touched)
                                              |
    joint targets <- [motor decoder] <- fly motor neurons

plus the level of drive injected into command neurons that really exist in
the simulated network (DNg100, MDN — row indices verified by query, see
`docs/trained_adapter/DESIGN.md` §2.2).

Parameters are shared **per thoracic segment (T1/T2/T3), not per leg**,
with a single global left/right scalar. This is deliberate and it is the
main thing keeping the result interesting: six independent per-leg
parameter sets would let the optimiser hand-build a gait leg by leg, which
is exactly the outcome that would make "the connectome coordinated the
legs" unfalsifiable.

Search space
------------
CMA-ES searches an unbounded, roughly unit-scale vector `z` in R^71. Each
entry maps to its physical range through a logistic squash, so:

  * bounds are respected without clipping (clipping creates flat regions
    the covariance cannot read, and the search stalls against them);
  * one `sigma0` is meaningful for every parameter, even though a sensory
    gain and a joint offset are not remotely in the same units.

`z = 0` is the DEFAULT configuration, not the midpoint of each range —
each parameter carries an offset chosen so the search STARTS at a
known-sane point (mostly Phase 4's frozen constants) rather than in the
middle of an arbitrary interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

SEGMENTS = ("T1", "T2", "T3")
# Matches ANTAGONIST_PAIRS order in interface/motor_neuron_to_joint.py.
PAIRS = ("coxa", "femur", "tibia")

# name -> (shape, low, high, default). Order defines the vector layout and
# MUST NOT be reordered — checkpoints and the config hash depend on it.
PARAM_SPEC: tuple[tuple[str, tuple[int, ...], float, float, float], ...] = (
    # --- sensory encoder: chordotonal, per segment ------------------------
    ("chord_gain",      (3,), 0.0,   6.0,   1.0),    # Phase 4 cliff ~4.25; start well below
    ("chord_offset",    (3,), -2.0,  2.0,   0.0),    # the knob RESULTS.md §5 suggests
    ("chord_cap",       (3,), 0.5,   6.0,   2.5),    # Phase 4's measured SENSORY_CURRENT_CAP
    ("chord_mix",       (3,), 0.0,   1.0,   0.5),    # position weight; velocity = 1 - this
    # --- sensory encoder: hair plate, per segment -------------------------
    ("hp_gain",         (3,), 0.0,   6.0,   1.0),
    ("hp_offset",       (3,), -2.0,  2.0,   0.0),
    ("hp_cap",          (3,), 0.5,   6.0,   2.5),
    ("hp_threshold",    (3,), 0.30,  0.95,  0.75),   # Phase 4's HAIR_PLATE_THRESHOLD_FRAC
    # --- chordotonal range fractionation, per segment ---------------------
    ("chord_sigma",     (3,), 0.02,  0.50,  0.10),   # Phase 4's FRACTIONATION_SIGMA
    ("chord_band",      (3,), -0.5,  0.5,   0.0),    # shifts the preferred-band centres
    # --- global sensory ----------------------------------------------------
    ("lr_sensory_ratio", (),  0.5,   2.0,   1.0),
    # --- command drive -----------------------------------------------------
    # Default is the REAL connectome's pre-registered activity-matched
    # drive (PREREGISTRATION.md, RESULTS.md: matched_drive=382.8125,
    # media/trained_adapter/drive_matching.json), not Pugliese's own 380 --
    # the pre-registration commits to training "relative to that start."
    # Regression-tested against the recorded JSON so the two cannot drift
    # apart silently (tests/test_adapter.py).
    ("dng100_level",    (),   0.0,   600.0, 382.8125),
    ("mdn_level",       (),   0.0,   400.0, 5.0),    # ~off; MDN is exploratory only
    ("command_asymmetry", (), -0.5,  0.5,   0.0),    # L/R drive imbalance -> turning
    ("command_ramp_s",  (),   0.001, 0.50,  0.02),   # matches Phase 4's pulse_start
    # --- motor decoder: per (pair, segment) --------------------------------
    ("motor_gain",      (3, 3), 0.05, 1.20, 0.50),   # DEFAULT_GAIN_RAD
    ("motor_scale",     (3, 3), 0.50, 20.0, 3.00),   # DEFAULT_RATE_SCALE_HZ
    ("motor_offset",    (3, 3), -0.30, 0.30, 0.0),
    ("motor_tau",       (3,),  0.0005, 0.05, 0.001),  # ~= no filtering at neural dt
    # --- adhesion gating, per segment --------------------------------------
    # Gate: w * (per-leg summed motor rate - its 50 ms running mean) > theta.
    #
    # `w` may go NEGATIVE so the optimiser can find the polarity itself --
    # which half of the rhythm is stance is our modelling choice, and one we
    # would rather not make by assertion. Initialised positive.
    #
    # Threshold bounds span the measured deviation scale: |rate - running
    # mean| has p50 0.37, p95 6.41, p99 9.05, max 18.39 Hz across four
    # untrained replicates, so +/-9 covers the usable range without the
    # gate saturating at either bound.
    ("adhesion_weight",    (3,), -2.0, 2.0, 1.0),
    ("adhesion_threshold", (3,), -9.0, 9.0, 0.0),
)

N_PARAMS = sum(int(np.prod(shape)) if shape else 1 for _n, shape, *_ in PARAM_SPEC)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _slices():
    """(name, shape, low, high, default, slice) for each block."""
    out, i = [], 0
    for name, shape, low, high, default in PARAM_SPEC:
        n = int(np.prod(shape)) if shape else 1
        out.append((name, shape, low, high, default, slice(i, i + n)))
        i += n
    return out


_SLICES = _slices()

# Search stages deliberately expose whole interface blocks, never arbitrary
# individual entries.  This is both a compute-saving device and an honesty
# constraint: R1a asks whether the connectome's *output* can walk the body;
# it cannot silently solve that question through the sensory encoder.
PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "sensory": (
        "chord_gain", "chord_offset", "chord_cap", "chord_mix",
        "hp_gain", "hp_offset", "hp_cap", "hp_threshold",
        "chord_sigma", "chord_band", "lr_sensory_ratio",
    ),
    "command": ("dng100_level", "mdn_level", "command_asymmetry",
                "command_ramp_s"),
    "motor": ("motor_gain", "motor_scale", "motor_offset", "motor_tau"),
    "adhesion": ("adhesion_weight", "adhesion_threshold"),
}

# `output` is the ground-walking bridge's first rung. `sensory` is R1b: it
# takes an R1a solution as a warm start and asks whether feedback improves it.
TRAINING_STAGES: dict[str, tuple[str, ...]] = {
    "output": ("command", "motor", "adhesion"),
    "sensory": ("sensory",),
    "full": tuple(PARAMETER_GROUPS),
}


def parameter_names_for_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Names in stable vector order for whole named parameter blocks."""
    requested = tuple(groups)
    unknown = set(requested) - set(PARAMETER_GROUPS)
    if unknown:
        raise ValueError(f"unknown parameter group(s): {sorted(unknown)}")
    selected = {name for group in requested for name in PARAMETER_GROUPS[group]}
    return tuple(name for name, *_rest in PARAM_SPEC if name in selected)


def parameter_indices_for_groups(groups: Iterable[str]) -> np.ndarray:
    """Indices in the full 71-D CMA coordinate vector for ``groups``."""
    names = set(parameter_names_for_groups(groups))
    return np.concatenate([np.arange(sl.start, sl.stop)
                           for name, _shape, _low, _high, _default, sl in _SLICES
                           if name in names]).astype(int)


def parameter_indices_for_stage(stage: str) -> np.ndarray:
    """The trainable coordinates for a named, documented training stage."""
    try:
        groups = TRAINING_STAGES[stage]
    except KeyError as exc:
        raise ValueError(
            f"unknown training stage {stage!r}; choose from "
            f"{sorted(TRAINING_STAGES)}") from exc
    return parameter_indices_for_groups(groups)

# z-offset per block so that z = 0 yields the DEFAULT physical value.
_Z0 = np.zeros(N_PARAMS)
for _name, _shape, _low, _high, _default, _sl in _SLICES:
    _frac = (_default - _low) / (_high - _low)
    _Z0[_sl] = _logit(min(max(_frac, 1e-6), 1 - 1e-6))


@dataclass(frozen=True)
class AdapterParams:
    """Physical-space adapter parameters. Build via `from_z`, not directly."""
    values: dict = field(default_factory=dict)

    def __getattr__(self, item):
        try:
            return self.values[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @property
    def segment_index(self) -> dict:
        return {seg: i for i, seg in enumerate(SEGMENTS)}

    def as_flat_dict(self) -> dict:
        """JSON-safe, for hashing and checkpoints."""
        out = {}
        for name, _shape, *_rest in PARAM_SPEC:
            v = self.values[name]
            out[name] = float(v) if np.isscalar(v) or np.ndim(v) == 0 else np.asarray(v).tolist()
        return out


def from_z(z: np.ndarray) -> AdapterParams:
    """Map an unbounded search vector to physical parameters."""
    z = np.asarray(z, dtype=float).ravel()
    if z.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got {z.shape}")
    squashed = 1.0 / (1.0 + np.exp(-(z + _Z0)))
    values = {}
    for name, shape, low, high, _default, sl in _SLICES:
        phys = low + (high - low) * squashed[sl]
        values[name] = float(phys[0]) if not shape else phys.reshape(shape)
    return AdapterParams(values)


def to_z(params: AdapterParams) -> np.ndarray:
    """Inverse of `from_z`: physical parameters back to search coordinates.

    Needed to warm-start a search from a physically-specified adapter (e.g.
    a calibrated decoder amplitude) rather than from the defaults. Exact
    round-trip with `from_z` up to floating point, asserted in
    tests/test_adapter.py.

    Values are clipped a hair inside their bounds before the logit, because
    a parameter sitting exactly on a bound maps to +/-inf in z — which CMA-ES
    cannot start from. `from_z(to_z(p))` therefore returns a value
    fractionally inside the bound rather than exactly on it.
    """
    z = np.zeros(N_PARAMS)
    eps = 1e-9
    for name, _shape, low, high, _default, sl in _SLICES:
        phys = np.asarray(params.values[name], dtype=float).ravel()
        frac = np.clip((phys - low) / (high - low), eps, 1.0 - eps)
        z[sl] = np.log(frac / (1.0 - frac)) - _Z0[sl]
    return z


def default_z() -> np.ndarray:
    """The search's starting point: zeros, which decode to the defaults."""
    return np.zeros(N_PARAMS)


def default_params() -> AdapterParams:
    return from_z(default_z())
