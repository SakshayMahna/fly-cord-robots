"""The trainable adapter — 65 parameters, and nothing else.

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
CMA-ES searches an unbounded, roughly unit-scale vector `z` in R^65. Each
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
    ("dng100_level",    (),   0.0,   600.0, 380.0),  # Pugliese's own verified value
    ("mdn_level",       (),   0.0,   400.0, 5.0),    # ~off; MDN is exploratory only
    ("command_asymmetry", (), -0.5,  0.5,   0.0),    # L/R drive imbalance -> turning
    ("command_ramp_s",  (),   0.001, 0.50,  0.02),   # matches Phase 4's pulse_start
    # --- motor decoder: per (pair, segment) --------------------------------
    ("motor_gain",      (3, 3), 0.05, 1.20, 0.50),   # DEFAULT_GAIN_RAD
    ("motor_scale",     (3, 3), 0.50, 20.0, 3.00),   # DEFAULT_RATE_SCALE_HZ
    ("motor_offset",    (3, 3), -0.30, 0.30, 0.0),
    ("motor_tau",       (3,),  0.0005, 0.05, 0.001),  # ~= no filtering at neural dt
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


def default_z() -> np.ndarray:
    """The search's starting point: zeros, which decode to the defaults."""
    return np.zeros(N_PARAMS)


def default_params() -> AdapterParams:
    return from_z(default_z())
