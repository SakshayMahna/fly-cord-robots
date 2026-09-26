"""Interleg phase coupling with LEARNED offsets — OUR ADDITION.

Why this exists
---------------
The connectome does not phase-couple its six leg CPGs. They run at
*different frequencies* (median spread 3.03 Hz across published
replicates), which Pugliese et al. also report and which survives removing
every per-neuron parameter heterogeneity. Three separate attempts to get a
gait out of that failed at the same point:

  * raising stride amplitude flips the fly (legs uncoordinated);
  * diffusive electrical coupling locks them **in phase** — all six legs
    lift together, the one gait that cannot support a body;
  * fixing the adhesion gate makes each leg cycle, but at *independent*
    phases, so at random moments all six release.

Stable walking needs three legs down in a stable triangle at all times.
That is a statement about interleg phase.

What is and is not assumed
--------------------------
**No gait is written here.** Each leg has a free phase offset, every one
initialised to 0 (i.e. in phase), and the coupling strength starts at
exactly 0 — so the default adapter is the uncoupled model and any phase
pattern has to be *discovered* by the search against the reward.

That distinction is the whole point. The CPG robotics literature obtains
tripod by designing a phase-difference matrix (FlyGym's own generator is
literally `make_tripod_cpg_network`); writing one here would make "the
connectome coordinated the legs" unfalsifiable. Training the offsets and
then **reporting what they converged to** is a result instead: if they land
on tripod, that is evidence about what this body and this reward select
for, not an assumption we supplied. If they land on something else, that is
also a result, and an interesting one.

Mechanism
---------
Kuramoto mean-field coupling with per-oscillator offsets:

    current_i = K * sin(Psi + phi_i - theta_i)

where `theta_i` is leg i's phase, read from the connectome's own motor
output by the validated causal resonator estimator
(`adapter/causal_phase.py`, mean circular correlation 0.902 against the
offline method), and `Psi` is the mean phase over legs that actually have a
rhythm. The current is injected into that leg's CPG triad neurons
(`adapter/cpg_groups.py`).

**No body sensor is involved** — phase comes from connectome output only —
so the no-bypass rule still holds, exactly as it did for the adhesion gate.

Legs with no measurable rhythm are excluded from both the mean field and
the injection: there is no phase to read, and driving them on a meaningless
estimate would be noise, not coupling.
"""

from __future__ import annotations

import numpy as np

# Below this the coupling is treated as exactly off, so that `z = 0` is the
# uncoupled model bit for bit. The logistic squash cannot return a bound
# exactly (it returns ~2e-16 for a default of 0.0), and an ablation that is
# "almost" zero is not an ablation.
OFF_EPS = 1e-9


class InterlegCoupling:
    """Stateful: holds the phase estimator, so construct once per trial and
    step in lockstep with the neural model."""

    def __init__(self, params, cpg_rows: dict, leg_order: tuple,
                 dt: float = 0.001, min_amplitude: float = 1e-6):
        self.strength = float(np.asarray(params.coupling_strength).ravel()[0])
        self.offsets = np.asarray(params.coupling_phase, dtype=float).ravel()
        self.enabled = abs(self.strength) > OFF_EPS
        self.leg_order = leg_order
        self.min_amplitude = float(min_amplitude)
        # Row indices per leg, in `leg_order`; legs absent from the circuit
        # map contribute nothing.
        self.rows = [np.asarray(cpg_rows.get(leg, []), dtype=int)
                     for leg in leg_order]
        self.estimator = None
        if self.enabled:
            from fly_robot.adapter.causal_phase import CausalPhaseEstimator
            self.estimator = CausalPhaseEstimator(dt=dt, n_legs=len(leg_order))
        self.last_phase = np.zeros(len(leg_order))
        self.last_current = np.zeros(len(leg_order))

    def step(self, leg_rate: np.ndarray, n_neurons: int):
        """Returns (n_neurons,) coupling current for this step.

        `leg_rate` is the per-leg summed motor rate, in `leg_order`.
        """
        if not self.enabled:
            return None
        phase = self.estimator.update(leg_rate)
        self.last_phase = phase.copy()

        # Only legs with a real oscillation take part. `z` is the
        # estimator's complex state; its magnitude is the rhythm's amplitude.
        amp = np.abs(self.estimator.z)
        active = amp > self.min_amplitude
        if active.sum() < 2:
            self.last_current[:] = 0.0
            return None

        # Mean field over participating legs, with each leg's offset removed
        # so that Psi is the common phase the offsets are measured against.
        z_mean = np.mean(np.exp(1j * (phase[active] - self.offsets[active])))
        psi = float(np.angle(z_mean))

        current = np.zeros(n_neurons)
        self.last_current[:] = 0.0
        for i, rows in enumerate(self.rows):
            if not active[i] or rows.size == 0:
                continue
            drive = self.strength * np.sin(psi + self.offsets[i] - phase[i])
            current[rows] = drive
            self.last_current[i] = drive
        return current

    def report(self) -> dict:
        return {"enabled": bool(self.enabled),
                "strength": self.strength,
                "offsets_rad": self.offsets.tolist(),
                "offsets_deg": np.degrees(self.offsets).tolist()}
