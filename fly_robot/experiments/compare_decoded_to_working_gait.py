"""Is the connectome's decoded joint motion in the same RANGE as a gait
that actually walks?

A diagnostic, run before spending a long CMA-ES budget. The restricted-action
gate (`baselines/restricted_action_cpg.py`) established that a working gait
exists inside the adapter's 18-DOF action space (9.8 mm/s, 70% of
unrestricted). So for those same 18 DOFs we can ask a much sharper question
than "does it walk": **how far is the untrained decoder's output from a
trajectory known to work?**

Three numbers per DOF, for the connectome path and the working CPG path:

  * peak-to-peak excursion (rad) — is the amplitude even comparable?
  * dominant frequency (Hz) — is it stepping at a plausible rate?
  * per-leg phase spread — is there any interleg structure at all?

Why this is worth doing first: CMA-ES searches `motor_gain` / `motor_scale`
/ `motor_offset` around their defaults with sigma0=0.5 in z-space. If the
decoded excursion is (say) 30x too small, the useful region may be far from
where the search starts, and 20 hours would be spent crawling toward it.
That is a fixable starting-point problem, not a fact about the connectome —
and it is much cheaper to find here.

**This does not tune anything and never enters a reported result.** The CPG
trajectory is used only as a measuring stick (`GROUND_BRIDGE.md` §5: a
conventional controller may calibrate the interface, but must be absent at
evaluation time and named outright).

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.compare_decoded_to_working_gait
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from flygym import Simulation
from flygym.compose.fly.base_fly import ActuatorType
from flygym_demo.complex_terrain.common import apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import (
    CPGController, make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

from fly_robot.adapter.parameters import default_params
from fly_robot.baselines.restricted_action_cpg import adapter_controllable_dof_names
from fly_robot.bodies.neuromechfly import build_free_fly
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

MATCHED_DRIVE = 382.8125     # PREREGISTRATION.md: the real network's matched drive
DURATION_S = 2.0


def dominant_hz(x: np.ndarray, dt: float) -> float:
    """Spectral peak of a single DOF trace, in the 2-20 Hz band the project's
    rhythm analysis already uses (`interleg_coordination.BAND_HZ`)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    freqs = np.fft.rfftfreq(len(x), dt)
    power = np.abs(np.fft.rfft(x)) ** 2
    band = (freqs >= 2.0) & (freqs <= 20.0)
    if not band.any():
        return 0.0
    return float(freqs[band][np.argmax(power[band])])


def connectome_traces(replicate: int = 0) -> tuple[np.ndarray, list[str]]:
    """Joint-angle traces from the untrained adapter on free ground, R1a
    condition (sensory path off), at the pre-registered matched drive."""
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
        stim_current=MATCHED_DRIVE)
    res = run_trial(model, mg, sensory_groups=sg, rig="ground",
                    duration_s=DURATION_S, seed=replicate,
                    adapter=default_params(), adapter_sensory=False)
    n = res.terminated_at_step or res.n_steps_planned
    return np.asarray(res.joint_angles[:n], dtype=float), list(res.meta and [])


def cpg_traces(seed: int = 0) -> tuple[np.ndarray, list, float]:
    """Joint-angle traces from the restricted CPG — the gait we measured
    walking inside the very same 18-DOF action space."""
    fly, world, *_rest = build_free_fly()
    sim = Simulation(world)
    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    controller = CPGController(
        cpg_network=make_tripod_cpg_network(sim.timestep, seed=seed),
        preprogrammed_steps=PreprogrammedSteps(), output_dof_order=dof_order)
    sim.reset()
    sim.warmup()

    allowed = set(adapter_controllable_dof_names())
    keep = np.array([d.name in allowed for d in dof_order], dtype=bool)
    neutral = np.asarray(controller.step().joint_angles, dtype=float).copy()

    traces = []
    for _ in range(int(DURATION_S / sim.timestep)):
        action = controller.step()
        angles = neutral.copy()
        angles[keep] = np.asarray(action.joint_angles, dtype=float)[keep]
        apply_locomotion_action(sim, fly.name, replace(action, joint_angles=angles))
        sim.step()
        traces.append(angles.copy())
    return np.asarray(traces), dof_order, sim.timestep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="media/trained_adapter/decoded_vs_working_gait.json")
    args = ap.parse_args()

    print("running the working (restricted CPG) gait...", flush=True)
    cpg, dof_order, cpg_dt = cpg_traces()
    print("running the untrained connectome on ground (R1a condition)...", flush=True)
    conn, _ = connectome_traces()

    names = [d.name for d in dof_order]
    allowed = adapter_controllable_dof_names()
    idx = [names.index(n) for n in allowed]

    rows = []
    print(f"\n{'DOF':38s}{'conn p2p':>10}{'cpg p2p':>9}{'ratio':>8}"
          f"{'conn Hz':>9}{'cpg Hz':>8}")
    for name, i in zip(allowed, idx):
        c_p2p = float(np.ptp(conn[:, i]))
        k_p2p = float(np.ptp(cpg[:, i]))
        ratio = c_p2p / k_p2p if k_p2p > 1e-9 else float("nan")
        c_hz = dominant_hz(conn[:, i], NEURAL_DT)
        k_hz = dominant_hz(cpg[:, i], cpg_dt)
        rows.append({"dof": name, "connectome_p2p_rad": c_p2p,
                     "cpg_p2p_rad": k_p2p, "ratio": ratio,
                     "connectome_hz": c_hz, "cpg_hz": k_hz})
        print(f"{name:38s}{c_p2p:10.4f}{k_p2p:9.4f}{ratio:8.3f}"
              f"{c_hz:9.2f}{k_hz:8.2f}")

    ratios = np.array([r["ratio"] for r in rows if np.isfinite(r["ratio"])])
    c_p2p_all = np.array([r["connectome_p2p_rad"] for r in rows])
    k_p2p_all = np.array([r["cpg_p2p_rad"] for r in rows])
    c_hz_all = np.array([r["connectome_hz"] for r in rows])
    k_hz_all = np.array([r["cpg_hz"] for r in rows])

    print(f"\nmean peak-to-peak: connectome {c_p2p_all.mean():.4f} rad, "
          f"working CPG {k_p2p_all.mean():.4f} rad")
    print(f"amplitude ratio (connectome/CPG): median {np.median(ratios):.3f}, "
          f"min {ratios.min():.3f}, max {ratios.max():.3f}")
    print(f"dominant Hz: connectome median {np.median(c_hz_all):.2f}, "
          f"working CPG median {np.median(k_hz_all):.2f}")

    # A stated-in-advance reading of the numbers, so the interpretation is
    # not invented after seeing them.
    med = float(np.median(ratios))
    print("\nreading:")
    if med < 0.2:
        print(f"  amplitude is {1/med:.0f}x TOO SMALL. The decoder's default "
              "gain/scale start far from any walking region; raising\n"
              "  motor_gain's default (or its bound) is a cheaper fix than "
              "asking CMA-ES to travel that distance.")
    elif med > 5.0:
        print(f"  amplitude is {med:.0f}x TOO LARGE — likely thrashing rather "
              "than stepping; the default is past the useful region.")
    else:
        print("  amplitude is within an order of magnitude of a gait that "
              "walks, so the search starts in a plausible region.")
    if np.median(c_hz_all) < 1.0:
        print("  the decoded joints are NOT oscillating in-band: the problem is "
              "upstream of amplitude (rhythm is not reaching the joints).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"matched_drive": MATCHED_DRIVE, "duration_s": DURATION_S,
                   "rows": rows,
                   "median_amplitude_ratio": med,
                   "connectome_mean_p2p_rad": float(c_p2p_all.mean()),
                   "cpg_mean_p2p_rad": float(k_p2p_all.mean()),
                   "connectome_median_hz": float(np.median(c_hz_all)),
                   "cpg_median_hz": float(np.median(k_hz_all))}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
