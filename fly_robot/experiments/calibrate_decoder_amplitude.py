"""Find a decoder starting point whose joint excursion matches a gait that
walks — and measure what that does to actual forward speed.

Motivation, measured not assumed
--------------------------------
`compare_decoded_to_working_gait.py` found that the untrained decoder
produces the **right frequency** (12.00 Hz, matching the working restricted
CPG almost exactly on every DOF) but only **~1/11th of the amplitude**
(0.062 rad mean peak-to-peak vs 0.707). The decoder is

    joint = neutral + offset + gain * tanh((rate_pos - rate_neg) / scale)

so excursion is governed by `motor_gain` (a direct multiplier, bounds
[0.05, 1.20], default 0.50) and `motor_scale` (divides the tanh argument,
bounds [0.50, 20.0], default 3.00). Raising gain to its bound gives 2.4x;
dropping scale to its bound gives ~6x more tanh argument while unsaturated.
The ~11x gap is therefore reachable **inside the existing bounds** — the
default simply sits in a weak corner of them.

This sweeps that corner and reports, for each (gain, scale), both the
excursion ratio against the working gait AND the forward speed the fly
actually achieves. Speed is the real objective, so a configuration that
walks would show up here in minutes instead of after a 20-hour search.

Honesty notes
-------------
* The CPG's amplitude is used **only as a measuring stick** for the starting
  point. No CPG output enters the runtime path: every trial here is
  `DNg100 -> frozen connectome -> motor neurons -> decoder -> body`, with
  the sensory path off (R1a condition). `GROUND_BRIDGE.md` §5 permits
  exactly this and requires it be stated outright.
* This is a **starting point**, not a result. It sets where CMA-ES begins;
  it does not replace the search, and its own speeds are reported as
  untrained-with-calibrated-constants, never as "trained".
* Uniform gain/scale across all (pair, segment) entries is used here on
  purpose — per-entry tuning is CMA-ES's job, and hand-tuning 18 entries
  would be the per-condition hand tuning this project forbids.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.calibrate_decoder_amplitude
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fly_robot.adapter import reward_ground
from fly_robot.adapter.parameters import (
    N_PARAMS, default_params, default_z, from_z, to_z,
)
from fly_robot.sim.closed_loop import run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

MATCHED_DRIVE = 382.8125
DURATION_S = 2.0
# Working-gait reference, measured in compare_decoded_to_working_gait.py.
CPG_MEAN_P2P_RAD = 0.7066

GAINS = (0.50, 0.80, 1.20)
SCALES = (3.00, 1.50, 0.80, 0.50)


def build_params(gain: float, scale: float):
    """Default adapter with motor_gain / motor_scale set uniformly."""
    p = default_params()
    values = dict(p.values)
    values["motor_gain"] = np.full_like(np.asarray(p.motor_gain, dtype=float), gain)
    values["motor_scale"] = np.full_like(np.asarray(p.motor_scale, dtype=float), scale)
    return type(p)(values=values)


def run_one(gain: float, scale: float, replicate: int, driven_idx: np.ndarray):
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
        stim_current=MATCHED_DRIVE)
    res = run_trial(model, mg, sensory_groups=sg, rig="ground",
                    duration_s=DURATION_S, seed=replicate,
                    adapter=build_params(gain, scale), adapter_sensory=False)
    n = res.terminated_at_step or res.n_steps_planned
    angles = np.asarray(res.joint_angles[:n], dtype=float)
    p2p = float(np.ptp(angles[:, driven_idx], axis=0).mean())
    breakdown = reward_ground.evaluate(res)
    return {"gain": gain, "scale": scale, "replicate": replicate,
            "mean_p2p_rad": p2p, "ratio_vs_cpg": p2p / CPG_MEAN_P2P_RAD,
            "speed_mm_s": float(breakdown.raw["speed_mm_s"]),
            "dx_mm": float(breakdown.raw["dx_mm"]),
            "upright_mean": float(breakdown.raw["upright_mean"]),
            "terminated": bool(res.terminated_at_step is not None),
            "peak_n_active": int(res.peak_n_active_neurons or 0),
            "reward": float(breakdown.total)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="media/trained_adapter/decoder_calibration.json")
    args = ap.parse_args()

    # The 18 DOFs the decoder can move, resolved from the interface itself.
    from fly_robot.baselines.restricted_action_cpg import adapter_controllable_dof_names
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=args.replicates[0], param_seed=PILOT_PARAM_SEED,
        duration_s=0.05, stim_current=MATCHED_DRIVE)
    probe = run_trial(model, mg, sensory_groups=sg, rig="ground", duration_s=0.05,
                      seed=0, adapter=default_params(), adapter_sensory=False)
    names = list(probe.joint_names) if getattr(probe, "joint_names", None) else None
    if names is None:
        # joint_angles columns follow the body's actuated DOF order; rebuild it.
        from flygym import Simulation
        from flygym.compose.fly.base_fly import ActuatorType
        from fly_robot.bodies.neuromechfly import build_free_fly
        fly, world, *_ = build_free_fly()
        Simulation(world)
        names = [d.name for d in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)]
    allowed = set(adapter_controllable_dof_names())
    driven_idx = np.array([i for i, n in enumerate(names) if n in allowed])
    if len(driven_idx) != 18:
        raise RuntimeError(f"expected 18 driven DOFs, matched {len(driven_idx)}")

    print(f"working-gait reference: {CPG_MEAN_P2P_RAD:.4f} rad mean p2p "
          f"(restricted CPG, 9.822 mm/s)\n")
    print(f"{'gain':>6}{'scale':>7}{'p2p rad':>10}{'ratio':>8}"
          f"{'speed mm/s':>12}{'upright':>9}{'term':>6}{'reward':>9}")

    rows = []
    for gain in GAINS:
        for scale in SCALES:
            per_rep = [run_one(gain, scale, rep, driven_idx)
                       for rep in args.replicates]
            rows.extend(per_rep)
            mean = {k: float(np.mean([r[k] for r in per_rep]))
                    for k in ("mean_p2p_rad", "ratio_vs_cpg", "speed_mm_s",
                              "upright_mean", "reward")}
            term = any(r["terminated"] for r in per_rep)
            print(f"{gain:6.2f}{scale:7.2f}{mean['mean_p2p_rad']:10.4f}"
                  f"{mean['ratio_vs_cpg']:8.3f}{mean['speed_mm_s']:12.3f}"
                  f"{mean['upright_mean']:9.3f}{str(term):>6}"
                  f"{mean['reward']:9.4f}", flush=True)

    # Best by SPEED (the objective), and closest by amplitude (the calibration
    # target) -- reported separately, because they need not agree and the
    # difference is itself informative.
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["gain"], r["scale"]), []).append(r)
    agg = [{"gain": g, "scale": s,
            "speed_mm_s": float(np.mean([r["speed_mm_s"] for r in v])),
            "ratio_vs_cpg": float(np.mean([r["ratio_vs_cpg"] for r in v])),
            "reward": float(np.mean([r["reward"] for r in v])),
            "any_terminated": bool(any(r["terminated"] for r in v))}
           for (g, s), v in by_cfg.items()]

    # A terminated trial reports speed 0.000 because the fly FLIPPED, not
    # because it stood still -- so selecting on speed alone happily returns a
    # configuration that falls over. Warm-starting a 20-hour search there
    # would be worse than not calibrating at all. Survivors only.
    survivors = [a for a in agg if not a["any_terminated"]]
    if not survivors:
        raise RuntimeError(
            "every configuration in the sweep terminated (flipped). There is "
            "no stable warm start to write; widen the sweep or treat stability "
            "as the thing to solve first.")
    fastest = max(survivors, key=lambda a: a["speed_mm_s"])
    closest = min(survivors, key=lambda a: abs(a["ratio_vs_cpg"] - 1.0))
    n_flipped = len(agg) - len(survivors)
    print(f"\n{n_flipped} of {len(agg)} configurations FLIPPED the fly "
          f"(their 0.000 mm/s is termination, not stillness)")
    print(f"fastest SURVIVING: gain={fastest['gain']:.2f} "
          f"scale={fastest['scale']:.2f} "
          f"-> {fastest['speed_mm_s']:+.3f} mm/s (amplitude "
          f"{fastest['ratio_vs_cpg']:.2f}x the working gait)")
    print(f"closest to 1x: gain={closest['gain']:.2f} scale={closest['scale']:.2f} "
          f"-> {closest['speed_mm_s']:+.3f} mm/s (amplitude "
          f"{closest['ratio_vs_cpg']:.2f}x)")
    print(f"\ndefault for comparison: gain=0.50 scale=3.00 -> "
          f"{[a for a in agg if a['gain']==0.50 and a['scale']==3.00][0]['speed_mm_s']:+.3f} mm/s")

    # Write the fastest configuration as a CMA-ES warm start. Chosen by SPEED
    # rather than by amplitude match: amplitude was only ever a proxy for
    # "somewhere a walking solution could live".
    z = to_z(build_params(fastest["gain"], fastest["scale"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cpg_reference_p2p_rad": CPG_MEAN_P2P_RAD,
               "gains": list(GAINS), "scales": list(SCALES),
               "replicates": args.replicates, "rows": rows, "aggregate": agg,
               "fastest": fastest, "closest_amplitude_match": closest,
               # Written in the shape --warm-start reads, so R1a can start
               # from the calibrated decoder without hand-editing anything.
               "best": {"z": np.asarray(z, dtype=float).tolist(),
                        "score": None, "generation": None,
                        "provenance": "calibrate_decoder_amplitude.py: "
                                      "uniform motor_gain/motor_scale chosen "
                                      "by measured forward speed"}}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
