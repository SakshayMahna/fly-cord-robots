"""Look at what the trained adapter is ACTUALLY doing — not just its score.

A scalar reward cannot distinguish walking from a degenerate strategy that
happens to earn the same number. This renders video of the top candidate
and runs explicit detectors for the failure modes an optimiser is most
likely to find, so that a good score is never reported as "it walked"
without someone having looked.

The detectors below are diagnostics, NOT penalties. Adding a penalty
changes the objective and is a decision for the project owner, so this
script reports and proposes; it never edits the reward.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.inspect_trained_adapter --run-name pilot
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fly_robot.adapter.parameters import PARAM_SPEC, default_params, from_z
from fly_robot.adapter.reward import evaluate
from fly_robot.analysis.interleg_coordination import (
    LEGS, TRANSIENT_S, coordination,
)
from fly_robot.interface.motor_neuron_to_joint import LEG_NAME_TO_FLYGYM_PREFIX
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

WALKING_REF = 0.95


def leg_joint_columns(all_jointdofs):
    """DOF indices belonging to each leg, for per-leg activity measures."""
    out = {}
    for (seg, side), prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        out[(seg, side)] = [i for i, d in enumerate(all_jointdofs)
                            if d.name.startswith(f"{prefix}_")
                            or d.name == f"c_thorax-{prefix}_coxa-pitch"]
    return out


def diagnose(result, leg_cols) -> dict:
    """Behavioural description plus degenerate-strategy detectors."""
    keep = int(TRANSIENT_S / NEURAL_DT)
    angles = result.joint_angles[keep:]
    av = result.ball_angvel[keep:]
    pitch = float(np.mean(av[:, 1]))
    speed = -pitch / WALKING_REF           # forward is negative pitch

    motion = np.abs(np.diff(angles, axis=0)).mean(axis=0)   # per-DOF motion
    per_leg = {leg: float(motion[cols].sum()) for leg, cols in leg_cols.items()}
    total_motion = sum(per_leg.values()) or 1e-12
    leg_share = {leg: v / total_motion for leg, v in per_leg.items()}
    top_leg = max(leg_share, key=leg_share.get)

    rhythm, coord = coordination(result.motor_rates, NEURAL_DT)

    # temporal concentration of ball rotation: what fraction of the total
    # |rotation| happens in the busiest 10% of the trial?
    mag = np.abs(av[:, 1])
    k = max(1, len(mag) // 10)
    concentration = float(np.sort(mag)[-k:].sum() / (mag.sum() + 1e-12))

    d = {
        "forward_speed_frac_of_walking": speed,
        "pitch_rad_s": pitch,
        "roll_rad_s": float(np.mean(av[:, 0])),
        "yaw_rad_s": float(np.mean(av[:, 2])),
        "joint_motion_total": float(motion.sum()),
        "static_offset_rad": float(np.abs(angles.mean(axis=0)).max()),
        "n_rhythmic": int(rhythm.rhythmic.sum()),
        "dominant_hz": float(np.median(rhythm.dominant_hz[rhythm.rhythmic]))
        if rhythm.rhythmic.any() else float("nan"),
        "tripod_index": float(coord.tripod_index) if coord.has_rhythm else float("nan"),
        "n_active": int(result.n_active_neurons),
        "top_leg": f"{top_leg[0]}-{top_leg[1]}",
        "top_leg_share": float(leg_share[top_leg]),
        "rotation_concentration": concentration,
        "leg_share": {f"{s}-{d_}": round(v, 3) for (s, d_), v in leg_share.items()},
    }

    # --- degenerate-strategy detectors ------------------------------------
    flags = []
    if d["joint_motion_total"] > 0.5 and abs(speed) < 0.02:
        flags.append(("in_place_vibration",
                      "legs move a lot but the ball barely turns"))
    if d["top_leg_share"] > 0.45:
        flags.append(("single_leg_dominance",
                      f"{d['top_leg']} accounts for "
                      f"{d['top_leg_share']:.0%} of all joint motion"))
    if d["static_offset_rad"] > 0.25:
        flags.append(("posture_exploit",
                      f"a joint sits {d['static_offset_rad']:.2f} rad off neutral "
                      "on average — a held pose, not a gait"))
    if d["n_active"] > 1500:
        flags.append(("saturation", f"n_active={d['n_active']} exceeds Pugliese's "
                                    "1500 oversaturation criterion"))
    if speed < -0.05:
        flags.append(("backward_walking",
                      f"net travel is BACKWARD at {-speed:.2f} of walking speed"))
    if concentration > 0.5:
        flags.append(("impulsive_rotation",
                      f"{concentration:.0%} of ball rotation happens in the "
                      "busiest 10% of the trial — a shove, not sustained gait"))
    if abs(d["yaw_rad_s"]) > 2 * abs(pitch) and abs(d["yaw_rad_s"]) > 0.05:
        flags.append(("turning_in_place",
                      "yaw dominates pitch — spinning rather than walking"))
    if d["n_rhythmic"] > 0 and d["dominant_hz"] > 18.0:
        flags.append(("high_frequency_buzz",
                      f"dominant rhythm {d['dominant_hz']:.1f} Hz sits at the top "
                      "of the 2-20 Hz analysis band — may be buzzing, not stepping"))
    d["flags"] = flags
    return d


def describe(label, d, score):
    print(f"\n=== {label} ===")
    print(f"  reward                {score.total:+.4f}")
    for k, v in score.terms.items():
        print(f"     {k:14s} {v:+.4f}  (raw {score.raw[k]:+.4f})")
    print(f"  forward speed         {d['forward_speed_frac_of_walking']:+.4f} "
          f"of reference walking ({d['pitch_rad_s']:+.4f} rad/s pitch)")
    print(f"  rhythmic legs         {d['n_rhythmic']}/6   dominant "
          f"{d['dominant_hz']:.1f} Hz   tripod {d['tripod_index']:+.3f}")
    print(f"  n_active              {d['n_active']}")
    print(f"  joint motion          {d['joint_motion_total']:.4f}   "
          f"max static offset {d['static_offset_rad']:.3f} rad")
    print(f"  per-leg motion share  {d['leg_share']}")
    if d["flags"]:
        print("  DEGENERATE-STRATEGY FLAGS:")
        for name, why in d["flags"]:
            print(f"     [{name}] {why}")
    else:
        print("  no degenerate-strategy flags raised")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="pilot")
    ap.add_argument("--out-dir", default="media/trained_adapter")
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.out_dir) / args.run_name
    with open(run_dir / "best.json") as f:
        best = json.load(f)["best"]
    trained = from_z(np.array(best["z"]))
    print(f"top candidate: score {best['score']:+.4f} from generation "
          f"{best['generation']}")

    clips = run_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)

    model, mg, sg, _wt, _info = build_trial_components(
        replicate=args.replicate, param_seed=PILOT_PARAM_SEED,
        duration_s=args.duration)
    from flygym.compose.fly.base_fly import ActuatorType  # noqa: F401
    from fly_robot.bodies.neuromechfly import build_ball_fly
    fly, _w, _m, _d, *_c = build_ball_fly()
    leg_cols = leg_joint_columns(fly.get_jointdofs_order())

    rows = {}
    for label, params in (("untrained (defaults)", default_params()),
                          ("trained (top candidate)", trained)):
        video = None
        if not args.no_video:
            tag = "trained" if "trained" in label else "untrained"
            video = {c: clips / f"{tag}_{c}.mp4"
                     for c in ("side", "opposite_side", "top_down")}
        res = run_trial(model, mg, sensory_groups=sg, on_ball=True,
                        duration_s=args.duration, seed=args.replicate,
                        adapter=params, video_paths=video)
        d = diagnose(res, leg_cols)
        describe(label, d, evaluate(res))
        rows[label] = d
        if video:
            print(f"  video -> {clips}/{tag}_*.mp4")

    # what the search actually changed
    print("\n=== parameters the search moved most (trained vs default) ===")
    dflt = default_params()
    deltas = []
    for name, shape, low, high, _default in PARAM_SPEC:
        a = np.atleast_1d(np.asarray(dflt.values[name], dtype=float))
        b = np.atleast_1d(np.asarray(trained.values[name], dtype=float))
        rel = np.abs(b - a) / (high - low)
        deltas.append((float(rel.max()), name, a.ravel()[int(np.argmax(rel))],
                       b.ravel()[int(np.argmax(rel))]))
    for rel, name, a, b in sorted(deltas, reverse=True)[:10]:
        print(f"  {name:20s} {a:10.4f} -> {b:10.4f}   ({rel:.0%} of its range)")

    with open(run_dir / "inspection.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    print(f"\nwrote {run_dir}/inspection.json")


if __name__ == "__main__":
    main()
