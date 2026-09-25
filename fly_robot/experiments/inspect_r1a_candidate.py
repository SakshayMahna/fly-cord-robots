"""Render video of R1a's current best candidate — look before believing.

This project's own standing rule: render top candidates and report what
they are actually doing, not just their score, before treating a number as
evidence. R1a's best score plateaued (+0.3154 at gen 18 to +0.3178 at gen
38, 20 generations for +0.0024) with forward progress essentially at zero
the whole time — this renders that exact candidate to see what "+0.32,
no progress" looks like physically: standing still safely, walking in
place, thrashing, or something else the reward doesn't capture.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.inspect_r1a_candidate \
        --checkpoint media/trained_adapter/r1a_ground/checkpoint.pkl
"""

import argparse
import pickle

import numpy as np

from fly_robot.adapter import reward_ground
from fly_robot.adapter.parameters import from_z
from fly_robot.sim.closed_loop import run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

MATCHED_DRIVE = 382.8125


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replicates", type=int, nargs="+", default=None,
                    help="default: the eligible pool recorded in the checkpoint")
    ap.add_argument("--duration-s", type=float, default=4.0)
    ap.add_argument("--out-prefix", default="media/trained_adapter/r1a_best")
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        state = pickle.load(f)
    z = np.asarray(state["best"]["z"], dtype=float)
    gen = state["best"]["generation"]
    score = state["best"]["score"]
    print(f"best candidate: gen {gen}, score {score:+.4f}")
    params = from_z(z)

    replicates = args.replicates or state["config"].get("eligible_replicates") or [0]
    print(f"rendering replicates: {replicates}\n")

    for rep in replicates:
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=rep, param_seed=PILOT_PARAM_SEED, duration_s=args.duration_s,
            stim_current=MATCHED_DRIVE)
        video_paths = {"top_down": f"{args.out_prefix}_r{rep}_top.mp4",
                       "side": f"{args.out_prefix}_r{rep}_side.mp4"}
        res = run_trial(model, mg, sensory_groups=sg, rig="ground",
                        duration_s=args.duration_s, seed=rep, adapter=params,
                        adapter_sensory=False, video_paths=video_paths)
        b = reward_ground.evaluate(res)
        print(f"replicate {rep}: dx={b.raw['dx_mm']:+.3f}mm "
              f"dy={b.raw['dy_mm']:+.3f}mm speed={b.raw['speed_mm_s']:+.3f}mm/s "
              f"upright_mean={b.raw['upright_mean']:.3f} "
              f"n_rhythmic={b.raw['n_rhythmic']}/6 "
              f"terminated={bool(res.terminated_at_step is not None)} "
              f"peak_n_active={res.peak_n_active_neurons} "
              f"reward={b.total:+.4f}")
        print(f"  -> {video_paths['top_down']}, {video_paths['side']}")


if __name__ == "__main__":
    main()
