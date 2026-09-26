"""Train the ground-walking adapter with CMA-ES. The connectome is never trained.

The lean settings are the **pilot**, approved as a pipeline shakedown only:
a success is informative, a failure at 2 episodes/candidate is NOT
reportable as a negative result (`docs/trained_adapter/DESIGN.md` §5.4).
Before any negative claim, §5.7's protocol applies — measure the score
noise, size the episode count, re-run powered (E = 6).

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.train_adapter --run-name r1a_ground --stage output

Resumes automatically from the last checkpoint if one exists; pass
--fresh to start over.
"""

import argparse
import json
from pathlib import Path

from fly_robot.training.cma_trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="pilot")
    ap.add_argument("--condition", default="real", choices=("real", "C1", "C2"))
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=2,
                    help="2 = lean pilot; 6 = powered (DESIGN.md §5.7)")
    ap.add_argument("--generations", type=int, default=250)
    ap.add_argument("--workers", type=int, default=6,
                    help="6 is the measured local-throughput knee; benchmark a VM first")
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--trial-s", type=float, default=4.0)
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--out-dir", default="media/trained_adapter")
    ap.add_argument("--stage", choices=("output", "sensory", "full"),
                    default="output",
                    help="output=R1a, sensory=R1b warm-start refinement, full=exploratory")
    ap.add_argument("--gap-conductance", type=float, default=0.0,
                    help="electrical coupling between leg CPGs (OUR addition; "
                         "0 = published model exactly). 20 locks all six legs "
                         "to one frequency -- see GAP_JUNCTIONS.md")
    ap.add_argument("--warm-start", type=Path,
                    help="best.json from a prior run; required for a meaningful R1b")
    args = ap.parse_args()

    initial_z = []
    if args.warm_start:
        payload = json.loads(args.warm_start.read_text())
        candidate = payload.get("best", payload)
        if "z" not in candidate:
            ap.error(f"{args.warm_start} does not contain a best.z vector")
        initial_z = candidate["z"]
    if args.stage == "sensory" and not initial_z:
        ap.error("--stage sensory requires --warm-start R1a/best.json")

    # C1 is the degree- and sign-preserving shuffle, interface edges held
    # out — Phase 4's primary wiring control, reused unchanged.
    shuffle_seed = 20260921 if args.condition == "C1" else None

    cfg = TrainConfig(
        run_name=args.run_name, condition=args.condition,
        population=args.population, episodes=args.episodes,
        generations=args.generations, workers=args.workers,
        sigma0=args.sigma0, seed=args.seed, trial_s=args.trial_s,
        shuffle_seed=shuffle_seed, out_dir=args.out_dir, stage=args.stage,
        adapter_sensory=args.stage != "output", initial_z=initial_z,
        gap_conductance=args.gap_conductance,
    )
    best, history = Trainer(cfg).run(resume=not args.fresh)
    print(f"\nbest score {best['score']:+.4f} at generation {best['generation']}")
    print(f"wrote {cfg.out_dir}/{cfg.run_name}/best.json")


if __name__ == "__main__":
    main()
