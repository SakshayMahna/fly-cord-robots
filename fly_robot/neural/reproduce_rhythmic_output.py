"""Reproduce Pugliese et al.'s rhythmic motor-neuron result — Phase 0's
original goal — using their own unmodified simulation code, in two modes:

1. `t1` — their published T1 (front leg) network, their T1 config,
   stimulating DNg100. A validation that our environment/pipeline
   reproduces their published rhythmic motor output, not something we
   invented.

2. `full_vnc` — our own addition, not in the paper: the SAME dynamics
   equations and the SAME unmodified DNg100 stimulation, but run on the
   true whole-VNC network (all six legs) instead of their T1-restricted
   network. Nothing about their model equations, connectome weights, or
   neurotransmitter signs is modified; we only feed the simulator a
   larger real slice of the same connectome.

Usage:
    python -m fly_robot.neural.reproduce_rhythmic_output --mode t1
    python -m fly_robot.neural.reproduce_rhythmic_output --mode full_vnc
"""

import argparse
from pathlib import Path

from fly_robot.neural.single_simulation import run_experiment
from fly_robot.neural.oscillation_scoring import score_and_plot_by_segment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["t1", "full_vnc"], required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--out-dir", default="media/simulation")
    parser.add_argument("--adaptive", action="store_true",
                         help="Use Pugliese et al.'s documented doubling/halving stimulus search "
                              "instead of a fixed stimI (see Methods, 'Descending neuron activation screen').")
    args = parser.parse_args()

    experiment = "DNg100_Stim" if args.mode == "t1" else "FullVNC_DNg100_Stim"
    wTable, R, simParams = run_experiment(experiment, rtol=args.rtol, atol=args.atol, adaptive=args.adaptive)
    score_and_plot_by_segment(wTable, R, out_prefix=args.mode, out_dir=Path(args.out_dir), dt=float(simParams.dt))
