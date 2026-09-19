"""Check that `steppable_rate_model` reproduces Pugliese et al.'s own
published full-VNC output.

The steppable model changes three things about how their rate equation is
integrated (fixed-step RK4 instead of adaptive Dopri5; sparse instead of
dense matvec; time-varying input allowed) — see that module's docstring.
None of those touch the connectome, but all three could change the
numbers, so this script quantifies by how much, against the only ground
truth available: their real published 128-replicate simulation output
(Zenodo record 22260924).

Method: rebuild the exact per-neuron parameters of one published
replicate using THEIR OWN `prepare_neuron_params` with the seed from
their real `run_config.yaml`, integrate open-loop (no sensory input),
and compare to that replicate's published trace.

The four published runs used seeds 641, 98, 445 and 424 (32 replicates
each, concatenated in the order their archive filename lists them). We
verify that mapping empirically rather than assuming it.

Usage:
    python -m fly_robot.neural.validate_steppable_model
    python -m fly_robot.neural.validate_steppable_model --replicate 0 --substeps 2
"""

import argparse
import time

import numpy as np

from fly_robot.neural import pugliese_paths  # noqa: F401 — sys.path side effect
from fly_robot.neural.replicate_ensemble import load_published_full_vnc_replicates
from fly_robot.neural.steppable_rate_model import SteppableRateModel, build_neuron_set

import jax
from src.simulation.vnc_sim import prepare_neuron_params, prepare_sim_params
from src.utils.sim_utils import load_wTable
from src.utils.path_utils import create_fresh_config_with_paths

# From the four `logs/run_config.yaml` files in their Zenodo archive, in
# the order their stacked-output filename concatenates the run ids
# (30662613-33195857-33195876-33195882).
PUBLISHED_SEEDS = [641, 98, 445, 424]
REPLICATES_PER_SEED = 32


def published_replicate_to_seed(replicate: int) -> tuple[int, int]:
    """Maps a row of the published 128-replicate stack to (seed, index
    within that seed's 32). Verified empirically by this script — if the
    concatenation order were different, the traces simply would not
    match and the comparison below would show it."""
    return PUBLISHED_SEEDS[replicate // REPLICATES_PER_SEED], replicate % REPLICATES_PER_SEED


def build_published_replicate(replicate: int, substeps: int = 1):
    """Rebuilds one published replicate's neuron parameters via Pugliese's
    own `prepare_neuron_params`, and wraps them in a SteppableRateModel."""
    seed, within = published_replicate_to_seed(replicate)
    config = create_fresh_config_with_paths(
        experiment="FullVNC_DNg100_Stim", paths_template="fly_robot",
        run_id=f"validate_rep{replicate}",
    )
    config.experiment.n_replicates = REPLICATES_PER_SEED
    config.experiment.seed = seed

    wTable = load_wTable(config.experiment.dfPath)
    neuron_params = prepare_neuron_params(config, wTable)
    sim_params = prepare_sim_params(config, 1, len(neuron_params.W))

    one = jax.tree.map(lambda x: x[within], neuron_params)
    neurons = build_neuron_set(
        np.asarray(neuron_params.W), one.tau, one.a, one.threshold, one.fr_cap,
        exc_mult=sim_params.exc_multiplier, inh_mult=sim_params.inh_multiplier,
    )
    model = SteppableRateModel(
        neurons, baseline_input=np.asarray(neuron_params.input_currents[0][within]),
        pulse_start=sim_params.pulse_start, pulse_end=sim_params.pulse_end,
        substeps=substeps,
    )
    return model, wTable, sim_params


def compare(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """Compares two (n_neurons, n_timesteps) traces on the things that
    actually matter downstream: which neurons are recruited, and how
    closely the recruited ones track."""
    n = min(ours.shape[1], theirs.shape[1])
    ours, theirs = ours[:, :n], theirs[:, :n]

    ours_active = ours.max(axis=1) > 0.01
    theirs_active = theirs.max(axis=1) > 0.01
    both = ours_active & theirs_active

    peak_err = np.abs(ours.max(axis=1) - theirs.max(axis=1))
    corr = np.full(ours.shape[0], np.nan)
    for i in np.where(both)[0]:
        a, b = ours[i], theirs[i]
        if a.std() > 1e-9 and b.std() > 1e-9:
            corr[i] = np.corrcoef(a, b)[0, 1]

    return {
        "n_active_ours": int(ours_active.sum()),
        "n_active_theirs": int(theirs_active.sum()),
        "n_active_both": int(both.sum()),
        "jaccard": float(both.sum() / max((ours_active | theirs_active).sum(), 1)),
        "max_abs_err_overall": float(np.abs(ours - theirs).max()),
        "median_peak_err_active": float(np.median(peak_err[both])) if both.any() else np.nan,
        "median_corr_active": float(np.nanmedian(corr[both])) if both.any() else np.nan,
        "frac_active_corr_gt_0.99": float(np.nanmean(corr[both] > 0.99)) if both.any() else np.nan,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", type=int, default=0,
                        help="Which of the 128 published replicates to reproduce")
    parser.add_argument("--substeps", type=int, default=1,
                        help="RK4 substeps per 1ms step (integration refinement)")
    args = parser.parse_args()

    print(f"Loading published replicate {args.replicate} ...")
    _, R_stack, dt = load_published_full_vnc_replicates()
    theirs = R_stack[args.replicate]
    seed, within = published_replicate_to_seed(args.replicate)
    print(f"  -> their run used seed {seed}, replicate index {within} within that run")

    model, _, sim_params = build_published_replicate(args.replicate, substeps=args.substeps)
    n_steps = theirs.shape[1] - 1
    print(f"Integrating {n_steps} steps of {dt}s with RK4 x{args.substeps} (sparse) ...")
    t0 = time.time()
    ours = model.run_open_loop(n_steps, dt=dt)
    elapsed = time.time() - t0
    print(f"  -> {elapsed:.1f}s wall-clock for {n_steps * dt:.1f}s simulated "
          f"({elapsed / (n_steps * dt):.1f}x real time)")

    stats = compare(ours, theirs)
    print("\nAgreement with their published trace:")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")
