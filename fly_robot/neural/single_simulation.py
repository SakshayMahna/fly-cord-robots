"""Run one instance of Pugliese et al.'s own (unmodified) firing-rate
simulation code — either a single fixed-stimulus run, or their documented
adaptive stimulus-search procedure. For running many replicates and
combining them, see `replicate_ensemble.py`.
"""

import time

from fly_robot.neural import pugliese_paths  # noqa: F401 — import side effect sets sys.path/env before the imports below

import jax
jax.config.update("jax_default_device", jax.devices("cpu")[0])
import jax.numpy as jnp
import numpy as np

from src.simulation.vnc_sim import (
    prepare_neuron_params, prepare_sim_params, run_single_simulation, reweight_connectivity,
)
from src.utils.sim_utils import load_wTable
from src.utils.path_utils import create_fresh_config_with_paths, register_custom_resolvers

register_custom_resolvers()


def run_single_simulation_helper(W, neuronParams, simParams, key):
    """Verbatim from Pugliese's Tutorial 1 notebook — not our method."""
    W_masked = W * neuronParams.W_mask
    W_reweighted = reweight_connectivity(W_masked, simParams.exc_multiplier, simParams.inh_multiplier)
    return run_single_simulation(
        W_reweighted, neuronParams.tau, neuronParams.a, neuronParams.threshold,
        neuronParams.fr_cap, neuronParams.input_currents[0], simParams.noise_stdv,
        simParams.t_axis, simParams.T, simParams.dt, simParams.pulse_start,
        simParams.pulse_end, simParams.r_tol, simParams.a_tol, key,
    )


def adaptive_stim_single(W, neuronParams_single, simParams, key, initial_stim_mask,
                          n_active_lower=5, n_active_upper=500,
                          n_high_fr_upper=100, high_fr_threshold=100.0,
                          max_iters=10):
    """Faithful re-implementation, for a single simulation, of the doubling
    /halving stimulus-search procedure documented in Pugliese et al.'s
    Methods ("Descending neuron activation screen": "If the simulation was
    underactive, Istim was doubled... If it was oversaturated, Istim was
    halved... repeated up to a maximum of 10 times") and implemented for
    their batched runner in `vnc_sim._adjust_stimulation_for_batch`
    (src/simulation/vnc_sim.py:1143). We can't reuse that function directly
    — it's written for their pmap/batch/checkpoint runner — so this mirrors
    its exact thresholds and bound-narrowing logic for the n_replicates=1
    case we need. `n_active` here uses their exact criterion from that
    function: sum(R, axis=time) > 0, i.e. any nonzero activity at all, not
    the stricter >0.01 threshold used elsewhere for motor-neuron scoring.
    """
    input_currents = neuronParams_single.input_currents
    stim_value = float(input_currents[initial_stim_mask][0])
    next_highest, next_lowest = None, None

    for it in range(max_iters):
        params_iter = neuronParams_single._replace(input_currents=input_currents)
        R = run_single_simulation_helper(W, params_iter, simParams, key)
        R = np.array(R)

        n_active = int((R.sum(axis=1) > 0).sum())
        n_high_fr = int((R.max(axis=1) > high_fr_threshold).sum())
        print(f"  [adaptive] iter {it + 1}/{max_iters}: stimI={stim_value:.2f}, "
              f"n_active={n_active}, n_high_fr={n_high_fr}")

        oversaturated = (n_active > n_active_upper) or (n_high_fr > n_high_fr_upper)
        underactive = n_active < n_active_lower

        if not oversaturated and not underactive:
            print(f"  [adaptive] converged: stimI={stim_value:.2f}")
            return R, stim_value, True

        if oversaturated:
            next_highest = stim_value
            stim_value = stim_value / 2 if next_lowest is None else (stim_value + next_lowest) / 2
        else:  # underactive
            next_lowest = stim_value
            stim_value = stim_value * 2 if next_highest is None else (stim_value + next_highest) / 2

        input_currents = jnp.where(initial_stim_mask, stim_value, input_currents)

    print(f"  [adaptive] did not converge after {max_iters} iterations (last stimI={stim_value:.2f})")
    return R, stim_value, False


def run_experiment(experiment_name: str, rtol=1e-4, atol=1e-7, seed=0, adaptive=False):
    # config_dir left at its default ("../../configs", relative to
    # path_utils.py's own location inside the Pugliese repo) — Hydra's
    # initialize() requires a relative path, so we can't point it at our
    # absolute PUGLIESE_REPO path directly.
    config = create_fresh_config_with_paths(
        experiment=experiment_name, paths_template="fly_robot",
        run_id=f"phase0_{experiment_name}",
    )
    config.experiment.n_replicates = 1
    config.sim.rtol = rtol
    config.sim.atol = atol

    wTable = load_wTable(config.experiment.dfPath)
    neuronParams = prepare_neuron_params(config, wTable)
    nNeurons = len(neuronParams.W)
    simParams = prepare_sim_params(config, 1, nNeurons)

    neuronParams_single = jax.tree.map(lambda x: x[0], neuronParams)
    key = jax.random.PRNGKey(seed)

    t0 = time.time()
    if adaptive:
        stim_mask = neuronParams_single.input_currents != 0
        R, final_stim, converged = adaptive_stim_single(
            neuronParams.W, neuronParams_single, simParams, key, stim_mask,
        )
        R = jnp.asarray(R)
        print(f"[{experiment_name}] adaptive stim search {'converged' if converged else 'did NOT converge'} "
              f"at stimI={final_stim:.2f}")
    else:
        R = run_single_simulation_helper(neuronParams.W, neuronParams_single, simParams, key)
        R.block_until_ready()
    elapsed = time.time() - t0
    print(f"[{experiment_name}] simulated {nNeurons} neurons, {simParams.T}s @ dt={simParams.dt} "
          f"in {elapsed:.1f}s wall-clock")

    return wTable, np.array(R), simParams
