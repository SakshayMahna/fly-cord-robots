"""Assemble the pieces a closed-loop trial needs, from a config.

Kept separate from `closed_loop.py` so the loop itself stays a thin,
readable stepping function and everything about *which* connectome,
*which* neuron parameters and *which* control manipulation is applied
lives in one auditable place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fly_robot.interface.joint_to_sensory_neuron import build_sensory_groups
from fly_robot.interface.motor_neuron_to_joint import build_motor_neuron_groups
from fly_robot.neural import pugliese_paths  # noqa: F401 — sys.path side effect
from fly_robot.neural.connectome_controls import degree_preserving_shuffle
from fly_robot.neural.steppable_rate_model import SteppableRateModel, build_neuron_set

import jax
from src.simulation.vnc_sim import prepare_neuron_params, prepare_sim_params
from src.utils.sim_utils import load_wTable
from src.utils.path_utils import create_fresh_config_with_paths

# Pre-registered seed sets. Pilot seeds are the parameter draws already
# used throughout the audit; main seeds are a draw never used for any
# tuning or exploratory decision.
PILOT_PARAM_SEED = 641
PILOT_REPLICATE_INDICES = tuple(range(8))
MAIN_PARAM_SEED = 20260919
MAIN_REPLICATE_INDICES = tuple(range(20))

DEFAULT_STIM_CURRENT = 380.0   # Pugliese's own verified value
DNG100_ROWS = (59, 282)


def build_trial_components(replicate: int, param_seed: int = PILOT_PARAM_SEED,
                            n_replicates: int = 32,
                            stim_current: float = DEFAULT_STIM_CURRENT,
                            duration_s: float = 4.0,
                            shuffle_seed: int | None = None,
                            shuffle_protected: bool = True,
                            normalise_sensory_sides: bool = True,
                            circuit_csv: str = "data/circuit_map/all_legs_circuit.csv"):
    """Returns `(neural_model, motor_groups, sensory_groups, wtable, info)`.

    `shuffle_seed=None` uses the real connectome. Any other value applies
    the C1/C1b degree-preserving shuffle — a labelled control; the real
    matrix is never modified in place.
    """
    config = create_fresh_config_with_paths(
        experiment="FullVNC_DNg100_Stim", paths_template="fly_robot",
        run_id=f"closed_loop_s{param_seed}_r{replicate}",
    )
    config.experiment.n_replicates = n_replicates
    config.experiment.seed = param_seed
    # Stimulation runs for the whole trial, ending one neural step short of
    # T exactly as Pugliese's own config does (pulseEnd 1.999 for T 2.0).
    config.sim.T = float(duration_s)
    config.sim.pulseEnd = float(duration_s) - 0.001

    wtable = load_wTable(config.experiment.dfPath)
    neuron_params = prepare_neuron_params(config, wtable)
    sim_params = prepare_sim_params(config, 1, len(neuron_params.W))
    one = jax.tree.map(lambda x: x[replicate], neuron_params)

    weights = sp.csr_matrix(np.asarray(neuron_params.W))
    shuffle_report = None
    if shuffle_seed is not None:
        protected_rows = protected_cols = None
        if shuffle_protected:
            sensory = np.array(wtable.index[wtable["class"].isin(
                ["sensory neuron", "sensory ascending"])])
            motor = np.array(wtable.index[wtable["class"] == "motor neuron"])
            protected_rows = np.concatenate([sensory, np.array(DNG100_ROWS)])
            protected_cols = motor
        weights, shuffle_report = degree_preserving_shuffle(
            weights, seed=shuffle_seed, protected_rows=protected_rows,
            protected_cols=protected_cols)

    neurons = build_neuron_set(
        weights, one.tau, one.a, one.threshold, one.fr_cap,
        exc_mult=sim_params.exc_multiplier, inh_mult=sim_params.inh_multiplier)

    baseline_input = np.zeros(len(wtable))
    baseline_input[list(DNG100_ROWS)] = stim_current

    model = SteppableRateModel(
        neurons, baseline_input=baseline_input,
        pulse_start=sim_params.pulse_start, pulse_end=sim_params.pulse_end)

    motor_groups = build_motor_neuron_groups(circuit_csv, wtable)
    sensory_groups = build_sensory_groups(
        wtable, normalise_sides=normalise_sensory_sides)

    info = {"param_seed": param_seed, "replicate": replicate,
            "stim_current": stim_current, "duration_s": duration_s,
            "pulse_start": sim_params.pulse_start, "pulse_end": sim_params.pulse_end,
            "shuffled": shuffle_seed is not None,
            "shuffle_protected": shuffle_protected if shuffle_seed is not None else None,
            "shuffle_report": shuffle_report}
    return model, motor_groups, sensory_groups, wtable, info
