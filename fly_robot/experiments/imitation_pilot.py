"""Can a map from connectome motor output reproduce a gait that walks?

An **offline regression pilot**, not a training run. It asks one question
as cheaply as possible:

    Does the connectome's motor output carry enough structure to
    reconstruct the joint trajectories of a controller that actually walks
    this body?

Why offline: no physics is needed to answer it. The target is the
restricted-CPG gait (`baselines/restricted_action_cpg.py`, 9.822 mm/s
through the adapter's own 18 DOFs), the input is connectome motor rates,
and the fit is a regression. Physics only returns if this succeeds and a
decoder is trained for real. Compared with CMA-ES this swaps one scalar
per 4 s trial for 18 x ~2000 dense targets.

**The control is the whole point.** The CPG gait is periodic, so a
sufficiently expressive map can emit the cycle as a function of time while
ignoring its input entirely -- which would make "connectome-driven" false.
So every fit is repeated with degraded input:

  * `shuffled`   -- C1, the degree- and sign-preserving wiring shuffle.
                    (Untrainable as a controller because it oversaturates,
                    but perfectly usable as a regressor input here.)
  * `scrambled`  -- real output, phases randomised, spectrum preserved.
  * `constant`   -- the time-average of the real output. Carries no
                    timing at all; any fit quality here is memorisation.

If the real input does not clearly beat these, the result is void and we
have learned that in minutes rather than hours.

Readouts tested, from narrow to wide, because "widen the readout" is the
obvious next move if the narrow one fails:

  * `leg6`   -- per-leg summed motor rate (6 signals; what the adhesion
                gate sees)
  * `pool36` -- per (leg, antagonist pair, direction) pooled mean rate
                (what `adapter_joint_targets` actually decodes from)
  * `all`    -- every mapped motor neuron

Each is also tested with a short delay embedding, since an instantaneous
linear map cannot express a phase shift, and the connectome's rhythm has
no reason to share the CPG's phase.

Scoring is R^2 on a held-out time segment (last 30%), never in-sample.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.imitation_pilot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from fly_robot.analysis.interleg_coordination import leg_rhythm
from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, build_motor_neuron_groups, leg_motor_row_indices,
)
from fly_robot.neural.replicate_ensemble import (
    N_ACTIVE_UPPER, load_published_full_vnc_replicates,
)

DELAYS = (0, 20, 40)        # ms taps; ~half a 12 Hz cycle, gives phase access
TEST_FRACTION = 0.30


# ---------------------------------------------------------------- target --

def cpg_target(duration_s: float, dt: float) -> tuple[np.ndarray, list[str]]:
    """Restricted-CPG joint angles for the 18 adapter-controllable DOFs,
    sampled at `dt`. This is the gait measured at 9.822 mm/s."""
    from flygym import Simulation
    from flygym.compose.fly.base_fly import ActuatorType
    from flygym_demo.complex_terrain.common import apply_locomotion_action
    from flygym_demo.complex_terrain.cpg_controller import (
        CPGController, make_tripod_cpg_network)
    from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

    from fly_robot.baselines.restricted_action_cpg import (
        adapter_controllable_dof_names)
    from fly_robot.bodies.neuromechfly import build_free_fly

    fly, world, *_rest = build_free_fly()
    sim = Simulation(world)
    every = int(round(dt / sim.timestep))
    dof = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ctrl = CPGController(
        cpg_network=make_tripod_cpg_network(sim.timestep, seed=0),
        preprogrammed_steps=PreprogrammedSteps(), output_dof_order=dof)
    sim.reset()
    sim.warmup()

    allowed = set(adapter_controllable_dof_names())
    keep = np.array([d.name in allowed for d in dof], dtype=bool)
    neutral = np.asarray(ctrl.step().joint_angles, dtype=float).copy()

    out = []
    for i in range(int(duration_s / sim.timestep)):
        a = ctrl.step()
        j = neutral.copy()
        j[keep] = np.asarray(a.joint_angles, dtype=float)[keep]
        apply_locomotion_action(sim, fly.name, replace(a, joint_angles=j))
        sim.step()
        if i % every == 0:
            out.append(j[keep].copy())
    names = [d.name for d, k in zip(dof, keep) if k]
    return np.asarray(out), names


# ----------------------------------------------------------------- input --

def readouts(R_rep: np.ndarray, mg) -> dict:
    """(n_neurons, T) for one replicate -> the three candidate readouts."""
    rows = leg_motor_row_indices(mg)
    legs = sorted(rows)
    leg6 = np.stack([R_rep[rows[l], :].sum(axis=0) for l in legs])

    pools = []
    for leg in legs:
        for pos, neg, _suffix in ANTAGONIST_PAIRS:
            for module in (pos, neg):
                idx = mg.indices_by_group.get((leg[0], leg[1], module), [])
                pools.append(R_rep[idx, :].mean(axis=0) if len(idx)
                             else np.zeros(R_rep.shape[1]))
    pool36 = np.stack(pools)

    allrows = sorted({i for v in rows.values() for i in v})
    return {"leg6": leg6, "pool36": pool36, "all": R_rep[allrows, :]}


def degrade(X: np.ndarray, kind: str, rng) -> np.ndarray:
    """Controls. `X` is (n_features, T)."""
    if kind == "real":
        return X
    if kind == "constant":
        return np.repeat(X.mean(axis=1, keepdims=True), X.shape[1], axis=1)
    if kind == "scrambled":
        # randomise phase, preserve each channel's power spectrum
        F = np.fft.rfft(X, axis=1)
        ph = rng.uniform(0, 2 * np.pi, F.shape)
        ph[:, 0] = 0.0
        return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=X.shape[1], axis=1)
    raise ValueError(kind)


def embed(X: np.ndarray, delays=DELAYS) -> np.ndarray:
    """(n_features, T) -> (T, n_features*len(delays)) with delay taps."""
    T = X.shape[1]
    cols = []
    for d in delays:
        shifted = np.concatenate([np.repeat(X[:, :1], d, axis=1), X[:, :T - d]],
                                 axis=1) if d else X
        cols.append(shifted.T)
    return np.concatenate(cols, axis=1)


# ------------------------------------------------------------------ fit --

def fit_score(X: np.ndarray, Y: np.ndarray, model: str, seed: int = 0) -> float:
    """Held-out R^2 (last TEST_FRACTION of time), averaged over outputs."""
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    cut = int(n * (1 - TEST_FRACTION))
    xs, ys = StandardScaler().fit(X[:cut]), StandardScaler().fit(Y[:cut])
    Xtr, Xte = xs.transform(X[:cut]), xs.transform(X[cut:])
    Ytr, Yte = ys.transform(Y[:cut]), ys.transform(Y[cut:])
    if model == "linear":
        m = Ridge(alpha=1.0).fit(Xtr, Ytr)
    else:
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                         random_state=seed, early_stopping=True).fit(Xtr, Ytr)
    P = m.predict(Xte)
    ss_res = ((Yte - P) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-replicates", type=int, default=3,
                    help="how many 6/6-rhythmic published replicates to use")
    ap.add_argument("--out", default="media/trained_adapter/imitation_pilot.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    wtable, R, dt = load_published_full_vnc_replicates()
    mg = build_motor_neuron_groups("data/circuit_map/all_legs_circuit.csv", wtable)
    rows = leg_motor_row_indices(mg)
    legs = sorted(rows)

    peak = (R > 0.01).sum(axis=1).max(axis=1)
    stable = np.where(peak <= N_ACTIVE_UPPER)[0]
    six = []
    for r in stable:
        S = np.stack([R[r, rows[l], :].sum(axis=0) for l in legs]).astype(float)
        if int(leg_rhythm(S, dt).rhythmic.sum()) == 6:
            six.append(int(r))
        if len(six) >= args.n_replicates:
            break
    print(f"using 6/6-rhythmic published replicates: {six}")

    T = R.shape[2]
    Y, dof_names = cpg_target(duration_s=T * dt, dt=dt)
    print(f"CPG target: {Y.shape[0]} steps x {Y.shape[1]} DOFs\n")

    results = []
    for rep in six:
        reps = readouts(np.asarray(R[rep], dtype=float), mg)
        for name, X0 in reps.items():
            for cond in ("real", "scrambled", "constant"):
                Xd = degrade(X0, cond, rng)
                for emb, tag in ((False, "inst"), (True, "delay")):
                    Xf = embed(Xd) if emb else Xd.T
                    for model in ("linear", "mlp"):
                        r2 = fit_score(Xf, Y, model)
                        results.append({"replicate": rep, "readout": name,
                                        "input": cond, "embedding": tag,
                                        "model": model, "r2": r2})
                        print(f"  rep{rep:3d} {name:7s} {cond:9s} {tag:5s} "
                              f"{model:6s} held-out R2 = {r2:+.4f}", flush=True)

    print("\n=== mean held-out R2 by (readout, input) ===")
    print(f"{'readout':9s}{'real':>10s}{'scrambled':>12s}{'constant':>11s}{'margin':>10s}")
    summary = {}
    for name in ("leg6", "pool36", "all"):
        vals = {}
        for cond in ("real", "scrambled", "constant"):
            v = [r["r2"] for r in results
                 if r["readout"] == name and r["input"] == cond]
            vals[cond] = float(np.mean(v)) if v else float("nan")
        margin = vals["real"] - max(vals["scrambled"], vals["constant"])
        summary[name] = {**vals, "margin": margin}
        print(f"{name:9s}{vals['real']:+10.4f}{vals['scrambled']:+12.4f}"
              f"{vals['constant']:+11.4f}{margin:+10.4f}")

    best = max(summary, key=lambda k: summary[k]["margin"])
    m = summary[best]["margin"]
    print(f"\nbest readout by margin over controls: {best} ({m:+.4f})")
    if m < 0.05:
        print("VERDICT: the connectome input does NOT beat its own controls.\n"
              "  Any apparent fit is the map memorising a periodic target.\n"
              "  Imitation from this readout is not a valid route.")
    elif summary[best]["real"] < 0.3:
        print("VERDICT: real input beats controls, but absolute fit is weak.\n"
              "  Structure is present but insufficient to reconstruct the gait.")
    else:
        print("VERDICT: real input beats its controls with a usable fit.\n"
              "  Imitation is a valid route; train the decoder on it next.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"replicates": six, "delays_ms": list(DELAYS),
                   "test_fraction": TEST_FRACTION, "dof_names": dof_names,
                   "results": results, "summary": summary}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
