"""Coupling-strength bound sweep for Rung 2 Option A — WRITE-side safety
only, using an OPEN-LOOP reference signal.

**This does NOT use `CausalPhaseEstimator`.** That estimator failed
validation (`validate_causal_phase.py`: mean circular correlation 0.301
against the offline method, minimum -0.251 — see
`docs/trained_adapter/RUNG2_DESIGN.md`) and per your instruction is not to
be used by the conductor until it passes. Using it here to shape the
current injected in this sweep would make any saturation result
uninterpretable — a failure could be the broken estimator, not the
coupling strength.

Instead: precompute a real per-leg phase trace ONCE, offline, from an
actual untrained trial (the established, already-validated whole-trace
Hilbert method), then REPLAY it open-loop as a fixed function of time
while a fresh trial runs, injecting

    current(leg, t) = clip(K * sin(phase_error(leg, t)), -cap, cap)

into that leg's 3 CPG-triad rows, where `phase_error` uses the same
tripod target structure Option A proposes (same group = target 0,
opposite group = target `preferred_phase_offset`, defaulting to pi).

This isolates exactly one question: **for a current shaped like a real
phase-locked correction, how large can the coupling strength K get before
the network saturates or loses rhythm?** It does not test whether the
correction is accurate (the estimator's job, which failed) or whether it
actually improves coordination (a question for once both pieces work) --
only whether the ACT of injecting current of this magnitude into the CPG
triad is safe. Reading a fixed current cap `cap = 2.5` from Phase 4's own
measured `SENSORY_CURRENT_CAP` as an upper search bound, since it is the
only previously-measured "safe-ish order of magnitude" for current into
this connectome; not assumed to transfer, hence the sweep.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.experiments.sweep_coupling_bound
"""

import argparse
import json

import numpy as np

from fly_robot.adapter.cpg_groups import build_cpg_groups
from fly_robot.adapter.parameters import default_params
from fly_robot.analysis.interleg_coordination import LEGS, TRANSIENT_S, coordination, leg_rhythm
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components
from fly_robot.training.replicate_filter import STABILITY_MAX_N_ACTIVE

MATCHED_DRIVE = 382.8125
REFERENCE_REPLICATE = 0       # 6/6 rhythmic in the untrained baseline
SWEEP_REPLICATES = (0, 1, 4, 6)   # pilot seeds only, subset of the eligible pool
DURATION_S = 4.0
CAP = 2.5    # Phase 4's SENSORY_CURRENT_CAP, used as the search ceiling, not an assumption
PREFERRED_PHASE_OFFSET = np.pi   # tripod target

TRIPOD_A = {("T1", "LHS"), ("T2", "RHS"), ("T3", "LHS")}


def build_reference_phase():
    """Offline whole-trace phase for every leg, from one real untrained
    trial -- the fixed, precomputed reference this sweep replays."""
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=REFERENCE_REPLICATE, param_seed=PILOT_PARAM_SEED,
        duration_s=DURATION_S, stim_current=MATCHED_DRIVE)
    res = run_trial(model, mg, sensory_groups=sg, rig="ground", duration_s=DURATION_S,
                    seed=REFERENCE_REPLICATE, adapter=default_params())
    n = res.terminated_at_step or res.n_steps_planned
    rhythm = leg_rhythm(res.motor_rates[:, :n], NEURAL_DT)
    keep = int(TRANSIENT_S / NEURAL_DT)
    # Pad the pre-transient window with the first post-transient phase so
    # the reference has a value for every step of a fresh trial.
    phase = np.zeros((6, n))
    phase[:, keep:] = rhythm.phase
    phase[:, :keep] = rhythm.phase[:, :1]
    return phase, rhythm.rhythmic


def phase_error_current(phase_ref, active_legs, step, K, cpg_rows, n_neurons):
    """(n_neurons,) additive current for this step, from the precomputed
    reference phase -- open loop, does not read the live trial at all."""
    current = np.zeros(n_neurons)
    t = min(step, phase_ref.shape[1] - 1)
    for i, leg in enumerate(LEGS):
        if not active_legs[i] or leg not in cpg_rows:
            continue
        error = 0.0
        for j, other in enumerate(LEGS):
            if j == i or not active_legs[j]:
                continue
            target = 0.0 if (leg in TRIPOD_A) == (other in TRIPOD_A) else PREFERRED_PHASE_OFFSET
            error += np.sin(phase_ref[j, t] - phase_ref[i, t] - target)
        error /= max(sum(active_legs) - 1, 1)
        inj = float(np.clip(K * error, -CAP, CAP))
        current[cpg_rows[leg]] = inj
    return current


def run_with_injection(replicate, K, phase_ref, active_legs, cpg_rows):
    model, mg, sg, _wt, _info = build_trial_components(
        replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=DURATION_S,
        stim_current=MATCHED_DRIVE)
    n_neurons = model.neurons.n_neurons
    model.reset()
    rates_trace = []
    max_rate = 0.0
    unstable = False
    for step in range(int(DURATION_S / NEURAL_DT)):
        extra = (phase_error_current(phase_ref, active_legs, step, K, cpg_rows, n_neurons)
                if K != 0 else None)
        rates = model.step(NEURAL_DT, extra_input=extra)
        rates_trace.append(rates.copy())
        max_rate = max(max_rate, float(rates.max()))
        if not np.isfinite(rates).all() or max_rate >= 999.0:
            unstable = True
            break
    n_active = int((model.rates > 0.01).sum())
    return np.array(rates_trace), n_active, max_rate, unstable


def leg_motor_summed(rates_trace_tn, motor_groups):
    """rates_trace_tn: (T, n_neurons). Returns (6, T) per-leg summed rate."""
    from fly_robot.interface.motor_neuron_to_joint import leg_motor_row_indices
    rows = leg_motor_row_indices(motor_groups)
    out = np.zeros((6, rates_trace_tn.shape[0]))
    for i, leg in enumerate(LEGS):
        idxs = rows.get(leg, [])
        if idxs:
            out[i] = rates_trace_tn[:, idxs].sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Ks", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--out", default="media/trained_adapter/coupling_bound_sweep.json")
    args = ap.parse_args()

    print("building the reference phase trace (replicate "
          f"{REFERENCE_REPLICATE})...", flush=True)
    phase_ref, active_legs = build_reference_phase()
    print(f"  active (rhythmic) legs in reference: "
          f"{[f'{l[0]}-{l[1]}' for l,a in zip(LEGS, active_legs) if a]}\n")

    _m, mg, _sg, wt, _i = build_trial_components(
        replicate=REFERENCE_REPLICATE, param_seed=PILOT_PARAM_SEED, duration_s=0.1)
    cpg_groups_by_leg = build_cpg_groups("data/circuit_map/all_legs_circuit.csv", wt)
    cpg_rows = {leg: idxs for leg, idxs in cpg_groups_by_leg.items()}

    results = []
    print(f"{'K':>6}{'rep':>5}{'n_active':>10}{'max_fr':>9}{'unstable':>10}"
          f"{'n_rhythmic':>12}{'tripod_idx':>12}{'saturated':>11}")
    for K in args.Ks:
        for rep in SWEEP_REPLICATES:
            rates_trace, n_active, max_fr, unstable = run_with_injection(
                rep, K, phase_ref, active_legs, cpg_rows)
            motor_summed = leg_motor_summed(rates_trace, mg)
            # coordination() computes rhythm AND tripod_index in one pass;
            # replaces the earlier leg_rhythm()-only call so the positive
            # test (does K actually move interleg phase?) is answered
            # alongside the safety numbers already gathered, from the same
            # trials rather than a second run.
            rhythm, coord = coordination(motor_summed, NEURAL_DT)
            n_rhythmic = int(rhythm.rhythmic.sum())
            tripod_idx = float(coord.tripod_index) if coord.has_rhythm else None
            saturated = n_active > STABILITY_MAX_N_ACTIVE
            results.append({"K": K, "replicate": rep, "n_active": n_active,
                           "max_fr": max_fr, "unstable": unstable,
                           "n_rhythmic": n_rhythmic, "tripod_index": tripod_idx,
                           "saturated": saturated})
            tstr = f"{tripod_idx:+.3f}" if tripod_idx is not None else "  n/a"
            print(f"{K:>6.1f}{rep:>5}{n_active:>10}{max_fr:>9.1f}"
                  f"{str(unstable):>10}{n_rhythmic:>12}{tstr:>12}"
                  f"{str(saturated):>11}", flush=True)

    print("\n=== per-K summary (safety) ===")
    safe_bound = None
    for K in args.Ks:
        rows = [r for r in results if r["K"] == K]
        any_saturated = any(r["saturated"] or r["unstable"] for r in rows)
        median_rhythmic = float(np.median([r["n_rhythmic"] for r in rows]))
        ok = not any_saturated and median_rhythmic >= 2
        print(f"  K={K:5.1f}: any_saturated={any_saturated}  "
              f"median_n_rhythmic={median_rhythmic}  safe={ok}")
        if ok:
            safe_bound = K

    print(f"\nHighest safe K tested: {safe_bound}")
    if safe_bound is None:
        print("NO SAFE BOUND FOUND even at the smallest K tested -- "
              "per your instruction, fall back to Option C.")
    elif safe_bound == max(args.Ks):
        print("Safe at the LARGEST K tested -- the sweep did not find the "
              "ceiling; extend --Ks upward before committing this as the bound.")

    # --- POSITIVE TEST: does K actually move interleg phase at all? ------
    # Safety alone is not enough -- your point exactly: the 2.5 current
    # cap was calibrated for SENSORY neurons (threshold ~3.1), while CPG
    # neurons sit at threshold median ~35.6. "No saturation" could equally
    # mean "harmless because too weak to do anything." This is the direct
    # check: compare each K's tripod_index distribution against K=0's, on
    # the SAME replicates, so any shift is attributable to the coupling
    # rather than replicate-to-replicate variation.
    print("\n=== per-K summary (effect on tripod_index) ===")
    baseline_tripod = {r["replicate"]: r["tripod_index"]
                       for r in results if r["K"] == 0.0 and r["tripod_index"] is not None}
    print(f"  K=0.0 baseline tripod_index per replicate: "
          f"{ {k: round(v,3) for k,v in baseline_tripod.items()} }")
    any_effect = False
    for K in args.Ks:
        rows = [r for r in results if r["K"] == K and r["tripod_index"] is not None]
        if not rows:
            print(f"  K={K:5.1f}: no rhythmic trials to measure tripod_index on")
            continue
        deltas = [r["tripod_index"] - baseline_tripod[r["replicate"]]
                 for r in rows if r["replicate"] in baseline_tripod]
        mean_tripod = float(np.mean([r["tripod_index"] for r in rows]))
        mean_delta = float(np.mean(deltas)) if deltas else float("nan")
        print(f"  K={K:5.1f}: mean tripod_index={mean_tripod:+.4f}  "
              f"mean delta vs K=0={mean_delta:+.4f}  (n={len(rows)})")
        if K > 0 and deltas and abs(mean_delta) > 0.02:
            any_effect = True

    print(f"\nMEASURABLE EFFECT on tripod_index within K<=64 "
          f"(|mean delta| > 0.02 at any tested K): {any_effect}")
    if not any_effect:
        print("The proposed cap (2.5, Phase 4's SENSORY_CURRENT_CAP) shows "
              "NO measurable coordination effect at any tested K up to the "
              "proposed bound. Per your instruction: measure a CPG-specific "
              "current cap next -- largest current that shifts phase without "
              "saturation or rhythm loss. Not yet done in this run.")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference_replicate": REFERENCE_REPLICATE,
                   "sweep_replicates": list(SWEEP_REPLICATES),
                   "Ks": args.Ks, "results": results,
                   "safe_bound": safe_bound, "measurable_effect": any_effect}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
