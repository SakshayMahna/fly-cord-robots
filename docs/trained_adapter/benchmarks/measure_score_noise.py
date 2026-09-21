"""How noisy is a candidate's score, before any training exists?

Two things depend on this number and neither can be done honestly without
it:

  1. **Reward scaling.** Weighting terms that have wildly different natural
     spreads is how a reward ends up secretly optimising one term. Each
     term's empirical SD is what puts them on comparable footing.
  2. **Episode count.** An ES search can only resolve fitness differences
     bigger than the noise in its own fitness estimates. Averaging E
     episodes shrinks that noise as sigma/sqrt(E), so "how many episodes per
     candidate" is a calculation, not a preference — and a NEGATIVE result
     at too few episodes says nothing about the hypothesis.

The adapter does not exist yet, so this measures the *observables the reward
will be built from*, at the frozen Phase 4 defaults. That is the right
reference point: it is the untrained starting configuration.

Variance is decomposed, because the two sources call for different things:

  ACROSS-REPLICATE  different stochastic neuron-parameter draws. Phase 4
                    showed this is large (replicate 1 stayed stable across
                    the whole gain sweep while replicate 0 bifurcated), and
                    it is irreducible - it is what an episode should sample.
  WITHIN-REPLICATE  same neurons, perturbed initial joint state. If this is
                    small, episodes should vary the replicate rather than
                    the initial condition.

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        docs/trained_adapter/benchmarks/measure_score_noise.py
"""

import argparse
import json
import time

import numpy as np

from fly_robot.analysis.interleg_coordination import TRANSIENT_S, coordination
from fly_robot.sim.closed_loop import NEURAL_DT, run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

TRIAL_S = 4.0
# Candidates only. The pool actually used is COMPUTED by
# `replicate_filter.resolve_pool` from each replicate's own adapter-off
# baseline. The first version of this script hard-coded "exclude 2",
# inherited from Phase 4, and so measured replicate 5 — whose baseline
# n_active is 3,794 — as though it were a normal draw. That is what made
# the across-replicate SD 0.176 instead of 0.0066, and the trainer then
# inherited the same bad pool.
CANDIDATE_REPLICATES = tuple(range(8))
JOINT_NOISE_RAD = 0.02


def observables(result) -> dict:
    """Every quantity the reward in DESIGN.md §3 is built from."""
    keep = int(TRANSIENT_S / NEURAL_DT)
    angles = result.joint_angles[keep:]
    rhythm, coord = coordination(result.motor_rates, NEURAL_DT)

    out = {
        "n_active": float(result.n_active_neurons),
        "max_fr": float(result.max_firing_rate),
        "n_rhythmic": float(rhythm.rhythmic.sum()),
        "tripod": float(coord.tripod_index)
        if coord.has_rhythm and np.isfinite(coord.tripod_index) else np.nan,
        # posture: how far joints sit from their own settled pose
        "joint_excursion": float(np.abs(angles - angles.mean(axis=0)).mean()),
        # energy proxy: squared per-step joint motion
        "energy": float((np.diff(angles, axis=0) ** 2).sum(axis=1).mean()),
    }
    if result.ball_angvel is not None:
        av = result.ball_angvel[keep:]
        out["progress_pitch"] = float(np.mean(av[:, 1]))     # forward axis
        out["straightness_roll"] = float(np.mean(np.abs(av[:, 0])))
        out["straightness_yaw"] = float(np.mean(np.abs(av[:, 2])))
    return out


def run_arm(name, cases):
    rows = []
    for label, replicate, seed, noise in cases:
        model, mg, sg, _wt, _info = build_trial_components(
            replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=TRIAL_S)
        t0 = time.time()
        res = run_trial(model, mg, sensory_groups=sg, feedback_gain=0.0,
                        on_ball=True, duration_s=TRIAL_S, seed=seed,
                        encoder_mode="deviation", initial_joint_noise_rad=noise)
        obs = observables(res)
        obs.update(arm=name, label=label, replicate=replicate, seed=seed,
                   wall_s=round(time.time() - t0, 1))
        rows.append(obs)
        print(f"  {name:8s} {label:12s} pitch={obs.get('progress_pitch', float('nan')):+.4f} "
              f"rhythmic={obs['n_rhythmic']:.0f}/6 n_active={obs['n_active']:.0f} "
              f"({obs['wall_s']}s)", flush=True)
    return rows


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        out[k] = dict(n=len(v), mean=mean, sd=sd,
                      cv=abs(sd / mean) if mean != 0 else float("nan"),
                      lo=float(v.min()), hi=float(v.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media/trained_adapter/score_noise.json")
    args = ap.parse_args()

    from fly_robot.training.replicate_filter import resolve_pool
    print("resolving eligible replicates from adapter-off baselines...")
    pool, baselines = resolve_pool(CANDIDATE_REPLICATES, PILOT_PARAM_SEED,
                                   duration_s=TRIAL_S)
    print(f"  eligible pool: {pool}\n")

    print("ACROSS-REPLICATE arm (different neuron draws, identical everything else)")
    across = run_arm("across", [(f"rep{r}", r, 0, 0.0) for r in pool])

    print(f"\nWITHIN-REPLICATE arm (replicate {pool[0]}, perturbed start, "
          f"noise={JOINT_NOISE_RAD} rad)")
    within = run_arm("within",
                     [(f"seed{s}", pool[0], s, JOINT_NOISE_RAD) for s in range(6)])

    keys = ["progress_pitch", "straightness_roll", "straightness_yaw",
            "n_rhythmic", "tripod", "n_active", "joint_excursion", "energy"]
    sa, sw = summarise(across, keys), summarise(within, keys)

    print(f"\n{'observable':<20} {'across: mean':>13} {'sd':>10} {'cv':>7}   "
          f"{'within: mean':>13} {'sd':>10} {'cv':>7}")
    for k in keys:
        a, w = sa.get(k), sw.get(k)
        if a is None:
            continue
        ws = (f"{w['mean']:13.4f} {w['sd']:10.4f} {w['cv']:7.2f}" if w
              else " " * 32)
        print(f"{k:<20} {a['mean']:13.4f} {a['sd']:10.4f} {a['cv']:7.2f}   {ws}")

    print("\nepisodes needed to resolve an effect of size D (in SD units),")
    print("using SE = sd/sqrt(E) and requiring D >= 2*SE:")
    for d in (0.5, 1.0, 1.5, 2.0):
        print(f"   D = {d:.1f} sd  ->  E >= {int(np.ceil((2.0 / d) ** 2))}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(across=sa, within=sw, rows=across + within,
                       trial_s=TRIAL_S, 
                       joint_noise_rad=JOINT_NOISE_RAD,
                       eligible_pool=list(pool), baselines=baselines), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
