"""Can a KNOWN-GOOD controller walk through the connectome's action space?

This is a feasibility gate, not a baseline, and it is deliberately cheap
(~seconds per seed) because it answers a question that otherwise costs
24,000 physics trials to guess at: **is the adapter's action space even
capable of walking?**

The adapter can only move 18 of the fly's 42 actuated leg DOFs — the three
pitch axes per leg that `ANTAGONIST_PAIRS` maps a motor-neuron antagonist
pair onto (coxa-pitch, trochanterfemur-pitch, tibia-pitch). The other 24
(roll/yaw axes, tarsus, femur reductor, and every DOF whose motor module is
unmapped or silent) are held at their neutral pose for the whole trial.
That is a property of the interface, not of the connectome.

So: take FlyGym's own CPG controller, which walks this body at 14.025 mm/s
with all 42 DOFs, and give it **only those same 18 DOFs**, holding the
other 24 at neutral exactly as the adapter does. Then read the result:

| outcome | what it means | what to do |
|---|---|---|
| restricted CPG walks | the 18-DOF action space is sufficient; a failure to walk is then about the connectome's output or the decoder | train R1a |
| restricted CPG cannot walk | the ACTION SPACE is the blocker; no amount of CMA-ES on the adapter can fix it | widen the interface (one justified DOF mapping at a time) before spending a search budget |

Either outcome is worth knowing and neither is a claim about the
connectome. **This controller is not connectome-driven and is never part
of any reported walking result** — it is a measuring instrument for the
body/interface, in the same spirit as `baselines/run_baselines.py`, and is
labelled as our own diagnostic addition (CLAUDE.md: anything not from the
fly is named as an addition).

Usage:
    MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
        -m fly_robot.baselines.restricted_action_cpg
    # add --video to render the restricted gait for inspection
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from flygym import Simulation
from flygym.compose.fly.base_fly import ActuatorType
from flygym_demo.complex_terrain.common import apply_locomotion_action
from flygym_demo.complex_terrain.cpg_controller import (
    CPGController, make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

from fly_robot.bodies.neuromechfly import build_free_fly
from fly_robot.interface.motor_neuron_to_joint import (
    ANTAGONIST_PAIRS, LEG_NAME_TO_FLYGYM_PREFIX, _dof_name_for_pair,
)

DURATION_S = 2.0
SEEDS = (0, 1, 2)


def adapter_controllable_dof_names() -> list[str]:
    """Exactly the DOFs `adapter_joint_targets` can move — derived from the
    interface's own tables, never hand-listed, so this cannot drift out of
    sync with the decoder it is supposed to mirror."""
    names = []
    for (_segment, _side), leg_prefix in LEG_NAME_TO_FLYGYM_PREFIX.items():
        for _pos, _neg, suffix in ANTAGONIST_PAIRS:
            names.append(_dof_name_for_pair(leg_prefix, suffix))
    return names


def run(restrict: bool, duration_s: float = DURATION_S, seed: int = 0,
        video_paths: dict | None = None) -> dict:
    """One CPG trial. `restrict=False` reproduces the unrestricted baseline
    in this same script, so the comparison is within-script and not against
    a number copied from another run."""
    fly, world, _m, _d, cam_side, cam_opp, cam_top = build_free_fly()
    sim = Simulation(world)
    if video_paths:
        by_name = {"side": cam_side, "opposite_side": cam_opp, "top_down": cam_top}
        targets = {by_name[k]: v for k, v in video_paths.items()}
        sim.set_renderer(list(targets.keys()))

    dof_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    controller = CPGController(
        cpg_network=make_tripod_cpg_network(sim.timestep, seed=seed),
        preprogrammed_steps=PreprogrammedSteps(), output_dof_order=dof_order)

    sim.reset()
    sim.warmup()

    # The pose unmapped DOFs are held at. The adapter holds them at the
    # body's neutral angles; here we take the controller's own first action
    # as that reference, which is the CPG's neutral stance for this body --
    # read from the controller rather than assumed, so the two conditions
    # differ ONLY in which DOFs are allowed to move afterwards.
    allowed = set(adapter_controllable_dof_names())
    # `dof_order` holds JointDOF objects, not strings; `.name` is the same
    # accessor the decoder itself uses to build `dof_index_by_name`
    # (motor_neuron_to_joint.py, closed_loop.py), so the two agree by
    # construction. Asserted below rather than trusted.
    keep = np.array([d.name in allowed for d in dof_order], dtype=bool)
    if int(keep.sum()) != len(allowed):
        raise RuntimeError(
            f"DOF name mismatch: {int(keep.sum())} of {len(allowed)} "
            "adapter-controllable DOFs found in the simulation's actuated "
            "order. A silent mismatch here would freeze every DOF and make "
            "the gate report 'cannot walk' for the wrong reason.")
    neutral = np.asarray(controller.step().joint_angles, dtype=float).copy()

    track, quats = [], []
    for i in range(int(duration_s / sim.timestep)):
        action = controller.step()
        if restrict:
            # Mask the JOINT ANGLES only; adhesion is left exactly as the CPG
            # commands it, so the DOF restriction is the single variable.
            angles = neutral.copy()
            angles[keep] = np.asarray(action.joint_angles, dtype=float)[keep]
            action = replace(action, joint_angles=angles)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        if video_paths:
            sim.render_as_needed()
        if i % 10 == 0:
            track.append(sim.get_body_positions(fly.name)[0].copy())
            quats.append(sim.get_body_rotations(fly.name)[0].copy())
    if video_paths:
        sim.renderer.save_video(targets)

    track, quats = np.array(track), np.array(quats)
    upright = 1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)
    return {"restricted": restrict, "seed": seed,
            "n_dofs_driven": int(keep.sum()) if restrict else int(len(dof_order)),
            "speed_mm_s": float(track[-1, 0] - track[0, 0]) / duration_s,
            "dx_mm": float(track[-1, 0] - track[0, 0]),
            "abs_dy_mm": abs(float(track[-1, 1] - track[0, 1])),
            "upright_min": float(upright.min()),
            "z_end": float(track[-1, 2])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration-s", type=float, default=DURATION_S)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--video", action="store_true",
                    help="render the restricted gait (seed 0) for inspection")
    ap.add_argument("--out", default="media/trained_adapter/restricted_action_cpg.json")
    args = ap.parse_args()

    allowed = adapter_controllable_dof_names()
    print(f"adapter-controllable DOFs: {len(allowed)}")
    for name in allowed[:3]:
        print(f"   e.g. {name}")

    rows = []
    for restrict in (False, True):
        label = "restricted (18 DOF)" if restrict else "unrestricted (all DOF)"
        for seed in args.seeds:
            row = run(restrict, duration_s=args.duration_s, seed=seed)
            rows.append(row)
            print(f"  {label:22s} seed {seed}: "
                  f"speed={row['speed_mm_s']:+7.3f} mm/s  "
                  f"|dy|={row['abs_dy_mm']:5.2f}  "
                  f"upright_min={row['upright_min']:.3f}  "
                  f"z_end={row['z_end']:.2f}  "
                  f"({row['n_dofs_driven']} DOFs driven)", flush=True)

    def speeds(restrict):
        return [r["speed_mm_s"] for r in rows if r["restricted"] is restrict]

    full, restricted = speeds(False), speeds(True)
    print(f"\nunrestricted: mean {np.mean(full):+.3f} mm/s "
          f"(sd {np.std(full):.3f}, n={len(full)})")
    print(f"restricted:   mean {np.mean(restricted):+.3f} mm/s "
          f"(sd {np.std(restricted):.3f}, n={len(restricted)})")

    # A deliberately blunt threshold: a third of the unrestricted speed, and
    # forward. This is a feasibility gate, not a performance measure -- the
    # question is "can this action space walk at all", so the bar is low on
    # purpose and stated before the numbers are read.
    threshold = max(0.33 * float(np.mean(full)), 1.0)
    verdict = float(np.mean(restricted)) >= threshold
    print(f"\ngate: restricted mean >= {threshold:.3f} mm/s "
          f"(1/3 of unrestricted, floor 1.0)")
    print(f"VERDICT: the 18-DOF action space "
          f"{'CAN' if verdict else 'CANNOT'} support walking")
    if not verdict:
        print("  -> the ACTION SPACE is the blocker, not the connectome.\n"
              "     Widen the motor interface before spending a CMA-ES budget.")
    else:
        print("  -> action space is sufficient; a walking failure in R1a is\n"
              "     attributable to the connectome's output or the decoder.")

    if args.video:
        Path("media/trained_adapter").mkdir(parents=True, exist_ok=True)
        print("\nrendering restricted gait (seed 0)...", flush=True)
        run(True, duration_s=args.duration_s, seed=args.seeds[0],
            video_paths={"top_down":
                         "media/trained_adapter/restricted_action_cpg_top.mp4"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"adapter_controllable_dofs": allowed,
                   "n_adapter_controllable": len(allowed),
                   "duration_s": args.duration_s, "rows": rows,
                   "unrestricted_mean_mm_s": float(np.mean(full)),
                   "restricted_mean_mm_s": float(np.mean(restricted)),
                   "gate_threshold_mm_s": threshold,
                   "action_space_can_walk": verdict}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
