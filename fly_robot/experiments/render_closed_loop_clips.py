"""Render video clips of specific closed-loop pilot states, for the video.

Not a data-generating experiment — the numbers behind these clips are
already established in `docs/closed_loop/RESULTS.md`. This script exists
purely to LOOK at what those numbers correspond to physically, using the
same validated `run_trial` the pilot itself used (rendering is opt-in via
`video_paths` and does not touch the neural/physics path — see
`sim/closed_loop.py`).

Four real, labelled states from the pilot's own fine sweep (Amendment 1/2,
`deviation` encoder). Two different pilot replicates are used, on purpose:

  no_drive, stable  — pilot replicate 1, which per the fine-sweep data
                      (docs/closed_loop/LOG.md) stays stable across the
                      ENTIRE sweep including g_fb=4.5 (n_active 380-388
                      throughout). Good for a clean "nothing happens"
                      state, useless for showing the cliff.
  transition, seizure — pilot replicate 0, which per the same data
                      actually crosses it: n_active 522/543 at g_fb<=3.5,
                      then 4040/4629 at g_fb=4.0/4.5. This distinction
                      matters — an earlier version of this script used
                      replicate 1 for all four clips and "transition"/
                      "seizure" silently rendered two more copies of the
                      same stable state, because that replicate never
                      destabilizes in the tested range. Caught by
                      checking n_active against the pilot's own recorded
                      table before treating the clips as done.

  no_drive   — DNg100 stimulation itself set to 0. No descending command
               at all. The network does nothing; this is the cleanest
               "non-movement" — not a claim about the sensory pathway,
               just: nothing is driving the animal.
  stable     — g_fb=3.5, the last pre-cliff sweep level. Real
               connectome-driven walking motion (Phase 3's established
               open-loop pattern) with feedback active but demonstrably
               NOT doing anything (measured effect ~0.00025 vs g_fb=0,
               n_rhythmic=4/6, matching baseline) — the "negligible" half
               of the reported result.
  transition — g_fb=4.0, the level immediately after the cliff for THIS
               replicate (n_active 543->4040). Named "transition" but
               note the reported finding itself: there is no gradual
               ramp — the jump between the last stable level and this one
               IS the whole transition, which is part of what makes the
               result a bifurcation rather than a dose-response.
  seizure    — g_fb=4.5, solidly past the cliff (n_active 4629). Firing
               rates saturate near the practical ceiling observed in the
               pilot (~230 Hz), rhythm is gone — the "catastrophic" half
               of the reported result.

Each clip is rendered from three cameras (side, opposite-side, top-down —
top-down is the one that reliably shows leg movement without wing
occlusion, per docs/logs/2026-09-19.md).

Usage:
    MUJOCO_GL=cgl python -m fly_robot.experiments.render_closed_loop_clips
"""

import argparse
from pathlib import Path

from fly_robot.sim.closed_loop import run_trial
from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

# replicate 1 stays stable through the whole sweep (good for no_drive/stable);
# replicate 0 actually crosses the cliff (needed for transition/seizure) —
# see the module docstring for why these are NOT interchangeable.
CLIPS = {
    "no_drive": dict(feedback_gain=0.0, use_sensory=False, stim_current=0.0, replicate=1),
    "stable": dict(feedback_gain=3.5, use_sensory=True, stim_current=380.0, replicate=1),
    "transition": dict(feedback_gain=4.0, use_sensory=True, stim_current=380.0, replicate=0),
    "seizure": dict(feedback_gain=4.5, use_sensory=True, stim_current=380.0, replicate=0),
}
DURATION_S = 4.0
ENCODER_MODE = "deviation"


def render(out_dir: Path, duration_s: float = DURATION_S):
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in CLIPS.items():
        replicate = cfg["replicate"]
        print(f"\n=== {name}: g_fb={cfg['feedback_gain']}, "
              f"stim_current={cfg['stim_current']}, replicate={replicate} ===", flush=True)
        model, motor_groups, sensory_groups, _wtable, info = build_trial_components(
            replicate=replicate, param_seed=PILOT_PARAM_SEED, duration_s=duration_s,
            stim_current=cfg["stim_current"],
        )
        video_paths = {
            "side": out_dir / f"{name}_side.mp4",
            "opposite_side": out_dir / f"{name}_opposite_side.mp4",
            "top_down": out_dir / f"{name}_top_down.mp4",
        }
        result = run_trial(
            model, motor_groups,
            sensory_groups=sensory_groups if cfg["use_sensory"] else None,
            feedback_gain=cfg["feedback_gain"], on_ball=True,
            duration_s=duration_s, seed=replicate, encoder_mode=ENCODER_MODE,
            video_paths=video_paths,
        )
        print(f"  n_active={result.n_active_neurons}  max_fr={result.max_firing_rate:.1f} Hz  "
              f"unstable={result.unstable}  wall={result.wall_clock_s:.1f}s")
        print(f"  wrote {list(video_paths.values())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("media/closed_loop/clips"))
    parser.add_argument("--duration", type=float, default=DURATION_S)
    args = parser.parse_args()
    render(args.out_dir, args.duration)
