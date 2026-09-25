# Runbook — R1a ground walking (v2)

Everything is implemented, tested and committed. This file is the complete
operating procedure: the commands to run, what healthy looks like, when to
stop, and the traps that have already bitten this project once each.

State at time of writing (2026-09-25): **nothing is running.** 101 tests
pass. Reward is `2026-09-25-ground-locomotion-gated`, hash `707b7c68…`.

---

## 1. Launch

```bash
cd /Users/sakshaymahna/Documents/fly-cord-robots
MUJOCO_GL=cgl PYTHONPATH="$PWD" nohup .venv/bin/python \
    -m fly_robot.experiments.train_adapter \
    --run-name r1a_ground_v2 --stage output --episodes 6 \
    --generations 250 --workers 6 --out-dir media/trained_adapter \
    > media/trained_adapter/r1a_ground_v2_stdout.log 2>&1 < /dev/null &
disown
```

**Verify the startup block before walking away.** It must read:

```
=== real / r1a_ground_v2 ===
reward config hash: 707b7c688908bd61...        <- NOT 0d648542 (ungated), NOT 88f681a4 (ball)
ground stage: output (40 active parameters; sensory=off)
  sign test PASSED
  eligible pool: [0, 1, 3, 4, 6, 7]
```

If the hash is anything else, **stop and investigate** — that is exactly
the check that was skipped for 70 generations of a run that turned out to
be optimising ball rotation (`GROUND_BRIDGE.md` §1).

Expect ~200 s/generation at 6 workers → **~14 h for 250 generations**.
Checkpoints every generation; safe to stop and resume.

## 2. Monitor

```bash
grep -E "^gen " media/trained_adapter/r1a_ground_v2_stdout.log | tail -10
```

Richer view, including the per-term breakdown that matters most:

```bash
.venv/bin/python -c "
import json, numpy as np
h = json.load(open('media/trained_adapter/r1a_ground_v2/history.json'))
print('generations:', len(h))
for r in h[-8:]:
    print(f\"gen {r['generation']:3d}: best={r['best']:+.4f} med={r['median']:+.4f} \"
          f\"prog={r.get('term_progress',0):+.5f} rhy={r.get('term_rhythmicity',0):+.4f} \"
          f\"sat={r.get('term_saturation',0):+.4f} term={r.get('frac_terminated',0):.2f}\")
"
```

### What healthy progress looks like

| signal | expected | meaning |
|---|---|---|
| `term_saturation` | should sit **much closer to 0** than run 1's −0.5585 | the narrowed drive bound is working |
| `frac_terminated` | falls from ~0.20 to <0.05 within ~25 gens | stability being solved (it did this in run 1) |
| **`term_progress`** | **must rise above ~0.001 and keep climbing** | **the one that decides the experiment** |
| `term_rhythmicity` | now *coupled* to progress — it can only rise if the fly moves | the gate is doing its job |

**The decisive signal is `term_progress`.** In run 1 it never left zero. If
it is still ~0 by generation 60–80 with the gate in place, that is a real
result, not a reason to keep waiting.

## 3. Inspect before believing any score

Standing rule in this project, and it is what caught run 1:

```bash
MUJOCO_GL=cgl PYTHONPATH="$PWD" .venv/bin/python \
    -m fly_robot.experiments.inspect_r1a_candidate \
    --checkpoint media/trained_adapter/r1a_ground_v2/checkpoint.pkl \
    --replicates 0 1 3
```

Writes `media/trained_adapter/r1a_best_r{rep}_{top,side}.mp4` and prints
real dx/dy/upright per replicate. **A score is not evidence until the video
has been looked at** — run 1's +0.3379 candidate was completely motionless.

> Note: this overwrites the run-1 videos at the same paths. Move them first
> if you want both:
> `mkdir -p media/trained_adapter/run1_videos && mv media/trained_adapter/r1a_best_r*.mp4 media/trained_adapter/run1_videos/`

## 4. Stop cleanly

**Wait for a generation boundary** (checkpoint is written *before* the
`gen N` line prints, so a freshly printed line means that generation is
safely saved).

```bash
# 1. current count
grep -cE "^gen " media/trained_adapter/r1a_ground_v2_stdout.log
# 2. wait for it to increase, then:
kill -TERM $(pgrep -f "run-name r1a_ground_v2")
sleep 4
# 3. THE TRAP: workers do NOT die with the parent.
ps -eo pid,%cpu | awk '$2+0>50{print $1, $2}'
# 4. kill those PIDs LITERALLY (variable expansion has failed here repeatedly)
kill -TERM <pid1> <pid2> ...
# 5. confirm idle
ps aux | grep -i "[m]ultiprocessing.spawn" | awk '{print $2, $3"%"}'
```

**Why step 3–4 matter:** Python's multiprocessing re-execs workers with a
generic `spawn_main` command line that does **not** contain
`train_adapter`, so `pkill -f train_adapter` misses them and they keep
running at ~100% CPU each, orphaned. This has happened on every stop so
far. Kill by literal PID — passing a shell variable has repeatedly produced
`illegal pid` in this environment.

## 5. Resume

Identical launch command. It auto-resumes from the last checkpoint and
prints `resuming from generation N`.

> Reading that line: the log is appended across runs, so
> `grep "resuming" | tail -1` can return a **stale** line from an earlier
> resume if the new one has not printed yet. Use `tail -1 <log>` and check
> the *last line of the file* instead.

A resume under a changed reward or stage is **refused**, by design — that
guard is why run 1's checkpoint cannot contaminate v2.

## 6. If `term_progress` is still flat by gen 60–80

Do not just keep running. In order of preference:

1. **Inspect the video** (§3) — is it motionless, marching in place, or
   moving and being scored badly? These imply different fixes.
2. Check whether `term_saturation` is still large. If so the drive bound
   needs narrowing further, or the fault is elsewhere in the 40 params.
3. If the fly moves its legs with real excursion but does not translate,
   that is the **coordination** failure this project predicted
   (`GROUND_BRIDGE.md` §7b: amplitude without interleg phase coupling makes
   it fall, not walk). That is the pre-registered trigger for **Rung 2**,
   the bounded interleg conductor — which is still blocked on its causal
   phase estimator failing validation on one leg (`RUNG2_DESIGN.md` §10).

A flat result here is a **publishable finding**, not a failure: it would
say the connectome's output alone, through a constrained interface, does
not produce locomotion without added coordination. Report it; do not tune
until it passes.

---

## What changed since run 1 (both committed)

1. **Locomotion gate on the rhythm terms.** `rhythmicity` and
   `coordination` are multiplied by `clip(max(0, dx)/1 mm, 0, 1)`. Run 1's
   best candidate scored +0.3379 while travelling −0.010 mm; it now scores
   −0.0210. Genuine walking keeps full credit. Gated on *forward* dx so it
   cannot be bought by walking backwards.
2. **`dng100_level` upper bound 600 → 400.** Measured: the saturation cliff
   is between drive 392 (`n_active` 577) and 395 (1,153), hitting 3,919 by
   400. Under the old bound, one sigma of CMA-ES sampling reached drive
   446 — far past the cliff — so ~47% of samples on that axis were spent
   on trials earning a −2.0 penalty. Now one sigma reaches 389.4 (safe).
   **The starting drive is unchanged**: z=0 still decodes to 382.8125
   exactly, and the pre-registered matched value is untouched.
