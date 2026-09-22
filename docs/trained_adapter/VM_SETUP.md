# Compute setup — DigitalOcean VM, or local

*2026-09-22. Setup steps only — no training launches from this document.
You create and pay for the VM yourself; these are the exact commands to
run once it exists. Same run commands work locally on the M3 Pro.*

## 0. What's actually needed, checked rather than assumed

Two things are **not in git** and will not arrive via `git clone`:

| what | size | why it's missing | how to get it on the VM |
|---|---:|---|---|
| `data/circuit_map/all_legs_circuit.csv` | 88 KB | `data/` is entirely gitignored (connectome dumps, mostly too large to track) — but this specific file is small and essential; nothing regenerates it without re-running Phase 1's neuprint circuit extraction | **copy it, don't regenerate it** — `scp` from this machine |
| `external/Pugliese_cpg_2025` | 375 MB | `external/` is gitignored (their repo, MIT-licensed reuse) | `git clone` fresh on the VM, per `README.md` |

Confirmed by checking what training actually reads
(`sim/trial_setup.py`, `training/replicate_filter.py`): those are the only
two external dependencies. `data/replicate_eligibility/` is a cache and
regenerates itself; the pre-measured values are also saved at
`docs/trained_adapter/artifacts/replicate_baselines_seed641.json` if you
want to skip the wait.

**Training itself needs no GL/rendering context.** `_evaluate_one` in
`training/cma_trainer.py` never passes `video_paths` to `run_trial`, so no
renderer is created during training — confirmed by reading the call site,
not assumed. `MUJOCO_GL` therefore should not matter for the training run
itself. It does matter for the **inspection video** step afterward; see
§5.

## 1. Create the droplet

Web console (console.digitalocean.com), or `doctl` if you have it
installed:

```bash
doctl compute droplet create fly-adapter-train \
  --region nyc3 \
  --size c-32 \
  --image ubuntu-24-04-x64 \
  --ssh-keys <your-ssh-key-fingerprint>
```

`c-32` is the CPU-Optimized 32 vCPU / 64 GB plan, **$1.00/hr**, verified
directly against digitalocean.com
(`artifacts/vm_pricing_2026-09-22.md`). Any region is fine; pick one close
to you for lower SSH latency, it does not affect compute.

## 2. Persistent storage for checkpoints — survives the droplet, not just the run

The trainer checkpoints every generation
(`Trainer.save_checkpoint`, atomic write), but that checkpoint lives on
the droplet's own disk by default — destroying the droplet destroys it
too. Attach a **Volume** (DigitalOcean's block storage, billed and
managed separately from the droplet) and point `--out-dir` at it:

```bash
doctl compute volume create fly-checkpoints \
  --region nyc3 --size 20GiB \
  --fs-type ext4

doctl compute volume-action attach <volume-id> <droplet-id>
```

On the droplet:

```bash
sudo mkdir -p /mnt/checkpoints
sudo mount -o discard,defaults /dev/disk/by-id/scsi-0DO_Volume_fly-checkpoints /mnt/checkpoints
sudo chown $(whoami) /mnt/checkpoints
# persist across reboots
echo '/dev/disk/by-id/scsi-0DO_Volume_fly-checkpoints /mnt/checkpoints ext4 defaults,nofail,discard 0 0' \
  | sudo tee -a /etc/fstab
```

20 GiB is generous — a single run's `history.json` + `checkpoint.pkl` is
under 200 KB (measured: the pilot's 121-generation checkpoint was 156 KB).
The volume can be detached and reattached to a fresh droplet if this one
needs to be destroyed mid-run — the checkpoint survives independently.

## 3. System packages and Python

```bash
ssh root@<droplet-ip>

apt-get update
apt-get install -y python3.12 python3.12-venv python3-pip git build-essential \
  libglfw3 libglfw3-dev libgl1-mesa-dev libosmesa6-dev
```

(The `mesa`/`glfw` packages are for MuJoCo's headless rendering path,
needed for §5, not for training itself per §0.)

## 4. Clone, install, and bring across the one file that isn't in git

```bash
git clone https://github.com/SakshayMahna/fly-cord-robots.git
cd fly-cord-robots

git clone https://github.com/smpuglie/Pugliese_cpg_2025.git external/Pugliese_cpg_2025

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m fly_robot.neural.setup_pugliese_configs
```

From **this machine** (not the VM), copy the one essential untracked file:

```bash
scp data/circuit_map/all_legs_circuit.csv root@<droplet-ip>:~/fly-cord-robots/data/circuit_map/
```

(`data/circuit_map/` needs to exist first: `mkdir -p data/circuit_map` on
the VM before the `scp`, or `scp` will fail silently into the wrong
place.)

## 5. Calibration run — do this before launching training

Ten minutes, measures actual per-trial and per-worker throughput on this
specific machine rather than trusting the extrapolation from an 11-core
Mac (`DESIGN.md` §5.6 flags that extrapolation as carrying roughly a
factor-of-two error bar):

```bash
cd ~/fly-cord-robots
source .venv/bin/activate
PYTHONPATH=$PWD python docs/trained_adapter/benchmarks/bench_population_throughput.py \
  --trials 3 --procs 8 16 32
```

This exercises the full `run_trial` path with no rendering, so
`MUJOCO_GL` should not need to be set for it to succeed. **If it fails on
a GL-related error anyway**, that means FlyGym/MuJoCo initialises some GL
context even without rendering — not yet observed on this project but not
ruled out either. Try `MUJOCO_GL=osmesa` (the correct headless *software*
backend for a machine with no GPU; `MUJOCO_GL=egl`, used elsewhere in this
project's docs, needs an actual GPU and **will not work on this VM
shape**, which has none). Report back whichever way it goes — this is
exactly what the calibration run is for.

**Report the actual `trials/hr` at 8/16/32 workers before launching
anything further.**

## 6. Launch (after your approval, not from this document)

Run commands, made to work identically here or on the M3 Pro — only
`--workers` and `--out-dir` differ:

```bash
# On the DigitalOcean VM:
PYTHONPATH=$PWD MUJOCO_GL=osmesa nohup .venv/bin/python \
  -m fly_robot.experiments.train_adapter \
  --run-name rung1 --population 16 --episodes 6 --generations 250 \
  --workers 30 --out-dir /mnt/checkpoints \
  > /mnt/checkpoints/rung1_stdout.log 2>&1 &

# On the local M3 Pro:
MUJOCO_GL=cgl PYTHONPATH=$PWD nohup .venv/bin/python \
  -m fly_robot.experiments.train_adapter \
  --run-name rung1 --population 16 --episodes 6 --generations 250 \
  --workers 6 --out-dir media/trained_adapter \
  > media/trained_adapter/rung1_stdout.log 2>&1 &
```

`--workers 30` on a 32-vCPU box leaves headroom for the OS and the parent
process; adjust after the calibration run shows where throughput actually
plateaus (§5.6 measured this Mac's own throughput peaking around 8-10
workers before efficiency fell off — the VM's plateau point is unmeasured
and is part of what the calibration run establishes).

**Resuming** (after any interruption, on either machine) is the same
command with the same `--run-name` and `--out-dir` — the trainer finds the
checkpoint automatically and continues from the next generation
(`Trainer.run`, `resume=True` is the default; pass `--fresh` only to
deliberately discard a checkpoint).

## 7. Getting results back off the VM

```bash
scp -r root@<droplet-ip>:/mnt/checkpoints/rung1 media/trained_adapter/
```

Then run inspection **locally**, where `MUJOCO_GL=cgl` is already known to
work end-to-end (used throughout this project's video rendering so far):

```bash
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python \
  -m fly_robot.experiments.inspect_trained_adapter --run-name rung1
```

## 8. Shutting down without losing anything

The Volume (§2) persists independently of the droplet. Once results are
copied off (§7), the droplet can be destroyed; the volume can be kept
(small ongoing cost) or destroyed too once you're confident nothing more
is needed from it.

```bash
doctl compute droplet delete fly-adapter-train
# only after confirming results are safely copied off:
doctl compute volume delete fly-checkpoints
```
