# VM pricing check — 2026-09-22

*Looked up against primary/near-primary sources at the time of writing.
Prices change; re-verify before actually renting. Target: 32 vCPU, >=32 GB
RAM, general-purpose CPU (no GPU needed — see DESIGN.md's compute-route
decision).*

## Quotes

| provider | instance | vCPU | RAM | $/hr (on-demand) | $/hr source |
|---|---|---:|---:|---:|---|
| **DigitalOcean** | CPU-Optimized Droplet, 32 vCPU | 32 | 64 GB | **$1.00** | digitalocean.com/pricing/droplets (primary, fetched directly) |
| **AWS EC2** | c7g.8xlarge (Graviton3, ARM) | 32 | 64 GB | **$1.16** | economize.cloud, which quotes AWS's own published on-demand rate for us-east-1 |
| **Hetzner Cloud** | CCX53 (dedicated vCPU, AMD) | 32 | 128 GB | **~$0.93** (€0.855, converted) | costgoat.com pricing calculator — **third-party, not fetched from hetzner.com directly** (Hetzner's own pricing page is JS-rendered and did not return numbers to a static fetch; no browser tool was available to load it interactively) |

## Recommendation: DigitalOcean CPU-Optimized, 32 vCPU / 64 GB — $1.00/hr

Cheapest of the three at a verified primary-source price, simplest billing
(per-second, no reserved-instance complexity), and the RAM (64 GB) is
comfortably above the measured requirement (steady-state ~18 GB for 32
workers at 0.57 GB each, `DESIGN.md` §5.6).

**Cost for Stage A1** (`RUN_PLAN.md`): ~170 core-hours total across the
real run and the wiring controls, at measured per-trial cost. At $1.00/hr
for 32 cores that is roughly **$1.00/hr x wall-hours**, and wall-hours
depend on measured efficiency (extrapolated 0.35-0.6, see DESIGN.md §5.6):

| efficiency | wall time | cost @ $1.00/hr |
|---|---:|---:|
| eta = 0.35 (this Mac's measured efficiency) | ~15 h | ~$15 |
| eta = 0.6 (homogeneous server, extrapolated) | ~9 h | ~$9 |

**This is an estimate, not a quote** — actual wall time depends on the real
per-core throughput on that specific instance, which is exactly what the
calibration run is for. Cost is low enough in either case that the
decision should be driven by wall time and reliability, not price.

## Caveats, stated rather than glossed

- **Hetzner's number is secondhand.** I could not load hetzner.com's own
  pricing table (JS-rendered, no browser tool available in this
  environment) and used a third-party calculator instead. If Hetzner is
  chosen, its price should be re-verified against hetzner.com directly
  before committing spend.
- **AWS's c7g uses Graviton3 (ARM64), not x86.** This project's dependency
  stack (jax, scipy, mujoco) has ARM wheels and should work, but this has
  **not been tested** on ARM. DigitalOcean and Hetzner are both x86,
  matching the development machine exactly.
- All three quotes are **on-demand** rates; spot/preemptible pricing would
  be substantially cheaper but risks losing an in-progress generation
  (checkpointing every generation limits the damage — see `RUN_PLAN.md` —
  but a mid-generation preemption still wastes that generation's compute).
- Prices are current as of this check (2026-09-22) and are not guaranteed
  to hold; re-verify before renting.
