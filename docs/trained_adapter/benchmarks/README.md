# Compute-gate benchmarks

The scripts that produced every measured number in `../DESIGN.md` §5, kept
so the gate is reproducible rather than asserted. They measure only; none
of them changes the model, and none is part of a trainer.

Run from the repo root with the venv, e.g.:

```bash
MUJOCO_GL=cgl PYTHONPATH=$PWD .venv/bin/python docs/trained_adapter/benchmarks/bench_components.py
```

| script | answers | key result (M3 Pro, 11 cores) |
|---|---|---|
| `bench_components.py` | where does a trial's time go? | neural 88.7%, physics 9.0%; component sum 24.7 s vs recorded pilot median 25.2 s |
| `reachability.py` | is there a lossless subnetwork crop? | no — only 846 / 23,532 (3.6%) provably inert |
| `bench_activeset.py` | does exploiting rate-vector sparsity help? | 11.8× on the matvec, difference exactly 0.0 |
| `bench_faststep.py` | full-step speedup, and the saturated worst case | 6.3× stable; 0.74× (slower) at full density → needs a density guard |
| `bench_parallel2.py` | how many CPU cores are worth using? | saturates at ~6 workers, 3.82× |
| `bench_endtoend.py` | does it hold through the real `run_trial`? | 33.1 s → 9.75 s (3.40×), outputs bit-identical |
| `bench_population_throughput.py` | trials/hour with N workers, and RAM per worker | 1,331 trials/hr at 10 workers; 0.57 GB steady/worker, 7.4 GB transient build peak |

`bench_population_throughput.py` is the one to re-run on a rented VM before
committing to a long training run — §5.6's 16- and 32-core figures are
extrapolated from an 11-core heterogeneous laptop and carry roughly a
factor-of-two error bar. It takes about ten minutes.

`bench_parallel2.py` writes a cached copy of `W_eff` and the settled rate
vector into the scratchpad on first run; delete those to regenerate.

Not measured here, and deliberately not estimated from these numbers:
batched rollout throughput on a GPU. See `../DESIGN.md` §5.5 for why, and
for the benchmark that would settle it.
