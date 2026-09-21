# Log — trained adapter (Phase 5)

Chronological, newest last. Kept separate from `docs/logs/` for the same
reason `docs/closed_loop/LOG.md` is: this is one continuous arc, not daily
findings. `DESIGN.md` is the current state; this is how it got there.

---

## 2026-09-20 — compute gate

Measured the gate before designing around it, and the headline inverted the
assumption the brief was built on: **the connectome is 88.7% of a trial's
cost and the body is 9.0%.** Any effort aimed at accelerating MuJoCo is
aimed at a tenth of the problem. The component timings sum to 24.7 s against
a recorded pilot median of 25.2 s across 172 trials, which is what makes the
breakdown trustworthy rather than indicative.

Three things followed from looking at that number properly.

**A lossless speedup, found by noticing a mismatch between two kinds of
sparsity.** W is 0.248% dense and stored sparsely, so that was already
exploited. What was not is that the *rate vector* is sparse too, and
dynamically so: only 471 of 23,532 neurons are ever nonzero in a stable
trial. scipy's CSR matvec touches all 1,372,404 stored entries regardless of
how many entries of R are zero. Gathering only the active columns touches
33,514 — 2.44% — and, because a term `w × 0.0` is exactly zero and the
surviving terms are visited in the same order, the result is identical to
the last bit. Measured end-to-end through the real `run_trial`: 33.1 s →
9.75 s, `max|diff| = 0.000e+00` on motor rates, joint angles and sensory
drive.

**But it inverts, which is the part worth remembering.** Speedup falls to
1.03× at 4,629 active neurons (the runaway regime) and to **0.74× — slower
than before** with all 23,532 active, because gathering columns costs more
than it saves once most columns are active. An optimisation that becomes a
pessimisation in exactly the regime the pilot already identified as
dangerous is a trap; hence the density guard, and hence two of the seven
tests are specifically about the fallback firing.

**No useful crop exists, and this was worth testing rather than assuming.**
A neuron that is not forward-reachable from the driven set can never leave
zero (at R=0 with no input, `act` is 0 and so is dR/dt), so deleting it
would be exactly lossless. That is only **846 of 23,532 — 3.6%**. The
network reaches 13,541 neurons within two hops. So "crop the network" is off
the table as a cheap win, and any *larger* crop would be a topology change,
which the honesty rules forbid.

**On the GPU question**, the brief asked for a Colab measurement that this
machine cannot produce, and rather than estimate one, recorded the
structural reasons a GPU looks wrong here: the loop is sequential in time,
batching the population would require batching physics, FlyGym 2.1.0 has
**zero** references to mjx, and 69 of the body's 70 geoms are meshes — MJX's
most limited path. Reported as reasons, with the missing measurement named
as missing.

---

## 2026-09-21 — decisions applied

Folder moved `phase5/` → `docs/trained_adapter/`; the first draft had used
the brief's path in violation of the project's own "no phase-numbered
names" rule, flagged at the time rather than silently chosen.

**Landed the active-column matvec** behind `active_column_matvec=True` with
`ACTIVE_COLUMN_NNZ_FRACTION = 0.35` as the guard. Tests assert
`np.array_equal`, not `assert_allclose` — a tolerance would pass even if the
optimisation were perturbing a trajectory, and `AUDIT.md` §4 already
established this system is phase-sensitive between any two numerically
distinct runs, so "close" would be exactly the wrong bar. Full suite 32
passed, including Phase 4's `g_fb = 0` bit-identity gate.

**Recovered the frozen motor interface's config hash.** Phase 4 recorded
`04be9dec181ec2e20…` in `AUDIT.md` §5 but the code that computed it was
never committed, so the hash was quotable but not checkable. Recovered the
serialisation by trying candidates against the recorded digest —
`json.dumps(cfg, sort_keys=True)` over the four documented fields — and the
regression test now recomputes it from the live module constants. A test
that restated the constant would have been worthless; this one actually
fails if someone edits a gain.

**Found a per-worker memory problem, and then that it was mostly
illusory.** Building trial components peaks at **7.39 GB** RSS and settles
at **3.29 GB**, because `neuron_params.W` is materialised dense (23,532² ×
4 B = 2.22 GB) before being converted to the 5.5 MB sparse copy the model
actually keeps. At 3.3 GB/worker, a 32-worker VM would need ~105 GB. But the
retained memory is held by JAX's caches, not by anything needed:
`jax.clear_caches()` plus deleting live arrays takes a worker to **0.55 GB**
and the model still steps correctly (441 active neurons after 200 steps, as
expected). That is a 6× reduction in the number that drives VM sizing. The
transient 7.4 GB build peak remains, so pool startup must be serialised —
or, better, workers should load a cached sparse matrix rather than each
rebuilding from the dense one.

**A debugging fix worth recording because it cost 43 minutes.** The
population-throughput benchmark hung: 43 minutes elapsed against an expected
~11, with the parent Python idle at 10 MB RSS and no output past the point
the first version had reached in under a minute.

**Suspected cause, not isolated:** `mp.Barrier` passed through
`Pool(initargs=...)`. macOS spawns rather than forks, and a raw `Barrier`
plausibly does not survive that path, whereas a raw `Lock` does — which
would explain why the hang appeared only once the barrier was added. Fixed
by switching to a `Manager()`-backed barrier and adding `timeout=900` to the
wait, after which it ran normally. That is consistent with the diagnosis but
does not prove it; the mechanism was never isolated, because the fix was
cheap and the measurement was the point.

One piece of the original reasoning was **wrong and is corrected here**:
"no worker processes" was cited as evidence, but it was an artifact of
`pgrep -f bench_population_throughput` — spawned workers run under a
`multiprocessing.spawn` command line and never match that pattern, so they
are invisible to it whether or not they exist. The real evidence was the
elapsed time and the silent output.

The
barrier was added in the first place because without it the measurement was
wrong, not just slow: builds are serialised behind a lock, so early workers
were running trials while later ones were still building, and the reported
"contention" was partly queue skew. The first, unsynchronised run showed
nproc=4 as *slower* than nproc=2, which is what prompted the re-measure.

**Throughput measured, and the metric choice matters.** 1,331 trials/hour at
10 workers; single trial ~8.2–8.7 s uncontended, against 25.2 s as the pilot
ran it. Throughput is reported **gated by the slowest worker**, not the
mean, because an ES generation is synchronous — the update needs every
candidate's fitness, so a generation costs what its slowest rollout costs.

Two artefacts are recorded in `DESIGN.md` §5.6 rather than smoothed away.
The 1-worker baseline was noisy (other work on the machine), which produces
an impossible 132% efficiency at 2 workers and a dip below it at 4; the
scaling column should not be read as a curve. And the M3 Pro is
**heterogeneous** — 6 performance cores, 5 efficiency — so workers on
E-cores set the pace under max-gating. That means the extrapolation to
homogeneous server cores is, unusually, more likely to be pessimistic than
optimistic. The 16- and 32-core figures still carry about a factor-of-two
error bar and want a calibration run on the real VM before any long booking.

Useful consequence of the memory finding: RAM sizing is set by the 0.57 GB
steady-state figure, not the 7.4 GB build peak — **provided** pool startup
is serialised. Thirty-two workers each rebuilding from the dense matrix
concurrently would want ~235 GB and would OOM. That is the most likely way
to waste a rented VM, so it is written into the design rather than left to
be discovered.

**Score noise measured, and it reframed the reward.** Ran the untrained
frozen defaults across the 7 pilot replicates to get the spread of every
observable the reward will be built from — needed both for scaling the
reward terms and for sizing the episode count.

The dominant result was not the spread itself but **where it comes from**.
One replicate (5) is saturated at `n_active = 3,794`, and it alone
contributes ~97% of the variance in the progress term: SD across all seven
replicates is 0.176, against **0.0066** once the pre-registered stability
filter (`n_active ≤ 1500`) is applied. **The filter is worth 27× in noise.**

And the mechanism matters more than the number: rep 5 produces **ball pitch
+0.471 rad/s, 30× any healthy replicate.** A seizing network thrashes and
spins the ball. So saturation is not merely scientifically wrong here — it
is *instrumentally attractive* to an optimiser looking for ball rotation.
That converted the saturation penalty from a precaution into the
largest-weighted term in `REWARD.md`, sized (−2.00, hinged at 1,500) to beat
the progress available that way by a measured ~6×.

**Episode count: the binding term is coordination, not progress.** Using
SE = σ/√E with Δ ≥ 2·SE on the filtered pool, progress needs only E ≥ 1 to
resolve 2% of walking, while the tripod index needs **E ≥ 6** to resolve a
+0.2 change and rhythmicity E ≥ 4 for one leg. Coordination is the noisy
quantity — which is unsurprising, since it is the thing the project is
actually asking about. Powered-run recommendation is therefore E = 6, with
the explicit caveat that this sizes *terminal-effect detection*, not
CMA-ES's ability to rank nearby candidates generation over generation; that
needs the within-generation spread, which does not exist until a search has
run, and is a specific deliverable of the lean pilot.

**A caveat recorded rather than glossed:** the within-replicate arm found
almost no variance (pitch varying in the 4th decimal, `n_active` identical
to the unit across 6 perturbed starts). That is substantially **by
construction** — the measurement ran at `g_fb = 0`, where there is no path
from body to neurons at all, so the neural trajectory *cannot* depend on
initial pose. It supports sampling the replicate rather than the initial
condition, but must be re-measured once the adapter runs a nonzero feedback
gain.

**Caught a sign hazard that would have been very hard to see later.**
`PREREGISTRATION.md` §2 records forward walking as **−0.95 rad/s** on the
pitch axis. So the reward's progress term has to be `−pitch`. With the sign
flipped, the trainer would learn to walk backwards and the scalar score
would read as success throughout. `REWARD.md` §2 makes this the first thing
in the document and requires a sign test — assert the synthetic tripod gait
scores positive and its time-reversal negative — rather than a comment.
Worth noting the untrained connectome sits at **+0.005 rad/s**, i.e.
essentially stationary and marginally *backward*.
