"""Does exploiting the SPARSITY OF THE RATE VECTOR speed up the step?

scipy's CSR matvec touches all 1.37M stored entries no matter how many
entries of R are zero. But in a real stable trial only ~400-540 of 23,532
neurons are ever above zero, and an exactly-zero rate contributes exactly
zero to W @ R. Gathering only the active COLUMNS of W (CSC) is therefore
algebraically identical, not an approximation.

Measured against a REAL rate vector taken from an actual open-loop run,
not a synthetic dense one.
"""
import time
import numpy as np
import scipy.sparse as sp

from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, build_trial_components

model, motor_groups, sensory_groups, wtable, info = build_trial_components(
    replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=4.0)
W = model.neurons.w_eff                 # CSR [post, pre]
Wc = sp.csc_matrix(W)                   # CSC: column j = presynaptic neuron j
n = W.shape[0]

# --- get REAL rate vectors by running the model open loop ---------------
print("running 1.0 s open loop to capture realistic rate vectors...")
model.reset()
snapshots = {}
for step in range(1000):
    r = model.step(0.001)
    if step + 1 in (100, 300, 600, 1000):
        snapshots[step + 1] = r.copy()

for k, r in snapshots.items():
    nz = int((r > 0).sum())
    print(f"  t={k/1000:.1f}s  nonzero rates: {nz} / {n} ({nz/n*100:.2f}%)  "
          f"max={r.max():.1f} Hz")

r = snapshots[1000]
active = np.flatnonzero(r)
print(f"\nusing t=1.0s vector: {len(active)} active of {n}")

# fraction of W's nonzeros actually in the active columns
cols_nnz = np.diff(Wc.indptr)
print(f"nnz in active columns: {cols_nnz[active].sum()} of {W.nnz} "
      f"({cols_nnz[active].sum()/W.nnz*100:.2f}%)")


def timeit(fn, n_rep):
    fn()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0) / n_rep


dense_s = timeit(lambda: W @ r, 200)


def active_matvec():
    idx = np.flatnonzero(r)
    return Wc[:, idx] @ r[idx]


act_s = timeit(active_matvec, 200)

# verify identical
v1 = W @ r
v2 = active_matvec()
print(f"\nmax abs difference between the two: {np.abs(v1 - v2).max():.3e} "
      f"(should be ~0 -- same arithmetic, fewer zero terms)")

print(f"\nCSR dense-vector matvec : {dense_s*1e3:7.3f} ms")
print(f"CSC active-column matvec: {act_s*1e3:7.3f} ms   "
      f"speedup {dense_s/act_s:.1f}x")
print(f"\n-> projected neural cost per 4 s trial: "
      f"{dense_s*4*4000:.1f} s  ->  {act_s*4*4000:.1f} s")
