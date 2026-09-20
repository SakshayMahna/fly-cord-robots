"""Is there a LOSSLESS crop of the simulated network?

Argument: the rate equation is dR/dt = (act - R)/tau with
act = max(fr_cap*tanh((a/fr_cap)*(total - threshold)), 0) and threshold > 0.
A neuron starting at R=0 whose total input is <= 0 has act = 0 and hence
dR/dt = 0 forever. So any neuron NOT forward-reachable (following pre->post
edges, any sign) from the set of externally-driven neurons is identically
zero for all time, and deleting it changes nothing at all.

Driven set = DNg100 (stimulation) U leg proprioceptors (sensory feedback).
This is NOT a modelling approximation and NOT a topology change to the
connectome -- it is declining to integrate rows that are provably 0.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from fly_robot.sim.trial_setup import PILOT_PARAM_SEED, DNG100_ROWS, build_trial_components
from fly_robot.interface.joint_to_sensory_neuron import build_sensory_groups

model, motor_groups, sensory_groups, wtable, info = build_trial_components(
    replicate=1, param_seed=PILOT_PARAM_SEED, duration_s=4.0)

W = model.neurons.w_eff            # [post, pre]
n = W.shape[0]
A = (W != 0).astype(np.int8)       # adjacency, post<-pre
A_fwd = sp.csr_matrix(A)           # row=post, col=pre  => A_fwd.T propagates pre->post

sensory_idx = sorted({i for idxs in sensory_groups.indices_by_group.values() for i in idxs})
motor_idx = sorted({i for idxs in motor_groups.indices_by_group.values() for i in idxs})
print(f"n={n}  DNg100={list(DNG100_ROWS)}  sensory={len(sensory_idx)}  motor={len(motor_idx)}")

def forward_closure(seed_idx):
    """Everything reachable following pre->post edges."""
    reached = np.zeros(n, dtype=bool)
    reached[seed_idx] = True
    frontier = reached.copy()
    hops = 0
    sizes = [reached.sum()]
    while frontier.any():
        # post neurons receiving from frontier: A[post, pre] with pre in frontier
        newly = (A_fwd @ frontier.astype(np.int8)) > 0
        newly &= ~reached
        reached |= newly
        frontier = newly
        hops += 1
        sizes.append(int(reached.sum()))
        if hops > 12:
            break
    return reached, sizes

for name, seeds in [
    ("DNg100 only", list(DNG100_ROWS)),
    ("DNg100 + leg proprioceptors", list(DNG100_ROWS) + sensory_idx),
]:
    reached, sizes = forward_closure(seeds)
    print(f"\nforward closure from {name}:")
    print("   cumulative reached by hop:", sizes)
    print(f"   TOTAL reachable = {reached.sum()} / {n} "
          f"({reached.sum()/n*100:.1f}%)   provably-inert = {n - reached.sum()}")
    mot = np.zeros(n, dtype=bool); mot[motor_idx] = True
    print(f"   leg motor neurons reachable: {(reached & mot).sum()} / {len(motor_idx)}")

# How many are actually recruited in a real stable trial? (upper bound on what matters)
print("\nfor comparison, recorded n_active in pilot trials: "
      "388-543 stable, 4040-4629 runaway (docs/closed_loop/RESULTS.md)")

# Rows with no outgoing edges at all / no incoming at all
outdeg = np.asarray((A_fwd != 0).sum(axis=0)).ravel()   # col sums = out-degree of pre
indeg = np.asarray((A_fwd != 0).sum(axis=1)).ravel()
print(f"neurons with zero in-degree: {(indeg==0).sum()}, "
      f"zero out-degree: {(outdeg==0).sum()}")
