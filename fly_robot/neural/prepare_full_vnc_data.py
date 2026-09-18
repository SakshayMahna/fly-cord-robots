"""Convert Pugliese's full-VNC MANC files into the exact formats their own
`load_W` / `load_wTable` functions accept (.npy and .csv), so we can run
their unmodified simulation code on the full six-leg network instead of
their T1-only front-leg network.

This is a pure format conversion — same neuron order, same synapse
weights, same connectome data. We are not touching connectivity values,
just re-saving to formats their loaders recognize (`load_W`/`load_wTable`
in `external/Pugliese_cpg_2025/src/utils/sim_utils.py` only accept
.npy/.csv, not .npz/.feather — even though, confusingly, their own real
production config references a .feather wPath directly; see CHANGELOG
2026-09-19. We don't know what code version they ran that with, so we
convert rather than guess).

Two source pairs are available in their repo's "manc full vnc data/"
folder — pass `--source` to pick:

- `20251006` (default): the file their ACTUAL Fig. 4 run used
  (confirmed from their real `run_config.yaml`, extracted from their
  Zenodo archive — see CHANGELOG 2026-09-19). 23,532 neurons, matching
  the paper's stated count exactly. Use this if you want a real chance of
  reproducing their result; combine with `--stim-i 380` in the experiment
  config (their actual verified value, not derivable from the paper text
  alone).
- `20260522_allSynapses`: a newer MANC snapshot (23,628 neurons) we tried
  first, before finding the config above. Lacks a synapse-count floor
  (74% of nonzero entries are below the 5-synapse floor Pugliese's
  Methods describe for T1) — this script applies that floor for it. Even
  with the floor and correct bilateral stimulation, our attempt with this
  file did not reproduce rhythmic motor output (see CHANGELOG); root
  cause not fully resolved. Kept available for anyone who wants to debug
  that further, not recommended as the default path to a working sim.

Usage:
    python -m fly_robot.neural.prepare_full_vnc_data \
        --pugliese-repo external/Pugliese_cpg_2025 \
        --source 20251006 \
        --out-dir data/pugliese_sim_cache
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    "20251006": {
        "wtable": "wTable_20251006.feather",
        "w": "W_20251006.feather",  # wide DataFrame: index=bodyId_pre, columns=bodyId_post
        "apply_synapse_floor": False,  # this is their actual verified file; leave as-is
    },
    "20260522_allSynapses": {
        "wtable": "wTable_20260522_allSynapses.feather",
        "w": "W_20260522_allSynapses.npz",  # pickled dense array, arr_0
        "apply_synapse_floor": True,
    },
}


def load_W_matrix(src_dir: Path, w_filename: str, wtable: pd.DataFrame) -> np.ndarray:
    path = src_dir / w_filename
    if path.suffix == ".npz":
        return np.load(path, allow_pickle=True)["arr_0"]
    elif path.suffix == ".feather":
        w_df = pd.read_feather(path)  # index (bodyId_pre) and columns (bodyId_post, int64) preserved by feather
        w_df.columns = w_df.columns.astype(np.int64)
        # Reindex explicitly to wtable's bodyId order rather than assuming
        # it already matches — cheap safety check, not a performance concern
        # at this size.
        w_df = w_df.reindex(index=wtable["bodyId"], columns=wtable["bodyId"])
        return w_df.to_numpy()
    else:
        raise ValueError(f"Don't know how to load {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pugliese-repo", required=True, type=Path)
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="20251006")
    parser.add_argument("--out-dir", default="data/pugliese_sim_cache", type=Path)
    args = parser.parse_args()

    src_dir = args.pugliese_repo / "data" / "manc full vnc data"
    spec = SOURCES[args.source]

    wtable = pd.read_feather(src_dir / spec["wtable"]).reset_index(drop=True)
    W = load_W_matrix(src_dir, spec["w"], wtable)

    assert W.shape == (len(wtable), len(wtable)), (
        f"W shape {W.shape} doesn't match wTable rows {len(wtable)} — "
        "row/column order assumption is wrong, do not proceed."
    )
    assert not np.isnan(W).any(), "NaNs in W after reindexing — bodyId mismatch between wTable and W"

    if spec["apply_synapse_floor"]:
        n_before = int((W != 0).sum())
        W[np.abs(W) < 5] = 0
        n_after = int((W != 0).sum())
        print(f"Applied synapse-count floor of 5: {n_before} -> {n_after} nonzero entries "
              f"({(1 - n_after / n_before) * 100:.1f}% removed)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    w_path = args.out_dir / "W_full_vnc.npy"
    df_path = args.out_dir / "wTable_full_vnc.csv"

    # float32 matches the simulation's actual runtime precision (the repo's
    # run_hydra.py sets jax_enable_x64=False), and roughly halves disk use
    # versus the source float64 — no precision is lost that the simulator
    # would have used anyway.
    np.save(w_path, W.astype(np.float32))
    wtable.to_csv(df_path)  # index=True (default) — load_wTable reads it back with index_col=0

    print(f"Source: {args.source}")
    print(f"Wrote {w_path} ({W.nbytes / 2**20:.1f} MB source -> {w_path.stat().st_size / 2**20:.1f} MB)")
    print(f"Wrote {df_path} ({len(wtable)} rows)")
