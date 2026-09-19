"""Shared path/environment bootstrap for everything that touches
Pugliese's cloned repo (external/Pugliese_cpg_2025, gitignored — see
docs/logs/2026-09-19.md). Previously each module that needed this
(prepare_full_vnc_data.py, setup_pugliese_configs.py, and the since-split
single_simulation.py / replicate_ensemble.py / oscillation_scoring.py /
reproduce_rhythmic_output.py) recomputed PROJECT_ROOT/PUGLIESE_REPO
independently; centralized here instead.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUGLIESE_REPO = PROJECT_ROOT / "external" / "Pugliese_cpg_2025"

os.environ.setdefault("FLY_ROBOT_REPO", str(PROJECT_ROOT))
sys.path.insert(0, str(PUGLIESE_REPO))
