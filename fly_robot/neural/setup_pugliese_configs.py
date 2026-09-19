"""Install our Hydra config additions into a freshly cloned Pugliese repo.

`external/Pugliese_cpg_2025/` is gitignored (their repo + data, ~375MB,
not ours to version — see CHANGELOG 2026-09-18). Our own two small config
additions (a paths file pointing at this project's layout, and an
experiment config for the full-VNC test) live here instead, tracked, and
get copied into place after cloning.

Usage (after `git clone https://github.com/smpuglie/Pugliese_cpg_2025.git
external/Pugliese_cpg_2025`):
    python -m fly_robot.neural.setup_pugliese_configs
"""

import shutil
from pathlib import Path

from fly_robot.neural.pugliese_paths import PROJECT_ROOT, PUGLIESE_REPO

THIS_DIR = Path(__file__).resolve().parent

FILES = {
    "paths_fly_robot.yaml": PUGLIESE_REPO / "configs" / "paths" / "fly_robot.yaml",
    "experiment_FullVNC_DNg100_Stim.yaml": PUGLIESE_REPO / "configs" / "experiment" / "FullVNC_DNg100_Stim.yaml",
}

if __name__ == "__main__":
    if not PUGLIESE_REPO.exists():
        raise SystemExit(
            f"{PUGLIESE_REPO} not found. Clone it first:\n"
            f"  git clone https://github.com/smpuglie/Pugliese_cpg_2025.git {PUGLIESE_REPO}"
        )
    for src_name, dest in FILES.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(THIS_DIR / "pugliese_extra_configs" / src_name, dest)
        print(f"Installed {dest}")
    print(f"\nSet FLY_ROBOT_REPO={PROJECT_ROOT} in your environment "
          f"(fly_robot.neural.pugliese_paths does this automatically on import) "
          f"before running simulations.")
