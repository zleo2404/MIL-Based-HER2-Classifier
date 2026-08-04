"""Helpers to snapshot everything needed to reproduce a run later."""
import shutil
import subprocess
from pathlib import Path
from typing import Union

import yaml

from her2_mil.config import Config


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def _get_pip_freeze() -> str:
    try:
        return subprocess.check_output(["pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return "unavailable"


def save_run_metadata(run_dir: Union[str, Path], cfg: Config, config_path: str) -> None:
    """Save the exact config used, the resolved config (with defaults
    filled in), the git commit hash and `pip freeze` output alongside the
    run's outputs, so any run can be traced back to what produced it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config_used.yaml")

    with open(run_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    with open(run_dir / "environment.txt", "w") as f:
        f.write(f"git_commit: {_get_git_commit()}\n\n")
        f.write("pip freeze:\n")
        f.write(_get_pip_freeze())
