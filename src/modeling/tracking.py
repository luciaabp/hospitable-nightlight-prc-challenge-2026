"""Experiment log in ``outputs/experiments``: one ``<datetime>_<name>`` folder each plus a row in ``index.csv``."""

import json
import logging
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FILE = "index.csv"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    raise TypeError(f"No serializable: {type(value)}")


class ExperimentTracker:
    def __init__(self, experiments_dir: Path, name: str):
        self.experiments_dir = Path(experiments_dir)
        self.created_at = datetime.now()
        self.experiment_id = f"{self.created_at:%Y%m%d-%H%M%S}_{name}"
        self.root = self.experiments_dir / self.experiment_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.save_text("command.txt", shlex.join(sys.orig_argv) + "\n")

    def path(self, relative: str) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(self, relative: str, data) -> None:
        self.path(relative).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default))

    def save_csv(self, relative: str, frame: pd.DataFrame) -> None:
        frame.to_csv(self.path(relative), index=False)

    def save_parquet(self, relative: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.path(relative), index=False)

    def save_text(self, relative: str, text: str) -> None:
        self.path(relative).write_text(text)

    def save_figure(self, relative: str, figure) -> None:
        import matplotlib.pyplot as plt

        figure.savefig(self.path(relative), dpi=120, bbox_inches="tight")
        plt.close(figure)

    def attach_log_file(self) -> None:
        handler = logging.FileHandler(self.path("run.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)

    def append_to_index(self, row: dict) -> None:
        import fcntl

        index_path = self.experiments_dir / INDEX_FILE
        # lock: several parallel experiments may finish at once
        with open(self.experiments_dir / f"{INDEX_FILE}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            new_row = pd.DataFrame([row])
            if index_path.exists():
                new_row = pd.concat([pd.read_csv(index_path), new_row], ignore_index=True)
            new_row.to_csv(index_path, index=False)


def _run(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def environment_info() -> dict:
    import catboost
    import optuna
    import sklearn

    status = _run(["git", "status", "--porcelain"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "libraries": {
            "catboost": catboost.__version__,
            "optuna": optuna.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_uncommitted_files": len(status.splitlines()) if status else 0,
        "gpus": (_run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) or "").splitlines(),
    }
