"""Read the ``training_*`` files and save them under the same names."""

import logging
from pathlib import Path

import pandas as pd

from .config import SOURCE_FILE

log = logging.getLogger(__name__)


def list_training_files(raw_dir: Path, pattern: str) -> list[Path]:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{pattern}' files in {raw_dir}")
    return files


def load_training(files: list[Path]) -> pd.DataFrame:
    """Load every file into one DataFrame, remembering each row's source."""
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame[SOURCE_FILE] = path.name
        frames.append(frame)
        log.info("read %-42s %9s rows", path.name, f"{len(frame):,}")
    return pd.concat(frames, ignore_index=True)


def save_by_source_file(df: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Save each row to a file named like its source file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_per_file = {}
    for name, part in df.groupby(SOURCE_FILE, sort=True):
        part.drop(columns=SOURCE_FILE).to_parquet(output_dir / name, index=False)
        rows_per_file[name] = len(part)
        log.info("saved %-42s %9s rows", name, f"{len(part):,}")
    return rows_per_file
