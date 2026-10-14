"""Paths, column names and small helpers shared by every step of the chain.

``ROOT`` is the folder holding this file: the checkout or, inside a run, ``RUN/code`` (``reproduce.py`` copies the
sources there and lays the inputs out under the same relative paths, see ``chain/runner.py``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Layout relative to ROOT (the same inside every run directory).
CHALLENGE_DIR = Path("data/projects/prc_data_challenge")
RAW_DIR = CHALLENGE_DIR / "raw"                  # training_2025-*.parquet, ranking, submitting, metar
SUBMISSIONS_DIR = CHALLENGE_DIR / "submissions"  # specialist v62 (submission_v62.parquet)
XPLANE_DIR = CHALLENGE_DIR / "xplane"            # X-Plane apt.dat files
ADSB_DIR = Path("adsb")                          # daily adsb.lol cuts and the ADS-B events built from them
VARIANTS_DIR = Path("variants")                  # base-model clones: base_reference, D2sub, FG4sub, FG4sub_s<k>
CANDIDATES_DIR = Path("outputs/final/candidates")

RAW = ROOT / RAW_DIR
SUBMISSIONS = ROOT / SUBMISSIONS_DIR
ADSB = ROOT / ADSB_DIR
VARIANTS = ROOT / VARIANTS_DIR
CANDIDATES = ROOT / CANDIDATES_DIR
LOCAL_DATA = ROOT / "local_data"
FG4_EVENTS = VARIANTS / "FG4sub/data/adsb/events.parquet"  # own ADS-B events of the final base model

ID, PRED = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
LOBT_WINDOW_SEC = 3606  # in 2025 |BLOCK - LOBT| <= 3606 s for every departure with LOBT
NINE = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM")  # the ten airports minus LIRF


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_ids(frame: pd.DataFrame, name: str = "frame") -> None:
    if frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError(f"Invalid or duplicate IDs: {name}")


def prediction(path: Path, ids: np.ndarray, column: str = PRED) -> np.ndarray:
    """``column`` of a prediction file in the order of ``ids`` (NaN where an ID is missing)."""
    return pd.read_parquet(path).set_index(ID)[column].reindex(ids).to_numpy(float)


def own_adsb(ids: np.ndarray) -> np.ndarray:
    """Rows of ``ids`` with an own ADS-B event (FG4 events)."""
    return np.isin(ids, pd.read_parquet(FG4_EVENTS, columns=[ID])[ID].to_numpy())


def fg4_seed_file(k: int) -> Path:
    """Base-model prediction of FG4 seed ``k`` (0 is FG4sub itself)."""
    folder = VARIANTS / ("FG4sub" if k == 0 else f"FG4sub_s{k}") / "submissions"
    return sorted(folder.glob("*_v37.parquet"))[0]
