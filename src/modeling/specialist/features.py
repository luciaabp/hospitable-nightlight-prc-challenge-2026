"""Feature matrix of one airport: blocks S (surface), T (stand) and C (common).

Training and ranking go through the same function, so their schemas cannot drift apart.

* S: ``interim_surface_ensemble`` columns listed in its manifest, minus month features.
* T: stand occupancy computed from raw departures and arrivals.
* C: ``COMMON`` columns from ``interim_nm``.

Departure BLOCK and taxi are read only to build the target and the imputed-block flag
(supervision); they never enter the matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CLOCK, COMMON, DELAY, ID, IMPUTED_TOLERANCE_SEC, MONTH_FEATURES, NM_DIR,
                     RAW_DIR, SURFACE_DIR, TARGET)
from .stand import stand_features

RAW_COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", CLOCK, "BLOCK_TIME_UTC_mvt",
               "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "STAND_mvt", "AIRCRAFT_TYPE_mvt",
               "AIRCRAFT_OPERATOR_flt", "FLIGHT_mvt"]


@dataclass
class AirportData:
    meta: pd.DataFrame            # ID, clock, month, delay, missing_nm (+ target when training)
    x: pd.DataFrame               # feature matrix, aligned with meta
    y: np.ndarray | None          # official taxi-out (training only)
    imputed: np.ndarray | None    # |BLOCK - SCHED| <= tolerance (training only)


def surface_columns(surface_dir: Path = SURFACE_DIR) -> list[str]:
    columns = json.loads((surface_dir / "manifest.json").read_text())["feature_columns"]
    return [c for c in columns if c not in MONTH_FEATURES]


def prepare_common(common: pd.DataFrame) -> pd.DataFrame:
    common = common[COMMON].copy()
    for column in COMMON:
        if pd.api.types.is_bool_dtype(common[column]) or pd.api.types.is_numeric_dtype(common[column]):
            # Nullable Int64/boolean columns carry <NA>, which CatBoost rejects.
            common[column] = common[column].astype("float64")
        else:
            common[column] = common[column].astype("string").fillna("Unknown").astype(str)
    return common


def _files(directory: Path, dataset: str) -> list[Path]:
    files = sorted(directory.glob("training_*.parquet")) if dataset == "training" else [directory / "ranking.parquet"]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"Missing {dataset} files in {directory}")
    return files


def load_airport(airport: str, dataset: str, surface_dir: Path = SURFACE_DIR, nm_dir: Path = NM_DIR,
                 raw_dir: Path = RAW_DIR) -> AirportData:
    """``dataset`` is ``"training"`` (all of 2025) or ``"ranking"``."""
    if dataset not in ("training", "ranking"):
        raise ValueError(dataset)
    training = dataset == "training"
    surface = pd.concat([pd.read_parquet(p, filters=[("ADEP_mvt", "==", airport)])
                         for p in _files(surface_dir, dataset)], ignore_index=True)
    if not surface[ID].is_unique:
        raise ValueError(f"{airport}: duplicated IDs in surface features")
    raw_columns = RAW_COLUMNS + ([TARGET] if training else [])
    raw = pd.concat([pd.read_parquet(p, columns=raw_columns,
                                     filters=[[("ADEP_mvt", "==", airport)], [("ADES_mvt", "==", airport)]])
                     for p in _files(raw_dir, dataset)], ignore_index=True)
    departures = raw[raw.PHASE_mvt.eq("DEP") & raw.ADEP_mvt.eq(airport)].set_index(ID)
    if not departures.index.is_unique:
        raise ValueError(f"{airport}: duplicated departure IDs in raw data")
    departures = departures.loc[surface[ID]]
    arrivals = raw[raw.PHASE_mvt.eq("ARR") & raw.ADES_mvt.eq(airport)]
    supervision = ["BLOCK_TIME_UTC_mvt"] + ([TARGET] if training else [])
    stand = stand_features(departures.drop(columns=supervision).reset_index(),
                           arrivals.drop(columns=[TARGET] if training else []))
    extra = pd.concat([pd.read_parquet(p, columns=[ID, *COMMON], filters=[("ADEP_mvt", "==", airport)])
                       for p in _files(nm_dir, dataset)]).drop_duplicates(ID).set_index(ID)
    missing = ~surface[ID].isin(extra.index)
    if missing.any():
        raise ValueError(f"{airport}: {int(missing.sum())} departures missing from {nm_dir}")
    common = prepare_common(extra.reindex(surface[ID]).reset_index(drop=True))
    x = pd.concat([surface[surface_columns(surface_dir)], stand.reset_index(drop=True), common], axis=1)

    meta = surface[[ID, CLOCK]].copy()
    meta["month"] = meta[CLOCK].dt.strftime("%Y-%m")
    meta["delay"] = surface[DELAY].to_numpy(float)
    meta["missing_nm"] = departures["AOBT_3_flt"].isna().to_numpy()
    y = imputed = None
    if training:
        y = surface[TARGET].to_numpy(float)
        np.testing.assert_array_equal(y, departures[TARGET].to_numpy(float))
        meta[TARGET] = y
        gap = (departures.BLOCK_TIME_UTC_mvt - departures.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
        imputed = gap.le(IMPUTED_TOLERANCE_SEC).to_numpy()
    return AirportData(meta=meta, x=x, y=y, imputed=imputed)
