"""Load the ``local_data/interim`` features and prepare them for CatBoost."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feature_engineering.aircraft_backfill import FLAG as AIRCRAFT_BACKFILL_FLAG
from src.feature_engineering.aircraft_backfill import backfill_aircraft, load_flight_list
from .config import ExperimentConfig

log = logging.getLogger(__name__)

ID_COLUMN = "MVT_ID_mvt"
TIME_COLUMN = "MVT_TIME_UTC_mvt"
BLOCK_COLUMN = "BLOCK_TIME_UTC_mvt"
SCHED_COLUMN = "SCHED_TIME_UTC_mvt"
AOBT_COLUMN = "AOBT_3_flt"  # Network Manager off-block: present even when the airport's block is imputed
UNMEASURED_BLOCK = "calzos_sin_medir"
# CatBoost rejects nulls in categorical features
MISSING_CATEGORY = "NA"


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # id, month, airport and flags for error analysis (never features)

    def subset(self, rows: np.ndarray) -> "Dataset":
        return Dataset(
            self.X.iloc[rows].reset_index(drop=True),
            self.y.iloc[rows].reset_index(drop=True),
            self.meta.iloc[rows].reset_index(drop=True),
        )


def prepare_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for column in config.categorical_features:
        X[column] = df[column].astype(object).where(df[column].notna(), MISSING_CATEGORY).astype(str)
    for column in config.numeric_features:
        X[column] = pd.to_numeric(df[column].astype("Float64"), errors="coerce").astype(float)
    return X


def apply_aircraft_backfill(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Identical in training and prediction; applied per airport after filtering airports."""
    if not config.aircraft_backfill_dir:
        return df
    airports = config.airports or sorted(df["airport"].dropna().astype(str).unique())
    flight_list = load_flight_list(config.aircraft_backfill_dir, airports)
    return backfill_aircraft(df, flight_list, config.aircraft_specs_path,
                             config.aircraft_backfill_tolerance)


def load_dataset(config: ExperimentConfig) -> Dataset:
    files = sorted(config.interim_dir.glob(config.file_pattern))
    if not files:
        raise FileNotFoundError(f"No '{config.file_pattern}' files in {config.interim_dir}")

    columns = list(dict.fromkeys(
        config.features + [config.target, ID_COLUMN, TIME_COLUMN, BLOCK_COLUMN, SCHED_COLUMN, AOBT_COLUMN,
                           "airport", "no_flight_plan", UNMEASURED_BLOCK]
    ))
    # The fill flag is not in the parquets: backfill_aircraft creates it below.
    columns = [c for c in columns if c != AIRCRAFT_BACKFILL_FLAG]
    df = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
    log.info("read %d files: %s rows", len(files), f"{len(df):,}")

    if config.airports:
        df = df[df["airport"].isin(config.airports)]
        log.info("filtered to airports %s: %s rows", config.airports, f"{len(df):,}")
        if df.empty:
            raise ValueError(f"No rows for airports {config.airports}")

    df = df[df[config.target].notna()]
    if config.exclude_unmeasured_block:
        df = df[~df[UNMEASURED_BLOCK].astype(bool)]
        log.info("rows with unmeasured block excluded: %s left", f"{len(df):,}")
    if config.max_rows is not None and len(df) > config.max_rows:
        df = df.sample(n=config.max_rows, random_state=config.seed)
        log.info("random subsample of %s rows", f"{len(df):,}")
    df = df.reset_index(drop=True)
    df = apply_aircraft_backfill(df, config)

    meta = pd.DataFrame({
        ID_COLUMN: df[ID_COLUMN].to_numpy(),
        "month": df[TIME_COLUMN].dt.strftime("%Y-%m").to_numpy(),
        "airport": df["airport"].astype(str).to_numpy(),
        "no_flight_plan": df["no_flight_plan"].astype(bool).to_numpy(),
        # only to train and evaluate the mixture: absent in the ranking and never features
        "unmeasured_block": df[UNMEASURED_BLOCK].astype(bool).to_numpy(),
        "block_sched_gap_sec": (df[BLOCK_COLUMN] - df[SCHED_COLUMN]).dt.total_seconds().abs().to_numpy(),
        # taxi measured with the NM off-block: an alternative label when the airport's block
        # is imputed (see run_mixture --taxi-label-from-aobt)
        "taxi_proxy_nm_sec": (df[TIME_COLUMN] - df[AOBT_COLUMN]).dt.total_seconds().to_numpy(),
    })
    return Dataset(prepare_features(df, config), df[config.target].astype(float), meta)
