"""Load ``interim_nm`` and prepare the features as in the v26 training.

Rows follow the sorted monthly files; CatBoost reproduces the validated models bit for bit only in that order.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .adsb import FEATURES as ADSB_FEATURES
from .timeline import timeline_features
from .traffic import departures_taxiing
from .config import (BLOCK, CATEGORICAL, CONTEXT_DELAY, CONTEXT_FEATURES, CONTEXT_RUNWAY,
                     CONTEXT_RUNWAY_CATEGORICAL, FEATURES, FLIGHT, FORWARD_GROUPS, GROUPED_INBOUND_FEATURES, ID,
                     INBOUND_FEATURES,
                     LIRF_DESTINATION,
                     MISSING_CATEGORY, NUMERIC, QUALITY_COLUMNS, SCHED, STATE_ADSB_FEATURES, STATE_FEATURES, TARGET,
                     TIME, TIMELINE_COLUMNS, TIMELINE_FEATURES, WX_FEATURES, PipelineConfig)

DESTINATION = "ADES_mvt"  # flight destination (LIRF)
# TIMELINE_COLUMNS: the flight's own times (timeline.py).
TRAINING_COLUMNS = list(dict.fromkeys(FEATURES + QUALITY_COLUMNS + TIMELINE_COLUMNS
                                      + [ID, TIME, BLOCK, SCHED, TARGET, FLIGHT, DESTINATION]))
# EOBT_1_flt: date anchor of the block time for flights with NM (dayshift.py) and a timeline time.
RANKING_COLUMNS = list(dict.fromkeys(FEATURES + QUALITY_COLUMNS + TIMELINE_COLUMNS
                                     + [ID, TIME, SCHED, FLIGHT, "EOBT_1_flt", DESTINATION]))


def training_files(config: PipelineConfig) -> list[Path]:
    files = sorted(config.interim_dir.glob(config.training_pattern))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 months of 2025 in {config.interim_dir}, found {len(files)}")
    return files


def with_taxiing(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Replace ``departures_taxiing`` with the causal version."""
    if config.departures_taxiing == "batch":
        return frame
    if config.departures_taxiing != "causal":
        raise ValueError(f"Unknown departures_taxiing: {config.departures_taxiing}")
    frame = frame.copy()
    frame["departures_taxiing"] = departures_taxiing(frame)
    return frame


def read_training(config: PipelineConfig) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(p, columns=TRAINING_COLUMNS) for p in training_files(config)],
                      ignore_index=True)
    frame = with_taxiing(frame, config)
    frame = frame[frame[TARGET].notna()].reset_index(drop=True)
    if frame[ID].duplicated().any():
        raise ValueError("Repeated movement IDs in the training")
    return with_timeline(with_wx_features(with_context_features(with_adsb_features(frame, config), config), config),
                         config)


def read_ranking(config: PipelineConfig) -> pd.DataFrame:
    frame = pd.read_parquet(config.interim_dir / config.ranking_name, columns=RANKING_COLUMNS)
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Ranking IDs missing or repeated")
    frame = with_taxiing(frame, config)
    frame = with_adsb_features(frame.reset_index(drop=True), config)
    return with_timeline(with_wx_features(with_context_features(frame, config), config), config)


def with_timeline(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the flight's own times if a component uses them; row order unchanged."""
    if not config.timeline_components:
        return frame
    return pd.concat([frame, timeline_features(frame)], axis=1)


def with_adsb_features(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the ADS-B context by ID without changing the row order (CatBoost depends on it)."""
    if config.adsb_features is None:
        return frame
    context = pd.read_parquet(config.adsb_features, columns=[ID] + ADSB_FEATURES)
    missing = ~frame[ID].isin(context[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.adsb_features}: rebuild the context")
    return frame.merge(context, on=ID, how="left", validate="1:1")


def with_context_features(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the pre-takeoff context by ID without changing the row order."""
    if config.context_features is None:
        return frame
    context = pd.read_parquet(config.context_features, columns=[ID] + CONTEXT_FEATURES)
    missing = ~frame[ID].isin(context[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.context_features}: rebuild the context")
    return frame.merge(context, on=ID, how="left", validate="1:1")


def with_inbound_features(lirf: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the aircraft's previous arrival to the LIRF departures by ID, order unchanged."""
    if config.lirf_inbound_features is None:
        return lirf
    context = pd.read_parquet(config.lirf_inbound_features, columns=[ID] + INBOUND_FEATURES)
    missing = ~lirf[ID].isin(context[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} LIRF departures without a row in {config.lirf_inbound_features}: "
                         "rebuild with python -m pipeline.inbound prepare")
    return lirf.merge(context, on=ID, how="left", validate="1:1")


def with_grouped_inbound(other: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the aircraft's previous arrival to the nine-airport departures by ID, order unchanged."""
    if config.grouped_inbound_features is None:
        return other
    table = pd.read_parquet(config.grouped_inbound_features, columns=[ID] + GROUPED_INBOUND_FEATURES)
    missing = ~other[ID].isin(table[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.grouped_inbound_features}: rebuild with "
                         "python -m pipeline.inbound prepare --airports nine")
    return other.merge(table, on=ID, how="left", validate="1:1")


def state_columns(config: PipelineConfig) -> list[str]:
    """Runway queue columns; the ADS-B ones only with ADS-B context (they go to the ADS-B regressor)."""
    if config.state_features is None:
        return []
    adsb = config.adsb_features is not None and config.state_adsb
    return list(STATE_FEATURES) + (list(STATE_ADSB_FEATURES) if adsb else [])


def with_state(other: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the runway queue to the nine-airport departures by ID, order unchanged."""
    columns = state_columns(config)
    if not columns:
        return other
    table = pd.read_parquet(config.state_features, columns=[ID] + columns)
    missing = ~other[ID].isin(table[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.state_features}: rebuild with "
                         "python -m pipeline.state prepare")
    return other.merge(table, on=ID, how="left", validate="1:1")


def forward_columns(config: PipelineConfig) -> list[str]:
    """Batch-context columns; the ADS-B family only with ADS-B context."""
    if config.forward_features is None:
        return []
    columns = [c for group in config.forward_groups if group != "adsb" for c in FORWARD_GROUPS[group]]
    adsb = config.adsb_features is not None and "adsb" in config.forward_groups
    return columns + (list(FORWARD_GROUPS["adsb"]) if adsb else [])


def with_forward(other: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the batch context to the nine-airport departures by ID, order unchanged."""
    columns = forward_columns(config)
    if not columns:
        return other
    table = pd.read_parquet(config.forward_features, columns=[ID] + columns)
    missing = ~other[ID].isin(table[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.forward_features}: rebuild with "
                         "python -m pipeline.forward prepare")
    return other.merge(table, on=ID, how="left", validate="1:1")


def lirf_columns(config: PipelineConfig) -> list[str]:
    """LIRF numeric features added to the 36 of v26 (regressor and classifiers; the gate uses only the 36)."""
    inbound = list(INBOUND_FEATURES) if config.lirf_inbound_features is not None else []
    return inbound + (list(TIMELINE_FEATURES) if "lirf" in config.timeline_components else [])


def lirf_categorical(config: PipelineConfig) -> list[str]:
    """LIRF categorical features added to the 36: the destination."""
    return list(LIRF_DESTINATION) if config.lirf_destination else []


def with_destination(lirf: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Flight destination (ICAO airport and country): scheduled information, visible in the ranking."""
    if not config.lirf_destination:
        return lirf
    destination = lirf[DESTINATION].astype("string").str.strip().str.upper()
    return lirf.assign(ades=destination, ades_country=destination.str[:2])


def adsb_columns(config: PipelineConfig) -> list[str]:
    return list(ADSB_FEATURES) if config.adsb_features is not None else []


def with_wx_features(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add the extended METAR by ID without changing the row order."""
    if config.wx_features is None:
        return frame
    table = pd.read_parquet(config.wx_features, columns=[ID] + WX_FEATURES)
    missing = ~frame[ID].isin(table[ID])
    if missing.any():
        raise ValueError(f"{int(missing.sum())} departures without a row in {config.wx_features}: rebuild the extended METAR")
    return frame.merge(table, on=ID, how="left", validate="1:1")


def extra_columns(config: PipelineConfig) -> tuple[list[str], list[str]]:
    """Features added to the 36 of v26: (numeric, categorical)."""
    tail = (list(WX_FEATURES) if config.wx_features is not None else []) + (
        list(TIMELINE_FEATURES) if "grouped" in config.timeline_components else []) + (
        list(GROUPED_INBOUND_FEATURES) if config.grouped_inbound_features is not None else []) + state_columns(config) + (
        forward_columns(config))
    if config.context_features is None:
        return adsb_columns(config) + tail, []
    return adsb_columns(config) + CONTEXT_DELAY + CONTEXT_RUNWAY + tail, list(CONTEXT_RUNWAY_CATEGORICAL)


def features(frame: pd.DataFrame, extra: list[str] = (), extra_categorical: list[str] = ()) -> pd.DataFrame:
    """Categoricals as text with 'NA' (CatBoost rejects nulls); numerics as float; ``extra*`` appended at the end."""
    X = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL:
        X[column] = frame[column].astype(object).where(frame[column].notna(), MISSING_CATEGORY).astype(str)
    for column in [*NUMERIC, *extra]:
        X[column] = pd.to_numeric(frame[column].astype("Float64"), errors="coerce").astype(float)
    for column in extra_categorical:
        X[column] = frame[column].astype(object).where(frame[column].notna(), MISSING_CATEGORY).astype(str)
    return X


def month(frame: pd.DataFrame) -> np.ndarray:
    return frame[TIME].dt.strftime("%Y-%m").to_numpy()


def imputed_block(frame: pd.DataFrame, tolerance: float) -> np.ndarray:
    """Block time filled with the scheduled time: the label is delay + taxi."""
    return (frame[BLOCK] - frame[SCHED]).dt.total_seconds().abs().le(tolerance).to_numpy()


def day_shift(y: np.ndarray, imputed: np.ndarray) -> np.ndarray:
    """Candidates for a one-day shift; closeness to the schedule takes precedence."""
    residual = np.asarray(y, dtype=float) - 86400
    return ~np.asarray(imputed, dtype=bool) & (residual >= 0) & (residual <= 7200)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(config: PipelineConfig) -> dict:
    files = training_files(config) + [config.interim_dir / config.ranking_name, config.template]
    if config.adsb_features is not None:
        files.append(config.adsb_features)
    if config.context_features is not None:
        files.append(config.context_features)
    if config.lirf_inbound_features is not None:
        files.append(config.lirf_inbound_features)
    if config.wx_features is not None:
        files.append(config.wx_features)
    if config.grouped_inbound_features is not None:
        files.append(config.grouped_inbound_features)
    if config.state_features is not None:
        files.append(config.state_features)
    if config.forward_features is not None:
        files.append(config.forward_features)
    return {str(p): sha256(p) for p in files}
