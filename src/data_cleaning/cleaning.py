"""Stages 2-4: cleaning of the departure label (arrivals are never changed).

Stages 2 (drop) and 3 (cap) can be disabled in ``CleaningConfig``; for the challenge they should be, because the ranking
keeps those rows with their original label.
"""

import pandas as pd

from .config import (
    DEPARTURE,
    FLAG_BLOCK_NOT_MEASURED,
    FLAG_NO_FLIGHT_PLAN,
    PHASE,
    TARGET,
    TAXI_TIME,
    CleaningConfig,
)


def drop_unrecoverable(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    """Stage 2: drop departures without flight plan AND with an unmeasured block time."""
    unrecoverable = (
        (df[PHASE] == DEPARTURE) & df[FLAG_NO_FLIGHT_PLAN] & df[FLAG_BLOCK_NOT_MEASURED]
    )
    if not config.drop_unrecoverable:
        return df.copy(), {"enabled": False, "rows_dropped": 0,
                           "rows_that_would_be_dropped": int(unrecoverable.sum())}
    return df[~unrecoverable].copy(), {"enabled": True, "rows_dropped": int(unrecoverable.sum())}


def cap_unmeasured_block(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    """Stage 3: clean label and, if enabled, cap rows with an unmeasured block time at ``cap_quantile`` of sound rows."""
    df = df.copy()
    is_departure = df[PHASE] == DEPARTURE
    healthy = is_departure & ~df[FLAG_NO_FLIGHT_PLAN] & ~df[FLAG_BLOCK_NOT_MEASURED]
    cap_sec = float(df.loc[healthy, TAXI_TIME].quantile(config.cap_quantile))

    # the label exists only for departures; empty for arrivals
    df[TARGET] = df[TAXI_TIME].astype(float).where(is_departure)

    over_cap = df[FLAG_BLOCK_NOT_MEASURED] & (df[TARGET] > cap_sec)
    if not config.cap_unmeasured_block:
        return df, {"enabled": False, "cap_sec": round(cap_sec, 1), "rows_clipped": 0,
                    "rows_that_would_be_clipped": int(over_cap.sum())}

    df.loc[over_cap, TARGET] = cap_sec
    return df, {"enabled": True, "cap_sec": round(cap_sec, 1), "rows_clipped": int(over_cap.sum())}


def clip_negative_target(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    """Stage 4: raise negative values to ``min_target_sec`` (clock offsets of a few seconds)."""
    df = df.copy()
    below = df[TARGET] < config.min_target_sec
    report = {
        "rows_clipped": int(below.sum()),
        "original_min_value": float(df.loc[below, TARGET].min()) if below.any() else None,
    }
    df[TARGET] = df[TARGET].clip(lower=config.min_target_sec)
    return df, report
