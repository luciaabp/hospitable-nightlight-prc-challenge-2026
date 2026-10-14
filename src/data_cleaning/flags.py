"""Stage 1: flag the two departure quality problems.

- ``sin_plan_vuelo``: no Network Manager flight (empty ``FLIGHT_ID_mvt``); also visible in the ranking.
- ``calzos_sin_medir``: block time equals scheduled time, so the label is "delay + taxi-out"; training only.
"""

import pandas as pd

from .config import (
    BLOCK_TIME,
    DEPARTURE,
    FLAG_BLOCK_NOT_MEASURED,
    FLAG_NO_FLIGHT_PLAN,
    FLIGHT_ID,
    PHASE,
    SCHED_TIME,
    CleaningConfig,
)


def add_quality_flags(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    is_departure = df[PHASE] == DEPARTURE

    df[FLAG_NO_FLIGHT_PLAN] = df[FLIGHT_ID].isna()

    block_minus_sched = (df[BLOCK_TIME] - df[SCHED_TIME]).dt.total_seconds().abs()
    df[FLAG_BLOCK_NOT_MEASURED] = is_departure & (block_minus_sched <= config.block_sched_tolerance_sec)

    dep = df[is_departure]
    report = {
        "departures": int(len(dep)),
        "no_flight_plan": int(dep[FLAG_NO_FLIGHT_PLAN].sum()),
        "unmeasured_block": int(dep[FLAG_BLOCK_NOT_MEASURED].sum()),
        "both": int((dep[FLAG_NO_FLIGHT_PLAN] & dep[FLAG_BLOCK_NOT_MEASURED]).sum()),
    }
    return df, report
