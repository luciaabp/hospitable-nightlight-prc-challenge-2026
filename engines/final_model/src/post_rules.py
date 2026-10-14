"""Frozen post-prediction rules used by the final ensemble."""
from __future__ import annotations
import numpy as np
import pandas as pd
import movement_features as F
MS = 'to_takeoff_from_SCHED_TIME_UTC_mvt'

ROME_INTERCEPT, ROME_SLOPE = (54310.76, 0.38)
ROME_MIN_SECONDS, ROME_MAX_SECONDS = (15 * 3600, 30 * 3600)

def rome_rows(df: pd.DataFrame) -> np.ndarray:
    ms = df[MS].to_numpy(float)
    return (df[F.AIRPORT].astype(str) == 'LIRF').to_numpy() & df['FLIGHT_ID_mvt'].isna().to_numpy() & (ms > ROME_MIN_SECONDS) & (ms <= ROME_MAX_SECONDS)

def rome_rule(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    ms = df[MS].to_numpy(float)
    sel = rome_rows(df)
    out = np.asarray(pred, float).copy()
    out[sel] = ROME_INTERCEPT + ROME_SLOPE * ms[sel]
    return out
RULES = {'roma': rome_rule}

def apply(df: pd.DataFrame, pred: np.ndarray, rules: list[str]) -> np.ndarray:
    for name in rules:
        if name not in RULES:
            raise ValueError(f'Unknown post-processing rule: {name}')
        pred = RULES[name](df, pred)
    return pred
