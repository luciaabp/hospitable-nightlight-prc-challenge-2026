"""Imputed-block history per flight number in LIRF: share and log1p count of earlier departures of the same
``FLIGHT_mvt`` with imputed blocks. A training row only sees earlier months; the ranking sees all 2025. Only useful
with recent history (January 2026, not July).
"""
import numpy as np
import pandas as pd

from .config import FNHIST_FEATURES


def history_until(flight: np.ndarray, imputed: np.ndarray, periods: np.ndarray, cutoff) -> pd.DataFrame:
    """Share and count per flight number from the rows of months <= cutoff."""
    past = periods <= cutoff
    if not past.any():
        return pd.DataFrame(columns=["mean", "size"]).astype(float)
    table = pd.DataFrame({"fn": flight[past], "imp": imputed[past].astype(float)})
    return table.groupby("fn").imp.agg(["mean", "size"])


def apply_history(table: pd.DataFrame, flight: np.ndarray) -> pd.DataFrame:
    fn = pd.Series(flight)
    if len(table):
        rate = fn.map(table["mean"]).to_numpy(float)
        count = fn.map(table["size"]).fillna(0).to_numpy(float)
    else:
        rate, count = np.full(len(fn), np.nan), np.zeros(len(fn))
    return pd.DataFrame({FNHIST_FEATURES[0]: rate, FNHIST_FEATURES[1]: np.log1p(count)})


def asof_features(flight: np.ndarray, imputed: np.ndarray, periods: np.ndarray) -> pd.DataFrame:
    """Each row gets the history of the months strictly before its own."""
    result = pd.DataFrame(np.nan, index=np.arange(len(flight)), columns=FNHIST_FEATURES)
    for period in sorted(set(periods)):
        rows = np.flatnonzero(periods == period)
        table = history_until(flight, imputed, periods, period - 1)
        result.iloc[rows] = apply_history(table, flight[rows]).to_numpy()
    return result


def flight_keys(frame: pd.DataFrame) -> np.ndarray:
    return frame["FLIGHT_mvt"].fillna("NA").astype(str).to_numpy()


def periods_of(months: np.ndarray) -> np.ndarray:
    return pd.PeriodIndex(months, freq="M").to_numpy()
