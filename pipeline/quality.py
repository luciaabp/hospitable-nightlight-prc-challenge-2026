"""LIRF record-quality context: visible flags of other flights, no labels.

Two versions with the same 24 columns (smoothed share and support of 6 flags, in two scopes): ``causal`` (default),
only flights airborne before the flight itself; ``batch`` (v26), the whole fixed 2 h block and day, kept only to
reproduce v26/v46.
"""
import numpy as np
import pandas as pd

from .config import QUALITY_COLUMNS


def quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """The six per-movement flags behind the aggregates."""
    f = frame[QUALITY_COLUMNS].reset_index(drop=True)
    t = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)
    s = pd.to_datetime(f.SCHED_TIME_UTC_mvt, utc=True)
    a = pd.to_datetime(f.AOBT_3_flt, utc=True)
    flight = f.FLIGHT_mvt.fillna("").astype(str)
    return pd.DataFrame({
        "no_nm": f.CALLSIGN_flt.isna().astype(float),
        "aircraft_missing": f.AIRCRAFT_TYPE_mvt.isna().astype(float),
        "prefix_repeated": flight.str.match(r"^[A-Z]([A-Z]{2})\1").astype(float),
        "nm_incoherent": (a > t).astype(float).where(a.notna()),
        "schedule_previous_day": (s.dt.floor("D") < t.dt.floor("D")).astype(float).where(s.notna()),
        "schedule_minute": (s.dt.second == 0).astype(float).where(s.notna()),
    }, index=frame.index)


def context(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "causal":
        return quality_context_causal(frame)
    if mode == "batch":
        return quality_context(frame)
    raise ValueError(f"Unknown quality context: {mode}")


# scope: (keys, window in s)
CAUSAL_WINDOWS = {"airport_2h": (["airport"], 2 * 3600), "airline_day": (["airport", "airline"], 24 * 3600)}


def quality_context_causal(frame: pd.DataFrame) -> pd.DataFrame:
    """24 columns from the flights airborne in the window before the flight's own takeoff (excluding it)."""
    f = frame[QUALITY_COLUMNS].reset_index(drop=True)
    takeoff = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).astype("datetime64[ns, UTC]").array.asi8 / 1e9
    flags = quality_flags(frame).reset_index(drop=True)
    result = {}
    for name, (keys, window) in CAUSAL_WINDOWS.items():
        key = f[keys].astype(object).where(f[keys].notna(), "NA").astype(str).agg("|".join, axis=1)
        rates = {col: np.full(len(f), np.nan) for col in flags}
        supports = {col: np.zeros(len(f)) for col in flags}
        for rows in key.groupby(key).indices.values():
            order = rows[np.argsort(takeoff[rows], kind="stable")]
            times = takeoff[order]
            # [takeoff - window, takeoff): simultaneous takeoffs and the flight itself do not enter
            low, high = np.searchsorted(times, times - window, "left"), np.searchsorted(times, times, "left")
            for col in flags:
                values = flags[col].to_numpy(float)[order]
                valid = ~np.isnan(values)
                total = np.concatenate([[0.0], np.cumsum(np.where(valid, values, 0.0))])
                count = np.concatenate([[0], np.cumsum(valid)])
                n = count[high] - count[low]
                rates[col][order] = (total[high] - total[low] + 1) / (n + 2)
                supports[col][order] = np.log1p(n)
        for col in flags:
            result[f"quality_{name}_{col}"] = rates[col]
            result[f"quality_{name}_{col}_support"] = supports[col]
    return pd.DataFrame(result, index=frame.index)


def quality_context(frame: pd.DataFrame) -> pd.DataFrame:
    """24 columns: smoothed share and support of each flag by airport/2 h and airport/airline/day."""
    f = frame[QUALITY_COLUMNS].reset_index(drop=True)
    t = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)
    flags = quality_flags(frame).reset_index(drop=True)
    keys = pd.DataFrame({"airport": f.airport.fillna("NA").astype(str),
                         "bin": t.dt.floor("2h"), "day": t.dt.floor("D"),
                         "airline": f.airline.fillna("NA").astype(str)})
    result = {}
    for name, columns in [("airport_2h", ["airport", "bin"]), ("airline_day", ["airport", "airline", "day"])]:
        group = flags.groupby([keys[c] for c in columns], dropna=False)
        count = group.transform("count") - flags.notna().astype(int)
        total = group.transform("sum") - flags.fillna(0)
        # Neutral 0.5 prior with strength 2; the support lets the gate discount small groups.
        for col in flags:
            result[f"quality_{name}_{col}"] = (total[col] + 1) / (count[col] + 2)
            result[f"quality_{name}_{col}_support"] = np.log1p(count[col])
    return pd.DataFrame(result, index=frame.index)
