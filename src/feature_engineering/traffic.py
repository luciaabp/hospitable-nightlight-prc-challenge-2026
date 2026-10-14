"""Traffic: surface congestion and runway load, from the flights of the same DataFrame.

Congestion uses only taxis measured with the Network Manager block time (``taxi_proxy_nm_sec`` from ``delays.py``);
block times after takeoff are discarded. Counts are incomplete in the first minutes of the period.
"""

import numpy as np
import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"
OFFBLOCK_NM = "AOBT_3_flt"
TAXI_PROXY = "taxi_proxy_nm_sec"  # NM block times only; empty without AOBT

WINDOW_30M = 30 * 60
WINDOW_60M = 60 * 60

FEATURES = [
    "departures_taxiing",
    "departures_prev_30m",
    "arrivals_prev_30m",
    "recent_taxi_proxy_mean_60m",
    "recent_taxi_proxy_count_60m",
    "runway_departures_prev_30m",
]

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _seconds(timestamps: pd.Series) -> np.ndarray:
    return (timestamps - _EPOCH).dt.total_seconds().to_numpy(dtype=float)


def _count_before(sorted_times: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """Number of instants in [t - window, t)."""
    return np.searchsorted(sorted_times, t, "left") - np.searchsorted(sorted_times, t - window, "left")


def _mean_before(times: np.ndarray, values: np.ndarray, t: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of ``values`` in [t - window, t), ignoring nulls."""
    order = np.argsort(times, kind="stable")
    sorted_times, sorted_values = times[order], values[order]
    valid = ~np.isnan(sorted_values)
    cum_sum = np.concatenate([[0.0], np.cumsum(np.where(valid, sorted_values, 0.0))])
    cum_count = np.concatenate([[0], np.cumsum(valid)])

    lo = np.searchsorted(sorted_times, t - window, "left")
    hi = np.searchsorted(sorted_times, t, "left")
    n = cum_count[hi] - cum_count[lo]
    return np.where(n > 0, (cum_sum[hi] - cum_sum[lo]) / np.maximum(n, 1), np.nan), n.astype(float)


def _taxiing_at_offblock(offblock: np.ndarray, takeoff: np.ndarray) -> np.ndarray:
    """Number of other departures on the surface (offblock <= t < takeoff) at each flight's off-block."""
    result = np.full(len(offblock), np.nan)
    on_surface = ~np.isnan(offblock) & (takeoff > offblock)

    starts, ends = np.sort(offblock[on_surface]), np.sort(takeoff[on_surface])
    t = offblock[on_surface]
    taxiing = np.searchsorted(starts, t, "right") - np.searchsorted(ends, t, "right")
    result[on_surface] = taxiing - 1  # excluding the flight itself
    return result


def add_traffic_features(departures: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """Add the features to ``departures`` using every movement (departures and arrivals)."""
    df = departures.copy()
    result = {name: np.full(len(df), np.nan) for name in FEATURES}

    dep_airport = df["ADEP_mvt"].astype(str).to_numpy()
    takeoff = _seconds(df[TAKEOFF])
    offblock = _seconds(df[OFFBLOCK_NM])
    # only measured, consistent taxis: no AOBT or block after takeoff do not count
    taxi_proxy = df[TAXI_PROXY].to_numpy(dtype=float)
    taxi_proxy = np.where(np.isfinite(taxi_proxy) & (taxi_proxy >= 0), taxi_proxy, np.nan)

    arrivals = movements[movements["PHASE_mvt"] == "ARR"]
    arr_airport = arrivals["ADES_mvt"].astype(str).to_numpy()
    arr_time = _seconds(arrivals[TAKEOFF])

    for icao in np.unique(dep_airport):
        rows = np.flatnonzero(dep_airport == icao)
        t = takeoff[rows]
        result["departures_taxiing"][rows] = _taxiing_at_offblock(offblock[rows], t)
        result["departures_prev_30m"][rows] = _count_before(np.sort(t), t, WINDOW_30M)
        result["arrivals_prev_30m"][rows] = _count_before(np.sort(arr_time[arr_airport == icao]), t, WINDOW_30M)
        mean_60m, count_60m = _mean_before(t, taxi_proxy[rows], t, WINDOW_60M)
        result["recent_taxi_proxy_mean_60m"][rows] = mean_60m
        result["recent_taxi_proxy_count_60m"][rows] = count_60m

    # uses the normalized runway (airport.py): an unknown runway has no runway traffic
    runway_key = (
        (df["ADEP_mvt"].astype(str) + ":" + df["runway"].astype(str))
        .where(df["runway"].notna())
        .to_numpy(dtype=object)
    )
    for key in pd.unique(runway_key[pd.notna(runway_key)]):
        rows = np.flatnonzero(runway_key == key)
        t = takeoff[rows]
        result["runway_departures_prev_30m"][rows] = _count_before(np.sort(t), t, WINDOW_30M)

    for name, values in result.items():
        df[name] = values
    for name in ["departures_prev_30m", "arrivals_prev_30m"]:
        df[name] = df[name].astype("int32")  # always defined
    return df
