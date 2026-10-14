"""Causal ``departures_taxiing``: departures on the surface at off-block that took off **before**.

    departures_taxiing(B) = #{A != B at the airport: offblock_A <= offblock_B < takeoff_A < takeoff_B}

Off-block is ``AOBT_3_flt``; without it, or after takeoff, the feature is empty (as in v26).
"""
import numpy as np
import pandas as pd

from .config import TIME

OFFBLOCK = "AOBT_3_flt"
VECTOR_WINDOW = 256  # longer windows (taxis of hours) are counted row by row


def _seconds(values: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]")
    out = ts.array.asi8.astype(float) / 1e9
    out[ts.isna().to_numpy()] = np.nan
    return out


def departures_taxiing(frame: pd.DataFrame, causal: bool = True) -> np.ndarray:
    """Per batch departure; ``causal=False`` reproduces the v26 definition (to check it)."""
    airport = frame["airport"].astype(str).to_numpy()
    offblock, takeoff = _seconds(frame[OFFBLOCK]), _seconds(frame[TIME])
    on_surface = ~np.isnan(offblock) & ~np.isnan(takeoff) & (takeoff > offblock)
    result = np.full(len(frame), np.nan)
    for code in np.unique(airport):
        rows = np.flatnonzero((airport == code) & on_surface)
        if not len(rows):
            continue
        t, end = offblock[rows], takeoff[rows]
        if not causal:
            starts, ends = np.sort(t), np.sort(end)
            result[rows] = np.searchsorted(starts, t, "right") - np.searchsorted(ends, t, "right") - 1
            continue
        # departures with takeoff in (offblock_B, takeoff_B) and offblock <= offblock_B; B itself is out
        order = np.argsort(end, kind="stable")
        sorted_end, sorted_off = end[order], t[order]
        low = np.searchsorted(sorted_end, t, "right")
        high = np.searchsorted(sorted_end, end, "left")
        width = high - low
        count = np.zeros(len(rows))
        short = width <= VECTOR_WINDOW
        for k in range(min(VECTOR_WINDOW, int(width.max()) if len(width) else 0)):
            idx = low + k
            inside = short & (idx < high)
            count[inside] += sorted_off[np.minimum(idx, len(rows) - 1)][inside] <= t[inside]
        for i in np.flatnonzero(~short):
            count[i] = np.count_nonzero(sorted_off[low[i]:high[i]] <= t[i])
        result[rows] = count
    return result
