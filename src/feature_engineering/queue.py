"""Queue: departure queue and takeoff rate of the aircraft's own runway.

Queue features look at the aircraft's off-block (``AOBT_3_flt``) and are empty without AOBT or when it is after
takeoff; rate features look at the window before its takeoff. Requires the normalized ``runway`` (``airport.py``).
"""

import numpy as np
import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"
OFFBLOCK_NM = "AOBT_3_flt"

WINDOW_10M = 10 * 60

FEATURES = [
    "queue_same_runway",            # departures of its runway already moving when it leaves the stand
    "offblocks_same_runway_prev_10m",  # off-blocks towards its runway in the previous 10 min
]

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _seconds(timestamps: pd.Series) -> np.ndarray:
    return (timestamps - _EPOCH).dt.total_seconds().to_numpy(dtype=float)


def add_queue_features(departures: pd.DataFrame) -> pd.DataFrame:
    df = departures.copy()
    result = {name: np.full(len(df), np.nan) for name in FEATURES}

    takeoff = _seconds(df[TAKEOFF])
    offblock = _seconds(df[OFFBLOCK_NM])
    # valid off-blocks: present and before takeoff
    valid = np.isfinite(offblock) & (takeoff > offblock)

    key = (df["airport"].astype(str) + ":" + df["runway"].astype(str)).where(df["runway"].notna())
    key = key.to_numpy(dtype=object)

    for runway in pd.unique(key[pd.notna(key)]):
        rows = np.flatnonzero(key == runway)
        t_off, t_take, ok = offblock[rows], takeoff[rows], valid[rows]

        # --- queue at off-block: left the stand earlier and not yet airborne
        starts = np.sort(t_off[ok])
        ends = np.sort(t_take[ok])
        on_surface = (np.searchsorted(starts, t_off, "right")
                         - np.searchsorted(ends, t_off, "right"))
        result["queue_same_runway"][rows] = np.where(ok, on_surface - 1, np.nan)  # excluding itself

        # --- off-blocks towards this runway in the 10 min before its own
        result["offblocks_same_runway_prev_10m"][rows] = np.where(
            ok,
            np.searchsorted(starts, t_off, "left") - np.searchsorted(starts, t_off - WINDOW_10M, "left"),
            np.nan)

    for name, values in result.items():
        df[name] = values
    return df
