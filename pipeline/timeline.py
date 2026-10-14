"""The flight's own times as explicit differences between its planned times, AOBT_3 and takeoff.

AOBT_3, EOBT_1, LOBT and IOBT against the schedule, plan revisions, takeoff - IOBT/LOBT, planned duration
ARVT_1 - EOBT_1, and the minutes/seconds patterns of AOBT_3 (NM often derives it from a standard taxi). Everything is
the flight's own and before its taxi, except AOBT_3 (accepted exception) and the takeoff (the label anchor).
"""
import numpy as np
import pandas as pd

from .config import SCHED, TIME, TIMELINE_COLUMNS, TIMELINE_FEATURES


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True)


def timeline_features(frame: pd.DataFrame) -> pd.DataFrame:
    """``TIMELINE_FEATURES`` per departure, on ``frame``'s index. Without NM, empty except the takeoff seconds."""
    t, s = _utc(frame[TIME]), _utc(frame[SCHED])
    lobt, iobt, eobt, arvt1, aobt = (_utc(frame[c]) for c in TIMELINE_COLUMNS)

    def seconds(a, b):
        return (a - b).dt.total_seconds().to_numpy(dtype=float)

    return pd.DataFrame({
        "tl_x_min": np.floor(seconds(t, aobt) / 60),
        "tl_takeoff_sec": t.dt.second.to_numpy(dtype=float),
        "tl_aobt_sched": seconds(aobt, s),
        "tl_eobt_sched": seconds(eobt, s),
        "tl_lobt_sched": seconds(lobt, s),
        "tl_iobt_sched": seconds(iobt, s),
        "tl_eobt_iobt": seconds(eobt, iobt),
        "tl_eobt_lobt": seconds(eobt, lobt),
        "tl_takeoff_lobt": seconds(t, lobt),
        "tl_takeoff_iobt": seconds(t, iobt),
        "tl_planned_duration": seconds(arvt1, eobt),
        "tl_aobt_minute_mod5": (aobt.dt.minute % 5).to_numpy(dtype=float),
    }, index=frame.index)[TIMELINE_FEATURES]
