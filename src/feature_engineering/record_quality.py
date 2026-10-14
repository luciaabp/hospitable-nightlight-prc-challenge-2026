"""Batch-visible, label-free, leave-one-out record quality context.

Requires the complete movement batch (also at inference), not real-time arrivals.
Never reads BLOCK, TAXITIME or calzos_sin_medir. UTC bins never cross months.
"""
import numpy as np
import pandas as pd

VISIBLE_COLUMNS = ["airport", "airline", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                   "AOBT_3_flt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]


def record_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-movement marks the aggregates are built from, one row per flight.

    Kept apart because the gate needs the flight's own marks, while the base
    models only ever see their leave-one-out aggregates.
    """
    f = frame[VISIBLE_COLUMNS].reset_index(drop=True)
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


def record_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame[VISIBLE_COLUMNS].reset_index(drop=True)
    t = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)
    flags = record_quality_flags(frame).reset_index(drop=True)
    keys = pd.DataFrame({"airport": f.airport.fillna("NA").astype(str),
                         "bin": t.dt.floor("2h"), "day": t.dt.floor("D"),
                         "airline": f.airline.fillna("NA").astype(str)})
    result = {}
    for name, columns in [("airport_2h", ["airport", "bin"]),
                          ("airline_day", ["airport", "airline", "day"])]:
        group = flags.groupby([keys[c] for c in columns], dropna=False)
        count = group.transform("count") - flags.notna().astype(int)
        total = group.transform("sum") - flags.fillna(0)
        # Neutral 0.5 prior, strength 2; support lets the gate discount sparse bins.
        for col in flags:
            result[f"quality_{name}_{col}"] = (total[col] + 1) / (count[col] + 2)
            result[f"quality_{name}_{col}_support"] = np.log1p(count[col])
    return pd.DataFrame(result, index=frame.index)
