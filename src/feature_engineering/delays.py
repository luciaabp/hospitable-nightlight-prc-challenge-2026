"""Delays with respect to the Network Manager milestones and the schedule, in seconds."""

import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"

FEATURES = [
    "takeoff_minus_EOBT_1_flt",
    "takeoff_minus_IOBT_flt",
    "takeoff_minus_LOBT_flt",
    "AOBT_3_flt_minus_LOBT_flt",
    "LOBT_flt_minus_IOBT_flt",
    "EOBT_1_flt_minus_SCHED_TIME_UTC_mvt",
    "iobt_missing",
    "lobt_missing",
    "sched_missing",
    "AOBT_3_flt_minus_EOBT_1_flt",
    "takeoff_minus_SCHED_TIME_UTC_mvt",
    "takeoff_minus_AOBT_3_flt",
    "taxi_proxy_nm_sec",
    "aobt_missing",
    "aobt_after_takeoff",
]


def _seconds(end: pd.Series, start: pd.Series) -> pd.Series:
    return (end - start).dt.total_seconds()


def add_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["takeoff_minus_EOBT_1_flt"] = _seconds(df[TAKEOFF], df["EOBT_1_flt"])
    df["AOBT_3_flt_minus_EOBT_1_flt"] = _seconds(df["AOBT_3_flt"], df["EOBT_1_flt"])
    df["takeoff_minus_SCHED_TIME_UTC_mvt"] = _seconds(df[TAKEOFF], df["SCHED_TIME_UTC_mvt"])

    # the flight plan goes through IOBT (initial), LOBT (last planned) and EOBT (current estimate); their
    # differences tell how the departure kept slipping.
    df["takeoff_minus_IOBT_flt"] = _seconds(df[TAKEOFF], df["IOBT_flt"])
    df["takeoff_minus_LOBT_flt"] = _seconds(df[TAKEOFF], df["LOBT_flt"])
    df["AOBT_3_flt_minus_LOBT_flt"] = _seconds(df["AOBT_3_flt"], df["LOBT_flt"])
    df["LOBT_flt_minus_IOBT_flt"] = _seconds(df["LOBT_flt"], df["IOBT_flt"])
    df["EOBT_1_flt_minus_SCHED_TIME_UTC_mvt"] = _seconds(df["EOBT_1_flt"], df["SCHED_TIME_UTC_mvt"])
    df["iobt_missing"] = df["IOBT_flt"].isna()
    df["lobt_missing"] = df["LOBT_flt"].isna()
    df["sched_missing"] = df["SCHED_TIME_UTC_mvt"].isna()

    # taxi proxy measured with the Network Manager block time; empty without AOBT
    df["taxi_proxy_nm_sec"] = _seconds(df[TAKEOFF], df["AOBT_3_flt"])
    df["aobt_missing"] = df["AOBT_3_flt"].isna()
    # inconsistency: block time after takeoff (audit of 2026-09-16)
    df["aobt_after_takeoff"] = df["taxi_proxy_nm_sec"] < 0

    # mixed feature: without AOBT_3 it uses the scheduled time, so it can be taxi or delay.
    # Kept as a flight predictor, never as a congestion measure (see traffic.py).
    df["takeoff_minus_AOBT_3_flt"] = df["taxi_proxy_nm_sec"].fillna(df["takeoff_minus_SCHED_TIME_UTC_mvt"])
    return df
