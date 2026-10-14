"""Leakage-safe row-wise features for the surface ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import AIRPORT_TIMEZONES, TARGET

ID = "MVT_ID_mvt"
CLOCK = "MVT_TIME_UTC_mvt"
CATEGORIES = [
    "ADEP_mvt", "ADES_FILED_flt", "RUNWAY_mvt", "STAND_mvt",
    "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
    "FLIGHT_RULE_mvt", "FLIGHT_mvt",
]
PLAN_TIMES = ["SCHED_TIME_UTC_mvt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "AOBT_3_flt"]
INPUTS = [ID, "PHASE_mvt", CLOCK, *CATEGORIES, *PLAN_TIMES]


def valid_targets(frame: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    return target.notna() & np.isfinite(target) & target.gt(0)


def departures(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[
        raw["PHASE_mvt"].eq("DEP") & raw["ADEP_mvt"].isin(AIRPORT_TIMEZONES)
    ].copy().reset_index(drop=True)
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Departure IDs must be non-null and unique")
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(INPUTS) - set(frame)
    if missing:
        raise ValueError(f"Missing challenge inputs: {sorted(missing)}")
    if not frame["PHASE_mvt"].eq("DEP").all() or not frame["ADEP_mvt"].isin(AIRPORT_TIMEZONES).all():
        raise ValueError("Only challenge departures supported")
    features = pd.DataFrame(index=frame.index)
    for column in CATEGORIES:
        features[column] = (
            frame[column].astype("string").fillna("Unknown").str.strip().str.upper().astype(str)
        )
    for name, columns in [
        ("airport_stand", ["ADEP_mvt", "STAND_mvt"]),
        ("airport_runway", ["ADEP_mvt", "RUNWAY_mvt"]),
        ("route", ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]),
    ]:
        features[name] = features[columns].agg(":".join, axis=1)
    times = {
        column: pd.to_datetime(frame[column], utc=True, errors="coerce")
        for column in [CLOCK, *PLAN_TIMES]
    }
    for column in PLAN_TIMES:
        features["takeoff_minus_" + column] = (times[CLOCK] - times[column]).dt.total_seconds()
        features[column + "_missing"] = times[column].isna().astype("int8")
    for first, second in [
        ("AOBT_3_flt", "LOBT_flt"),
        ("AOBT_3_flt", "EOBT_1_flt"),
        ("LOBT_flt", "IOBT_flt"),
        ("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
    ]:
        features[first + "_minus_" + second] = (times[first] - times[second]).dt.total_seconds()
    for airport, timezone in AIRPORT_TIMEZONES.items():
        indices = frame.index[frame["ADEP_mvt"].eq(airport)]
        for label, column in [("takeoff", CLOCK), ("scheduled", "SCHED_TIME_UTC_mvt")]:
            local = times[column].loc[indices].dt.tz_convert(timezone)
            features.loc[indices, label + "_hour"] = local.dt.hour + local.dt.minute / 60
            features.loc[indices, label + "_dow"] = local.dt.dayofweek
            features.loc[indices, label + "_month"] = local.dt.month
    block_proxy = (
        times["AOBT_3_flt"]
        .fillna(times["LOBT_flt"])
        .fillna(times["EOBT_1_flt"])
        .fillna(times["SCHED_TIME_UTC_mvt"])
        .fillna(times["IOBT_flt"])
    )
    features["taxi_proxy_sec"] = (times[CLOCK] - block_proxy).dt.total_seconds().fillna(900)
    return features
