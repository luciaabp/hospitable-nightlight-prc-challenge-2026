"""Aircraft: type, wake category and maximum takeoff weight."""

from pathlib import Path

import pandas as pd

SPECS_KEY = "AIRCRAFT_TYPE_flt"
SPECS_MTOW = "Max Take-Off Weight (kg)"

FEATURES = ["aircraft_type", "wake_category", "mtow", "aircraft_type_mismatch"]


def load_mtow(specs_path: Path) -> pd.Series:
    """MTOW (kg) by ICAO aircraft type code."""
    specs = pd.read_excel(specs_path)
    key = specs[SPECS_KEY].astype("string").str.strip().str.upper()
    return specs.assign(key=key).dropna(subset=["key"]).drop_duplicates("key").set_index("key")[SPECS_MTOW]


def add_aircraft_features(df: pd.DataFrame, specs_path: Path) -> pd.DataFrame:
    df = df.copy()
    type_mvt = df["AIRCRAFT_TYPE_mvt"].astype("string")
    type_flt = df["AIRCRAFT_TYPE_flt"].astype("string")
    df["aircraft_type"] = type_mvt.fillna(type_flt)
    # the type comes from the movement and the wake from the flight plan: if they disagree, the
    # combination may be inconsistent (audit of 2026-09-16)
    df["aircraft_type_mismatch"] = (type_mvt.notna() & type_flt.notna() & (type_mvt != type_flt)).fillna(False)
    df["wake_category"] = df["WK_TBL_CAT_flt"].astype("string")
    df["mtow"] = df["aircraft_type"].map(load_mtow(specs_path)).astype(float)
    return df
