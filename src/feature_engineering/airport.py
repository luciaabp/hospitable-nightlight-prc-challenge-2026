"""Airport: airport, stand, runway and their combinations (stand and runway codes are local to each airport)."""

import pandas as pd

FEATURES = ["airport", "stand", "runway", "airport_stand", "airport_runway", "route", "runway_unknown"]

# "NA" and similar arrive as text in RUNWAY_mvt: unknown runway, not a real one.
# "H" and "HEL" are valid codes (helicopters) and are kept.
RUNWAY_PLACEHOLDERS = {"NA", "N/A", "UNKNOWN", "UNK", ""}


def _join(*parts: pd.Series) -> pd.Series:
    """Join with ':'; empty if any part is missing."""
    joined = parts[0].astype("string")
    for part in parts[1:]:
        joined = joined + ":" + part.astype("string")
    return joined


def add_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["airport"] = df["ADEP_mvt"].astype("string")
    df["stand"] = df["STAND_mvt"].astype("string")
    runway = df["RUNWAY_mvt"].astype("string").str.strip().str.upper()
    df["runway_unknown"] = runway.isin(RUNWAY_PLACEHOLDERS) | runway.isna()
    # an unknown runway stays empty, so it forms no fictitious runway in keys or traffic
    df["runway"] = runway.where(~runway.isin(RUNWAY_PLACEHOLDERS))
    df["airport_stand"] = _join(df["airport"], df["stand"])
    df["airport_runway"] = _join(df["airport"], df["runway"])
    df["route"] = _join(df["airport"], df["stand"], df["runway"])
    return df
