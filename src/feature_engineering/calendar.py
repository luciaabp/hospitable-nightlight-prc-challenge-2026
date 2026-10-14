"""Calendar: local hour, weekday and month as sine and cosine.

Shares its name with the standard ``calendar`` module: run the package as a module (``python -m ...``).
"""

import numpy as np
import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"

TIMEZONES = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LTFM": "Europe/Istanbul",
    "LSZH": "Europe/Zurich",
}

FEATURES = ["sin_local_hour", "cos_local_hour", "sin_weekday", "cos_weekday", "sin_month", "cos_month"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = pd.Series(np.nan, index=df.index)
    weekday = pd.Series(np.nan, index=df.index)
    month = pd.Series(np.nan, index=df.index)

    for icao, timezone in TIMEZONES.items():
        mask = df["ADEP_mvt"] == icao
        if not mask.any():
            continue
        local = df.loc[mask, TAKEOFF].dt.tz_convert(timezone)
        hour.loc[mask] = local.dt.hour + local.dt.minute / 60
        weekday.loc[mask] = local.dt.dayofweek
        month.loc[mask] = local.dt.month

    for name, value, period in [("local_hour", hour, 24), ("weekday", weekday, 7), ("month", month - 1, 12)]:
        angle = 2 * np.pi * value / period
        df[f"sin_{name}"] = np.sin(angle)
        df[f"cos_{name}"] = np.cos(angle)
    return df
