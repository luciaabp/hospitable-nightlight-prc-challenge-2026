"""Weather events from the EUROCONTROL Airport Corner (the only event type with a measurable effect in 2025).

Times are UTC; an end date without time covers the whole day; ``TBD`` and ``[On Hold / Cancelled]`` events are dropped.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"

FEATURES = ["event_adverse_weather", "event_winter_ops"]

# matched on the original English text, before the bracketed annotation
WEATHER = re.compile(
    r"weather|thunderstorm|snow|freezing|fog|winter|wind|showers|lvp|de-icing|storm|visibility|cloudbase",
    re.IGNORECASE,
)
CANCELLED = re.compile(r"on hold / cancelled", re.IGNORECASE)
# the winter subtype also reads the Spanish annotation: "Adverse Weather Conditions
# (Previsión de nieve ...)" is only recognised as snow through it
WINTER = re.compile(r"snow|freezing|winter|de-?icing|nieve|engelante|deshielo|invernal", re.IGNORECASE)


def _parse_date(value, is_end: bool) -> pd.Timestamp:
    value = str(value).strip()
    if value.upper() == "TBD":
        return pd.NaT
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        day = pd.to_datetime(value, format="%d/%m/%Y", utc=True)
        return day + pd.Timedelta(days=1) if is_end else day
    return pd.to_datetime(value, format="%d/%m/%Y %H:%M", utc=True)


def load_weather_events(events_path: Path) -> pd.DataFrame:
    """Valid weather events with their [start, end) interval in UTC."""
    events = pd.read_csv(events_path, encoding="utf-8-sig")
    original_text = events["Evento"].str.split(" (", n=1, regex=False).str[0]

    events["start"] = events["Start Date"].map(lambda v: _parse_date(v, is_end=False))
    events["end"] = events["End Date"].map(lambda v: _parse_date(v, is_end=True))
    events["winter_ops"] = events["Evento"].str.contains(WINTER)

    valid = (
        original_text.str.contains(WEATHER)
        & ~events["Evento"].str.contains(CANCELLED)
        & events["start"].notna()
        & events["end"].notna()
    )
    return events.loc[valid, ["ICAO", "Evento", "start", "end", "winter_ops"]].reset_index(drop=True)


def add_event_features(df: pd.DataFrame, events_path: Path) -> pd.DataFrame:
    df = df.copy()
    events = load_weather_events(events_path)

    airport = df["ADEP_mvt"].astype(str).to_numpy()
    takeoff = df[TAKEOFF]
    adverse_weather = np.zeros(len(df), dtype=bool)
    winter_ops = np.zeros(len(df), dtype=bool)

    for event in events.itertuples(index=False):
        inside = (
            (airport == event.ICAO)
            & (takeoff >= event.start).to_numpy()
            & (takeoff < event.end).to_numpy()
        )
        adverse_weather |= inside
        if event.winter_ops:
            winter_ops |= inside

    df["event_adverse_weather"] = adverse_weather.astype("int8")
    df["event_winter_ops"] = winter_ops.astype("int8")
    return df
