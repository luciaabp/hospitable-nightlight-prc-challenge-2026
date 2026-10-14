"""Weather: METAR features of the departure airport.

Each departure gets two reports, the last before its takeoff and the last before its scheduled time (at most 90 min
old). Takeoff is used instead of block time because ``BLOCK_TIME_UTC_mvt`` is empty in the ranking. Visibility
thresholds are in the rounded miles of the IEM CSV.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

TAKEOFF = "MVT_TIME_UTC_mvt"
SCHEDULED = "SCHED_TIME_UTC_mvt"
MAX_METAR_AGE = pd.Timedelta("90min")

COLD_C = 3.0
VISIBILITY_BINS_MI = [-np.inf, 0.62, 3.11, 6.21, np.inf]  # 1.000 m, 5.000 m y 9.999 m
VISIBILITY_LABELS = ["<1 km", "1-5 km", "5-10 km", "≥10 km"]
UNKNOWN_VISIBILITY = re.compile(r"(?:^|\s)/{4}(?:\s|$)")
LOW_VISIBILITY_MI = 3.11  # under 5 km

PRECIPITATION_OR_ICE = re.compile(r"RA|DZ|SN|SG|PL|GS|GR|UP|FZ")
PHENOMENA = {
    "rain": re.compile(r"RA|DZ"),
    "snow": re.compile(r"SN|SG"),
    "fog_mist": re.compile(r"FG|BR"),
    "thunder": re.compile(r"TS"),
    "freezing": re.compile(r"FZ"),
    "hail": re.compile(r"GR|GS|PL"),
    "showers": re.compile(r"SH"),
}

# features computed for each instant (takeoff and scheduled time)
MOMENTS = {"": TAKEOFF, "sched_": SCHEDULED}
PER_MOMENT = ["temperature_c", "dewpoint_c", "humidity_pct", "wind_kt", "gust_kt", "wind_dir_sin",
               "wind_dir_cos", "visibility_mi", "cloud_base_ft", "precip_mm", "dewpoint_spread_c",
               "metar_age_min", "metar_missing"] + list(PHENOMENA) + ["low_visibility", "cold_moist"]

FEATURES = (["deicing_risk", "visibility_band", "freezing_or_snow"]
            + [f"{p}{n}" for p in MOMENTS for n in PER_MOMENT])


def _phenomenon(codes, pattern: re.Pattern) -> bool:
    """True if a phenomenon at the airport itself (not 'VC', in the vicinity) matches."""
    if not isinstance(codes, str):
        return False
    return any(pattern.search(code) for code in codes.split() if not code.startswith("VC"))


def load_metar(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    usecols = ["station", "valid", "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "p01i", "vsby",
               "skyl1", "wxcodes", "metar"]
    metar = pd.read_csv(path, usecols=usecols, na_values=["", "M"], low_memory=False)
    metar["t"] = pd.to_datetime(metar["valid"], utc=True).astype("datetime64[ns, UTC]")
    metar = metar[(metar["t"] >= start - MAX_METAR_AGE) & (metar["t"] <= end)].copy()

    metar["temperature_c"] = ((metar["tmpf"] - 32) * 5 / 9).round(1)
    metar["dewpoint_c"] = ((metar["dwpf"] - 32) * 5 / 9).round(1)
    metar["dewpoint_spread_c"] = (metar["temperature_c"] - metar["dewpoint_c"]).round(1)
    metar["humidity_pct"] = metar["relh"]
    metar["wind_kt"] = metar["sknt"]
    metar["gust_kt"] = metar["gust"].fillna(0)
    radians = np.deg2rad(metar["drct"])
    metar["wind_dir_sin"] = np.sin(radians)
    metar["wind_dir_cos"] = np.cos(radians)
    metar["cloud_base_ft"] = metar["skyl1"]
    metar["precip_mm"] = metar["p01i"] * 25.4

    unknown = metar["metar"].astype("string").str.contains(UNKNOWN_VISIBILITY, na=False)
    metar["visibility_mi"] = metar["vsby"].where(~unknown)
    metar["visibility_band"] = pd.cut(metar["visibility_mi"], bins=VISIBILITY_BINS_MI, right=False,
                                      labels=VISIBILITY_LABELS).astype("string")
    metar["low_visibility"] = metar["visibility_mi"] < LOW_VISIBILITY_MI

    for name, pattern in PHENOMENA.items():
        metar[name] = metar["wxcodes"].map(lambda c, p=pattern: _phenomenon(c, p))
    metar["freezing_or_snow"] = metar["freezing"] | metar["snow"]

    cold = metar["temperature_c"] <= COLD_C
    moist = metar["wxcodes"].map(lambda c: _phenomenon(c, PRECIPITATION_OR_ICE))
    metar["cold_moist"] = cold & moist
    metar["deicing_risk"] = pd.Series(
        np.select([cold & moist, cold], ["frío_húmedo", "frío_seco"], default="templado"), index=metar.index
    ).where(metar["temperature_c"].notna()).astype("string")

    columns = ["station", "t", "deicing_risk", "visibility_band", "freezing_or_snow"] + \
               [c for c in PER_MOMENT if c not in ("metar_age_min", "metar_missing")]
    return metar[columns]


def add_weather_features(df: pd.DataFrame, metar_path: Path) -> pd.DataFrame:
    df = df.copy()
    start = min(df[TAKEOFF].min(), df[SCHEDULED].min())
    end = max(df[TAKEOFF].max(), df[SCHEDULED].max())
    metar = load_metar(metar_path, start, end)
    metar_t = metar.rename(columns={"t": "t"})

    for prefix, moment in MOMENTS.items():
        base = metar_t.copy()
        base["t_metar"] = base["t"]
        flights = pd.DataFrame({
            "_row": np.arange(len(df)),
            "station": df["ADEP_mvt"].astype(str).to_numpy(),
            "t": pd.to_datetime(df[moment].astype("datetime64[ns, UTC]"), utc=True).to_numpy(),
        })
        valid = flights.dropna(subset=["t"]).sort_values("t")
        merged = pd.merge_asof(valid, base.sort_values("t"), on="t", by="station",
                              direction="backward", tolerance=MAX_METAR_AGE)
        merged = merged.set_index("_row")

        for name in PER_MOMENT:
            col = f"{prefix}{name}"
            if name == "metar_age_min":
                values = (merged["t"] - merged["t_metar"]).dt.total_seconds() / 60
            elif name == "metar_missing":
                values = merged["temperature_c"].isna()
            else:
                values = merged[name]
            series = pd.Series(np.nan, index=range(len(df)), dtype="object")
            series.loc[values.index] = values.to_numpy()
            df[col] = series.to_numpy()

        # the three historical categoricals come from the takeoff report
        if prefix == "":
            for name in ["deicing_risk", "visibility_band", "freezing_or_snow"]:
                series = pd.Series(np.nan, index=range(len(df)), dtype="object")
                series.loc[merged.index] = merged[name].to_numpy()
                df[name] = series.to_numpy()

    numeric = [f"{p}{n}" for p in MOMENTS for n in PER_MOMENT
                 if n not in list(PHENOMENA) + ["low_visibility", "cold_moist", "metar_missing"]]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [f"{p}{n}" for p in MOMENTS for n in list(PHENOMENA) + ["low_visibility", "cold_moist", "metar_missing"]]:
        df[col] = df[col].astype("boolean")
    df["temperature_c"] = df["temperature_c"].astype(float)
    df["freezing_or_snow"] = df["freezing_or_snow"].astype("boolean")
    df["visibility_band"] = df["visibility_band"].astype("string")
    df["deicing_risk"] = df["deicing_risk"].astype("string")
    return df
