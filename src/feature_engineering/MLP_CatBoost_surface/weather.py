"""Past-observation METAR joins for the surface ensemble."""

import numpy as np
import pandas as pd

REQUIRED = ["icao", "valid", "tmpf", "dwpf", "relh", "drct", "sknt", "vsby", "skyl1", "skyl2", "wxcodes"]


def prepare_metar(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(raw)
    if missing:
        raise ValueError(f"Missing METAR columns: {sorted(missing)}")
    metar = raw[REQUIRED].copy()
    metar["airport"] = metar["icao"].astype("string").str.strip().str.upper()
    metar["observed"] = pd.to_datetime(
        metar["valid"], utc=True, errors="coerce", format="mixed"
    ).astype("datetime64[ns, UTC]")
    if metar["airport"].isna().any() or metar["observed"].isna().any():
        raise ValueError("METAR station and timestamp must be known")
    if metar.duplicated(["airport", "observed"]).any():
        raise ValueError("Ambiguous duplicate METAR station/timestamp")
    metar["airport"] = metar["airport"].astype(str)
    prepared = metar[["airport", "observed"]].copy()
    for source, destination, scale, offset in [
        ("tmpf", "temperature_c", 5 / 9, -32 * 5 / 9),
        ("dwpf", "dewpoint_c", 5 / 9, -32 * 5 / 9),
        ("relh", "humidity_pct", 1, 0),
        ("sknt", "wind_kt", 1, 0),
        ("vsby", "visibility_m", 1609.344, 0),
        ("skyl1", "cloud_base1_ft", 1, 0),
        ("skyl2", "cloud_base2_ft", 1, 0),
    ]:
        prepared[destination] = pd.to_numeric(metar[source], errors="coerce") * scale + offset
    direction = pd.to_numeric(metar["drct"], errors="coerce")
    prepared["wind_direction_sin"] = np.sin(np.deg2rad(direction))
    prepared["wind_direction_cos"] = np.cos(np.deg2rad(direction))
    prepared["wind_direction_missing"] = direction.isna().astype(float)
    prepared["dewpoint_spread_c"] = prepared["temperature_c"] - prepared["dewpoint_c"]
    codes = metar["wxcodes"].fillna("").astype(str).str.upper()
    for name, pattern in {
        "rain": "RA|DZ", "snow": "SN|SG", "fog_mist": "FG|BR",
        "thunder": "TS", "freezing": "FZ", "hail": "GR|GS", "showers": "SH",
    }.items():
        prepared[name] = codes.str.contains(pattern, regex=True).astype(float)
    prepared["low_visibility"] = prepared["visibility_m"].lt(1000).where(
        prepared["visibility_m"].notna()
    ).astype(float)
    prepared["cold_moist"] = (
        prepared["temperature_c"].le(2) & prepared["humidity_pct"].ge(90)
    ).where(prepared["temperature_c"].notna() & prepared["humidity_pct"].notna()).astype(float)
    return prepared.sort_values(["observed", "airport"]).reset_index(drop=True)


def weather_features(flights: pd.DataFrame, prepared: pd.DataFrame, max_age="2h") -> pd.DataFrame:
    if pd.Timedelta(max_age) <= pd.Timedelta(0):
        raise ValueError("METAR maximum age must be positive")
    parts = []
    for label, column in [("takeoff", "MVT_TIME_UTC_mvt"), ("scheduled", "SCHED_TIME_UTC_mvt")]:
        left = pd.DataFrame({
            "airport": flights["ADEP_mvt"].astype(str).to_numpy(),
            "reference": pd.to_datetime(flights[column], utc=True, errors="coerce").to_numpy(),
            "row": np.arange(len(flights)),
        })
        left["reference"] = pd.to_datetime(left["reference"], utc=True).astype("datetime64[ns, UTC]")
        valid = left[left["reference"].notna()].sort_values("reference")
        joined = pd.merge_asof(
            valid, prepared, left_on="reference", right_on="observed", by="airport",
            direction="backward", tolerance=pd.Timedelta(max_age),
        )
        joined["age_minutes"] = (joined["reference"] - joined["observed"]).dt.total_seconds() / 60
        columns = [c for c in prepared if c not in ["airport", "observed"]] + ["age_minutes"]
        features = joined.set_index("row")[columns].reindex(range(len(flights)))
        features["missing"] = features["age_minutes"].isna().astype(float)
        features.index = flights.index
        parts.append(features.add_prefix(f"wx_{label}_"))
    return pd.concat(parts, axis=1)