"""Distance from the first ADS-B point to the assigned stand (2026-10-07). A stand's position is the median first point
of aircraft seen parked there (gs0 < 1 kt) in 2025 and 2026: positions only, no labels.

    python -m components.m2r.stand_features     # -> stand_features.parquet: s_d_stand, s_at_stand, s_stand_n
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import common

HERE = Path(__file__).resolve().parent
RAW = str(common.RAW) + "/"
EV = str(common.FG4_EVENTS)
ID = "MVT_ID_mvt"
QUALITY_COLUMNS = ["sq_dx", "sq_dy", "sq_distance", "sq_radius50", "sq_radius90", "sq_n", "sq_days",
                   "sq_distance_scaled", "sq_supported"]


def quality_features(query: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Estimate stand geometry from other day folds, without departure labels.

    The caller supplies the allowed reference months. Five deterministic UTC-day
    folds exclude the query day, including the query flight, from its reference.
    At least three parked observations across two days are needed for geometry.
    """
    needed = [ID, "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "adsb_lat0", "adsb_lon0", "adsb_gs0"]
    for source in (query, reference):
        if source[ID].isna().any() or source[ID].duplicated().any():
            raise ValueError("Stand features require unique, non-null movement IDs")
    queries, refs = query[needed].reset_index(drop=True).copy(), reference[needed].reset_index(drop=True).copy()
    keys = ["ADEP_mvt", "STAND_mvt"]
    for source in (queries, refs):
        source["STAND_mvt"] = source.STAND_mvt.astype("string").str.strip().replace("", pd.NA)
        source["day"] = pd.to_datetime(source.MVT_TIME_UTC_mvt, utc=True).dt.floor("D")
        source["fold"] = (source.day - pd.Timestamp("1970-01-01", tz="UTC")).dt.days % 5
        valid = source.adsb_lat0.between(-90, 90) & source.adsb_lon0.between(-180, 180)
        source.loc[~valid, ["adsb_lat0", "adsb_lon0"]] = np.nan
    parked = refs.adsb_gs0.ge(0) & refs.adsb_gs0.lt(1) & refs.adsb_lat0.notna() & refs.day.notna()
    refs = refs.loc[parked & refs.STAND_mvt.notna() & refs.ADEP_mvt.notna()]
    out = pd.DataFrame(np.nan, index=queries.index, columns=QUALITY_COLUMNS)
    out["sq_n"], out["sq_days"], out["sq_supported"] = 0.0, 0.0, 0.0
    for fold in range(5):
        selected = queries[queries.fold.eq(fold)]
        other = refs[refs.fold.ne(fold)].copy()
        if selected.empty or other.empty:
            continue
        groups = other.groupby(keys, observed=True)
        centres = groups.agg(lat=("adsb_lat0", "median"), lon=("adsb_lon0", "median"),
                             count=(ID, "size"), days=("day", "nunique"))
        ref_centres = centres.reindex(pd.MultiIndex.from_frame(other[keys]))
        latitude = ref_centres.lat.to_numpy(float)
        north = (other.adsb_lat0.to_numpy(float) - latitude) * 111200
        east = (other.adsb_lon0.to_numpy(float) - ref_centres.lon.to_numpy(float)) * 111200 * np.cos(np.radians(latitude))
        other["radius"] = np.hypot(east, north)
        radii = other.groupby(keys, observed=True).radius
        centres["radius50"], centres["radius90"] = radii.median(), radii.quantile(0.9)
        positions = centres.reindex(pd.MultiIndex.from_frame(selected[keys]))
        supported = positions["count"].ge(3).to_numpy() & positions.days.ge(2).to_numpy()
        latitude = positions.lat.to_numpy(float)
        north = (selected.adsb_lat0.to_numpy(float) - latitude) * 111200
        east = (selected.adsb_lon0.to_numpy(float) - positions.lon.to_numpy(float)) * 111200 * np.cos(np.radians(latitude))
        distance = np.hypot(east, north)
        values = {"sq_dx": east, "sq_dy": north, "sq_distance": distance,
                  "sq_radius50": positions.radius50.to_numpy(float), "sq_radius90": positions.radius90.to_numpy(float),
                  "sq_distance_scaled": distance / np.maximum(positions.radius90.to_numpy(float), 25.0)}
        for column, value in values.items():
            out.loc[selected.index, column] = np.where(supported, value, np.nan)
        out.loc[selected.index, "sq_n"] = positions["count"].fillna(0).to_numpy(float)
        out.loc[selected.index, "sq_days"] = positions.days.fillna(0).to_numpy(float)
        out.loc[selected.index, "sq_supported"] = supported.astype(float)
    out.insert(0, ID, queries[ID].to_numpy())
    return out


def main():
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "STAND_mvt"]
    files = sorted(glob.glob(RAW + "training_*.parquet")) + [RAW + "ranking.parquet"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    d = d[d.PHASE_mvt == "DEP"].merge(pd.read_parquet(EV, columns=[ID, "adsb_gs0", "adsb_lat0", "adsb_lon0"]), on=ID)
    d["k"] = d.ADEP_mvt + "|" + d.STAND_mvt.fillna("?").astype(str)
    parked = (d.adsb_gs0 < 1.0) & d.adsb_lat0.notna() & d.STAND_mvt.notna()
    pos = d[parked].groupby("k").agg(lat=("adsb_lat0", "median"), lon=("adsb_lon0", "median"), n=("adsb_lat0", "size"))
    pos = pos[pos.n >= 3]
    p = pos.reindex(d.k)
    lat, lon = p.lat.to_numpy(float), p.lon.to_numpy(float)
    dy = (d.adsb_lat0.to_numpy(float) - lat) * 111.2e3
    dx = (d.adsb_lon0.to_numpy(float) - lon) * 111.2e3 * np.cos(np.radians(lat))
    out = pd.DataFrame({ID: d[ID].to_numpy(), "s_d_stand": np.hypot(dx, dy), "s_stand_n": p.n.to_numpy(float)})
    out["s_at_stand"] = ((out.s_d_stand < 50) & (d.adsb_gs0.to_numpy(float) < 1.0)).astype(float).where(out.s_d_stand.notna())
    out.to_parquet(HERE / "stand_features.parquet", index=False)
    print(out.describe().round(1).to_string())


if __name__ == "__main__":
    main()
