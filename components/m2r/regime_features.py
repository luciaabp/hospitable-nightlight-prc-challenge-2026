"""Day-regime features for M2r (2026-10-07), one row per departure, computed from other flights without departure labels.

r_x_*: median MVT - AOBT_3 of the airport's departures in +/- 60 min and in the day, over the 2025 airport median;
r_tin_*: the same with the arrivals' taxi-in; r_far_* / r_long_h60: share of traced departures with a stop >= 15 min
(far from the first point / anywhere); r_xx_h60, r_tinx_*: excess over the 2025 route median (airport x runway x
stand); r_n_h60: departures in +/- 60 min.

    python -m components.m2r.regime_features      # -> regime_features.parquet
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import common

HERE = Path(__file__).resolve().parent
RAW = str(common.RAW) + "/"
OB = common.ROOT / "components/offblock"
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
H = "120min"  # centred window: +/- 60 min


def rolling_median(t: pd.Series, v: pd.Series) -> np.ndarray:
    s = pd.Series(v.to_numpy(float), index=pd.DatetimeIndex(t)).sort_index()
    order = np.argsort(t.to_numpy(), kind="stable")
    r = s.rolling(H, center=True, min_periods=5).median().to_numpy()
    out = np.empty(len(t))
    out[order] = r
    return out


def main():
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt", "STAND_mvt", Y]
    files = sorted(glob.glob(RAW + "training_*.parquet")) + [RAW + "ranking.parquet"]
    al = pd.concat([pd.read_parquet(f, columns=cols).assign(y26=f.endswith("ranking.parquet")) for f in files],
                   ignore_index=True)
    dep = al[(al.PHASE_mvt == "DEP") & al.ADEP_mvt.isin(APTS)].copy()
    arr = al[(al.PHASE_mvt == "ARR") & al.ADES_mvt.isin(APTS) & al[Y].notna()].copy()
    dep["x"] = (dep.MVT_TIME_UTC_mvt - dep.AOBT_3_flt).dt.total_seconds()
    dep.loc[(dep.x < 0) | (dep.x > 7200), "x"] = np.nan
    c = pd.read_parquet(OB / "ob4_fixed.parquet", columns=[ID, "n_long", "d_first_long"]).set_index(ID)
    seen = dep[ID].isin(c.index)
    nl = c.n_long.reindex(dep[ID]).to_numpy(float)
    far = (nl > 0) & (c.d_first_long.reindex(dep[ID]).to_numpy(float) > 50)
    dep["far"] = np.where(seen, far.astype(float), np.nan)
    dep["long"] = np.where(seen, (nl > 0).astype(float), np.nan)
    dep["day"] = dep.MVT_TIME_UTC_mvt.dt.floor("D")
    # excess over the route (2025 median by airport x runway x stand; the airport's if the route has < 20 flights)
    for d, a_col, v in ((dep, "ADEP_mvt", "x"), (arr, "ADES_mvt", Y)):
        key = d[a_col] + "|" + d.RUNWAY_mvt.fillna("?").astype(str) + "|" + d.STAND_mvt.fillna("?").astype(str)
        g = d[~d.y26].groupby(key[~d.y26])[v].agg(["median", "size"])
        med = key.map(g["median"].where(g["size"] >= 20))
        d["exc"] = d[v] - med.fillna(d[a_col].map(d[~d.y26].groupby(a_col)[v].median()))
    arr["day"] = arr.MVT_TIME_UTC_mvt.dt.floor("D")
    ref_x = dep[~dep.y26].groupby("ADEP_mvt").x.median()
    ref_t = arr[~arr.y26].groupby("ADES_mvt")[Y].median()
    parts = []
    for a in APTS:
        d = dep[dep.ADEP_mvt == a].sort_values("MVT_TIME_UTC_mvt").copy()
        r = arr[arr.ADES_mvt == a].sort_values("MVT_TIME_UTC_mvt").copy()
        d["r_x_h60"] = rolling_median(d.MVT_TIME_UTC_mvt, d.x) / ref_x[a]
        d["r_far_h60"] = (pd.Series(d.far.to_numpy(), index=pd.DatetimeIndex(d.MVT_TIME_UTC_mvt))
                          .rolling(H, center=True, min_periods=5).mean().to_numpy())
        d["r_long_h60"] = (pd.Series(d.long.to_numpy(), index=pd.DatetimeIndex(d.MVT_TIME_UTC_mvt))
                           .rolling(H, center=True, min_periods=5).mean().to_numpy())
        d["r_xx_h60"] = rolling_median(d.MVT_TIME_UTC_mvt, d.exc)
        d["r_n_h60"] = (pd.Series(1.0, index=pd.DatetimeIndex(d.MVT_TIME_UTC_mvt)).rolling(H, center=True).sum()
                        .to_numpy())
        r["r_tin_h60"] = rolling_median(r.MVT_TIME_UTC_mvt, r[Y]) / ref_t[a]
        r["r_tinx_h60"] = rolling_median(r.MVT_TIME_UTC_mvt, r.exc)
        d = pd.merge_asof(d, r[["MVT_TIME_UTC_mvt", "r_tin_h60", "r_tinx_h60"]], on="MVT_TIME_UTC_mvt",
                          direction="nearest", tolerance=pd.Timedelta("30min"))
        dd = d.groupby("day").agg(r_x_day=("x", "median"), r_far_day=("far", "mean"))
        dd["r_x_day"] /= ref_x[a]
        dd["r_tin_day"] = r.groupby("day")[Y].median() / ref_t[a]
        dd["r_tinx_day"] = r.groupby("day").exc.median()
        d = d.merge(dd, left_on="day", right_index=True, how="left")
        parts.append(d[[ID, "r_x_h60", "r_x_day", "r_tin_h60", "r_tin_day", "r_far_h60", "r_far_day", "r_long_h60",
                       "r_xx_h60", "r_tinx_h60", "r_tinx_day", "r_n_h60"]])
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(HERE / "regime_features.parquet", index=False)
    yr = out[ID].isin(dep.loc[dep.y26, ID])
    print(out.shape)
    print(pd.concat({"2025": out[~yr].describe().T[["mean", "50%", "max"]],
                     "2026": out[yr].describe().T[["mean", "50%", "max"]]}, axis=1).round(3))


if __name__ == "__main__":
    main()
