"""M2: second model for the own-ADS-B departures of the nine airports (2026-10-07).

    python -m components.m2.m2 holdout   # train on 2025 without January and July, predict Jan + Jul 2025
    python -m components.m2.m2 ranking   # train on all 2025, predict the 2026 ranking

Environment: M2_SEED (11; the first seed writes pred_<mode>.parquet, the others pred_<mode>_s<seed>.parquet),
M2_THREADS, M2_LGB_ROUNDS (2500), M2_CB_ITERS (3000).

Predicts the gap y - v4 (off-block seen by rule v4) from detector candidates, first-point cell and the context tables;
mean of LightGBM and CatBoost, then the LOBT window (+/- 3606 s) and >= 0. LIRF is excluded.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import FG4_EVENTS as EV, LOCAL_DATA as LD, NINE, RAW, ROOT

HERE = Path(__file__).resolve().parent
OB = ROOT / "components/offblock"
TABLES = [LD / "context_features/features.parquet", LD / "forward_features/features.parquet",
          LD / "state_features/features.parquet", LD / "wx_features/features.parquet",
          LD / "inbound_features_nine/features.parquet", LD / "adsb_features_20260928/features_takeoff.parquet"]
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ARP = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
       "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LSZH": (47.4647, 8.5492),
       "LTFM": (41.2753, 28.7519)}
WINDOW = 3606.0
GAP_CLIP = (-1800.0, 5400.0)
CAT = ["apt", "stand", "runway", "airline", "actype", "wtc", "cell", "rwy_config", "geo_cross_names"]
COLS = [ID, "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
        "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt", Y]
SEED = int(os.environ.get("M2_SEED", "11"))
THREADS = int(os.environ.get("M2_THREADS", "24"))


def load(ranking: bool) -> pd.DataFrame:
    files = [RAW / "ranking.parquet"] if ranking else sorted(RAW.glob("training_2025-*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    d = d[(d.PHASE_mvt == "DEP") & d.ADEP_mvt.isin(NINE)]
    ev = pd.read_parquet(EV)
    d = d.merge(ev, on=ID).merge(pd.read_parquet(OB / "ob_v4.parquet"), on=ID)
    d = d.merge(pd.read_parquet(OB / "ob4_fixed.parquet"), on=ID, how="left")
    for f in TABLES:
        t = pd.read_parquet(f)
        t = t.loc[:, ~t.columns.duplicated()]
        d = d.merge(t, on=ID, how="left", suffixes=("", "_dup"))
    d = d[np.isfinite(d.v4.to_numpy(float))]
    return d.reset_index(drop=True)


def features(d: pd.DataFrame) -> pd.DataFrame:
    mvt = d.MVT_TIME_UTC_mvt
    f = pd.DataFrame(index=d.index)
    v4 = d.v4.astype(float)
    f["v4"] = v4
    for c in ["c_first", "c_move", "c_first_long", "c_last_long", "init_stop", "max_gs_before_long", "last_gap",
              "d_long", "d_first_long", "gs0", "n_long"]:
        f[c] = d[c].astype(float)
    for c in ["c_first", "c_move", "c_first_long", "c_last_long"]:
        f[c + "_m_v4"] = d[c].astype(float) - v4
    for c in ["adsb_taxi", "adsb_taxi_move", "adsb_gs0", "adsb_takeoff_err", "adsb_n_chao", "adsb_gap_max",
              "adsb_frac_mlat"]:
        f[c] = d[c].astype(float)
    lat0 = d.ADEP_mvt.map(lambda a: ARP[a][0]).astype(float)
    lon0 = d.ADEP_mvt.map(lambda a: ARP[a][1]).astype(float)
    f["dy"] = (d.adsb_lat0.astype(float) - lat0) * 111.2e3
    f["dx"] = (d.adsb_lon0.astype(float) - lon0) * 111.2e3 * np.cos(np.radians(lat0))
    f["cell"] = (d.ADEP_mvt + ":" + (f.dy // 150).astype("Int64").astype(str) + "|"
                 + (f.dx // 150).astype("Int64").astype(str))
    ob = mvt - pd.to_timedelta(v4, unit="s")
    for c in ["SCHED_TIME_UTC_mvt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt"]:
        f["ob_m_" + c] = (ob - d[c]).dt.total_seconds()
        f["mvt_m_" + c] = (mvt - d[c]).dt.total_seconds()
    f["nm"] = d.FLIGHT_ID_mvt.notna().astype(float)
    f["apt"] = d.ADEP_mvt
    f["stand"] = d.ADEP_mvt + ":" + d.STAND_mvt.fillna("?").astype(str)
    f["runway"] = d.ADEP_mvt + ":" + d.RUNWAY_mvt.fillna("?").astype(str)
    f["airline"] = d.CALLSIGN_flt.fillna(d.FLIGHT_mvt).astype(str).str[:3]
    f["actype"] = d.AIRCRAFT_TYPE_mvt.fillna("?").astype(str)
    f["wtc"] = d.WK_TBL_CAT_flt.fillna("?").astype(str)
    f["hour"] = mvt.dt.hour + mvt.dt.minute / 60
    f["dow"] = mvt.dt.dayofweek.astype(float)
    skip = set(COLS) | {"v4", "adsb_lat0", "adsb_lon0"} | set(f.columns)
    for c in d.columns:
        if c in skip or c.endswith("_dup"):
            continue
        if c in ("rwy_config", "geo_cross_names"):
            f[c] = d[c].fillna("?").astype(str)
        elif pd.api.types.is_numeric_dtype(d[c]) or pd.api.types.is_bool_dtype(d[c]):
            f[c] = d[c].astype(float)
    for c in CAT:
        f[c] = f[c].astype("category")
    return f


def window(d: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    g = (d.MVT_TIME_UTC_mvt - d.LOBT_flt).dt.total_seconds().to_numpy(float)
    lo, hi = np.nan_to_num(g - WINDOW, nan=-np.inf), np.nan_to_num(g + WINDOW, nan=np.inf)
    return np.maximum(np.clip(p, lo, hi), 0)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, path: Path) -> pd.DataFrame:
    import catboost as cb
    import lightgbm as lgb
    Xtr, Xte = features(tr), features(te)
    for c in CAT:
        Xte[c] = pd.Categorical(Xte[c].astype(str).where(Xte[c].astype(str).isin(Xtr[c].cat.categories)),
                                categories=Xtr[c].cat.categories)
    gap = np.clip(tr[Y].to_numpy(float) - tr.v4.to_numpy(float), *GAP_CLIP)
    print(f"variables {Xtr.shape[1]}, train {len(Xtr):,}, test {len(Xte):,}", flush=True)
    t0 = time.time()
    params = dict(objective="regression", learning_rate=0.04, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, cat_smooth=20,
                  max_cat_to_onehot=8, num_threads=THREADS, seed=SEED, verbose=-1)
    m = lgb.train(params, lgb.Dataset(Xtr, gap), num_boost_round=int(os.environ.get("M2_LGB_ROUNDS", "2500")))
    p_lgb = m.predict(Xte)
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    print(f"LightGBM {time.time() - t0:.0f} s; importance:\n{(imp / imp.sum()).head(30).round(3).to_string()}", flush=True)
    v4 = te.v4.to_numpy(float)
    part = pd.DataFrame({ID: te[ID].to_numpy(), "apt": te.ADEP_mvt.to_numpy(), "m2_lgb": window(te, v4 + p_lgb)})
    part.to_parquet(path.with_name(path.stem + "_lgb.parquet"), index=False)  # in case CatBoost fails
    t0 = time.time()
    cats = [Xtr.columns.get_loc(c) for c in CAT]
    tr_cb, te_cb = Xtr.copy(), Xte.copy()
    for c in CAT:  # CatBoost rejects NaN categories (unseen test categories arrive as NaN)
        tr_cb[c] = tr_cb[c].astype(object).where(tr_cb[c].notna(), "?").astype(str)
        te_cb[c] = te_cb[c].astype(object).where(te_cb[c].notna(), "?").astype(str)
    mc = cb.CatBoostRegressor(iterations=int(os.environ.get("M2_CB_ITERS", "3000")), learning_rate=0.08, depth=8, l2_leaf_reg=5, loss_function="RMSE",
                              thread_count=THREADS, random_seed=SEED, verbose=500, allow_writing_files=False)
    mc.fit(cb.Pool(tr_cb, gap, cat_features=cats))
    p_cb = mc.predict(cb.Pool(te_cb, cat_features=cats))
    print(f"CatBoost {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame({ID: te[ID].to_numpy(), "apt": te.ADEP_mvt.to_numpy(),
                        "m2_lgb": window(te, v4 + p_lgb), "m2_cb": window(te, v4 + p_cb)})
    out["m2"] = window(te, v4 + 0.5 * (p_lgb + p_cb))
    return out


def main():
    warnings.simplefilter("ignore")
    mode = sys.argv[1]
    tr = load(False)
    month = tr.MVT_TIME_UTC_mvt.dt.month
    if mode == "holdout":
        te, tr = tr[month.isin([1, 7])], tr[~month.isin([1, 7])]
    elif mode == "ranking":
        te = load(True)
    else:
        raise SystemExit("mode: holdout | ranking")
    path = HERE / (f"pred_{mode}.parquet" if SEED == 11 else f"pred_{mode}_s{SEED}.parquet")
    out = fit_predict(tr.reset_index(drop=True), te.reset_index(drop=True), path)
    out.to_parquet(path, index=False)
    print("→", path, len(out))


if __name__ == "__main__":
    main()
