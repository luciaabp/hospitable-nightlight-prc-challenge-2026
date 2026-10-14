"""Increment for LIRF departures without NM with MVT - SCHED between 1 h and 15 h (SCHED copy or real taxi), recipe v73.

    python -m components.lirf.lirf_rotation --oof OOF.parquet --out outputs/final/lirf_v73.parquet

Adds what the base model does not see: per-aircraft rotation (components/lirf/tail_rotation.py: previous arrival and
its in-block). Two LightGBM fit the residual y - base of the 2025 out-of-block base prediction (``--oof``: base e76 of
variants/base_reference, located by ``variants.py oof-path``), without and with the rotation features; the increment
is their difference, for the 2026 ranking months. The flight number / registration / route copy histories are kept in
the output for inspection only.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import LOCAL_DATA, RAW

TAIL = LOCAL_DATA / "tail_rotation"
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt"]
ROT = ["slack", "to_mvt", "in_late", "in_copy", "same_stand"]
HISTORY_KEYS = {"fn": "FLIGHT_mvt", "reg": "reg", "ruta": "ruta"}  # feature prefix -> history column
WITHOUT_ROTATION = ["delay", "nm", "g_LOBT_flt", "g_EOBT_1_flt", "g_AOBT_3_flt", "hour", "cia"]
V73 = (WITHOUT_ROTATION, WITHOUT_ROTATION + ROT)  # without, with the rotation
CATS = ["cia", "ADES_mvt", "RUNWAY_mvt", "stand_p", "AIRCRAFT_TYPE_mvt"]
MIN_DELAY = 3600
MAX_DELAY = 15 * 3600  # beyond this the base model's Rome rule applies (15-30 h)
COPY_SECONDS, SMOOTHING_WEIGHT = 60, 5
PARAMS = dict(objective="regression", learning_rate=0.02, num_leaves=7, min_data_in_leaf=80, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=8)
ROUNDS, SEEDS = 300, (0, 1, 2)


def frame(airport: str = "LIRF") -> pd.DataFrame:
    """LIRF departures of 2025 and of the ranking with the model features (no history)."""
    fs = [*sorted(RAW.glob("training_2025-*.parquet")), RAW / "ranking.parquet"]
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in fs], ignore_index=True)
    d = d[(d.PHASE_mvt == "DEP") & (d.ADEP_mvt == airport)].copy()
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d = d.merge(pd.read_parquet(TAIL / "rotation.parquet").drop(columns=["apt", "icao"]), on="MVT_ID_mvt", how="left")
    d = d.merge(pd.read_parquet(TAIL / "flights.parquet", columns=["MVT_ID_mvt", "reg"]), on="MVT_ID_mvt", how="left")
    s = lambda x: x.dt.total_seconds()  # noqa: E731
    inb = pd.to_datetime(d.in_BLOCK_TIME_UTC_mvt, utc=True)
    t = d.MVT_TIME_UTC_mvt
    d["mes"] = t.dt.strftime("%Y-%m")
    d["m"] = t.dt.month + 12 * (t.dt.year - 2025)  # 1-12 in 2025, 13 and 19 in the ranking
    d["delay"] = s(t - d.SCHED_TIME_UTC_mvt)
    d["nm"] = d.FLIGHT_ID_mvt.notna().astype(int)
    for c in ("LOBT_flt", "EOBT_1_flt", "AOBT_3_flt"):
        d["g_" + c] = s(t - d[c])
    d["hour"] = t.dt.hour
    d["dow"] = t.dt.dayofweek
    d["cia"] = d.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z]{2,3})", expand=False).fillna("?")
    d["stand_p"] = d.STAND_mvt.astype(str).str[:1]
    d["ruta"] = d.cia.astype(str) + "|" + d.ADES_mvt.astype(str)  # stable across years; the flight number is not
    for c in CATS:
        d[c] = d[c].astype(str).astype("category")
    d["slack"] = s(d.SCHED_TIME_UTC_mvt - inb)              # > 0: the aircraft was at the stand at scheduled time
    d["to_mvt"] = s(t - inb)                                # bound of y if BLOCK is not a copy
    d["in_late"] = s(inb - pd.to_datetime(d.in_SCHED_TIME_UTC_mvt, utc=True))
    d["in_copy"] = (d.in_late.abs() <= COPY_SECONDS).astype(float).where(inb.notna())
    d["same_stand"] = (d.STAND_mvt == d.in_STAND_mvt).astype(float).where(inb.notna())
    d["cp"] = (s(d.BLOCK_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).abs() <= COPY_SECONDS).astype(float).where(d.m <= 12)
    return d.reset_index(drop=True)


def history(d: pd.DataFrame, col: str, excluded: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed copy rate of the key on 2025 delayed departures, without the row's own month or `excluded`."""
    base = d[(d.delay > MIN_DELAY) & d.cp.notna() & ~d.m.isin(excluded)]
    tab = base.groupby([col, "m"], observed=True).cp.agg(["sum", "size"]).reset_index()
    tot = tab.groupby(col, observed=True)[["sum", "size"]].sum()
    s_all = d[col].map(tot["sum"]).astype(float).fillna(0).to_numpy()
    n_all = d[col].map(tot["size"]).astype(float).fillna(0).to_numpy()
    own = d[[col, "m"]].merge(tab, on=[col, "m"], how="left")
    s_ = s_all - own["sum"].fillna(0).to_numpy()
    n_ = n_all - own["size"].fillna(0).to_numpy()
    prior = np.where(d.nm == 1, base[base.nm == 1].cp.mean(), base[base.nm == 0].cp.mean())
    return np.where(d[col].notna(), (s_ + SMOOTHING_WEIGHT * prior) / (n_ + SMOOTHING_WEIGHT), np.nan), n_


def with_history(d: pd.DataFrame, excluded: list[int]) -> pd.DataFrame:
    d = d.copy()
    for pre, col in HISTORY_KEYS.items():
        d[f"{pre}_rate"], d[f"{pre}_n"] = history(d, col, excluded)
    return d


def increment(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    out = []
    for feats in reversed(V73):  # with, without
        cats = [c for c in feats if c in CATS]
        out.append(np.mean([lgb.train({**PARAMS, "seed": sd}, lgb.Dataset(train[feats], train.y - train.base,
                                                                          categorical_feature=cats), ROUNDS)
                            .predict(target[feats]) for sd in SEEDS], axis=0))
    return out[0] - out[1]


def training(d: pd.DataFrame, oof: Path) -> pd.DataFrame:
    o = pd.read_parquet(oof, columns=["MVT_ID_mvt", "y_true", "pred"]).rename(columns={"y_true": "y", "pred": "base"})
    tr = d.merge(o, on="MVT_ID_mvt")
    return tr[(tr.delay > MIN_DELAY) & (tr.y < 3 * 3600 + tr.delay.clip(lower=0))]  # no gross lotteries


def targets(d: pd.DataFrame, months) -> pd.DataFrame:
    return d[d.mes.isin(months) & (d.delay > MIN_DELAY) & (d.delay <= MAX_DELAY) & (d.nm == 0)]


def produce(oof: Path, out: Path) -> None:
    d = with_history(frame(), excluded=[])  # all 2025; each training row without its own month
    tr = training(d, oof)
    te = targets(d, ["2026-01", "2026-02", "2026-06", "2026-07"]).copy()
    te["inc"] = increment(tr, te)
    out.parent.mkdir(parents=True, exist_ok=True)
    te[["MVT_ID_mvt", "mes", "inc", "slack", "fn_n", "ruta_n", "reg_n"]].to_parquet(out, index=False)
    print(f"v73: {len(tr):,} training departures · {len(te)} targets {te.mes.value_counts().to_dict()}"
          f" · with rotation {te.slack.notna().mean():.0%} · history flight/route/aircraft {(te.fn_n > 0).mean():.0%}/"
          f"{(te.ruta_n > 0).mean():.0%}/{(te.reg_n > 0).mean():.0%}"
          f" · mean |inc| {te.inc.abs().mean():.0f} s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True, help="out-of-block base prediction of 2025 (y_true, pred)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    produce(a.oof, a.out)


if __name__ == "__main__":
    main()
