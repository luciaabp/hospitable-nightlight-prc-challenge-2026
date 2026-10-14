"""Runway queue per departure (optional): takeoff sequence, previous departure and taxi anomalies of earlier
departures, built only from **other** flights airborne before.

    python -m pipeline.state prepare [--output local_data/state_features]

For departure B with takeoff t_B only departures A of its runway with t_A < t_B and arrivals landed before enter; of B
only its runway, airline, AOBT_3 and t_B. Never any departure's block time nor B's own ADS-B. ``STATE_FEATURES`` go to
the three nine-airport taxi regressors; ``STATE_ADSB_FEATURES`` (ADS-B taxi and stop time of each A against a 7-day
runway baseline) only to the ADS-B regressor.

Output: ``features.parquet`` (one row per training and ranking departure) and ``manifest.json``.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID, ROOT, STATE_ADSB_FEATURES, STATE_FEATURES

log = logging.getLogger(__name__)

CHALLENGE = ROOT / "data" / "projects" / "prc_data_challenge"
MEASUREMENTS = ROOT / "local_data" / "adsb_features_20260928" / "measurements.parquet"
OUTPUT = ROOT / "local_data" / "state_features"
COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt",
           "WK_TBL_CAT_flt", "FLIGHT_mvt"]
BUSY_GAP = 150.0          # s: chained takeoffs in the busy period
MAX_X = 7200.0            # s: t - AOBT_3 outside [0, 2 h] is not a taxi window
ANOMALY_CLIP = 1800.0     # s
WAKE = {"L": 0.0, "M": 1.0, "H": 2.0, "J": 3.0}
# Clipping so that LightGBM's linear leaves do not extrapolate (as in the validation).
CLIP = {"seq_gap_prev_rwy": 3600, "seq_gap_prev2_rwy": 3600, "seq_gap_prev_mov_rwy": 3600, "seq_busy_len_rwy": 10800,
        "seq_busy_n_rwy": 60, "seq_arr_dep_ratio_30m": 10, "prev_aobt_gap": 3600}


def seconds(values: pd.Series) -> np.ndarray:
    v = pd.to_datetime(values, utc=True)
    out = v.astype("datetime64[ns, UTC]").array.asi8.astype(float) / 1e9
    out[v.isna().to_numpy()] = np.nan
    return out


def read_movements(raw_dir: Path = CHALLENGE / "raw", measurements: Path = MEASUREMENTS):
    """Departures (with their ADS-B segment, if any) and arrivals of training and ranking; never BLOCK."""
    parts = [pd.read_parquet(p, columns=COLUMNS).assign(training=True) for p in sorted(raw_dir.glob("training_*.parquet"))]
    parts.append(pd.read_parquet(raw_dir / "ranking.parquet", columns=COLUMNS).assign(training=False))
    raw = pd.concat(parts, ignore_index=True)
    if raw[ID].isna().any() or raw[ID].duplicated().any() or raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Movements without ID, with a repeated ID or without time")
    raw["runway"] = raw.RUNWAY_mvt.astype("string").str.strip().str.upper().fillna("NA").astype(str)
    raw["t"] = seconds(raw.MVT_TIME_UTC_mvt)
    dep = raw[raw.PHASE_mvt == "DEP"].copy()
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["route"] = (dep.airport + ":" + dep.STAND_mvt.astype("string").fillna("NA") + ":" + dep.runway).astype(str)
    dep["month"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    dep["aobt"] = seconds(dep.AOBT_3_flt)
    x = dep.t - dep.aobt
    dep["x"] = x.where(x.between(0, MAX_X))
    dep["wake"] = dep.WK_TBL_CAT_flt.map(WAKE)
    dep["airline"] = dep.FLIGHT_mvt.astype("string").str[:3].fillna("NA").astype(str)
    meas = pd.read_parquet(measurements, columns=[ID, "a_move_lead", "a_stop_time"])
    ok = meas.a_move_lead.between(0, MAX_X)
    meas["adsb_taxi"] = meas.a_move_lead.where(ok)
    meas["adsb_stop"] = meas.a_stop_time.where(ok)
    dep = dep.merge(meas[[ID, "adsb_taxi", "adsb_stop"]], on=ID, how="left", validate="1:1").reset_index(drop=True)
    arr = raw[raw.PHASE_mvt == "ARR"].copy()
    arr["airport"] = arr.ADES_mvt.astype(str)
    return dep, arr.reset_index(drop=True)


def count_before(sorted_t: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """Events in [t - window, t)."""
    return np.searchsorted(sorted_t, t, "left") - np.searchsorted(sorted_t, t - window, "left")


# ----------------------------------------------------------------------------------------------------
# Taxi anomalies of each departure A (they only feed later departures)
# ----------------------------------------------------------------------------------------------------
def route_median_previous_months(dep: pd.DataFrame, col: str) -> np.ndarray:
    """Median of ``col`` per route over the departures of months strictly before each row's month."""
    out = np.full(len(dep), np.nan)
    for m in sorted(dep.month.unique())[1:]:
        median = dep.loc[dep.month < m].groupby("route")[col].median()
        rows = (dep.month == m).to_numpy()
        out[rows] = dep.loc[rows, "route"].map(median).to_numpy()
    return out


def runway_median_7d(dep: pd.DataFrame, col: str) -> np.ndarray:
    """Median of ``col`` on the same runway with takeoff in [t - 7 days, t)."""
    out = np.full(len(dep), np.nan)
    for _, g in dep.groupby(["airport", "runway"], sort=False):
        g = g.sort_values("t", kind="stable")
        s = pd.Series(g[col].to_numpy(float), index=pd.to_datetime(g.t.to_numpy(), unit="s"))
        out[g.index.to_numpy()] = s.rolling("7D", closed="left", min_periods=5).median().to_numpy()
    return out


def anomalies(dep: pd.DataFrame) -> pd.DataFrame:
    dep = dep.copy()
    dep["nm_route"] = dep.x - route_median_previous_months(dep, "x")
    dep["adsb_rw7"] = dep.adsb_taxi - runway_median_7d(dep, "adsb_taxi")
    dep["stop_rw7"] = dep.adsb_stop - runway_median_7d(dep, "adsb_stop")
    for c in ["nm_route", "adsb_rw7", "stop_rw7"]:
        dep[c] = dep[c].clip(-ANOMALY_CLIP, ANOMALY_CLIP)
    return dep


# ----------------------------------------------------------------------------------------------------
# Features of B
# ----------------------------------------------------------------------------------------------------
def runway_block(g: pd.DataFrame, arr_t: np.ndarray) -> dict:
    """One runway's departures sorted by t. "Previous" is always strictly earlier."""
    t, aobt, x = g.t.to_numpy(), g.aobt.to_numpy(), g.x.to_numpy()
    n = len(t)
    before = np.searchsorted(t, t, "left")     # number of strictly earlier takeoffs
    p = before - 1
    has = p >= 0
    pi = np.maximum(p, 0)
    prev_t = np.where(has, t[pi], np.nan)
    out = {"seq_gap_prev_rwy": t - prev_t}
    p2 = np.where(has, p - 1, -1)
    out["seq_gap_prev2_rwy"] = t - np.where(p2 >= 0, t[np.maximum(p2, 0)], np.nan)
    arr_sorted = np.sort(arr_t)
    k = np.searchsorted(arr_sorted, t, "left") - 1
    last_arr = np.where(k >= 0, arr_sorted[np.maximum(k, 0)], np.nan) if len(arr_sorted) else np.full(n, np.nan)
    out["seq_gap_prev_mov_rwy"] = t - np.fmax(prev_t, last_arr)
    chained = has & ((t - prev_t) <= BUSY_GAP)
    start = t.copy()
    for i in np.flatnonzero(chained):          # p[i] < i: its start is already computed
        start[i] = start[p[i]]
    out["seq_busy_len_rwy"] = t - start
    out["seq_busy_n_rwy"] = (before - np.searchsorted(t, start, "left")).astype(float)
    for minutes in (5, 10, 15):
        out[f"seq_dep_rwy_{minutes}m"] = count_before(t, t, minutes * 60).astype(float)
    out["seq_arr_rwy_10m"] = count_before(arr_sorted, t, 600).astype(float)
    near = has & (out["seq_gap_prev_rwy"] <= 3600)

    def take(v):
        return np.where(near, np.asarray(v, float)[pi], np.nan)

    out["prev_wake"] = take(g.wake.to_numpy())
    out["prev_x"] = take(x)
    airline = g.airline.to_numpy()
    out["prev_same_airline"] = np.where(near, (airline[pi] == airline).astype(float), np.nan)
    out["prev_aobt_gap"] = aobt - take(aobt)
    # anomalies of the earlier ones: the previous, the previous 3, the queue (AOBT_3_B, t_B) and the previous 30 min
    valid = np.isfinite(x)
    q_lo = np.minimum(np.searchsorted(t, np.where(valid, aobt, np.inf), "right"), before)
    w_lo = np.searchsorted(t, t - 1800, "left")
    l3_lo = np.maximum(before - 3, 0)
    for a in ["nm_route", "adsb_rw7", "stop_rw7"]:
        v = g[a].to_numpy(float)
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        cn = np.concatenate([[0], np.cumsum(np.isfinite(v))])

        def mean(lo, hi):
            m = cn[hi] - cn[lo]
            return np.where(m > 0, (cs[hi] - cs[lo]) / np.maximum(m, 1), np.nan)

        out[f"{a}__prev"] = np.where(has, v[pi], np.nan)
        out[f"{a}__last3"] = mean(l3_lo, before)
        out[f"{a}__queue"] = np.where(valid, mean(q_lo, before), np.nan)
        out[f"{a}__rwy30"] = mean(w_lo, before)
    return out


def airport_features(d: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Features of an airport's departures ``d`` (anomalies already computed); ``a``: its arrivals."""
    d = d.sort_values("t", kind="stable").reset_index(drop=True)
    t = d.t.to_numpy()
    arr_t = np.sort(a.t.to_numpy())
    feats = pd.DataFrame({"seq_dep_apt_10m": count_before(t, t, 600).astype(float),
                          "seq_arr_dep_ratio_30m": (count_before(arr_t, t, 1800) + 1.0)
                          / (count_before(t, t, 1800) + 1.0)})
    parts = []
    for runway, g in d.groupby("runway", sort=False):
        frame = pd.DataFrame(runway_block(g, a.t.to_numpy()[a.runway.to_numpy() == runway]), index=g.index)
        if runway == "NA":
            frame[:] = np.nan
        parts.append(frame)
    feats = pd.concat([feats, pd.concat(parts).sort_index()], axis=1)
    for column, high in CLIP.items():
        feats[column] = feats[column].clip(upper=high)
    feats["prev_aobt_gap"] = feats["prev_aobt_gap"].clip(lower=-3600)
    feats.insert(0, ID, d[ID].to_numpy())
    return feats[[ID] + STATE_FEATURES + STATE_ADSB_FEATURES]


def state_features(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    dep = anomalies(dep)
    parts = [airport_features(dep[dep.airport == ap], arr[arr.airport == ap]) for ap in sorted(dep.airport.unique())]
    out = pd.concat(parts, ignore_index=True)
    return dep[[ID]].merge(out, on=ID, how="left", validate="1:1")


def prepare(raw_dir: Path = CHALLENGE / "raw", measurements: Path = MEASUREMENTS, output: Path = OUTPUT) -> Path:
    from .data import sha256
    dep, arr = read_movements(raw_dir, measurements)
    features = state_features(dep, arr)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "features.parquet"
    features.to_parquet(target, index=False)
    key = dep.airport + "|" + dep.month
    coverage = features.drop(columns=ID).notna().groupby(key.to_numpy()).mean()
    manifest = {"output": str(target), "output_sha256": sha256(target), "rows": int(len(features)),
                "busy_gap_sec": BUSY_GAP, "anomaly_clip_sec": ANOMALY_CLIP, "clip": CLIP,
                "inputs": {str(p): sha256(p) for p in sorted(raw_dir.glob("training_*.parquet"))
                           + [raw_dir / "ranking.parquet", measurements]},
                "coverage": coverage[["seq_gap_prev_rwy", "prev_x", "nm_route__last3", "adsb_rw7__last3"]]
                .round(4).to_dict(orient="index")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%s: %d departures", target, len(features))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.state", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--raw-dir", type=Path, default=CHALLENGE / "raw")
    parser.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(prepare(args.raw_dir, args.measurements, args.output))


if __name__ == "__main__":
    main()
