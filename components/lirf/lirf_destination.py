"""LIRF departures without NM: the same flight's arrival at its destination decides between SCHED copy and real taxi.

Used by components/blend/v82.py (``produce``). NM is matched to a departure only if |BLOCK - LOBT| <= 3606 s, so a
departure without NM is outside that window. When the destination is one of the ten airports its arrival row carries
NM: a copy (BLOCK = SCHED) fits only if |SCHED - LOBT| > 3606, a real taxi (BLOCK ~ AOBT_3) only if
AOBT_3 - LOBT > ~3000. In 2025 the rule decides 156 of 174 linked departures and gets 154 right. Prediction
(d = MVT - SCHED, rA = MVT - AOBT_3 of the arrival): copy 0.897 d + 0.103 rA; real 0.992 rA + 0.008 d. Other rows keep
the base.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ID, LOBT_WINDOW_SEC, PRED, RAW

COLS = [ID, "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "CALLSIGN_flt", "LOBT_flt", "AOBT_3_flt"]
TIMES = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
AIRPORT = "LIRF"
LANDING_H = (0.3, 8.0)            # landing at destination after the LIRF takeoff
REAL_MIN_GAP = 3000               # AOBT_3 - LOBT above this: the real block is outside the window
TAXI_RANGE = (120, 3600)          # plausible rA; outside, the link is another flight
DELAY_RANGE = (1800, 12 * 3600)   # the tail (> 12 h) follows the Rome rule and dayshift
P_COPY, P_REAL = 26 / 29, 130 / 131


def load(files) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    for c in TIMES:
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


def link(d: pd.DataFrame) -> pd.DataFrame:
    """LIRF departures without NM with the same flight's arrival (with NM) at destination: its LOBT and AOBT_3."""
    dep = d[(d.PHASE_mvt == "DEP") & (d.ADEP_mvt == AIRPORT) & d.FLIGHT_ID_mvt.isna() & d.FLIGHT_mvt.notna()]
    arr = d[(d.PHASE_mvt == "ARR") & (d.ADEP_mvt == AIRPORT) & d.FLIGHT_ID_mvt.notna()]
    arr = arr[["CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]].add_prefix("a_")
    keep = [ID, "FLIGHT_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    found = []
    for key in ("a_CALLSIGN_flt", "a_FLIGHT_mvt"):  # the destination keeps the ATC callsign or the flight number
        m = dep[keep].merge(arr, left_on="FLIGHT_mvt", right_on=key)
        h = (m.a_MVT_TIME_UTC_mvt - m.MVT_TIME_UTC_mvt).dt.total_seconds() / 3600
        found.append(m[(h > LANDING_H[0]) & (h < LANDING_H[1])])
    m = pd.concat(found).sort_values("a_MVT_TIME_UTC_mvt").drop_duplicates(ID).drop(columns=["a_CALLSIGN_flt", "a_FLIGHT_mvt"])
    s = lambda x: x.dt.total_seconds()  # noqa: E731
    m["delay"] = s(m.MVT_TIME_UTC_mvt - m.SCHED_TIME_UTC_mvt)
    m["g_sched"] = s(m.SCHED_TIME_UTC_mvt - m.a_LOBT_flt)
    m["g_aobt"] = s(m.a_AOBT_3_flt - m.a_LOBT_flt)
    m["taxi_aobt"] = s(m.MVT_TIME_UTC_mvt - m.a_AOBT_3_flt)
    return m.reset_index(drop=True)


def rule(m: pd.DataFrame) -> pd.DataFrame:
    """Cell of each linked departure (copy, real or none) and its prediction."""
    m = m.copy()
    ok = m.taxi_aobt.between(*TAXI_RANGE) & (m.delay > DELAY_RANGE[0]) & (m.delay < DELAY_RANGE[1])
    copy_fits = m.g_sched.abs() > LOBT_WINDOW_SEC
    real_fits = m.g_aobt > REAL_MIN_GAP
    m["rule"] = np.select([ok & copy_fits & ~real_fits, ok & real_fits & ~copy_fits], ["copy", "real"], "")
    m["pred_rule"] = np.select(
        [m.rule == "copy", m.rule == "real"],
        [P_COPY * m.delay + (1 - P_COPY) * m.taxi_aobt, P_REAL * m.taxi_aobt + (1 - P_REAL) * m.delay], np.nan)
    return m


def apply(base: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """The base with the rule's prediction on its rows; the rest untouched."""
    r = m[m.rule != ""].set_index(ID).pred_rule
    out = base.copy()
    hit = out[ID].isin(r.index).to_numpy()
    out.loc[hit, PRED] = out.loc[hit, ID].map(r).to_numpy()
    return out


def expected_delta_sse(base: np.ndarray, new: np.ndarray, m: pd.DataFrame) -> float:
    """Expected SSE change if the labels follow the rule's probabilities (two possible values per row)."""
    cp = np.where(m.rule == "copy", P_COPY, 1 - P_REAL)
    e = lambda p: cp * (p - m.delay) ** 2 + (1 - cp) * (p - m.taxi_aobt) ** 2  # noqa: E731
    return float((e(new) - e(base)).sum())


def produce(base_path: Path, out: Path, info_path: Path) -> None:
    d = load([RAW / "ranking.parquet"])
    m = rule(link(d))
    base = pd.read_parquet(base_path)[[ID, PRED]]
    res = apply(base, m)
    changed = (res[PRED].to_numpy() != base[PRED].to_numpy())
    dep = d[d.PHASE_mvt == "DEP"].set_index(ID).loc[base[ID]]
    target = ((dep.ADEP_mvt == AIRPORT) & dep.FLIGHT_ID_mvt.isna()).to_numpy()
    assert not (changed & ~target).any(), "the rule may only change LIRF departures without NM"
    assert np.isfinite(res[PRED]).all() and (res[PRED] >= 0).all()
    hit = m[m.rule != ""].copy()
    hit["base"] = hit[ID].map(base.set_index(ID)[PRED])
    dsse = expected_delta_sse(hit.base.to_numpy(), hit.pred_rule.to_numpy(), hit)
    hit["month"] = hit.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    info = {"base": str(base_path), "sha256": sha, "linked_rows": int(len(m)), "changed_rows": int(changed.sum()),
            "by_month_and_rule": {f"{k[0]} {k[1]}": int(v) for k, v in hit.groupby(["month", "rule"]).size().items()},
            "rms_vs_base": float(np.sqrt(((res[PRED] - base[PRED]) ** 2).mean())),
            "expected_delta_sse": dsse}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=2, ensure_ascii=False))
