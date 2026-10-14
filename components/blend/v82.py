"""Candidate v82: base-model bag, context model v70, LIRF rules, M2 bag and specialist v62.

    python -m components.blend.v82 --seeds 0 ... 31 --v62-adsb 0.022 --v62-lirf 0.182 --m2 0.3 --m2-bag --ventana \
        --out v82_m2_03_s32_m2bag_ventana

1. Base model: mean of the FG4 seeds on own-ADS-B rows; mean of D2sub and the FG4 seeds elsewhere.
2. components/blend/blend.py: 30 % of v70 off ADS-B (seven airports) and the LIRF rotation increment (Jan + Jul).
3. M2 (mean of its seed bag with --m2-bag) at weight --m2 on the own-ADS-B rows of the nine airports.
4. Specialist v62: 0.182 everywhere, except --v62-adsb on the own-ADS-B rows of the nine and --v62-lirf on LIRF.
5. LIRF departures without NM: capped at the largest possible label, then the destination rule
   (components/lirf/lirf_destination.py).
6. --ventana: departures without NM whose destination arrival bounds the label (>= MVT - LOBT + 3606 s) are raised to
   that bound + 300 s.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CANDIDATES, FG4_EVENTS, ID, LOBT_WINDOW_SEC, NINE, PRED, RAW, ROOT, SUBMISSIONS, VARIANTS,
                    fg4_seed_file, own_adsb, prediction)
from components.blend import blend
from components.lirf import lirf_destination

HERE = Path(__file__).resolve().parent
M2_DIR = ROOT / "components/m2"
V70 = ROOT / "outputs/pipeline/pipeline_20261003_anchor/submission.parquet"
LIRF_ROTATION = ROOT / "outputs/final/lirf_v73.parquet"
ROTATION_MONTHS = ("2026-01", "2026-02", "2026-06", "2026-07")
V62_W = 0.182
WINDOW_AIRPORTS = ("LTFM", "LIRF", "EGLL", "EDDM")
WINDOW_MIN_TAXI, WINDOW_MARGIN = 240.0, 300.0


def lirf_cap(ids: np.ndarray, p: np.ndarray) -> np.ndarray:
    """LIRF departures without NM delayed > 2 h: never above the largest possible label."""
    rk = pd.read_parquet(RAW / "ranking.parquet")
    rk = rk[rk.PHASE_mvt == "DEP"].set_index(ID).loc[ids]
    D = (rk.MVT_TIME_UTC_mvt - rk.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    sel = (rk.ADEP_mvt == "LIRF").to_numpy() & rk.FLIGHT_ID_mvt.isna().to_numpy() & (D > 2 * 3600)
    top = np.where(D > 12 * 3600, np.maximum(D, 86400 + 3600), D + 300)
    return np.where(sel & (p > top), top, p)


def window_bound(ids: np.ndarray) -> np.ndarray:
    """Lower bound of the label (NaN where it does not apply)."""
    cols = [ID, "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
            "LOBT_flt"]
    d = pd.read_parquet(RAW / "ranking.parquet", columns=cols)
    dep = d[(d.PHASE_mvt == "DEP") & d.FLIGHT_ID_mvt.isna() & d.FLIGHT_mvt.notna() & d.ADEP_mvt.isin(WINDOW_AIRPORTS)]
    arr = d[(d.PHASE_mvt == "ARR") & d.FLIGHT_ID_mvt.notna()]
    arr = arr[["ADEP_mvt", "ADES_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]].add_prefix("a_")
    found = []
    for key in ("a_CALLSIGN_flt", "a_FLIGHT_mvt"):
        m = dep[[ID, "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]].merge(
            arr, left_on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], right_on=[key, "a_ADEP_mvt", "a_ADES_mvt"])
        h = (m.a_MVT_TIME_UTC_mvt - m.MVT_TIME_UTC_mvt).dt.total_seconds() / 3600
        found.append(m[(h > 0.3) & (h < 8.0)])
    # unique labels: the two merges reuse index values, and .loc[m.index] below would pick up both rows
    m = pd.concat(found, ignore_index=True).sort_values("a_MVT_TIME_UTC_mvt").drop_duplicates(ID)
    gap =(m.MVT_TIME_UTC_mvt - m.a_LOBT_flt).dt.total_seconds()
    m = m[(gap - LOBT_WINDOW_SEC) < WINDOW_MIN_TAXI]
    lb = (gap + LOBT_WINDOW_SEC).loc[m.index]
    return pd.Series(lb.to_numpy(), index=m[ID].to_numpy()).reindex(ids).to_numpy(float)


def build(seeds, v62_adsb: float, v62_lirf: float, m2: float, m2_bag: bool, ventana: bool, out: Path) -> Path:
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    ids = tpl[ID].to_numpy()
    d2 = prediction(VARIANTS / "D2sub/submissions/replica_v37.parquet", ids)
    s = [prediction(fg4_seed_file(k), ids) for k in seeds]
    con = own_adsb(ids)
    hyb = tpl.copy()
    hyb[PRED] = np.where(con, np.mean(s, axis=0), np.mean([d2] + s, axis=0))
    tmp = HERE / f"_hybrid_{out.stem}.parquet"
    hyb.to_parquet(tmp, index=False)
    x, _ = blend.build(tmp, V70, FG4_EVENTS, LIRF_ROTATION, ROTATION_MONTHS)
    tmp.unlink()
    assert (x[ID].to_numpy() == ids).all()
    x = x[PRED].to_numpy()
    rk = pd.read_parquet(RAW / "ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"].set_index(ID).loc[ids]
    nine_adsb = con & rk.ADEP_mvt.isin(NINE).to_numpy()
    lirf = (rk.ADEP_mvt == "LIRF").to_numpy()
    if m2:
        files = [M2_DIR / "pred_ranking.parquet"]
        if m2_bag:
            files += sorted(M2_DIR.glob("pred_ranking_s[0-9]*[0-9].parquet"))
        q = np.mean([prediction(f, ids, "m2") for f in files], axis=0)
        k = nine_adsb & np.isfinite(q)
        x = np.where(k, (1 - m2) * x + m2 * q, x)
        print(f"M2 ({len(files)} seeds) at {m2}: {k.sum():,} rows")
    v62 = prediction(SUBMISSIONS / "submission_v62.parquet", ids)
    keep = 0.818 * x + 0.182 * v62  # literals of v81 (1 - 0.182 is not 0.818 in floating point)
    mix = lambda w: keep if w == V62_W else (1 - w) * x + w * v62  # noqa: E731
    p = np.where(nine_adsb, mix(v62_adsb), np.where(lirf, mix(v62_lirf), keep))
    p = lirf_cap(ids, p)
    assert np.isfinite(p).all() and (p >= 0).all()
    base = HERE / f"_base_{out.stem}.parquet"
    res = tpl.copy()
    res[PRED] = p
    res.to_parquet(base, index=False)
    lirf_destination.produce(base, out, out.with_name(f"{out.stem}_lirf_destination.json"))
    base.unlink()
    if ventana:
        res = pd.read_parquet(out)
        assert (res[ID].to_numpy() == ids).all()
        lb = window_bound(ids)
        p = res[PRED].to_numpy(float)
        hit = np.isfinite(lb) & (p < lb)
        res[PRED] = np.where(hit, lb + WINDOW_MARGIN, p)
        res.to_parquet(out, index=False)
        print(f"ventana: {hit.sum()} rows raised to the bound + {WINDOW_MARGIN:.0f} s "
              f"(rms of the change {np.sqrt(((res[PRED].to_numpy() - p) ** 2).mean()):.3f} s)")
    print(f"{out.name}: sha {hashlib.sha256(out.read_bytes()).hexdigest()[:8]}, seeds {len(seeds)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="FG4 seeds (0 is FG4sub)")
    ap.add_argument("--v62-adsb", type=float, default=V62_W)
    ap.add_argument("--v62-lirf", type=float, default=V62_W)
    ap.add_argument("--m2", type=float, default=0.0)
    ap.add_argument("--m2-bag", action="store_true", help="mean of every M2 seed (pred_ranking*_s*.parquet)")
    ap.add_argument("--ventana", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    build(a.seeds, a.v62_adsb, a.v62_lirf, a.m2, a.m2_bag, a.ventana, CANDIDATES / f"{a.out}.parquet")


if __name__ == "__main__":
    main()
