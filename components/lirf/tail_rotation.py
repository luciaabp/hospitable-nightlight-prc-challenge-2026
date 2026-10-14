"""Per-aircraft rotation at the ten airports from airborne adsb.lol ADS-B (public data).

    python -m components.lirf.tail_rotation [--out local_data/tail_rotation]

Each organizer flight is matched to the airborne track of its callsign within 30 min of its MVT, which gives the
aircraft; each departure then gets the last arrival of the same aircraft at the same airport in the previous 24 h.
Outputs: flights.parquet (MVT_ID_mvt -> icao, reg) and rotation.parquet (one row per matched departure).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import ADSB, LOCAL_DATA, RAW

CUT = ADSB / "cut"
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
        "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]


def segments(path: Path) -> pd.DataFrame:
    c = pd.read_parquet(path, columns=["icao", "reg", "t", "apt", "callsign"]).dropna(subset=["callsign"])
    c["apt"] = c["apt"].astype(str)
    c["cs"] = c["callsign"].astype(str).str.strip()
    c = c.sort_values(["apt", "cs", "icao", "t"])
    new_segment = (c.apt != c.apt.shift()) | (c.cs != c.cs.shift()) | (c.icao != c.icao.shift()) | (c.t.diff() > 3600)
    c["segment"] = new_segment.cumsum()
    return c.groupby("segment").agg(apt=("apt", "first"), cs=("cs", "first"), icao=("icao", "first"), reg=("reg", "first"),
                                  t0=("t", "min"), t1=("t", "max")).reset_index(drop=True)


def flights() -> pd.DataFrame:
    fs = [*sorted(RAW.glob("training_2025-*.parquet")), RAW / "ranking.parquet"]
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in fs], ignore_index=True)
    d["apt"] = np.where(d.PHASE_mvt == "DEP", d.ADEP_mvt, d.ADES_mvt)
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d["ts"] = (d.MVT_TIME_UTC_mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d["cs"] = d.CALLSIGN_flt.fillna(d.FLIGHT_mvt).astype(str).str.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=LOCAL_DATA / "tail_rotation")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.procs) as ex:
        tr = pd.concat(ex.map(segments, sorted(CUT.glob("*.parquet")), chunksize=4), ignore_index=True)
    d = flights()
    m = d[["MVT_ID_mvt", "apt", "cs", "ts"]].merge(tr, on=["apt", "cs"], how="inner")
    m = m[(m.ts >= m.t0 - 1800) & (m.ts <= m.t1 + 1800)]
    m["dist"] = np.minimum((m.ts - m.t0).abs(), (m.ts - m.t1).abs())
    m = m.sort_values("dist").drop_duplicates("MVT_ID_mvt")
    f = d.merge(m[["MVT_ID_mvt", "icao", "reg"]], on="MVT_ID_mvt", how="left")
    f[["MVT_ID_mvt", "apt", "PHASE_mvt", "icao", "reg"]].to_parquet(a.out / "flights.parquet", index=False)
    f["mes"] = f.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    print("matched departures by airport and month:")
    print(f[f.PHASE_mvt == "DEP"].groupby(["apt", "mes"]).icao.apply(lambda s: s.notna().mean()).unstack().round(2).to_string())

    arr = f[(f.PHASE_mvt == "ARR") & f.icao.notna()][["icao", "apt", "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                                                      "SCHED_TIME_UTC_mvt", "STAND_mvt"]]
    arr = arr.rename(columns={c: "in_" + c for c in arr.columns if c not in ("icao", "apt")}).sort_values("in_MVT_TIME_UTC_mvt")
    dep = f[(f.PHASE_mvt == "DEP") & f.icao.notna()][["MVT_ID_mvt", "icao", "apt", "MVT_TIME_UTC_mvt"]].sort_values("MVT_TIME_UTC_mvt")
    r = pd.merge_asof(dep, arr, left_on="MVT_TIME_UTC_mvt", right_on="in_MVT_TIME_UTC_mvt", by=["icao", "apt"],
                      direction="backward", tolerance=pd.Timedelta("24h"), allow_exact_matches=False)
    r = r[r.in_MVT_ID_mvt.notna()].drop(columns=["MVT_TIME_UTC_mvt"])
    r.to_parquet(a.out / "rotation.parquet", index=False)
    print(f"{len(r):,} departures with an arrival of the same aircraft -> {a.out / 'rotation.parquet'}")


if __name__ == "__main__":
    main()
