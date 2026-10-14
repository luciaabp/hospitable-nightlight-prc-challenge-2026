"""Relaxed own off-block from the adsb.lol cuts: more departures with ADS-B for the base model (variant D2sub).

    python -m components.base_model.adsb_relaxed --procs 16      # -> adsb/relaxed_events.parquet

The base model only keeps an event when it sees the same aircraft go from ground to air within 120 s, so it loses
whole taxis where runway reception is patchy. Here the takeoff comes from MVT_TIME_UTC_mvt: aircraft by callsign within
1 h of MVT, ground segment (gaps < 600 s) ending at the last ground point before MVT + 120 s. ``extended_events`` adds
the departures without a base event and ``rx_last_gap < max_gap`` to the base events (variants.py d2sub).
``load_cut`` and ``departures`` are shared with the own off-block detector (components/offblock).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import ADSB, ID, RAW

SEG_GAP = 600.0       # largest gap inside the ground segment
MATCH_SEC = 3600.0    # callsign seen within 1 h of MVT
MAX_LAST_GAP = 1800.0  # the segment must end within 30 min of MVT
LOOKBACK = 4 * 3600.0
COLS = ["rx_taxi", "rx_last_gap", "rx_taxi_move", "rx_n", "rx_gap_max", "rx_lat0", "rx_lon0", "rx_gs0", "rx_frac_mlat"]


def load_cut(cut_dir: Path, dia: str) -> pd.DataFrame:
    """The day's cut plus the last 4 h of the previous day (taxis starting before midnight)."""
    start = pd.Timestamp(dia, tz="UTC").timestamp()
    parts = []
    yesterday = cut_dir / f"{(date.fromisoformat(dia) - timedelta(days=1)).isoformat()}.parquet"
    if yesterday.exists():
        parts.append(pd.read_parquet(yesterday, filters=[("t", ">=", start - LOOKBACK)]))
    parts.append(pd.read_parquet(cut_dir / f"{dia}.parquet"))
    c = pd.concat(parts, ignore_index=True)
    c["apt"] = c["apt"].astype(str)
    c["callsign"] = c["callsign"].astype(str).str.strip()
    return c


def relaxed(cut: pd.DataFrame, deps: pd.DataFrame) -> pd.DataFrame:
    """deps: MVT_ID_mvt, apt, mvt (epoch s), callsign. One row per departure with a ground segment."""
    seen = cut.groupby(["apt", "callsign", "icao"], observed=True)["t"].agg(["min", "max"]).reset_index()
    m = deps.merge(seen, on=["apt", "callsign"])
    m["d"] = np.maximum(0, np.maximum(m["min"] - m["mvt"], m["mvt"] - m["max"]))
    m = m[m["d"] < MATCH_SEC].sort_values("d").drop_duplicates(ID)[[ID, "mvt", "icao"]]
    ground = cut[cut["chao"]].sort_values(["icao", "t"])
    by_aircraft = {k: (g["t"].to_numpy(), g["lat"].to_numpy(), g["lon"].to_numpy(), g["gs"].to_numpy(float),
                     (g["fonte"].astype(str) == "mlat").to_numpy())
                 for k, g in ground.groupby("icao", observed=True)}
    out = []
    for mid, mvt, icao in m.itertuples(index=False):
        if icao not in by_aircraft:
            continue
        t, lat, lon, gs, mlat = by_aircraft[icao]
        k = np.searchsorted(t, mvt + 120, side="right") - 1
        if k < 0 or mvt - t[k] > MAX_LAST_GAP:
            continue
        j = k
        while j > 0 and t[j] - t[j - 1] < SEG_GAP and mvt - t[j - 1] < LOOKBACK:
            j -= 1
        moving = np.flatnonzero(gs[j:k + 1] > 1.0)
        out.append((mid, mvt - t[j], mvt - t[k], mvt - t[j + moving[0]] if moving.size else np.nan, k - j + 1,
                    float(np.diff(t[j:k + 1]).max()) if k > j else 0.0, lat[j], lon[j], gs[j],
                    float(mlat[j:k + 1].mean())))
    return pd.DataFrame(out, columns=[ID, *COLS])


def departures(raw: Path) -> pd.DataFrame:
    """Departures of 2025 and the ranking: ID, airport, MVT (epoch s), day and callsign (or flight number)."""
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    paths = sorted(raw.glob("training_2025-*.parquet")) + [raw / "ranking.parquet"]
    d = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    d = d[d["PHASE_mvt"] == "DEP"]
    mvt = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    return pd.DataFrame({ID: d[ID].to_numpy(), "apt": d["ADEP_mvt"].astype(str).to_numpy(),
                         "mvt": (mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(),
                         "dia": mvt.dt.strftime("%Y-%m-%d").to_numpy(),
                         "callsign": d["CALLSIGN_flt"].fillna(d["FLIGHT_mvt"]).astype(str).str.strip().to_numpy()})


def _day(args) -> pd.DataFrame:
    cut_dir, day, deps = args
    return relaxed(load_cut(cut_dir, day), deps)


def events(procs: int) -> Path:
    cut_dir, out = ADSB / "cut", ADSB / "relaxed_events.parquet"
    deps = departures(RAW)
    days = sorted(p.stem for p in cut_dir.glob("*.parquet"))
    jobs = [(cut_dir, d, g.drop(columns="dia")) for d, g in deps[deps["dia"].isin(days)].groupby("dia")]
    with ProcessPoolExecutor(procs) as ex:
        r = pd.concat(ex.map(_day, jobs), ignore_index=True).drop_duplicates(ID)
    r.to_parquet(out, index=False)
    print(f"{len(r):,} departures with a ground segment ({len(jobs)} days) -> {out}")
    return out


def extended_events(max_gap: float) -> pd.DataFrame:
    """Base events plus the relaxed segments of departures without one whose last ground point is < max_gap s."""
    ev = pd.read_parquet(ADSB / "events.parquet")
    rx = pd.read_parquet(ADSB / "relaxed_events.parquet")
    rx = rx[(rx["rx_last_gap"] < max_gap) & ~rx[ID].isin(ev[ID])]
    add = pd.DataFrame({ID: rx[ID], "adsb_taxi": rx["rx_taxi"], "adsb_taxi_move": rx["rx_taxi_move"],
                        "adsb_gs0": rx["rx_gs0"], "adsb_takeoff_err": -rx["rx_last_gap"],
                        "adsb_n_chao": rx["rx_n"].astype(float), "adsb_gap_max": rx["rx_gap_max"],
                        "adsb_frac_mlat": rx["rx_frac_mlat"], "adsb_lat0": rx["rx_lat0"], "adsb_lon0": rx["rx_lon0"]})
    out = pd.concat([ev, add[ev.columns]], ignore_index=True)
    if out[ID].duplicated().any():
        raise SystemExit("MVT_ID_mvt repeated in the extended events")
    print(f"events {len(ev):,} -> {len(out):,} (+{len(add):,} relaxed, last ground point < {max_gap:.0f} s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--procs", type=int, default=16)
    events(ap.parse_args().procs)


if __name__ == "__main__":
    main()
