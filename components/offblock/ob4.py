"""Off-block candidates per departure from adsb.lol, to choose the ob2 rule with the 2025 labels.

Same aircraft and ground segment as ``ob2.py`` (relaxed: gaps < 600 s, last point < 1800 s from MVT). Seconds before
MVT of the first point, first moving point and end of the first/last stop >= 15 min, plus segment descriptors.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW
from components.offblock.ob2 import LONG_STOP, LOOKBACK, MATCH, MAX_LAST_GAP, SEG_GAP, STOP_KT, departures, load_cut

COLS = ["c_first", "c_move", "c_first_long", "c_last_long", "gs0", "n_long", "d_long", "d_first_long", "init_stop",
        "max_gs_before_long", "last_gap"]
TRACE_COLUMNS = [f"tr_{window}_{name}" for window in (60, 180)
                 for name in ("n", "span", "gap", "dx", "dy", "path", "speed", "stop", "observed")]


def initial_motion(times, latitudes, longitudes, speeds):
    """Summarise the existing segment start; gaps over 30 s do not imply observed motion."""
    times, latitudes, longitudes, speeds = [np.asarray(values, dtype=float)
                                           for values in (times, latitudes, longitudes, speeds)]
    if len({len(values) for values in (times, latitudes, longitudes, speeds)}) != 1:
        raise ValueError("Trace arrays must have equal lengths")
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError("Trace timestamps must be finite and sorted")
    result = dict.fromkeys(TRACE_COLUMNS, np.nan)
    if not len(times):
        return result
    for window in (60, 180):
        count = int(np.searchsorted(times, times[0] + window, side="right"))
        prefix = f"tr_{window}_"
        elapsed = np.diff(times[:count])
        valid_positions = (np.isfinite(latitudes[:count]) & np.isfinite(longitudes[:count])
                           & (np.abs(latitudes[:count]) <= 90) & (np.abs(longitudes[:count]) <= 180))
        latitude = np.where(valid_positions, latitudes[:count], np.nan)
        longitude = np.where(valid_positions, longitudes[:count], np.nan)
        north = (latitude - latitude[0]) * 111200
        east = (longitude - longitude[0]) * 111200 * np.cos(np.radians(latitude[0]))
        speed = speeds[:count]
        speed_ok = np.isfinite(speed) & (speed >= 0)
        observed = (elapsed > 0) & (elapsed <= 30) & speed_ok[:-1] & speed_ok[1:]
        connected = (elapsed > 0) & (elapsed <= 30) & valid_positions[:-1] & valid_positions[1:]
        values = {
            "n": count, "span": times[count - 1] - times[0],
            "gap": elapsed.max() if count > 1 else np.nan,
            "dx": east[-1] if count > 1 else np.nan,
            "dy": north[-1] if count > 1 else np.nan,
            "path": np.hypot(np.diff(east), np.diff(north))[connected].sum() if connected.any() else np.nan,
            "speed": np.median(speed[speed_ok]) if speed_ok.any() else np.nan,
            "stop": elapsed[observed & (speed[:-1] <= STOP_KT) & (speed[1:] <= STOP_KT)].sum() if observed.any() else np.nan,
            "observed": elapsed[observed].sum() / window,
        }
        result.update({prefix + name: float(value) for name, value in values.items()})
    return result


def _m(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat2 - lat1) * 111.2e3, (lon2 - lon1) * 111.2e3 * np.cos(np.radians(lat1))))


def detect(cut, deps, *, trace_features=False):
    vis = cut.groupby(["apt", "callsign", "icao"], observed=True)["t"].agg(["min", "max"]).reset_index()
    m = deps.merge(vis, on=["apt", "callsign"])
    m["d"] = np.maximum(0, np.maximum(m["min"] - m["mvt"], m["mvt"] - m["max"]))
    m = m[m.d < MATCH].sort_values("d").drop_duplicates("MVT_ID_mvt")[["MVT_ID_mvt", "mvt", "icao"]]
    g = cut[cut.chao].sort_values(["icao", "t"])
    by = {k: (v.t.to_numpy(), v.lat.to_numpy(), v.lon.to_numpy(), v.gs.to_numpy(float)) for k, v in g.groupby("icao", observed=True)}
    out = []
    trace_rows = []
    for mid, mvt, icao in m.itertuples(index=False):
        if icao not in by:
            continue
        t, lat, lon, gs = by[icao]
        k = np.searchsorted(t, mvt + 120, side="right") - 1
        if k < 0 or mvt - t[k] > MAX_LAST_GAP:
            continue
        j = k
        while j > 0 and t[j] - t[j - 1] < SEG_GAP and mvt - t[j - 1] < LOOKBACK:
            j -= 1
        tt, la, lo = t[j:k + 1], lat[j:k + 1], lon[j:k + 1]
        gg = np.nan_to_num(gs[j:k + 1], nan=0.0)
        n = len(tt)
        mov = gg > STOP_KT
        runs, i = [], 0
        while i < n:
            if not mov[i]:
                s = i
                while i < n and not mov[i]:
                    i += 1
                end_t = tt[i] if i < n else tt[n - 1]
                runs.append((s, i, end_t - tt[s]))
            else:
                i += 1
        longs = [r for r in runs if r[2] >= LONG_STOP and r[1] < n]
        im = np.flatnonzero(mov)
        c_move = mvt - tt[im[0]] if im.size else np.nan
        init_stop = (tt[im[0]] - tt[0]) if (im.size and not mov[0]) else 0.0
        if longs:
            (s1, e1, _), (s2, e2, _) = longs[0], longs[-1]
            row = (mvt - tt[e1], mvt - tt[e2], len(longs), _m(la[s1], lo[s1], la[s2], lo[s2]),
                   _m(la[0], lo[0], la[s1], lo[s1]), float(gg[:s1 + 1].max()))
        else:
            row = (np.nan, np.nan, 0, np.nan, np.nan, np.nan)
        out.append((mid, mvt - tt[0], c_move, *row[:2], float(gg[0]), *row[2:5], init_stop, row[5], mvt - tt[-1]))
        if trace_features:
            trace_rows.append(initial_motion(tt, la, lo, gs[j:k + 1]))
    result = pd.DataFrame(out, columns=["MVT_ID_mvt", "c_first", "c_move", "c_first_long", "c_last_long", "gs0", "n_long",
                                       "d_long", "d_first_long", "init_stop", "max_gs_before_long", "last_gap"])
    if trace_features:
        result = pd.concat([result, pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)], axis=1)
    return result


def _job(a):
    cut_dir, day, deps = a
    return detect(load_cut(cut_dir, day), deps)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    cut_dir, out, procs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    deps = departures(RAW)
    days = sorted(p.stem for p in cut_dir.glob("*.parquet"))
    jobs = [(cut_dir, d, g.drop(columns="dia")) for d, g in deps[deps.dia.isin(days)].groupby("dia")]
    with ProcessPoolExecutor(procs) as ex:
        r = pd.concat(ex.map(_job, jobs), ignore_index=True).drop_duplicates("MVT_ID_mvt")
    r.to_parquet(out, index=False)
    print(len(r), "->", out)
