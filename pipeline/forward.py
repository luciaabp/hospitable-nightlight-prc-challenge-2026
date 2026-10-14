"""Batch context per departure (optional): what is occupied and what is coming around each takeoff, with earlier and
**later** flights of the batch.

    python -m pipeline.forward prepare [--output local_data/forward_features]

No label or block time of any departure (departure BLOCK is dropped on reading; arrivals only contribute their
in-block time, visible in the ranking) and no ADS-B of the flight itself. Families (``FORWARD_GROUPS``): ``load``,
``sequence``, ``nm``, ``plan``, ``delay`` and ``adsb`` (ADS-B regressor only).

Output: ``features.parquet`` (one row per training and ranking departure) and ``manifest.json``.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import state
from .config import FORWARD_ADSB_FEATURES, FORWARD_FEATURES, ID, ROOT
from .context import median_before
from .traffic import departures_taxiing

log = logging.getLogger(__name__)

OUTPUT = ROOT / "local_data" / "forward_features"
FALLBACK = 900.0          # s: without a valid AOBT_3, the reference taxi starts 15 min before takeoff
ARRIVAL_MAX = 7200.0      # s: arrivals with taxi-in outside (0, 2 h] do not count as occupancy
NEAR = 3600.0             # s: "the next one" only if it takes off within the next hour
BEHIND_MAX = 12           # flights behind averaged in ``behind``
CLIP = {"fw_gap_next_rwy": 3600, "fw_busy_len_total": 10800, "fw_busy_len_after": 10800, "fw_busy_n_total": 60,
        "fw_busy_n_after": 60}
# Delays with day shifts in the scheduled time are bounded (linear-leaf LightGBM extrapolates).
CLIP_RANGE = {"fw_dep_delay_med_next_30m": (-7200, 21600), "fw_dep_delay_med_next_60m": (-7200, 21600),
              "fw_arr_delay_med_next_60m": (-7200, 21600), "fw_own_minus_delay_next_60m": (-21600, 21600)}
ANOMALIES = {"nm_route": "nm_route", "adsb_rw7": "adsb_rw7", "stop_rw7": "stop_rw7"}


def read_extras(raw_dir: Path = state.CHALLENGE / "raw") -> pd.DataFrame:
    """Scheduled time of every movement and in-block of **arrivals only** (departure BLOCK is the label)."""
    columns = [ID, "PHASE_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    parts = [pd.read_parquet(p, columns=columns) for p in sorted(raw_dir.glob("training_*.parquet"))]
    parts.append(pd.read_parquet(raw_dir / "ranking.parquet", columns=columns))
    raw = pd.concat(parts, ignore_index=True)
    block = raw["BLOCK_TIME_UTC_mvt"].where(raw["PHASE_mvt"] == "ARR")
    return pd.DataFrame({ID: raw[ID].to_numpy(), "sched": state.seconds(raw["SCHED_TIME_UTC_mvt"]),
                         "inblock": state.seconds(block)})


def active(starts: np.ndarray, ends: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """[start, end) intervals containing each reference instant; ``starts`` and ``ends`` sorted."""
    return (np.searchsorted(starts, reference, "right") - np.searchsorted(ends, reference, "right")).astype(float)


def runway_block(t, aobt, x, tau, anomalies: dict) -> dict:
    """Features of one runway's departures, sorted by ``t``. "Next" is always strictly later."""
    m = len(t)
    valid = np.isfinite(x)
    own_off = (valid & (x > 0)).astype(float)
    starts, ends = np.sort(aobt[valid]), np.sort(t[valid])
    out = {"fw_dep_active_off_rwy": active(starts, ends, tau) - own_off,
           "fw_dep_active_tko_rwy": active(starts, ends, t)}
    after = np.searchsorted(t, t, "right")      # first index with a strictly later takeoff
    before = np.searchsorted(t, t, "left")      # number of strictly earlier takeoffs
    has = after < m
    ni = np.minimum(after, m - 1)
    gap = np.where(has, t[ni] - t, np.nan)
    out["fw_gap_next_rwy"] = gap
    for minutes in (5, 10, 15):
        out[f"fw_dep_rwy_next_{minutes}m"] = (np.searchsorted(t, t + minutes * 60, "right") - after).astype(float)
    out["fw_dep_rwy_win_10m"] = (np.searchsorted(t, t + 300, "left") - np.searchsorted(t, t - 300, "left")
                                 - 1).astype(float)
    # busy period: takeoffs chained with gaps <= BUSY_GAP, forwards and backwards
    end_t, end_i = t.copy(), np.arange(m)
    for i in range(m - 1, -1, -1):
        j = after[i]
        if j < m and t[j] - t[i] <= state.BUSY_GAP:
            end_t[i], end_i[i] = end_t[j], end_i[j]
    start_t, start_i = t.copy(), np.arange(m)
    prev = before - 1
    for i in range(m):
        p = prev[i]
        if p >= 0 and t[i] - t[p] <= state.BUSY_GAP:
            start_t[i], start_i[i] = start_t[p], start_i[p]
    out["fw_busy_len_after"] = end_t - t
    out["fw_busy_n_after"] = np.maximum(end_i - (after - 1), 0).astype(float)
    out["fw_busy_len_total"] = end_t - start_t
    out["fw_busy_n_total"] = (end_i - start_i + 1).astype(float)
    near = has & (gap <= NEAR)
    out["fw_x_next"] = np.where(near, x[ni], np.nan)
    out["fw_aobt_gap_next"] = np.where(near, np.clip(aobt[ni] - aobt, -3600, 3600), np.nan)
    h1, h30 = np.searchsorted(t, t + NEAR, "right"), np.searchsorted(t, t + 1800, "right")
    next3_hi = np.minimum(after + 3, h1)
    for name, v in anomalies.items():
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        cn = np.concatenate([[0], np.cumsum(np.isfinite(v))])

        def mean(lo, hi):
            n = cn[hi] - cn[lo]
            return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)

        out[f"fw_{name}__next"] = np.where(near, v[ni], np.nan)
        out[f"fw_{name}__next3"] = mean(after, next3_hi)
        out[f"fw_{name}__fwd30"] = mean(after, h30)
        # those already taxiing when B takes off: off-block <= t_B and later takeoff (queue behind B)
        total, count = np.zeros(m), np.zeros(m)
        for k in range(BEHIND_MAX):
            idx = after + k
            index = np.minimum(idx, m - 1)
            ok = (idx < m) & valid[index] & (aobt[index] <= t) & np.isfinite(v[index])
            total += np.where(ok, v[index], 0.0)
            count += ok
        out[f"fw_{name}__behind"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def airport_features(d: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """Features of an airport's departures ``d`` (anomalies already computed); ``a``: its arrivals."""
    d = d.sort_values("t", kind="stable").reset_index(drop=True)
    n = len(d)
    t, aobt, x = d.t.to_numpy(float), d.aobt.to_numpy(float), d.x.to_numpy(float)
    valid = np.isfinite(x)
    tau = np.where(valid, aobt, t - FALLBACK)
    own_off = (valid & (x > 0)).astype(float)
    starts, ends = np.sort(aobt[valid]), np.sort(t[valid])
    out = {ID: d[ID].to_numpy(), "fw_dep_behind_off": d["fw_dep_behind_off"].to_numpy(float),
           "fw_dep_active_off": active(starts, ends, tau) - own_off, "fw_dep_active_tko": active(starts, ends, t)}
    land, inblock = a.t.to_numpy(float), a.inblock.to_numpy(float)
    ok = np.isfinite(inblock) & (inblock > land) & (inblock - land <= ARRIVAL_MAX)
    arr_starts, arr_ends = np.sort(land[ok]), np.sort(inblock[ok])
    out["fw_arr_active_off"] = active(arr_starts, arr_ends, tau)
    out["fw_arr_active_tko"] = active(arr_starts, arr_ends, t)
    out["fw_dep_apt_next_10m"] = (np.searchsorted(t, t + 600, "right") - np.searchsorted(t, t, "right")).astype(float)
    sched = d.sched.to_numpy(float)
    has_sched = np.isfinite(sched)
    dep_sched = np.sort(sched[has_sched])
    arr_sched = np.sort(a.sched.to_numpy(float)[np.isfinite(a.sched.to_numpy(float))])
    low = tau - 1800
    out["fw_sched_dep_prev_30m"] = (np.searchsorted(dep_sched, tau, "left") - np.searchsorted(dep_sched, low, "left")
                                    - (has_sched & (sched >= low) & (sched < tau))).astype(float)
    for minutes in (15, 30, 60):
        high = tau + minutes * 60
        out[f"fw_sched_dep_next_{minutes}m"] = (
            np.searchsorted(dep_sched, high, "right") - np.searchsorted(dep_sched, tau, "right")
            - (has_sched & (sched > tau) & (sched <= high))).astype(float)
    for minutes in (30, 60):
        high = tau + minutes * 60
        out[f"fw_sched_arr_next_{minutes}m"] = (np.searchsorted(arr_sched, high, "right")
                                                - np.searchsorted(arr_sched, tau, "right")).astype(float)
    # forward airport delay: ``median_before`` on reversed time looks at events in (t, t + window]
    t_ns, arr_ns = (t * 1e9).astype(np.int64), (a.t.to_numpy(float) * 1e9).astype(np.int64)
    dep_delay, arr_delay = t - sched, a.t.to_numpy(float) - a.sched.to_numpy(float)
    out["fw_dep_delay_med_next_30m"] = median_before(-t_ns, dep_delay, -t_ns, 30)
    out["fw_dep_delay_med_next_60m"] = median_before(-t_ns, dep_delay, -t_ns, 60)
    out["fw_arr_delay_med_next_60m"] = median_before(-arr_ns, arr_delay, -t_ns, 60)
    out["fw_own_minus_delay_next_60m"] = dep_delay - out["fw_dep_delay_med_next_60m"]
    names = ["fw_dep_active_off_rwy", "fw_dep_active_tko_rwy", "fw_gap_next_rwy", "fw_dep_rwy_next_5m",
             "fw_dep_rwy_next_10m", "fw_dep_rwy_next_15m", "fw_dep_rwy_win_10m", "fw_busy_len_after",
             "fw_busy_n_after", "fw_busy_len_total", "fw_busy_n_total", "fw_x_next", "fw_aobt_gap_next"] + [
        f"fw_{a_}__{k}" for a_ in ANOMALIES for k in ("next", "next3", "fwd30", "behind")]
    columns = {name: np.full(n, np.nan) for name in names}
    runway = d.runway.to_numpy()
    for code in np.unique(runway):
        if code == "NA":
            continue
        rows = np.flatnonzero(runway == code)
        block = runway_block(t[rows], aobt[rows], x[rows], tau[rows], {k: d[c].to_numpy(float)[rows]
                                                                        for k, c in ANOMALIES.items()})
        for name, values in block.items():
            columns[name][rows] = values
    out.update(columns)
    result = pd.DataFrame(out)
    for column, high in CLIP.items():
        result[column] = result[column].clip(upper=high)
    for column, (low, high) in CLIP_RANGE.items():
        result[column] = result[column].clip(low, high)
    return result


def forward_features(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    """Features of every departure of the batch."""
    dep = state.anomalies(dep)
    dep["fw_dep_behind_off"] = departures_taxiing(dep, causal=False) - departures_taxiing(dep, causal=True)
    parts = [airport_features(dep[dep.airport == ap], arr[arr.airport == ap]) for ap in sorted(dep.airport.unique())]
    out = pd.concat(parts, ignore_index=True)
    return dep[[ID]].merge(out[[ID] + FORWARD_FEATURES + FORWARD_ADSB_FEATURES], on=ID, how="left", validate="1:1")


def prepare(raw_dir: Path = state.CHALLENGE / "raw", measurements: Path = state.MEASUREMENTS,
            output: Path = OUTPUT) -> Path:
    from .data import sha256
    dep, arr = state.read_movements(raw_dir, measurements)
    extras = read_extras(raw_dir)
    dep = dep.merge(extras[[ID, "sched"]], on=ID, how="left", validate="1:1")
    arr = arr.merge(extras, on=ID, how="left", validate="1:1")
    features = forward_features(dep, arr)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "features.parquet"
    features.to_parquet(target, index=False)
    key = dep.airport + "|" + dep.month
    coverage = features.drop(columns=ID).notna().groupby(key.to_numpy()).mean()
    manifest = {"output": str(target), "output_sha256": sha256(target), "rows": int(len(features)),
                "fallback_sec": FALLBACK, "clip": CLIP,
                "inputs": {str(p): sha256(p) for p in sorted(raw_dir.glob("training_*.parquet"))
                           + [raw_dir / "ranking.parquet", measurements]},
                "coverage": coverage[["fw_dep_active_off", "fw_gap_next_rwy", "fw_nm_route__next3",
                                      "fw_adsb_rw7__next3", "fw_sched_dep_next_30m"]].round(4).to_dict(orient="index")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%s: %d departures", target, len(features))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.forward", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--raw-dir", type=Path, default=state.CHALLENGE / "raw")
    parser.add_argument("--measurements", type=Path, default=state.MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(prepare(args.raw_dir, args.measurements, args.output))


if __name__ == "__main__":
    main()
