"""Surface traffic counts around the estimated departure taxi interval."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import movement_features as F
from dataset_cache import DATA
from external_data import BLOCK, TIME
from weather_rotation import _raw_files

MARGIN = pd.Timedelta('6h')
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
ARR_COLS = ['PHASE_mvt', 'ADES_mvt', TIME, BLOCK]
COLS = ['sup_dep_taxiando_push', 'sup_dep_decolam_durante', 'sup_dep_decolam_durante_rwy', 'sup_arr_taxiando_push', 'sup_arr_pousam_durante', 'sup_dep_decolam_durante_min', 'sup_arr_pousam_durante_min']

def _seconds(s: pd.Series) -> np.ndarray:
    t = pd.to_datetime(s, utc=True, errors='coerce')
    return (t - EPOCH).dt.total_seconds().to_numpy(float)

def _groups(key: np.ndarray) -> dict:
    return pd.Series(np.arange(key.size)).groupby(key, sort=False).indices

def _arrivals(apts: set[str], t0: pd.Timestamp, t1: pd.Timestamp, data: Path) -> pd.DataFrame:
    parts = []
    for start, end, p in _raw_files(data):
        if end <= t0 - MARGIN or start > t1:
            continue
        raw = pd.read_parquet(p, columns=ARR_COLS)
        arr = raw[(raw['PHASE_mvt'] == 'ARR') & raw['ADES_mvt'].isin(apts)]
        del raw
        parts.append(pd.DataFrame({'apt': arr['ADES_mvt'].astype(str).to_numpy(), 'L': _seconds(arr[TIME]), 'B': _seconds(arr[BLOCK])}))
    if not parts:
        return pd.DataFrame({'apt': pd.Series(dtype='object'), 'L': pd.Series(dtype=float), 'B': pd.Series(dtype=float)})
    ch = pd.concat(parts, ignore_index=True).dropna()
    return ch[ch['B'] >= ch['L']]

def _count_departures(out: dict, apt: np.ndarray, rwy: np.ndarray, push: np.ndarray, mvt: np.ndarray, taxi: np.ndarray) -> None:
    for pos in _groups(apt).values():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        oi, mi = (push[pos], mvt[pos])
        order_o, order_m = (np.sort(oi), np.sort(mi))
        out['sup_dep_taxiando_push'][pos] = np.searchsorted(order_o, oi, 'right') - np.searchsorted(order_m, oi, 'right') - (taxi[pos] > 0)
        out['sup_dep_decolam_durante'][pos] = np.maximum(np.searchsorted(order_m, mi, 'left') - np.searchsorted(order_m, oi, 'right'), 0)
    for pos in _groups(rwy).values():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        order_m = np.sort(mvt[pos])
        out['sup_dep_decolam_durante_rwy'][pos] = np.maximum(np.searchsorted(order_m, mvt[pos], 'left') - np.searchsorted(order_m, push[pos], 'right'), 0)

def _count_arrivals(out: dict, apt: np.ndarray, push: np.ndarray, mvt: np.ndarray, ch: pd.DataFrame) -> None:
    empty = np.empty(0)
    airport = {name: (np.sort(g['L'].to_numpy()), np.sort(g['B'].to_numpy())) for name, g in ch.groupby('apt', sort=False)}
    for name, pos in _groups(apt).items():
        pos = pos[np.isfinite(mvt[pos])]
        if not pos.size:
            continue
        arrivals, blocks = airport.get(name, (empty, empty))
        oi, mi = (push[pos], mvt[pos])
        out['sup_arr_taxiando_push'][pos] = np.searchsorted(arrivals, oi, 'right') - np.searchsorted(blocks, oi, 'right')
        out['sup_arr_pousam_durante'][pos] = np.maximum(np.searchsorted(arrivals, mi, 'left') - np.searchsorted(arrivals, oi, 'right'), 0)

def counts(df: pd.DataFrame, pred: np.ndarray, data: Path=DATA) -> pd.DataFrame:
    n = len(df)
    out = {c: np.full(n, np.nan) for c in COLS}
    if n == 0:
        return pd.DataFrame(out, index=df.index)
    mvt = _seconds(df[TIME])
    taxi = np.clip(np.asarray(pred, float), 0.0, None)
    push = mvt - taxi
    apt = df[F.AIRPORT].astype(str).to_numpy()
    rwy = (df[F.AIRPORT].astype(str) + '|' + df['RUNWAY_mvt'].astype(str)).to_numpy()
    _count_departures(out, apt, rwy, push, mvt, taxi)
    known = np.isfinite(mvt)
    if known.any():
        t0 = EPOCH + pd.to_timedelta(float(np.nanmin(push[known])), unit='s')
        t1 = EPOCH + pd.to_timedelta(float(mvt[known].max()), unit='s')
        ch = _arrivals(set(apt[known]), t0, t1, data)
        if not ch.empty:
            _count_arrivals(out, apt, push, mvt, ch)
    per_minute = np.where(taxi > 0, taxi / 60.0, np.nan)
    out['sup_dep_decolam_durante_min'] = out['sup_dep_decolam_durante'] / per_minute
    out['sup_arr_pousam_durante_min'] = out['sup_arr_pousam_durante'] / per_minute
    return pd.DataFrame(out, index=df.index)
