"""Cross-fitted airport, stand and runway reference statistics."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import movement_features as F

KEYS = [[F.AIRPORT, 'STAND_mvt', 'RUNWAY_mvt'], [F.AIRPORT, 'STAND_mvt'], [F.AIRPORT, 'RUNWAY_mvt'], [F.AIRPORT]]
MIN_FLIGHTS = 10
CUTOFF_SECONDS = 7200.0
STATISTICS = ['cel_p50', 'cel_p90', 'cel_dp', 'cel_n']
GAP = os.environ.get('BASE_GAP_FEATURES', '0') == '1'
CELL_METERS = 150.0
GAP_KEYS = {'gap_cel': ([F.AIRPORT, 'celula'], 'adsb_taxi')}
GAP_INPUTS = ['adsb_taxi', 'adsb_taxi_move', 'adsb_gs0', 'adsb_lat0', 'adsb_lon0', 'FLIGHT_mvt']
GAP_COLS = [*GAP_KEYS, 'gap_d_stand', 'gap_taxi_mais']
COLS = [*STATISTICS, 'cel_nivel', *(GAP_COLS if GAP else [])]
N_CELLS = len(KEYS)

def _airline(df: pd.DataFrame) -> pd.Series:
    return df['FLIGHT_mvt'].astype('string').str.extract('^([A-Z]{2,3})', expand=False).fillna('?')

def _cell(df: pd.DataFrame) -> np.ndarray:
    lat, lon = (df['adsb_lat0'].to_numpy(float), df['adsb_lon0'].to_numpy(float))
    i = np.floor(lat * 111200.0 / CELL_METERS)
    j = np.floor(lon * 111200.0 * np.cos(np.radians(lat)) / CELL_METERS)
    ok = np.isfinite(i) & np.isfinite(j)
    out = np.full(len(df), '?', dtype=object)
    out[ok] = [f'{a:.0f}|{b:.0f}' for a, b in zip(i[ok], j[ok])]
    return out

def _fit_gap(df: pd.DataFrame, target: np.ndarray) -> dict:
    if not set(GAP_INPUTS) <= set(df.columns):
        return {}
    d = df.assign(cia=_airline(df).to_numpy(), celula=_cell(df))
    tabs = {}
    for name, (cols, ref) in GAP_KEYS.items():
        gap = np.clip(target - d[ref].to_numpy(float), -1800.0, 3600.0)
        ok = np.isfinite(gap)
        g = pd.DataFrame({'k': _key(d, cols)[ok], 'g': gap[ok]}).groupby('k')['g'].agg(['median', 'size'])
        tabs[name] = g.loc[g['size'] >= MIN_FLIGHTS, 'median']
    stationary = (d['adsb_gs0'].to_numpy(float) < 1.0) & np.isfinite(d['adsb_lat0'].to_numpy(float))
    pos = pd.DataFrame({'k': _key(d, [F.AIRPORT, 'STAND_mvt'])[stationary], 'lat': d['adsb_lat0'].to_numpy(float)[stationary], 'lon': d['adsb_lon0'].to_numpy(float)[stationary]}).groupby('k').agg(lat=('lat', 'median'), lon=('lon', 'median'), n=('lat', 'size'))
    tabs['pos'] = pos[pos['n'] >= 3]
    return tabs

def _apply_gap(df: pd.DataFrame, tabs: dict, out: pd.DataFrame) -> None:
    if not tabs or not set(GAP_INPUTS) <= set(df.columns):
        return
    d = df.assign(cia=_airline(df).to_numpy(), celula=_cell(df))
    for name, (cols, _) in GAP_KEYS.items():
        out[name] = pd.Series(_key(d, cols)).map(tabs[name]).to_numpy(float)
    pos = tabs['pos'].reindex(_key(d, [F.AIRPORT, 'STAND_mvt']))
    lat, lon = (pos['lat'].to_numpy(float), pos['lon'].to_numpy(float))
    dy = (d['adsb_lat0'].to_numpy(float) - lat) * 111200.0
    dx = (d['adsb_lon0'].to_numpy(float) - lon) * 111200.0 * np.cos(np.radians(lat))
    out['gap_d_stand'] = np.hypot(dx, dy)
    out['gap_taxi_mais'] = d['adsb_taxi'].to_numpy(float) + out['gap_cel'].to_numpy(float)
    without_adsb = ~np.isfinite(d['adsb_taxi'].to_numpy(float))
    out.loc[without_adsb, GAP_COLS] = np.nan

def _key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    k = df[cols[0]].astype('string').fillna('?')
    for col in cols[1:]:
        k = k + '|' + df[col].astype('string').fillna('?')
    return k.to_numpy()

def fit(df: pd.DataFrame, y) -> list[pd.DataFrame]:
    target = np.clip(np.asarray(y, float), 0.0, CUTOFF_SECONDS)
    ok = np.isfinite(target)
    tables = []
    for cols in KEYS:
        g = pd.DataFrame({'k': _key(df, cols)[ok], 'y': target[ok]}).groupby('k')['y']
        t = pd.DataFrame({'cel_p50': g.median(), 'cel_p90': g.quantile(0.9), 'cel_dp': g.std(), 'cel_n': g.size().astype(float)})
        tables.append(t[t['cel_n'] >= MIN_FLIGHTS])
    if GAP:
        tables.append(_fit_gap(df, np.asarray(y, float)))
    return tables

def apply(df: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    missing = np.ones(len(df), bool)
    for level, (cols, t) in enumerate(zip(KEYS, tables)):
        if t.empty or not missing.any():
            continue
        k = pd.Series(_key(df, cols))
        found = missing & k.isin(t.index).to_numpy()
        if not found.any():
            continue
        sub = t.reindex(k[found].to_numpy())
        for c in STATISTICS:
            out.iloc[np.flatnonzero(found), out.columns.get_loc(c)] = sub[c].to_numpy()
        out.iloc[np.flatnonzero(found), out.columns.get_loc('cel_nivel')] = float(level)
        missing &= ~found
    if len(tables) > N_CELLS:
        _apply_gap(df, tables[N_CELLS], out)
    return out

def apply_by_block(df: pd.DataFrame, block: np.ndarray, tables: dict[int, list[pd.DataFrame]]) -> pd.DataFrame:
    out = pd.DataFrame({c: np.full(len(df), np.nan) for c in COLS}, index=df.index)
    block = np.asarray(block)
    for k, tabs in tables.items():
        sel = np.flatnonzero(block == k)
        if sel.size:
            out.iloc[sel] = apply(df.iloc[sel], tabs).to_numpy()
    return out
