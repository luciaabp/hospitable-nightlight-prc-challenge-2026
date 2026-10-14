"""Arrival, stand and neighbouring-flight context without departure block leakage."""
from __future__ import annotations
import numpy as np
import pandas as pd
import movement_features as F
WINDOWS_MIN = (15, 60)
MAX_STAND_AGE = 86400
TAXI_IN_MAX = 3600
PROXY_MAX = 7200
COLS = [*[f'ctx_arr_{q}_{k}_{w}' for k in ('apt', 'rwy') for q in ('tin', 'n') for w in WINDOWS_MIN], 'ctx_stand_ult_idade', 'ctx_stand_ult_tin', *[f'ctx_viz_{side}_{k}_{w}' for k in ('apt', 'rwy') for side in ('pas', 'fut') for w in WINDOWS_MIN], 'ctx_viz_proprio_menos_pas', 'ctx_viz_proprio_menos_fut']
_UNSEEN = np.iinfo(np.int64).max

def _seconds(s: pd.Series) -> np.ndarray:
    v = s.to_numpy('datetime64[s]').astype('float64')
    v[s.isna().to_numpy()] = np.nan
    return v

def _codes(dep: pd.DataFrame, arr: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:

    def key(df: pd.DataFrame) -> pd.Series:
        k = df[cols[0]].astype('string')
        for col in cols[1:]:
            k = k + '|' + df[col].astype('string')
        return k
    codes, _ = pd.factorize(pd.concat([key(dep), key(arr)], ignore_index=True))
    return (codes[:len(dep)].astype('int64'), codes[len(dep):].astype('int64'))

def _keys(codes: np.ndarray, t: np.ndarray) -> np.ndarray:
    k = (codes << 32) + t.astype('int64')
    return np.where(codes < 0, _UNSEEN, k)

def _cumulative(codes: np.ndarray, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(t) & np.isfinite(v) & (codes >= 0)
    codes, t, v = (codes[ok], t[ok], v[ok])
    order = np.lexsort((t, codes))
    keys = _keys(codes[order], t[order])
    return (keys, np.concatenate([[0.0], np.cumsum(v[order], dtype='float64')]))

def _mean(keys: np.ndarray, cs: np.ndarray, lo: np.ndarray, hi: np.ndarray, side_lo: str, side_hi: str) -> tuple[np.ndarray, np.ndarray]:
    n = np.searchsorted(keys, hi, side_hi) - (i := np.searchsorted(keys, lo, side_lo))
    sum = cs[i + n] - cs[i]
    return (np.where(n > 0, sum / np.maximum(n, 1), np.nan), n)

def _query_key(codes: np.ndarray, t: np.ndarray, offset: int) -> np.ndarray:
    raw = np.where(np.isfinite(t), t + offset, 0).astype('int64')
    return np.where(np.isfinite(t) & (codes >= 0), (codes << 32) + raw, _UNSEEN)

def arrival_context_features(raw: pd.DataFrame) -> pd.DataFrame:
    phase = raw['PHASE_mvt'].to_numpy()
    cols = [F.ID, F.AIRPORT, 'RUNWAY_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt']
    dep = raw.loc[phase == 'DEP', [*cols, 'AOBT_3_flt']]
    arr = raw.loc[phase == 'ARR', [*cols, 'BLOCK_TIME_UTC_mvt', F.TARGET]]
    d_mvt = _seconds(dep['MVT_TIME_UTC_mvt'])
    a_mvt = _seconds(arr['MVT_TIME_UTC_mvt'])
    a_block = _seconds(arr['BLOCK_TIME_UTC_mvt'])
    a_tin = arr[F.TARGET].to_numpy('float64').clip(0, TAXI_IN_MAX)
    proxy = d_mvt - _seconds(dep['AOBT_3_flt'])
    proxy[~((proxy >= 0) & (proxy <= PROXY_MAX))] = np.nan
    groups = {'apt': [F.AIRPORT], 'rwy': [F.AIRPORT, 'RUNWAY_mvt']}
    feats: dict[str, np.ndarray] = {}
    for name, key in groups.items():
        d_cod, a_cod = _codes(dep, arr, key)
        keys, cs = _cumulative(a_cod, a_mvt, a_tin)
        for w in WINDOWS_MIN:
            lo = _query_key(d_cod, d_mvt, -w * 60)
            hi = _query_key(d_cod, d_mvt, 0)
            mean, n = _mean(keys, cs, lo, hi, 'right', 'right')
            feats[f'ctx_arr_tin_{name}_{w}'] = mean
            feats[f'ctx_arr_n_{name}_{w}'] = n.astype('float64')
        keys, cs = _cumulative(d_cod, d_mvt, proxy)
        for w in WINDOWS_MIN:
            past = _mean(keys, cs, _query_key(d_cod, d_mvt, -w * 60), _query_key(d_cod, d_mvt, 0), 'left', 'left')[0]
            future = _mean(keys, cs, _query_key(d_cod, d_mvt, 0), _query_key(d_cod, d_mvt, w * 60), 'right', 'right')[0]
            feats[f'ctx_viz_pas_{name}_{w}'] = past
            feats[f'ctx_viz_fut_{name}_{w}'] = future
    feats['ctx_viz_proprio_menos_pas'] = proxy - feats['ctx_viz_pas_apt_60']
    feats['ctx_viz_proprio_menos_fut'] = proxy - feats['ctx_viz_fut_apt_60']
    feats.update(_stand(dep, arr, d_mvt, a_block, a_tin))
    out = pd.DataFrame({F.ID: dep[F.ID].to_numpy()})
    for col in COLS:
        out[col] = feats[col]
    return out

def _stand(dep: pd.DataFrame, arr: pd.DataFrame, d_mvt: np.ndarray, a_block: np.ndarray, a_tin: np.ndarray) -> dict[str, np.ndarray]:
    d_cod, a_cod = _codes(dep, arr, [F.AIRPORT, 'STAND_mvt'])
    without_stand = dep['STAND_mvt'].isna().to_numpy()
    age = np.full(len(dep), np.nan)
    tin = np.full(len(dep), np.nan)
    ok = np.isfinite(a_block) & ~arr['STAND_mvt'].isna().to_numpy()
    if ok.any():
        order = np.lexsort((a_block[ok], a_cod[ok]))
        keys = _keys(a_cod[ok][order], a_block[ok][order])
        blocks = a_block[ok][order]
        tins = a_tin[ok][order]
        pos = np.searchsorted(keys, _query_key(d_cod, d_mvt, 0), 'right') - 1
        found = (pos >= 0) & ~without_stand & np.isfinite(d_mvt)
        found[found] = keys[pos[found]] >> 32 == d_cod[found]
        target = pos[found]
        age[found] = d_mvt[found] - blocks[target]
        tin[found] = tins[target]
    old = age > MAX_STAND_AGE
    age[old] = np.nan
    tin[old] = np.nan
    return {'ctx_stand_ult_idade': age, 'ctx_stand_ult_tin': tin}
