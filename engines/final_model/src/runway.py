"""Runway configuration, queue and off-block retention features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import movement_features as F
from dataset_cache import DATA
from external_data import TIME
from weather_rotation import _raw_files


AOBT = 'AOBT_3_flt'
EOBT = 'EOBT_1_flt'
SCHED = 'SCHED_TIME_UTC_mvt'
WAKE = 'WK_TBL_CAT_flt'
RWY = 'RUNWAY_mvt'
MOV_COLS = ['PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', RWY, TIME, WAKE]
DEP_COLS = ['PHASE_mvt', 'ADEP_mvt', RWY, TIME, AOBT, EOBT, SCHED]
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
MARGIN = pd.Timedelta('6h')
BLOCK = 600.0
WINDOW = 6
HOUR = 6
N_RECENT = 20
HALF_HOUR_WINDOW = 1800.0
WEIGHT = {'J': 2.0, 'H': 1.6, 'M': 1.0, 'L': 0.7}
DEFAULT_WEIGHT = 1.0
HEAVY = ('H', 'J')
MAX_HELD = 7200.0
DELAY_WINDOW = 900.0
HALF_LIFE = 600.0
EWMA_CUTOFF = 7200.0
EWMA_CHUNK = 21600.0
CAT = 'pista_cfg'
COLS_NUM = ['pista_cfg_mudou', 'pista_cfg_dep', 'pista_cfg_arr', 'pista_dominante', 'pista_parte_rwy', 'pista_fila_peso', 'pista_fila_pesadas', 'pista_span20', 'pista_dep_h30']
COLS = [CAT, *COLS_NUM]
COLS_RET = ['ret_overdue_rwy', 'ret_overdue_apt', 'ret_ativos_rwy', 'ret_atraso_15m', 'ret_ewma_dep']

def _seconds(s: pd.Series) -> np.ndarray:
    t = pd.to_datetime(s, utc=True, errors='coerce')
    return (t - EPOCH).dt.total_seconds().to_numpy(float)

def _groups(key: np.ndarray) -> dict:
    return pd.Series(np.arange(len(key))).groupby(key, sort=False).indices

def _weight(wk: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    cat = wk.astype(str).str.strip().str.upper().str[:1]
    return (cat.map(WEIGHT).fillna(DEFAULT_WEIGHT).to_numpy(float), cat.isin(HEAVY).to_numpy(float))

def _movements(apts: set[str], t0: float, t1: float, data: Path) -> dict:
    vocab_apt = {a: i for i, a in enumerate(sorted(apts))}
    vocab_rwy: dict[str, int] = {}
    margin = MARGIN.total_seconds()
    lim0 = EPOCH + pd.to_timedelta(t0 - margin, unit='s')
    lim1 = EPOCH + pd.to_timedelta(t1 + margin, unit='s')
    parts = []
    for start, end, p in _raw_files(data):
        if end <= lim0 or start > lim1:
            continue
        raw = pd.read_parquet(p, columns=MOV_COLS)
        phase = raw['PHASE_mvt'].to_numpy()
        name = np.where(phase == 'DEP', raw['ADEP_mvt'].to_numpy(), raw['ADES_mvt'].to_numpy())
        apt = pd.Series(name).map(vocab_apt).to_numpy(float)
        t = _seconds(raw[TIME])
        ok = np.isfinite(apt) & np.isfinite(t) & (t >= t0 - margin) & (t <= t1 + margin)
        if ok.any():
            codes, names = pd.factorize(raw[RWY][ok].astype(str).to_numpy())
            table = np.array([vocab_rwy.setdefault(n, len(vocab_rwy)) for n in names], dtype=np.int32)
            weight, heavy = _weight(raw[WAKE][ok])
            parts.append({'apt': apt[ok].astype(np.int16), 'rwy': table[codes], 't': t[ok], 'peso': weight, 'pesada': heavy, 'dep': phase[ok] == 'DEP'})
        del raw
    empty = {'apt': np.int16, 'rwy': np.int32, 't': float, 'peso': float, 'pesada': float, 'dep': bool}
    combined = {c: np.concatenate([p[c] for p in parts]) if parts else np.empty(0, dtype=kind) for c, kind in empty.items()}
    order = np.argsort(combined['t'], kind='stable')
    combined = {c: v[order] for c, v in combined.items()}
    combined['vocab_apt'] = vocab_apt
    combined['nomes_rwy'] = [n for n, _ in sorted(vocab_rwy.items(), key=lambda kv: kv[1])]
    return combined

def _count(bins: np.ndarray, runways: np.ndarray, nb: int, r: int) -> np.ndarray:
    if not bins.size:
        return np.zeros((nb, r))
    return np.bincount(bins * r + runways, minlength=nb * r).reshape(nb, r).astype(float)

def _window(cnt: np.ndarray) -> np.ndarray:
    nb, r = cnt.shape
    cs = np.vstack([np.zeros((1, r)), np.cumsum(cnt, axis=0)])
    i = np.arange(nb)
    return cs[np.clip(i + WINDOW // 2, None, nb)] - cs[np.clip(i - WINDOW // 2, 0, None)]

def _codes(win_dep: np.ndarray, win_arr: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    r = len(names)
    order = sorted(range(r), key=lambda i: names[i])
    active = np.concatenate([win_dep > 0, win_arr > 0], axis=1)
    unique, inverse = np.unique(active, axis=0, return_inverse=True)
    texts = []
    for row in unique:
        d = '+'.join((names[i] for i in order if row[i])) or '-'
        a = '+'.join((names[i] for i in order if row[r + i])) or '-'
        texts.append(f'D:{d} A:{a}')
    return (np.array(texts, dtype=object), np.asarray(inverse).reshape(-1))

def _configuration(num: dict, cod: np.ndarray, pos: np.ndarray, apt: str, t: np.ndarray, rwy: np.ndarray, dep: tuple, arr: tuple, names_rwy: list[str]) -> None:
    dep_t, dep_r = dep
    arr_t, arr_r = arr
    local, inverse = np.unique(np.concatenate([dep_r, arr_r, rwy]), return_inverse=True)
    names = [names_rwy[c] if c >= 0 else '?' for c in local]
    r = local.size
    d_idx, a_idx = (inverse[:dep_r.size], inverse[dep_r.size:dep_r.size + arr_r.size])
    own = inverse[dep_r.size + arr_r.size:]
    bins = np.floor(np.concatenate([dep_t, arr_t, t]) / BLOCK).astype(np.int64)
    b0, nb = (bins.min(), int(bins.max() - bins.min() + 1))
    bins -= b0
    win_d = _window(_count(bins[:dep_t.size], d_idx, nb, r))
    win_a = _window(_count(bins[dep_t.size:dep_t.size + arr_t.size], a_idx, nb, r))
    bi = bins[dep_t.size + arr_t.size:]
    texts, inverse_cfg = _codes(win_d, win_a, names)
    cod[pos] = [f'{apt}|{texts[k]}' for k in inverse_cfg[bi]]
    before = bi - HOUR
    num['pista_cfg_mudou'][pos] = np.where(before >= 0, (inverse_cfg[bi] != inverse_cfg[np.clip(before, 0, None)]).astype(float), np.nan)
    wd, wa = (win_d[bi], win_a[bi])
    num['pista_cfg_dep'][pos] = (wd > 0).sum(axis=1)
    num['pista_cfg_arr'][pos] = (wa > 0).sum(axis=1)
    total, own = (wd.sum(axis=1), wd[np.arange(bi.size), own])
    num['pista_dominante'][pos] = np.where(total > 0, (own >= wd.max(axis=1)).astype(float), np.nan)
    num['pista_parte_rwy'][pos] = np.divide(own, total, out=np.full(total.shape, np.nan), where=total > 0)

def _cadence(num: dict, pos: np.ndarray, t: np.ndarray, dep_t: np.ndarray) -> None:
    j = np.searchsorted(dep_t, t, 'left')
    before = j - N_RECENT
    num['pista_span20'][pos] = np.where(before >= 0, t - dep_t[np.clip(before, 0, None)], np.nan)
    j30 = np.searchsorted(dep_t, t - HALF_HOUR_WINDOW, 'left')
    num['pista_dep_h30'][pos] = (j - j30) * (3600.0 / HALF_HOUR_WINDOW)

def _queue(num: dict, pos: np.ndarray, t: np.ndarray, aobt: np.ndarray, rwy: np.ndarray, dep: tuple, weight_own: np.ndarray, heavy_own: np.ndarray) -> None:
    dep_t, dep_r, dep_weight, dep_heavy = dep
    for cod_rwy, rows in _groups(rwy).items():
        sel = dep_r == cod_rwy
        tt = dep_t[sel]
        acc_weight = np.concatenate([[0.0], np.cumsum(dep_weight[sel])])
        acc_heavy = np.concatenate([[0.0], np.cumsum(dep_heavy[sel])])
        a, m = (aobt[rows], t[rows])
        hi = np.searchsorted(tt, m, 'right')
        lo = np.searchsorted(tt, np.where(np.isfinite(a), a, m), 'right')
        positions = pos[rows]
        num['pista_fila_peso'][positions] = np.where(np.isfinite(a), np.maximum(acc_weight[hi] - acc_weight[lo] - weight_own[rows], 0.0), np.nan)
        num['pista_fila_pesadas'][positions] = np.where(np.isfinite(a), np.maximum(acc_heavy[hi] - acc_heavy[lo] - heavy_own[rows], 0.0), np.nan)

def _frame(cod: np.ndarray, num: dict, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame({CAT: pd.Categorical(cod)}, index=index)
    for c in COLS_NUM:
        out[c] = num[c]
    return out[COLS]

def columns(df: pd.DataFrame, data: Path=DATA) -> pd.DataFrame:
    n = len(df)
    num = {c: np.full(n, np.nan) for c in COLS_NUM}
    cod = np.full(n, None, dtype=object)
    if n == 0:
        return _frame(cod, num, df.index)
    t = _seconds(df[TIME])
    aobt = _seconds(df[AOBT]) if AOBT in df else np.full(n, np.nan)
    apt_name = df[F.AIRPORT].astype(str).to_numpy()
    rwy_name = df[RWY].astype(str).to_numpy()
    if WAKE in df:
        weight_own, heavy_own = _weight(df[WAKE])
    else:
        weight_own, heavy_own = (np.full(n, DEFAULT_WEIGHT), np.zeros(n))
    known = np.isfinite(t)
    if not known.any():
        return _frame(cod, num, df.index)
    mov = _movements(set(apt_name[known]), float(np.nanmin(t)), float(np.nanmax(t)), data)
    index = {name: i for i, name in enumerate(mov['nomes_rwy'])}
    rwy_cod = pd.Series(rwy_name).map(index).fillna(-1).to_numpy(np.int64)
    for name, pos in _groups(apt_name).items():
        pos = pos[known[pos]]
        code_apt = mov['vocab_apt'].get(name)
        if not pos.size or code_apt is None:
            continue
        at_airport = mov['apt'] == code_apt
        d, a = (at_airport & mov['dep'], at_airport & ~mov['dep'])
        _configuration(num, cod, pos, name, t[pos], rwy_cod[pos], (mov['t'][d], mov['rwy'][d]), (mov['t'][a], mov['rwy'][a]), mov['nomes_rwy'])
        _cadence(num, pos, t[pos], mov['t'][d])
        _queue(num, pos, t[pos], aobt[pos], rwy_cod[pos], (mov['t'][d], mov['rwy'][d], mov['peso'][d], mov['pesada'][d]), weight_own[pos], heavy_own[pos])
    return _frame(cod, num, df.index)

def _departures(apts: set[str], t0: float, t1: float, data: Path) -> dict:
    vocab_apt = {a: i for i, a in enumerate(sorted(apts))}
    vocab_rwy: dict[str, int] = {}
    margin = MARGIN.total_seconds()
    lim0 = EPOCH + pd.to_timedelta(t0 - margin, unit='s')
    lim1 = EPOCH + pd.to_timedelta(t1 + margin, unit='s')
    parts = []
    for start, end, p in _raw_files(data):
        if end <= lim0 or start > lim1:
            continue
        raw = pd.read_parquet(p, columns=DEP_COLS)
        apt = raw['ADEP_mvt'].map(vocab_apt).to_numpy(float)
        mvt = _seconds(raw[TIME])
        ok = (raw['PHASE_mvt'].to_numpy() == 'DEP') & np.isfinite(apt) & np.isfinite(mvt) & (mvt >= t0 - margin) & (mvt <= t1 + margin)
        if ok.any():
            codes, names = pd.factorize(raw[RWY][ok].astype(str).to_numpy())
            table = np.array([vocab_rwy.setdefault(n, len(vocab_rwy)) for n in names], dtype=np.int32)
            parts.append({'apt': apt[ok].astype(np.int16), 'rwy': table[codes], 'mvt': mvt[ok], 'aobt': _seconds(raw[AOBT])[ok], 'eobt': _seconds(raw[EOBT])[ok], 'sched': _seconds(raw[SCHED])[ok]})
        del raw
    types = {'apt': np.int16, 'rwy': np.int32, 'mvt': float, 'aobt': float, 'eobt': float, 'sched': float}
    combined = {c: np.concatenate([p[c] for p in parts]) if parts else np.empty(0, dtype=kind) for c, kind in types.items()}
    combined['vocab_apt'], combined['vocab_rwy'] = (vocab_apt, vocab_rwy)
    return combined

def _held(par: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    waiting = par['aobt'] - par['eobt']
    valid = sel & np.isfinite(waiting) & (waiting >= 0) & (waiting < MAX_HELD) & (par['mvt'] >= par['aobt'])
    return (np.sort(par['eobt'][valid]), np.sort(par['aobt'][valid]))

def _active(par: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = sel & np.isfinite(par['aobt']) & np.isfinite(par['mvt']) & (par['aobt'] <= par['mvt'])
    return (np.sort(par['aobt'][valid]), np.sort(par['mvt'][valid]))

def _count_until(ordered: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.searchsorted(ordered, t, 'right').astype(float)

def _recent_delay(par: dict, sel: np.ndarray, t: np.ndarray) -> np.ndarray:
    valid = sel & np.isfinite(par['aobt']) & np.isfinite(par['sched'])
    order = np.argsort(par['aobt'][valid], kind='stable')
    when = par['aobt'][valid][order]
    cs = np.concatenate([[0.0], np.cumsum((par['aobt'][valid] - par['sched'][valid])[order])])
    hi = np.searchsorted(when, t, 'left')
    lo = np.searchsorted(when, t - DELAY_WINDOW, 'left')
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)

def _ewma(mvt: np.ndarray, t: np.ndarray) -> np.ndarray:
    output = np.zeros(t.size)
    if not mvt.size:
        return output
    for chunk, rows in _groups(np.floor(t / EWMA_CHUNK)).items():
        origin = chunk * EWMA_CHUNK
        start = np.searchsorted(mvt, origin - EWMA_CUTOFF, 'left')
        end = np.searchsorted(mvt, origin + EWMA_CHUNK, 'left')
        ev = mvt[start:end]
        cs = np.concatenate([[0.0], np.cumsum(np.exp2((ev - origin) / HALF_LIFE))])
        tq = t[rows]
        hi = np.searchsorted(ev, tq, 'left')
        lo = np.searchsorted(ev, tq - EWMA_CUTOFF, 'left')
        output[rows] = np.exp2((origin - tq) / HALF_LIFE) * (cs[hi] - cs[lo])
    return output * (3600.0 * np.log(2.0) / HALF_LIFE)

def retention_features(df: pd.DataFrame, data: Path=DATA) -> pd.DataFrame:
    n = len(df)
    num = {c: np.full(n, np.nan) for c in COLS_RET}
    if n == 0:
        return pd.DataFrame(num, index=df.index)[COLS_RET]
    t = _seconds(df[AOBT]) if AOBT in df else np.full(n, np.nan)
    mvt = _seconds(df[TIME])
    apt_name = df[F.AIRPORT].astype(str).to_numpy()
    rwy_name = df[RWY].astype(str).to_numpy()
    known = np.isfinite(t)
    if not known.any():
        return pd.DataFrame(num, index=df.index)[COLS_RET]
    par = _departures(set(apt_name[known]), float(np.nanmin(t)), float(np.nanmax(t)), data)
    rwy_cod = pd.Series(rwy_name).map(par['vocab_rwy']).fillna(-1).to_numpy(np.int64)
    own = (np.isfinite(mvt) & (mvt > t)).astype(float)
    for name, pos in _groups(apt_name).items():
        pos = pos[known[pos]]
        code_apt = par['vocab_apt'].get(name)
        if not pos.size or code_apt is None:
            continue
        at_airport = par['apt'] == code_apt
        tq = t[pos]
        eobt_s, aobt_s = _held(par, at_airport)
        num['ret_overdue_apt'][pos] = _count_until(eobt_s, tq) - _count_until(aobt_s, tq)
        num['ret_atraso_15m'][pos] = _recent_delay(par, at_airport, tq)
        num['ret_ewma_dep'][pos] = _ewma(np.sort(par['mvt'][at_airport]), tq)
        for cod_rwy, rows in _groups(rwy_cod[pos]).items():
            on_runway = at_airport & (par['rwy'] == cod_rwy)
            positions, when = (pos[rows], tq[rows])
            eobt_r, aobt_r = _held(par, on_runway)
            num['ret_overdue_rwy'][positions] = _count_until(eobt_r, when) - _count_until(aobt_r, when)
            push, takeoff = _active(par, on_runway)
            num['ret_ativos_rwy'][positions] = np.maximum(_count_until(push, when) - _count_until(takeoff, when) - own[positions], 0.0)
    return pd.DataFrame(num, index=df.index)[COLS_RET]
