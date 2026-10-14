"""Match observed ADS-B takeoffs and ground segments to departure movements."""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from adsb_trace import ROOT_DIR

TAKEOFF_GAP_S = 120
SEG_GAP_S = 600
CS_TOL_S = 300
TIME_TOL_S = 90
DATA = Path(__file__).resolve().parent.parent / 'data'
EVENT_COLS = ['apt', 'takeoff', 'first_ground', 'first_move', 'gs0', 'n_chao', 'gap_max', 'frac_mlat', 'lat0', 'lon0', 'callsign', 'reg', 'tipo', 'icao']
FEATURES = ['adsb_taxi', 'adsb_taxi_move', 'adsb_gs0', 'adsb_takeoff_err', 'adsb_n_chao', 'adsb_gap_max', 'adsb_frac_mlat', 'adsb_lat0', 'adsb_lon0']
MOVE_KT = 1.0

def add_features(df: pd.DataFrame, root: Path=ROOT_DIR) -> pd.DataFrame:
    existing = [c for c in FEATURES if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)
    path = root / 'events.parquet'
    ev = pd.read_parquet(path, columns=['MVT_ID_mvt', *FEATURES]) if path.exists() else None
    if ev is None or ev.empty:
        for c in FEATURES:
            df[c] = np.nan
        return df
    keys = pd.Index(ev['MVT_ID_mvt'])
    if keys.has_duplicates:
        raise ValueError('events.parquet contains duplicate MVT_ID_mvt values; the join would duplicate rows')
    positions = keys.get_indexer(pd.Index(df['MVT_ID_mvt']))
    found = positions >= 0
    for c in FEATURES:
        v = ev[c].to_numpy(float)
        df[c] = np.where(found, v[positions], np.nan)
    return df

def takeoffs(cut: pd.DataFrame) -> pd.DataFrame:
    d = cut.sort_values(['icao', 't'], kind='stable').reset_index(drop=True)
    icao = d['icao'].to_numpy()
    t = d['t'].to_numpy(float)
    ground = d['chao'].to_numpy(bool)
    same_prev = np.r_[False, icao[1:] == icao[:-1]]
    gap_prev = np.r_[np.inf, np.diff(t)]
    cont = same_prev & np.r_[False, ground[:-1]] & ground & (gap_prev < SEG_GAP_S)
    run = np.cumsum(ground & ~cont)
    nxt_same = np.r_[same_prev[1:], False]
    nxt_air = np.r_[~ground[1:], False]
    nxt_gap = np.r_[gap_prev[1:], np.inf]
    last = np.flatnonzero(ground & nxt_same & nxt_air & (nxt_gap < TAKEOFF_GAP_S))
    if last.size == 0:
        return pd.DataFrame(columns=EVENT_COLS)
    g = d.loc[ground].assign(run=run[ground], gap=np.where(cont, gap_prev, 0.0)[ground], mlat=(d['fonte'].astype(str) == 'mlat').to_numpy()[ground])
    g['t_move'] = g['t'].where(g['gs'] > MOVE_KT)
    agg = g.groupby('run').agg(first_ground=('t', 'min'), first_move=('t_move', 'min'), gs0=('gs', 'first'), n_chao=('t', 'size'), gap_max=('gap', 'max'), frac_mlat=('mlat', 'mean'), lat0=('lat', 'first'), lon0=('lon', 'first'))
    ev = pd.DataFrame({'run': run[last], 'apt': d['apt'].to_numpy()[last + 1], 'takeoff': t[last + 1], 'callsign': d['callsign'].astype(str).to_numpy()[last + 1], 'reg': d['reg'].astype(str).to_numpy()[last], 'tipo': d['tipo'].astype(str).to_numpy()[last], 'icao': icao[last]}).join(agg, on='run')
    ev['callsign'] = ev['callsign'].replace({'nan': None, 'None': None})
    return ev[EVENT_COLS].reset_index(drop=True)

def match_events(dep: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.reset_index(drop=True).assign(ev_id=lambda x: x.index)
    ev['apt'] = ev['apt'].astype(str)
    dep = dep.assign(apt=dep['apt'].astype(str))
    cs = dep.dropna(subset=['callsign']).merge(ev.dropna(subset=['callsign'])[['ev_id', 'apt', 'callsign', 'takeoff']], on=['apt', 'callsign'])
    pairs = [cs[(cs['takeoff'] - cs['mvt']).abs() < CS_TOL_S].assign(pri=0)]
    for apt, dd in dep.groupby('apt'):
        ee = ev[ev['apt'] == apt].sort_values('takeoff')
        if ee.empty:
            continue
        dd = dd.sort_values('mvt')
        for direction in ('backward', 'forward'):
            m = pd.merge_asof(dd, ee[['ev_id', 'takeoff']], left_on='mvt', right_on='takeoff', direction=direction, tolerance=TIME_TOL_S).dropna(subset=['ev_id'])
            pairs.append(m.assign(pri=1))
    p = pd.concat(pairs, ignore_index=True)
    p['dist'] = (p['takeoff'] - p['mvt']).abs()
    p = p.sort_values(['pri', 'dist'], kind='stable')
    used_v, used_e, keep = (set(), set(), [])
    for i, v, e in zip(p.index, p['MVT_ID_mvt'].to_numpy(), p['ev_id'].to_numpy()):
        if v not in used_v and e not in used_e:
            used_v.add(v)
            used_e.add(e)
            keep.append(i)
    p = p.loc[keep]
    out = p[['MVT_ID_mvt', 'mvt', 'ev_id']].merge(ev.drop(columns=['apt', 'callsign']), on='ev_id')
    return out.drop(columns='ev_id')

def features(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({'MVT_ID_mvt': m['MVT_ID_mvt'].to_numpy(), 'adsb_taxi': (m['mvt'] - m['first_ground']).to_numpy(float), 'adsb_taxi_move': (m['mvt'] - m['first_move']).to_numpy(float), 'adsb_gs0': m['gs0'].to_numpy(float), 'adsb_takeoff_err': (m['takeoff'] - m['mvt']).to_numpy(float), 'adsb_n_chao': m['n_chao'].to_numpy(float), 'adsb_gap_max': m['gap_max'].to_numpy(float), 'adsb_frac_mlat': m['frac_mlat'].to_numpy(float), 'adsb_lat0': m['lat0'].to_numpy(float), 'adsb_lon0': m['lon0'].to_numpy(float)})

def load_departures() -> pd.DataFrame:
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt']
    paths = sorted(DATA.glob('training_2025-*.parquet')) + [DATA / 'ranking.parquet']
    d = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    d = d[d['PHASE_mvt'] == 'DEP']
    mvt = pd.to_datetime(d['MVT_TIME_UTC_mvt'], utc=True)
    return pd.DataFrame({'MVT_ID_mvt': d['MVT_ID_mvt'].to_numpy(), 'apt': d['ADEP_mvt'].astype(str).to_numpy(), 'mvt': (mvt - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().to_numpy(), 'dia': mvt.dt.strftime('%Y-%m-%d').to_numpy(), 'callsign': d['CALLSIGN_flt'].fillna(d['FLIGHT_mvt']).to_numpy()})

def daily_events(cut_dir: Path, day: str) -> pd.DataFrame:
    start = pd.Timestamp(day, tz='UTC').timestamp()
    parts = []
    yesterday = cut_dir / f'{(date.fromisoformat(day) - timedelta(days=1)).isoformat()}.parquet'
    if yesterday.exists():
        parts.append(pd.read_parquet(yesterday, filters=[('t', '>=', start - 7200)]))
    parts.append(pd.read_parquet(cut_dir / f'{day}.parquet'))
    ev = takeoffs(pd.concat(parts, ignore_index=True))
    return ev[(ev['takeoff'] >= start) & (ev['takeoff'] < start + 86400)]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT_DIR)
    a = ap.parse_args()
    cut_dir = a.root / 'cut'
    deps = load_departures()
    days = sorted((p.stem for p in cut_dir.glob('*.parquet')))
    out = []
    for i, day in enumerate(days, 1):
        dd = deps[deps['dia'] == day]
        if dd.empty:
            continue
        m = match_events(dd, daily_events(cut_dir, day))
        out.append(features(m))
        if i % 20 == 0 or i == len(days):
            print(f'[{i}/{len(days)}] {day}: {len(m)}/{len(dd)} matched', flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(a.root / 'events.parquet', compression='zstd', index=False)
    print(f'{len(res):,} departures with ADS-B events -> {a.root / 'events.parquet'}')
if __name__ == '__main__':
    main()
