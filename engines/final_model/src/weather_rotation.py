"""METAR, stand rotation, flight-plan consistency and airline features.
Weather source: IEM ASOS, https://mesonet.agron.iastate.edu/."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import movement_features as F
from dataset_cache import DATA
from external_data import BLOCK, ROOT_DIR, TIME, _download, _airline

APTS = ('EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM')
TOL_METAR = pd.Timedelta('2h')
NA_METAR = ['null', 'M']
MET_COLS = ['met_temp', 'met_spread', 'met_vento', 'met_rajada', 'met_vis', 'met_teto', 'met_neve', 'met_congel', 'met_trovoada', 'met_chuva', 'met_nevoa']
DEICING_C = 3.0
DEICING_SPREAD = 3.0
MAX_AGE = pd.Timedelta(seconds=86400)
ARR_COLS = ['PHASE_mvt', 'ADES_mvt', 'STAND_mvt', BLOCK, 'FLIGHT_mvt', 'AIRCRAFT_TYPE_mvt']
TOP_AIRLINES = 150
OTHER = 'outro'
QUEUE_COUNTS = [f'{g}_{q}_{d}_{w}m' for g, q, ws in (('apt', 'dep', (10, 30, 60)), ('apt', 'arr', (30, 60)), ('rwy', 'dep', (10, 30, 60))) for w in ws for d in ('prev', 'next')]
URL_METAR = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={apt}&data=tmpf&data=dwpf&data=sknt&data=gust&data=vsby&data=wxcodes&data=skyc1&data=skyl1&year1={start.year}&month1={start.month}&day1={start.day}&year2={end.year}&month2={end.month}&day2={end.day}&tz=Etc/UTC&format=onlycomma&latlon=no&missing=null&trace=T&direct=no'

def download(root: Path=ROOT_DIR) -> None:
    from dataset import time_bounds
    start, end = time_bounds(DATA)
    start = start.floor('D') - pd.Timedelta(days=1)
    end = end.floor('D') + pd.Timedelta(days=1)
    for apt in APTS:
        print(_download(URL_METAR.format(apt=apt, start=start, end=end), root / 'metar' / f'{apt}.csv'), flush=True)

def _movement_time(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df[TIME], utc=True)

def _observations(root: Path) -> pd.DataFrame:
    parts = []
    for apt in APTS:
        p = root / 'metar' / f'{apt}.csv'
        if not p.exists():
            raise SystemExit(f'Missing {p.name} in {p.parent}; run: python src/weather_rotation.py download')
        m = pd.read_csv(p, na_values=NA_METAR, low_memory=False)
        for col in ('tmpf', 'dwpf', 'sknt', 'gust', 'vsby', 'skyl1'):
            m[col] = pd.to_numeric(m[col], errors='coerce')
        wx = m['wxcodes'].astype('string').fillna('')
        parts.append(pd.DataFrame({'t': pd.to_datetime(m['valid'], utc=True, errors='coerce'), 'apt': apt, 'met_temp': (m['tmpf'] - 32) / 1.8, 'met_spread': (m['tmpf'] - m['dwpf']) / 1.8, 'met_vento': m['sknt'], 'met_rajada': m['gust'], 'met_vis': m['vsby'], 'met_teto': m['skyl1'], 'met_neve': wx.str.contains('SN').astype(float), 'met_congel': wx.str.contains('FZ').astype(float), 'met_trovoada': wx.str.contains('TS').astype(float), 'met_chuva': wx.str.contains('RA').astype(float), 'met_nevoa': wx.str.contains('FG|BR').astype(float)}))
    obs = pd.concat(parts, ignore_index=True)
    return obs.dropna(subset=['t']).sort_values('t', ignore_index=True)

def metar(df: pd.DataFrame, root: Path=ROOT_DIR) -> pd.DataFrame:
    query = pd.DataFrame({'i': np.arange(len(df)), 't': _movement_time(df).reset_index(drop=True), 'apt': df[F.AIRPORT].astype(str).reset_index(drop=True)}).dropna(subset=['t']).sort_values('t', ignore_index=True)
    matched = pd.merge_asof(query, _observations(root), on='t', by='apt', direction='backward', tolerance=TOL_METAR)
    out = matched.set_index('i')[MET_COLS].reindex(range(len(df)))
    out['met_degelo'] = ((out['met_temp'] <= DEICING_C) & ((out['met_spread'] <= DEICING_SPREAD) | (out[['met_neve', 'met_congel', 'met_chuva']].sum(axis=1) > 0))).astype(float).where(out['met_temp'].notna())
    out.index = df.index
    return out

def _raw_files(data: Path) -> list[tuple[pd.Timestamp, pd.Timestamp, Path]]:
    windows = []
    for p in sorted(data.glob('training_2025-*.parquet')):
        start, end = p.stem.split('_')[1:3]
        windows.append((pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'), p))
    ranking = data / 'ranking.parquet'
    if ranking.exists():
        from dataset import file_coverage
        start, end, _ = file_coverage(ranking)
        windows.append((start, end + pd.Timedelta(nanoseconds=1), ranking))
    return windows

def _arrivals(t0: pd.Timestamp, t1: pd.Timestamp, data: Path) -> pd.DataFrame:
    parts = []
    for start, end, p in _raw_files(data):
        if end <= t0 - MAX_AGE or start > t1:
            continue
        raw = pd.read_parquet(p, columns=ARR_COLS)
        arr = raw[raw['PHASE_mvt'] == 'ARR'].dropna(subset=[BLOCK, 'STAND_mvt'])
        del raw
        parts.append(pd.DataFrame({'t_a': pd.to_datetime(arr[BLOCK], utc=True), 'apt': arr['ADES_mvt'].astype(str), 'st': arr['STAND_mvt'].astype(str), 'cia_a': _airline(arr['FLIGHT_mvt']).astype(str), 'tipo_a': arr['AIRCRAFT_TYPE_mvt'].astype(str)}))
    columns = ['t_a', 'apt', 'st', 'cia_a', 'tipo_a']
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype='object') for c in columns}).astype({'t_a': 'datetime64[ns, UTC]'})
    arrivals = pd.concat(parts, ignore_index=True)
    return arrivals.dropna(subset=['t_a']).sort_values('t_a', ignore_index=True)[columns]

def stand_rotation(df: pd.DataFrame, data: Path=DATA) -> pd.DataFrame:
    query = pd.DataFrame({'i': np.arange(len(df)), 't': _movement_time(df).reset_index(drop=True), 'apt': df[F.AIRPORT].astype(str).reset_index(drop=True), 'st': df['STAND_mvt'].astype(str).reset_index(drop=True), 'cia': _airline(df['FLIGHT_mvt']).astype(str).reset_index(drop=True), 'tipo': df['AIRCRAFT_TYPE_mvt'].astype(str).reset_index(drop=True)}).dropna(subset=['t']).sort_values('t', ignore_index=True)
    empty = pd.DataFrame(np.nan, index=df.index, columns=['rot_mesma_cia', 'rot_mesmo_tipo', 'rot_idade'])
    if query.empty:
        return empty
    arrivals = _arrivals(query['t'].iloc[0], query['t'].iloc[-1], data)
    if arrivals.empty:
        return empty
    matched = pd.merge_asof(query, arrivals, left_on='t', right_on='t_a', by=['apt', 'st'], direction='backward', tolerance=MAX_AGE).set_index('i')
    found = matched['t_a'].notna()
    out = pd.DataFrame({'rot_mesma_cia': (matched['cia'] == matched['cia_a']).astype(float).where(found), 'rot_mesmo_tipo': (matched['tipo'] == matched['tipo_a']).astype(float).where(found), 'rot_idade': (matched['t'] - matched['t_a']).dt.total_seconds()}).reindex(range(len(df)))
    out.index = df.index
    return out

def flight_plan_consistency(df: pd.DataFrame) -> pd.DataFrame:

    def duration(end: str, start: str) -> pd.Series:
        return (pd.to_datetime(df[end], utc=True, errors='coerce') - pd.to_datetime(df[start], utc=True, errors='coerce')).dt.total_seconds()

    def equal(a: str, b: str) -> pd.Series:
        return (df[a].astype(str) == df[b].astype(str)).astype(float)
    plan, real = (duration('ARVT_1_flt', 'EOBT_1_flt'), duration('ARVT_3_flt', 'AOBT_3_flt'))
    return pd.DataFrame({'nm_dur_plan': plan, 'nm_dur_real': real, 'nm_dur_dif': real - plan, 'nm_dur_razao': real / plan.where(plan > 0), 'nm_ades_igual': equal('ADES_mvt', 'ADES_flt'), 'nm_ades_filed_igual': equal('ADES_flt', 'ADES_FILED_flt'), 'nm_tipo_igual': equal('AIRCRAFT_TYPE_mvt', 'AIRCRAFT_TYPE_flt'), 'nm_adep_igual': equal('ADEP_mvt', 'ADEP_flt')}, index=df.index)

def airline_vocabulary(df: pd.DataFrame, n: int=TOP_AIRLINES) -> list[str]:
    return sorted(_airline(df['FLIGHT_mvt']).value_counts().index[:n])

def airline_category(df: pd.DataFrame, airlines: list[str]) -> pd.Categorical:
    airline = _airline(df['FLIGHT_mvt']).astype(str)
    return pd.Categorical(airline.where(airline.isin(set(airlines)), OTHER), categories=[*airlines, OTHER])

def stand_prefix(stand: pd.Series) -> pd.Series:
    return stand.astype(str).str.extract('^([A-Za-z]+|\\d)', expand=False).fillna('?')

def queue(df: pd.DataFrame) -> pd.DataFrame:
    apt = df[F.AIRPORT].astype(str)
    out = df[[c for c in QUEUE_COUNTS if c in df]].astype(float).reset_index(drop=True)
    out['fila_rwy_hora'] = (df['rwy_dep_prev_60m'].astype(float) + df['rwy_dep_next_60m'].astype(float)).to_numpy()
    out['apt_rwy'] = pd.Categorical((apt + '|' + df['RUNWAY_mvt'].astype(str)).to_numpy())
    out['stand_p'] = pd.Categorical((apt + '|' + stand_prefix(df['STAND_mvt'])).to_numpy())
    return out

def weather_rotation_features(df: pd.DataFrame, airlines: list[str], root: Path=ROOT_DIR, data: Path=DATA, include_queue: bool | str=False) -> pd.DataFrame:
    parts = [metar(df, root), stand_rotation(df, data), flight_plan_consistency(df)]
    out = pd.concat([p.reset_index(drop=True).astype(float) for p in parts], axis=1)
    out['cia'] = airline_category(df, airlines)
    if include_queue:
        extra = queue(df)
        if include_queue == 'stand':
            extra = extra[['stand_p']]
        for c in extra.columns:
            out[c] = extra[c].array
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('action', choices=['download'])
    ap.add_argument('--root', type=Path, default=ROOT_DIR)
    download(ap.parse_args().root)
