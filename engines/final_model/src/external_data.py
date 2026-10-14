"""Airline schedule-copy rates, EUROCONTROL daily series and OPDI flight lists.
Sources: https://ansperformance.eu/csv/ and https://www.opdi.aero/."""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import movement_features as F
from dataset_cache import DATA, split_paths

ROOT_DIR = DATA / 'externo'
TIME = 'MVT_TIME_UTC_mvt'
SCHED = 'SCHED_TIME_UTC_mvt'
BLOCK = 'BLOCK_TIME_UTC_mvt'
SMOOTHING_WEIGHT = 20
COPY_SECONDS = 60
MIN_DELAY_SECONDS = 3600
TRAINING_MONTHS = tuple(range(1, 13))
TOL_CALLSIGN_S = 600
TOL_APT_S = 90
OPDI_COLS = ['icao24', 'flt_id', 'adep', 'ades', 'first_seen', 'last_seen']
SERIES = {'atfm_slot_adherence': ['FLT_DEP_1', 'FLT_DEP_REG_1', 'FLT_DEP_OUT_LATE_1'], 'all_pre_departure_delays': ['FLT_DEP_IFR_2', 'DLY_ALL_PRE_2'], 'atc_pre_departure_delays': ['FLT_DEP_3', 'DLY_ATC_PRE_3']}
URL_CSV = 'https://www.eurocontrol.int/performance/data/download/csv/{name}.csv'
URL_OPDI = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{year_month}.parquet'

def _download(url: str, destination: Path) -> str:
    if destination.exists():
        return f'{destination.name}: already exists'
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.part')
    subprocess.run(['curl', '-sfL', '--retry', '5', '-o', str(part), url], check=True)
    part.rename(destination)
    return f'{destination.name}: {destination.stat().st_size / 1000000.0:.1f} MB'

def download(root: Path=ROOT_DIR) -> None:
    from dataset import months
    periods = months(DATA)
    for name in SERIES:
        for year in sorted({int(period[:4]) for period in periods}):
            print(_download(URL_CSV.format(name=f'{name}_{year}'), root / f'{name}_{year}.csv'), flush=True)
    for year_month in (period.replace('-', '') for period in periods):
        print(_download(URL_OPDI.format(year_month=year_month), root / 'opdi' / f'flight_list_{year_month}.parquet'), flush=True)

def _airline(flight: pd.Series) -> pd.Series:
    return flight.astype('string').str.extract('^([A-Z]{2,3})', expand=False).fillna('?')

def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0)

class AirlineCopyRate:

    def __init__(self, k: float=SMOOTHING_WEIGHT) -> None:
        self.k = k
        self.counts: pd.DataFrame | None = None

    def fit(self, raw_dep: pd.DataFrame) -> 'AirlineCopyRate':
        t = pd.to_datetime(raw_dep[TIME], utc=True)
        sched = pd.to_datetime(raw_dep[SCHED], utc=True)
        block = pd.to_datetime(raw_dep[BLOCK], utc=True)
        tab = pd.DataFrame({'apt': raw_dep[F.AIRPORT].astype('string'), 'cia': _airline(raw_dep['FLIGHT_mvt']), 'nm': raw_dep['FLIGHT_ID_mvt'].isna().astype('int8'), 'mes': t.dt.month, 'n': 1.0, 'copias': ((block - sched).dt.total_seconds().abs() <= COPY_SECONDS).astype(float)})[(t - sched).dt.total_seconds() > MIN_DELAY_SECONDS]
        new = tab.groupby(['apt', 'cia', 'nm', 'mes'])[['n', 'copias']].sum()
        self.counts = new if self.counts is None else self.counts.add(new, fill_value=0)
        return self

    def transform(self, df: pd.DataFrame, training_months) -> pd.DataFrame:
        tab = self.counts.reset_index()
        tab = tab[tab['mes'].isin({int(m) for m in training_months})]
        tab['chave'] = tab['apt'] + '|' + tab['cia'] + '|' + tab['nm'].astype(str)
        month_txt = '|' + tab['mes'].astype(str)
        by_key = tab.groupby('chave')[['n', 'copias']].sum()
        by_key_month = tab.groupby(tab['chave'] + month_txt)[['n', 'copias']].sum()
        by_apt = tab.groupby('apt')[['n', 'copias']].sum()
        by_apt_month = tab.groupby(tab['apt'] + month_txt)[['n', 'copias']].sum()
        apt = df[F.AIRPORT].astype('string')
        key = apt + '|' + _airline(df['FLIGHT_mvt']) + '|' + df['nm_missing'].astype(int).astype(str)
        t = pd.to_datetime(df[TIME], utc=True)
        month = '|' + t.dt.month.astype('Int64').astype('string')

        def outside_month(k: pd.Series, k_month: pd.Series, total, by_month, col) -> pd.Series:
            return k.map(total[col]).fillna(0.0) - k_month.map(by_month[col]).fillna(0.0)
        n_apt = outside_month(apt, apt + month, by_apt, by_apt_month, 'n')
        c_apt = outside_month(apt, apt + month, by_apt, by_apt_month, 'copias')
        n = outside_month(key, key + month, by_key, by_key_month, 'n')
        c = outside_month(key, key + month, by_key, by_key_month, 'copias')
        rate_apt = c_apt / n_apt.where(n_apt > 0)
        rate = (c + self.k * rate_apt) / (n + self.k)
        ms = (t - pd.to_datetime(df[SCHED], utc=True)).dt.total_seconds()
        return pd.DataFrame({'ext_taxa_cia': rate, 'ext_taxa_cia_ms': rate * ms}, index=df.index)

def _series(root: Path, name: str) -> pd.DataFrame:
    files = sorted(root.glob(f'{name}_*.csv'))
    if not files:
        raise SystemExit(f'Missing {name}_*.csv in {root}; run: python src/external_data.py download')
    cols = SERIES[name]
    df = pd.concat((pd.read_csv(p, usecols=['FLT_DATE', 'APT_ICAO', *cols]) for p in files), ignore_index=True)
    key = pd.to_datetime(df['FLT_DATE']).dt.strftime('%Y-%m-%d').astype('string') + '|' + df['APT_ICAO'].astype('string')
    return df.groupby(key)[cols].sum(min_count=1)

def daily_series(df: pd.DataFrame, root: Path=ROOT_DIR) -> pd.DataFrame:
    t = pd.to_datetime(df[TIME], utc=True)
    key = t.dt.strftime('%Y-%m-%d').astype('string') + '|' + df[F.AIRPORT].astype('string')
    slot, pre, atc = (_series(root, name) for name in SERIES)
    reg = key.map(slot['FLT_DEP_REG_1'])
    return pd.DataFrame({'ext_reg_frac': _ratio(reg, key.map(slot['FLT_DEP_1'])), 'ext_late_frac': _ratio(key.map(slot['FLT_DEP_OUT_LATE_1']), reg), 'ext_reg_n': reg.astype(float), 'ext_pre_min': _ratio(key.map(pre['DLY_ALL_PRE_2']), key.map(pre['FLT_DEP_IFR_2'])), 'ext_atc_min': _ratio(key.map(atc['DLY_ATC_PRE_3']), key.map(atc['FLT_DEP_3']))}, index=df.index)

def _flight_list(root: Path, months: list[str]) -> pd.DataFrame:
    parts = []
    for year_month in months:
        p = root / 'opdi' / f'flight_list_{year_month}.parquet'
        if not p.exists():
            raise SystemExit(f'Missing {p.name} in {p.parent}; run: python src/external_data.py download')
        parts.append(pd.read_parquet(p, columns=OPDI_COLS))
    flights = pd.concat(parts, ignore_index=True)
    for col in ('first_seen', 'last_seen'):
        flights[col] = pd.to_datetime(flights[col], utc=True)
    flights = flights.sort_values(['icao24', 'first_seen'], ignore_index=True)
    same = flights['icao24'].eq(flights['icao24'].shift())
    flights['ext_solo_s'] = (flights['first_seen'] - flights['last_seen'].shift()).dt.total_seconds().where(same)
    previous = flights['ades'].shift().where(same)
    equal = (previous == flights['adep']).fillna(False).astype(float)
    flights['ext_mesmo_apt'] = equal.where(previous.notna() & flights['adep'].notna())
    return flights

def _asof(left: pd.DataFrame, right: pd.DataFrame, by: list[str], tol_s: int) -> pd.DataFrame:
    output = ['i', 'ext_solo_s', 'ext_mesmo_apt']
    if left.empty or right.empty:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in output})
    matched = pd.merge_asof(left, right, left_on='t', right_on='first_seen', by=by, tolerance=pd.Timedelta(seconds=tol_s), direction='nearest')
    return matched[matched['first_seen'].notna()][output]

def opdi(df: pd.DataFrame, root: Path=ROOT_DIR) -> pd.DataFrame:
    t = pd.to_datetime(df[TIME], utc=True).reset_index(drop=True)
    apt = df[F.AIRPORT].astype('string').reset_index(drop=True)
    flights = _flight_list(root, sorted(t.dropna().dt.strftime('%Y%m').unique()))
    flights = flights[flights['adep'].isin(set(apt.dropna()))]
    flights = pd.DataFrame({'apt': flights['adep'].astype('string'), 'cs': flights['flt_id'].astype('string').str.strip(), 'first_seen': flights['first_seen'], 'ext_solo_s': flights['ext_solo_s'], 'ext_mesmo_apt': flights['ext_mesmo_apt']}).sort_values('first_seen', ignore_index=True)
    left = pd.DataFrame({'i': np.arange(len(df)), 't': t, 'apt': apt, 'cs': df['CALLSIGN_flt'].astype('string').str.strip().reset_index(drop=True)}).dropna(subset=['t']).sort_values('t', ignore_index=True)
    has_callsign = left['cs'].notna() & (left['cs'] != '')
    matched = _asof(left[has_callsign], flights[flights['cs'].notna() & (flights['cs'] != '')], ['apt', 'cs'], TOL_CALLSIGN_S)
    remaining = left[~left['i'].isin(matched['i'])]
    matched = pd.concat([matched, _asof(remaining, flights, ['apt'], TOL_APT_S)], ignore_index=True)
    ground, same = (np.full(len(df), np.nan), np.full(len(df), np.nan))
    positions = matched['i'].to_numpy(int)
    ground[positions] = matched['ext_solo_s'].to_numpy(float)
    same[positions] = matched['ext_mesmo_apt'].to_numpy(float)
    return pd.DataFrame({'ext_solo_s': ground, 'ext_mesmo_apt': same}, index=df.index)

def external_features(df: pd.DataFrame, copy_rate: AirlineCopyRate, training_months, root: Path=ROOT_DIR) -> dict[str, np.ndarray]:
    parts = [copy_rate.transform(df, training_months), daily_series(df, root), opdi(df, root)]
    return {c: p[c].to_numpy(float) for p in parts for c in p.columns}
_COPY_RATES: AirlineCopyRate | None = None

def airline_copy_rates(run=None) -> AirlineCopyRate:
    global _COPY_RATES
    if _COPY_RATES is None:
        copy_rate = AirlineCopyRate()
        for p in split_paths('full2025'):
            raw = F.load([p])
            copy_rate.fit(raw[raw['PHASE_mvt'] == 'DEP'])
            del raw
        _COPY_RATES = copy_rate
    if run:
        run.log(f'schedule-copy rate: {len(_COPY_RATES.counts):,} groups (airport, airline, NM, month)')
    return _COPY_RATES
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('action', choices=['download'])
    ap.add_argument('--root', type=Path, default=ROOT_DIR)
    download(ap.parse_args().root)
