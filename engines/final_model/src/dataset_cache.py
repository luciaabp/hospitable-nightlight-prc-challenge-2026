"""Build label-safe feature splits and cache them by source and input identity."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import arrival_context as arrival_context_features
import movement_features as F
from adsb_events import add_features
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CACHE = DATA / 'cache'
TRUTH = 'y_true'
BLOCK = 'BLOCK_TIME_UTC_mvt'
SPLITS = ('blind2025', 'full2025', 'ranking2026')

def _read_parquet(path: Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas(split_blocks=True, self_destruct=True)

def split_paths(name: str) -> list[Path]:
    training = sorted(DATA.glob('training_2025-*.parquet'))
    match name:
        case 'full2025' | 'blind2025':
            return training
        case 'ranking2026':
            return [DATA / 'ranking.parquet']
    raise ValueError(f'Unknown split: {name}')

def cache_key(paths: list[Path]) -> str:
    h = hashlib.sha256(Path(F.__file__).read_bytes())
    h.update(Path(arrival_context_features.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    for p in paths:
        h.update(f'{p.name}:{p.stat().st_size}'.encode())
    return h.hexdigest()[:12]

def build_blind(raw: pd.DataFrame) -> pd.DataFrame:
    dep = raw['PHASE_mvt'] == 'DEP'
    truth = pd.Series(raw.loc[dep, F.TARGET].to_numpy(), index=raw.loc[dep, F.ID].to_numpy())
    blind = raw.copy()
    blind.loc[dep, BLOCK] = pd.NaT
    blind.loc[dep, F.TARGET] = np.nan
    out = F.build(blind)
    out[TRUTH] = out[F.ID].map(truth)
    return out

def load_split(name: str) -> pd.DataFrame:
    paths = split_paths(name)
    if not paths or any((not p.exists() for p in paths)):
        raise SystemExit(f'Missing input data for {name}; provide the organizer parquet files in data/')
    target = CACHE / f'{name}-{cache_key(paths)}.parquet'
    if target.exists():
        return add_features(_read_parquet(target))
    raw = F.load(paths)
    df = build_blind(raw) if name == 'blind2025' else F.build(raw)
    df = df.merge(arrival_context_features.arrival_context_features(raw), on=F.ID, how='left')
    del raw
    if name == 'full2025':
        df = df[df[F.TARGET].notna()]
    df = df.reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    df.to_parquet(tmp, compression='zstd', index=False)
    tmp.replace(target)
    for old in CACHE.glob(f'{name}-*.parquet'):
        if old != target:
            old.unlink()
    return add_features(df)
if __name__ == '__main__':
    from runtime import LOGS, Run
    with Run('cache', {'splits': list(SPLITS)}, registry=LOGS / 'infra.jsonl') as run:
        for split in SPLITS:
            with run.phase(split, 1 / len(SPLITS)):
                run.log(f'{split}: {len(load_split(split)):,} rows')
