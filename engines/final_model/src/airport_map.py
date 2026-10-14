"""Stand-to-runway graph distances from X-Plane airport layouts.
Source: https://gateway.x-plane.com, GPL v2+ scenery."""
from __future__ import annotations
import base64
import io
import json
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import movement_features as F
from dataset_cache import DATA

DIR = DATA / 'mapa'
TABLE = DIR / 'distancias.parquet'
API = 'https://gateway.x-plane.com/apiv1'
COLS = ['map_dist', 'map_dmin', 'map_extra']

def norm(s) -> str:
    return re.sub('[^A-Z0-9]', '', re.sub('-\\([A-Z]\\)$', '', str(s).upper().strip()))

def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'prc-taxiout-2026'})
    return json.load(urllib.request.urlopen(req, timeout=60))

def download(apt: str) -> Path:
    f = DIR / f'{apt}.dat'
    if not f.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        sid = _get(f'{API}/airport/{apt}')['airport']['recommendedSceneryId']
        blob = _get(f'{API}/scenery/{sid}')['scenery']['masterZipBlob']
        zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob)))
        f.write_bytes(zf.read(next((n for n in zf.namelist() if n.endswith('.dat')))))
    return f

def _xy(lat, lon, lat0: float) -> np.ndarray:
    lat, lon = (np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float))
    return np.c_[np.radians(lon) * 6371000.0 * np.cos(np.radians(lat0)), np.radians(lat) * 6371000.0]

def airport_distances(apt: str) -> pd.DataFrame:
    nodes, edges, stands, thresholds = ({}, [], {}, {})
    for row in download(apt).read_text(errors='ignore').splitlines():
        p = row.split()
        if not p:
            continue
        if p[0] == '1201':
            nodes[int(p[4])] = (float(p[1]), float(p[2]))
        elif p[0] == '1202':
            edges.append((int(p[1]), int(p[2])))
        elif p[0] == '1300' and len(p) > 6:
            stands[norm(' '.join(p[6:]))] = (float(p[1]), float(p[2]))
        elif p[0] == '100':
            thresholds[norm(p[8])] = (float(p[9]), float(p[10]))
            thresholds[norm(p[17])] = (float(p[18]), float(p[19]))
    ids = list(nodes)
    pos = {k: i for i, k in enumerate(ids)}
    ll = np.array([nodes[k] for k in ids])
    lat0 = float(ll[:, 0].mean())
    P = _xy(ll[:, 0], ll[:, 1], lat0)
    i, j = np.array([(pos[a], pos[b]) for a, b in edges if a in pos and b in pos]).T
    w = np.linalg.norm(P[i] - P[j], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(len(ids),) * 2).tocsr()
    _, comp = connected_components(G, directed=False)
    outside = comp != np.bincount(comp).argmax()

    def most_near(la: float, lo: float) -> int:
        d = np.linalg.norm(P - _xy(la, lo, lat0), axis=1)
        d[outside] = np.inf
        return int(np.argmin(d))
    runways = list(thresholds)
    D = dijkstra(G, indices=[most_near(*thresholds[r]) for r in runways])
    rows = [(s, r, float(D[k, most_near(*c)])) for s, c in stands.items() for k, r in enumerate(runways)]
    return pd.DataFrame(rows, columns=['stand', 'pista', 'dist']).assign(apt=apt)

def build_map(airports) -> pd.DataFrame:
    t = pd.concat([airport_distances(a) for a in sorted(airports)], ignore_index=True)
    t.to_parquet(TABLE, index=False)
    return t

@lru_cache(maxsize=1)
def _table() -> tuple[dict, dict, dict]:
    t = pd.read_parquet(TABLE)
    dist = {(a, s, r): d for a, s, r, d in zip(t['apt'], t['stand'], t['pista'], t['dist'])}
    dmin = t.groupby(['apt', 'stand'])['dist'].min().to_dict()
    idx: dict = {}
    for a, s in dmin:
        idx[a, s] = s
    for a, s in dmin:
        for rx in ('([A-Z]?\\d+[A-Z]?)$', '(\\d+[A-Z]?)$'):
            m = re.search(rx, s)
            if m:
                idx.setdefault((a, m.group(1)), s)
    return (dist, dmin, idx)

def columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    dist, dmin, idx = _table()
    apt = df[F.AIRPORT].astype(str).to_numpy()
    st = [idx.get((a, norm(s))) for a, s in zip(apt, df['STAND_mvt'].to_numpy())]
    rw = [norm(r) for r in df['RUNWAY_mvt'].to_numpy()]
    d = np.array([dist.get((a, s, r), np.nan) if s else np.nan for a, s, r in zip(apt, st, rw)], float)
    m = np.array([dmin.get((a, s), np.nan) if s else np.nan for a, s in zip(apt, st)], float)
    return {'map_dist': d, 'map_dmin': m, 'map_extra': d - m}
if __name__ == '__main__':
    rk = pd.read_parquet(DATA / 'ranking.parquet', columns=['ADEP_mvt', 'PHASE_mvt'])
    airports = rk.loc[rk['PHASE_mvt'] == 'DEP', 'ADEP_mvt'].unique()
    t = build_map(airports)
