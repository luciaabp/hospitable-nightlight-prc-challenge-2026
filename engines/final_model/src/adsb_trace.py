"""Clip public adsb.lol traces to airport bounding boxes. Source: adsb.lol, ODbL 1.0."""
import os
from pathlib import Path
import numpy as np
ROOT_DIR = Path(os.environ.get('PRC_ADSB_RAIZ', Path(__file__).resolve().parents[1] / 'data' / 'adsb'))
ALT_MAX_FT = 3000
BOX_DEG = 0.1
AIRPORTS = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861), 'EGLL': (51.47, -0.4543), 'EHAM': (52.3086, 4.7639), 'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626), 'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389), 'LTFM': (41.2753, 28.7519), 'LSZH': (47.4647, 8.5492)}
PREFERRED = 'https://raw.githubusercontent.com/adsblol/globe_history_{ano}/main/PREFERRED_RELEASES.txt'

def airport_boxes() -> np.ndarray:
    out = []
    for lat, lon in AIRPORTS.values():
        dlon = BOX_DEG / np.cos(np.radians(lat))
        out.append([lat - BOX_DEG, lat + BOX_DEG, lon - dlon, lon + dlon])
    return np.array(out)

def point_airport(lat: float, lon: float, cx: np.ndarray) -> int:
    hit = (cx[:, 0] <= lat) & (lat <= cx[:, 1]) & (cx[:, 2] <= lon) & (lon <= cx[:, 3])
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else -1
COLS = ['icao', 'reg', 'tipo', 't', 'apt', 'lat', 'lon', 'chao', 'alt', 'gs', 'track', 'vrate', 'fonte', 'callsign']

def _field(p: list, i: int):
    return p[i] if len(p) > i else None

def clip_trace(d: dict, cx: np.ndarray) -> list[tuple]:
    t0 = d['timestamp']
    icao, reg, kind = (d['icao'], d.get('r'), d.get('t'))
    cs = None
    rows = []
    for p in d.get('trace', []):
        if isinstance(_field(p, 8), dict) and p[8].get('flight'):
            cs = p[8]['flight'].strip()
        alt = p[3]
        ground = alt == 'ground'
        if not ground and (alt is None or alt > ALT_MAX_FT):
            continue
        a = point_airport(p[1], p[2], cx)
        if a < 0:
            continue
        rows.append((icao, reg, kind, t0 + p[0], a, p[1], p[2], ground, None if ground else alt, p[4], _field(p, 5), _field(p, 7), _field(p, 9), cs))
    return rows
