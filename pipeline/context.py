"""Pre-takeoff context per departure, built only from **other**, earlier flights (optional).

    python -m pipeline.context prepare [--output local_data/context_features]

For departure B with takeoff T only departures airborne before T and arrivals landed before T enter; BLOCK is never
used. Two families: airport delay (``CONTEXT_DELAY``) and runway state (``CONTEXT_RUNWAY``: configuration in use and
traffic on the runways crossed by the stand -> runway route, shortest path on the X-Plane taxi network).

Output: ``features.parquet`` (one row per training and ranking departure) and ``manifest.json``.
"""
import argparse
import heapq
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CONTEXT_DELAY, CONTEXT_FEATURES, CONTEXT_RUNWAY, ID,
                     MISSING_CATEGORY, ROOT)

log = logging.getLogger(__name__)

CHALLENGE = ROOT / "data" / "projects" / "prc_data_challenge"
APT_DIR = CHALLENGE / "xplane"
OUTPUT = ROOT / "local_data" / "context_features"
COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt",
           "STAND_mvt"]

WINDOW_RUNWAY = 30 * 60          # s: runway state
MIN_RUNWAY_MOVES = 200           # runways with fewer 2025 movements are left out of the configuration
MIN_CONFIG_ROWS = 100            # configurations with fewer 2025 departures become "<airport>|rare"
EARTH_RADIUS_M = 6_371_008.8
MAX_RAMP_SNAP_M = 150.0          # stand -> network node
MAX_RUNWAY_SNAP_M = 500.0        # access node to a runway end
ON_RUNWAY_M = 60.0               # runway-type edge rolling on the centreline


# ----------------------------------------------------------------------------------------------------
# X-Plane taxi network: frozen copy of the essentials of src/feature_engineering/airport_geometry.py,
# so that the submission does not change with it.
# ----------------------------------------------------------------------------------------------------
def normalize_stand(value: object) -> str:
    name = str(value).strip().upper()
    name = re.sub(r"^(?:GATE|STAND|RAMP|PARKING|PARK)\s+", "", name)
    return re.sub(r"\s+", "", name)


def _distance_m(point, points: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(point)
    lat2, lon2 = np.radians(points[:, 0]), np.radians(points[:, 1])
    haversine = (np.sin((lat2 - lat1) / 2) ** 2
                 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _segment_distance_and_fraction_m(point, segment) -> tuple[float, float]:
    start, end = segment
    lat0, lon0 = np.radians(point)
    scale_x = EARTH_RADIUS_M * np.cos(lat0)
    start_lat, start_lon = np.radians(start)
    end_lat, end_lon = np.radians(end)
    start_x = scale_x * ((start_lon - lon0 + np.pi) % (2 * np.pi) - np.pi)
    end_x = scale_x * ((end_lon - lon0 + np.pi) % (2 * np.pi) - np.pi)
    start_y, end_y = EARTH_RADIUS_M * (start_lat - lat0), EARTH_RADIUS_M * (end_lat - lat0)
    vector_x, vector_y = end_x - start_x, end_y - start_y
    denominator = vector_x * vector_x + vector_y * vector_y
    fraction = 0.0 if denominator == 0 else np.clip((-start_x * vector_x - start_y * vector_y) / denominator, 0.0, 1.0)
    return float(np.hypot(start_x + fraction * vector_x, start_y + fraction * vector_y)), float(fraction)


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _proper_cross(p1, p2, q1, q2) -> bool:
    """Segments p and q cross at an interior point of both (touching or overlapping does not count)."""
    return _orient(q1, q2, p1) * _orient(q1, q2, p2) < 0 and _orient(p1, p2, q1) * _orient(p1, p2, q2) < 0


class TaxiNetwork:
    """Shortest stand -> runway-end route and the physical runways it crosses."""

    def __init__(self, path: Path):
        self.parking: dict[str, list] = defaultdict(list)
        self.runway_segments: dict[str, list] = defaultdict(list)
        self.runway_endpoints: dict[str, list] = defaultdict(list)
        self.nodes: dict[int, tuple[float, float]] = {}
        edges, self.edge_type = [], {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            f = line.split()
            if not f:
                continue
            try:
                if f[0] == "100" and len(f) >= 20:
                    first, second = (float(f[9]), float(f[10])), (float(f[18]), float(f[19]))
                    for name, end in ((f[8].strip().upper(), first), (f[17].strip().upper(), second)):
                        self.runway_segments[name].append((first, second))
                        self.runway_endpoints[name].append(end)
                elif f[0] == "1201" and len(f) >= 5:
                    self.nodes[int(f[4])] = (float(f[1]), float(f[2]))
                elif f[0] == "1202" and len(f) >= 4:
                    if f[3].lower() in {"twoway", "oneway"}:
                        edges.append((int(f[1]), int(f[2]), f[3].lower() == "twoway"))
                    if len(f) >= 5:
                        self.edge_type[(int(f[1]), int(f[2]))] = self.edge_type[(int(f[2]), int(f[1]))] = f[4]
                elif f[0] == "1300" and len(f) >= 7:
                    name = normalize_stand(" ".join(f[6:]))
                    if name:
                        self.parking[name].append((float(f[1]), float(f[2])))
            except (ValueError, IndexError):
                continue
        graph = defaultdict(list)
        for start, end, twoway in edges:
            if start not in self.nodes or end not in self.nodes:
                continue
            weight = float(_distance_m(self.nodes[start], np.asarray([self.nodes[end]]))[0])
            graph[start].append((end, weight))
            if twoway:
                graph[end].append((start, weight))
        self.graph = dict(graph)
        connected = set(self.graph) | {end for neighbors in self.graph.values() for end, _ in neighbors}
        # access nodes to each runway end: within 500 m of the centreline, on its half
        self.runway_access = {}
        for runway, segments in self.runway_segments.items():
            access = []
            for endpoint, segment in zip(self.runway_endpoints[runway], segments):
                near_start = endpoint == segment[0]
                for node in connected:
                    distance, fraction = _segment_distance_and_fraction_m(self.nodes[node], segment)
                    if distance <= MAX_RUNWAY_SNAP_M and (fraction <= 0.5 if near_start else fraction >= 0.5):
                        access.append((node, distance))
            self.runway_access[runway] = access
        # physical runway: segment -> "09L/27R"; local planar coordinates for the crossings
        physical = defaultdict(set)
        for name, segments in self.runway_segments.items():
            for segment in segments:
                physical[tuple(sorted(segment))].add(name)
        self.physical = {segment: "/".join(sorted(names)) for segment, names in physical.items()}
        coordinates = np.array(list(self.nodes.values()))
        self.origin = (coordinates[:, 0].mean(), coordinates[:, 1].mean())
        self.physical_xy = {segment: self._planar(list(segment)) for segment in self.physical}
        self.paths = {}

    def _planar(self, points) -> np.ndarray:
        points = np.radians(np.asarray(points, float))
        lat0, lon0 = np.radians(self.origin)
        return np.c_[EARTH_RADIUS_M * np.cos(lat0) * (points[:, 1] - lon0), EARTH_RADIUS_M * (points[:, 0] - lat0)]

    def _nearest_node(self, point, max_distance: float) -> tuple[int, float] | None:
        node_ids = np.fromiter(sorted(self.graph), dtype=int)
        if not len(node_ids):
            return None
        distances = _distance_m(point, np.asarray([self.nodes[n] for n in node_ids]))
        nearest = int(np.argmin(distances))
        return (int(node_ids[nearest]), float(distances[nearest])) if distances[nearest] <= max_distance else None

    def _shortest_paths(self, stand: str):
        """Dijkstra from the nodes closest to the stand's ramp starts (cached per stand)."""
        if stand not in self.paths:
            seeds = [s for p in self.parking.get(stand, []) if (s := self._nearest_node(p, MAX_RAMP_SNAP_M)) is not None]
            distances, previous, heap = {}, {}, []
            for node, access in seeds:
                if access < distances.get(node, float("inf")):
                    distances[node], previous[node] = access, None
                    heapq.heappush(heap, (access, node))
            while heap:
                distance, node = heapq.heappop(heap)
                if distance != distances[node]:
                    continue
                for neighbor, weight in self.graph.get(node, ()):
                    if distance + weight < distances.get(neighbor, float("inf")):
                        distances[neighbor], previous[neighbor] = distance + weight, node
                        heapq.heappush(heap, (distance + weight, neighbor))
            self.paths[stand] = (distances, previous) if seeds else None
        return self.paths[stand]

    def crossed(self, stand: str, runway: str) -> str | None:
        """Physical runways crossed by the route, "08L/26R;..." ('' if none); None without a route."""
        state = self._shortest_paths(stand)
        access = self.runway_access.get(runway, [])
        if state is None or not access:
            return None
        distances, previous = state
        best = min(((distances[n] + a, n) for n, a in access if n in distances), default=None)
        if best is None:
            return None
        route, node = [], best[1]
        while node is not None:
            route.append(node)
            node = previous[node]
        route.reverse()
        own = {segment for segment, names in self.physical.items() if runway in names.split("/")}
        xy = self._planar([self.nodes[n] for n in route])
        crossed = set()
        for (u, v), pu, pv in zip(zip(route, route[1:]), xy, xy[1:]):
            for segment, (q1, q2) in self.physical_xy.items():
                if segment in own:
                    continue
                if _proper_cross(pu, pv, q1, q2):
                    crossed.add(self.physical[segment])
                elif self.edge_type.get((u, v)) == "runway":
                    direction = q2 - q1
                    normal = np.array([-direction[1], direction[0]]) / np.hypot(*direction)
                    if abs((pu - q1) @ normal) < ON_RUNWAY_M and abs((pv - q1) @ normal) < ON_RUNWAY_M:
                        crossed.add(self.physical[segment])
        return ";".join(sorted(crossed))


def crossed_runways(departures: pd.DataFrame, apt_dir: Path = APT_DIR) -> np.ndarray:
    """``geo_cross_names`` per departure (None without a route)."""
    keys = departures[["airport", "STAND_mvt", "runway"]].drop_duplicates()
    table = {}
    for airport, group in keys.groupby("airport"):
        path = apt_dir / airport / "apt.dat"
        if not path.exists():
            continue
        network = TaxiNetwork(path)
        for stand, runway in zip(group.STAND_mvt, group.runway):
            if isinstance(stand, str):
                table[(airport, stand, runway)] = network.crossed(normalize_stand(stand), runway)
        log.info("%s: route for %d of %d stand-runway pairs", airport,
                 sum(v is not None for (a, *_), v in table.items() if a == airport), len(group))
    return np.array([table.get(key) for key in zip(departures.airport, departures.STAND_mvt, departures.runway)],
                    dtype=object)


# ----------------------------------------------------------------------------------------------------
# Time context: only events strictly before each departure's takeoff
# ----------------------------------------------------------------------------------------------------
def _ns(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]").array.asi8.astype(np.int64)


def count_before(sorted_times: np.ndarray, t: np.ndarray, window_ns: int) -> np.ndarray:
    """Events in [t - window, t)."""
    return np.searchsorted(sorted_times, t, "left") - np.searchsorted(sorted_times, t - window_ns, "left")


def median_before(event_t: np.ndarray, event_v: np.ndarray, query_t: np.ndarray, minutes: int) -> np.ndarray:
    """Median of the events in [t - window, t) for each query; events at t do not enter."""
    events = pd.DataFrame({"t": event_t, "v": np.asarray(event_v, float), "q": -1})
    queries = pd.DataFrame({"t": query_t, "v": np.nan, "q": np.arange(len(query_t))})
    both = pd.concat([events, queries], ignore_index=True).sort_values(["t", "q"], kind="stable")
    series = pd.Series(both.v.to_numpy(), index=pd.to_datetime(both.t.to_numpy(), utc=True))
    median = series.rolling(f"{minutes}min", closed="left", min_periods=1).median().to_numpy()
    is_query = both.q.to_numpy() >= 0
    out = np.full(len(query_t), np.nan)
    out[both.q.to_numpy()[is_query]] = median[is_query]
    return out


def context_features(departures: pd.DataFrame, arrivals: pd.DataFrame, crossed: np.ndarray) -> pd.DataFrame:
    """Context per departure; ``training`` marks the 2025 movements, which alone decide frequent runways and
    configurations. ``crossed``: ``geo_cross_names`` of each departure."""
    window = WINDOW_RUNWAY * 10**9
    n = len(departures)
    values = {c: np.full(n, np.nan) for c in CONTEXT_DELAY[:3] + CONTEXT_RUNWAY}
    config = np.full(n, MISSING_CATEGORY, dtype=object)
    dep_t, arr_t = _ns(departures.MVT_TIME_UTC_mvt), _ns(arrivals.MVT_TIME_UTC_mvt)
    dep_delay = (dep_t - _ns(departures.SCHED_TIME_UTC_mvt)) / 1e9
    arr_delay = (arr_t - _ns(arrivals.SCHED_TIME_UTC_mvt)) / 1e9
    dep_airport, arr_airport = departures.airport.to_numpy(), arrivals.airport.to_numpy()
    dep_runway, arr_runway = departures.runway.to_numpy(), arrivals.runway.to_numpy()
    training, arr_training = departures.training.to_numpy(bool), arrivals.training.to_numpy(bool)
    for airport in pd.unique(dep_airport):
        di, ai = np.flatnonzero(dep_airport == airport), np.flatnonzero(arr_airport == airport)
        t = dep_t[di]
        moves = pd.Series(np.r_[dep_runway[di[training[di]]], arr_runway[ai[arr_training[ai]]]]).value_counts()
        runways = sorted(moves[moves >= MIN_RUNWAY_MOVES].index)
        arrivals_on = {r: count_before(np.sort(arr_t[ai][arr_runway[ai] == r]), t, window) for r in runways}
        departures_on = {r: count_before(np.sort(t[dep_runway[di] == r]), t, window) for r in runways}
        active_arr = [",".join(r for r in runways if arrivals_on[r][k] > 0) for k in range(len(di))]
        active_dep = [",".join(r for r in runways if departures_on[r][k] > 0) for k in range(len(di))]
        config[di] = [f"{airport}|A:{a}|D:{d}" for a, d in zip(active_arr, active_dep)]
        values["rwy_own_arr_30m"][di] = [arrivals_on[r][k] if r in arrivals_on else 0
                                         for k, r in enumerate(dep_runway[di])]
        names = crossed[di]
        cross_arr, cross_dep = np.full(len(di), np.nan), np.full(len(di), np.nan)
        for value in pd.unique(names[pd.notna(names)]):
            k = np.flatnonzero(names == value)
            runs = [r for phys in value.split(";") if phys for r in phys.split("/") if r in arrivals_on]
            cross_arr[k] = sum(arrivals_on[r][k] for r in runs) if runs else 0
            cross_dep[k] = sum(departures_on[r][k] for r in runs) if runs else 0
        values["rwy_cross_arr_30m"][di], values["rwy_cross_dep_30m"][di] = cross_arr, cross_dep
        values["ctx_dep_delay_med_30m"][di] = median_before(t, dep_delay[di], t, 30)
        values["ctx_dep_delay_med_60m"][di] = median_before(t, dep_delay[di], t, 60)
        values["ctx_arr_delay_med_60m"][di] = median_before(arr_t[ai], arr_delay[ai], t, 60)
    counts = pd.Series(config[training]).value_counts()
    frequent = set(counts[counts >= MIN_CONFIG_ROWS].index)
    out = pd.DataFrame({ID: departures[ID].to_numpy()})
    out["ctx_dep_delay_med_30m"] = values["ctx_dep_delay_med_30m"]
    out["ctx_dep_delay_med_60m"] = values["ctx_dep_delay_med_60m"]
    out["ctx_arr_delay_med_60m"] = values["ctx_arr_delay_med_60m"]
    out["ctx_own_minus_dep_delay_60m"] = dep_delay - values["ctx_dep_delay_med_60m"]
    out["rwy_config"] = [c if c in frequent else c.split("|")[0] + "|rare" for c in config]
    for column in CONTEXT_RUNWAY:
        out[column] = values[column]
    out["geo_cross_names"] = crossed
    return out[[ID] + CONTEXT_FEATURES + ["geo_cross_names"]]


def read_movements(raw_dir: Path = CHALLENGE / "raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw training (2025) and ranking departures and arrivals; ``training`` marks the 2025 ones."""
    parts = []
    for path in sorted(raw_dir.glob("training_*.parquet")) + [raw_dir / "ranking.parquet"]:
        parts.append(pd.read_parquet(path, columns=COLUMNS).assign(training=path.name.startswith("training_")))
    raw = pd.concat(parts, ignore_index=True)
    if raw[ID].isna().any() or raw[ID].duplicated().any() or raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Movements without ID, with a repeated ID or without time")
    runway = raw.RUNWAY_mvt.astype("string").str.strip().str.upper()
    raw["runway"] = runway.where(runway.notna() & (runway != "NA"), MISSING_CATEGORY).astype(str)
    departures = raw[raw.PHASE_mvt == "DEP"].assign(airport=lambda d: d.ADEP_mvt.astype(str)).reset_index(drop=True)
    arrivals = raw[raw.PHASE_mvt == "ARR"].assign(airport=lambda d: d.ADES_mvt.astype(str)).reset_index(drop=True)
    return departures, arrivals


def prepare(raw_dir: Path = CHALLENGE / "raw", apt_dir: Path = APT_DIR, output: Path = OUTPUT) -> Path:
    from .data import sha256
    departures, arrivals = read_movements(raw_dir)
    crossed = crossed_runways(departures, apt_dir)
    features = context_features(departures, arrivals, crossed)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "features.parquet"
    features.to_parquet(target, index=False)
    month = departures.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    key = departures.airport + "|" + month
    coverage = pd.DataFrame({"key": key, "route": pd.notna(crossed), "crosses": [bool(c) for c in crossed],
                             "delay": features.ctx_dep_delay_med_60m.notna(),
                             "rare_config": features.rwy_config.str.endswith("|rare")}).groupby("key").mean()
    manifest = {"output": str(target), "output_sha256": sha256(target), "rows": int(len(features)),
                "window_runway_sec": WINDOW_RUNWAY, "min_runway_moves": MIN_RUNWAY_MOVES,
                "min_config_rows": MIN_CONFIG_ROWS,
                "inputs": {str(p): sha256(p) for p in sorted(raw_dir.glob("training_*.parquet"))
                           + [raw_dir / "ranking.parquet"] + sorted(apt_dir.glob("*/apt.dat"))},
                "coverage": coverage.round(4).to_dict(orient="index")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%s: %d departures", target, len(features))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.context", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--raw-dir", type=Path, default=CHALLENGE / "raw")
    parser.add_argument("--apt-dir", type=Path, default=APT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(prepare(args.raw_dir, args.apt_dir, args.output))


if __name__ == "__main__":
    main()
