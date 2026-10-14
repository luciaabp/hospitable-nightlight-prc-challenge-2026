"""Static operational geometry from X-Plane apt.dat: ramp starts (1300), runway ends (100) and taxiway network
(1201/1202)."""

from __future__ import annotations

import heapq
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "stand_runway_geodesic_m",
    "stand_nearest_runway_geodesic_m",
    "stand_runway_taxiway_m",
    "stand_runway_taxi_band",
    "stand_runway_estimated_turns",
    "stand_geometry_available",
    "stand_taxiway_component",
    "stand_zone",
]

EARTH_RADIUS_M = 6_371_008.8
MAX_RAMP_SNAP_M = 150.0
MAX_RUNWAY_SNAP_M = 500.0


def _normalize_name(value: object) -> str:
    name = str(value).strip().upper()
    name = re.sub(r"^(?:GATE|STAND|RAMP|PARKING|PARK)\s+", "", name)
    return re.sub(r"\s+", "", name)


def _distance_m(point: tuple[float, float], points: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(point)
    lat2 = np.radians(points[:, 0])
    lon2 = np.radians(points[:, 1])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _distance_to_segments_m(
    point: tuple[float, float], segments: list[tuple[tuple[float, float], tuple[float, float]]]
) -> float:
    return min((_segment_distance_and_fraction_m(point, segment)[0] for segment in segments), default=float("nan"))


def _segment_distance_and_fraction_m(
    point: tuple[float, float], segment: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[float, float]:
    start, end = segment
    lat0, lon0 = np.radians(point)
    scale_x = EARTH_RADIUS_M * np.cos(lat0)
    start_lat, start_lon = np.radians(start)
    end_lat, end_lon = np.radians(end)
    start_delta_lon = (start_lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    end_delta_lon = (end_lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    start_x, start_y = scale_x * start_delta_lon, EARTH_RADIUS_M * (start_lat - lat0)
    end_x, end_y = scale_x * end_delta_lon, EARTH_RADIUS_M * (end_lat - lat0)
    vector_x, vector_y = end_x - start_x, end_y - start_y
    denominator = vector_x * vector_x + vector_y * vector_y
    fraction = 0.0 if denominator == 0 else np.clip(
        (-(start_x) * vector_x - start_y * vector_y) / denominator,
        0.0,
        1.0,
    )
    closest_x = start_x + fraction * vector_x
    closest_y = start_y + fraction * vector_y
    return float(np.hypot(closest_x, closest_y)), float(fraction)


@dataclass
class AirportGeometry:
    parking: dict[str, list[tuple[float, float]]]
    runways: dict[str, list[tuple[float, float]]]
    runway_segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]]
    runway_endpoints: dict[str, list[tuple[float, float]]]
    nodes: dict[int, tuple[float, float]]
    edges: list[tuple[int, int, bool]]
    graph_cache: dict[int, list[tuple[int, float]]] | None = field(default=None, init=False)
    component_cache: dict[int, str] | None = field(default=None, init=False)
    runway_access: dict[str, list[tuple[int, float]]] = field(default_factory=dict, init=False)
    feature_cache: dict[str, dict[str, dict[str, float | str]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        graph = self._graph()
        incoming_nodes = {end for neighbors in graph.values() for end, _ in neighbors}
        connected_nodes = set(graph) | incoming_nodes
        for runway, segments in self.runway_segments.items():
            runway_candidates = []
            endpoints = self.runway_endpoints.get(runway, [])
            for endpoint, segment in zip(endpoints, segments):
                near_start = endpoint == segment[0]
                candidates = []
                for node in connected_nodes:
                    segment_distance, fraction = _segment_distance_and_fraction_m(self.nodes[node], segment)
                    on_near_half = fraction <= 0.5 if near_start else fraction >= 0.5
                    if segment_distance <= MAX_RUNWAY_SNAP_M and on_near_half:
                        candidates.append((segment_distance, node))
                runway_candidates.extend((node, distance) for distance, node in candidates)
            self.runway_access[runway] = runway_candidates

    @classmethod
    def from_file(cls, path: Path) -> "AirportGeometry":
        parking: dict[str, list[tuple[float, float]]] = defaultdict(list)
        runways: dict[str, list[tuple[float, float]]] = defaultdict(list)
        runway_segments: dict[
            str, list[tuple[tuple[float, float], tuple[float, float]]]
        ] = defaultdict(list)
        runway_endpoints: dict[str, list[tuple[float, float]]] = defaultdict(list)
        nodes: dict[int, tuple[float, float]] = {}
        edges: list[tuple[int, int, bool]] = []

        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                if fields[0] == "100" and len(fields) >= 20:
                    first_name, second_name = fields[8].strip().upper(), fields[17].strip().upper()
                    first_end = (float(fields[9]), float(fields[10]))
                    second_end = (float(fields[18]), float(fields[19]))
                    runways[first_name].append(first_end)
                    runways[second_name].append(second_end)
                    runway_segments[first_name].append((first_end, second_end))
                    runway_segments[second_name].append((first_end, second_end))
                    runway_endpoints[first_name].append(first_end)
                    runway_endpoints[second_name].append(second_end)
                elif fields[0] == "1201" and len(fields) >= 5:
                    nodes[int(fields[4])] = (float(fields[1]), float(fields[2]))
                elif fields[0] == "1202" and len(fields) >= 4:
                    direction = fields[3].lower()
                    if direction in {"twoway", "oneway"}:
                        edges.append((int(fields[1]), int(fields[2]), direction == "twoway"))
                elif fields[0] == "1300" and len(fields) >= 7:
                    name = _normalize_name(" ".join(fields[6:]))
                    if name:
                        parking[name].append((float(fields[1]), float(fields[2])))
            except (ValueError, IndexError):
                continue

        return cls(dict(parking), dict(runways), dict(runway_segments), dict(runway_endpoints), nodes, edges)

    def _graph(self) -> dict[int, list[tuple[int, float]]]:
        if self.graph_cache is not None:
            return self.graph_cache
        graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for start, end, twoway in self.edges:
            if start not in self.nodes or end not in self.nodes:
                continue
            weight = float(_distance_m(self.nodes[start], np.asarray([self.nodes[end]]))[0])
            graph[start].append((end, weight))
            if twoway:
                graph[end].append((start, weight))
        self.graph_cache = dict(graph)
        return self.graph_cache

    def _nearest_node(
        self,
        point: tuple[float, float],
        max_distance: float,
        eligible_nodes: set[int],
    ) -> tuple[int, float] | None:
        if not eligible_nodes:
            return None
        node_ids = np.fromiter(sorted(eligible_nodes), dtype=int)
        coordinates = np.asarray([self.nodes[node] for node in node_ids])
        distances = _distance_m(point, coordinates)
        nearest = int(np.argmin(distances))
        if distances[nearest] > max_distance:
            return None
        return int(node_ids[nearest]), float(distances[nearest])

    def _components(self, graph: dict[int, list[tuple[int, float]]]) -> dict[int, str]:
        if self.component_cache is not None:
            return self.component_cache
        undirected: dict[int, set[int]] = defaultdict(set)
        for start, neighbors in graph.items():
            for end, _ in neighbors:
                undirected[start].add(end)
                undirected[end].add(start)
        component_by_node: dict[int, str] = {}
        component_index = 0
        for root in sorted(self.nodes):
            if root in component_by_node:
                continue
            label = f"{component_index:04d}"
            component_index += 1
            queue = deque([root])
            component_by_node[root] = label
            while queue:
                node = queue.popleft()
                for neighbor in undirected[node]:
                    if neighbor not in component_by_node:
                        component_by_node[neighbor] = label
                        queue.append(neighbor)
        self.component_cache = component_by_node
        return self.component_cache

    @staticmethod
    def _shortest_paths(
        graph: dict[int, list[tuple[int, float]]], seeds: list[tuple[int, float]]
    ) -> tuple[dict[int, float], dict[int, int | None]]:
        distances: dict[int, float] = {}
        previous: dict[int, int | None] = {}
        heap: list[tuple[float, int]] = []
        for node, access_distance in seeds:
            if access_distance < distances.get(node, float("inf")):
                distances[node] = access_distance
                previous[node] = None
                heapq.heappush(heap, (access_distance, node))
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != distances[node]:
                continue
            for neighbor, weight in graph.get(node, ()):
                candidate = distance + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(heap, (candidate, neighbor))
        return distances, previous

    def _estimated_turns(self, node: int, previous: dict[int, int | None]) -> int:
        turns = 0
        while (parent := previous[node]) is not None:
            grandparent = previous[parent]
            if grandparent is not None:
                first = np.asarray(self.nodes[parent]) - self.nodes[grandparent]
                second = np.asarray(self.nodes[node]) - self.nodes[parent]
                latitude_scale = np.cos(np.radians(self.nodes[parent][0]))
                first[1] *= latitude_scale
                second[1] *= latitude_scale
                norms = np.linalg.norm(first) * np.linalg.norm(second)
                if norms > 0 and np.dot(first, second) / norms < np.cos(np.pi / 4):
                    turns += 1
            node = parent
        return turns

    def features_for_stand(self, stand: object, airport: str) -> dict[str, dict[str, float | str]]:
        stand_key = _normalize_name(stand)
        if stand_key in self.feature_cache:
            return self.feature_cache[stand_key]
        parking_points = self.parking.get(stand_key, [])
        if not parking_points:
            return {}

        graph = self._graph()
        incoming_nodes = {end for neighbors in graph.values() for end, _ in neighbors}
        ramp_seeds = [
            seed
            for point in parking_points
            if (seed := self._nearest_node(point, MAX_RAMP_SNAP_M, set(graph))) is not None
        ]
        paths, previous = self._shortest_paths(graph, ramp_seeds)
        components = self._components(graph)
        if ramp_seeds:
            nearest_seed = min(ramp_seeds, key=lambda item: item[1])
            component = f"{airport}:{components.get(nearest_seed[0], 'NA')}"
        else:
            component = pd.NA

        nearest_runway = min(
            (
                _distance_to_segments_m(parking_point, segments)
                for parking_point in parking_points
                for segments in self.runway_segments.values()
            ),
            default=float("nan"),
        )
        output: dict[str, dict[str, float | str]] = {}
        for runway, runway_points in self.runways.items():
            segments = self.runway_segments.get(runway, [])
            geodesic = min(
                _distance_to_segments_m(parking_point, segments)
                for parking_point in parking_points
            )
            runway_seeds = self.runway_access.get(runway, [])
            best_access = min(
                ((paths[node] + access, node) for node, access in runway_seeds if node in paths),
                default=None,
            )
            taxiway_distance = best_access[0] if best_access is not None else float("nan")
            taxi_band = (
                "unreachable" if np.isnan(taxiway_distance) else
                "short" if taxiway_distance < 1_000 else
                "medium" if taxiway_distance < 2_500 else "long"
            )
            output[runway] = {
                "stand_runway_geodesic_m": geodesic,
                "stand_nearest_runway_geodesic_m": nearest_runway,
                "stand_runway_taxiway_m": taxiway_distance,
                "stand_runway_taxi_band": taxi_band,
                "stand_runway_estimated_turns": self._estimated_turns(best_access[1], previous) if best_access else float("nan"),
                "stand_geometry_available": 1.0,
                "stand_taxiway_component": component,
            }
        self.feature_cache[stand_key] = output
        return output


@lru_cache(maxsize=32)
def _load_airport_geometry(path: str) -> AirportGeometry | None:
    apt_path = Path(path)
    return AirportGeometry.from_file(apt_path) if apt_path.is_file() else None


def add_airport_geometry_features(
    df: pd.DataFrame, apt_dir: Path
) -> pd.DataFrame:
    """Add per-flight distances with exact stand and runway joins."""
    df = df.copy()
    df["stand_zone"] = (
        df["STAND_mvt"].astype("string").str.strip().str.upper()
        .str.extract(r"^([A-Z]+)", expand=False)
        .fillna("numeric")
        .replace({"NA": "unknown", "N/A": "unknown", "UNKNOWN": "unknown", "UNK": "unknown"})
    )

    keys = df[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].copy()
    keys["airport_key"] = keys["ADEP_mvt"].astype("string").str.upper()
    keys["stand_key"] = keys["STAND_mvt"].map(_normalize_name)
    keys["runway_key"] = keys["RUNWAY_mvt"].astype("string").str.strip().str.upper()
    unique_keys = keys[["airport_key", "stand_key", "runway_key"]].drop_duplicates()

    records = []
    for airport, group in unique_keys.groupby("airport_key", dropna=True, sort=False):
        geometry = _load_airport_geometry(str((apt_dir / str(airport) / "apt.dat").resolve()))
        if geometry is None:
            continue
        runway_names = set(group["runway_key"].dropna())
        for stand in group["stand_key"].dropna().unique():
            stand_features = geometry.features_for_stand(stand, str(airport))
            for runway in runway_names:
                values = stand_features.get(runway)
                if values is not None:
                    records.append({
                        "airport_key": airport,
                        "stand_key": stand,
                        "runway_key": runway,
                        **values,
                    })

    for feature in FEATURES[:-1]:
        df[feature] = np.nan
    df["stand_geometry_available"] = 0.0
    df["stand_taxiway_component"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["stand_runway_taxi_band"] = pd.Series(pd.NA, index=df.index, dtype="string")
    if records:
        lookup = pd.DataFrame(records).set_index(["airport_key", "stand_key", "runway_key"])
        row_keys = pd.MultiIndex.from_arrays(
            [keys["airport_key"], keys["stand_key"], keys["runway_key"]],
            names=lookup.index.names,
        )
        matched = lookup.reindex(row_keys)
        for feature in FEATURES[:-1]:
            if feature == "stand_runway_taxi_band":
                continue
            values = matched[feature]
            if feature == "stand_geometry_available":
                values = values.fillna(0.0)
            df[feature] = values.to_numpy()
        df["stand_taxiway_component"] = matched["stand_taxiway_component"].astype("string").to_numpy()
        df["stand_runway_taxi_band"] = matched["stand_runway_taxi_band"].astype("string").to_numpy()
    return df
