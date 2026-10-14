"""Target-free surface activity from a complete movement context."""

import numpy as np
import pandas as pd

from .base import CLOCK, ID

CONTEXT_COLUMNS = [
    ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", CLOCK,
    "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "RUNWAY_mvt",
]
SECOND = 10**9
METRICS = (
    ["taxi_out_active_proxy", "taxi_in_active", "surface_active_proxy"]
    + [f"{name}_prev_{minutes}m" for minutes in [15, 30, 60] for name in ["departures", "arrivals", "nm_starts"]]
    + [f"planned_next_{minutes}m" for minutes in [15, 30]]
    + [f"net_ground_flow_{minutes}m" for minutes in [60, 180]]
    + ["seconds_since_departure", "seconds_since_arrival"]
    + [
        f"{name}_{minutes}m"
        for minutes in [30, 60]
        for name in ["completed_dep_proxy_mean", "completed_arr_taxi_mean", "completed_arr_count"]
    ]
    + ["runway_taxi_out_active_proxy", "runway_departures_prev_30m"]
)


def timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def nanoseconds(series: pd.Series) -> np.ndarray:
    return timestamps(series).array.asi8


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("UNKNOWN").str.strip().str.upper().astype(str)


class SurfaceContext:
    def __init__(self, raw: pd.DataFrame):
        if raw[ID].isna().any() or raw[ID].duplicated().any():
            raise ValueError("Movement context IDs must be unique and non-null")
        departures = raw.loc[raw["PHASE_mvt"].eq("DEP")].copy()
        arrivals = raw.loc[raw["PHASE_mvt"].eq("ARR")].copy()
        departures["airport"] = normalize(departures["ADEP_mvt"])
        arrivals["airport"] = normalize(arrivals["ADES_mvt"])
        for frame in [departures, arrivals]:
            frame["end"] = timestamps(frame[CLOCK])
            frame["runway"] = normalize(frame["RUNWAY_mvt"])
        departures["start"] = timestamps(departures["AOBT_3_flt"]).fillna(
            timestamps(departures["LOBT_flt"])
        )
        duration = (departures["end"] - departures["start"]).dt.total_seconds()
        departures["interval_valid"] = duration.gt(0) & duration.le(6 * 3600)
        arrivals["start"] = arrivals["end"]
        arrivals["end"] = timestamps(arrivals["BLOCK_TIME_UTC_mvt"])
        duration = (arrivals["end"] - arrivals["start"]).dt.total_seconds()
        arrivals["interval_valid"] = duration.gt(0) & duration.le(2 * 3600)
        departures["scheduled"] = timestamps(departures["SCHED_TIME_UTC_mvt"])
        self.groups = {}
        for airport in sorted(set(departures["airport"]) | set(arrivals["airport"])):
            self.groups[airport] = (
                departures[departures["airport"].eq(airport)],
                arrivals[arrivals["airport"].eq(airport)],
            )

    @staticmethod
    def sorted_time(series: pd.Series) -> np.ndarray:
        return np.sort(nanoseconds(series.dropna()))

    @staticmethod
    def counts(event_times: np.ndarray, reference: np.ndarray, minutes: int) -> np.ndarray:
        return np.searchsorted(event_times, reference, "left") - np.searchsorted(
            event_times, reference - minutes * 60 * SECOND, "left"
        )

    @staticmethod
    def active(rows: pd.DataFrame, reference: np.ndarray) -> np.ndarray:
        valid = rows.loc[rows["interval_valid"]]
        starts = SurfaceContext.sorted_time(valid["start"])
        ends = SurfaceContext.sorted_time(valid["end"])
        return np.searchsorted(starts, reference, "right") - np.searchsorted(ends, reference, "right")

    @staticmethod
    def completed_mean(rows: pd.DataFrame, reference: np.ndarray, minutes: int):
        valid = rows.loc[rows["interval_valid"]].sort_values("end")
        ends = nanoseconds(valid["end"])
        durations = (valid["end"] - valid["start"]).dt.total_seconds().to_numpy()
        cumulative = np.r_[0.0, np.cumsum(durations)]
        right = np.searchsorted(ends, reference, "left")
        left = np.searchsorted(ends, reference - minutes * 60 * SECOND, "left")
        count = right - left
        mean = np.divide(
            cumulative[right] - cumulative[left], count,
            out=np.full(len(reference), np.nan), where=count > 0,
        )
        return mean, count

    def features(self, query: pd.DataFrame) -> pd.DataFrame:
        if query[ID].isna().any() or query[ID].duplicated().any():
            raise ValueError("Query IDs must be unique and non-null")
        original_index = query.index
        query = query.reset_index(drop=True).copy()
        takeoff = timestamps(query[CLOCK])
        start = timestamps(query["AOBT_3_flt"]).fillna(timestamps(query["LOBT_flt"]))
        network_manager = start.notna()
        for column in ["EOBT_1_flt", "SCHED_TIME_UTC_mvt", "IOBT_flt"]:
            if column in query:
                start = start.fillna(timestamps(query[column]))
        duration = (takeoff - start).dt.total_seconds()
        fallback = start.isna() | duration.lt(0) | duration.gt(6 * 3600)
        start = start.where(~fallback, takeoff - pd.Timedelta(minutes=15))
        result = pd.DataFrame(index=query.index)
        result["traffic_reference_fallback"] = fallback.astype(float)
        result["traffic_reference_nm"] = (network_manager & ~fallback).astype(float)
        for label, clock in [("offblock", start), ("takeoff", takeoff)]:
            output = pd.DataFrame(np.nan, index=query.index, columns=METRICS)
            for airport, grouped_indices in query.groupby("ADEP_mvt").groups.items():
                indices = np.asarray(grouped_indices)
                indices = indices[clock.iloc[indices].notna().to_numpy()]
                if not len(indices) or airport not in self.groups:
                    continue
                reference = nanoseconds(clock.iloc[indices])
                departures, arrivals = self.groups[airport]
                own = departures.set_index(ID).reindex(query.loc[indices, ID])
                own_start = nanoseconds(own["start"])
                own_end = nanoseconds(own["end"])
                own_valid = own["interval_valid"].fillna(False).to_numpy(dtype=bool)
                own_active = own_valid & (own_start <= reference) & (own_end > reference)
                values = {
                    "taxi_out_active_proxy": self.active(departures, reference) - own_active,
                    "taxi_in_active": self.active(arrivals, reference),
                }
                values["surface_active_proxy"] = values["taxi_out_active_proxy"] + values["taxi_in_active"]
                departure_ends = self.sorted_time(departures["end"])
                arrival_starts = self.sorted_time(arrivals["start"])
                departure_starts = self.sorted_time(departures.loc[departures["interval_valid"], "start"])
                for minutes in [15, 30, 60]:
                    for name, events, own_event in [
                        ("departures", departure_ends, own_end),
                        ("arrivals", arrival_starts, None),
                        ("nm_starts", departure_starts, own_start),
                    ]:
                        count = self.counts(events, reference, minutes)
                        if own_event is not None:
                            own_present = (own_event >= reference - minutes * 60 * SECOND) & (own_event < reference)
                            if name == "nm_starts":
                                own_present &= own_valid
                            count = count - own_present
                        values[f"{name}_prev_{minutes}m"] = count
                scheduled = self.sorted_time(departures["scheduled"])
                own_scheduled = nanoseconds(own["scheduled"])
                for minutes in [15, 30]:
                    count = np.searchsorted(scheduled, reference + minutes * 60 * SECOND, "left") - np.searchsorted(
                        scheduled, reference, "left"
                    )
                    count -= (own_scheduled >= reference) & (own_scheduled < reference + minutes * 60 * SECOND)
                    values[f"planned_next_{minutes}m"] = count
                for minutes in [60, 180]:
                    departure_count = self.counts(departure_ends, reference, minutes) - (
                        (own_end >= reference - minutes * 60 * SECOND) & (own_end < reference)
                    )
                    values[f"net_ground_flow_{minutes}m"] = self.counts(
                        arrival_starts, reference, minutes
                    ) - departure_count
                for name, events in [("departure", departure_ends), ("arrival", arrival_starts)]:
                    positions = np.searchsorted(events, reference, "left") - 1
                    last = np.full(len(reference), np.nan)
                    valid = positions >= 0
                    last[valid] = (reference[valid] - events[positions[valid]]) / SECOND
                    values[f"seconds_since_{name}"] = last
                for minutes in [30, 60]:
                    values[f"completed_dep_proxy_mean_{minutes}m"], _ = self.completed_mean(
                        departures, reference, minutes
                    )
                    mean, count = self.completed_mean(arrivals, reference, minutes)
                    values[f"completed_arr_taxi_mean_{minutes}m"] = mean
                    values[f"completed_arr_count_{minutes}m"] = count
                runway_active = np.empty(len(reference))
                runway_past = np.empty(len(reference))
                query_runway = normalize(query.loc[indices, "RUNWAY_mvt"]).to_numpy()
                for runway in np.unique(query_runway):
                    keep = query_runway == runway
                    group = departures[departures["runway"].eq(runway)]
                    runway_active[keep] = self.active(group, reference[keep]) - own_active[keep]
                    runway_past[keep] = self.counts(self.sorted_time(group["end"]), reference[keep], 30)
                    runway_past[keep] -= (
                        (own_end[keep] >= reference[keep] - 30 * 60 * SECOND) & (own_end[keep] < reference[keep])
                    )
                values["runway_taxi_out_active_proxy"] = runway_active
                values["runway_departures_prev_30m"] = runway_past
                for column, values_array in values.items():
                    output.loc[indices, column] = values_array
            output["context_missing"] = output.isna().all(axis=1).astype(float)
            result = pd.concat([result, output.add_prefix(f"traffic_{label}_")], axis=1)
        result.index = original_index
        return result.astype("float32")
