"""Stand occupancy context from arrival in-block times and departure takeoffs (block T).

Another aircraft blocking in at the departure's stand before takeoff implies the
departure already left it. Only arrival BLOCK, SCHED, NM AOBT and takeoff clocks are
read: departure BLOCK and taxi labels never enter. Ranking publishes arrival BLOCK
for the whole period, so arrivals after takeoff are also visible at inference.
"""

import numpy as np
import pandas as pd

from .config import CLOCK, ID

SECOND = 10**9
UNKNOWN_STANDS = {"", "UNKNOWN", "NAN", "NONE", "UNK", "ZZZ", "-"}
COLUMNS = [
    "stand_last_arr_gap_s", "stand_second_arr_gap_s", "stand_last_arr_same_type",
    "stand_last_arr_same_operator", "stand_last_arr_minus_sched_s",
    "stand_last_arr_minus_nm_offblock_s", "stand_arrivals_prev_60m",
    "stand_next_arr_gap_s", "stand_prev_dep_gap_s", "stand_known",
]


def _key(frame: pd.DataFrame, airport: str) -> pd.Series:
    stand = frame["STAND_mvt"].astype("string").str.strip().str.upper()
    stand = stand.where(~stand.isin(UNKNOWN_STANDS))
    return (frame[airport].astype("string").str.strip().str.upper() + ":" + stand).astype(object)


def _ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]").array.asi8


def _operator(frame: pd.DataFrame) -> pd.Series:
    fallback = frame["FLIGHT_mvt"].astype("string").str.strip().str.upper().str[:3]
    return frame["AIRCRAFT_OPERATOR_flt"].astype("string").str.strip().str.upper().fillna(fallback)


def _same(left: pd.Series, right: pd.Series) -> np.ndarray:
    known = left.notna().to_numpy() & right.notna().to_numpy()
    same = left.fillna("").to_numpy(str) == right.fillna("").to_numpy(str)
    return np.where(known, same.astype(float), np.nan)


def stand_features(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """One row per departure, aligned to ``departures.index``.

    departures: ID, ADEP_mvt, STAND_mvt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt,
    AOBT_3_flt, AIRCRAFT_TYPE_mvt, AIRCRAFT_OPERATOR_flt, FLIGHT_mvt.
    arrivals: ADES_mvt, STAND_mvt, BLOCK_TIME_UTC_mvt, AIRCRAFT_TYPE_mvt,
    AIRCRAFT_OPERATOR_flt, FLIGHT_mvt.
    """
    if departures[ID].duplicated().any():
        raise ValueError("Departure IDs must be unique")
    dep = departures.reset_index(drop=True).assign(
        key=lambda f: _key(f, "ADEP_mvt"), operator=lambda f: _operator(f))
    arr = arrivals.loc[arrivals["BLOCK_TIME_UTC_mvt"].notna()].reset_index(drop=True).assign(
        key=lambda f: _key(f, "ADES_mvt"), operator=lambda f: _operator(f))
    arr = arr.loc[arr["key"].notna()]
    out = pd.DataFrame(np.nan, index=dep.index, columns=COLUMNS)
    out["stand_known"] = dep["key"].notna().astype(float)
    takeoff_all = _ns(dep[CLOCK])
    arrival_groups = arr.groupby("key").indices
    departure_groups = dep.loc[dep["key"].notna() & dep[CLOCK].notna()].groupby("key").indices
    for key, rows in departure_groups.items():
        takeoff = takeoff_all[rows]
        # Previous departure from the same stand, by takeoff, excluding the flight itself.
        order = np.argsort(takeoff, kind="stable")
        previous = np.full(len(rows), np.nan)
        previous[order[1:]] = (takeoff[order[1:]] - takeoff[order[:-1]]) / SECOND
        out.loc[rows, "stand_prev_dep_gap_s"] = previous
        if key not in arrival_groups:
            out.loc[rows, "stand_arrivals_prev_60m"] = 0.0
            continue
        group = arr.iloc[arrival_groups[key]].sort_values("BLOCK_TIME_UTC_mvt", kind="stable")
        blocks = _ns(group["BLOCK_TIME_UTC_mvt"])
        # Strictly before takeoff: an arrival exactly at takeoff counts as later.
        right = np.searchsorted(blocks, takeoff, "left")
        last, second = right - 1, right - 2
        has_last, has_second, has_next = last >= 0, second >= 0, right < len(blocks)
        gap = np.full(len(rows), np.nan)
        gap[has_last] = (takeoff[has_last] - blocks[last[has_last]]) / SECOND
        out.loc[rows, "stand_last_arr_gap_s"] = gap
        second_gap = np.full(len(rows), np.nan)
        second_gap[has_second] = (takeoff[has_second] - blocks[second[has_second]]) / SECOND
        out.loc[rows, "stand_second_arr_gap_s"] = second_gap
        following = np.full(len(rows), np.nan)
        following[has_next] = (blocks[right[has_next]] - takeoff[has_next]) / SECOND
        out.loc[rows, "stand_next_arr_gap_s"] = following
        out.loc[rows, "stand_arrivals_prev_60m"] = right - np.searchsorted(blocks, takeoff - 3600 * SECOND, "left")
        if has_last.any():
            with_last = rows[has_last]
            previous_arrival = group.iloc[last[has_last]]
            own = dep.iloc[with_last]
            out.loc[with_last, "stand_last_arr_same_type"] = _same(
                own["AIRCRAFT_TYPE_mvt"].astype("string").str.strip().str.upper(),
                previous_arrival["AIRCRAFT_TYPE_mvt"].astype("string").str.strip().str.upper())
            out.loc[with_last, "stand_last_arr_same_operator"] = _same(own["operator"], previous_arrival["operator"])
            block = blocks[last[has_last]]
            for column, clock in [("stand_last_arr_minus_sched_s", "SCHED_TIME_UTC_mvt"),
                                  ("stand_last_arr_minus_nm_offblock_s", "AOBT_3_flt")]:
                reference = pd.to_datetime(own[clock], utc=True).astype("datetime64[ns, UTC]")
                value = (block - reference.array.asi8) / SECOND
                out.loc[with_last, column] = np.where(reference.notna().to_numpy(), value, np.nan)
    out.index = departures.index
    return out.astype("float32")
