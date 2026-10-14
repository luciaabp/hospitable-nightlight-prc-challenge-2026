"""Fill an absent aircraft type from the OpenSky flight list, conservatively.

Linking a movement to a flight is not the same as recovering a variable: most
movements without a Network Manager cross already carry their type, and the
flight list only adds one where the movement has none. It is filled only when a
single flight matches, because a wrong type is worse than an absent one.

The list covers 2025 and both ranking months, so the same fill is available when
predicting. It reads no label and no hidden column.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .aircraft import load_mtow

log = logging.getLogger(__name__)
COLUMNS = ["flt_id", "dof", "adep", "first_seen", "typecode"]
FLAG = "aircraft_type_backfilled"
DEFAULT_TOLERANCE = 60


def load_flight_list(directory, airports=None, pattern="flight_list_*.parquet"):
    """Departures with a known type, keyed by airport and first observation."""
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{pattern}' under {directory}")
    frame = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in files], ignore_index=True)
    # flt_id is space padded in the source; without stripping nothing ever matches.
    frame["flt_id"] = frame.flt_id.str.strip()
    frame = frame[frame.typecode.notna() & frame.adep.notna()]
    if airports:
        frame = frame[frame.adep.isin(airports)]
    # Timezone-aware values become an object array in numpy, which searchsorted
    # cannot compare; everything here is UTC, so the offset is dropped instead.
    frame["first_seen"] = pd.to_datetime(frame.first_seen, utc=True).dt.tz_localize(None)
    return frame.dropna(subset=["first_seen"]).sort_values("first_seen").reset_index(drop=True)


def _unique_match(times, candidates, tolerance):
    """Index of the only candidate within the window, or -1 when it is not unique."""
    seconds = np.timedelta64(int(tolerance), "s")
    low = np.searchsorted(candidates, times - seconds, "left")
    high = np.searchsorted(candidates, times + seconds, "right")
    return np.where(high - low == 1, low, -1)


def backfill_aircraft(frame, flight_list, specs_path, tolerance=DEFAULT_TOLERANCE,
                      time_column="MVT_TIME_UTC_mvt", airport_column="airport"):
    """Fill aircraft_type and mtow where absent; mark what was filled."""
    result = frame.copy()
    if "aircraft_type" not in result:
        raise KeyError("aircraft_type must exist before it can be filled")
    filled = np.zeros(len(result), dtype=bool)
    types = result.aircraft_type.astype("string").to_numpy(copy=True)
    missing = pd.isna(types)
    times = pd.to_datetime(result[time_column], utc=True).dt.tz_localize(None).to_numpy()
    airports = result[airport_column].astype(str).to_numpy()
    for airport, group in flight_list.groupby("adep", sort=False):
        rows = np.flatnonzero(missing & (airports == airport))
        if not len(rows):
            continue
        # Rows must be in time order for the search; the source is already sorted.
        order = np.argsort(times[rows], kind="stable")
        ordered = rows[order]
        found = _unique_match(times[ordered], group.first_seen.to_numpy(), tolerance)
        hit = found >= 0
        types[ordered[hit]] = group.typecode.to_numpy()[found[hit]]
        filled[ordered[hit]] = True
    result["aircraft_type"] = pd.array(types, dtype="string")
    result[FLAG] = filled
    if filled.any():
        # mtow follows the type, so it is recomputed for exactly the filled rows.
        recovered = result.loc[filled, "aircraft_type"].map(load_mtow(Path(specs_path))).astype(float)
        result.loc[filled, "mtow"] = recovered.to_numpy()
    log.info("aircraft type filled for %d of %d movements without one", filled.sum(), missing.sum())
    return result
