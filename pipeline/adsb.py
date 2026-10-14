"""Surface ADS-B context per departure, built only from **other**, earlier flights.

    python -m pipeline.adsb prepare [--cutoff takeoff] [--output local_data/adsb_features]

A flight's own ADS-B taxi is never one of its features (that would measure the label from outside). Step 1 measures
each observed departure's ground segment (OPDI events and adsb.lol traces, matched by callsign, airport and takeoff;
BLOCK is never used); step 2 gives each departure B the medians of those measures over flights that took off before
B's cutoff (``--cutoff takeoff``, or ``strict``: before min(AOBT_3, SCHED, EOBT_1)), minus a 7-day baseline.
Rotation (the aircraft's previous arrival) is computed and saved but not used by the model.

Output: ``features_<cutoff>.parquet`` (one row per training and ranking departure), ``measurements.parquet``,
``rotation.parquet`` and ``manifest.json``.
"""
import argparse
import gzip
import io
import json
import logging
import tarfile
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq

from .config import ID, ROOT, TIME

log = logging.getLogger(__name__)

CHALLENGE = ROOT / "data" / "projects" / "prc_data_challenge"
OPDI_EVENTS = CHALLENGE / "eurocontrol" / "flight_events"
ADSBLOL_RAW = ROOT / "data/adsblol_history"
OPDI_CACHE = ROOT / "local_data" / "opdi_surface"
ADSBLOL_CACHE = ROOT / "local_data" / "adsblol_points"
ADSBLOL_EVENTS = ROOT / "local_data" / "adsblol_events"
OUTPUT = ROOT / "local_data" / "adsb_features"

# Centre (mean of the runway ends), elevation in ft.
AIRPORTS = {"EDDF": (50.0305, 8.5520, 364), "EDDM": (48.3538, 11.7861, 1487), "EGLL": (51.4710, -0.4615, 83),
            "EHAM": (52.3100, 4.7600, -11), "LEBL": (41.2970, 2.0830, 12), "LEMD": (40.4940, -3.5670, 2000),
            "LFPG": (49.0097, 2.5479, 392), "LIRF": (41.8003, 12.2389, 14), "LSZH": (47.4582, 8.5555, 1416),
            "LTFM": (41.2753, 28.7519, 325)}
HALF_LAT = 0.055   # ~6 km half-height of the box
MAX_AGL = 2500     # ft: airborne points kept to anchor the takeoff
PART_BYTES = 2_000_000_000  # size of every .tar.a? part except the last
EVENT_HALF_LAT = 0.2   # ~22 km: wide box to see takeoffs and landings even without ground coverage
EVENT_MAX_AGL = 6000   # ft

OPDI_TYPES = ["exit-parking_position", "entry-taxiway", "entry-runway", "exit-apron"]

WINDOW = 3 * 3600      # s before takeoff where the ground segment is searched
MATCH_TOLERANCE = 600  # s: last ground point at most this far from the takeoff
BREAK_GAP = 20 * 60    # s without points that cut the ground segment
STOP_KT = 1.5          # below this the aircraft is stopped

# Measures of the flight's own ground segment: NEVER features of that same flight.
OPDI_MEASURES = ["o_rwy_lead", "o_park_first", "o_park_first_match", "o_park_first_use", "o_park_match_lead",
                 "o_n_park", "o_twy_first", "o_apron_first"]
ADSBLOL_MEASURES = ["a_last_lead", "a_first_lead", "a_first_gs", "a_npts", "a_maxgap", "a_park_dur",
                    "a_move_lead", "a_path_km", "a_stop_time", "a_nstops", "a_mean_gs", "a_first_dist_km",
                    "a_prior_ground"]
MEASURES = OPDI_MEASURES + ADSBLOL_MEASURES
DEPARTURE_COLUMNS = [ID, "airport", "runway", "CALLSIGN_flt", TIME, "stand", "SCHED_TIME_UTC_mvt", "EOBT_1_flt",
                     "AOBT_3_flt"]

# Context: (feature, measure, window in s, level). Medians over flights before the cutoff.
CONTEXT = [
    ("c_opark_60m", "o_park_match_lead", 3600, "airport"),
    ("c_opark_30m", "o_park_match_lead", 1800, "airport"),
    ("c_otwy_30m", "o_twy_first", 1800, "airport"),
    ("c_otwy_15m", "o_twy_first", 900, "airport"),
    ("c_orwy_30m", "o_rwy_lead", 1800, "airport"),
    ("c_otwy_rwy_30m", "o_twy_first", 1800, "runway"),
    ("c_amove_60m", "a_move_parked", 3600, "airport"),
    ("c_afirst_30m", "a_first_lead", 1800, "airport"),
    ("c_afirst_rwy_30m", "a_first_lead", 1800, "runway"),
    ("c_astop_30m", "a_stop_time", 1800, "airport"),
    ("c_alast_30m", "a_last_lead", 1800, "airport"),
]
CONTEXT_VALUES = [name for name, *_ in CONTEXT]
# No counts: the number of flights seen depends on ADS-B coverage, not on congestion.
CONTEXT_FEATURES = CONTEXT_VALUES
BASELINE_DAYS = 7      # baseline: median of the daily medians of the previous 7 days
BASELINE_MIN_DAYS = 2
PARKED_MIN_SEC = 60  # adsb.lol: only a start seen from parked counts, which matches the off-block


def normalise(values: pd.Series) -> pd.Series:
    return values.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


# --- OPDI ------------------------------------------------------------------------------------------

def _opdi_file(args) -> str:
    path, out = args
    table = pq.read_table(path, columns=["flight_id", "type", "event_time", "info"],
                          filters=[("type", "in", OPDI_TYPES)])
    frame = table.to_pandas()
    info = frame.pop("info").map(lambda s: orjson.loads(s) if s else {})
    frame["apt"] = info.map(lambda x: x.get("osm_airport"))
    keep = frame.apt.isin(AIRPORTS).to_numpy()
    frame, info = frame[keep].copy(), info[keep]
    frame["ref"] = info.map(lambda x: x.get("osm_ref"))
    frame["use_s"] = info.map(lambda x: x.get("opdi_time_in_use_s")).astype(float)
    frame["cs"] = info.map(lambda x: (x.get("osn_flight_id") or "").strip() or None)
    frame["event_time"] = pd.to_datetime(frame.event_time).dt.tz_localize("UTC")
    frame.to_parquet(out, index=False)
    return path.name


def extract_opdi(source: Path = OPDI_EVENTS, cache: Path = OPDI_CACHE, workers: int = 16) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    todo = [(p, cache / p.name) for p in sorted(source.glob("flight_events_*.parquet"))
            if not (cache / p.name).exists()]
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_opdi_file, todo):
            log.info("OPDI %s", name)
    return sorted(cache.glob("flight_events_*.parquet"))


def opdi_measures(departures: pd.DataFrame, cache: Path = OPDI_CACHE) -> pd.DataFrame:
    events = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("flight_events_*.parquet"))],
                       ignore_index=True).drop_duplicates(["flight_id", "type", "event_time", "ref"])
    callsign = events.dropna(subset=["cs"]).groupby("flight_id").cs.agg(lambda s: s.mode().iat[0])
    events["cs"] = normalise(events.flight_id.map(callsign))
    dep = departures[[ID, "airport", "CALLSIGN_flt", TIME, "stand"]].copy()
    dep["cs"] = normalise(dep.CALLSIGN_flt)
    dep = dep.dropna(subset=["cs"])

    runway = events[events.type == "entry-runway"][["flight_id", "apt", "cs", "event_time"]]
    pairs = dep.merge(runway, left_on=["airport", "cs"], right_on=["apt", "cs"])
    pairs["rwy_lead"] = (pairs[TIME] - pairs.event_time).dt.total_seconds()
    pairs = pairs[(pairs.rwy_lead > -60) & (pairs.rwy_lead < 1200)]
    pairs = pairs.sort_values("rwy_lead").drop_duplicates([ID, "flight_id"])
    # A movement with two candidate tracks is ambiguous: dropped.
    pairs = pairs[pairs.groupby(ID).flight_id.transform("nunique").eq(1)].drop_duplicates(ID)

    surface = events[events.type != "entry-runway"][["flight_id", "apt", "type", "event_time", "ref", "use_s"]]
    cand = pairs[[ID, "flight_id", "airport", TIME, "rwy_lead", "stand"]].merge(
        surface, left_on=["flight_id", "airport"], right_on=["flight_id", "apt"])
    cand["lead"] = (cand[TIME] - cand.event_time).dt.total_seconds()
    cand = cand[(cand.lead >= cand.rwy_lead) & (cand.lead < WINDOW)]

    out = pairs.set_index(ID)[["rwy_lead"]].rename(columns={"rwy_lead": "o_rwy_lead"})
    first = cand.groupby([ID, "type"]).lead.max().unstack("type")
    out["o_twy_first"] = first.get("entry-taxiway")
    out["o_apron_first"] = first.get("exit-apron")
    park = cand[cand.type == "exit-parking_position"].sort_values("lead", ascending=False)
    park = park.assign(match=(normalise(park.ref).str.lstrip("0") == normalise(park.stand).str.lstrip("0"))
                       .fillna(False).astype(bool))
    head = park.drop_duplicates(ID).set_index(ID)
    out["o_park_first"] = head.lead
    out["o_park_first_match"] = head.match.astype(float)
    out["o_park_first_use"] = head.use_s
    out["o_park_match_lead"] = park[park.match].drop_duplicates(ID).set_index(ID).lead
    out["o_n_park"] = park.groupby(ID).size()
    return out.reset_index()[[ID] + OPDI_MEASURES]


# --- adsb.lol: extraction per day --------------------------------------------------------------------

class _Chain(io.RawIOBase):
    """The ``.tar.aa``, ``.tar.ab``... parts read as one stream."""

    def __init__(self, paths):
        self.files, self.i = [open(p, "rb") for p in paths], 0

    def readable(self):
        return True

    def readinto(self, buffer):
        while self.i < len(self.files):
            n = self.files[self.i].readinto(buffer)
            if n:
                return n
            self.i += 1
        return 0


def _events_near(t, lat, lon, agl, gs, track, callsigns, clat, clon):
    """Takeoffs and landings of a trace near an airport (wide box, low altitude), extrapolated to the ground."""
    near = ((np.abs(lat - clat) < EVENT_HALF_LAT) & (np.abs(lon - clon) < EVENT_HALF_LAT / np.cos(np.radians(clat)))
            & np.isfinite(agl) & (agl <= EVENT_MAX_AGL))
    idx = np.flatnonzero(near)
    if len(idx) < 3:
        return []
    airborne = (agl[idx] > 0) & (gs[idx] >= 50)
    ground_times = t[idx][agl[idx] == 0]
    breaks = np.flatnonzero((np.diff(t[idx]) > 300) | (airborne[1:] != airborne[:-1])) + 1
    events = []
    for run in np.split(np.arange(len(idx)), breaks):
        if len(run) < 3 or not airborne[run[0]]:
            continue
        rows = idx[run]
        tt, aa = t[rows], agl[rows]
        if aa.min() > 3000 or tt[-1] <= tt[0]:
            continue
        rate = (aa[-1] - aa[0]) / (tt[-1] - tt[0])
        k = min(5, len(rows)) - 1
        if rate > 3:
            low, local = 0, (aa[k] - aa[0]) / max(tt[k] - tt[0], 1)
            local = local if local > 1 else rate
            when = tt[0] - np.clip(aa[0] / local, 0, 300)
            before = ground_times[ground_times <= tt[0]]
            gap = tt[0] - before.max() if len(before) else np.nan
            kind = "DEP"
        elif rate < -3:
            low, local = len(rows) - 1, (aa[-1] - aa[-1 - k]) / max(tt[-1] - tt[-1 - k], 1)
            local = local if local < -1 else rate
            when = tt[-1] + np.clip(aa[-1] / -local, 0, 300)
            after = ground_times[ground_times >= tt[-1]]
            gap = after.min() - tt[-1] if len(after) else np.nan
            kind = "ARR"
        else:
            continue
        events.append((kind, when, aa[low], rate, len(rows), track[rows[low]], callsigns[rows[low]], gap))
    return events


def _trace(blob: bytes, points: bool = True, events: bool = True):
    """Ground or low points (narrow box) and takeoff/landing events (wide box); damaged traces are skipped."""
    try:
        raw = gzip.decompress(blob)
    except (OSError, EOFError, zlib.error):
        raw = blob
    try:
        data = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return None, None
    trace = data.get("trace") or []
    if not trace:
        return None, None
    t = data["timestamp"] + np.array([p[0] for p in trace], float)
    lat = np.array([p[1] for p in trace], float)
    lon = np.array([p[2] for p in trace], float)
    altitude = [p[3] for p in trace]
    ground = np.array([a == "ground" for a in altitude])
    baro = np.array([np.nan if a is None or a == "ground" else float(a) for a in altitude])
    gs = np.array([np.nan if p[4] is None else p[4] for p in trace], float)
    track = np.array([np.nan if p[5] is None else p[5] for p in trace], float)
    callsigns, current = [], None
    for point in trace:
        detail = point[8] if len(point) > 8 else None
        if detail and detail.get("flight"):
            current = detail["flight"].strip() or current
        callsigns.append(current)
    callsigns = np.array(callsigns, dtype=object)
    point_frames, event_rows = [], []
    for code, (clat, clon, elevation) in AIRPORTS.items():
        if not ((np.abs(lat - clat) < EVENT_HALF_LAT) & (np.abs(lon - clon) < EVENT_HALF_LAT / np.cos(np.radians(clat)))).any():
            continue
        agl = np.where(ground, 0.0, baro - elevation)
        if events:
            event_rows += [(code, *e) for e in _events_near(t, lat, lon, agl, np.nan_to_num(gs), track, callsigns, clat, clon)]
        if points:
            near = ((np.abs(lat - clat) < HALF_LAT) & (np.abs(lon - clon) < HALF_LAT / np.cos(np.radians(clat)))
                    & (ground | (np.isfinite(baro) & (baro - elevation <= MAX_AGL))))
            if near.any():
                point_frames.append(pd.DataFrame({"t": t[near], "lat": lat[near], "lon": lon[near],
                                                  "alt": np.where(ground[near], -1.0, baro[near]), "gs": gs[near],
                                                  "ground": ground[near], "cs": callsigns[near], "apt": code,
                                                  "icao": data["icao"]}))
    frame_points = pd.concat(point_frames) if point_frames else None
    frame_events = None
    if event_rows:
        frame_events = pd.DataFrame(event_rows, columns=["apt", "kind", "time", "agl_low", "rate", "n", "track", "cs",
                                                         "ground_gap"])
        frame_events["icao"], frame_events["reg"], frame_events["typecode"] = data["icao"], data.get("r"), data.get("t")
    return frame_points, frame_events


def _trace_batch(args):
    batch, points, events = args
    results = [_trace(blob, points, events) for blob in batch]
    frames = [r[0] for r in results if r[0] is not None]
    found = [r[1] for r in results if r[1] is not None]
    return (pd.concat(frames) if frames else None), (pd.concat(found) if found else None)


def _write(frame: pd.DataFrame, out: Path) -> None:
    temporary = out.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.rename(out)


def extract_adsblol_day(day_dir: Path, points_out: Path | None, events_out: Path | None, processes: int = 40):
    """One pass over the day's tar; writes the requested files (None: not computed)."""
    parts = sorted(day_dir.glob("*.tar.a?"))
    archive = tarfile.open(fileobj=io.BufferedReader(_Chain(parts), 1 << 22), mode="r|")
    want_points, want_events = points_out is not None, events_out is not None

    def batches():
        batch = []
        for member in archive:
            if member.isfile() and "/traces/" in member.name:
                batch.append(archive.extractfile(member).read())
                if len(batch) >= 400:
                    yield batch, want_points, want_events
                    batch = []
        if batch:
            yield batch, want_points, want_events

    with Pool(processes) as pool:
        results = list(pool.imap_unordered(_trace_batch, batches()))
    counts = {}
    if want_points:
        points = pd.concat([r[0] for r in results if r[0] is not None], ignore_index=True)
        _write(points, points_out)
        counts["points"] = len(points)
    if want_events:
        events = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
        _write(events, events_out)
        counts["events"] = len(events)
    return counts


def complete_days(raw: Path = ADSBLOL_RAW) -> list[Path]:
    """Days with every part downloaded: no ``.part`` and not the last of a running download."""
    days = sorted(d for d in raw.glob("*/*") if d.is_dir() and any(d.glob("*.tar.a?")))
    ready = []
    for day in days:
        if any(day.glob("*.part")):
            continue
        # Parts are exactly 2e9 bytes except the last: if the last one is that size, parts are missing.
        last = sorted(day.glob("*.tar.a?"))[-1]
        if (day / ".complete").exists() or last.stat().st_size != PART_BYTES:
            ready.append(day)
    return ready


def extract_adsblol(raw: Path = ADSBLOL_RAW, cache: Path = ADSBLOL_CACHE, events_cache: Path = ADSBLOL_EVENTS,
                    processes: int = 40) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    events_cache.mkdir(parents=True, exist_ok=True)
    for day in complete_days(raw):
        points_out, events_out = cache / f"{day.name}.parquet", events_cache / f"{day.name}.parquet"
        todo = (None if points_out.exists() else points_out, None if events_out.exists() else events_out)
        if todo == (None, None):
            continue
        start = time.time()
        try:
            counts = extract_adsblol_day(day, *todo, processes=processes)
        except (tarfile.ReadError, EOFError) as error:
            log.warning("adsb.lol %s incomplete, skipped: %s", day.name, error)
            continue
        log.info("adsb.lol %s: %s, %.0f s", day.name, counts, time.time() - start)


# --- adsb.lol: per-departure features ----------------------------------------------------------------

def load_points(days: list[str], cache: Path = ADSBLOL_CACHE) -> pd.DataFrame | None:
    frames = [pd.read_parquet(cache / f"{d}.parquet", columns=["t", "lat", "lon", "gs", "ground", "apt", "icao", "cs"])
              for d in days if (cache / f"{d}.parquet").exists()]
    if not frames:
        return None
    points = pd.concat(frames, ignore_index=True).sort_values(["icao", "t"], kind="stable")
    # The callsign only travels in some messages: propagated within each aircraft.
    points["cs"] = points.groupby("icao").cs.transform(lambda s: s.ffill().bfill())
    points = points.dropna(subset=["cs"])
    points["cs"] = normalise(points.cs)
    points["on_ground"] = points.ground | (points.gs < 40)
    points["gs"] = points.gs.fillna(0.0)
    return points.sort_values(["apt", "cs", "t"], kind="stable").reset_index(drop=True)


def segment(t, lat, lon, gs, on_ground, icao, takeoff) -> dict | None:
    """Descriptors of the ground segment ending at takeoff. Arrays of one callsign and airport."""
    best = None
    for aircraft in np.unique(icao):
        rows = (icao == aircraft) & on_ground & (t <= takeoff + 60)
        if rows.any():
            gap = takeoff - t[rows].max()
            if best is None or abs(gap) < abs(best[1]):
                best = (aircraft, gap)
    if best is None or abs(best[1]) > MATCH_TOLERANCE:
        return None
    keep = icao == best[0]
    t, lat, lon, gs, on_ground = t[keep], lat[keep], lon[keep], gs[keep], on_ground[keep]
    last = np.flatnonzero(on_ground & (t <= takeoff + 60))[-1]
    first = last
    while first > 0 and on_ground[first - 1] and t[first] - t[first - 1] <= BREAK_GAP:
        first -= 1
    ts, las, los, speed = t[first:last + 1], lat[first:last + 1], lon[first:last + 1], gs[first:last + 1]
    n = len(ts)
    dt = np.diff(ts)
    ky = 111_195.0
    kx = ky * np.cos(np.radians(las.mean()))
    step = np.hypot(np.diff(las) * ky, np.diff(los) * kx)
    moving = speed >= STOP_KT
    k = int(np.argmax(moving)) if moving.any() else n   # first moving point
    move = min(k, n - 1)
    out = {"a_last_lead": takeoff - ts[-1], "a_first_lead": takeoff - ts[0], "a_first_gs": speed[0],
           "a_npts": n, "a_maxgap": dt.max() if n > 1 else 0.0,
           "a_park_dur": ts[min(k, n) - 1] - ts[0] if k > 0 else 0.0,
           "a_move_lead": takeoff - ts[move],
           "a_first_dist_km": np.hypot((las[0] - las[-1]) * ky, (los[0] - los[-1]) * kx) / 1000,
           "a_prior_ground": float(first > 0 and on_ground[:first].any())}
    if move < n - 1:
        stopped = (speed[move:-1] < STOP_KT) & (speed[move + 1:] < STOP_KT)
        out.update(a_path_km=step[move:].sum() / 1000, a_stop_time=dt[move:][stopped].sum(),
                   a_nstops=int(stopped[0]) + int((np.diff(stopped.astype(int)) == 1).sum()),
                   a_mean_gs=float(np.average(speed[move:-1], weights=np.maximum(dt[move:], 1e-3))))
    else:
        out.update(a_path_km=0.0, a_stop_time=0.0, a_nstops=0, a_mean_gs=np.nan)
    return out


def adsblol_measures(departures: pd.DataFrame, cache: Path = ADSBLOL_CACHE) -> pd.DataFrame:
    dep = departures.dropna(subset=["CALLSIGN_flt"])
    days = sorted(set(dep[TIME].dt.strftime("%Y-%m-%d"))
                  | set((dep[TIME] - pd.Timedelta(seconds=WINDOW)).dt.strftime("%Y-%m-%d")))
    points = load_points(days, cache)
    if points is None:
        return pd.DataFrame(columns=[ID] + ADSBLOL_MEASURES)
    keys = (points.apt.astype(str) + "|" + points.cs.astype(str)).to_numpy()
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    index = dict(zip(keys[starts], zip(starts, np.r_[starts[1:], len(keys)])))
    t, lat, lon = points.t.to_numpy(), points.lat.to_numpy(), points.lon.to_numpy()
    gs, on_ground, icao = points.gs.to_numpy(), points.on_ground.to_numpy(), points.icao.to_numpy()
    takeoffs = (dep[TIME] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    rows = []
    for mid, airport, callsign, takeoff in zip(dep[ID], dep.airport, normalise(dep.CALLSIGN_flt), takeoffs):
        span = index.get(f"{airport}|{callsign}")
        if span is None:
            continue
        a, b = span
        lo = a + np.searchsorted(t[a:b], takeoff - WINDOW)
        hi = a + np.searchsorted(t[a:b], takeoff + MATCH_TOLERANCE)
        if hi - lo < 2:
            continue
        found = segment(t[lo:hi], lat[lo:hi], lon[lo:hi], gs[lo:hi], on_ground[lo:hi], icao[lo:hi], takeoff)
        if found is not None:
            rows.append({ID: mid, **found})
    return pd.DataFrame(rows, columns=[ID] + ADSBLOL_MEASURES)


def _adsblol_month(args):
    part, cache = args
    return adsblol_measures(part, cache)


# --- rotation: the previous arrival of the same aircraft ------------------------------------------------------

ROTATION = ["r_since_landing", "r_since_inblock", "r_inblock_minus_sched", "r_inbound_delay", "r_sched_turn",
            "r_same_stand", "r_inbound_taxi_in"]
MOVEMENT_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                    "SCHED_TIME_UTC_mvt", "CALLSIGN_flt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt"]
MATCH_SEC = 180          # ADS-B takeoff or landing against MVT_TIME
HEADING_TOLERANCE = 35   # degrees between takeoff track and runway heading
PREVIOUS_WINDOW = 36 * 3600
# Model features: only other flights' context. Rotation is computed and saved (rotation.parquet) but not used.
FEATURES = CONTEXT_FEATURES


def load_events(days: list[str], cache: Path = ADSBLOL_EVENTS) -> pd.DataFrame:
    frames = [pd.read_parquet(cache / f"{d}.parquet") for d in days if (cache / f"{d}.parquet").exists()]
    if not frames:
        return pd.DataFrame(columns=["apt", "kind", "time", "track", "cs", "icao"])
    events = pd.concat(frames, ignore_index=True).drop_duplicates(["icao", "apt", "kind", "time"])
    events["csn"] = normalise(events.cs)
    return events


def _runway_heading(runway: pd.Series) -> np.ndarray:
    digits = runway.astype("string").str.extract(r"^(\d{1,2})")[0]
    return pd.to_numeric(digits, errors="coerce").to_numpy(dtype=float) * 10


def match_movements(movements: pd.DataFrame, events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Each PRC movement with its ADS-B event (same airport, +/- 3 min, consistent callsign and runway): identity only."""
    phase = "DEP" if kind == "DEP" else "ARR"
    mv = movements[movements.PHASE_mvt == phase].copy()
    mv["apt"] = mv.ADEP_mvt if phase == "DEP" else mv.ADES_mvt
    mv["t_mvt"] = _seconds(mv.MVT_TIME_UTC_mvt)
    mv["csn"] = normalise(mv.CALLSIGN_flt)
    mv["heading"] = _runway_heading(mv.RUNWAY_mvt)
    ev = events[events.kind == kind][["apt", "icao", "time", "csn", "track"]].copy()
    # Match by MATCH_SEC buckets: each movement is compared only with its bucket and the neighbours.
    ev["bucket"] = np.floor(ev.time / MATCH_SEC).astype("int64")
    mv["bucket"] = np.floor(mv.t_mvt / MATCH_SEC).astype("int64")
    left = mv[[ID, "apt", "t_mvt", "csn", "heading", "bucket"]]
    pairs = pd.concat([left.assign(bucket=left.bucket + shift).merge(ev, on=["apt", "bucket"], suffixes=("", "_a"))
                       for shift in (-1, 0, 1)], ignore_index=True)
    pairs = pairs[(pairs.time - pairs.t_mvt).abs() <= MATCH_SEC]
    both = (pairs.csn.notna() & pairs.csn_a.notna()).to_numpy(bool)
    same = (pairs.csn == pairs.csn_a).fillna(False).to_numpy(bool)
    turn = np.abs((pairs.track.to_numpy(float) - pairs.heading.to_numpy(float) + 180) % 360 - 180)
    pairs = pairs[~(both & ~same) & ~(turn > HEADING_TOLERANCE)].assign(same=same[~(both & ~same) & ~(turn > HEADING_TOLERANCE)])
    pairs["score"] = (pairs.time - pairs.t_mvt).abs() - 600 * pairs.same
    pairs = pairs.sort_values("score").drop_duplicates(ID).drop_duplicates(["icao", "time"])
    return pairs[[ID, "icao", "time"]].rename(columns={"time": f"{kind.lower()}_time"})


def rotation_features(movements: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Each identified departure with its previous arrival at the same airport by the same aircraft."""
    dep = movements[movements.PHASE_mvt == "DEP"].merge(match_movements(movements, events, "DEP"), on=ID)
    arrivals = events[events.kind == "ARR"][["apt", "icao", "time"]].sort_values("time")
    dep["apt"] = dep.ADEP_mvt
    dep["t_mvt"] = _seconds(dep.MVT_TIME_UTC_mvt)
    prev = pd.merge_asof(dep.sort_values("t_mvt").assign(limit=lambda f: f.t_mvt - 300.0).sort_values("limit"),
                         arrivals.rename(columns={"time": "landing"}), left_on="limit", right_on="landing",
                         by=["apt", "icao"], direction="backward")
    prev = prev[(prev.t_mvt - prev.landing) <= PREVIOUS_WINDOW]
    inbound = match_movements(movements, events, "ARR").merge(
        movements[[ID, "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt",
                   "TAXITIME_SEC_mvt"]], on=ID).drop(columns=ID)
    inbound = inbound.rename(columns={"arr_time": "landing", "MVT_TIME_UTC_mvt": "in_mvt", "BLOCK_TIME_UTC_mvt": "in_block",
                                      "SCHED_TIME_UTC_mvt": "in_sched", "STAND_mvt": "in_stand",
                                      "TAXITIME_SEC_mvt": "in_taxi"})
    prev = prev.merge(inbound, on=["icao", "landing"], how="left")
    out = pd.DataFrame({ID: prev[ID].to_numpy()})
    in_block = _seconds(prev.in_block)
    out["r_since_landing"] = (prev.t_mvt - prev.landing).to_numpy()
    out["r_since_inblock"] = prev.t_mvt.to_numpy() - in_block
    out["r_inblock_minus_sched"] = in_block - _seconds(prev.SCHED_TIME_UTC_mvt)
    out["r_inbound_delay"] = _seconds(prev.in_mvt) - _seconds(prev.in_sched)
    out["r_sched_turn"] = _seconds(prev.SCHED_TIME_UTC_mvt) - _seconds(prev.in_sched)
    same = normalise(prev.in_stand) == normalise(prev.STAND_mvt)
    out["r_same_stand"] = same.astype("Float64").where(prev.in_stand.notna()).astype(float).to_numpy()
    out["r_inbound_taxi_in"] = prev.in_taxi.to_numpy(dtype=float)
    return out


def read_movements(raw_dir: Path = CHALLENGE / "raw") -> pd.DataFrame:
    """Raw training and ranking departures and arrivals (ranking arrivals keep their BLOCK)."""
    files = sorted(raw_dir.glob("training_*.parquet")) + [raw_dir / "ranking.parquet"]
    frame = pd.concat([pd.read_parquet(p, columns=MOVEMENT_COLUMNS) for p in files], ignore_index=True)
    frame = frame.dropna(subset=[ID, "MVT_TIME_UTC_mvt"])
    frame[ID] = frame[ID].astype("int64")
    return frame


def _rotation_month(args):
    part, events_cache = args
    first, last = part.MVT_TIME_UTC_mvt.min(), part.MVT_TIME_UTC_mvt.max()
    days = pd.date_range((first - pd.Timedelta(seconds=PREVIOUS_WINDOW)).normalize(), last.normalize(), freq="D")
    events = load_events([d.strftime("%Y-%m-%d") for d in days], events_cache)
    if events.empty:
        return pd.DataFrame(columns=[ID] + ROTATION)
    return rotation_features(part, events)


def rotation_table(movements: pd.DataFrame, events_cache: Path = ADSBLOL_EVENTS) -> pd.DataFrame:
    """Rotation of each departure, by month; arrivals in the 36 h before the month count too."""
    month = movements.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    jobs = []
    for m in sorted(month.unique()):
        start = movements.MVT_TIME_UTC_mvt[month == m].min() - pd.Timedelta(seconds=PREVIOUS_WINDOW)
        part = movements[(month == m) | ((movements.PHASE_mvt == "ARR") & (movements.MVT_TIME_UTC_mvt >= start)
                                         & (movements.MVT_TIME_UTC_mvt < movements.MVT_TIME_UTC_mvt[month == m].min()))]
        jobs.append((part, events_cache))
    with ProcessPoolExecutor(min(len(jobs), 14)) as pool:
        parts = list(pool.map(_rotation_month, jobs))
    table = pd.concat(parts, ignore_index=True).drop_duplicates(ID)
    return table.astype({ID: float, **{c: float for c in ROTATION}})  # IDs as in interim_nm (float64)


# --- context: only other flights before the cutoff ---------------------------------------------------

def _seconds(values: pd.Series) -> np.ndarray:
    return (values - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(dtype=float)


def cutoff_seconds(departures: pd.DataFrame, cutoff: str) -> np.ndarray:
    """Instant of B up to which other flights' information is allowed. Never after takeoff."""
    takeoff = _seconds(departures[TIME])
    if cutoff == "takeoff":
        return takeoff
    if cutoff == "strict":
        others = [_seconds(departures[c]) for c in ("SCHED_TIME_UTC_mvt", "EOBT_1_flt", "AOBT_3_flt")]
        return np.fmin.reduce([takeoff, *others])  # fmin ignores NaN
    raise ValueError(f"Unknown cutoff: {cutoff}")


def _window_median(times: np.ndarray, values: np.ndarray, cutoffs: np.ndarray, window: float):
    """Median and count of ``values`` with (sorted) ``times`` in [cutoff - window, cutoff)."""
    lo = np.searchsorted(times, cutoffs - window, "left")
    hi = np.searchsorted(times, cutoffs, "left")
    median = np.full(len(cutoffs), np.nan)
    for i in np.flatnonzero(hi > lo):
        median[i] = np.median(values[lo[i]:hi[i]])
    return median, (hi - lo).astype(float)


def _daily_baseline(takeoff: np.ndarray, values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Median of the daily medians of the ``BASELINE_DAYS`` full days before the cutoff day."""
    day = np.floor(takeoff / 86400).astype("int64")
    daily = pd.Series(values).groupby(day).median()
    if daily.empty:
        return np.full(len(cutoffs), np.nan)
    full = daily.reindex(np.arange(daily.index.min(), daily.index.max() + 1))
    # the baseline of day d uses days d-BASELINE_DAYS ... d-1
    baseline = full.rolling(BASELINE_DAYS, min_periods=BASELINE_MIN_DAYS).median().shift(1)
    return baseline.reindex(np.floor(cutoffs / 86400).astype("int64")).to_numpy()


def context_features(departures: pd.DataFrame, measurements: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """One row per departure: for each ``CONTEXT``, recent median minus the 7-day baseline.

    A measure becomes available at its flight's takeoff, so B never sees its own. The difference with the previous
    week absorbs the changes in ADS-B receiver coverage between 2025 and 2026.
    """
    frame = departures[[ID, "airport", "runway", TIME]].merge(measurements, on=ID, how="left", validate="1:1")
    frame["a_move_parked"] = frame.a_move_lead.where(frame.a_park_dur >= PARKED_MIN_SEC)
    takeoff = _seconds(frame[TIME])
    cutoffs = cutoff_seconds(departures, cutoff)  # the left merge keeps the order
    out = pd.DataFrame({ID: frame[ID].to_numpy()})
    for name, measure, window, level in CONTEXT:
        key = frame.airport.astype(str)
        if level == "runway":
            key = (key + ":" + frame.runway.astype(str)).where(frame.runway.notna())
        result = np.full(len(frame), np.nan)
        for _, rows in frame.groupby(key, sort=False).indices.items():
            values = frame[measure].to_numpy(dtype=float)[rows]
            valid = np.isfinite(values)
            order = np.argsort(takeoff[rows][valid], kind="stable")
            times, series = takeoff[rows][valid][order], values[valid][order]
            recent, _ = _window_median(times, series, cutoffs[rows], window)
            result[rows] = recent - _daily_baseline(times, series, cutoffs[rows])
        out[name] = result
    return out


# --- baseline at the start of a ranking month -------------------------------------------------------------

def baseline_days(departures: pd.DataFrame, events_cache: Path = ADSBLOL_EVENTS) -> list[str]:
    """Days without challenge departures among the ``BASELINE_DAYS`` before the first day of each covered stretch."""
    days = pd.DatetimeIndex(sorted(set(departures[TIME].dt.strftime("%Y-%m-%d"))))
    have = set(days)
    out = set()
    for day in days:
        if day - pd.Timedelta(days=1) in have:
            continue
        for k in range(1, BASELINE_DAYS + 1):
            candidate = day - pd.Timedelta(days=k)
            if candidate not in have and (events_cache / f"{candidate:%Y-%m-%d}.parquet").exists():
                out.add(f"{candidate:%Y-%m-%d}")
    return sorted(out)


def pseudo_departures(days: list[str], departures: pd.DataFrame, events_cache: Path = ADSBLOL_EVENTS) -> pd.DataFrame:
    """adsb.lol takeoffs with callsign on ``days`` as departures without a challenge row (negative ``ID``).

    They only feed later departures' context (baseline and windows), never model rows; e.g. July 2026 has no June.
    """
    frames = [pd.read_parquet(events_cache / f"{d}.parquet") for d in days if (events_cache / f"{d}.parquet").exists()]
    columns = list(departures.columns)
    if not frames:
        return departures.iloc[:0].copy()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[(ev.kind == "DEP") & ev.cs.notna()].copy()
    ev["cs"] = normalise(ev.cs)
    ev = ev.dropna(subset=["cs"]).drop_duplicates(["apt", "icao", "time"]).reset_index(drop=True)
    used = departures.dropna(subset=["runway"]).groupby("airport").runway.agg(lambda s: sorted(set(s)))
    runway = []
    for airport, track in zip(ev.apt, ev.track):
        names = used.get(airport, [])
        headings = _runway_heading(pd.Series(names, dtype="string"))
        match = [n for n, h in zip(names, headings)
                 if np.isfinite(h) and abs((track - h + 180) % 360 - 180) <= HEADING_TOLERANCE]
        runway.append(match[0] if len(match) == 1 else None)
    out = pd.DataFrame({ID: -np.arange(1, len(ev) + 1, dtype=float), "airport": ev.apt.to_numpy(),
                        "runway": runway, "CALLSIGN_flt": ev.cs.to_numpy(),
                        TIME: pd.to_datetime(ev.time.to_numpy(), unit="s", utc=True)})
    for column in columns:
        if column not in out:
            out[column] = None
        if pd.api.types.is_datetime64_any_dtype(departures[column]):
            out[column] = pd.to_datetime(out[column], utc=True).astype(departures[column].dtype)
    return out[columns]


# --- full preparation ----------------------------------------------------------------------------

def read_departures(interim_dir: Path) -> pd.DataFrame:
    files = sorted(interim_dir.glob("training_*.parquet")) + [interim_dir / "ranking.parquet"]
    return pd.concat([pd.read_parquet(p, columns=DEPARTURE_COLUMNS) for p in files], ignore_index=True)


def flight_measurements(departures: pd.DataFrame, processes: int = 40) -> pd.DataFrame:
    """Ground-segment measures of each observed departure. Intermediate product, never a feature."""
    opdi = opdi_measures(departures)
    log.info("OPDI: %d departures measured", len(opdi))
    by_month = [(part, ADSBLOL_CACHE) for _, part in departures.groupby(departures[TIME].dt.strftime("%Y-%m"))]
    with ProcessPoolExecutor(min(len(by_month), 14)) as pool:
        adsblol = pd.concat(list(pool.map(_adsblol_month, by_month)), ignore_index=True)
    log.info("adsb.lol: %d departures measured", len(adsblol))
    measured = opdi.merge(adsblol, on=ID, how="outer")
    return measured.astype({c: float for c in MEASURES}).sort_values(ID).reset_index(drop=True)


def prepare(interim_dir: Path, output: Path, cutoffs=("takeoff",), processes: int = 40, extract: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    departures = read_departures(interim_dir)
    if extract:
        extract_opdi()
        extract_adsblol(processes=processes)
    pseudo_days = baseline_days(departures)
    pseudo = pseudo_departures(pseudo_days, departures)
    log.info("baseline: %d departures rebuilt from %s", len(pseudo), pseudo_days)
    measured = flight_measurements(pd.concat([departures, pseudo], ignore_index=True), processes)
    # With the ADS-B takeoff, the OPDI runway entry is biased: not used.
    measured.loc[measured[ID] < 0, "o_rwy_lead"] = np.nan
    measured[measured[ID] < 0].to_parquet(output / "measurements_pseudo.parquet", index=False)
    measured[measured[ID] >= 0].to_parquet(output / "measurements.parquet", index=False)
    context_departures = pd.concat([departures, pseudo], ignore_index=True)
    meta = departures[[ID, "airport"]].assign(month=departures[TIME].dt.strftime("%Y-%m"))
    rotation = rotation_table(read_movements())
    rotation.to_parquet(output / "rotation.parquet", index=False)
    log.info("rotation: %d departures with a previous arrival", int(rotation.r_since_landing.notna().sum()))
    coverage = {}
    for cutoff in cutoffs:
        table = context_features(context_departures, measured, cutoff)
        table = table[table[ID] >= 0].merge(rotation, on=ID, how="left", validate="1:1")
        table = table[[ID] + CONTEXT_FEATURES + ROTATION]  # the model reads only FEATURES
        table.to_parquet(output / f"features_{cutoff}.parquet", index=False)
        joined = meta.merge(table, on=ID)
        coverage[cutoff] = {name: {f"{a}|{m}": round(float(v), 4) for (a, m), v in
                                   joined[name].notna().groupby([joined.airport, joined.month]).mean().items()}
                            for name in ("c_otwy_30m", "c_afirst_30m", "r_since_landing", "r_since_inblock")}
    manifest = {"departures": int(len(departures)), "measured": int((measured[ID] >= 0).sum()),
                "baseline_days": pseudo_days, "pseudo_departures": int(len(pseudo)),
                "pseudo_measured": int((measured[ID] < 0).sum()),
                "opdi_files": sorted(p.name for p in OPDI_CACHE.glob("*.parquet")),
                "adsblol_days": sorted(p.stem for p in ADSBLOL_CACHE.glob("*.parquet")),
                "adsblol_event_days": sorted(p.stem for p in ADSBLOL_EVENTS.glob("*.parquet")),
                "context": [list(c) for c in CONTEXT], "features": FEATURES, "coverage": coverage}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.adsb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "extract"])
    parser.add_argument("--interim-dir", type=Path, default=ROOT / "local_data" / "interim_nm")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cutoff", nargs="+", choices=["takeoff", "strict"], default=["takeoff"])
    parser.add_argument("--processes", type=int, default=40)
    parser.add_argument("--no-extract", action="store_true",
                        help="Use only the days already extracted (does not change days already prepared)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.stage == "extract":
        extract_opdi()
        extract_adsblol(processes=args.processes)
        return
    manifest = prepare(args.interim_dir, args.output, tuple(args.cutoff), args.processes, not args.no_extract)
    print(json.dumps({k: manifest[k] for k in ["departures", "measured"]}, indent=1))


if __name__ == "__main__":
    main()
