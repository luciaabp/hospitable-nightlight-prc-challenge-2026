"""Build the departure features; arrivals are used for traffic only."""

import logging
from pathlib import Path

import pandas as pd

from . import aircraft, airport, airport_geometry, calendar, delays, events, flight, queue, traffic, weather

log = logging.getLogger(__name__)

CATEGORIES = {
    "delays": delays.FEATURES,
    "airport": airport.FEATURES,
    "airport_geometry": airport_geometry.FEATURES,
    "flight": flight.FEATURES,
    "traffic": traffic.FEATURES,
    "queue": queue.FEATURES,
    "weather": weather.FEATURES,
    "events": events.FEATURES,
    "calendar": calendar.FEATURES,
    "aircraft": aircraft.FEATURES,
}


def build_features(
    movements: pd.DataFrame,
    metar_path: Path,
    aircraft_specs_path: Path,
    events_path: Path,
    xplane_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply every feature group in order. ``movements`` must include departures and arrivals.

    The X-Plane stand geometry is added only with ``xplane_dir`` (context pipeline); the specialist runs without it.
    """
    departures = movements[movements["PHASE_mvt"] == "DEP"].copy()
    log.info("departures: %s of %s movements", f"{len(departures):,}", f"{len(movements):,}")

    departures = delays.add_delay_features(departures)
    departures = airport.add_airport_features(departures)
    if xplane_dir is not None:
        departures = airport_geometry.add_airport_geometry_features(departures, xplane_dir)
    departures = flight.add_flight_features(departures)
    departures = traffic.add_traffic_features(departures, movements)  # uses the delays proxy
    departures = queue.add_queue_features(departures)  # uses the normalized runway from airport
    departures = weather.add_weather_features(departures, metar_path)
    departures = events.add_event_features(departures, events_path)
    departures = calendar.add_calendar_features(departures)
    departures = aircraft.add_aircraft_features(departures, aircraft_specs_path)
    return departures
