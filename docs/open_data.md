# Open data

Besides the organizer files (`training_2025-*.parquet`, `ranking.parquet`, `submitting.parquet` and the organizer's
`metar.parquet`), the submission uses only public data. This page lists each source, where it comes from, how it is
fetched and which part of the model reads it.

| source | provider | licence / terms | fetched by | read by |
|---|---|---|---|---|
| ADS-B traces (`globe_history`) | adsb.lol | ODbL 1.0 | `scripts/download_adsblol_days.py`, or during the run | base model, off-block detector, context model |
| Flight events (OPDI v002) | EUROCONTROL | open data with attribution | `scripts/download_opdi_events.py` | context model (`pipeline/adsb.py`) |
| Flight list (OPDI v002) | EUROCONTROL | open data with attribution | base engine, stage `base_reference` | base model (`external_data.py`) |
| Daily performance series | EUROCONTROL (ansperformance.eu) | public | base engine, stage `base_reference` | base model (`external_data.py`) |
| METAR (ASOS) | Iowa Environmental Mesonet, Iowa State University | public | IEM ASOS download (`--metar-csv`); base engine, stage `base_reference` | specialist, context model, base model |
| Airport layouts (`apt.dat`) | X-Plane Scenery Gateway | GPL v2+ | Gateway (`--xplane`); base engine via the Gateway API | specialist, context model, base model |
| Airport events | EUROCONTROL Airport Corner | public | compiled by hand into `--taxi-events` | specialist, context model |

No source needs an account or a key. All of them are downloaded once and then read locally; a run never writes to
them.

## adsb.lol ADS-B traces

- **What**: daily snapshots of every aircraft trace received by the adsb.lol network (readsb `globe_history`).
- **Where**: GitHub releases of `adsblol/globe_history_2025` and `adsblol/globe_history_2026`, one release per day
  (`v<YYYY.MM.DD>-planes-readsb-prod-0`, or `staging` when there is no `prod`), split into `.tar.aa`, `.tar.ab`, …
  parts of about 2 GB.
- **Download**: `python scripts/download_adsblol_days.py --start 2025-01-01 --end 2026-07-31 [--root DIR]` stores the
  parts under `<root>/<year>/<day>/` and marks finished days with `.complete`. Pass the folder as `--adsblol-store`;
  days missing from it are downloaded during the run.
- **Use**:
  - `components/base_model/base_model.py cut` keeps the points around the ten airports (`adsb/cut/<day>.parquet`);
    the base engine (`adsb_events.py`) builds its off-block/takeoff events from them.
  - `components/base_model/adsb_relaxed.py` adds relaxed ground segments (variant D2sub).
  - `components/offblock/` is our own off-block detector (ob2 → ob4 → rule v4), used by the FG4 variants and the
    correctors.
  - `pipeline/adsb.py` measures the ground segment of **other, earlier** departures to build surface-congestion
    context.
- A flight's own ADS-B taxi is never one of its features in the context model; see the module docstrings for the
  cut-off rules.

## EUROCONTROL OPDI

The Open Performance Data Initiative (<https://www.opdi.aero/>), version v002.

- **Flight events** — `https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/`, one parquet
  per 10-day window (`flight_events_<start>_<end>.parquet`, windows anchored on 2022-01-01).
  - Download: `python scripts/download_opdi_events.py --months 2025-01 … 2026-07 --output-dir DIR` estimates the
    volume; add `--download` to fetch it. Pass `DIR` as `--opdi-events`.
  - Use: `pipeline/adsb.py` reads the surface events `exit-parking_position`, `entry-taxiway`, `entry-runway` and
    `exit-apron` to measure ground segments of earlier departures. With an ADS-B takeoff the OPDI runway entry is
    biased and is not used.
- **Flight list** — `…/OPDI/v002/flight_list/flight_list_<YYYYMM>.parquet`, one per month.
  - Downloaded by the base engine (`engines/final_model/src/external_data.py download`) inside stage `base_reference`.
  - Use: aircraft ground time since its previous landing (`ext_opdi_*` columns of the base model).

## EUROCONTROL daily performance series

- **Where**: `https://www.eurocontrol.int/performance/data/download/csv/<name>_<year>.csv` (published on
  <https://ansperformance.eu/csv/>).
- **Files**: `atfm_slot_adherence`, `all_pre_departure_delays` and `atc_pre_departure_delays`, per year.
- **Download**: base engine, `src/external_data.py download`, stage `base_reference`.
- **Use**: slot adherence and total/ATC pre-departure delay per (UTC day, airport) in the base model.

## METAR (Iowa Environmental Mesonet)

The organizer's `metar.parquet` is used as given. In addition we use the public METAR archive of the Iowa
Environmental Mesonet ASOS service (<https://mesonet.agron.iastate.edu/request/download.phtml>).

- **Stations**: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM; times in UTC.
- **`--metar-csv`** (`metar_iem.csv`): one comma-separated file for the ten stations over the training and ranking
  months, with at least the columns `station`, `valid`, `tmpf`, `dwpf`, `relh`, `drct`, `sknt`, `gust`, `p01i`,
  `vsby`, `skyl1`, `wxcodes` and `metar` (missing values as `M` or empty).
  - Use: `src/feature_engineering/weather.py` (specialist and `pipeline.prepare`) attaches to each departure the last
    report before takeoff and the last before scheduled time, at most 90 min old: temperature, dew point, wind, gust,
    visibility band, cloud base, precipitation and phenomena (rain, snow, fog, thunder, freezing, hail, showers).
    `pipeline/weather.py` reuses these features in the context model.
- **Base engine**: `engines/final_model/src/weather_rotation.py download` downloads its own copy, one request per airport, through
  `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (`tmpf`, `dwpf`, `sknt`, `gust`, `vsby`, `wxcodes`,
  `skyc1`, `skyl1`), and uses the last report up to 2 h before the movement (`met_*` columns).

## X-Plane airport layouts

- **What**: `apt.dat` of the recommended scenery of each airport on the X-Plane Scenery Gateway
  (<https://gateway.x-plane.com>, API `https://gateway.x-plane.com/apiv1`).
- **Records read**: `100` (runway ends), `1201`/`1202` (taxi network nodes and edges) and `1300` (stands).
- **`--xplane`**: folder with `<ICAO>/apt.dat` for the ten airports.
  - `src/feature_engineering/airport_geometry.py`: stand geometry and taxi distance (specialist, `pipeline.prepare`).
  - `pipeline/context.py`: runways crossed on the stand → runway shortest path.
- **Base engine**: `engines/final_model/src/airport_map.py` downloads `data/mapa/<ICAO>.dat` through the Gateway API and
  computes `map_dist`, `map_dmin` and `map_extra`. When the API is down, pass the files with `--base-model-map`.

## EUROCONTROL Airport Corner events

- **What**: the public list of planned and ongoing events at European airports (works, capacity reductions, adverse
  weather, winter operations, …), <https://ext.eurocontrol.int/airport_corner_public/events>.
- **`--taxi-events`** ([data/taxiout_events.csv](../data/taxiout_events.csv)): our compilation of the events of the
  ten airports from that page, with the columns `ICAO`, `Airport Name`, `Evento` (event: original English text,
  sometimes followed by a Spanish annotation in parentheses), `Start Date` and `End Date` (UTC,
  `dd/mm/yyyy[ HH:MM]`; an end date without time covers the whole day). The file is kept exactly as used in the
  reference run; the winter-operations flag also matches the Spanish words of the annotations.
- **Use**: `src/feature_engineering/events.py` (specialist and `pipeline.prepare`). Only weather events are kept, the
  only type with a measurable effect in 2025; `TBD` dates and `[On Hold / Cancelled]` events are dropped. Each
  departure gets `event_adverse_weather` (takeoff inside a weather event of its airport) and `event_winter_ops`
  (the event is about snow, freezing or de-icing).

## Our own tables

### Aircraft type table (`--aircraft-specs`)

`data/unique_aircraft_types.xlsx` is a derived table we compiled from public aircraft references, with additional
approximate values; it is not an independently published public dataset.

- **How it was built**: we extracted the unique ICAO aircraft type codes from the training and ranking files and
  combined them with `data/aircraft_specifications.xlsx`. The resulting workbook preserves all 243 rows of the
  specifications table and its 224 available maximum takeoff weight (MTOW) values. It adds 14 MTOW values marked
  as approximate, using published type information or values copied from similar types; five rows remain without
  MTOW. The resulting table therefore contains 238 MTOW values.
- **Traceability**: the specifications workbook's `Fuentes` (sources) sheet provides source links for its 224 MTOW
  values, and `Guia` (guide) explains their interpretation. The derived workbook's `Trazabilidad` (traceability)
  sheet distinguishes original values from approximations and describes the additions, but does not provide
  specific source links for them. Its `Notas` (notes) sheet reports 13 added MTOW values; the actual data and
  traceability sheets contain 14.
- **Interpretation**: original MTOW values refer to the configuration identified by the source, not necessarily
  the highest certified MTOW across all variants of an ICAO type or the value for a particular aircraft.
  The added approximate values should not be treated as verified aircraft-specific specifications.
- **Columns read**: `AIRCRAFT_TYPE_flt` (ICAO type code, the key) and `Max Take-Off Weight (kg)`.
- **Use**: `src/feature_engineering/aircraft.py` maps each departure's type (movement type, else flight-plan type) to
  `mtow`; `src/feature_engineering/aircraft_backfill.py` recomputes `mtow` for rows whose type was filled from the
  flight list. The specialist and `pipeline.prepare` use `mtow` as a feature.

## Attribution

- ADS-B data © adsb.lol contributors, Open Database License (ODbL) 1.0.
- OPDI and performance data © EUROCONTROL, Open Performance Data Initiative.
- Airport events from the EUROCONTROL Airport Corner.
- METAR from the Iowa Environmental Mesonet, Iowa State University.
- Airport data from the X-Plane Scenery Gateway, GPL v2 or later.
