# PRC Data Challenge 2026 — taxi-out time

## The challenge

The PRC Data Challenge is the open data-science competition of EUROCONTROL's Performance Review Commission (PRC). The
2026 edition asks for the **taxi-out time** of departures: the seconds between off-block (leaving the stand) and
takeoff, `TAXITIME_SEC_mvt`, for every movement `MVT_ID_mvt` listed in the submission template.

- **Airports**: ten European hubs — EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM.
- **Data**: movement records (schedule, block and runway times, stand, runway, aircraft type, airline) joined with
  Network Manager flight data (LOBT, EOBT, AOBT, …) where available, plus METAR. The organizers provide twelve
  monthly training files for 2025 with labels and a ranking set without labels; teams may add open data. The ranking
  set first covered January and July 2026; the final dataset (`final_ranking.parquet` / `final_submitting.parquet`)
  adds February and June 2026.
- **Metric**: root-mean-square error in seconds on the ranking set (leaderboard score; lower is better).

## This repository

It rebuilds our final submission (**v87**) from the organizer's files and open data, with one command.
Training uses 2025 only; no label of a predicted month is used as a feature.

- **Final submission**: [output/hospitable-nightlight_final.parquet](output/hospitable-nightlight_final.parquet) —
  recipe v87 on the final dataset (January, February, June and July 2026; 670,790 departures,
  sha256 `166741b4…`).
- [output/submission_v87.parquet](output/submission_v87.parquet) — the same recipe on the original January/July
  ranking set (344,841 departures, sha256 `cd501a95…`), kept as the historical reference. Predictions for shared
  IDs are not identical, because the features are rebuilt from the larger ranking set.

- Best measured leaderboard score: **v86 = 231.6335** (sha256 `8324c6bc…`).
- **v87** = v86 with the stand-quality corrector (`components/quality`). Its local holdout gain is −5.6 M s²
  (≈ −0.05 s projected).
- Open data used: adsb.lol ADS-B traces, EUROCONTROL OPDI flight events and flight list, EUROCONTROL daily
  performance series, EUROCONTROL Airport Corner events, IEM METAR, X-Plane airport layouts — see [docs/open_data.md](docs/open_data.md).

## Getting the code

The source code is published at <https://github.com/Indra-ATM/hospitable-nightlight>:

```bash
git clone https://github.com/Indra-ATM/hospitable-nightlight.git
cd hospitable-nightlight
```

Clone it with Git (do not copy the files by hand): `python reproduce.py check` compares every source file byte for
byte with `config/source_manifest.json`, and `.gitattributes` keeps line endings as LF on every platform.

## How the prediction is built

```
 organizer files ─┬─ specialist v62 (src/, CatBoost experts) ──────────────────────────────┐
 + open data      ├─ base model (engines/final_model) × 34 trainings                        │
                  │    base_reference ─ base ADS-B events ──▶ out-of-block base ─▶ LIRF rotation
                  │    D2sub ── + relaxed ADS-B segments                                     │
                  │    FG4sub, FG4sub_s1…s31 ── own off-block detector events, 32 seeds      │
                  ├─ context model v70 (pipeline/) ──────────────────────────────────────────┤
                  └─ M2 (4 seeds) ─▶ M2r ─▶ stand-quality corrector                          │
                                                                                             ▼
                v82 = blend(base models, v70, LIRF rules, M2, v62)  ─▶  v86 = v82 + M2r  ─▶  v87 = v86 + quality
```

1. **Base model** — All variants build on this engine, each using its own ADS-B events: base_reference (engine-provided events), D2sub (with relaxed ground segments), and the 32 FG4 seeds (events from our own off-block detector; BASE_GAP_FEATURES=1 enables additional hidden-taxi features).
2. **Specialist v62** — CatBoost experts for LIRF, flights without NM and FCO (`src/`, run by `specialist.py`).
3. **Context model v70** — anchored CatBoost models on context tables (`pipeline/`, frozen config `config/context.json`).
4. **Correctors** on own-ADS-B departures of the nine airports other than LIRF: M2 (`components/m2`), M2r with
   day-regime and stand features (`components/m2r`) and the stand-quality LightGBM (`components/quality`).
5. **LIRF rules** — aircraft rotation increment and the destination-arrival rule (`components/lirf`).
6. **Blends** — `components/blend/v82.py`, `components/blend/v86.py` and `generate.py` (v87).

Every weight and recipe value is in `config/recipe.json`, with its origin.

## Reproducing from scratch

### 1. Environment

- Python 3.12 on Linux, `setpriv` and `curl` (both standard on Ubuntu).
- The reference run used 64 cores, 247 GB of RAM (five base trainings in parallel at ~17 GB each) and tens of GB of
  disk under the run directory.
- A CUDA GPU is needed only to **retrain** the specialist; without one, pass a frozen bundle with
  `--specialist-models` (prepare + predict only).
- Network access for the open-data downloads. The organizer files are not included and must be obtained from the
  challenge; the two hand-compiled tables are in [data/](data/).

Create an isolated environment and install the pinned runtime dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pins are exact versions, but not a platform-specific lockfile; GPU drivers, system libraries and the external
datasets remain environment inputs.

### 2. Inputs

| argument | content | how to get it |
|---|---|---|
| `--raw` | `training_2025-*.parquet` (12), `ranking.parquet`, `submitting.parquet`, `metar.parquet` | organizer files |
| `--ranking`, `--template` | the final dataset's `final_ranking.parquet` / `final_submitting.parquet` (default: `RAW/ranking.parquet` / `RAW/submitting.parquet`) | organizer files |
| `--metar-csv` | METAR of the ten airports, IEM ASOS CSV | Iowa Environmental Mesonet ASOS download |
| `--aircraft-specs` | aircraft type table | [data/unique_aircraft_types.xlsx](data/unique_aircraft_types.xlsx): our own table, MTOW of every type in the data looked up by hand (see [docs/open_data.md](docs/open_data.md)) |
| `--taxi-events` | airport events table | [data/taxiout_events.csv](data/taxiout_events.csv): compiled from the EUROCONTROL Airport Corner public events |
| `--xplane` | X-Plane `apt.dat` files of the ten airports | X-Plane Scenery Gateway |
| `--opdi-events` | EUROCONTROL OPDI flight events | `python scripts/download_opdi_events.py --months 2025-01 … --download` |
| `--adsblol-store` | adsb.lol `globe_history` archives (read only) | `python scripts/download_adsblol_days.py --start 2025-01-01 --end 2026-07-31` |
| `--base-model-map` | optional: folder of X-Plane `<ICAO>.dat` for the base model map | when the Gateway API is down |
| `--specialist-models` | optional: frozen specialist bundle | output of a previous specialist training |

Days missing from `--adsblol-store` are downloaded during the run.

### 3. Run

```bash
python reproduce.py check     # the sources match config/source_manifest.json and every recipe value is confirmed
python reproduce.py plan      # every stage, its dependencies and outputs
python reproduce.py run --run-dir RUN \
  --raw /path/raw --ranking /path/raw/final_ranking.parquet --template /path/raw/final_submitting.parquet \
  --metar-csv metar_iem.csv --aircraft-specs data/unique_aircraft_types.xlsx --taxi-events data/taxiout_events.csv \
  --xplane /path/xplane --opdi-events /path/eurocontrol/flight_events --adsblol-store /path/adsblol_history \
  [--specialist-models /path/final_models] [--base-model-map /path/xplane_maps] \
  --jobs 5 --procs 16 --threads 12 --retry-interrupted
python reproduce.py status --run-dir RUN
```

The submission is `RUN/submission_v87.parquet` (report: `RUN/submission_v87.json`). On the final dataset the
reference run (with `--specialist-models`) produced `output/hospitable-nightlight_final.parquet` (sha256
`166741b4…`). Omit `--ranking`/`--template` to rebuild the historical January/July submission.

- A run copies the sources into `RUN/code` and links the inputs there; editing the checkout never changes a run in
  progress. The inputs are hashed and checked on every invocation.
- Rerunning the same command resumes: completed stages (`RUN/stages/<stage>.json`) are skipped, each stage logs to
  `RUN/logs/<stage>.log`.
- `bash scripts/run_unattended.sh RUN [run options]` resumes a run without supervision (one supervisor per run).
- With the original January/July data, `--verify-reference` requires the historical v86/v87 hashes.

See [docs/reproduction.md](docs/reproduction.md) for every stage.

### Final assembly only

From already computed component predictions:

```bash
python generate.py --template submitting.parquet --ranking ranking.parquet --baseline v86.parquet \
  --original pred_ranking_c.parquet --quality pred_ranking_quality_s11_2500.parquet --events events.parquet \
  --output output/submission.parquet [--verify-reference]
```

`v87 = max(v86 + 0.978 · 0.37 · 0.5 · (quality_lgb − original_lgb), 0)` on own-ADS-B departures outside LIRF; every
other row keeps v86.

## Repository layout

| path | content |
|---|---|
| `reproduce.py` | single entry point: `check`, `plan`, `run`, `status`, `manifest` |
| `chain/` | stage engine (`runner.py`) and the stage list (`stages.py`) |
| `common.py` | paths, column names and helpers shared by every step |
| `dataset.py` | dataset coverage and ID checks (stage `inspect`) |
| `specialist.py` | isolated specialist v62 run: prepare, train, predict |
| `generate.py` | final assembly v86 → v87 |
| `src/` | cleaning, departure features and the specialist models |
| `pipeline/` | context model v70: feature tables, training and prediction |
| `engines/final_model/` | final-only base engine and self-contained recipe |
| `components/base_model/` | base-model setup, adsb.lol cuts, relaxed events, variants (D2sub, FG4) |
| `components/offblock/` | own off-block detector: ob2 segments, ob4 candidates, rule v4 |
| `components/lirf/` | tail rotation, LIRF rotation increment, LIRF destination rule |
| `components/m2/`, `components/m2r/`, `components/quality/` | correctors on own-ADS-B departures |
| `components/blend/` | blends: v70 + rotation, candidates v82 and v86 |
| `config/` | `recipe.json` (frozen recipe), `context.json` (v70), `specialist/`, `source_manifest.json` |
| `scripts/` | open-data downloads, unattended runner |
| `data/` | hand-compiled inputs: aircraft type table (and its source workbook), airport events |
| `docs/` | open-data sources and the stage-by-stage reproduction guide |
| `output/` | final submission, historical January/July v87 submission and its report |

After editing any source, run `python reproduce.py manifest` so that `check` and new runs accept the tree. Every
commit must leave `python reproduce.py check` clean (exit code 0, no `source_problems`): a new run refuses to start
otherwise. Generate the manifest from a clean checkout, because untracked files under the source folders also enter it.

## Legacy terminology

Code, comments and messages are in English. A few Spanish and Portuguese names remain on purpose, because they are
column or feature names of the trained models, keys of the frozen recipe, command-line flags, or values of the input
files; renaming them would change the model inputs or the interfaces of the reference run.

| term | English meaning | where |
|---|---|---|
| `imputado` | imputed | NM-history flag features (`*_imputado`) |
| `calzos_sin_medir` | unmeasured blocks | data-cleaning flag |
| `sin_plan_vuelo` | no flight plan | data-cleaning flag and specialist feature |
| `ventana` | window | `config/recipe.json`, `components/blend/v82.py --ventana` |
| `frío_húmedo`, `frío_seco`, `templado` | cold and wet, cold and dry, mild | values of the `deicing_risk` feature |
| `Evento` | event | column of `data/taxiout_events.csv` |
| `mes`, `dia`, `cia`, `ruta` | month, day, airline, route | intermediate columns of the LIRF and base-model components |
| `chao`, `fonte`, `tipo`, `reg` | on ground, source, aircraft type, registration | columns of the adsb.lol cuts (`adsb/cut/`) shared with the base engine |
| `externo/`, `mapa/`, `distancias.parquet` | external data, map, distances | base-engine data folders |
| `pista`, `copias`, `peso`, `fila`, `mudou`, `mesmo`, `viz`, … | runway, copies, weight, queue, changed, same, neighbour | feature names inside `engines/final_model` |


## Licensing and data handling

The code is licensed under the GNU General Public License v3.0 or later; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

External data have separate provider terms and attribution requirements; see [docs/open_data.md](docs/open_data.md).


## Limits

- January and July 2025 were reused for model selection, so holdout gains are optimistic.
- The rules specific to the ranking months (v70 anchors, LIRF rotation, flight history, LIRF blend) were selected on
  January/July 2026; for the final dataset, January rules are applied to February and July rules to June with the
  same weights, thresholds and training settings, without validation on the added months.
- `output/` holds reference outputs only, never inputs for a run.
