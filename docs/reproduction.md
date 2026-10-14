# Reproducing v87 from raw data: the stages

`python reproduce.py plan` prints this list from `chain/stages.py`. Every command runs in `RUN/code` with
`PYTHONPATH=RUN/code`; paths below are relative to it. Recipe values come from `config/recipe.json` → `chain`.

## Specialist

| stage | command | output |
|---|---|---|
| `inspect` | `python dataset.py --raw …` | `RUN/dataset.json` |
| `specialist` | `python specialist.py all …` (GPU) or `prepare` + `predict --models-dir` | `RUN/specialist/submission_specialist.parquet`, linked as `submission_v62.parquet` |

## Base model and ADS-B

| stage | command | output |
|---|---|---|
| `base_reference` | `python -m components.base_model.base_model setup`, then the engine's `src/external_data.py download`, `src/weather_rotation.py download`, `src/airport_map.py` | `variants/base_reference` with its open data |
| `adsb_cut` | `python -m components.base_model.base_model cut` (resumable per day) | `adsb/cut/<day>.parquet` |
| `base_events` | engine `src/adsb_events.py`, `src/dataset_cache.py` | `adsb/events.parquet`, data splits |
| `relaxed_events` | `python -m components.base_model.adsb_relaxed` | `adsb/relaxed_events.parquet` |
| `d2sub` | `python -m components.base_model.variants d2sub --max-gap 1200` | `variants/D2sub` |
| `offblock` | `python -m components.offblock.ob2`, `.ob4`, `.ob_v4` | `components/offblock/ob2_all.parquet`, `ob4_fixed.parquet`, `ob_v4.parquet` |
| `fg4sub` | `python -m components.base_model.variants fg4 FG4sub --max-gap 1200` | `variants/FG4sub` |
| `base_reference_v37`, `base_d2sub`, `base_fg4_s0` … `s31` | engine `src/training.py submit 37` with `TEAM_NAME`, `SEED_SHIFT=k`, `BASE_GAP_FEATURES` (1 only for FG4) | `variants/<variant>/submissions/<team>_v37.parquet` |

The engine's out-of-block prediction cache is keyed by a hash of the base config, of `blocked_training.py` and the engine
modules it imports, and of the data files; `variants.py oof-path` finds it for the LIRF rotation.

## Context model (pipeline/)

| stage | command | output |
|---|---|---|
| `pipeline_prepare` | `python -m pipeline.prepare … --xplane-dir …` | `local_data/pipeline_prep/interim_nm` |
| `features_adsb`, `features_context`, `features_inbound_lirf`, `features_inbound_nine`, `features_weather`, `features_state`, `features_forward` | `python -m pipeline.<table> prepare` | `local_data/<table>_features/` |
| `context_v70` | `python -m pipeline all --run pipeline_20261003_anchor` (config `config/context.json`) | `outputs/pipeline/pipeline_20261003_anchor/submission.parquet` |

## Correctors

| stage | command | output |
|---|---|---|
| `m2_s11` … `m2_s14` | `python -m components.m2.m2 ranking` with `M2_SEED` | `components/m2/pred_ranking[_s<seed>].parquet` |
| `m2r` | `components.m2r.regime_features`, `.stand_features`, then `.m2r ranking` (`M2R_TAG=c`, `M2R_STAND=1`) | `components/m2r/pred_ranking_c.parquet` |
| `quality` | `python -m components.quality.quality --mode ranking --seed 11 --rounds 2500` | `components/quality/pred_ranking_quality_s11_2500.parquet` |

## LIRF rules and blends

| stage | command | output |
|---|---|---|
| `tail_rotation` | `python -m components.lirf.tail_rotation` | `local_data/tail_rotation/` |
| `lirf_rotation` | `variants oof-path variants/base_reference 20260930-130008-e76_base`, then `python -m components.lirf.lirf_rotation --oof …` | `outputs/final/lirf_v73.parquet` |
| `candidate_v82` | `python -m components.blend.v82 --seeds 0 … 31 --v62-adsb 0.022 --v62-lirf 0.182 --m2 0.3 --m2-bag --ventana` | `outputs/final/candidates/v82_m2_03_s32_m2bag_ventana.parquet` |
| `candidate_v86` | `python -m components.blend.v86 --seeds 32 --tag c --wm 0 --wr 0.37` | `outputs/final/candidates/v87_rc_037_s32.parquet` (v86) |
| `assemble_v87` | `python generate.py …` | `RUN/submission_v87.parquet` |

All recipe values were confirmed against the original experiment files on 2026-10-09: M2 seeds 11–14,
`--wm 0 --wr 0.37` for v86, D2sub gap 1200 s (adds exactly the 218,846 historical relaxed events), G4 events equal to
the historical FG4sub events, LIRF OOF from the base engine with the base events.

## Unattended runs

- `--jobs N`: stages whose dependencies are complete run in parallel; the long chain (context pipeline → M2 → M2r →
  quality) starts before the 34 base trainings.
- `--retry-interrupted`: a stage cut short is retried up to `--max-attempts` times, with a growing wait. Resumable
  stages rerun in place; for the others the partial outputs move to `RUN/interrupted/`.
- `scripts/run_unattended.sh` keeps one supervisor per run and writes `RUN/supervisor.json`.
- Commands run under `setpriv --pdeathsig`, so no training survives its runner.
