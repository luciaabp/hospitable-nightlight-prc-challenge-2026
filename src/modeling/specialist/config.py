"""Constants shared by features, training, validation and submission building."""

from pathlib import Path

ID = "MVT_ID_mvt"
CLOCK = "MVT_TIME_UTC_mvt"
TARGET = "TAXITIME_SEC_mvt"
DELAY = "takeoff_minus_SCHED_TIME_UTC_mvt"

FCO = "LIRF"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"]

VALIDATION_MONTHS = ["2025-07", "2025-09", "2025-11", "2025-12"]
SELECTION_MONTHS = VALIDATION_MONTHS[:2]
CONFIRMATION_MONTHS = VALIDATION_MONTHS[2:]

SURFACE_DIR = Path("local_data/interim_surface_ensemble")
NM_DIR = Path("local_data/interim_nm")
RAW_DIR = Path("local_data/raw")

# |BLOCK - SCHED| within this tolerance marks a block time copied from the schedule.
IMPUTED_TOLERANCE_SEC = 60

# Month features let a handful of boundary rows (local month = next month) define the
# whole outer month in walk-forward folds: +153 s mean SHAP shift on LFPG July 2025.
MONTH_FEATURES = ["takeoff_month", "scheduled_month"]

# Block C, read from interim_nm. Label-derived columns (calzos_sin_medir,
# TAXI_TARGET_SEC, BLOCK) are never read.
COMMON = [
    "ADES_mvt", "airline", "market_segment", "airline_prefix_source", "flight_prefix_repeated",
    "flight_id_len", "callsign_all_alpha", "aobt_after_takeoff", "sin_plan_vuelo", "deicing_risk",
    "event_adverse_weather", "event_winter_ops", "visibility_band", "freezing_or_snow",
    "aircraft_type", "aircraft_type_mismatch", "wake_category", "mtow", "market_segment_imputado",
    "wake_category_imputado", "eobt_imputado", "recent_taxi_proxy_mean_60m",
    "recent_taxi_proxy_count_60m", "departures_taxiing", "sin_local_hour", "cos_local_hour",
]

SEEDS = [1, 42]
MAX_TREES = 4000
INNER_MONTHS = 2
EARLY_STOPPING_ROUNDS = 200
REFIT_TREE_FACTOR = 1.1
PARAMS = dict(depth=8, learning_rate=0.05, l2_leaf_reg=10, border_count=254,
              max_ctr_complexity=2, task_type="GPU", thread_count=8,
              allow_writing_files=False, verbose=False)
