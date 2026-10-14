"""Cleaning configuration: paths, column names and thresholds."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- columns of the original dataset ----------------------------------------
PHASE = "PHASE_mvt"
FLIGHT_ID = "FLIGHT_ID_mvt"
BLOCK_TIME = "BLOCK_TIME_UTC_mvt"
SCHED_TIME = "SCHED_TIME_UTC_mvt"
TAXI_TIME = "TAXITIME_SEC_mvt"

DEPARTURE = "DEP"

# --- columns added by the cleaning -------------------------------------------
FLAG_NO_FLIGHT_PLAN = "sin_plan_vuelo"      # no Network Manager match (observable in the ranking)
FLAG_BLOCK_NOT_MEASURED = "calzos_sin_medir"  # block time = scheduled time (NOT observable in the ranking)
TARGET = "TAXI_TARGET_SEC"                  # clean training label (departures only)

# internal column with each row's source file; not saved
SOURCE_FILE = "_source_file"


@dataclass(frozen=True)
class CleaningConfig:
    raw_dir: Path = PROJECT_ROOT / "local_data" / "raw"
    output_dir: Path = PROJECT_ROOT / "local_data" / "processed"
    file_pattern: str = "training_*.parquet"
    report_name: str = "cleaning_report.json"

    # if |block - scheduled| <= this, the block time was not measured
    block_sched_tolerance_sec: float = 5.0
    # cap for rows with an unmeasured block time: this quantile of the sound rows
    cap_quantile: float = 0.999
    # smallest allowed label
    min_target_sec: float = 0.0

    # stage 2: drop departures without flight plan and with an unmeasured block time
    drop_unrecoverable: bool = True
    # stage 3: cap the label of departures with an unmeasured block time
    cap_unmeasured_block: bool = True
