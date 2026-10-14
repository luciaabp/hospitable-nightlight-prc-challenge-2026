"""Training cleaning, end to end: flags, then stages 2-4, then one cleaned file per source file."""

import json
import logging

import pandas as pd

from .cleaning import cap_unmeasured_block, clip_negative_target, drop_unrecoverable
from .config import DEPARTURE, PHASE, TARGET, TAXI_TIME, CleaningConfig
from .flags import add_quality_flags
from .io import list_training_files, load_training, save_by_source_file

log = logging.getLogger(__name__)


def _target_summary(values: pd.Series) -> dict:
    return {
        "rows": int(values.size),
        "median_min": round(float(values.median()) / 60, 2),
        "max_hours": round(float(values.max()) / 3600, 2),
        "negative": int((values < 0).sum()),
        "pct_over_1h": round(100 * float((values > 3600).mean()), 3),
    }


def clean(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    """Apply the stages in order. Reads and writes no files."""
    before = _target_summary(df.loc[df[PHASE] == DEPARTURE, TAXI_TIME])

    df, flags = add_quality_flags(df, config)
    log.info("1 · flags     : %s", flags)

    df, dropped = drop_unrecoverable(df, config)
    log.info("2 · drop      : %s", dropped)

    df, capped = cap_unmeasured_block(df, config)
    log.info("3 · cap       : %s", capped)

    df, clipped = clip_negative_target(df, config)
    log.info("4 · negative  : %s", clipped)

    after = _target_summary(df.loc[df[PHASE] == DEPARTURE, TARGET])
    report = {
        "options": {
            "drop_unrecoverable": config.drop_unrecoverable,
            "cap_unmeasured_block": config.cap_unmeasured_block,
        },
        "stages": {
            "1_flag_problems": flags,
            "2_drop_unrecoverable": dropped,
            "3_cap_unmeasured_block": capped,
            "4_clip_negative": clipped,
        },
        "departures_before": before,
        "departures_after": after,
        "arrivals": int((df[PHASE] != DEPARTURE).sum()),
    }
    return df, report


def run(config: CleaningConfig) -> dict:
    files = list_training_files(config.raw_dir, config.file_pattern)
    log.info("%d files in %s", len(files), config.raw_dir)

    df, report = clean(load_training(files), config)
    report["files"] = save_by_source_file(df, config.output_dir)
    report_path = config.output_dir / config.report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("report in %s", report_path)
    return report
