"""Fill the Network Manager block of flights without NM from the history of the same flight number.

Only months before the one being filled are used (July from January-June, the 2026 ranking from all 2025).

    python -m src.feature_engineering.nm_history --input-dir local_data/interim_fix --output-dir local_data/interim_nm
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPEATED_PREFIX = re.compile(r"^([A-Z])([A-Z]{2})\2(.*)$")
# flags of what was filled: the model must be able to distrust an imputed value
FEATURES = ["market_segment_imputado", "wake_category_imputado", "eobt_imputado"]


def flight_key(flight: pd.Series) -> pd.Series:
    """Flight number without the repeated prefix: ITYTY672 -> ITY672."""
    def clean(value):
        if not isinstance(value, str):
            return None
        m = REPEATED_PREFIX.match(value)
        return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else value
    return flight.map(clean)


def build_history(frames: list[pd.DataFrame]) -> dict:
    """Usual value per flight number, from the rows that did match NM."""
    if not frames:
        return {}
    hist = pd.concat(frames, ignore_index=True)
    hist = hist[hist["MARKET_SEGMENT_flt"].notna() | hist["WK_TBL_CAT_flt"].notna() | hist["EOBT_1_flt"].notna()]
    hist = hist[hist["_flight_key"].notna()]
    if hist.empty:
        return {}
    mode = lambda s: s.value_counts().idxmax() if s.notna().any() else None  # noqa: E731
    time_offset = (hist["EOBT_1_flt"] - hist["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return {
        "market_segment": hist.groupby("_flight_key")["MARKET_SEGMENT_flt"].agg(mode).dropna().to_dict(),
        "wake_category": hist.groupby("_flight_key")["WK_TBL_CAT_flt"].agg(mode).dropna().to_dict(),
        "eobt_offset_sec": time_offset.groupby(hist["_flight_key"]).median().dropna().to_dict(),
    }


def apply_history(df: pd.DataFrame, history: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    flight = df["_flight_key"]
    report = {}

    for column, key in [("market_segment", "market_segment"), ("wake_category", "wake_category")]:
        missing = df[column].isna()
        values = flight.map(history.get(key, {}))
        df[f"{column}_imputado"] = missing & values.notna()
        df[column] = df[column].where(~missing, values).astype("string")
        report[column] = int(df[f"{column}_imputado"].sum())

    # EOBT is rebuilt as schedule + the flight's usual offset
    missing = df["takeoff_minus_EOBT_1_flt"].isna()
    offset = flight.map(history.get("eobt_offset_sec", {}))
    estimated = df["takeoff_minus_SCHED_TIME_UTC_mvt"] - offset
    df["eobt_imputado"] = missing & estimated.notna()
    df["takeoff_minus_EOBT_1_flt"] = df["takeoff_minus_EOBT_1_flt"].where(~missing, estimated)
    report["takeoff_minus_EOBT_1_flt"] = int(df["eobt_imputado"].sum())

    report["rows_without_nm"] = int(df["no_flight_plan"].astype(bool).sum())
    return df, report


def month_of(path: Path) -> str:
    """Month of the file, for sorting: the ranking goes last (2026)."""
    m = re.search(r"(\d{4})-(\d{2})-\d{2}", path.name)
    return f"{m.group(1)}-{m.group(2)}" if m else "9999-99"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fill the NM block from the history of the same flight.")
    parser.add_argument("--input-dir", type=Path, default=Path("local_data/interim_fix"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_data/interim_nm"))
    parser.add_argument("--ranking-name", default="ranking.parquet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    training = sorted(args.input_dir.glob("training_*.parquet"), key=month_of)
    ranking = args.input_dir / args.ranking_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hist_cols = ["_flight_key", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"]
    previous_frames: list[pd.DataFrame] = []
    for path in training:
        df = pd.read_parquet(path)
        df["_flight_key"] = flight_key(df["FLIGHT_mvt"].astype("string"))
        df, report = apply_history(df, build_history(previous_frames))
        log.info("%s: %s rows without NM | filled %s", path.name, f"{report['rows_without_nm']:,}",
                 {k: v for k, v in report.items() if k != "rows_without_nm"})
        df.drop(columns="_flight_key").to_parquet(args.output_dir / path.name, index=False)
        previous_frames.append(df[hist_cols])

    if ranking.exists():
        df = pd.read_parquet(ranking)
        df["_flight_key"] = flight_key(df["FLIGHT_mvt"].astype("string"))
        # the ranking is 2026: filled with all 2025
        df, report = apply_history(df, build_history(previous_frames))
        log.info("%s: %s rows without NM | filled %s", ranking.name, f"{report['rows_without_nm']:,}",
                 {k: v for k, v in report.items() if k != "rows_without_nm"})
        df.drop(columns="_flight_key").to_parquet(args.output_dir / ranking.name, index=False)


if __name__ == "__main__":
    main()
