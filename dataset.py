"""Dataset coverage and validation, independent of the historical ranking months."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd


ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"


def movement_files(raw: Path) -> list[Path]:
    training = sorted(raw.glob("training_*.parquet"))
    if not training:
        raise FileNotFoundError(f"No training files in {raw}")
    ranking = raw / "ranking.parquet"
    if not ranking.is_file():
        raise FileNotFoundError(ranking)
    return [*training, ranking]


@lru_cache(maxsize=128)
def _coverage(path: str, size: int, modified: int):
    times = pd.to_datetime(pd.read_parquet(path, columns=[TIME])[TIME], utc=True, errors="raise").dropna()
    if times.empty:
        raise ValueError(f"No movement timestamps in {path}")
    return times.min(), times.max(), tuple(sorted(times.dt.strftime("%Y-%m").unique()))


def file_coverage(path: Path):
    stat = path.stat()
    return _coverage(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def months(raw: Path) -> list[str]:
    return sorted({month for path in movement_files(raw) for month in file_coverage(path)[2]})


def time_bounds(raw: Path):
    coverage = [file_coverage(path) for path in movement_files(raw)]
    return min(item[0] for item in coverage), max(item[1] for item in coverage)


def inspect(raw: Path) -> dict:
    files = movement_files(raw)
    unsupported = [path.name for path in files[:-1] if not path.name.startswith("training_2025-")]
    if unsupported:
        raise ValueError(f"The frozen recipe trains on 2025; unsupported training files: {unsupported}")
    template = pd.read_parquet(raw / "submitting.parquet", columns=[ID])
    ranking = pd.read_parquet(files[-1], columns=[ID, TIME, "PHASE_mvt", "ADEP_mvt"])
    for name, frame in (("template", template), ("ranking", ranking)):
        if frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"Null or duplicate IDs in {name}")
    if template.empty:
        raise ValueError("Empty template")
    departures = ranking.loc[ranking.PHASE_mvt.eq("DEP")].set_index(ID)
    missing = template.loc[~template[ID].isin(departures.index), ID]
    if len(missing):
        raise ValueError(f"Template has {len(missing)} IDs without a ranking departure")
    selected = departures.loc[template[ID]]
    times = pd.to_datetime(selected[TIME], utc=True, errors="raise")
    if times.isna().any() or selected.ADEP_mvt.isna().any():
        raise ValueError("Template departures need movement timestamps and airports")
    for path in files[:-1]:
        training_ids = pd.read_parquet(path, columns=[ID])[ID]
        if template[ID].isin(training_ids).any():
            raise ValueError(f"Prediction IDs overlap training: {path.name}")
    return {
        "training_files": [path.name for path in files[:-1]],
        "ranking_rows": len(ranking), "submission_rows": len(template),
        "ranking_months": sorted(times.dt.strftime("%Y-%m").unique().tolist()),
        "external_data_months": months(raw),
        "submission_by_month": times.dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.raw), indent=2))


if __name__ == "__main__":
    main()