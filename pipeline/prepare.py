"""Raw data -> ``interim_nm`` in a new folder, with the ``src`` code.

    python -m pipeline.prepare --output-dir local_data/pipeline_prep [--compare-to local_data/interim_nm]

Cleaning without dropping or capping, departure features per monthly file, the ranking with the same steps and an ID
check against the template, then the time-respecting NM imputation. Never writes into an existing folder;
``--compare-to`` checks the columns the pipeline reads against another preparation.
"""
import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning.config import CleaningConfig
from src.data_cleaning.pipeline import clean, run as run_cleaning
from src.evaluation.prepare_ranking import check_ids
from src.feature_engineering import featuring, nm_history

from .config import ID, ROOT
from .data import RANKING_COLUMNS, TRAINING_COLUMNS

log = logging.getLogger(__name__)
CHALLENGE = ROOT / "data" / "projects" / "prc_data_challenge"


def prepare(raw_dir: Path, output_dir: Path, metar: Path, aircraft_specs: Path, events: Path,
            xplane_dir: Path | None = None) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists: the preparation overwrites nothing")
    processed, fix, nm = output_dir / "processed_nodrop", output_dir / "interim_fix", output_dir / "interim_nm"
    cleaning = replace(CleaningConfig(), raw_dir=raw_dir, output_dir=processed,
                       drop_unrecoverable=False, cap_unmeasured_block=False)
    processed.mkdir(parents=True)
    run_cleaning(cleaning)
    fix.mkdir()
    for path in sorted(processed.glob("training_*.parquet")):
        departures = featuring.build_features(pd.read_parquet(path), metar, aircraft_specs, events, xplane_dir)
        departures.to_parquet(fix / path.name, index=False)
        log.info("%s: %d departures", path.name, len(departures))
    raw = pd.read_parquet(raw_dir / "ranking.parquet")
    cleaned, _ = clean(raw, cleaning)
    if len(cleaned) != len(raw):
        raise RuntimeError("Cleaning changed the number of ranking rows")
    departures = featuring.build_features(cleaned, metar, aircraft_specs, events, xplane_dir)
    ids = check_ids(departures, pd.read_parquet(raw_dir / "submitting.parquet"))
    if not ids["ok"]:
        raise RuntimeError(f"Ranking and template do not match: {ids}")
    departures.to_parquet(fix / "ranking.parquet", index=False)
    nm_history.main(["--input-dir", str(fix), "--output-dir", str(nm)])
    return nm


def compare(new_dir: Path, reference_dir: Path) -> dict:
    """File-by-file equality of the columns the pipeline reads."""
    report = {}
    for reference in sorted(reference_dir.glob("training_*.parquet")) + [reference_dir / "ranking.parquet"]:
        columns = RANKING_COLUMNS if reference.name == "ranking.parquet" else TRAINING_COLUMNS
        old = pd.read_parquet(reference, columns=columns).sort_values(ID).reset_index(drop=True)
        new = pd.read_parquet(new_dir / reference.name, columns=columns).sort_values(ID).reset_index(drop=True)
        if len(old) != len(new) or not np.array_equal(old[ID].to_numpy(), new[ID].to_numpy()):
            report[reference.name] = {"status": "different rows", "rows": [len(old), len(new)]}
            continue
        differing = {}
        for column in columns:
            a, b = old[column], new[column]
            same = (a == b) | (a.isna() & b.isna())
            if not same.all():
                differing[column] = int((~same).sum())
        report[reference.name] = {"status": "identical" if not differing else "different", "differing": differing}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=CHALLENGE / "raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metar", type=Path, default=CHALLENGE / "raw" / "metar_iem.csv")
    parser.add_argument("--aircraft-specs", type=Path,
                        default=ROOT / "data" / "unique_aircraft_types.xlsx")
    parser.add_argument("--events", type=Path, default=ROOT / "data" / "taxiout_events.csv")
    parser.add_argument("--xplane-dir", type=Path, help="X-Plane apt.dat folder: adds the stand geometry features")
    parser.add_argument("--compare-to", type=Path, help="Reference interim_nm folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    nm = prepare(args.raw_dir, args.output_dir, args.metar, args.aircraft_specs, args.events, args.xplane_dir)
    if args.compare_to:
        report = compare(nm, args.compare_to)
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
