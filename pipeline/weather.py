"""Extended METAR per departure (optional), for the regressors of the groups where it helps (``wx_groups``).

    python -m pipeline.weather prepare [--source local_data/pipeline_prep_20260925/interim_fix]

Takes the ``WX_FEATURES`` (METAR at or before takeoff and scheduled time) from the ``interim`` built by
``python -m pipeline.prepare``. Output: ``features.parquet`` (``ID`` + ``WX_FEATURES``) and ``manifest.json``.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import ID, ROOT, WX_FEATURES

log = logging.getLogger(__name__)
SOURCE = ROOT / "local_data" / "pipeline_prep_20260925" / "interim_fix"
OUTPUT = ROOT / "local_data" / "wx_features"


def prepare(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    from .data import sha256
    files = sorted(source.glob("training_*.parquet")) + [source / "ranking.parquet"]
    table = pd.concat([pd.read_parquet(p, columns=[ID] + WX_FEATURES) for p in files], ignore_index=True)
    if table[ID].isna().any() or table[ID].duplicated().any():
        raise ValueError("IDs missing or repeated in the extended METAR")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "features.parquet"
    table.to_parquet(target, index=False)
    manifest = {"output": str(target), "output_sha256": sha256(target), "rows": int(len(table)),
                "coverage": {c: round(float(table[c].notna().mean()), 4) for c in WX_FEATURES},
                "inputs": {str(p): sha256(p) for p in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%s: %d departures", target, len(table))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.weather", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(prepare(args.source, args.output))


if __name__ == "__main__":
    main()
