"""Build target-independent weather and surface ensemble matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import FEATURE_CODE_SHA256
from .base import CLOCK, ID, INPUTS, TARGET, build_features, departures, valid_targets
from .surface import CONTEXT_COLUMNS, SurfaceContext
from .weather import prepare_metar, weather_features

def source_signature(raw_dir: Path) -> dict:
    training = sorted(raw_dir.glob("training*.parquet"))
    if not training:
        raise FileNotFoundError(f"No training files in {raw_dir}")
    files = [*training, raw_dir / "ranking.parquet", raw_dir / "metar.parquet", raw_dir / "submitting.parquet"]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "code": FEATURE_CODE_SHA256,
        "files": [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns] for path in files],
    }


def prepare(raw_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    signature = source_signature(raw_dir)
    training_files = sorted(raw_dir.glob("training*.parquet"))
    for stale in [*output.glob("training_*.parquet"), output / "ranking.parquet"]:
        stale.unlink(missing_ok=True)
    metar = prepare_metar(pd.read_parquet(raw_dir / "metar.parquet"))
    columns = list(dict.fromkeys([*INPUTS, *CONTEXT_COLUMNS]))
    audits = []
    feature_columns = None
    datasets = [("training", training_files), ("ranking", [raw_dir / "ranking.parquet"])]
    for name, paths in datasets:
        raw = pd.concat(
            [
                pd.read_parquet(path, columns=columns + ([TARGET] if name == "training" else []))
                .assign(_source_file=path.name)
                for path in paths
            ],
            ignore_index=True,
        )
        context = SurfaceContext(raw[CONTEXT_COLUMNS])
        movements = departures(raw)
        if name == "training":
            movements = movements.loc[
                valid_targets(movements) & movements[CLOCK].notna()
            ].sort_values(CLOCK).reset_index(drop=True)
        base = build_features(movements).drop(columns="FLIGHT_mvt")
        print(name, "base features", len(movements), flush=True)
        features = pd.concat(
            [base, weather_features(movements, metar), context.features(movements)], axis=1
        )
        if not features.index.equals(movements.index) or features.columns.duplicated().any():
            raise ValueError("Feature alignment failure")
        if feature_columns is None:
            feature_columns = list(features.columns)
        elif list(features.columns) != feature_columns:
            raise ValueError("Training and ranking feature schemas differ")
        metadata = movements[[ID, CLOCK] + ([TARGET] if name == "training" else [])].copy()
        metadata["missing_nm"] = movements["AOBT_3_flt"].isna().to_numpy()
        dataset = pd.concat([metadata, features], axis=1)
        source_files = movements["_source_file"]
        written = []
        for path in paths:
            part = dataset.loc[source_files.eq(path.name)].reset_index(drop=True)
            destination = output / (path.name if name == "training" else "ranking.parquet")
            part.to_parquet(destination, index=False)
            written.append({"file": destination.name, "rows": len(part)})
        for airport, indices in movements.groupby("ADEP_mvt").groups.items():
            surface = features.loc[indices, "traffic_offblock_surface_active_proxy"]
            audits.append({
                "dataset": name,
                "airport": airport,
                "n": len(indices),
                "surface_median": float(surface.median()),
                "surface_p95": float(surface.quantile(0.95)),
                "surface_max": float(surface.max()),
                "reference_fallback": float(features.loc[indices, "traffic_reference_fallback"].mean()),
                "context_missing": float(features.loc[indices, "traffic_offblock_context_missing"].mean()),
            })
        signature[name] = written
        print(name, "saved", len(features.columns), "features in", len(written), "files; context", len(raw), "movements", flush=True)
        del raw, context, movements, base, features, metadata, dataset
    pd.DataFrame(audits).to_csv(output / "traffic_audit.csv", index=False)
    signature["layout"] = "monthly training parquets plus ranking.parquet"
    signature["feature_columns"] = feature_columns
    (output / "manifest.json").write_text(json.dumps(signature, indent=2) + "\n")
    return signature


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("local_data/interim_surface_ensemble"))
    args = parser.parse_args(argv)
    prepare(args.raw_dir, args.output)


if __name__ == "__main__":
    main()