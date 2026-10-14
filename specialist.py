"""Specialist v62: prepare, train and predict in an isolated run directory (stage ``specialist``).

    python specialist.py all --run-dir R --raw RAW --metar M.csv --aircraft-specs A.xlsx --events E.csv [--device 0]
    python specialist.py prepare ...; python specialist.py predict --run-dir R --models-dir FROZEN_BUNDLE

The run copies src/ and config/specialist/ into R/engine and runs every module there; training needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dataset import inspect, movement_files
from common import ROOT, sha256

SEEDS = (1, 42)


def signature(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_inputs(run: Path) -> dict:
    manifest = json.loads((run / "run.json").read_text())
    for expected in manifest["inputs"]:
        if signature(Path(expected["path"])) != expected:
            raise ValueError(f"Input changed; create a new run: {expected['path']}")
    return manifest


def execute(run: Path, module: str, *arguments) -> None:
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8")
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=run / "engine", env=environment, check=True)


def initialize(run: Path, raw: Path, metar: Path, aircraft_specs: Path, events: Path) -> dict:
    if run.exists():
        raise FileExistsError(f"Use a new run directory: {run}")
    report = inspect(raw)
    paths = [*movement_files(raw), raw / "submitting.parquet", raw / "metar.parquet",
             metar, aircraft_specs, events]
    manifest = {
        "recipe": "specialist_v62", "dataset": report,
        "inputs": [signature(path) for path in dict.fromkeys(paths)],
        "metar": str(metar.resolve()), "aircraft_specs": str(aircraft_specs.resolve()),
        "events": str(events.resolve()), "seeds": list(SEEDS),
    }
    run.mkdir(parents=True)
    engine = run / "engine"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", engine / "src", ignore=ignore)
    shutil.copytree(ROOT / "config" / "specialist", engine / "configs" / "specialist", ignore=ignore)
    current_raw = engine / "local_data" / "raw"
    current_raw.mkdir(parents=True)
    for path in [*movement_files(raw), raw / "submitting.parquet", raw / "metar.parquet"]:
        (current_raw / path.name).symlink_to(path.resolve())
    (current_raw / "metar_iem.csv").symlink_to(metar.resolve())
    (engine / "local_data" / "interim_nm").symlink_to("prepared/interim_nm", target_is_directory=True)
    (run / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(run: Path, raw: Path, metar: Path, aircraft_specs: Path, events: Path) -> None:
    manifest = initialize(run, raw, metar, aircraft_specs, events)
    execute(run, "pipeline.prepare", "--raw-dir", "local_data/raw", "--output-dir", "local_data/prepared",
            "--metar", manifest["metar"], "--aircraft-specs", manifest["aircraft_specs"],
            "--events", manifest["events"])
    execute(run, "src.feature_engineering.MLP_CatBoost_surface.prepare", "--raw-dir", "local_data/raw",
            "--output", "local_data/interim_surface_ensemble")
    (run / "prepared.json").write_text(json.dumps({"status": "complete"}) + "\n")


def train(run: Path, device: str) -> None:
    verify_inputs(run)
    if not (run / "prepared.json").is_file():
        raise ValueError("Preparation has not completed")
    models = Path("outputs/final_models")
    if (run / "engine" / models).exists():
        raise FileExistsError("Models already exist; use a new run for retraining")
    execute(run, "src.evaluation.specialist.build", "--device", device, "--seeds", *SEEDS,
            "--models-dir", models / "specialist")
    for expert in ("fco", "no_nm"):
        execute(run, f"src.modeling.specialist.{expert}", "final", "--device", device,
                "--seeds", *SEEDS, "--output", models / f"specialist_{expert}")
    for seed in (42, 1):
        name = f"structured_fco_d6_s{seed}_retrained"
        execute(run, "src.modeling.run_structured", "--name", name,
                "--interim-dir", "local_data/interim_nm",
                "--params-file", "configs/specialist/structured_params.json",
                "--airports", "LIRF", "--iterations", 600, "--depth", 6,
                "--valid-months", "2025-07", "2025-09", "2025-11", "2025-12",
                "--penalties", 0.001, 0.01, 0.1, 1.0, "--seed", seed,
                "--devices", device, "--threads", 8, "--save-models")
        experiments = list((run / "engine" / "outputs" / "experiments").glob(f"*_{name}"))
        if len(experiments) != 1:
            raise ValueError(f"Expected one structured training run for {name}: {experiments}")
        execute(run, "src.modeling.finalize_structured", "--experiment", experiments[0],
                "--output", models / name, "--devices", device, "--variants", "gate3_quality")
    (run / "trained.json").write_text(json.dumps({"status": "complete", "seeds": list(SEEDS)}) + "\n")


def predict(run: Path, models: Path | None = None) -> Path:
    verify_inputs(run)
    if not (run / "prepared.json").is_file():
        raise ValueError("Preparation has not completed")
    if models is None:
        if not (run / "trained.json").is_file():
            raise ValueError("Training has not completed; provide --models-dir to reuse a complete model bundle")
        models = run / "engine" / "outputs" / "final_models"
    output = run / "submission_specialist.parquet"
    execute(run, "src.evaluation.specialist.submission", "--output", output,
            "--specialist-models", models / "specialist", "--fco-models", models / "specialist_fco",
            "--no-nm-models", models / "specialist_no_nm", "--structured-models",
            models / "structured_fco_d6_s42_retrained", models / "structured_fco_d6_s1_retrained")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "train", "predict", "all"):
        command = subcommands.add_parser(stage)
        command.add_argument("--run-dir", type=Path, required=True)
        if stage in ("prepare", "all"):
            for name in ("raw", "metar", "aircraft-specs", "events"):
                command.add_argument(f"--{name}", type=Path, required=True)
        if stage in ("train", "all"):
            command.add_argument("--device", default="0")
        if stage == "predict":
            command.add_argument("--models-dir", type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    if args.stage in ("prepare", "all"):
        prepare(run, args.raw.resolve(), args.metar.resolve(), args.aircraft_specs.resolve(), args.events.resolve())
    if args.stage in ("train", "all"):
        train(run, args.device)
    if args.stage in ("predict", "all"):
        models = getattr(args, "models_dir", None)
        print(predict(run, models.resolve() if models else None))


if __name__ == "__main__":
    main()