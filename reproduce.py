
from __future__ import annotations

import argparse
import calendar
import json
import shutil
import sys
import time
from pathlib import Path

from chain.runner import RECIPE, Run, blockers, directory_signature, link, schedule, select
from chain.stages import stages
from common import CHALLENGE_DIR as CHALLENGE, RAW_DIR as RAW, ROOT, SUBMISSIONS_DIR as SUBMISSIONS, XPLANE_DIR, sha256

# Everything a run copies into RUN/code; config/source_manifest.json lists a hash for each of their files.
SOURCE_ITEMS = ("chain", "components", "config", "engines/final_model", "pipeline", "scripts", "src", "common.py", "dataset.py",
                "generate.py", "reproduce.py", "requirements.txt", "specialist.py")
MANIFEST = ROOT / "config" / "source_manifest.json"
IGNORED = ("__pycache__", "*.pyc", ".pytest_cache")
FILE_INPUTS = ("ranking", "template", "metar_csv", "aircraft_specs", "taxi_events")
DIRECTORY_INPUTS = ("xplane", "opdi_events", "adsblol_store")
OPTIONAL_INPUTS = ("specialist_models", "base_model_map")
# adsb.lol publishes a new day every day: days added after the run was created do not change its inputs.
APPEND_ONLY_INPUTS = ("adsblol_store",)


def signature(path: Path) -> dict:
    path = path.resolve()
    if path.is_dir():
        return {"path": str(path), "directory": directory_signature(path)}
    return {"path": str(path), "sha256": sha256(path)}


def collect_inputs(args) -> dict:
    raw = args.raw.resolve()
    training = sorted(raw.glob("training_2025-*.parquet"))
    if not training:
        raise FileNotFoundError(f"No training_2025-*.parquet in {raw}")
    paths = {"training": training, "metar_parquet": raw / "metar.parquet",
             "ranking": args.ranking or raw / "ranking.parquet", "template": args.template or raw / "submitting.parquet"}
    for name in FILE_INPUTS + DIRECTORY_INPUTS:
        if name not in paths:
            paths[name] = getattr(args, name)
    for name in OPTIONAL_INPUTS:
        if getattr(args, name, None) is not None:
            paths[name] = getattr(args, name)
    missing = [name for name, value in paths.items() if value is None]
    if missing:
        raise SystemExit(f"Missing inputs for a new run: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    for name, value in paths.items():
        for path in value if isinstance(value, list) else [value]:
            if not path.exists():
                raise FileNotFoundError(f"{name}: {path}")
    return {name: [signature(p) for p in value] if isinstance(value, list) else signature(value)
            for name, value in paths.items()}


def verify_inputs(inputs: dict, created_utc: str) -> None:
    created = calendar.timegm(time.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ"))
    for name, value in inputs.items():
        for recorded in value if isinstance(value, list) else [value]:
            path = Path(recorded["path"])
            current = ({"path": str(path.resolve()), "directory": directory_signature(path.resolve(), before=created)}
                       if name in APPEND_ONLY_INPUTS else signature(path))
            if current != recorded:
                raise ValueError(f"Input {name} changed since the run was created: {recorded['path']}. "
                                 "Create a new run directory.")


def source_files() -> list[Path]:
    files = []
    for item in SOURCE_ITEMS:
        path = ROOT / item
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        files += [f for f in candidates if f.is_file() and f != MANIFEST
                  and not any(part == "__pycache__" or part.endswith(".pyc") for part in f.relative_to(ROOT).parts)]
    return files


def write_manifest() -> None:
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest)} files -> {MANIFEST}")


def verify_sources() -> list[str]:
    """Differences between the checkout and config/source_manifest.json; an incomplete copy must not run."""
    if not MANIFEST.is_file():
        return [f"missing {MANIFEST.relative_to(ROOT)} (python reproduce.py manifest)"]
    manifest = json.loads(MANIFEST.read_text())
    present = {str(path.relative_to(ROOT)) for path in source_files()}
    problems = [f"missing {relative}" for relative in manifest if relative not in present]
    problems += [f"differs {relative}" for relative in manifest
                 if relative in present and sha256(ROOT / relative) != manifest[relative]]
    problems += [f"not in the manifest {relative}" for relative in sorted(present - set(manifest))]
    return problems


def create(run_path: Path, inputs: dict) -> None:
    """Copy the sources and lay the inputs out where the legacy scripts look for them."""
    code = run_path / "code"
    run_path.mkdir(parents=True)
    for item in SOURCE_ITEMS:
        source = ROOT / item
        if source.is_dir():
            shutil.copytree(source, code / item, ignore=shutil.ignore_patterns(*IGNORED))
        else:
            code.mkdir(exist_ok=True)
            shutil.copy2(source, code / item)
    raw = code / RAW
    for entry in inputs["training"]:
        link(Path(entry["path"]), raw / Path(entry["path"]).name)
    # The final dataset ships as final_ranking/final_submitting; every script reads the canonical names.
    link(Path(inputs["ranking"]["path"]), raw / "ranking.parquet")
    link(Path(inputs["template"]["path"]), raw / "submitting.parquet")
    link(Path(inputs["metar_parquet"]["path"]), raw / "metar.parquet")
    link(Path(inputs["metar_csv"]["path"]), raw / "metar_iem.csv")
    link(Path(inputs["xplane"]["path"]), code / XPLANE_DIR)
    link(Path(inputs["opdi_events"]["path"]), code / CHALLENGE / "eurocontrol" / "flight_events")
    link(Path(inputs["adsblol_store"]["path"]), code / "data" / "adsblol_history")
    (code / SUBMISSIONS).mkdir(parents=True)
    for directory in ("stages", "logs"):
        (run_path / directory).mkdir()


def command_run(args) -> None:
    run_path = args.run_dir.resolve()
    manifest_path = run_path / "run.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_inputs(manifest["inputs"], manifest["created_utc"])
        if args.raw is not None:
            raise SystemExit("The run already exists; its inputs are fixed. Omit --raw or use a new --run-dir.")
    else:
        if args.raw is None:
            raise SystemExit("A new run needs --raw and the other inputs")
        problems = verify_sources()
        if problems:
            raise SystemExit("The source tree differs from config/source_manifest.json:\n  " + "\n  ".join(problems))
        inputs = collect_inputs(args)
        create(run_path, inputs)
        manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version,
                    "inputs": inputs, "recipe": RECIPE, "verify_reference": args.verify_reference, "upload": False}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    run = Run(run_path, manifest["inputs"], args.procs, args.threads, args.device, manifest["verify_reference"])
    failed = schedule(select(stages(), args.until, args.only), run, args.jobs, args.retry_interrupted,
                      args.max_attempts)
    (run_path / "last_invocation.json").write_text(json.dumps(
        {"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "failed": failed}, indent=2) + "\n")
    if failed:
        raise SystemExit("Stages not completed:\n  " + "\n  ".join(f"{k}: {v}" for k, v in failed.items()))


def command_plan(_args) -> None:
    for stage in stages():
        flags = []
        if blockers(stage):
            flags.append(f"BLOCKED: confirm {', '.join(blockers(stage))}")
        if stage.note:
            flags.append(stage.note)
        print(f"{stage.name:18s} {stage.summary}")
        print(f"{'':18s} needs: {', '.join(stage.needs) or '-'} | outputs: {', '.join(stage.outputs)}")
        for flag in flags:
            print(f"{'':18s} ! {flag}")


def command_check(_args) -> None:
    problems = verify_sources()
    unconfirmed = sorted({key for stage in stages() for key in blockers(stage)})
    print(json.dumps({"source_problems": problems, "unconfirmed_recipe_values": unconfirmed}, indent=2))
    if problems or unconfirmed:
        raise SystemExit(1)


def command_status(args) -> None:
    run_path = args.run_dir.resolve()
    for stage in stages():
        record = run_path / "stages" / f"{stage.name}.json"
        marker = run_path / "stages" / f"{stage.name}.started"
        if record.exists():
            state = json.loads(record.read_text())["status"]
        elif marker.exists():
            started = json.loads(marker.read_text())
            state = f"started {started['started_utc']} (attempt {started['attempts']}; running or interrupted)"
        else:
            state = "pending"
        print(f"{stage.name:20s} {state}")


def command_manifest(_args) -> None:
    write_manifest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("check")
    commands.add_parser("manifest")
    status = commands.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--raw", type=Path, help="training_2025-*.parquet, metar.parquet and, by default, ranking/template")
    run.add_argument("--ranking", type=Path, help="default RAW/ranking.parquet (final dataset: final_ranking.parquet)")
    run.add_argument("--template", type=Path, help="default RAW/submitting.parquet (final: final_submitting.parquet)")
    run.add_argument("--metar-csv", type=Path, help="IEM ASOS METAR CSV")
    run.add_argument("--aircraft-specs", type=Path)
    run.add_argument("--taxi-events", type=Path, help="airport events CSV (data/taxiout_events.csv)")
    run.add_argument("--xplane", type=Path, help="X-Plane apt.dat directory for pipeline.context")
    run.add_argument("--opdi-events", type=Path, help="OPDI flight_events directory for pipeline.adsb")
    run.add_argument("--adsblol-store", type=Path, help="adsb.lol globe_history archives (read only)")
    run.add_argument("--verify-reference", action="store_true",
                     help="Require the historical v86/v87 hashes (only meaningful with the original January/July data)")
    run.add_argument("--until", help="Stop after this stage")
    run.add_argument("--only", help="Run a single stage (its dependencies must be complete)")
    run.add_argument("--procs", type=int, default=16)
    run.add_argument("--threads", type=int, default=16)
    run.add_argument("--device", default="0", help="GPU for the specialist")
    run.add_argument("--specialist-models", type=Path,
                     help="frozen specialist bundle (specialist, specialist_fco, specialist_no_nm, structured_fco_*): "
                          "prepare + predict instead of retraining, which needs a CUDA GPU")
    run.add_argument("--base-model-map", type=Path,
                     help="folder of X-Plane <ICAO>.dat files for the base model map (the gateway API is not always up)")
    run.add_argument("--jobs", type=int, default=1, help="stages run at the same time")
    run.add_argument("--retry-interrupted", action="store_true",
                     help="retry interrupted or failed stages, setting their partial work aside")
    run.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    {"plan": command_plan, "check": command_check, "status": command_status, "run": command_run,
     "manifest": command_manifest}[args.command](args)


if __name__ == "__main__":
    main()
