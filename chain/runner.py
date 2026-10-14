"""Stage engine of reproduce.py: command execution, stage records, retries and the parallel scheduler.

A run lives in RUN/: RUN/code is a frozen copy of the sources with the inputs linked under the relative paths of
common.py, RUN/stages/<stage>.json records each completed stage (commands, timings, output hashes) and
RUN/logs/<stage>.log its output."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common import ID, PRED as PREDICTION, RAW_DIR as RAW, ROOT, sha256

RECIPE = json.loads((ROOT / "config" / "recipe.json").read_text())
CHAIN = RECIPE["chain"]
RETRY_WAIT_SECONDS = 900
# Variables that would redirect a step (the base-model engine reads PRC_ADSB_RAIZ, SEED_SHIFT, BASE_GAP_FEATURES and
# TEAM_NAME; M2/M2r read M2_* and M2R_*): each command gets only the values its stage sets.
SCRUBBED_VARIABLES = ("PRC_ADSB_RAIZ", "SEED_SHIFT", "TEAM_NAME", "BASE_GAP_FEATURES", "PRC_VERIFY_REFERENCE",
                      "PYTHONPATH")
SCRUBBED_PREFIXES = ("M2_", "M2R_")
SETPRIV = shutil.which("setpriv")


# --- run, commands and stages -------------------------------------------------------------------------------------

@dataclass
class Run:
    path: Path
    inputs: dict
    procs: int
    threads: int
    device: str
    verify_reference: bool

    @property
    def code(self) -> Path:
        return self.path / "code"

    def environment(self) -> dict:
        """Inherited environment without the variables that redirect the scripts (each variant reads its own events)."""
        environment = {name: value for name, value in os.environ.items()
                       if name not in SCRUBBED_VARIABLES and not name.startswith(SCRUBBED_PREFIXES)}
        environment.update(PYTHONPATH=str(self.code), PRC_VERIFY_REFERENCE="1" if self.verify_reference else "0")
        return environment


@dataclass
class Command:
    argv: list
    cwd: Path
    env: dict = field(default_factory=dict)
    stdout: Path | None = None  # also keep stdout in this file (JSON reports)


@dataclass
class Stage:
    name: str
    summary: str
    commands: Callable[[Run], list]  # Command or an in-process callable (links, small files)
    outputs: tuple[str, ...]          # relative to the run directory
    needs: tuple[str, ...] = ()
    recipe_keys: tuple[str, ...] = ()  # chain values that must be confirmed (non-null)
    note: str = ""                     # reconstruction or open question shown by `plan`
    predictions: tuple[str, ...] = ()  # outputs checked as complete, finite, non-negative template predictions
    scratch: tuple[str, ...] = ()      # partial work moved aside before retrying an interrupted stage
    resumable: bool = False            # the commands resume their own partial work: retry in place


def directory_signature(path: Path, before: float | None = None) -> dict:
    """Files, sizes and mtimes: hashing adsb.lol archives or ADS-B cuts would take hours.

    With ``before`` (epoch seconds), only the files last modified earlier count: an archive that only grows keeps
    its signature, while a changed or deleted older file still changes it."""
    files = sorted(p for p in path.rglob("*") if p.is_file() and (before is None or p.stat().st_mtime < before))
    listing = "\n".join(f"{p.relative_to(path)}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files)
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "listing_sha256": hashlib.sha256(listing.encode()).hexdigest()}


def chain_value(key: str):
    value = CHAIN
    for part in key.split("."):
        value = value[part]
    return value


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=target.is_dir())


# --- execution ----------------------------------------------------------------------------------------------------

def execute(command: Command | Callable[[], None], run: Run, log: Path) -> dict:
    if callable(command):
        command()
        return {"in_process": getattr(command, "__name__", "callable")}
    env = {**run.environment(), **command.env}
    started = time.monotonic()
    print(f"  $ {' '.join(map(str, command.argv))}  (log: {log})", flush=True)
    with log.open("a") as stream:
        stream.write(f"\n$ (cd {command.cwd}) {' '.join(map(str, command.argv))}\n")
        stream.flush()
        # Long trainings write straight to the log so that progress is visible with `tail -f`.
        argv = [SETPRIV, "--pdeathsig", "TERM", *map(str, command.argv)] if SETPRIV else command.argv
        process = subprocess.run(argv, cwd=command.cwd, env=env, text=True, stderr=stream,
                                 stdout=subprocess.PIPE if command.stdout is not None else stream)
        if command.stdout is not None:
            stream.write(process.stdout)
    if command.stdout is not None:
        command.stdout.write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}: {' '.join(map(str, command.argv))}")
    return {"argv": list(map(str, command.argv)), "cwd": str(command.cwd), "env": command.env,
            "seconds": round(time.monotonic() - started, 1)}


def output_signature(path: Path) -> dict:
    if path.is_dir():
        return directory_signature(path)
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def check_prediction(path: Path, template: Path) -> dict:
    import numpy as np
    import pandas as pd

    expected = pd.read_parquet(template, columns=[ID])[ID]
    frame = pd.read_parquet(path)
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError(f"{path}: null or duplicate IDs")
    if set(frame[ID]) != set(expected):
        raise ValueError(f"{path}: IDs differ from the template")
    values = frame[PREDICTION].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: non-finite or negative predictions")
    return {"rows": len(frame), "template_order": bool(frame[ID].equals(expected)),
            "mean": float(values.mean()), "p50": float(np.median(values)), "max": float(values.max())}


def blockers(stage: Stage) -> list[str]:
    return [key for key in stage.recipe_keys if chain_value(key) is None]


class StageFailed(RuntimeError):
    """A started attempt failed; retrying it may help (network, killed process)."""


def stage_complete(run: Run, name: str) -> bool:
    record = run.path / "stages" / f"{name}.json"
    return record.is_file() and json.loads(record.read_text()).get("status") == "complete"


def set_aside(stage: Stage, run: Run, attempt: int) -> list[str]:
    """Move the partial outputs and scratch of an interrupted attempt out of the way; nothing is deleted."""
    if stage.resumable:
        return []
    destination = run.path / "interrupted" / f"{stage.name}-{attempt}"
    moved = []
    for relative in (*stage.outputs, *stage.scratch):
        source = run.path / relative
        if source.exists() or source.is_symlink():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)
            moved.append(relative)
    return moved


def run_stage(stage: Stage, run: Run, retry: bool = False, max_attempts: int = 3) -> None:
    record_path = run.path / "stages" / f"{stage.name}.json"
    marker = run.path / "stages" / f"{stage.name}.started"
    outputs = {name: run.path / name for name in stage.outputs}
    if record_path.exists():
        record = json.loads(record_path.read_text())
        current = {name: output_signature(path) for name, path in outputs.items() if path.exists()}
        if record.get("status") == "complete" and current == record["outputs"]:
            print(f"[{stage.name}] complete, outputs unchanged", flush=True)
            return
        raise RuntimeError(f"[{stage.name}] outputs differ from {record_path}; inspect the run or start a new one")
    attempts = json.loads(marker.read_text())["attempts"] if marker.exists() else 0
    existing = [name for name, path in outputs.items() if path.exists()]
    if attempts or existing:
        if not retry:
            if existing:
                raise RuntimeError(f"[{stage.name}] outputs exist without a stage record (interrupted run?): {existing}."
                                   " Use --retry-interrupted to set them aside and retry")
        elif attempts >= max_attempts:
            raise RuntimeError(f"[{stage.name}] gave up after {attempts} attempts; see logs/{stage.name}.log")
        else:
            moved = set_aside(stage, run, attempts)
            state = "resuming in place" if stage.resumable else f"moved aside: {moved or 'nothing'}"
            print(f"[{stage.name}] retrying after attempt {attempts} ({state})", flush=True)
    missing_values = blockers(stage)
    if missing_values:
        raise SystemExit(f"[{stage.name}] unconfirmed recipe values in config/recipe.json chain: {missing_values}")
    for need in stage.needs:
        if not stage_complete(run, need):
            raise RuntimeError(f"[{stage.name}] needs stage {need}")
    log = run.path / "logs" / f"{stage.name}.log"
    print(f"[{stage.name}] {stage.summary}", flush=True)
    started = time.time()
    marker.write_text(json.dumps({"attempts": attempts + 1, "started_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))}) + "\n")
    try:
        executed = [execute(command, run, log) for command in stage.commands(run)]
        missing = [name for name, path in outputs.items() if not path.exists()]
        if missing:
            raise RuntimeError(f"[{stage.name}] finished without its outputs: {missing}")
        template = run.code / RAW / "submitting.parquet"
        checks = {name: check_prediction(run.path / name, template) for name in stage.predictions}
    except Exception as error:
        raise StageFailed(f"[{stage.name}] attempt {attempts + 1} failed: {error}") from error
    record = {"status": "complete", "summary": stage.summary, "note": stage.note, "attempts": attempts + 1,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
              "seconds": round(time.time() - started, 1), "commands": executed, "checks": checks,
              "outputs": {name: output_signature(path) for name, path in outputs.items()}}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    marker.unlink()
    print(f"[{stage.name}] complete in {record['seconds']:.0f} s", flush=True)


def run_stage_retrying(stage: Stage, run: Run, retry: bool, max_attempts: int) -> None:
    while True:
        try:
            return run_stage(stage, run, retry, max_attempts)
        except StageFailed as error:
            if not retry:
                raise
            attempts = json.loads((run.path / "stages" / f"{stage.name}.started").read_text())["attempts"]
            if attempts < max_attempts:
                print(f"{error}; retrying in {RETRY_WAIT_SECONDS * attempts // 60} min", flush=True)
                time.sleep(RETRY_WAIT_SECONDS * attempts)


def schedule(selected: list[Stage], run: Run, jobs: int, retry: bool, max_attempts: int) -> dict[str, str]:
    """Start every stage whose needs are complete, at most `jobs` at a time, in list order. A failed stage blocks the
    stages that need it and nothing else. Returns the stages that did not complete, with the reason."""
    names = {stage.name for stage in selected}
    pending, running, failed = list(selected), {}, {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        while pending or running:
            for stage in list(pending):
                if len(running) >= jobs:
                    break
                blocked = [need for need in stage.needs if need in failed]
                absent = [need for need in stage.needs if need not in names and not stage_complete(run, need)]
                if blocked or absent:
                    pending.remove(stage)
                    failed[stage.name] = f"blocked by {blocked}" if blocked else f"needs stages not run: {absent}"
                    print(f"[{stage.name}] not started: {failed[stage.name]}", flush=True)
                elif all(stage_complete(run, need) for need in stage.needs):
                    pending.remove(stage)
                    running[pool.submit(run_stage_retrying, stage, run, retry, max_attempts)] = stage
            if not running:
                for stage in pending:
                    failed[stage.name] = "unschedulable"
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                stage = running.pop(future)
                try:
                    future.result()
                except BaseException as error:  # noqa: BLE001 - SystemExit from unconfirmed recipe values too
                    failed[stage.name] = str(error)
                    print(f"[{stage.name}] FAILED: {error}", flush=True)
    return failed


def select(all_stages: list[Stage], until: str | None, only: str | None) -> list[Stage]:
    names = [stage.name for stage in all_stages]
    for name in (until, only):
        if name is not None and name not in names:
            raise SystemExit(f"Unknown stage {name}; see `python reproduce.py plan`")
    if only:
        return [all_stages[names.index(only)]]
    return all_stages[:names.index(until) + 1] if until else all_stages
