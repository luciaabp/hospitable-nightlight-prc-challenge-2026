"""Train the frozen ensemble and write predictions in submission-template order."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import cell_reference
import movement_features as features
import post_rules
import runtime
from adsb_events import ROOT_DIR as ADSB_ROOT
from blocked_training import oof_base
from residuals import (cell_tables, corrected_prediction, corrector_frame, fit_corrector,
                       reorder_rows)
from dataset_cache import CACHE, DATA, TRUTH, load_split
from estimator import build_model, leaky_columns, prepare, rule_rows, validate_config
from external_data import TRAINING_MONTHS, airline_copy_rates, external_features
from runtime import ROOT, Run, release_memory
from weather_rotation import airline_vocabulary, weather_rotation_features

OOF_CACHE = CACHE / "oof_base"
OUT = ROOT / "submissions"
ROUNDS_SCALE = 1.2
RECIPE = ROOT / "recipe.json"


def load_recipe(path: Path = RECIPE) -> dict:
    """Read final configurations directly, without experiment IDs or registry lookups."""
    recipe = json.loads(path.read_text())
    members = recipe["members"]
    if not members or len({member["id"] for member in members}) != len(members):
        raise ValueError("The recipe must contain nonempty, unique ensemble members")
    allowed = {"model", "base", "base_config", "adsb", "rounds", "seed", "corrector",
               "external_data", "weather_rotation", "plan_distance", "exclude_rule_rows",
               "retention", "airport_geometry", "cell_reference", "queue", "surface",
               "corrector_params", "runway"}
    for member in members:
        config = member["config"]
        if set(config) - allowed:
            raise ValueError(f"Unsupported final-model options: {sorted(set(config) - allowed)}")
        if config["model"] != "stack_cf" or config["corrector"] != "ensemble":
            raise ValueError("The final recipe requires the cross-fitted residual ensemble")
        if not all(config.get(key) is True for key in (
                "adsb", "external_data", "weather_rotation", "plan_distance", "airport_geometry", "cell_reference")):
            raise ValueError("All mandatory final-model feature blocks must be enabled")
        validate_config(config["base_config"])
    if any(rule not in post_rules.RULES for rule in recipe["post_rules"]):
        raise ValueError("The recipe contains an unsupported post-processing rule")
    return recipe


def oof_key(base_config: dict) -> str:
    """Hash the base configuration, its source dependencies and input file metadata."""
    digest = hashlib.sha256(json.dumps(base_config, sort_keys=True).encode())
    source = ROOT / "src"
    visited = set()
    pending = ["blocked_training"]
    while pending:
        name = pending.pop()
        path = source / f"{name}.py"
        if name in visited or not path.exists():
            continue
        visited.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                pending.append(node.module.split(".")[0])
    for name in sorted(visited):
        path = source / f"{name}.py"
        digest.update(path.name.encode() + path.read_bytes())
    paths = [*sorted(DATA.glob("*.parquet")), *sorted((DATA / "mapa").glob("*.parquet")),
             ADSB_ROOT / "events.parquet"]
    for path in paths:
        path = path.resolve()
        stat = path.stat() if path.exists() else None
        digest.update((f"{path}:{stat.st_size}:{stat.st_mtime_ns}" if stat else f"{path}:-").encode())
    return digest.hexdigest()[:16]


def base_groups(members: list[dict]) -> list[tuple[dict, list[dict]]]:
    groups = {}
    for member in members:
        config = member["config"]["base_config"]
        groups.setdefault(json.dumps(config, sort_keys=True), (config, []))[1].append(member)
    return list(groups.values())


def build_submission(template: pd.DataFrame, ids, predictions,
                     max_fill_fraction: float = 0.001) -> tuple[pd.DataFrame, int]:
    by_id = pd.Series(np.asarray(predictions, float), index=np.asarray(ids))
    output = template[[features.ID]].copy()
    output[features.TARGET] = output[features.ID].map(by_id)
    missing = output[features.TARGET].isna()
    count = int(missing.sum())
    if count > max_fill_fraction * len(output):
        raise ValueError(f"{count} template IDs have no prediction; check ranking/template alignment")
    if count:
        output.loc[missing, features.TARGET] = float(np.median(predictions))
    return output, count


def final_base(base_config: dict, full: pd.DataFrame, ranking: pd.DataFrame, run: Run):
    config = dict(base_config)
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in config:
            config[key] = round(config[key] * ROUNDS_SCALE)
    columns = prepare(full, [ranking])
    dropped = leaky_columns(full, ranking, columns)
    columns = [column for column in columns if column not in dropped]
    run.log(f"Final base: {len(full):,} training rows, {len(ranking):,} ranking rows")
    return build_model(config).fit(full, columns, run=run).predict(ranking)


def feature_options(config: dict, frame: pd.DataFrame, copy_rates, airlines, cells) -> dict:
    return {
        "external": external_features(frame, copy_rates, TRAINING_MONTHS),
        "weather": weather_rotation_features(frame, airlines, include_queue=config.get("queue", False)),
        "surface": config.get("surface", False),
        "cell_features": cells,
        "runway": config.get("runway", False),
        "retention": config.get("retention", False),
    }


def train_member(config: dict, full: pd.DataFrame, ranking: pd.DataFrame, run: Run):
    base_config = config["base_config"]
    copy_rates = airline_copy_rates(run)
    airlines = airline_vocabulary(full)
    block_tables, all_tables, month_to_block = cell_tables(full, run)
    blind = load_split("blind2025")
    path = OOF_CACHE / f"{oof_key(base_config)}.parquet"
    if path.exists():
        out_of_fold = pd.read_parquet(path)
    else:
        out_of_fold = oof_base(base_config, full, blind, ranking, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        out_of_fold.to_parquet(temporary, index=False)
        temporary.replace(path)
    base = out_of_fold["pred"].to_numpy(float)
    ordered = reorder_rows(blind, out_of_fold[features.ID].to_numpy())
    del blind
    release_memory()
    blocks = pd.Series(out_of_fold["mes"].to_numpy()).map(month_to_block).to_numpy()
    cells = cell_reference.apply_by_block(ordered, blocks, block_tables)
    rules = rule_rows(ordered, base_config["nm_min_ms"]) if config.get("exclude_rule_rows") else None
    inputs = corrector_frame(ordered, base, **feature_options(config, ordered, copy_rates, airlines, cells))
    del ordered, cells
    release_memory()
    target = out_of_fold[TRUTH].to_numpy(float)
    if rules is not None:
        inputs, target, base = inputs[~rules], target[~rules], base[~rules]
    model = fit_corrector(inputs, target, base, rounds=config["rounds"], params=config.get("corrector_params"))
    return model, copy_rates, airlines, all_tables


def predict_group(base_config: dict, members: list[dict], full, ranking, run) -> list[np.ndarray]:
    trained = [(member["config"], train_member(member["config"], full, ranking, run)) for member in members]
    base = final_base(base_config, full, ranking, run)
    predictions = []
    for config, (model, copy_rates, airlines, tables) in trained:
        cells = cell_reference.apply(ranking, tables)
        rules = rule_rows(ranking, base_config["nm_min_ms"]) if config.get("exclude_rule_rows") else None
        predictions.append(corrected_prediction(
            model, ranking, base, rule=rules,
            **feature_options(config, ranking, copy_rates, airlines, cells)))
    del trained, base
    release_memory()
    return predictions


def predict_ensemble(members, rules, run, loader=load_split):
    """Reload inputs for each base because reference fitting mutates feature frames."""
    predictions = []
    full = ranking = None
    for base_config, group in base_groups(members):
        del full, ranking
        release_memory()
        full, ranking = loader("full2025"), loader("ranking2026")
        ids = ranking[features.ID].to_numpy(copy=True)
        predictions.extend(pd.Series(values, index=ids)
                           for values in predict_group(base_config, group, full, ranking, run))
    ids = ranking[features.ID].to_numpy()
    aligned = [prediction.reindex(ids).to_numpy(float) for prediction in predictions]
    if any(np.isnan(values).any() for values in aligned):
        raise ValueError("Base groups produced inconsistent ranking IDs or missing predictions")
    return ids, post_rules.apply(ranking, np.mean(aligned, axis=0), rules)


def submit(version: int) -> None:
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME")
    if not team:
        raise SystemExit("Set TEAM_NAME in the environment or engine .env file")
    recipe = load_recipe()
    for member in recipe["members"]:
        base_config = member["config"]["base_config"]
        base_config["seed"] = int(base_config.get("seed", 0)) + int(os.environ.get("SEED_SHIFT", "0"))
    OUT.mkdir(exist_ok=True)
    with Run(f"submit_v{version}", {"members": [member["id"] for member in recipe["members"]],
                                    "source": runtime.src_hash()}) as run:
        with run.phase("inputs", 0.05):
            template = pd.read_parquet(DATA / "submitting.parquet")
        with run.phase("ensemble", 0.85):
            ids, predictions = predict_ensemble(recipe["members"], recipe["post_rules"], run)
        with run.phase("submission", 0.10):
            output, filled = build_submission(template, ids, predictions)
            path = OUT / f"{team}_v{version}.parquet"
            output.to_parquet(path, index=False)
            run.set(output=str(path.relative_to(ROOT)), filled=filled)
            run.log(f"Wrote {len(output):,} predictions to {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["submit"])
    parser.add_argument("version", type=int)
    arguments = parser.parse_args()
    submit(arguments.version)