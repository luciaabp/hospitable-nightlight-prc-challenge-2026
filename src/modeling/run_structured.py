"""Causal monthly cross-fitting and matched ablations for constrained mixtures.

python -m src.modeling.run_structured --help
All monthly base fits use fixed iterations, never the prediction month's labels.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.feature_engineering.record_quality import (VISIBLE_COLUMNS, record_quality_features,
                                                     record_quality_flags)
from .config import ExperimentConfig, NM_FLAG_FEATURES
from .data import load_dataset, ID_COLUMN
from .regime_metrics import attribution, compare_variants
from .structured_mixture import (ConvexGate, DAY, day_prior_estimate, day_shift_mask,
                                 day_signal_frame, experts)
from .tracking import ExperimentTracker, environment_info

log = logging.getLogger(__name__)


class Variant(NamedTuple):
    three: bool               # third candidate, taxi + one day
    quality: bool             # leave-one-out record-quality context
    day: bool = False         # the flight's own day-regime coordinates
    banded_prior: bool = False  # day prior conditioned on the visible band


VARIANTS = {"gate2": Variant(False, False), "gate2_quality": Variant(False, True),
            "gate3": Variant(True, False), "gate3_quality": Variant(True, True),
            "gate3_day": Variant(True, False, day=True),
            "gate3_quality_day": Variant(True, True, day=True),
            "gate3_quality_day_prior": Variant(True, True, day=True, banded_prior=True)}


def day_band(delay):
    """The visible band the banded prior conditions on; the same rule as the gate column."""
    return np.asarray(delay, dtype=float) > DAY/2


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(pred))**2)))


def temporal_splits(months, valid_months, min_history=2):
    available = sorted(set(months))
    for month in valid_months:
        if month not in available or available.index(month) < min_history:
            raise ValueError(f"Insufficient history or missing month: {month}")
    return [(month, np.flatnonzero(months < month), np.flatnonzero(months == month))
            for month in available[min_history:] if month <= max(valid_months)]


def fit_base(X, y, imputed, shifted, train, valid, params, cats):
    models = {}
    predictions = {}
    for name, exclusion in [("taxi", imputed), ("clean_taxi", imputed | shifted)]:
        rows = train[~exclusion[train]]
        if not len(rows):
            raise ValueError(f"No ordinary taxi examples for {name}")
        model = CatBoostRegressor(**params)
        model.fit(X.iloc[rows], y[rows], cat_features=cats)
        models[name] = model
        predictions[name] = model.predict(X.iloc[valid])
    cp = dict(params, loss_function="Logloss", eval_metric="Logloss")
    if len(np.unique(imputed[train])) == 1:
        models["classifier"] = float(imputed[train].mean())
        predictions["probability"] = np.full(len(valid), models["classifier"])
    else:
        classifier = CatBoostClassifier(**cp)
        classifier.fit(X.iloc[train], imputed[train].astype(int), cat_features=cats)
        models["classifier"] = classifier
        predictions["probability"] = classifier.predict_proba(X.iloc[valid])[:, 1]
    return models, predictions


def gate_frame(X, quality, flags, base, spec):
    # Cap categorical capacity in ConvexGate; preserve existing base model features.
    result = X.copy()
    if spec.quality:
        result = pd.concat([result, quality], axis=1)
    if spec.day:
        # Only meaningful with the third candidate present.
        signals = day_signal_frame(base.delay.to_numpy(), flags.reset_index(drop=True))
        result = pd.concat([result, signals.set_axis(result.index)], axis=1)
    result["base_taxi"] = base.clean_taxi.to_numpy() if spec.three else base.taxi.to_numpy()
    result["base_delay"] = base.delay.to_numpy()
    result["base_probability"] = base.probability.to_numpy()
    result["base_gap"] = result.base_delay - result.base_taxi
    return result


def fit_gate(X, quality, flags, base, y, rows, spec, penalty, day_prior, shifted=None):
    frame = gate_frame(X.iloc[rows].reset_index(drop=True), quality.iloc[rows].reset_index(drop=True),
                       flags.iloc[rows].reset_index(drop=True), base.iloc[rows], spec)
    rates = None
    if spec.banded_prior:
        if shifted is None:
            raise ValueError("A banded prior needs the day-shift supervision of the fitting rows")
        # Estimated on the fitting rows only, which are always earlier months.
        rates = day_prior_estimate(day_band(base.delay.to_numpy()[rows]), shifted[rows])
    return ConvexGate(spec.three, penalty, day_prior, day_rates=rates).fit(
        frame, frame.base_taxi, frame.base_delay, frame.base_probability, y[rows])


def predict_gate(model, X, quality, flags, base, rows, spec):
    frame = gate_frame(X.iloc[rows].reset_index(drop=True), quality.iloc[rows].reset_index(drop=True),
                       flags.iloc[rows].reset_index(drop=True), base.iloc[rows], spec)
    return model.predict(frame, frame.base_taxi, frame.base_delay, frame.base_probability)


def run(args):
    config = ExperimentConfig(name=args.name, interim_dir=Path(args.interim_dir), airports=args.airports,
                              seed=args.seed, task_type=args.task_type, devices=args.devices,
                              cv_strategy="walk_forward", valid_months=args.valid_months, n_trials=0,
                              fit_final_model=args.fit_final, final_max_iterations=args.iterations,
                              aircraft_backfill_dir=getattr(args, "aircraft_backfill_dir", None))
    config.numeric_features = config.numeric_features + NM_FLAG_FEATURES
    dataset = load_dataset(config)
    X, y, meta = dataset.X, dataset.y.to_numpy(), dataset.meta
    # Compute context before airport filtering, on the entire visible batch; join by ID.
    visible = list(dict.fromkeys(VISIBLE_COLUMNS + [ID_COLUMN]))
    records = pd.concat([pd.read_parquet(p, columns=visible) for p in sorted(config.interim_dir.glob(config.file_pattern))], ignore_index=True)
    if records[ID_COLUMN].duplicated().any() or meta[ID_COLUMN].duplicated().any():
        raise ValueError("Movement IDs must be unique")
    quality = record_quality_features(records)
    quality.index = records[ID_COLUMN]
    quality = quality.loc[meta[ID_COLUMN]].reset_index(drop=True)
    flags = record_quality_flags(records)
    flags.index = records[ID_COLUMN]
    flags = flags.loc[meta[ID_COLUMN]].reset_index(drop=True)
    timestamps = records.set_index(ID_COLUMN).loc[meta[ID_COLUMN], "MVT_TIME_UTC_mvt"].reset_index(drop=True)
    del records
    months = meta.month.to_numpy()
    imputed = meta.block_sched_gap_sec.le(60).to_numpy()
    shifted = day_shift_mask(y, imputed)
    prediction_months = args.valid_months + ([max(months)] if args.fit_final else [])
    splits = temporal_splits(months, prediction_months)
    tracker = ExperimentTracker(Path(args.experiments_dir), args.name)
    tracker.save_json("config.json", {**vars(args), "base_config": config.to_dict(),
                                     "day_rule": "not imputed and 0 <= y-86400 <= 7200",
                                     "context": "visible batch, leave-one-out, UTC groups",
                                     "base_early_stopping": False})
    tracker.save_json("environment.json", environment_info())
    params = json.loads(Path(args.params_file).read_text())
    for key in ["od_type", "od_wait", "early_stopping_rounds"]:
        params.pop(key, None)
    params.update(iterations=args.iterations, random_seed=args.seed, task_type=args.task_type,
                  allow_writing_files=False, verbose=False, thread_count=args.threads)
    if args.task_type == "GPU":
        params["devices"] = args.devices
    else:
        params.pop("devices", None)
    if getattr(args, "depth", None) is not None:
        params["depth"] = args.depth
    tracker.save_json("params.json", params)
    base = pd.DataFrame(np.nan, index=X.index, columns=["taxi", "clean_taxi", "probability"])
    base["delay"] = X.takeoff_minus_SCHED_TIME_UTC_mvt
    output = meta.copy()
    output["day"] = timestamps.dt.strftime("%Y-%m-%d")
    output["y_true"], output["imputed"], output["day_shift_candidate"] = y, imputed, shifted
    selection, metrics = [], []
    for month, train, valid in splits:
        log.info("Base %s: train=%d, predict=%d", month, len(train), len(valid))
        models, predictions = fit_base(X, y, imputed, shifted, train, valid, params, config.categorical_features)
        for col, values in predictions.items():
            base.loc[valid, col] = values
        if args.save_models:
            joblib.dump(models, tracker.path(f"models/{month}_base.joblib"))
        del models
        tracker.save_parquet("base_predictions.parquet", pd.concat([meta, base], axis=1))
        if month not in args.valid_months:
            continue
        inner = np.flatnonzero((months < month) & base.taxi.notna().to_numpy())
        inner_months = sorted(set(months[inner]))
        if len(inner_months) < 2:
            raise ValueError("Gate selection needs at least two historical OOF months (outer month >= fifth month)")
        tuning_month = inner_months[-1]
        fitting = inner[months[inner] < tuning_month]
        tuning = inner[months[inner] == tuning_month]
        baseline = experts(base.loc[valid, "taxi"], base.loc[valid, "delay"])
        p = base.loc[valid, "probability"].to_numpy()
        output.loc[valid, "baseline"] = baseline[:, 0]*(1-p) + baseline[:, 1]*p
        cleaned = experts(base.loc[valid, "clean_taxi"], base.loc[valid, "delay"])
        output.loc[valid, "clean_baseline"] = cleaned[:, 0]*(1-p) + cleaned[:, 1]*p
        for variant, spec in VARIANTS.items():
            candidates = []
            # Day prior for tuning uses only history before the internal holdout.
            prior_tune = max(1e-7, float(shifted[months < tuning_month].mean()))
            for penalty in args.penalties:
                gate = fit_gate(X, quality, flags, base, y, fitting, spec, penalty, prior_tune, shifted)
                score = rmse(y[tuning], predict_gate(gate, X, quality, flags, base, tuning, spec))
                candidates.append((score, penalty))
            _, penalty = min(candidates)
            prior = max(1e-7, float(shifted[train].mean()))
            gate = fit_gate(X, quality, flags, base, y, inner, spec, penalty, prior, shifted)
            output.loc[valid, variant] = predict_gate(gate, X, quality, flags, base, valid, spec)
            selection.append({"month": month, "variant": variant, "penalty": penalty,
                              "inner_months": inner_months, "tuning_month": tuning_month,
                              "tuning_scores": candidates, "optimization": gate.optimization})
            if args.save_models:
                joblib.dump(gate, tracker.path(f"models/{month}_{variant}.joblib"))
        for variant in ["baseline", "clean_baseline", *VARIANTS]:
            score = rmse(y[valid], output.loc[valid, variant])
            metrics.append({"month": month, "variant": variant, "rows": len(valid), "rmse": score})
            log.info("%s %s RMSE %.3f", month, variant, score)
        tracker.save_json("selection.json", selection)
        tracker.save_csv("metrics.csv", pd.DataFrame(metrics))
        tracker.save_parquet("predictions.parquet", output[output.baseline.notna()])
    evaluated = output[output.baseline.notna()].copy()
    summaries = []
    for variant in ["baseline", "clean_baseline", *VARIANTS]:
        for cohort, mask in {"all": np.ones(len(evaluated), bool), "over12h_imputed": evaluated.imputed & (evaluated.y_true>43200),
                             "over12h_other": ~evaluated.imputed & (evaluated.y_true>43200),
                             "under12h": evaluated.y_true<=43200}.items():
            part = evaluated.loc[mask]
            summaries.append({"variant": variant, "cohort": cohort, "rows": len(part),
                              "rmse": rmse(part.y_true, part[variant]) if len(part) else None})
    tracker.save_csv("summary.csv", pd.DataFrame(summaries))
    # Regime accounting beside RMSE: the two failure directions move independently.
    tracker.save_csv("regimes.csv", compare_variants(evaluated, ["baseline", "clean_baseline", *VARIANTS]))
    tracker.save_csv("regime_attribution.csv", attribution(evaluated, ["baseline", *VARIANTS]))
    # Paired daily error differences and leave-one-day-out sensitivity, without selecting models on it.
    daily = []
    for variant in VARIANTS:
        dif = (evaluated.y_true-evaluated[variant])**2 - (evaluated.y_true-evaluated.baseline)**2
        grouped = pd.DataFrame({"day": evaluated.day, "sse_difference": dif, "rows": 1}).groupby("day").sum().reset_index()
        grouped["variant"] = variant
        daily.append(grouped)
    tracker.save_csv("paired_days.csv", pd.concat(daily, ignore_index=True))
    if args.fit_final:
        log.info("Fitting final components on all training months")
        all_rows = np.arange(len(y))
        models, _ = fit_base(X, y, imputed, shifted, all_rows, all_rows[:1], params, config.categorical_features)
        joblib.dump(models, tracker.path("models/final_base.joblib"))
        inner = np.flatnonzero(base.taxi.notna().to_numpy())
        tuning_month = max(months[inner])
        fitting = inner[months[inner] < tuning_month]
        tuning = inner[months[inner] == tuning_month]
        if not len(fitting):
            raise ValueError("Insufficient OOF history for final gates")
        for variant, spec in VARIANTS.items():
            scores = []
            prior_tune = max(1e-7, float(shifted[months < tuning_month].mean()))
            for penalty in args.penalties:
                gate = fit_gate(X, quality, flags, base, y, fitting, spec, penalty, prior_tune, shifted)
                scores.append((rmse(y[tuning], predict_gate(gate, X, quality, flags, base, tuning, spec)), penalty))
            _, penalty = min(scores)
            gate = fit_gate(X, quality, flags, base, y, inner, spec, penalty, max(1e-7, float(shifted.mean())), shifted)
            joblib.dump(gate, tracker.path(f"models/final_{variant}.joblib"))
            selection.append({"month": "final", "variant": variant, "penalty": penalty,
                              "tuning_month": tuning_month, "tuning_scores": scores,
                              "optimization": gate.optimization})
        tracker.save_json("selection.json", selection)
    overall = {row["variant"]: row["rmse"] for row in summaries if row["cohort"] == "all"}
    tracker.save_text("summary.md", "# Structured mixture experiment\n\n" +
                      "Fixed-iteration causal base models; internally selected gates.\n\n" +
                      "\n".join(f"- {variant}: {score:.3f} s" for variant, score in overall.items()))
    tracker.append_to_index({"experiment_id": tracker.experiment_id, "created_at": str(tracker.created_at),
                             "name": args.name, "rows": len(y), "n_features": X.shape[1],
                             "cv_strategy": "walk_forward", "valid_months": ",".join(args.valid_months),
                             "score_name": "matched_baseline_rmse", "score": overall["baseline"],
                             "seed": args.seed, "airports": ",".join(args.airports),
                             "experiment_type": "structured_mixture", "path": str(tracker.root)})
    tracker.save_json("status.json", {"status": "complete"})
    log.info("Results: %s", tracker.root)
    return tracker.root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", default="local_data/interim_nm")
    parser.add_argument("--params-file", default="outputs/experiments/20260916-113801_wf_global_nm_flags/cv/params.json")
    parser.add_argument("--experiments-dir", default="outputs/experiments")
    parser.add_argument("--name", default="structured_fco")
    parser.add_argument("--airports", nargs="+", default=["LIRF"])
    parser.add_argument("--valid-months", nargs="+", default=["2025-07", "2025-09", "2025-11", "2025-12"])
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, help="Optional base depth override for pilot runs")
    parser.add_argument("--penalties", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aircraft-backfill-dir",
                        help="Folder with flight_list_*.parquet to fill the missing aircraft type")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--fit-final", action="store_true", help="Fit and save an all-training inference bundle")
    args = parser.parse_args()
    if args.iterations < 1 or any(p <= 0 for p in args.penalties):
        parser.error("Iterations and penalties must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
