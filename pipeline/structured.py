"""LIRF: structured three-candidate mixture (``gate3_quality``) and classifier with history.

Per seed: monthly causal OOF from March (clean-taxi regressor and imputed-block classifier trained on earlier months,
600 fixed trees), final gate (penalty chosen on December, refit on March-December), final models on all 2025 and an
optional walk-forward validation. Seeds 42 and 1 reproduce the LIRF part of v26 (and of v46 with January history).
"""
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from . import fnhist
from .config import (CATEGORICAL, DELAY, FEATURES, FNHIST_FEATURES, ID, INBOUND_WEIGHT_CAP_SEC, INBOUND_WEIGHT_SEC,
                     LIRF_DESTINATION, PipelineConfig, lirf_params)
from .data import day_shift
from .gate import fit_gate, predict_gate

log = logging.getLogger(__name__)
FINAL = "final"
# Final gate with the ``lirf_gate_month_penalty`` penalty (months of ``lirf_gate_penalty_months``).
MONTH_GATE = "final_gate3_quality_month.joblib"


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2)))


def model_dir(config: PipelineConfig, seed: int) -> Path:
    return config.run_dir / "models" / f"lirf_s{seed}"


def probability(classifier, X) -> np.ndarray:
    if isinstance(classifier, float):  # a fold without both classes
        return np.full(len(X), classifier)
    return classifier.predict_proba(X)[:, 1]


def classifier_weight(X: pd.DataFrame, config: PipelineConfig) -> np.ndarray | None:
    """LIRF classifier weights: 1 + min(max(d, 0), 12 h) / 30 min with the inbound features; none without."""
    if config.lirf_inbound_features is None:
        return None
    delay = np.clip(np.nan_to_num(X[DELAY].to_numpy(dtype=float)), 0, INBOUND_WEIGHT_CAP_SEC)
    return 1 + delay / INBOUND_WEIGHT_SEC


def categorical(X: pd.DataFrame) -> list[str]:
    """CatBoost categoricals: those of v26 and, optionally, the destination."""
    return [c for c in X.columns if c in CATEGORICAL or c in LIRF_DESTINATION]


def _fit_classifier(X, target, params, weight=None):
    if len(np.unique(target)) == 1:
        return float(target.mean())
    model = CatBoostClassifier(**params)
    model.fit(X, target.astype(int), cat_features=categorical(X), sample_weight=weight)
    return model


def _weights(job: dict, rows) -> np.ndarray | None:
    return None if job.get("weight") is None else job["weight"][rows]


def _base_job(job: dict) -> dict:
    """Clean-taxi regressor + classifier; predicts ``predict`` and optionally returns the models."""
    X, y, imputed, shifted, train = job["X"], job["y"], job["imputed"], job["shifted"], job["train"]
    rows = train[~(imputed | shifted)[train]]
    if not len(rows):
        raise ValueError("No ordinary taxi examples")
    regressor = CatBoostRegressor(**job["params"])
    regressor.fit(X.iloc[rows], y[rows], cat_features=categorical(X))
    classifier = _fit_classifier(X.iloc[train], imputed[train], job["classifier_params"], _weights(job, train))
    predict = X.iloc[job["predict"]]
    result = {"key": job["key"], "clean_taxi": regressor.predict(predict),
              "probability": probability(classifier, predict)}
    if job["keep"]:
        result["models"] = {"clean_taxi": regressor, "classifier": classifier}
    return result


def _fnhist_job(job: dict) -> dict:
    """Classifier with flight-number history (as-of on each training row)."""
    X, imputed, train = job["X"], job["imputed"], job["train"]
    classifier = _fit_classifier(X.iloc[train], imputed[train], job["classifier_params"], _weights(job, train))
    result = {"key": job["key"], "probability": probability(classifier, X.iloc[job["predict"]])}
    if job["keep"]:
        result["model"] = classifier
    return result


def _gate_job(job: dict) -> dict:
    """Choose the penalty on the last available OOF month and refit on all earlier OOF."""
    X, quality, base, y, shifted, months = (job[k] for k in ["X", "quality", "base", "y", "shifted", "months"])
    target, penalties = job["target"], job["penalties"]
    has_oof = base.clean_taxi.notna().to_numpy()
    before = months < target if target != FINAL else np.ones(len(months), bool)
    inner = np.flatnonzero(has_oof & before)
    tuning_month = max(months[inner])
    fitting = inner[months[inner] < tuning_month]
    tuning = inner[months[inner] == tuning_month]
    if not len(fitting):
        raise ValueError(f"The {target} gate needs at least two earlier OOF months")
    prior_tune = max(1e-7, float(shifted[months < tuning_month].mean()))
    scores = []
    for penalty in penalties:
        gate = fit_gate(X.iloc[fitting], quality.iloc[fitting], base.iloc[fitting], y[fitting], penalty, prior_tune)
        scores.append((rmse(y[tuning], predict_gate(gate, X.iloc[tuning], quality.iloc[tuning],
                                                    base.iloc[tuning])), penalty))
    _, penalty = min(scores)
    prior = max(1e-7, float(shifted[before].mean()))
    gate = fit_gate(X.iloc[inner], quality.iloc[inner], base.iloc[inner], y[inner], penalty, prior)
    result = {"target": target, "gate": gate, "penalty": penalty, "tuning_month": tuning_month,
              "tuning_scores": scores, "optimization": gate.optimization}
    if target != FINAL:
        valid = np.flatnonzero(months == target)
        rows = base.iloc[valid]
        result["valid"] = valid
        result["control"] = predict_gate(gate, X.iloc[valid], quality.iloc[valid], rows)
        result["fnhist"] = predict_gate(gate, X.iloc[valid], quality.iloc[valid],
                                        rows.assign(probability=job["fnhist_probability"]))
    return result


def train(config: PipelineConfig, seeds: list[int], X: pd.DataFrame, y: np.ndarray, months: np.ndarray,
          imputed: np.ndarray, quality: pd.DataFrame, flights: np.ndarray, ids: np.ndarray) -> None:
    """Every row is a 2025 LIRF departure, in file order; the gate only uses the 36 v26 features of ``X``."""
    seeds = [s for s in seeds if not (model_dir(config, s) / "manifest.json").exists()]
    if not seeds:
        log.info("LIRF already trained")
        return
    shifted = day_shift(y, imputed)
    periods = fnhist.periods_of(months)
    history = fnhist.asof_features(flights, imputed, periods)
    XH = pd.concat([X, history], axis=1)
    weight = classifier_weight(X, config)
    available = sorted(set(months))
    oof_months = available[config.lirf_first_oof_index:]
    all_rows = np.arange(len(y))

    jobs, fn_jobs = [], []
    for seed in seeds:
        common = {"y": y, "imputed": imputed, "shifted": shifted, "params": lirf_params(config, seed),
                  "classifier_params": lirf_params(config, seed, classifier=True), "weight": weight}
        for m in oof_months:
            jobs.append({**common, "X": X, "key": (seed, m), "train": np.flatnonzero(months < m),
                         "predict": np.flatnonzero(months == m), "keep": False})
        jobs.append({**common, "X": X, "key": (seed, FINAL), "train": all_rows, "predict": all_rows[:1],
                     "keep": True})
        fn_common = {"X": XH, "imputed": imputed, "weight": weight,
                     "classifier_params": lirf_params(config, seed, classifier=True, fnhist=True)}
        fn_jobs.append({**fn_common, "key": (seed, FINAL), "train": all_rows, "predict": all_rows[:1], "keep": True})
        if config.validate:
            for m in config.valid_months:
                fn_jobs.append({**fn_common, "key": (seed, m), "train": np.flatnonzero(months < m),
                                "predict": np.flatnonzero(months == m), "keep": False})
    # Cache of the expensive part: if the gate stage fails, the CatBoost fits are not repeated.
    cache = config.run_dir / "cache" / f"lirf_catboost_s{'_'.join(map(str, seeds))}.joblib"
    if cache.exists():
        base_results, fn_results = joblib.load(cache)
        log.info("LIRF: CatBoost fits read from %s", cache)
    else:
        start = time.time()
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            pending = pool.map(_base_job, jobs), pool.map(_fnhist_job, fn_jobs)
            base_results, fn_results = (list(results) for results in pending)
        log.info("LIRF: %d base fits and %d with history in %.0f s", len(jobs), len(fn_jobs), time.time() - start)
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((base_results, fn_results), cache)

    bases, finals, fn_final, fn_valid = {}, {}, {}, {}
    for seed in seeds:
        base = pd.DataFrame({"clean_taxi": np.nan, "probability": np.nan}, index=all_rows)
        base["delay"] = X[DELAY].to_numpy(dtype=float)
        bases[seed] = base
    for result in base_results:
        seed, m = result["key"]
        if m == FINAL:
            finals[seed] = result["models"]
            continue
        rows = np.flatnonzero(months == m)
        bases[seed].loc[rows, "clean_taxi"] = result["clean_taxi"]
        bases[seed].loc[rows, "probability"] = result["probability"]
    for result in fn_results:
        seed, m = result["key"]
        if m == FINAL:
            fn_final[seed] = result["model"]
        else:
            fn_valid[(seed, m)] = result["probability"]

    gate_jobs = []
    for seed in seeds:
        common = {"X": X[FEATURES], "quality": quality, "base": bases[seed], "y": y, "shifted": shifted,
                  "months": months, "penalties": list(config.gate_penalties)}
        gate_jobs.append({**common, "key": seed, "target": FINAL})
        if config.validate:
            for m in config.valid_months:
                gate_jobs.append({**common, "key": seed, "target": m, "fnhist_probability": fn_valid[(seed, m)]})
    start = time.time()
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        gate_results = list(pool.map(_gate_job, gate_jobs))
    log.info("LIRF: %d gates in %.0f s", len(gate_jobs), time.time() - start)

    table = fnhist.history_until(flights, imputed, periods, pd.Period(max(months), "M"))
    for seed in seeds:
        out = model_dir(config, seed)
        out.mkdir(parents=True, exist_ok=True)
        results = [r for job, r in zip(gate_jobs, gate_results) if job["key"] == seed]
        final_gate = next(r for r in results if r["target"] == FINAL)
        joblib.dump(finals[seed], out / "final_base.joblib")
        joblib.dump(final_gate["gate"], out / "final_gate3_quality.joblib")
        joblib.dump(fn_final[seed], out / "fnhist_classifier.joblib")
        table.reset_index().to_parquet(out / "fnhist_history.parquet", index=False)
        oof = bases[seed].assign(**{ID: ids, "month": months, "y_true": y, "imputed": imputed})
        oof[FNHIST_FEATURES] = history.to_numpy()
        for r in results:
            if r["target"] == FINAL:
                continue
            oof.loc[r["valid"], "gate3_quality"] = r["control"]
            oof.loc[r["valid"], "gate3_quality_fnhist"] = r["fnhist"]
            oof.loc[r["valid"], "probability_fnhist"] = fn_valid[(seed, r["target"])]
        oof.to_parquet(out / "oof.parquet", index=False)
        selection = [{k: r[k] for k in ["target", "penalty", "tuning_month", "tuning_scores", "optimization"]}
                     for r in results]
        (out / "selection.json").write_text(json.dumps(selection, indent=1))
        manifest = {"seed": seed, "rows": int(len(y)), "oof_months": oof_months,
                    "final_penalty": final_gate["penalty"], "params": lirf_params(config, seed),
                    "imputed_rate": float(imputed.mean()), "day_shift_rows": int(shifted.sum()),
                    "fnhist_flights": int(len(table)), "model_features": list(X.columns),
                    "gate_features": list(FEATURES), "classifier_weighted": weight is not None,
                    "fnhist_params": lirf_params(config, seed, classifier=True, fnhist=True)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log.info("LIRF s%d saved to %s (final penalty %s)", seed, out, final_gate["penalty"])
    cache.unlink()


def fit_month_gate(config: PipelineConfig, X: pd.DataFrame, y: np.ndarray, imputed: np.ndarray,
                   quality: pd.DataFrame, ids: np.ndarray) -> None:
    """Final gate with ``lirf_gate_month_penalty`` from the OOF saved by ``train``; no CatBoost retraining."""
    penalty = config.lirf_gate_month_penalty
    if penalty is None:
        return
    shifted = day_shift(y, imputed)
    for seed in config.lirf_seeds:
        out = model_dir(config, seed)
        target, record = out / MONTH_GATE, out / "month_gate.json"
        if target.exists() and json.loads(record.read_text())["penalty"] == penalty:
            continue
        oof = pd.read_parquet(out / "oof.parquet")
        if not np.array_equal(oof[ID].to_numpy(), np.asarray(ids)):
            raise ValueError(f"oof.parquet of s{seed} does not follow the training order")
        base = oof[["clean_taxi", "probability", "delay"]]
        rows = np.flatnonzero(base.clean_taxi.notna().to_numpy())
        gate = fit_gate(X[FEATURES].iloc[rows], quality.iloc[rows], base.iloc[rows], y[rows], penalty,
                        max(1e-7, float(shifted.mean())))
        joblib.dump(gate, target)
        record.write_text(json.dumps({"penalty": penalty, "rows": int(len(rows)),
                                      "optimization": gate.optimization}, indent=1))
        log.info("LIRF s%d: gate with penalty %s for %s", seed, penalty, list(config.lirf_gate_penalty_months))


def validation_report(config: PipelineConfig) -> pd.DataFrame | None:
    """Walk-forward RMSE per month: current gate against classifier with history, per seed and mean."""
    frames = {}
    for seed in config.lirf_seeds:
        path = model_dir(config, seed) / "oof.parquet"
        if not path.exists():
            return None
        oof = pd.read_parquet(path)
        frames[seed] = oof[oof.gate3_quality.notna()].reset_index(drop=True) if "gate3_quality" in oof else None
    if any(f is None for f in frames.values()):
        return None
    first = next(iter(frames.values()))
    if any(not np.array_equal(f[ID].to_numpy(), first[ID].to_numpy()) for f in frames.values()):
        raise ValueError("Seed OOF with different rows")
    rows = []
    variants = {f"s{seed}": frame for seed, frame in frames.items()}
    mean = first[[ID, "month", "y_true"]].copy()
    for column in ["gate3_quality", "gate3_quality_fnhist"]:
        mean[column] = np.mean([f[column].to_numpy() for f in frames.values()], axis=0)
    variants["mean"] = mean
    for name, frame in variants.items():
        for period, part in [("all", frame), *frame.groupby("month")]:
            control, treated = rmse(part.y_true, part.gate3_quality), rmse(part.y_true, part.gate3_quality_fnhist)
            rows.append({"seeds": name, "month": period, "rows": len(part), "rmse_gate3_quality": control,
                         "rmse_fnhist": treated, "delta": treated - control})
    return pd.DataFrame(rows)


def predict(config: PipelineConfig, seed: int, frame: pd.DataFrame, X: pd.DataFrame,
            quality: pd.DataFrame, months: np.ndarray) -> pd.DataFrame:
    """Pieces and prediction per flight, with and without history; ``frame``: LIRF rows only."""
    out = model_dir(config, seed)
    models = joblib.load(out / "final_base.joblib")
    gate = joblib.load(out / "final_gate3_quality.joblib")
    classifier = joblib.load(out / "fnhist_classifier.joblib")
    table = pd.read_parquet(out / "fnhist_history.parquet").set_index("fn")
    history = fnhist.apply_history(table, fnhist.flight_keys(frame))
    base = pd.DataFrame({"clean_taxi": models["clean_taxi"].predict(X),
                         "probability": probability(models["classifier"], X),
                         "delay": X[DELAY].to_numpy(dtype=float)})
    result = pd.DataFrame({ID: frame[ID].to_numpy(), "month": months})
    result[["clean_taxi", "probability", "delay"]] = base.to_numpy()
    result[FNHIST_FEATURES] = history.to_numpy()
    result["probability_fnhist"] = probability(classifier, pd.concat([X.reset_index(drop=True), history], axis=1))
    result["gate3_quality"] = predict_gate(gate, X[FEATURES], quality, base)
    result["gate3_quality_fnhist"] = predict_gate(gate, X[FEATURES], quality,
                                                  base.assign(probability=result.probability_fnhist.to_numpy()))
    rows = np.flatnonzero(np.isin(months, list(config.lirf_gate_penalty_months)))
    if config.lirf_gate_month_penalty is not None and len(rows):
        month_gate = joblib.load(out / MONTH_GATE)
        recorded = json.loads((out / "month_gate.json").read_text())["penalty"]
        if recorded != config.lirf_gate_month_penalty:
            raise ValueError(f"The month gate of s{seed} has penalty {recorded}: run train")
        part, part_quality, part_base = X[FEATURES].iloc[rows], quality.iloc[rows], base.iloc[rows]
        result.loc[rows, "gate3_quality"] = predict_gate(month_gate, part, part_quality, part_base)
        result.loc[rows, "gate3_quality_fnhist"] = predict_gate(
            month_gate, part, part_quality, part_base.assign(probability=result.probability_fnhist.to_numpy()[rows]))
    if not np.isfinite(result[["gate3_quality", "gate3_quality_fnhist"]].to_numpy()).all():
        raise ValueError("Non-finite LIRF predictions")
    return result
