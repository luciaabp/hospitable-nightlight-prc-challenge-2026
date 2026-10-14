"""Train, predict, save and load the three CatBoost models of one airport.

    direct   = regressor of the official taxi-out, all departures
    taxi     = regressor of the taxi-out, departures with a measured block time only
    p        = classifier of block times copied from the schedule
    hybrid   = p * delay + (1 - p) * taxi   when delay > taxi, else taxi
    specialist = w * direct + (1 - w) * hybrid, averaged over seeds

Trees are chosen by early stopping on the last training months and refit on every
training month with ``REFIT_TREE_FACTOR`` times those trees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from .config import (EARLY_STOPPING_ROUNDS, INNER_MONTHS, MAX_TREES, PARAMS, REFIT_TREE_FACTOR, SEEDS)
from .features import AirportData

MODEL_NAMES = ("direct", "taxi", "classifier")


def combine(taxi, delay, probability) -> np.ndarray:
    """Blend the taxi estimate with the apparent delay by the copied-block probability."""
    taxi, delay, probability = (np.asarray(v, dtype=float) for v in (taxi, delay, probability))
    use_delay = np.isfinite(delay) & (delay > taxi)
    return np.where(use_delay, probability * delay + (1 - probability) * taxi, taxi)


def copied_block_weight(delay) -> np.ndarray:
    """Sample weight for the copied-block classifier: a misclassified flight costs about its delay squared."""
    return np.clip(1 + (np.maximum(np.nan_to_num(np.asarray(delay, float)), 0) / 3600) ** 2, 1, 25)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def fit(kind, x, y, train, inner_eval, seed, device, max_trees=MAX_TREES, weight=None):
    """Choose trees on ``inner_eval`` rows (part of ``train``), then refit on all of ``train``.

    ``weight`` optionally gives a per-row sample weight (aligned with ``x``).
    """
    model_class = CatBoostRegressor if kind == "regressor" else CatBoostClassifier
    params = dict(PARAMS, loss_function="RMSE" if kind == "regressor" else "Logloss",
                  random_seed=seed, devices=str(device))
    categorical = x.select_dtypes(include="object").columns.tolist()
    inner_fit = train & ~inner_eval
    w = (lambda mask: None) if weight is None else (lambda mask: np.asarray(weight)[mask])
    probe = model_class(iterations=max_trees, early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
    probe.fit(x.loc[inner_fit], y[inner_fit], sample_weight=w(inner_fit), cat_features=categorical,
              eval_set=(x.loc[inner_eval], y[inner_eval]), use_best_model=True)
    trees = max(int(round((probe.get_best_iteration() + 1) * REFIT_TREE_FACTOR)), 50)
    model = model_class(iterations=trees, **params)
    model.fit(x.loc[train], y[train], sample_weight=w(train), cat_features=categorical)
    return model


def train_airport(data: AirportData, train: np.ndarray, device="0", seeds=SEEDS,
                  inner_months=INNER_MONTHS, max_trees=MAX_TREES) -> dict:
    """Return ``{seed: {"direct", "taxi", "classifier"}}`` trained on the ``train`` rows."""
    months = sorted(data.meta.month[train].unique())[-inner_months:]
    inner_eval = train & data.meta.month.isin(months).to_numpy()
    measured = ~data.imputed
    models = {}
    for seed in seeds:
        models[seed] = {
            "direct": fit("regressor", data.x, data.y, train, inner_eval, seed, device, max_trees),
            "taxi": fit("regressor", data.x, data.y, train & measured, inner_eval & measured,
                        seed, device, max_trees),
            "classifier": fit("classifier", data.x, data.imputed.astype(int), train, inner_eval,
                              seed, device, max_trees),
        }
    return models


def predict(models: dict, x: pd.DataFrame, delay) -> pd.DataFrame:
    """Per-seed components plus seed-averaged ``direct`` and ``hybrid`` for the rows of ``x``."""
    out = pd.DataFrame(index=x.index)
    for seed, m in models.items():
        out[f"direct_s{seed}"] = np.maximum(m["direct"].predict(x), 0)
        taxi = np.maximum(m["taxi"].predict(x), 0)
        out[f"probability_s{seed}"] = m["classifier"].predict_proba(x)[:, 1]
        out[f"hybrid_s{seed}"] = combine(taxi, delay, out[f"probability_s{seed}"])
    out["direct"] = out[[f"direct_s{s}" for s in models]].mean(axis=1)
    out["hybrid"] = out[[f"hybrid_s{s}" for s in models]].mean(axis=1)
    return out


def specialist(predictions: pd.DataFrame, w_direct: float) -> np.ndarray:
    return np.maximum(w_direct * predictions["direct"] + (1 - w_direct) * predictions["hybrid"], 0).to_numpy()


def save_models(models: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for seed, m in models.items():
        for name in MODEL_NAMES:
            m[name].save_model(str(directory / f"s{seed}_{name}.cbm"))


def load_models(directory: Path, seeds=SEEDS) -> dict:
    models = {}
    for seed in seeds:
        models[seed] = {}
        for name in MODEL_NAMES:
            model = CatBoostClassifier() if name == "classifier" else CatBoostRegressor()
            model.load_model(str(directory / f"s{seed}_{name}.cbm"))
            models[seed][name] = model
    return models
