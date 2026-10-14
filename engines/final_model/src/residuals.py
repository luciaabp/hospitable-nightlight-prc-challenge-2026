"""The final residual ensemble and its cross-fitted feature tables."""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import airport_map
import arrival_context
import cell_reference
import movement_features as features
from adsb_events import FEATURES as ADSB_FEATURES
from blocked_training import month_blocks
from estimator import clip_window, lobt_window
from runway import columns as runway_features, retention_features
from runtime import release_memory
from surface import counts as surface_counts

SEED_SHIFT = int(os.environ.get("SEED_SHIFT", "0"))
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, verbose=-1, num_threads=12,
              seed=SEED_SHIFT, deterministic=True, force_row_wise=True)
AIRPORT_PARAMS = {**PARAMS, "min_data_in_leaf": 100}
CATBOOST = dict(iterations=800, depth=8, learning_rate=0.08,
                loss_function="RMSE", random_seed=SEED_SHIFT)


def plan_distances(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    gaps = frame.reindex(columns=[f"to_takeoff_from_{column}" for column in features.PLAN_REFS]).to_numpy(float)
    distances = gaps - np.asarray(prediction, float)[:, None]
    output = {f"dist_{column.split('_')[0].lower()}": distances[:, index]
              for index, column in enumerate(features.PLAN_REFS)}
    nearest = np.where(np.isnan(distances), np.inf, np.abs(distances)).min(axis=1)
    output["dist_min"] = np.where(np.isfinite(nearest), nearest, np.nan)
    output["n_planos"] = np.isfinite(gaps).sum(1).astype(float)
    output["n_planos_longos"] = (gaps > 3600).sum(1).astype(float)
    return output


def corrector_frame(frame: pd.DataFrame, base: np.ndarray, external: dict,
                    weather: pd.DataFrame, cell_features: pd.DataFrame,
                    surface: bool = False, runway: bool = False, retention: bool = False):
    columns = [features.AIRPORT, "nm_missing", "hour",
               *[column for column in frame if column.startswith("to_takeoff_from_")],
               *[column for column in arrival_context.COLS if column in frame]]
    inputs = frame[columns].copy()
    inputs[features.AIRPORT] = inputs[features.AIRPORT].astype("category")
    inputs["pred"] = np.asarray(base, float)
    lower, upper = lobt_window(frame)
    inputs["dist_lo"] = inputs["pred"].to_numpy(float) - lower
    inputs["dist_hi"] = upper - inputs["pred"].to_numpy(float)
    inputs = inputs.join(frame[ADSB_FEATURES])
    inputs["adsb_menos_pred"] = inputs["adsb_taxi_move"] - inputs["pred"]
    for name, values in external.items():
        inputs[name] = np.asarray(values, float)
    for name in weather:
        inputs[name] = weather[name].array
    for name, values in plan_distances(frame, inputs["pred"].to_numpy(float)).items():
        inputs[name] = values
    if surface:
        for name, values in surface_counts(frame, inputs["pred"].to_numpy(float)).items():
            inputs[name] = np.asarray(values, float)
    for name, values in airport_map.columns(frame).items():
        inputs[name] = values
    for name in cell_reference.COLS:
        inputs[name] = np.asarray(cell_features[name], float)
    inputs["cel_pred_menos_p50"] = inputs["pred"].to_numpy(float) - inputs["cel_p50"].to_numpy(float)
    if runway:
        extra = runway_features(frame)
        for name in extra:
            inputs[name] = extra[name].array
    if retention:
        for name, values in retention_features(frame).items():
            inputs[name] = np.asarray(values, float)
    return inputs


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame if isinstance(frame[column].dtype, pd.CategoricalDtype)]


def catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(deep=False)
    for column in categorical_columns(frame):
        output[column] = output[column].astype(str).fillna("nan")
    return output


class ResidualEnsemble:
    def __init__(self, global_model, airport_models: dict, catboost):
        self.global_model = global_model
        self.airport_models = airport_models
        self.catboost = catboost

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        global_prediction = np.asarray(self.global_model.predict(frame), float)
        airport_prediction = np.asarray(self.global_model.predict(frame), float)
        airports = frame[features.AIRPORT].astype(str).to_numpy()
        for airport, model in self.airport_models.items():
            selected = airports == airport
            if selected.any():
                airport_prediction[selected] = model.predict(frame[selected])
        catboost_prediction = np.asarray(self.catboost.predict(catboost_frame(frame)), float)
        return np.mean([global_prediction, airport_prediction, catboost_prediction], axis=0)


def fit_corrector(inputs: pd.DataFrame, target, base, rounds: int, params: dict | None = None):
    from catboost import CatBoostRegressor, utils

    residual = np.asarray(target, float) - np.asarray(base, float)
    global_params = {**PARAMS, **(params or {})}
    airport_params = {**AIRPORT_PARAMS, **(params or {})}
    global_model = lgb.train(global_params, lgb.Dataset(inputs, residual), rounds)
    global_model.free_dataset()
    release_memory()
    airports = inputs[features.AIRPORT].astype(str).to_numpy()
    airport_models = {}
    for airport in np.unique(airports):
        selected = airports == airport
        model = lgb.train(airport_params, lgb.Dataset(inputs[selected], residual[selected]), rounds)
        model.free_dataset()
        airport_models[airport] = model
    release_memory()
    catboost_params = {**CATBOOST, "verbose": 0}
    catboost_params.update({"task_type": "GPU"} if utils.get_gpu_device_count() > 0 else {"thread_count": 12})
    catboost = CatBoostRegressor(**catboost_params)
    catboost.fit(catboost_frame(inputs), residual, cat_features=categorical_columns(inputs))
    return ResidualEnsemble(global_model, airport_models, catboost)


def corrected_prediction(model, frame, base, rule=None, **feature_options):
    inputs = corrector_frame(frame, base, **feature_options)
    predictions = clip_window(np.asarray(base, float) + model.predict(inputs), frame)
    if rule is not None:
        predictions[rule] = np.asarray(base, float)[rule]
    return predictions


def reorder_rows(frame: pd.DataFrame, ids) -> pd.DataFrame:
    """Reorder by ID while releasing source columns to limit peak memory."""
    keys = pd.Index(frame[features.ID])
    if keys.has_duplicates:
        raise ValueError(f"Duplicate {features.ID} values")
    positions = keys.get_indexer(pd.Index(np.asarray(ids)))
    if (positions < 0).any():
        raise KeyError("Requested IDs are absent from the source frame")
    columns = [features.ID, *[column for column in frame if column != features.ID]]
    return pd.DataFrame({column: frame.pop(column).array.take(positions) for column in columns}, copy=False)


def cell_tables(train: pd.DataFrame, run=None):
    months = train["MVT_TIME_UTC_mvt"].dt.month
    month_to_block = month_blocks(months)
    months_by_block = {}
    for month, block in month_to_block.items():
        months_by_block.setdefault(block, []).append(month)
    columns = list(dict.fromkeys([*(column for key in cell_reference.KEYS for column in key),
                                  *(column for column in cell_reference.GAP_INPUTS if cell_reference.GAP and column in train)]))
    keys = train[columns]
    tables = {block: cell_reference.fit(keys[~months.isin(held_out)], train.loc[~months.isin(held_out), features.TARGET])
              for block, held_out in sorted(months_by_block.items())}
    if run:
        run.log(f"Cell references: {len(tables)} month blocks")
    return tables, cell_reference.fit(keys, train[features.TARGET]), month_to_block