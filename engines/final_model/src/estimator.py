"""The final taxi estimator: schedule-copy mixture, airport regressions and LOBT bounds."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import movement_features as features
import post_rules
from runtime import DEVICE, release_memory
from weather_rotation import airline_vocabulary, weather_rotation_features

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=255, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, cat_smooth=20,
              max_cat_to_onehot=8, num_threads=12, deterministic=True,
              force_row_wise=True, verbose=-1)
SCHED_GAP = "to_takeoff_from_SCHED_TIME_UTC_mvt"
MIN_AIRPORT_ROWS = 20000
NM_MIN_ROWS = 50
LOBT_WINDOW_SECONDS = 3606


def validate_config(config: dict) -> None:
    fixed = {"model": "two_stage_nm", "nm_split_ms": False, "clip_lobt": True,
             "arrival_context": True, "weather_rotation": True, "airport_regressors": True}
    allowed = {*fixed, "cls_rounds", "reg_rounds", "seed", "nm_min_ms", "target_cap", "classifier_weight"}
    if set(config) - allowed or any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("The base configuration must use the frozen final estimator")
    if config.get("classifier_weight") not in (None, "quad"):
        raise ValueError("Only unweighted or squared-error classifier weights are supported")
    if config["nm_min_ms"] <= 0 or config["target_cap"] <= 0:
        raise ValueError("The NM threshold and target cap must be positive")
    if DEVICE != "cpu":
        raise ValueError("The frozen base estimator requires PRC_DEVICE=cpu")


def prepare(train: pd.DataFrame, others: list[pd.DataFrame]) -> list[str]:
    """Fit the P10 reference and airline vocabulary, then apply them to every frame."""
    reference = features.fit_reference(train)
    for frame in (train, *others):
        frame["ref_p10"] = features.apply_reference(frame, reference)
    features.as_categories([train, *others])
    columns = features.feature_columns(train) + [column for column in train if column.startswith("adsb_")]
    columns += [column for column in train if column.startswith("ctx_")]
    airlines = airline_vocabulary(train)
    for frame in (train, *others):
        extra = weather_rotation_features(frame, airlines)
        for column in extra:
            frame[column] = extra[column].array
    return columns + list(extra.columns)


def leaky_columns(train: pd.DataFrame, ranking: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in train and column in ranking
            and train[column].isna().mean() < 0.5 and ranking[column].isna().mean() > 0.95]


def combine(probability, schedule_gap, regression) -> np.ndarray:
    probability, schedule_gap, regression = (np.asarray(value, float)
                                              for value in (probability, schedule_gap, regression))
    return np.clip(np.where(np.isnan(schedule_gap), regression,
                            probability * schedule_gap + (1 - probability) * regression), 0, None)


def classifier_weights(schedule_gap) -> np.ndarray:
    schedule_gap = np.asarray(schedule_gap, float)
    weights = np.clip(np.abs(schedule_gap - 900), 60.0, None) ** 2
    weights = np.where(np.isnan(schedule_gap), 0.0, weights)
    return weights / weights[weights > 0].mean()


def lobt_window(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gap = (frame["MVT_TIME_UTC_mvt"] - frame["LOBT_flt"]).dt.total_seconds().to_numpy(float)
    return gap - LOBT_WINDOW_SECONDS, gap + LOBT_WINDOW_SECONDS


def clip_window(predictions, frame: pd.DataFrame) -> np.ndarray:
    lower, upper = lobt_window(frame)
    bounded = np.clip(np.asarray(predictions, float), np.nan_to_num(lower, nan=-np.inf),
                       np.nan_to_num(upper, nan=np.inf))
    return np.clip(bounded, 0, None)


def _line(schedule_gap, target) -> tuple[float, float]:
    slope, intercept = np.polyfit(schedule_gap, target, 1)
    return float(intercept), float(slope)


def fit_lines(schedule_gap, target, airports):
    schedule_gap, target, airports = np.asarray(schedule_gap, float), np.asarray(target, float), np.asarray(airports)
    valid = ~np.isnan(schedule_gap) & ~np.isnan(target)
    schedule_gap, target, airports = schedule_gap[valid], target[valid], airports[valid]
    lines = {str(airport): _line(schedule_gap[airports == airport], target[airports == airport])
             for airport in np.unique(airports) if (airports == airport).sum() >= NM_MIN_ROWS}
    return lines, _line(schedule_gap, target)


def apply_lines(schedule_gap, airports, lines: dict, fallback) -> np.ndarray:
    schedule_gap = np.asarray(schedule_gap, float)
    intercepts = np.array([lines.get(str(airport), fallback)[0] for airport in airports])
    slopes = np.array([lines.get(str(airport), fallback)[1] for airport in airports])
    return np.clip(intercepts + slopes * schedule_gap, 0, None)


def rule_rows(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    gap = frame[SCHED_GAP].to_numpy(float)
    use_line = (frame["nm_missing"] == 1).to_numpy() & ~np.isnan(gap) & (gap > threshold)
    return use_line | post_rules.rome_rows(frame)


class TaxiEstimator:
    """Fit one base with the exact components used by the final recipe."""

    def __init__(self, config: dict):
        validate_config(config)
        self.config = config
        self.params = {**PARAMS, "seed": int(config["seed"])}

    def fit(self, train, columns, run=None):
        self.columns = columns
        missing_nm = train[train["nm_missing"] == 1]
        self.lines, self.fallback = fit_lines(missing_nm[SCHED_GAP], missing_nm[features.TARGET],
                                              missing_nm[features.AIRPORT].astype(str).to_numpy())
        copied = (train["BLOCK_TIME_UTC_mvt"] - train["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs() <= 60
        classifier_params = {**self.params, "objective": "binary", "metric": "binary_logloss"}
        weight = classifier_weights(train[SCHED_GAP]) if self.config.get("classifier_weight") else None
        rounds = self.config["cls_rounds"]
        callbacks = [run.lgb_callback(rounds, "classifier", start=0.0, span=0.5)] if run else []
        self.classifier = lgb.train(classifier_params, lgb.Dataset(train[columns], copied.astype("int8"), weight=weight),
                                    rounds, callbacks=callbacks)
        self.classifier.free_dataset()
        release_memory()
        selected = list(dict.fromkeys([*columns, features.AIRPORT, "nm_missing", features.TARGET]))
        normal = train[selected][~copied]
        target = normal[features.TARGET].clip(upper=self.config["target_cap"])
        rounds = self.config["reg_rounds"]
        callbacks = [run.lgb_callback(rounds, "regressor", start=0.5, span=0.5)] if run else []
        self.regressor = lgb.train(self.params, lgb.Dataset(normal[columns], target), rounds, callbacks=callbacks)
        self.regressor.free_dataset()
        release_memory()
        self.airport_regressors = {}
        airports = normal[features.AIRPORT].astype(str).to_numpy()
        inputs = normal[columns]
        for airport in np.unique(airports):
            selected = airports == airport
            if selected.sum() >= MIN_AIRPORT_ROWS:
                model = lgb.train(self.params, lgb.Dataset(inputs[selected], target[selected]), rounds)
                model.free_dataset()
                self.airport_regressors[airport] = model
        release_memory()
        return self

    def predict(self, frame):
        inputs = frame[self.columns]
        airports = frame[features.AIRPORT].astype(str).to_numpy()
        global_prediction = np.asarray(self.regressor.predict(inputs), float)
        regression = global_prediction.copy()
        for airport, model in self.airport_regressors.items():
            selected = airports == airport
            if selected.any():
                regression[selected] = 0.5 * (global_prediction[selected] + np.asarray(model.predict(inputs[selected]), float))
        gap = frame[SCHED_GAP].to_numpy(float)
        probability = np.asarray(self.classifier.predict(inputs), float)
        lower, upper = lobt_window(frame)
        probability = np.where((gap < lower) | (gap > upper), 0.0, probability)
        prediction = combine(probability, gap, regression)
        lines = apply_lines(gap, airports, self.lines, self.fallback)
        selected = (frame["nm_missing"] == 1).to_numpy() & ~np.isnan(lines) & (gap > self.config["nm_min_ms"])
        prediction[selected] = lines[selected]
        return clip_window(prediction, frame)


def build_model(config: dict):
    return TaxiEstimator(config)