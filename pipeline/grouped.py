"""Nine airports: grouped mixture (the non-LIRF part of v26).

Per seed: one taxi regressor per airport group (measured blocks only) and one imputed-block classifier.

    prediction = max(p d + (1 - p) r, 0)   if d > r
                 max(r, 0)                 otherwise

Seed 42 reproduces the models of ``outputs/final_models/2026-09-17_grouped9`` bit for bit; seeds are averaged. Optional
feature families (ADS-B, pre-takeoff context, extended METAR, own times, previous arrival, runway queue, batch context)
and the AOBT_3-anchored members only change the regressors; the classifier always uses the 36 v26 features.
"""
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .adsb import CONTEXT_VALUES
from .config import (CATEGORICAL, CONTEXT_DELAY, CONTEXT_FEATURES, CONTEXT_RUNWAY_CATEGORICAL, DELAY, FEATURES,
                     FORWARD_FEATURES, GROUPED_INBOUND_FEATURES, GROUPS, ID, LGB_PARAMS, OTHER_AIRPORTS, STATE_FEATURES,
                     TIME, TIMELINE_FEATURES,
                     WX_FEATURES,
                     PipelineConfig, grouped_params)

log = logging.getLogger(__name__)


def capped_taxi(taxi, delay, margin: float | None) -> np.ndarray:
    """``min(r, max(d, 0) + margin)``: with a measured block, the label only exceeds the delay if the aircraft leaves
    before the scheduled time."""
    taxi = np.asarray(taxi, dtype=float)
    if margin is None:
        return taxi
    bound = np.maximum(np.nan_to_num(np.asarray(delay, dtype=float), nan=0.0), 0) + margin
    return np.minimum(taxi, bound)


def _lgb_frame(X: pd.DataFrame, categories: list | None = None, columns: list[str] = FEATURES) -> pd.DataFrame:
    """Categoricals as pandas ``category``; at prediction, with the training levels."""
    frame = X[list(columns)].copy()
    for i, column in enumerate(c for c in columns if c in CATEGORICAL):
        levels = pd.Index(frame[column].unique()) if categories is None else categories[i]
        frame[column] = pd.Categorical(frame[column], categories=levels)
    return frame


def fit_lgb(X: pd.DataFrame, y: np.ndarray, seed: int, threads: int | None, rounds: int,
            columns: list[str] = FEATURES) -> lgb.Booster:
    params = dict(LGB_PARAMS, seed=seed, num_threads=threads or 0)
    categorical = [c for c in columns if c in CATEGORICAL]
    return lgb.train(params, lgb.Dataset(_lgb_frame(X, columns=columns), y, categorical_feature=categorical), rounds)


def predict_lgb(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """With the features the booster was trained on."""
    return booster.predict(_lgb_frame(X, booster.pandas_categorical, booster.feature_name()))


def has_context(X: pd.DataFrame) -> bool:
    return all(c in X.columns for c in CONTEXT_FEATURES)


# The runway queue (except its ADS-B features, which go to the ADS-B regressor) too, and the batch context
# (except the ADS-B family).
OWN_FLIGHT = TIMELINE_FEATURES + GROUPED_INBOUND_FEATURES + STATE_FEATURES + FORWARD_FEATURES


def timeline_columns(X: pd.DataFrame) -> list[str]:
    """The flight's own times and previous arrival, if ``X`` has them: in every taxi regressor."""
    return [c for c in OWN_FLIGHT if c in X.columns]


def regressor_columns(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Taxi regressor features: (CatBoost, its categoricals, LightGBM)."""
    timeline = timeline_columns(X)
    if has_context(X):
        return (FEATURES + CONTEXT_FEATURES + timeline, CATEGORICAL + CONTEXT_RUNWAY_CATEGORICAL,
                FEATURES + CONTEXT_DELAY + timeline)
    return FEATURES + timeline, CATEGORICAL, FEATURES + timeline


def combine(taxi, delay, probability) -> np.ndarray:
    taxi, delay, probability = (np.asarray(v, dtype=float) for v in (taxi, delay, probability))
    use_delay = np.isfinite(delay) & (delay > taxi)
    return np.maximum(np.where(use_delay, probability * delay + (1 - probability) * taxi, taxi), 0)


def group_of(airports: pd.Series) -> np.ndarray:
    lookup = {airport: group for group, members in GROUPS.items() for airport in members}
    groups = airports.map(lookup)
    if groups.isna().any():
        raise ValueError(f"Airports without a group: {sorted(airports[groups.isna()].unique())}")
    return groups.to_numpy()


def adsb_columns(X: pd.DataFrame) -> list[str]:
    """ADS-B features of ``X``: those that belong to no other family."""
    return [c for c in X.columns if c not in FEATURES + CONTEXT_FEATURES + WX_FEATURES + OWN_FLIGHT]


def wx_columns(X: pd.DataFrame, config: PipelineConfig, group: str) -> list[str]:
    """Extended METAR of ``group``'s regressor: only in the ``wx_groups`` groups."""
    return [c for c in WX_FEATURES if c in X.columns] if group in config.wx_groups else []


def has_adsb(X: pd.DataFrame) -> np.ndarray:
    """Departures with some ADS-B context value (the ``_n`` counts are 0 without data and do not count)."""
    columns = [c for c in CONTEXT_VALUES if c in X.columns]
    return X[columns].notna().any(axis=1).to_numpy() if columns else np.zeros(len(X), bool)


def model_dir(config: PipelineConfig, seed: int) -> Path:
    return config.run_dir / "models" / f"grouped_s{seed}"


ANCHOR_CLIP_SEC = 7200.0


def anchor_of(frame: pd.DataFrame) -> np.ndarray:
    """Anchor ``clip(t - AOBT_3, 0, 2 h)`` per departure; NaN without ``AOBT_3``."""
    seconds = (pd.to_datetime(frame[TIME], utc=True) - pd.to_datetime(frame["AOBT_3_flt"], utc=True)).dt.total_seconds()
    return np.clip(seconds.to_numpy(dtype=float), 0, ANCHOR_CLIP_SEC)


def uses_anchor(config: PipelineConfig) -> bool:
    return bool(config.grouped_anchor_cb_weight or config.grouped_anchor_lgb_weight)


def _check_anchor(config: PipelineConfig) -> None:
    if not (0 <= config.grouped_anchor_cb_weight <= 1 and 0 <= config.grouped_anchor_lgb_weight <= 1):
        raise ValueError("The anchored members' weight must be between 0 and 1")
    if config.grouped_anchor_lgb_weight and not config.grouped_lgb_weight:
        raise ValueError("The anchored LightGBM needs LightGBM")


def anchor_models(config: PipelineConfig, adsb: bool) -> list[str]:
    """Files of the anchored members requested by ``config``; ``adsb``: whether there are ADS-B features."""
    names = []
    for group, members in GROUPS.items():
        if config.grouped_anchor_cb_weight:
            names += [f"{group}_taxi_anchor.cbm"] + ([f"{group}_taxi_adsb_anchor.cbm"] if adsb else [])
        if config.grouped_anchor_lgb_weight:
            airports = list(members) if config.grouped_airport_lgb_weight else []
            names += [f"{name}_taxi_lgb_anchor.txt" for name in [group] + airports]
            if adsb and config.grouped_lgb_adsb:
                names += [f"{name}_taxi_lgb_adsb_anchor.txt" for name in [group] + [
                    a for a in airports if config.adsb_airports is None or a in config.adsb_airports]]
    return names


def _in_months(frame: pd.DataFrame, months: tuple[str, ...] | None) -> np.ndarray:
    if months is None:
        return np.ones(len(frame), bool)
    return np.isin(pd.to_datetime(frame[TIME], utc=True).dt.strftime("%Y-%m").to_numpy(), list(months))


def _train_anchored(config: PipelineConfig, seed: int, out: Path, X: pd.DataFrame, y: np.ndarray,
                    airports: pd.Series, groups: np.ndarray, measured: np.ndarray, anchor: np.ndarray,
                    columns: tuple[list[str], list[str], list[str]], extra: list[str], lgb_rounds: dict) -> list[dict]:
    """Anchored members with the features, parameters and rounds of their plain member (only the missing ones)."""
    cb_columns, cb_categorical, lgb_columns = columns
    has = np.isfinite(anchor)
    residual = y - np.nan_to_num(anchor)
    wanted = set(anchor_models(config, bool(extra)))
    entries = []
    for group in GROUPS:
        rows = np.flatnonzero((groups == group) & measured & has)
        wx = wx_columns(X, config, group)
        members = [(f"{group}_taxi_anchor.cbm", rows, cb_columns + wx, "cb"),
                   (f"{group}_taxi_adsb_anchor.cbm", rows, cb_columns + extra + wx, "cb")]
        for name in [group] + list(GROUPS[group]):
            member_rows = rows if name == group else rows[airports.iloc[rows].to_numpy() == name]
            members += [(f"{name}_taxi_lgb_anchor.txt", member_rows, lgb_columns + wx, "lgb"),
                        (f"{name}_taxi_lgb_adsb_anchor.txt", member_rows, lgb_columns + wx + extra, "lgb")]
        for name, member_rows, member_columns, kind in members:
            target = out / name
            if name not in wanted or target.exists():
                continue
            start = time.time()
            if kind == "cb":
                model = CatBoostRegressor(**grouped_params(config, seed, config.grouped_iterations[group]))
                model.fit(Pool(X.iloc[member_rows][member_columns], residual[member_rows], cat_features=cb_categorical))
                model.save_model(str(target))
            else:
                fit_lgb(X.iloc[member_rows], residual[member_rows], seed, config.grouped_threads, lgb_rounds[group],
                        member_columns).save_model(str(target))
            log.info("grouped s%d %s: %d rows with AOBT_3, %.0f s", seed, name, len(member_rows), time.time() - start)
        entries.append({"group": group, "rows_with_aobt3": int(len(rows))})
    return entries


def train(config: PipelineConfig, seed: int, X: pd.DataFrame, y: np.ndarray, airports: pd.Series,
          imputed: np.ndarray, anchor: np.ndarray | None = None) -> Path:
    """``X``, ``y``, ``airports`` and ``imputed``: nine-airport departures in file order; ``anchor`` only with
    anchored members."""
    out = model_dir(config, seed)
    _check_anchor(config)
    if uses_anchor(config) and anchor is None:
        raise ValueError("The anchored members need each departure's anchor (grouped.anchor_of)")
    airport_weight = config.grouped_airport_lgb_weight
    if not 0 <= airport_weight <= 1 or (airport_weight and not config.grouped_lgb_weight):
        raise ValueError("The per-airport blend needs LightGBM and a weight between 0 and 1")
    cb_weight, lgb_adsb = config.grouped_airport_cb_weight, config.grouped_lgb_adsb
    if not 0 <= cb_weight <= 1:
        raise ValueError("The per-airport CatBoost weight must be between 0 and 1")
    if lgb_adsb and not config.grouped_lgb_weight:
        raise ValueError("The ADS-B LightGBM needs LightGBM")
    airport_models_ready = (not airport_weight or all(
        (out / f"{airport}_taxi_lgb.txt").exists() for airport in OTHER_AIRPORTS)) and (not cb_weight or all(
        (out / f"{airport}_taxi.cbm").exists() for airport in OTHER_AIRPORTS)) and (
        not lgb_adsb or config.adsb_features is None or all(
            (out / f"{group}_taxi_lgb_adsb.txt").exists() for group in GROUPS)) and all(
        (out / name).exists() for name in anchor_models(config, bool(adsb_columns(X))))
    if (out / "manifest.json").exists() and airport_models_ready:
        log.info("grouped s%d already trained: %s", seed, out)
        return out
    if set(airports.unique()) != set(OTHER_AIRPORTS):
        raise ValueError("The grouped mixture is trained on exactly the nine non-LIRF airports")
    out.mkdir(parents=True, exist_ok=True)
    groups = group_of(airports)
    manifest = {"seed": seed, "groups": {g: list(a) for g, a in GROUPS.items()}, "models": [],
                "label_max_sec": config.grouped_label_max_sec, "lgb_weight": config.grouped_lgb_weight,
                "airport_lgb_weight": airport_weight, "airport_cb_weight": cb_weight, "lgb_adsb": lgb_adsb,
                "anchor_cb_weight": config.grouped_anchor_cb_weight,
                "anchor_lgb_weight": config.grouped_anchor_lgb_weight}
    extra = adsb_columns(X)
    if extra and config.grouped_lgb_weight and not has_context(X):
        raise ValueError("The ADS-B regressor's LightGBM is only validated together with the pre-takeoff context")
    cb_columns, cb_categorical, lgb_columns = regressor_columns(X)
    lgb_rounds = config.grouped_context_lgb_rounds if has_context(X) else config.grouped_lgb_rounds
    manifest.update(regressor_features=cb_columns, lgb_features=lgb_columns)
    kept = ~(y > config.grouped_label_max_sec) if config.grouped_label_max_sec is not None else np.ones(len(y), bool)
    for group in GROUPS:
        rows = np.flatnonzero((groups == group) & ~imputed & kept)
        iterations = config.grouped_iterations[group]
        wx = wx_columns(X, config, group)
        variants = [("", cb_columns + wx)] + ([("_adsb", cb_columns + extra + wx)] if extra else [])
        for suffix, columns in variants:
            target = out / f"{group}_taxi{suffix}.cbm"
            if not target.exists():
                start = time.time()
                model = CatBoostRegressor(**grouped_params(config, seed, iterations))
                model.fit(Pool(X.iloc[rows][columns], y[rows], cat_features=cb_categorical))
                model.save_model(str(target))
                log.info("grouped s%d %s%s: %d measured rows, %d trees, %.0f s", seed, group, suffix, len(rows),
                         iterations, time.time() - start)
        if cb_weight:
            for airport in GROUPS[group]:
                airport_rows = rows[airports.iloc[rows].to_numpy() == airport]
                for suffix, columns in variants:
                    if suffix and config.adsb_airports is not None and airport not in config.adsb_airports:
                        continue
                    target = out / f"{airport}_taxi{suffix}.cbm"
                    if not target.exists():
                        start = time.time()
                        model = CatBoostRegressor(**grouped_params(config, seed, iterations))
                        model.fit(Pool(X.iloc[airport_rows][columns], y[airport_rows], cat_features=cb_categorical))
                        model.save_model(str(target))
                        log.info("grouped s%d %s%s: %d measured rows, %d trees, %.0f s", seed, airport, suffix,
                                 len(airport_rows), iterations, time.time() - start)
        entry = {"group": group, "rows_measured": int(len(rows)), "iterations": iterations,
                 "rows_with_adsb": int(has_adsb(X.iloc[rows]).sum()) if extra else 0,
                 "rows_dropped_label_max": int(((groups == group) & ~imputed & ~kept).sum()), "wx_features": wx}
        target = out / f"{group}_taxi_lgb.txt"
        if config.grouped_lgb_weight and not target.exists():
            start = time.time()
            rounds = lgb_rounds[group]
            booster = fit_lgb(X.iloc[rows], y[rows], seed, config.grouped_threads, rounds, lgb_columns + wx)
            booster.save_model(str(target))
            log.info("grouped s%d %s LightGBM: %d rows, %d rounds, %.0f s", seed, group, len(rows), rounds,
                     time.time() - start)
        if config.grouped_lgb_weight:
            entry["lgb_rounds"] = lgb.Booster(model_file=str(target)).current_iteration()
        if lgb_adsb and extra:
            target = out / f"{group}_taxi_lgb_adsb.txt"
            if not target.exists():
                start = time.time()
                booster = fit_lgb(X.iloc[rows], y[rows], seed, config.grouped_threads, lgb_rounds[group],
                                  lgb_columns + wx + extra)
                booster.save_model(str(target))
                log.info("grouped s%d %s LightGBM ADS-B: %d rows, %d rounds, %.0f s", seed, group, len(rows),
                         lgb_rounds[group], time.time() - start)
            if airport_weight:
                for airport in GROUPS[group]:
                    if config.adsb_airports is not None and airport not in config.adsb_airports:
                        continue
                    airport_rows = rows[airports.iloc[rows].to_numpy() == airport]
                    target = out / f"{airport}_taxi_lgb_adsb.txt"
                    if not target.exists():
                        start = time.time()
                        booster = fit_lgb(X.iloc[airport_rows], y[airport_rows], seed, config.grouped_threads,
                                          lgb_rounds[group], lgb_columns + wx + extra)
                        booster.save_model(str(target))
                        log.info("grouped s%d %s LightGBM ADS-B: %d rows, %d rounds, %.0f s", seed, airport,
                                 len(airport_rows), lgb_rounds[group], time.time() - start)
        if airport_weight:
            entry["airport_models"] = []
            for airport in GROUPS[group]:
                airport_rows = rows[airports.iloc[rows].to_numpy() == airport]
                target = out / f"{airport}_taxi_lgb.txt"
                if not target.exists():
                    start = time.time()
                    booster = fit_lgb(X.iloc[airport_rows], y[airport_rows], seed, config.grouped_threads,
                                      lgb_rounds[group], lgb_columns + wx)
                    booster.save_model(str(target))
                    log.info("grouped s%d %s LightGBM: %d rows, %d rounds, %.0f s", seed, airport,
                             len(airport_rows), lgb_rounds[group], time.time() - start)
                entry["airport_models"].append({"airport": airport, "rows_measured": int(len(airport_rows)),
                                               "lgb_rounds": lgb_rounds[group]})
        manifest["models"].append(entry)
    if uses_anchor(config):
        manifest["anchor"] = _train_anchored(config, seed, out, X, y, airports, groups, ~imputed & kept, anchor,
                                             (cb_columns, cb_categorical, lgb_columns), extra, lgb_rounds)
    target = out / "classifier.cbm"
    if not target.exists():
        start = time.time()
        classifier = CatBoostClassifier(**grouped_params(config, seed, config.grouped_classifier_iterations,
                                                         classifier=True))
        classifier.fit(Pool(X[FEATURES], imputed.astype(int), cat_features=CATEGORICAL))
        classifier.save_model(str(target))
        log.info("grouped s%d classifier: %d rows, %.0f s", seed, len(X), time.time() - start)
    manifest.update(adsb_features=extra, classifier_rows=int(len(X)),
                    classifier_iterations=config.grouped_classifier_iterations,
                    imputed_rate=float(imputed.mean()),
                    params=grouped_params(config, seed, 0))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def predict(config: PipelineConfig, seed: int, frame: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Per-flight prediction with its pieces: taxi, probability, delay and blend."""
    airport_weight = config.grouped_airport_lgb_weight
    if not 0 <= airport_weight <= 1 or (airport_weight and not config.grouped_lgb_weight):
        raise ValueError("The per-airport blend needs LightGBM and a weight between 0 and 1")
    if not 0 <= config.grouped_airport_cb_weight <= 1:
        raise ValueError("The per-airport CatBoost weight must be between 0 and 1")
    _check_anchor(config)
    out = model_dir(config, seed)
    groups = group_of(frame["airport"].astype(str))
    extra = adsb_columns(X)
    covered = has_adsb(X) if extra else np.zeros(len(X), bool)
    if config.adsb_airports is not None:
        covered = covered & frame["airport"].astype(str).isin(config.adsb_airports).to_numpy()
    cb_base, cb_adsb = np.full(len(X), np.nan), np.full(len(X), np.nan)
    for group in GROUPS:
        rows = groups == group
        if rows.any():
            model = CatBoostRegressor()
            model.load_model(str(out / f"{group}_taxi.cbm"))
            cb_base[rows] = model.predict(X[rows][model.feature_names_])
        if (rows & covered).any():
            model = CatBoostRegressor()
            model.load_model(str(out / f"{group}_taxi_adsb.cbm"))
            cb_adsb[rows & covered] = model.predict(X[rows & covered][model.feature_names_])
    airports = frame["airport"].astype(str).to_numpy()
    if config.grouped_airport_cb_weight:
        weight = config.grouped_airport_cb_weight
        for airport in OTHER_AIRPORTS:
            rows = airports == airport
            if rows.any():
                model = CatBoostRegressor()
                model.load_model(str(out / f"{airport}_taxi.cbm"))
                cb_base[rows] = (1 - weight) * cb_base[rows] + weight * model.predict(X[rows][model.feature_names_])
            if (rows & covered).any():
                model = CatBoostRegressor()
                model.load_model(str(out / f"{airport}_taxi_adsb.cbm"))
                both = rows & covered
                cb_adsb[both] = (1 - weight) * cb_adsb[both] + weight * model.predict(X[both][model.feature_names_])
    cb = np.where(covered, cb_adsb, cb_base)
    anchor = anchor_of(frame) if uses_anchor(config) else None
    use_cb_anchor = use_lgb_anchor = np.zeros(len(X), bool)
    if config.grouped_anchor_cb_weight:
        # Anchored CatBoost (ADS-B version on ADS-B regressor rows) inside the CatBoost term.
        offset = np.full(len(X), np.nan)
        for group in GROUPS:
            rows = groups == group
            for suffix, part in [("", rows & ~covered), ("_adsb", rows & covered)]:
                if part.any():
                    model = CatBoostRegressor()
                    model.load_model(str(out / f"{group}_taxi{suffix}_anchor.cbm"))
                    offset[part] = model.predict(X[part][model.feature_names_])
        use_cb_anchor = _in_months(frame, config.grouped_anchor_cb_months) & np.isfinite(anchor)
        weight = config.grouped_anchor_cb_weight
        cb = np.where(use_cb_anchor, (1 - weight) * cb + weight * (anchor + offset), cb)
    if config.grouped_lgb_weight:
        taxi_lgb = np.full(len(X), np.nan)
        for group in GROUPS:
            rows = groups == group
            if rows.any():
                taxi_lgb[rows] = predict_lgb(lgb.Booster(model_file=str(out / f"{group}_taxi_lgb.txt")), X[rows])
        if airport_weight:
            for airport in OTHER_AIRPORTS:
                rows = frame["airport"].astype(str).to_numpy() == airport
                if rows.any():
                    airport_taxi = predict_lgb(lgb.Booster(model_file=str(out / f"{airport}_taxi_lgb.txt")), X[rows])
                    taxi_lgb[rows] = (1 - airport_weight) * taxi_lgb[rows] + airport_weight * airport_taxi
        taxi_lgb_covered = taxi_lgb
        if config.grouped_lgb_adsb and covered.any():
            # On ADS-B rows the LightGBM also sees the ADS-B features (like the ADS-B CatBoost).
            taxi_lgb_adsb = np.full(len(X), np.nan)
            for group in GROUPS:
                rows = (groups == group) & covered
                if rows.any():
                    taxi_lgb_adsb[rows] = predict_lgb(
                        lgb.Booster(model_file=str(out / f"{group}_taxi_lgb_adsb.txt")), X[rows])
            if airport_weight:
                for airport in OTHER_AIRPORTS:
                    rows = (airports == airport) & covered
                    if rows.any():
                        airport_taxi = predict_lgb(
                            lgb.Booster(model_file=str(out / f"{airport}_taxi_lgb_adsb.txt")), X[rows])
                        taxi_lgb_adsb[rows] = (1 - airport_weight) * taxi_lgb_adsb[rows] + airport_weight * airport_taxi
            taxi_lgb_covered = np.where(covered, taxi_lgb_adsb, taxi_lgb)
        if config.grouped_anchor_lgb_weight:
            # Anchored LightGBM (group and per airport, ADS-B version where the term has it).
            def anchored(suffix, mask):
                values = np.full(len(X), np.nan)
                for group in GROUPS:
                    rows = (groups == group) & mask
                    if rows.any():
                        values[rows] = predict_lgb(
                            lgb.Booster(model_file=str(out / f"{group}_taxi_lgb{suffix}_anchor.txt")), X[rows])
                if airport_weight:
                    for airport in OTHER_AIRPORTS:
                        rows = (airports == airport) & mask
                        if rows.any():
                            airport_taxi = predict_lgb(
                                lgb.Booster(model_file=str(out / f"{airport}_taxi_lgb{suffix}_anchor.txt")), X[rows])
                            values[rows] = (1 - airport_weight) * values[rows] + airport_weight * airport_taxi
                return values

            offset = anchored("", np.ones(len(X), bool))
            if config.grouped_lgb_adsb and covered.any():
                offset = np.where(covered, anchored("_adsb", covered), offset)
            use_lgb_anchor = _in_months(frame, config.grouped_anchor_lgb_months) & np.isfinite(anchor)
            weight = config.grouped_anchor_lgb_weight
            taxi_lgb_covered = np.where(use_lgb_anchor, (1 - weight) * taxi_lgb_covered + weight * (anchor + offset),
                                        taxi_lgb_covered)

        def blend(catboost, lgb_term):
            mixed = (1 - config.grouped_lgb_weight) * catboost + config.grouped_lgb_weight * lgb_term
            return np.where(np.abs(lgb_term - catboost) > (config.grouped_lgb_guard_sec or np.inf), catboost, mixed)

        taxi_base, taxi = blend(cb_base, taxi_lgb), blend(cb, taxi_lgb_covered)
    else:
        taxi_base, taxi = cb_base, cb
    classifier = CatBoostClassifier()
    classifier.load_model(str(out / "classifier.cbm"))
    probability = classifier.predict_proba(X[FEATURES])[:, 1]
    delay = X[DELAY].to_numpy(dtype=float)
    taxi = capped_taxi(taxi, delay, config.grouped_taxi_margin_sec)
    result = pd.DataFrame({ID: frame[ID].to_numpy(), "group": groups, "taxi": taxi, "taxi_v26": taxi_base,
                           "adsb": covered, "probability": probability, "delay": delay,
                           "anchor_cb": use_cb_anchor, "anchor_lgb": use_lgb_anchor})
    result["prediction"] = combine(taxi, delay, probability)
    if not np.isfinite(result.prediction).all():
        raise ValueError("Non-finite predictions in the grouped mixture")
    return result
