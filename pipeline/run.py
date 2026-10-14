"""Pipeline stages: ``train`` (per-seed models, resumable) and ``predict`` (checked submission)."""
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd

from . import dayshift, fnhist, grouped, nonm, structured
from .grouped import combine
from .config import ID, LEGACY_V26, PREDICTION, ROME, ROOT, SUBMISSIONS, TARGET, PipelineConfig
from .data import (extra_columns, features, imputed_block, input_fingerprint, lirf_categorical, lirf_columns, month,
                   read_ranking, read_training, sha256, with_destination, with_forward, with_grouped_inbound,
                   with_inbound_features, state_columns, with_state)
from . import quality as record_quality
from .submission import assemble, check, compare

log = logging.getLogger(__name__)

# Historical submissions the pipeline must reproduce with its own models (controls).
REFERENCES = {
    # grouped mixture s42 + LIRF gate3_quality s42/s1; 286.4992 on the ranking
    "v26": SUBMISSIONS / "submission_v26.parquet",
    # v26 + classifier with history on LIRF January 2026
    "v46": SUBMISSIONS / "submission_v46.parquet",
}
# Comparison only (no control): earlier submissions of this pipeline, 284.9063 and 279.8098 (the best).
COMPARE_ONLY = {"v51": SUBMISSIONS / "submission_v51.parquet",
                "v54": SUBMISSIONS / "submission_v54.parquet"}


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    return {"python": platform.python_version(), "hostname": platform.node(), "git_commit": commit,
            "libraries": {name: version(name) for name in
                          ["catboost", "pandas", "numpy", "scipy", "scikit-learn", "joblib"]}}


def save_config(config: PipelineConfig) -> None:
    """Record the configuration and the input fingerprint; a run never mixes different data."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = config.run_dir / "inputs.json"
    fingerprint = input_fingerprint(config)
    if inputs.exists():
        recorded = json.loads(inputs.read_text())
        if recorded != fingerprint:
            raise RuntimeError(f"The inputs changed since the first training of {config.run_dir}; "
                               "use another --run so as not to mix models trained on different data")
    else:
        inputs.write_text(json.dumps(fingerprint, indent=1))
        (config.run_dir / "environment.json").write_text(json.dumps(environment(), indent=1))
    (config.run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=1))


def load_config(run_dir) -> PipelineConfig:
    return PipelineConfig.from_dict(json.loads((run_dir / "config.json").read_text()))


def train(config: PipelineConfig, components: list[str]) -> None:
    save_config(config)
    frame = read_training(config)
    months, y = month(frame), frame[TARGET].to_numpy(dtype=float)
    imputed = imputed_block(frame, config.imputed_tolerance_sec)
    rome = (frame["airport"] == ROME).to_numpy()
    log.info("training: %d departures, %d of LIRF, %.2f %% with imputed block", len(frame), rome.sum(),
             100 * imputed.mean())
    if "grouped" in components:
        other = with_forward(with_state(with_grouped_inbound(frame[~rome].reset_index(drop=True), config), config),
                             config)
        X = features(other, *extra_columns(config))
        anchor = grouped.anchor_of(other) if grouped.uses_anchor(config) else None
        for seed in config.grouped_seeds:
            grouped.train(config, seed, X, y[~rome], other["airport"].astype(str), imputed[~rome], anchor)
        del X, other
    if "lirf" in components:
        lirf = with_destination(with_inbound_features(frame[rome].reset_index(drop=True), config), config)
        X_lirf = features(lirf, lirf_columns(config), lirf_categorical(config))
        quality = record_quality.context(lirf, config.lirf_quality)
        structured.train(config, list(config.lirf_seeds), X_lirf, y[rome], months[rome], imputed[rome], quality,
                         fnhist.flight_keys(lirf), lirf[ID].to_numpy())
        structured.fit_month_gate(config, X_lirf, y[rome], imputed[rome], quality, lirf[ID].to_numpy())
        report = structured.validation_report(config)
        if report is not None:
            report.to_csv(config.run_dir / "lirf_validation.csv", index=False)
            log.info("LIRF walk-forward (Jul/Sep/Nov/Dec 2025):\n%s",
                     report[report.month == "all"].to_string(index=False))


def _months(months) -> str:
    return "all months" if months is None else str(list(months))


def _mean(frames: dict, column: str) -> np.ndarray:
    first = next(iter(frames.values()))
    if any(not np.array_equal(f[ID].to_numpy(), first[ID].to_numpy()) for f in frames.values()):
        raise ValueError("Seeds with different rows")
    return np.mean([f[column].to_numpy() for f in frames.values()], axis=0)


def predict(config: PipelineConfig, name: str = "submission") -> dict:
    destination = config.run_dir / f"{name}.parquet"
    if destination.exists():
        raise FileExistsError(f"{destination} already exists: use another --name")
    recorded = json.loads((config.run_dir / "inputs.json").read_text())
    training_now = {k: v for k, v in input_fingerprint(config).items() if "training_" in k}
    if any(recorded.get(k) != v for k, v in training_now.items()):
        raise RuntimeError("The training changed since the models were trained")
    template = pd.read_parquet(config.template)
    ranking = read_ranking(config)
    months = month(ranking)
    rome = (ranking["airport"] == ROME).to_numpy()
    meta = pd.DataFrame({ID: ranking[ID].to_numpy(), "airport": ranking["airport"].astype(str).to_numpy(),
                         "month": months})

    other = with_forward(with_state(with_grouped_inbound(ranking[~rome].reset_index(drop=True), config), config),
                         config)
    X_other = features(other, *extra_columns(config))
    by_seed = {seed: grouped.predict(config, seed, other, X_other) for seed in config.grouped_seeds}
    detail_other = by_seed[config.grouped_seeds[0]][[ID, "group"]].copy()
    for seed, part in by_seed.items():
        detail_other[f"grouped_s{seed}"] = part.prediction.to_numpy()
    if config.adsb_features is not None:
        first = by_seed[config.grouped_seeds[0]]
        detail_other["adsb"] = first.adsb.to_numpy()
        # Same blend with the v26 regressor: isolates the per-row effect of the ADS-B context.
        detail_other["prediction_v26_taxi"] = np.mean(
            [combine(p.taxi_v26, p.delay, p.probability) for p in by_seed.values()], axis=0)
    detail_other["prediction"] = _mean(by_seed, "prediction")

    lirf = with_destination(with_inbound_features(ranking[rome].reset_index(drop=True), config), config)
    X_lirf, lirf_months = features(lirf, lirf_columns(config), lirf_categorical(config)), months[rome]
    quality = record_quality.context(lirf, config.lirf_quality)
    by_seed_lirf = {seed: structured.predict(config, seed, lirf, X_lirf, quality, lirf_months)
                    for seed in config.lirf_seeds}
    control, with_history = _mean(by_seed_lirf, "gate3_quality"), _mean(by_seed_lirf, "gate3_quality_fnhist")
    use_history = np.isin(lirf_months, config.fnhist_months)
    detail_lirf = by_seed_lirf[config.lirf_seeds[0]][[ID, "fn_hist_imputed_rate", "fn_hist_log_n"]].copy()
    for seed, part in by_seed_lirf.items():
        detail_lirf[f"lirf_s{seed}"] = part.gate3_quality.to_numpy()
        detail_lirf[f"lirf_fnhist_s{seed}"] = part.gate3_quality_fnhist.to_numpy()
    detail_lirf["uses_fnhist"] = use_history
    gate_prediction = np.where(use_history, with_history, control)
    raw, mixture_check = nonm.apply(lirf, by_seed_lirf, gate_prediction, lirf_months, config)
    detail_lirf["uses_mixture"] = nonm.rows(lirf, X_lirf["takeoff_minus_SCHED_TIME_UTC_mvt"], lirf_months, config)
    detail_lirf["prediction_gate"] = np.maximum(gate_prediction, 0)
    lirf_prediction, dayshift_check = raw, {"mode": config.lirf_dayshift}
    if config.lirf_dayshift != "none":
        banded = config.lirf_dayshift == "banded"
        parts = dayshift.components(lirf, X_lirf, by_seed_lirf, config.fnhist_months,
                                    [structured.model_dir(config, s) for s in config.lirf_seeds], timeline=banded)
        training = read_training(config)
        table, prior = dayshift.flight_prior(training, config.imputed_tolerance_sec)
        bands = dayshift.band_prior(training, config.imputed_tolerance_sec) if banded else None
        del training
        changes = dayshift.tail(parts, raw, config.lirf_dayshift, table, prior, bands)
        lirf_prediction = raw.copy()
        lirf_prediction[changes.index] = changes.new.to_numpy()
        detail_lirf["dayshift_gap_h"] = parts.gap.to_numpy() / 3600
        detail_lirf.loc[changes.index, "dayshift_pi"] = changes.pi.to_numpy()
        columns = [ID, dayshift.FLIGHT, "nm_times_dropped", "gap", "delay", "clean_taxi", "alt", "classifier_p",
                   "empirical_p", *(["band_p"] if banded else []), "pi", "previous", "new"]
        if banded:
            dayshift_check["band_rates"] = dict(zip(["<=15.5h", "15.5-24h", ">24h"], np.round(bands, 4).tolist()))
        dayshift_check.update({"tail_prior": prior, "tail_rows": int(len(changes)),
                               "changed_rows": int((changes.new != changes.previous).sum()),
                               "changes": json.loads(changes[columns].assign(gap=changes.gap / 3600)
                                                     .round(4).to_json(orient="records"))})
    detail_lirf["prediction"] = np.maximum(lirf_prediction, 0)

    detail = meta.merge(pd.concat([detail_other, detail_lirf], ignore_index=True), on=ID, validate="1:1")
    submission = assemble(template, detail.set_index(ID).prediction)

    checks = {"rows": int(len(submission)), "lirf_rows": int(rome.sum()),
              "lirf_rows_with_fnhist": int(use_history.sum()),
              "lirf_gate_month": {"penalty": config.lirf_gate_month_penalty,
                                  "months": list(config.lirf_gate_penalty_months),
                                  "rows": int(np.isin(lirf_months, list(config.lirf_gate_penalty_months)).sum())
                                  if config.lirf_gate_month_penalty is not None else 0},
              "lirf_negative_clipped": int((lirf_prediction < 0).sum()),
              "lirf_dayshift": dayshift_check,
              "lirf_mixture": mixture_check,
              "rows_by_month": {m: int(n) for m, n in meta.month.value_counts().sort_index().items()},
              "fnhist_coverage_by_month": {m: float((detail_lirf.fn_hist_log_n[lirf_months == m] > 0).mean())
                                           for m in sorted(set(lirf_months))}}
    if config.lirf_inbound_features is not None:
        checks["lirf_inbound_linked_by_month"] = {m: float(X_lirf.inb_linked[lirf_months == m].mean())
                                                  for m in sorted(set(lirf_months))}
    if config.adsb_features is not None:
        covered = pd.Series(grouped.has_adsb(X_other))
        keys = other["airport"].astype(str).str.cat(months[~rome], sep="|")
        checks["adsb_features_coverage"] = {k: float(v) for k, v in covered.groupby(keys.to_numpy()).mean().items()}
    if config.context_features is not None:
        keys = other["airport"].astype(str).str.cat(months[~rome], sep="|").to_numpy()
        coverage = pd.DataFrame({"delay": X_other.ctx_dep_delay_med_60m.notna().to_numpy(),
                                 "crossings": X_other.rwy_cross_arr_30m.notna().to_numpy(),
                                 "rare_config": X_other.rwy_config.str.endswith("|rare").to_numpy()})
        checks["context_features_coverage"] = coverage.groupby(keys).mean().round(4).to_dict(orient="index")
    if len(by_seed) > 1:
        # Seed = f + noise of variance s^2: averaging k seeds removes s^2 (1 - 1/k) from a single seed's MSE.
        spread = pd.DataFrame({"airport": other["airport"].astype(str).to_numpy(), "variance": np.var(
            np.column_stack([p.prediction.to_numpy() for p in by_seed.values()]), axis=1, ddof=1)})
        k = len(by_seed)
        checks["grouped_seed_spread"] = {
            airport: {"seed_sd_rms": float(np.sqrt(part.variance.mean())),
                      "expected_mse_reduction": float(part.variance.mean() * (1 - 1 / k))}
            for airport, part in [("ALL", spread), *spread.groupby("airport")]}
    comparisons, controls = {}, {}
    grouped_recipe = {k: getattr(config, k) for k in LEGACY_V26 if k.startswith("grouped_")}
    if config.adsb_features is not None:
        # With ADS-B context the regressors are no longer v26's: the reconstruction does not apply.
        controls = {key: {"status": "not applicable: ADS-B context changes the grouped regressors"}
                    for key in REFERENCES}
    elif config.context_features is not None:
        controls = {key: {"status": "not applicable: pre-departure context changes the grouped regressors"}
                    for key in REFERENCES}
    elif config.lirf_inbound_features is not None or config.lirf_destination:
        controls = {key: {"status": "not applicable: LIRF uses the inbound-arrival or destination features "
                                    "(improvements 6, 8)"} for key in REFERENCES}
    elif config.lirf_quality != LEGACY_V26["lirf_quality"]:
        controls = {key: {"status": f"not applicable: LIRF quality context is {config.lirf_quality}, v26 used batch"}
                    for key in REFERENCES}
    elif config.departures_taxiing != LEGACY_V26["departures_taxiing"]:
        controls = {key: {"status": "not applicable: departures_taxiing is causal, v26 used batch"}
                    for key in REFERENCES}
    elif any(grouped_recipe[k] != LEGACY_V26[k] for k in grouped_recipe):
        controls = {key: {"status": f"not applicable: grouped recipe {grouped_recipe} is not v26's (--recipe-v26)"}
                    for key in REFERENCES}
    elif 42 in config.grouped_seeds and {42, 1} <= set(config.lirf_seeds):
        # The same models with the historical recipes must reproduce v26 and v46 exactly.
        pair = {seed: by_seed_lirf[seed] for seed in (42, 1)}
        pair_control, pair_history = _mean(pair, "gate3_quality"), _mean(pair, "gate3_quality_fnhist")
        recipes = {"v26": pair_control, "v46": np.where(lirf_months == "2026-01", pair_history, pair_control)}
        for key, lirf_values in recipes.items():
            values = pd.concat([pd.Series(by_seed[42].prediction.to_numpy(), index=other[ID]),
                                pd.Series(lirf_values, index=lirf[ID])])
            rebuilt = assemble(template, values)
            detail[f"control_{key}"] = values.loc[detail[ID]].to_numpy()
            path = REFERENCES[key]
            if not path.exists():
                controls[key] = {"reference": str(path), "status": "reference missing"}
                continue
            reference = pd.read_parquet(path).set_index(ID).loc[template[ID], PREDICTION].to_numpy()
            diff = np.abs(rebuilt[PREDICTION].to_numpy() - reference)
            controls[key] = {"reference": str(path), "reference_sha256": sha256(path),
                             "max_abs_diff": float(diff.max()), "rows_differing": int((diff > 1e-6).sum())}
    for key, path in {**REFERENCES, **COMPARE_ONLY}.items():
        if path.exists():
            comparisons[key] = compare(submission, pd.read_parquet(path), meta)

    check(submission, template)
    submission.to_parquet(destination, index=False)
    if not pd.read_parquet(destination).equals(submission):
        raise RuntimeError("Rereading the submission does not match")
    detail.to_parquet(config.run_dir / f"{name}_detail.parquet", index=False)
    validation = config.run_dir / "lirf_validation.csv"
    verification = {
        "status": "verified", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(destination), "output_sha256": sha256(destination),
        "recipe": {"departures_taxiing": config.departures_taxiing,
                   "other_airports": f"grouped mixture, mean of seeds {list(config.grouped_seeds)}, max(., 0); "
                                     f"regressor {grouped_recipe}"
                                                 + (f"; per-airport weight inside LightGBM: {config.grouped_airport_lgb_weight}"
                                                     if config.grouped_airport_lgb_weight else "")
                                     + (f"; regressors with ADS-B context from {config.adsb_features}"
                                        f" on {list(config.adsb_airports) if config.adsb_airports else 'all'}"
                                        if config.adsb_features is not None else "")
                                     + (f"; pre-takeoff context from {config.context_features} (CatBoost: "
                                        "airport delay + runway state; LightGBM: airport delay)"
                                        if config.context_features is not None else "")
                                     + (f"; runway queue from {config.state_features}: "
                                        f"{len(state_columns(config))} features"
                                        if config.state_features is not None else "")
                                     + (f"; members anchored on AOBT_3: CatBoost "
                                        f"{config.grouped_anchor_cb_weight} in {_months(config.grouped_anchor_cb_months)}"
                                        f", LightGBM {config.grouped_anchor_lgb_weight} in "
                                        f"{_months(config.grouped_anchor_lgb_months)}"
                                        if grouped.uses_anchor(config) else "")
                                     + (f"; batch context from {config.forward_features}: families "
                                        f"{list(config.forward_groups)}"
                                        if config.forward_features is not None else ""),
                   "LIRF": f"gate3_quality (quality context {config.lirf_quality}), mean of seeds "
                           f"{list(config.lirf_seeds)}; classifier with history "
                           f"by flight number in {list(config.fnhist_months)}; day-shift tail: "
                           f"{config.lirf_dayshift}; without NM with {config.lirf_mixture_delay} s delay in "
                           f"{list(config.lirf_mixture_months)}: p*d + (1 - p)*r"
                           + (f"; aircraft's previous arrival from {config.lirf_inbound_features} in regressor and "
                              "classifiers, classifiers weighted by delay"
                              if config.lirf_inbound_features is not None else "")
                           + ("; destination (airport and country) in regressor and classifiers"
                              if config.lirf_destination else "")
                           + (f"; gate with penalty {config.lirf_gate_month_penalty} in "
                              f"{list(config.lirf_gate_penalty_months)} months"
                              if config.lirf_gate_month_penalty is not None else "")},
        "inputs": {"ranking_sha256": sha256(config.interim_dir / config.ranking_name),
                   "template_sha256": sha256(config.template)},
        "checks": checks, "controls": controls, "difference_vs": comparisons,
        "lirf_validation": (pd.read_csv(validation).query("month == 'all'").to_dict("records")
                            if validation.exists() else None),
    }
    (config.run_dir / f"{name}_verification.json").write_text(json.dumps(verification, indent=1, default=str))
    return verification
