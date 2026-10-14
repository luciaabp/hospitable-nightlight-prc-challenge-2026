"""Submission: every row from models trained on 2025 data or from rules written as code.

No frozen predictions and no leaderboard-derived corrections.

| Rows | Prediction |
|---|---|
| EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LSZH, LTFM | airport specialist, capped by delay |
| LFPG with NM | LFPG specialist, capped by delay |
| LFPG without NM | pooled no-NM expert, transfer hybrid, capped by delay |
| FCO with NM, delay <= 12 h | FCO specialist with the delay-weighted copied-block classifier |
| FCO, other rows | structured FCO model retrained from the repository code (gate3_quality, seeds 42 and 1) |
| FCO without NM, delay <= 12 h, July 2026 | July cap on the structured model (``--july-cap``, on by default) |
| FCO without NM, delay > 12 h | critical-flight rule, dominant mode, 2025 mean taxi for the day candidate |
| every airport except FCO | LOBT bounds |

Inputs are the saved final models (see ``src/modeling/specialist/README.md``); no GPU is needed.

    python -m src.evaluation.specialist.submission --output outputs/submissions/specialist_v49.parquet \
        --expect data/submissions/submission_v62.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.evaluation.build_critical_fco_submission import critical_departures, replacement
from src.evaluation.build_fco_july_calibrated_submission import calibrated_components, july_noncritical_mask
from src.evaluation.specialist.lobt_bounds import apply_bounds, lobt_lead, nm_proxy
from src.evaluation.specialist.policy import cap_by_delay
from src.feature_engineering.record_quality import VISIBLE_COLUMNS
from src.modeling.predict_structured import StructuredPredictor
from src.modeling.specialist.config import AIRPORTS, FCO, ID, RAW_DIR, SEEDS, TARGET
from src.modeling.specialist.features import load_airport
from src.modeling.specialist.no_nm import ranking_pool
from src.modeling.specialist.train import combine, load_models, predict, specialist

DEFAULTS = {
    "specialist_models": Path("outputs/final_models/specialist"),
    "specialist_weights": Path("configs/specialist/weights.json"),
    "fco_models": Path("outputs/final_models/specialist_fco"),
    "no_nm_models": Path("outputs/final_models/specialist_no_nm"),
    "structured_models": [Path("outputs/final_models/structured_fco_d6_s42_retrained"),
                          Path("outputs/final_models/structured_fco_d6_s1_retrained")],
    "structured_ranking": Path("local_data/interim_nm/ranking.parquet"),
}
MAX_DELAY_SEC = 43200.0
CRITICAL_MODE = "dominant"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_hybrid(root: Path, features: pd.DataFrame, meta: pd.DataFrame, seeds, prefix="") -> pd.Series:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if len(meta) != len(features) or meta[ID].isna().any() or meta[ID].duplicated().any():
        raise ValueError("Expert features need aligned, unique movement IDs")
    if meta.empty:
        return pd.Series(dtype=float, index=pd.Index([], name=ID))
    parts = []
    for seed in seeds:
        directory = root / f"s{seed}"
        taxi = CatBoostRegressor()
        classifier = CatBoostClassifier()
        taxi.load_model(str(directory / f"{prefix}taxi.cbm"))
        classifier.load_model(str(directory / f"{prefix}classifier_weighted.cbm"))
        columns = taxi.feature_names_
        if columns != classifier.feature_names_ or set(columns).difference(features.columns):
            raise ValueError("Expert model feature schema differs from ranking")
        current = features.loc[:, columns]
        taxi_values = np.maximum(taxi.predict(current), 0)
        probability = classifier.predict_proba(current)[:, 1]
        parts.append(combine(taxi_values, meta.delay.to_numpy(), probability))
    values = np.mean(parts, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite expert predictions")
    return pd.Series(values, index=meta[ID].to_numpy())


def structured_fco(models, ranking_path: Path, july_cap: bool) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Retrained structured FCO ensemble, optionally with the July cap, on the full ranking batch."""
    ranking = pd.read_parquet(ranking_path)
    predictions = []
    for folder in models:
        predictor = StructuredPredictor(folder)
        visible = list(dict.fromkeys([ID] + predictor.config.features + VISIBLE_COLUMNS))
        predictions.append(predictor.predict(ranking[visible]))
    components = calibrated_components(predictions)
    value = components["gate3_quality"].copy()
    capped = pd.Series(False, index=value.index)
    if july_cap:
        eligible = ranking.set_index(ID).pipe(july_noncritical_mask)
        ids = eligible.index[eligible]
        value.loc[ids] = components.loc[ids, "calibrated"]
        capped.loc[ids] = True
    return value, capped, ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--no-july-cap", dest="july_cap", action="store_false")
    parser.add_argument("--expect", type=Path, help="Optional submission to compare against")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=default,
                            nargs="+" if isinstance(default, list) else None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in DEFAULTS}
    if args.output.exists():
        raise FileExistsError(args.output)
    template = pd.read_parquet(RAW_DIR / "submitting.parquet")
    raw = pd.read_parquet(RAW_DIR / "ranking.parquet",
                          columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                   "AOBT_3_flt", "LOBT_flt"])
    ranking = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(template[ID])
    if ranking.ADEP_mvt.isna().any():
        raise ValueError("Template IDs missing from the ranking departures")
    airport = ranking.ADEP_mvt.to_numpy()
    delay = (ranking.MVT_TIME_UTC_mvt - ranking.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(float)
    with_nm = ranking.AOBT_3_flt.notna().to_numpy()
    values = pd.Series(np.nan, index=template[ID].to_numpy())
    source = pd.Series("", index=values.index, dtype=object)

    # 1. Specialists (9 airports; LFPG rows without NM are replaced in step 2).
    weights = json.loads(inputs["specialist_weights"].read_text())
    for name in AIRPORTS:
        if name not in airport:
            continue
        data = load_airport(name, "ranking")
        models = load_models(inputs["specialist_models"] / name, args.seeds)
        components = predict(models, data.x, data.meta.delay.to_numpy())
        pred = cap_by_delay(specialist(components, weights[name]), data.meta.delay.to_numpy())
        values.loc[data.meta[ID].to_numpy()] = pred
        source.loc[data.meta[ID].to_numpy()] = "specialist"

    # 2. LFPG without NM: pooled no-NM expert (transfer).
    lfpg_no_nm = values.index[(airport == "LFPG") & ~with_nm]
    if len(lfpg_no_nm):
        training = json.loads((inputs["no_nm_models"] / f"s{args.seeds[0]}" / "training.json").read_text())
        meta, features = ranking_pool(training["columns"])
        hybrid = predict_hybrid(inputs["no_nm_models"], features, meta, args.seeds, prefix="transfer_")
        no_nm = pd.Series(cap_by_delay(hybrid.to_numpy(), meta.delay.to_numpy()), index=hybrid.index)
        values.loc[lfpg_no_nm] = no_nm.reindex(lfpg_no_nm).to_numpy()
    source.loc[lfpg_no_nm] = "no-NM expert"

    # 3. FCO: structured model everywhere first, then the FCO specialist and the critical rule.
    structured, capped, structured_ranking = structured_fco(inputs["structured_models"], inputs["structured_ranking"],
                                                            args.july_cap)
    fco_ids = values.index[airport == FCO]
    if structured.reindex(fco_ids).isna().any():
        raise ValueError("Structured model does not cover every FCO row")
    values.loc[fco_ids] = structured.reindex(fco_ids).to_numpy()
    source.loc[fco_ids] = np.where(capped.reindex(fco_ids).to_numpy(), "structured + July cap", "structured")
    routed = values.index[(airport == FCO) & with_nm & (delay <= MAX_DELAY_SEC)]
    if len(routed):
        current = load_airport(FCO, "ranking")
        fco_specialist = predict_hybrid(inputs["fco_models"], current.x, current.meta, args.seeds)
        values.loc[routed] = fco_specialist.reindex(routed).to_numpy()
    source.loc[routed] = "FCO specialist"
    critical = critical_departures(structured_ranking)
    previous = values.reindex(critical[ID]).to_numpy()
    values.loc[critical[ID].to_numpy()] = replacement(critical, None, previous=previous, mode=CRITICAL_MODE)
    source.loc[critical[ID].to_numpy()] = "critical rule"

    # 4. LOBT bounds outside FCO.
    outside = airport != FCO
    bounded = apply_bounds(values.to_numpy(), lobt_lead(ranking), nm_proxy(ranking))
    changed_by_bounds = outside & (bounded != values.to_numpy())
    values[outside] = bounded[outside]

    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} rows without prediction")
    result = template[[ID]].copy()
    result[TARGET] = np.maximum(values.to_numpy(float), 0)
    result = result.astype(template.dtypes.to_dict())
    if not np.isfinite(result[TARGET]).all():
        raise ValueError("Non-finite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(args.output), result)
    detail = pd.DataFrame({ID: result[ID], "airport": airport, "source": source.to_numpy(), "with_nm": with_nm,
                           "delay": delay, "lobt_bounded": changed_by_bounds, TARGET: result[TARGET]})
    detail.to_parquet(args.output.with_name(args.output.stem + "_detail.parquet"), index=False)
    verification = {
        "output": str(args.output), "output_sha256": sha256(args.output), "rows": len(result),
        "july_cap": args.july_cap, "seeds": args.seeds, "lobt_bounded_rows": int(changed_by_bounds.sum()),
        "rows_by_source": detail.groupby(["airport", "source"]).size().rename("rows").reset_index().to_dict("records"),
        "inputs": {k: [str(p) for p in v] if isinstance(v, list) else str(v) for k, v in inputs.items()},
        "specialist_weights": weights,
    }
    if args.expect:
        expected = pd.read_parquet(args.expect)
        np.testing.assert_array_equal(expected[ID], result[ID])
        diff = result[TARGET].to_numpy() - expected[TARGET].to_numpy()
        verification["versus"] = {"path": str(args.expect), "rows_different": int((diff != 0).sum()),
                                  "rms_diff_by_airport": pd.Series(diff).groupby(airport).apply(
                                      lambda d: float(np.sqrt(np.mean(d ** 2)))).round(1).to_dict()}
    args.output.with_name(args.output.stem + "_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verification.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
