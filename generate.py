"""Final assembly v86 -> v87: the stand-quality LightGBM replaces the M2r LightGBM term (any ranking; optionally
verifies the historical v87)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ID, PRED as TARGET, ROOT, require_ids, sha256 as digest


def assemble(template, baseline, original, quality, movements, events, recipe):
    if template.empty:
        raise ValueError("Template must not be empty")
    for name, frame in (("template", template), ("baseline", baseline),
                        ("original", original), ("quality", quality), ("events", events)):
        require_ids(frame, name)
    if list(baseline.columns) != [ID, TARGET]:
        raise ValueError("Baseline must contain exactly ID and prediction columns")
    if not baseline[ID].equals(template[ID]):
        raise ValueError("Baseline IDs, types or order differ from template")
    if not np.isfinite(baseline[TARGET]).all() or baseline[TARGET].lt(0).any():
        raise ValueError("Invalid baseline predictions")
    if set(original[ID]) != set(quality[ID]):
        raise ValueError("Corrector coverage differs")
    if not set(original[ID]).issubset(set(template[ID])):
        raise ValueError("Corrector contains IDs outside template")
    for frame, columns in ((original, ["m2", "m2_lgb"]), (quality, ["m2_lgb"])):
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError("Invalid corrector predictions")
    departures = movements.loc[movements.PHASE_mvt.eq("DEP")].copy()
    require_ids(departures, "departures")
    if not set(template[ID]).issubset(set(departures[ID])):
        raise ValueError("Ranking is missing template departures")
    departures = departures.set_index(ID).loc[template[ID]]
    if departures.ADEP_mvt.isna().any():
        raise ValueError("Missing departure airport")
    old = original.set_index(ID)
    quality_lgb = quality.set_index(ID).m2_lgb.reindex(old.index)
    hybrid = old.m2 + recipe["lightgbm_weight"] * (quality_lgb - old.m2_lgb)
    old_values = old.m2.reindex(template[ID]).to_numpy()
    new_values = hybrid.reindex(template[ID]).to_numpy()
    allowed = np.isfinite(old_values) & np.isfinite(new_values)
    allowed &= template[ID].isin(events[ID]).to_numpy()
    allowed &= departures.ADEP_mvt.ne(recipe["protected_airport"]).to_numpy()
    gap = (pd.to_datetime(departures.MVT_TIME_UTC_mvt, utc=True)
           - pd.to_datetime(departures.LOBT_flt, utc=True)).dt.total_seconds().to_numpy()
    window = recipe["lobt_window_seconds"]
    candidate = baseline.copy()
    previous = baseline.loc[allowed, TARGET].to_numpy()
    proposed = np.maximum(previous + recipe["ensemble_weight"] * recipe["corrector_weight"]
                          * (new_values[allowed] - old_values[allowed]), 0)
    # The label lies in the LOBT window (|BLOCK - LOBT| <= window): the correction stops at its bound, and a row whose
    # window is empty keeps the baseline. No row of the historical v87 reaches either case.
    lower, upper = np.maximum(gap[allowed] - window, 0), gap[allowed] + window
    bounded = np.isfinite(gap[allowed])
    empty = bounded & (lower > upper)
    corrected = np.clip(np.where(empty, previous, proposed),
                        np.where(bounded & ~empty, lower, -np.inf), np.where(bounded & ~empty, upper, np.inf))
    window_clipped = int((corrected != proposed).sum())
    candidate.loc[allowed, TARGET] = corrected
    changed = candidate[TARGET].to_numpy() != baseline[TARGET].to_numpy()
    values = candidate[TARGET].to_numpy()
    violations = changed & np.isfinite(gap) & (
        (values < gap - window - 1e-6) | (values > gap + window + 1e-6))
    if violations.any():
        raise ValueError(f"LOBT window violations: {violations.sum()}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid candidate predictions")
    delta = values - baseline[TARGET].to_numpy()
    energy = np.sort(np.square(delta))[::-1]
    total = energy.sum()
    report = {
        "rows": len(candidate), "changed_rows": int(changed.sum()),
        "protected_rows": int((~allowed).sum()), "window_violations": int(violations.sum()),
        "window_clipped_rows": window_clipped,
        "rms_change": float(np.sqrt(np.square(delta).mean())),
        "top100_change_energy_share": float(energy[:100].sum() / total) if total else 0.0,
    }
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "baseline", "original", "quality", "ranking", "events"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/submission.parquet")
    parser.add_argument("--verify-reference", action="store_true",
                        help="Require the exact historical v86 input and v87 output hashes")
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Output already exists; choose another --output")
    recipe = json.loads((ROOT / "config/recipe.json").read_text())
    if args.verify_reference and digest(args.baseline) != recipe["baseline_sha256"]:
        raise ValueError("This reconstruction requires the frozen v86 baseline")
    paths = {name: getattr(args, name) for name in
             ("template", "baseline", "original", "quality", "ranking", "events")}
    inputs = {name: digest(path) for name, path in paths.items()}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    candidate, report = assemble(
        frames["template"], frames["baseline"], frames["original"], frames["quality"],
        frames["ranking"], frames["events"], recipe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(".pending.parquet")
    if pending.exists():
        raise FileExistsError(pending)
    candidate.to_parquet(pending, index=False)
    pd.testing.assert_frame_equal(candidate, pd.read_parquet(pending))
    output_hash = digest(pending)
    if args.verify_reference and output_hash != recipe["expected_sha256"]:
        raise ValueError(f"Reconstruction differs from v87: {output_hash}; retained {pending}")
    args.output.hardlink_to(pending)
    pending.unlink()
    report.update(recipe="v87", sha256=output_hash, inputs=inputs, upload=False,
                  historical_reference_verified=args.verify_reference)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()