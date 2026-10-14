"""Stand-quality corrector: the M2r LightGBM (tag c) plus stand-geometry quality features.

    python -m components.quality.quality --mode ranking --seed 11 --rounds 2500 --threads 12
        # -> components/quality/pred_ranking_quality_s11_2500.parquet (m2_lgb), used by generate.py (v87)

The quality features (components/m2r/stand_features.quality_features) describe how well the stand position is known
from other UTC-day folds of the training months; no departure label enters them. ``holdout`` trains on 2025 without
January and July and predicts those two months.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CANDIDATES, ID, require_ids, sha256

HERE = Path(__file__).resolve().parent
BASELINE_SUBMISSION = CANDIDATES / "v87_rc_037_s32.parquet"
BASELINE_SHA256 = "8324c6bcd6131681a106396e485f69d73ae2e9e73aee4202810e881690bc7940"
POSITION_COLUMNS = [ID, "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "adsb_lat0", "adsb_lon0", "adsb_gs0"]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(mode: str, seed: int, rounds: int, threads: int, prediction_threads: int, out: Path) -> Path:
    # m2r reads its configuration from the environment when imported: LightGBM only, stand features on.
    os.environ.update(M2_THREADS=str(threads), M2_SEED=str(seed), M2R_STAND="1", M2R_NO_CB="1")
    from components.m2r import m2r, stand_features

    path = out / f"pred_{mode}_quality_s{seed}_{rounds}.parquet"
    if path.exists() or path.with_name(path.stem + "_lgb.parquet").exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {path}")
    started = time.monotonic()
    # The v86 hash only identifies the historical experiment; the model never reads v86. A new dataset has
    # another v86, so the exact hash is enforced only when reproducing the historical reference.
    baseline_sha256 = sha256(BASELINE_SUBMISSION) if BASELINE_SUBMISSION.exists() else None
    if os.environ.get("PRC_VERIFY_REFERENCE") == "1" and baseline_sha256 != BASELINE_SHA256:
        raise ValueError("The agreed v86 baseline has changed")
    all_training = m2r.load(False)
    require_ids(all_training, "training")
    if mode == "holdout":
        held = all_training.MVT_TIME_UTC_mvt.dt.month.isin([1, 7])
        training, testing = all_training[~held].copy(), all_training[held].copy()
    else:
        training, testing = all_training.copy(), m2r.load(True)
    del all_training
    query = pd.concat([training[POSITION_COLUMNS], testing[POSITION_COLUMNS]], ignore_index=True)
    quality = stand_features.quality_features(query, training)
    quality.to_parquet(out / f"quality_{mode}.parquet", index=False)
    training = training.merge(quality, on=ID, how="left", validate="one_to_one")
    testing = testing.merge(quality, on=ID, how="left", validate="one_to_one")
    del query, quality
    metadata = {"features": stand_features.QUALITY_COLUMNS, "seed": seed, "threads": threads,
                "prediction_threads": prediction_threads, "lgb_rounds": rounds, "mode": mode,
                "train_rows": len(training), "test_rows": len(testing),
                "reference_policy": "training months only; other UTC-day folds for stand quality",
                "baseline_submission": {"path": str(BASELINE_SUBMISSION), "sha256": baseline_sha256}}
    write_json(path.with_suffix(".json"), metadata)
    result = m2r.fit_predict(training.reset_index(drop=True), testing.reset_index(drop=True), path, lgb_rounds=rounds,
                             checkpoint=path.with_suffix(".lgb.txt"), prediction_threads=prediction_threads)
    require_ids(result, "prediction")
    if not np.isfinite(result.m2).all():
        raise ValueError("Invalid model predictions")
    result.to_parquet(path, index=False)
    metadata.update(seconds=time.monotonic() - started, sha256=sha256(path))
    write_json(path.with_suffix(".json"), metadata)
    print(f"Saved {path}: {metadata['seconds']:.1f} s", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["holdout", "ranking"], default="ranking")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=2500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    if min(args.threads, args.prediction_threads, args.rounds) <= 0:
        parser.error("threads and rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    train(args.mode, args.seed, args.rounds, args.threads, args.prediction_threads, args.out)


if __name__ == "__main__":
    main()
