"""Walk-forward validation of one airport specialist.

For each validation month and seed, the three models train only on earlier months and
predict that month. Each fold writes ``<month>_s<seed>.parquet``; a rerun resumes the
missing folds if configuration and code are unchanged.

    python -m src.modeling.specialist.validate --airport LTFM --device 0 \
        --output outputs/experiments/specialist/LTFM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import AIRPORTS, CLOCK, ID, INNER_MONTHS, MAX_TREES, PARAMS, SEEDS, TARGET, VALIDATION_MONTHS
from .features import load_airport
from .train import predict, save_models, train_airport

PACKAGE = Path(__file__).parent


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"airport": args.airport, "seeds": args.seeds, "months": VALIDATION_MONTHS,
              "inner_months": args.inner_months, "max_trees": args.max_trees, "params": PARAMS,
              "code_sha256": code_hash()}
    manifest = args.output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Output belongs to another configuration or code version")
    manifest.write_text(json.dumps(config, indent=2))
    data = load_airport(args.airport, "training")
    for seed in args.seeds:
        for month in VALIDATION_MONTHS:
            target = args.output / f"{month}_s{seed}.parquet"
            if target.exists():
                continue
            begin = time.monotonic()
            train = data.meta.month.lt(month).to_numpy()
            valid = data.meta.month.eq(month).to_numpy()
            models = train_airport(data, train, args.device, [seed], args.inner_months, args.max_trees)
            out = data.meta.loc[valid, [ID, CLOCK, "month", "delay", "missing_nm", TARGET]].reset_index(drop=True)
            components = predict(models, data.x.loc[valid], data.meta.delay[valid].to_numpy())
            for column in [f"direct_s{seed}", f"hybrid_s{seed}", f"probability_s{seed}"]:
                out[column.rsplit("_s", 1)[0]] = components[column].to_numpy()
            if not np.isfinite(out[["direct", "hybrid"]]).all().all():
                raise ValueError(f"{args.airport} {month}: non-finite predictions")
            save_models(models, args.output / "models" / f"{month}")
            out.to_parquet(target, index=False)
            rmse = lambda c: float(np.sqrt(np.mean((out[TARGET] - out[c]) ** 2)))  # noqa: E731
            print(args.airport, seed, month, "rmse direct", round(rmse("direct"), 2), "hybrid", round(rmse("hybrid"), 2),
                  "trees", [m.tree_count_ for m in models[seed].values()], round(time.monotonic() - begin, 1), "s",
                  flush=True)
    (args.output / "complete.json").write_text(json.dumps({"status": "complete", "code_sha256": config["code_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airport", choices=AIRPORTS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--inner-months", type=int, default=INNER_MONTHS)
    parser.add_argument("--max-trees", type=int, default=MAX_TREES)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
