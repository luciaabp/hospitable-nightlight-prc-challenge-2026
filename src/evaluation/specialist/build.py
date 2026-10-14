"""Final specialist models on all of 2025, one directory per airport.

One process per GPU, disjoint airports:

    python -m src.evaluation.specialist.build --airports LTFM EGLL EDDF LEBL LFPG \
        --device 0 --models-dir outputs/final_models/specialist
    python -m src.evaluation.specialist.build --airports EHAM LSZH EDDM LEMD \
        --device 1 --models-dir outputs/final_models/specialist

The submission is assembled from the saved models by ``submission.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.modeling.specialist.config import AIRPORTS, SEEDS
from src.modeling.specialist.features import load_airport
from src.modeling.specialist.train import save_models, train_airport
from src.modeling.specialist.validate import code_hash


def train(args):
    for airport in args.airports:
        directory = args.models_dir / airport
        if directory.exists():
            raise FileExistsError(f"{directory} exists; use a new --models-dir")
        data = load_airport(airport, "training")
        models = train_airport(data, np.ones(len(data.meta), bool), args.device, args.seeds)
        save_models(models, directory)
        info = {"airport": airport, "code_sha256": code_hash(), "rows": len(data.meta), "seeds": args.seeds,
                "trees": {str(s): {n: m.tree_count_ for n, m in ms.items()} for s, ms in models.items()}}
        (directory / "training.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airports", nargs="+", choices=AIRPORTS, default=AIRPORTS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models-dir", type=Path, required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
