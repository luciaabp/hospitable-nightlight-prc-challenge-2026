"""Load saved experimental components and predict a complete visible batch.

Models are trusted local joblib files. Fold models are for reproducibility;
only a `final` bundle is trained with all available training labels.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering.record_quality import record_quality_features, record_quality_flags
from .config import ExperimentConfig
from .data import ID_COLUMN, apply_aircraft_backfill, prepare_features
from .run_structured import VARIANTS, gate_frame
from .structured_mixture import experts


class StructuredPredictor:
    def __init__(self, experiment, period="final"):
        root = Path(experiment)
        settings = json.loads((root / "config.json").read_text())["base_config"]
        settings.pop("features", None)
        self.config = ExperimentConfig(**settings)
        self.models = joblib.load(root / f"models/{period}_base.joblib")
        # Bundles predate later variants; load the gates this bundle actually has.
        self.gates = {name: joblib.load(root / f"models/{period}_{name}.joblib") for name in VARIANTS
                      if (root / f"models/{period}_{name}.joblib").exists()}
        if not self.gates:
            raise FileNotFoundError(f"No {period} gates in {root}")

    def predict(self, visible_batch):
        frame = visible_batch.reset_index(drop=True)
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError("Duplicate movement IDs")
        if self.config.airports:
            frame = frame.loc[frame.airport.isin(self.config.airports)].reset_index(drop=True)
        if frame.empty:
            raise ValueError("No rows for this model's airports")
        frame = apply_aircraft_backfill(frame, self.config)
        X = prepare_features(frame, self.config)
        quality = record_quality_features(frame)
        flags = record_quality_flags(frame)
        base = pd.DataFrame({key: self.models[key].predict(X) for key in ["taxi", "clean_taxi"]})
        classifier = self.models["classifier"]
        base["probability"] = classifier.predict_proba(X)[:, 1] if hasattr(classifier, "predict_proba") else classifier
        base["delay"] = X.takeoff_minus_SCHED_TIME_UTC_mvt
        output = frame[[ID_COLUMN]].copy()
        output[["taxi", "clean_taxi", "probability", "delay"]] = base[["taxi", "clean_taxi", "probability", "delay"]]
        candidates = experts(base.taxi, base.delay)
        output["baseline"] = candidates[:, 0]*(1-base.probability) + candidates[:, 1]*base.probability
        cleaned = experts(base.clean_taxi, base.delay)
        output["clean_baseline"] = cleaned[:, 0]*(1-base.probability) + cleaned[:, 1]*base.probability
        for name in self.gates:
            inputs = gate_frame(X, quality, flags, base, VARIANTS[name])
            output[name] = self.gates[name].predict(inputs, inputs.base_taxi, inputs.base_delay, inputs.base_probability)
        if not np.isfinite(output.drop(columns=ID_COLUMN).to_numpy()).all():
            raise ValueError("Non-finite predictions")
        return output
