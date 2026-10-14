"""Experiment configuration: data, features, cross-validation, search and hardware."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.feature_engineering.aircraft_backfill import FLAG as AIRCRAFT_BACKFILL_FLAG

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# features from src/feature_engineering
CATEGORICAL_FEATURES = [
    "airport",
    "stand",
    "runway",
    "airport_stand",
    "airport_runway",
    "route",
    "airline",
    "market_segment",
    "deicing_risk",
    "visibility_band",
    "aircraft_type",
    "wake_category",
]

NUMERIC_FEATURES = [
    # delays
    "takeoff_minus_EOBT_1_flt",
    "AOBT_3_flt_minus_EOBT_1_flt",
    "takeoff_minus_SCHED_TIME_UTC_mvt",
    "takeoff_minus_AOBT_3_flt",
    # flight
    "no_flight_plan",
    # traffic
    "departures_taxiing",
    "departures_prev_30m",
    "arrivals_prev_30m",
    "recent_taxi_proxy_mean_60m",
    "runway_departures_prev_30m",
    # weather and events
    "temperature_c",
    "freezing_or_snow",
    "event_adverse_weather",
    "event_winter_ops",
    # calendar
    "sin_local_hour",
    "cos_local_hour",
    "sin_weekday",
    "cos_weekday",
    "sin_month",
    "cos_month",
    # aircraft
    "mtow",
]


# Flags of what was filled from the same flight's history (feature_engineering/nm_history.py).
# Only in local_data/interim_nm; added with --add-nm-flags.
NM_FLAG_FEATURES = ["market_segment_imputado", "wake_category_imputado", "eobt_imputado"]


@dataclass
class ExperimentConfig:
    name: str = "catboost"

    # data
    interim_dir: Path = PROJECT_ROOT / "local_data" / "interim"
    file_pattern: str = "training_*.parquet"
    target: str = "TAXI_TARGET_SEC"
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    numeric_features: list[str] = field(default_factory=lambda: list(NUMERIC_FEATURES))
    # features removed from the lists above (ablations)
    dropped_features: list[str] = field(default_factory=list)
    # train and validate on these airports only (ICAO codes); None = all
    airports: list[str] | None = None
    # exclude rows with an imputed block time (not applicable to the ranking)
    exclude_unmeasured_block: bool = False
    # fill the missing aircraft type from the OpenSky flight_list (feature_engineering/aircraft_backfill.py).
    # Available for 2025 and the two ranking months, so the fill is the same in training and prediction.
    aircraft_backfill_dir: str | None = None
    aircraft_backfill_tolerance: int = 60
    aircraft_specs_path: str = str(PROJECT_ROOT / "data" / "unique_aircraft_types.xlsx")
    # random subsample for quick tests; None = every row
    max_rows: int | None = None

    # evaluation cross-validation: "group_month" (blocks of consecutive months),
    # "months" (one fold per valid_months month) or "kfold" (random)
    cv_strategy: str = "group_month"
    n_folds: int = 4  # unused with "months"
    valid_months: list[str] = field(default_factory=lambda: ["2025-01", "2025-07"])  # only with "months"
    # search cross-validation; None = same as cv_strategy
    search_cv_strategy: str | None = None

    # hyperparameter search (0 trials = CatBoost defaults)
    n_trials: int = 30
    search_sample_frac: float = 0.3
    # fixed hyperparameters (e.g. from another experiment); if given, no search
    fixed_params: dict | None = None

    # training
    # tree cap only without search; with search, "iterations" is one more hyperparameter
    final_max_iterations: int = 5000
    early_stopping_rounds: int = 100
    fit_final_model: bool = True

    # hardware and reproducibility
    task_type: str = "GPU"
    devices: str = "0"
    seed: int = 42

    # tracking
    experiments_dir: Path = PROJECT_ROOT / "outputs" / "experiments"

    def __post_init__(self):
        if self.aircraft_backfill_dir is not None and AIRCRAFT_BACKFILL_FLAG not in self.numeric_features:
            self.numeric_features = self.numeric_features + [AIRCRAFT_BACKFILL_FLAG]
        known = set(self.categorical_features) | set(self.numeric_features)
        unknown = sorted(set(self.dropped_features) - known)
        if unknown:
            raise ValueError(f"Features to remove that do not exist: {unknown}")
        self.categorical_features = [c for c in self.categorical_features if c not in self.dropped_features]
        self.numeric_features = [c for c in self.numeric_features if c not in self.dropped_features]

    @property
    def features(self) -> list[str]:
        return self.categorical_features + self.numeric_features

    def to_dict(self) -> dict:
        data = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        data["features"] = self.features
        return data
