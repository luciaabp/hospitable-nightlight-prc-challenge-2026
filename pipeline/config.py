"""Single source of truth: inputs, features, models and what goes into the submission.

Every value is fixed by an earlier validation; changing one makes a different model that must be validated again.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import ROOT, SUBMISSIONS

ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"
SCHED = "SCHED_TIME_UTC_mvt"
FLIGHT = "FLIGHT_mvt"
TARGET = "TAXI_TARGET_SEC"
PREDICTION = "TAXITIME_SEC_mvt"
DELAY = "takeoff_minus_SCHED_TIME_UTC_mvt"
MISSING_CATEGORY = "NA"

ROME = "LIRF"
# Airport groups learned on the months before July 2025 (experiment grouped_mixture_9_full).
GROUPS = {
    "group_1": ("EDDM", "LSZH"),
    "group_2": ("EDDF", "EHAM", "LEBL", "LEMD", "LFPG"),
    "group_3": ("EGLL", "LTFM"),
}
OTHER_AIRPORTS = tuple(sorted(a for members in GROUPS.values() for a in members))

# The 36 features of v26.
CATEGORICAL = [
    "airport", "stand", "runway", "airport_stand", "airport_runway", "route", "airline",
    "market_segment", "deicing_risk", "visibility_band", "aircraft_type", "wake_category",
]
NUMERIC = [
    "takeoff_minus_EOBT_1_flt", "AOBT_3_flt_minus_EOBT_1_flt", DELAY, "takeoff_minus_AOBT_3_flt",
    "no_flight_plan",
    "departures_taxiing", "departures_prev_30m", "arrivals_prev_30m", "recent_taxi_proxy_mean_60m",
    "runway_departures_prev_30m",
    "temperature_c", "freezing_or_snow", "event_adverse_weather", "event_winter_ops",
    "sin_local_hour", "cos_local_hour", "sin_weekday", "cos_weekday", "sin_month", "cos_month",
    "mtow",
    # NM imputation flags by flight number (nm_history)
    "market_segment_imputado", "wake_category_imputado", "eobt_imputado",
]
FEATURES = CATEGORICAL + NUMERIC

# Pre-takeoff context (pipeline/context.py), from other flights before takeoff only. Runway state goes to CatBoost
# only: LightGBM's linear leaves extrapolate with it.
CONTEXT_DELAY = ["ctx_dep_delay_med_30m", "ctx_dep_delay_med_60m", "ctx_arr_delay_med_60m",
                 "ctx_own_minus_dep_delay_60m"]
CONTEXT_RUNWAY_CATEGORICAL = ["rwy_config"]
CONTEXT_RUNWAY = ["rwy_own_arr_30m", "rwy_cross_arr_30m", "rwy_cross_dep_30m"]
CONTEXT_FEATURES = CONTEXT_DELAY + CONTEXT_RUNWAY_CATEGORICAL + CONTEXT_RUNWAY

# Extended METAR (pipeline/weather.py): only in the regressors of the wx_groups groups.
_WX_NAMES = ["dewpoint_spread_c", "humidity_pct", "wind_kt", "gust_kt", "cloud_base_ft", "precip_mm", "visibility_mi",
             "rain", "snow", "fog_mist", "freezing", "showers", "cold_moist", "low_visibility"]
WX_FEATURES = _WX_NAMES + ["sched_temperature_c"] + [f"sched_{n}" for n in _WX_NAMES]

# Record-quality context: columns visible in the ranking, never labels.
QUALITY_COLUMNS = ["airport", "airline", TIME, SCHED, "AOBT_3_flt", "AIRCRAFT_TYPE_mvt", "CALLSIGN_flt", FLIGHT]

# LIRF flight-number history for the imputed-block classifier.
FNHIST_FEATURES = ["fn_hist_imputed_rate", "fn_hist_log_n"]

# LIRF: the aircraft's previous arrival and the airline's imputed-block rate before takeoff (pipeline/inbound.py),
# in the LIRF regressor and classifiers, not in the gate.
INBOUND_FEATURES = ["inb_linked", "inb_late_vs_sched", "inb_to_takeoff", "inb_imputed", "inb_arr_delay",
                    "inb_sched_turn", "inb_same_type",
                    "al_arr_imp_24h", "al_arr_imp_24h_support", "al_arr_imp_168h", "al_arr_imp_168h_support"]
# LIRF classifiers weight each departure by 1 + min(max(d, 0), 12 h) / 30 min: the blend error grows with (d - r)^2.
INBOUND_WEIGHT_SEC = 1800.0
# The previous arrival also in the nine-airport taxi regressors (only the 7 link features).
GROUPED_INBOUND_FEATURES = [c for c in INBOUND_FEATURES if c.startswith("inb_")]
INBOUND_WEIGHT_CAP_SEC = 12 * 3600.0

# The flight's own times as explicit differences (pipeline/timeline.py), in the taxi regressors and in the LIRF
# regressor and classifiers.
TIMELINE_COLUMNS = ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt"]
TIMELINE_FEATURES = ["tl_x_min", "tl_takeoff_sec", "tl_aobt_sched", "tl_eobt_sched", "tl_lobt_sched", "tl_iobt_sched",
                     "tl_eobt_iobt", "tl_eobt_lobt", "tl_takeoff_lobt", "tl_takeoff_iobt", "tl_planned_duration",
                     "tl_aobt_minute_mod5"]

# LIRF: flight destination (airport and country, ADES_mvt); imputed blocks depend strongly on it.
LIRF_DESTINATION = ["ades", "ades_country"]

# Runway queue (state.py): STATE_FEATURES in the three nine-airport taxi regressors; STATE_ADSB_FEATURES only in the
# ADS-B regressor.
_STATE_AGGREGATES = ["prev", "last3", "queue", "rwy30"]
STATE_SEQUENCE = ["seq_dep_apt_10m", "seq_arr_dep_ratio_30m", "seq_gap_prev_rwy", "seq_gap_prev2_rwy",
                  "seq_gap_prev_mov_rwy", "seq_busy_len_rwy", "seq_busy_n_rwy", "seq_dep_rwy_5m", "seq_dep_rwy_10m",
                  "seq_dep_rwy_15m", "seq_arr_rwy_10m"]
STATE_PREVIOUS = ["prev_wake", "prev_x", "prev_same_airline", "prev_aobt_gap"]
STATE_FEATURES = STATE_SEQUENCE + STATE_PREVIOUS + [f"nm_route__{a}" for a in _STATE_AGGREGATES]
STATE_ADSB_FEATURES = [f"{m}__{a}" for m in ["adsb_rw7", "stop_rw7"] for a in _STATE_AGGREGATES]

# Batch context with earlier and later flights (forward.py), never labels or departure block times. By family;
# FORWARD_ADSB_FEATURES only in the ADS-B regressor.
FORWARD_LOAD = ["fw_dep_active_off", "fw_dep_behind_off", "fw_dep_active_tko", "fw_dep_active_off_rwy",
                "fw_dep_active_tko_rwy", "fw_arr_active_off", "fw_arr_active_tko"]
FORWARD_SEQUENCE = ["fw_gap_next_rwy", "fw_dep_rwy_next_5m", "fw_dep_rwy_next_10m", "fw_dep_rwy_next_15m",
                    "fw_dep_apt_next_10m", "fw_dep_rwy_win_10m", "fw_busy_len_total", "fw_busy_n_total",
                    "fw_busy_len_after", "fw_busy_n_after"]
FORWARD_NM = ["fw_nm_route__next", "fw_nm_route__next3", "fw_nm_route__fwd30", "fw_nm_route__behind", "fw_x_next",
              "fw_aobt_gap_next"]
FORWARD_PLAN = ["fw_sched_dep_prev_30m", "fw_sched_dep_next_15m", "fw_sched_dep_next_30m", "fw_sched_dep_next_60m",
                "fw_sched_arr_next_30m", "fw_sched_arr_next_60m"]
FORWARD_ADSB = [f"fw_{m}__{a}" for m in ["adsb_rw7", "stop_rw7"] for a in ["next", "next3", "fwd30", "behind"]]
FORWARD_DELAY = ["fw_dep_delay_med_next_30m", "fw_dep_delay_med_next_60m", "fw_arr_delay_med_next_60m",
                 "fw_own_minus_delay_next_60m"]
FORWARD_GROUPS = {"load": FORWARD_LOAD, "sequence": FORWARD_SEQUENCE, "nm": FORWARD_NM, "plan": FORWARD_PLAN,
                  "delay": FORWARD_DELAY, "adsb": FORWARD_ADSB}
FORWARD_FEATURES = FORWARD_LOAD + FORWARD_SEQUENCE + FORWARD_NM + FORWARD_PLAN + FORWARD_DELAY
FORWARD_ADSB_FEATURES = list(FORWARD_ADSB)

# Common tree hyperparameters (search of wf_global_nm_flags).
TREE_PARAMS = {
    "learning_rate": 0.07108700938420189,
    "l2_leaf_reg": 8.508449673906318,
    "random_strength": 12.92599985300091,
    "bagging_temperature": 0.9488188928055175,
    "max_ctr_complexity": 2,
    "bootstrap_type": "Bayesian",
    "border_count": 254,
}


# v26/v46 recipe for the options added later (and for --recipe-v26).
LEGACY_V26 = {"grouped_label_max_sec": None, "grouped_lgb_weight": 0.0, "grouped_taxi_margin_sec": None,
              "lirf_dayshift": "none", "lirf_mixture_months": (), "lirf_quality": "batch",
              "departures_taxiing": "batch", "lirf_inbound_features": None, "lirf_fnhist_iterations": None,
              "lirf_destination": False, "wx_features": None, "timeline_components": (),
              "grouped_inbound_features": None, "state_features": None, "grouped_airport_lgb_weight": 0.0,
              "forward_features": None, "grouped_airport_cb_weight": 0.0, "grouped_lgb_adsb": False,
              "grouped_anchor_cb_weight": 0.0, "grouped_anchor_lgb_weight": 0.0,
              "lirf_gate_month_penalty": None}

# LightGBM of the nine-airport taxi regressor.
LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.03, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "linear_tree": True, "linear_lambda": 1.0, "cat_smooth": 20, "cat_l2": 10, "min_data_per_group": 100,
    "max_cat_threshold": 64, "verbose": -1,
}


@dataclass
class PipelineConfig:
    run_name: str
    interim_dir: Path = ROOT / "local_data" / "interim_nm"
    ranking_name: str = "ranking.parquet"
    training_pattern: str = "training_*.parquet"
    template: Path = ROOT / "data" / "projects" / "prc_data_challenge" / "raw" / "submitting.parquet"
    output_root: Path = ROOT / "outputs" / "pipeline"

    # Imputed block time: |BLOCK - SCHED| <= 60 s. Supervision only, never a feature.
    imputed_tolerance_sec: float = 60.0

    # --- nine airports: grouped mixture -------------------------------------------------------
    grouped_seeds: tuple[int, ...] = (42, 1)
    grouped_depth: int = 10
    # (best iteration + 1) fold mean x final rows / mean fold rows.
    grouped_iterations: dict[str, int] = field(default_factory=lambda: {
        "group_1": 2512, "group_2": 2856, "group_3": 2114})
    grouped_classifier_iterations: int = 1744
    grouped_threads: int | None = None  # None: every core, as in v26
    # Taxi regressor: without labels > 6 h (three 2025 artifacts), blended with a linear-tree LightGBM and capped at
    # r <= max(d, 0) + 30 min.
    grouped_label_max_sec: float | None = 6 * 3600.0
    # LightGBM weight in the taxi blend (0.35). Applied at prediction: --grouped-lgb-weight.
    grouped_lgb_weight: float = 0.35
    grouped_airport_lgb_weight: float = 0.0
    # Per-airport CatBoost inside the CatBoost term (0 by default).
    grouped_airport_cb_weight: float = 0.0
    # LightGBM with the ADS-B features on rows with the ADS-B regressor.
    grouped_lgb_adsb: bool = False
    # Members anchored on AOBT_3: they learn y - clip(t - AOBT_3, 0, 2 h) on departures with AOBT_3 and blend with
    # their weight inside the CatBoost or LightGBM term; the months limit where they apply (None: all).
    grouped_anchor_cb_weight: float = 0.0
    grouped_anchor_cb_months: tuple[str, ...] | None = None
    grouped_anchor_lgb_weight: float = 0.0
    grouped_anchor_lgb_months: tuple[str, ...] | None = None
    # Fixed LightGBM rounds: mean optimal round of the validation x final rows / fold fit rows.
    grouped_lgb_rounds: dict[str, int] = field(default_factory=lambda: {
        "group_1": 520, "group_2": 466, "group_3": 581})
    # If CatBoost and LightGBM differ by more than 1 h, CatBoost only (linear leaves extrapolate out of distribution).
    grouped_lgb_guard_sec: float | None = 3600.0
    grouped_taxi_margin_sec: float | None = 1800.0

    # --- LIRF: structured mixture of three candidates (gate3_quality) --------------------------
    lirf_seeds: tuple[int, ...] = (42, 1)
    lirf_depth: int = 6
    lirf_iterations: int = 600
    lirf_threads: int = 8
    lirf_first_oof_index: int = 2  # OOF from the third month (March): two months of history at least
    gate_penalties: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    valid_months: tuple[str, ...] = ("2025-07", "2025-09", "2025-11", "2025-12")
    # Own gate penalty in lirf_gate_penalty_months; off by default (no gain in the ranking).
    lirf_gate_month_penalty: float | None = None
    lirf_gate_penalty_months: tuple[str, ...] = ("2026-07",)

    # LIRF tail with real block > 12 h after the anchor (dayshift.py): the label can only be the delay or taxi + 1 day.
    # "dominant" moves towards the mean of both; "none" keeps the gate (v26/v46).
    lirf_dayshift: str = "dominant"

    # LIRF record-quality context (quality.py): "causal", only flights already airborne; "batch" reproduces v26/v46.
    lirf_quality: str = "causal"

    # departures_taxiing (traffic.py): "causal", only departures already airborne; "batch" is the v26 column.
    departures_taxiing: str = "causal"

    # LIRF in summer (nonm.py): departures without NM delayed 1-3 h get p*d + (1 - p)*r instead of the gate (July).
    lirf_mixture_months: tuple[str, ...] = ("2026-07",)
    lirf_mixture_delay: tuple[float, float] = (3600.0, 3 * 3600.0)

    # --- flight-number history ---------------------------------------------------------------
    # Ranking months where the classifier uses the history (January 2026: 98.6 % coverage; July is not validated).
    fnhist_months: tuple[str, ...] = ("2026-01",)
    # Trees of the classifier with history (only in fnhist_months). None: lirf_iterations.
    lirf_fnhist_iterations: int | None = 1200
    # Destination (LIRF_DESTINATION) in the LIRF regressor and classifiers.
    lirf_destination: bool = True

    # --- LIRF: the aircraft's previous arrival (pipeline/inbound.py) -------------------------------
    # features.parquet from ``python -m pipeline.inbound prepare``; None: without it.
    lirf_inbound_features: Path | None = None

    # --- ADS-B features (pipeline/adsb.py) --------------------------------------------------------
    # features_takeoff.parquet: other flights' context + aircraft rotation, only in the nine-airport taxi regressors.
    # None: the v26/v46 model, without ADS-B.
    adsb_features: Path | None = None
    # Airports where the ADS-B regressor replaces the v26 one (None: all with context).
    # Applied at prediction: changing it needs no retraining.
    adsb_airports: tuple[str, ...] | None = None

    # --- pre-takeoff context (pipeline/context.py) ----------------------------------------
    # features.parquet from ``python -m pipeline.context prepare``; only in the nine-airport taxi regressors.
    context_features: Path | None = None
    # LightGBM rounds with context: same rule as grouped_lgb_rounds.
    grouped_context_lgb_rounds: dict[str, int] = field(default_factory=lambda: {
        "group_1": 508, "group_2": 746, "group_3": 795})

    # --- extended METAR (pipeline/weather.py) -----------------------------------------------------------
    # features.parquet from ``python -m pipeline.weather prepare``; only in the regressors of the wx_groups groups.
    wx_features: Path | None = None
    wx_groups: tuple[str, ...] = ("group_1",)

    # --- the flight's own times (pipeline/timeline.py) -----------------------------------------
    # Components receiving TIMELINE_FEATURES: "grouped" and "lirf"; computed from interim_nm.
    timeline_components: tuple[str, ...] = ("grouped", "lirf")

    # --- previous arrival in the nine airports (pipeline/inbound.py) ----------------------------------------------
    # File from ``python -m pipeline.inbound prepare --airports nine``; None: without it.
    grouped_inbound_features: Path | None = None

    # Runway queue in the nine-airport regressors (state.py, optional).
    state_features: Path | None = None
    state_adsb: bool = True   # False: without the ADS-B queue features

    # Batch context with later flights (forward.py, optional); "adsb" only goes to the ADS-B regressor.
    forward_features: Path | None = None
    forward_groups: tuple[str, ...] = ("load", "sequence", "nm", "plan")

    workers: int = 6
    validate: bool = True

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: dict) -> "PipelineConfig":
        # Runs older than these options are read with the recipe they were trained with.
        values = {**LEGACY_V26, **values}
        paths = {"interim_dir", "template", "output_root", "adsb_features", "context_features",
                 "lirf_inbound_features", "wx_features", "grouped_inbound_features",
                 "state_features", "forward_features"}
        tuples = {"grouped_seeds", "lirf_seeds", "gate_penalties", "valid_months", "fnhist_months", "adsb_airports",
                  "lirf_mixture_months", "lirf_mixture_delay", "wx_groups", "timeline_components",
                  "forward_groups", "lirf_gate_penalty_months", "grouped_anchor_cb_months",
                  "grouped_anchor_lgb_months"}
        return cls(**{k: Path(v) if k in paths and v is not None else tuple(v) if k in tuples and v is not None else v
                      for k, v in values.items()})


def grouped_params(config: PipelineConfig, seed: int, iterations: int, classifier: bool = False) -> dict:
    loss = "Logloss" if classifier else "RMSE"
    params = dict(TREE_PARAMS, loss_function=loss, eval_metric=loss, iterations=iterations,
                  depth=config.grouped_depth, random_seed=seed, task_type="CPU",
                  allow_writing_files=False, verbose=False)
    if config.grouped_threads:
        params["thread_count"] = config.grouped_threads
    return params


def lirf_params(config: PipelineConfig, seed: int, classifier: bool = False, fnhist: bool = False) -> dict:
    loss = "Logloss" if classifier else "RMSE"
    iterations = config.lirf_iterations
    if fnhist and config.lirf_fnhist_iterations is not None:
        iterations = config.lirf_fnhist_iterations
    return dict(TREE_PARAMS, loss_function=loss, eval_metric=loss, iterations=iterations,
                depth=config.lirf_depth, random_seed=seed, task_type="CPU", allow_writing_files=False,
                verbose=False, thread_count=config.lirf_threads)
