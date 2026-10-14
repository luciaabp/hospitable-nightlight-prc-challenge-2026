"""python -m pipeline {train,predict,all} --run NAME [options]

    train    train the run's missing models (resumable by seed and component)
    predict  predict the ranking and write the checked submission to outputs/pipeline/NAME/
    all      train + predict
"""
import argparse
import json
import logging
import os

# Before importing numpy: the LIRF processes already share the cores among themselves.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from pathlib import Path  # noqa: E402

from .config import LEGACY_V26, PipelineConfig  # noqa: E402
from .run import load_config, predict, train  # noqa: E402


def build_config(args) -> PipelineConfig:
    defaults = PipelineConfig(run_name=args.run)
    if args.output_root:
        defaults.output_root = Path(args.output_root)
    existing = defaults.run_dir / "config.json"
    config = load_config(defaults.run_dir) if existing.exists() else defaults
    overrides = {"grouped_seeds": args.grouped_seeds, "lirf_seeds": args.lirf_seeds,
                 "fnhist_months": args.fnhist_months, "workers": args.workers, "adsb_airports": args.adsb_airports,
                 "lirf_dayshift": args.lirf_dayshift, "lirf_mixture_months": args.lirf_mixture_months,
                 "grouped_lgb_weight": args.grouped_lgb_weight,
                 "grouped_airport_lgb_weight": getattr(args, "grouped_airport_lgb_weight", None),
                 "grouped_airport_cb_weight": getattr(args, "grouped_airport_cb_weight", None),
                 "grouped_lgb_adsb": getattr(args, "grouped_lgb_adsb", None),
                 "forward_groups": getattr(args, "forward_groups", None),
                 "departures_taxiing": getattr(args, "departures_taxiing", None),
                 "lirf_quality": getattr(args, "lirf_quality", None),
                 "lirf_gate_month_penalty": getattr(args, "lirf_gate_month_penalty", None),
                 "lirf_gate_penalty_months": getattr(args, "lirf_gate_penalty_months", None),
                 "grouped_anchor_cb_weight": getattr(args, "grouped_anchor_cb_weight", None),
                 "grouped_anchor_cb_months": getattr(args, "grouped_anchor_cb_months", None),
                 "grouped_anchor_lgb_weight": getattr(args, "grouped_anchor_lgb_weight", None),
                 "grouped_anchor_lgb_months": getattr(args, "grouped_anchor_lgb_months", None)}
    if existing.exists() and args.interim_dir and Path(args.interim_dir) != config.interim_dir:
        parser_error(f"{args.run} already uses {config.interim_dir}; use another --run for other data")
    if args.interim_dir:
        config.interim_dir = Path(args.interim_dir)
    if existing.exists() and args.adsb_features and Path(args.adsb_features) != config.adsb_features:
        parser_error(f"{args.run} already uses adsb_features={config.adsb_features}; use another --run")
    if args.adsb_features:
        config.adsb_features = Path(args.adsb_features)
    if existing.exists() and args.context_features and Path(args.context_features) != config.context_features:
        parser_error(f"{args.run} already uses context_features={config.context_features}; use another --run")
    if args.context_features:
        config.context_features = Path(args.context_features)
    if (existing.exists() and args.inbound_features
            and Path(args.inbound_features) != config.lirf_inbound_features):
        parser_error(f"{args.run} already uses lirf_inbound_features={config.lirf_inbound_features}; use another --run")
    if args.inbound_features:
        config.lirf_inbound_features = Path(args.inbound_features)
    if (existing.exists() and args.grouped_inbound_features
            and Path(args.grouped_inbound_features) != config.grouped_inbound_features):
        parser_error(f"{args.run} already uses grouped_inbound_features={config.grouped_inbound_features}; use another --run")
    if args.grouped_inbound_features:
        config.grouped_inbound_features = Path(args.grouped_inbound_features)
    if existing.exists() and args.state_features and Path(args.state_features) != config.state_features:
        parser_error(f"{args.run} already uses state_features={config.state_features}; use another --run")
    if args.state_features:
        config.state_features = Path(args.state_features)
    forward = getattr(args, "forward_features", None)
    if existing.exists() and forward and Path(forward) != config.forward_features:
        parser_error(f"{args.run} already uses forward_features={config.forward_features}; use another --run")
    if forward:
        config.forward_features = Path(forward)
    if existing.exists() and args.wx_features and Path(args.wx_features) != config.wx_features:
        parser_error(f"{args.run} already uses wx_features={config.wx_features}; use another --run")
    if args.wx_features:
        config.wx_features = Path(args.wx_features)
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, tuple(value) if isinstance(value, list) else value)
    if args.no_validate:
        config.validate = False
    if args.recipe_v26:
        if existing.exists() and any(getattr(config, k) != v for k, v in LEGACY_V26.items() if k.startswith("grouped_")):
            parser_error(f"{args.run} was trained with another regressor recipe; use another --run")
        for key, value in LEGACY_V26.items():
            setattr(config, key, value)
    return config


def parser_error(message):
    raise SystemExit(f"error: {message}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["train", "predict", "all"])
    parser.add_argument("--run", required=True, help="Run name: outputs/pipeline/<run>/")
    parser.add_argument("--components", nargs="+", choices=["grouped", "lirf"], default=["grouped", "lirf"],
                        help="What to train (default all); useful to split over two terminals")
    parser.add_argument("--grouped-seeds", nargs="+", type=int)
    parser.add_argument("--lirf-seeds", nargs="+", type=int)
    parser.add_argument("--fnhist-months", nargs="*", help="Ranking months with flight history (default 2026-01)")
    parser.add_argument("--workers", type=int, help="LIRF processes (8 threads each)")
    parser.add_argument("--lirf-dayshift", choices=["none", "dominant", "mixture", "banded"],
                        help="LIRF day-shift tail (default dominant); predict option")
    parser.add_argument("--lirf-mixture-months", nargs="*",
                        help="LIRF months with p*d + (1 - p)*r on departures without NM delayed 1-3 h "
                             "(default 2026-07; empty: none); predict option")
    parser.add_argument("--grouped-lgb-weight", type=float,
                        help="LightGBM weight in the nine-airport taxi blend (default 0.35); predict option, no retraining")
    parser.add_argument("--grouped-airport-lgb-weight", type=float, choices=[0.0, 0.5],
                        help="Per-airport weight inside the LightGBM term (default 0); "
                             "needs train for the per-airport models")
    parser.add_argument("--no-validate", action="store_true", help="Skip the LIRF walk-forward validation")
    parser.add_argument("--grouped-airport-cb-weight", type=float, choices=[0.0, 0.5],
                        help="Per-airport CatBoost weight inside the CatBoost term (default 0); needs "
                             "train for the per-airport models")
    parser.add_argument("--grouped-lgb-adsb", action="store_const", const=True,
                        help="LightGBM also sees the ADS-B features on rows with the ADS-B regressor; needs train")
    parser.add_argument("--grouped-anchor-cb-weight", type=float,
                        help="Weight of the AOBT_3-anchored CatBoost inside the CatBoost term (default 0; "
                             "0.5 validated in July); needs train")
    parser.add_argument("--grouped-anchor-cb-months", nargs="+",
                        help="Months where the anchored CatBoost applies (default all); predict option")
    parser.add_argument("--grouped-anchor-lgb-weight", type=float,
                        help="Weight of the AOBT_3-anchored LightGBM inside the LightGBM term (default 0; "
                             "0.5 validated in January); needs train")
    parser.add_argument("--grouped-anchor-lgb-months", nargs="+",
                        help="Months where the anchored LightGBM applies (default all); predict option")
    parser.add_argument("--recipe-v26", action="store_true",
                        help="v26/v46 recipe: unfiltered regressor, no LightGBM or cap, no LIRF tail")
    parser.add_argument("--interim-dir")
    parser.add_argument("--adsb-airports", nargs="+", help="Airports using the ADS-B regressor (default "
                                                          "all with context); predict option")
    parser.add_argument("--adsb-features", help="features_takeoff.parquet from python -m pipeline.adsb prepare: "
                                               "ADS-B features in the nine-airport regressors")
    parser.add_argument("--context-features", help="features.parquet from python -m pipeline.context prepare: "
                                                  "pre-takeoff context in the nine-airport regressors")
    parser.add_argument("--inbound-features", help="features.parquet from python -m pipeline.inbound prepare: the "
                                                  "aircraft's previous arrival in the LIRF regressor and classifiers")
    parser.add_argument("--grouped-inbound-features", help="features.parquet from python -m pipeline.inbound prepare "
                                                          "--airports nine: previous arrival in the nine-airport "
                                                          "regressors")
    parser.add_argument("--wx-features", help="features.parquet from python -m pipeline.weather prepare: extended "
                                             "METAR in the regressors of the wx_groups groups (default EDDM+LSZH)")
    parser.add_argument("--state-features", help="features.parquet from python -m pipeline.state prepare: runway "
                                                "queue in the nine-airport regressors; its ADS-B features only in "
                                                "the ADS-B regressor")
    parser.add_argument("--output-root")
    parser.add_argument("--forward-features", help="features.parquet from python -m pipeline.forward prepare: batch "
                                                  "context with later flights in the nine-airport regressors")
    parser.add_argument("--forward-groups", nargs="+", choices=["load", "sequence", "nm", "plan", "delay", "adsb"],
                        help="Batch-context families used by the model (default load sequence nm plan)")
    parser.add_argument("--departures-taxiing", choices=["causal", "batch"],
                        help="departures_taxiing: causal (only departures already airborne) or batch (v26, whole batch)")
    parser.add_argument("--lirf-quality", choices=["causal", "batch"],
                        help="LIRF record-quality context: causal or batch (v26, whole batch)")
    parser.add_argument("--lirf-gate-month-penalty", type=float,
                        help="LIRF gate penalty in the --lirf-gate-penalty-months (off by default); train refits it "
                             "from oof.parquet without retraining CatBoost")
    parser.add_argument("--lirf-gate-penalty-months", nargs="*",
                        help="Ranking months with that gate (default 2026-07; empty: none)")
    parser.add_argument("--name", default="submission", help="Name of the submission files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = build_config(args)
    if args.stage in ("train", "all"):
        train(config, args.components)
    if args.stage in ("predict", "all"):
        verification = predict(config, args.name)
        print(json.dumps({k: verification[k] for k in ["output", "output_sha256", "checks", "controls"]},
                         indent=1, default=str))


if __name__ == "__main__":
    main()
