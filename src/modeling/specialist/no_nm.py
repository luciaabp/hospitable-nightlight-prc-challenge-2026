"""Pooled expert for departures without NM data, all ten airports.

Departures without an NM match (no AOBT, no flight plan) are 0.7–2 % of flights but hold a large
share of the error (57 % of FCO's). A single model is trained for them across airports, with the
airport as a feature, using the specialist matrix minus every column derived from the flight's own
NM record (clocks, flight-plan fields, own-AOBT-anchored traffic, stand vs own AOBT). The regime
flag ``sin_plan_vuelo`` and context computed from other flights are kept.

Variants:

    native    trained on departures without NM only
    transfer  departures without NM plus a TRANSFER_SHARE sample of departures with NM, whose
              physical taxi patterns are learned with the same NM-free columns

Per variant: taxi regressor (measured blocks), copied-block classifier weighted by
clip(1 + (delay/3600)^2, 1, 25), and direct regressor. Early stopping on departures without NM
of the last two training months; refit on all training rows.

The submission uses the transfer hybrid for LFPG departures without NM.

    python -m src.modeling.specialist.no_nm train --device 0 --seeds 1 --output outputs/experiments/specialist_no_nm
    python -m src.modeling.specialist.no_nm evaluate --root outputs/experiments/specialist_no_nm

    # Final transfer models on all of 2025 and ranking predictions for departures without NM.
    python -m src.modeling.specialist.no_nm final --device 0 --seeds 1 --output outputs/final_models/specialist_no_nm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AIRPORTS, CLOCK, FCO, ID, INNER_MONTHS, SEEDS, TARGET, VALIDATION_MONTHS
from .features import load_airport
from .train import combine, copied_block_weight, fit, rmse

ALL_AIRPORTS = AIRPORTS + [FCO]
TRANSFER_SHARE = 0.15
OWN_NM_COLUMNS = {"taxi_proxy_sec", "stand_last_arr_minus_nm_offblock_s", "aobt_after_takeoff", "eobt_imputado"}
SPECIALIST_ROOT = Path("outputs/experiments/specialist")
SPECIALIST_WEIGHTS = Path("configs/specialist/weights.json")
STRUCTURED_OOF = Path("outputs/diagnostics/structured_mixture_final/ensemble_oof.parquet")
OLD_ARCHITECTURE_OOF = Path("outputs/diagnostics/20260921_airport_blends_clean_seeds/paired_predictions.parquet")


def nm_free_columns(columns) -> list[str]:
    """Drop features that come from the flight's own NM record."""
    return [c for c in columns if not ("_flt" in c or c in OWN_NM_COLUMNS or c.startswith("traffic_offblock_")
                                       or c.startswith("traffic_reference_"))]


def load_pool(seed: int = 2026):
    """No-NM departures of every airport plus a TRANSFER_SHARE sample of departures with NM."""
    rng = np.random.default_rng(seed)
    metas, xs, ys, imps = [], [], [], []
    columns = None
    for airport in ALL_AIRPORTS:
        data = load_airport(airport, "training")
        columns = columns or nm_free_columns(data.x.columns)
        missing = data.meta.missing_nm.to_numpy(bool)
        keep = missing | (rng.random(len(missing)) < TRANSFER_SHARE)
        meta = data.meta.loc[keep].reset_index(drop=True).assign(airport=airport)
        metas.append(meta)
        xs.append(data.x.loc[keep, columns].reset_index(drop=True))
        ys.append(data.y[keep])
        imps.append(data.imputed[keep])
        print(airport, "rows", int(keep.sum()), "without NM", int(missing.sum()), flush=True)
    x = pd.concat(xs, ignore_index=True)
    for c in x.select_dtypes(include="object").columns:
        x[c] = x[c].astype(str)
    return pd.concat(metas, ignore_index=True), x, np.concatenate(ys), np.concatenate(imps)


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    meta, x, y, imputed = load_pool()
    missing = meta.missing_nm.to_numpy(bool)
    delay = meta.delay.to_numpy(float)
    weight = copied_block_weight(delay)
    (args.output / "config.json").write_text(json.dumps({"transfer_share": TRANSFER_SHARE, "columns": list(x.columns),
                                                          "rows": len(meta), "rows_without_nm": int(missing.sum())},
                                                         indent=2))
    for seed in args.seeds:
        for month in VALIDATION_MONTHS:
            target = args.output / f"{month}_s{seed}.parquet"
            if target.exists():
                continue
            begin = time.monotonic()
            train_all = meta.month.lt(month).to_numpy()
            months = sorted(meta.month[train_all].unique())[-INNER_MONTHS:]
            inner = train_all & meta.month.isin(months).to_numpy() & missing   # stop on no-NM flights only
            valid = meta.month.eq(month).to_numpy() & missing
            out = meta.loc[valid, [ID, CLOCK, "month", "airport", "delay", TARGET]].reset_index(drop=True)
            out["imputed"] = imputed[valid]
            for variant, pool in [("native", missing), ("transfer", np.ones(len(meta), bool))]:
                tr = train_all & pool
                direct = fit("regressor", x, y, tr, inner, seed, args.device)
                taxi = fit("regressor", x, y, tr & ~imputed, inner & ~imputed, seed, args.device)
                clf = fit("classifier", x, imputed.astype(int), tr, inner, seed, args.device, weight=weight)
                xv = x.loc[valid]
                out[f"{variant}_direct"] = np.maximum(direct.predict(xv), 0)
                out[f"{variant}_taxi"] = np.maximum(taxi.predict(xv), 0)
                out[f"{variant}_p"] = clf.predict_proba(xv)[:, 1]
                out[f"{variant}_hybrid"] = combine(out[f"{variant}_taxi"], out.delay, out[f"{variant}_p"])
            out.to_parquet(target, index=False)
            print("no-NM", seed, month, {c: round(rmse(out[TARGET], out[c]), 1) for c in out if c.endswith(("_direct", "_hybrid"))},
                  round(time.monotonic() - begin), "s", flush=True)
    (args.output / f"complete_s{'_'.join(map(str, args.seeds))}.json").write_text(json.dumps({"status": "complete"}))


def ranking_pool(columns):
    metas, xs = [], []
    for airport in ALL_AIRPORTS:
        data = load_airport(airport, "ranking")
        missing = data.meta.missing_nm.to_numpy(bool)
        metas.append(data.meta.loc[missing].reset_index(drop=True).assign(airport=airport))
        xs.append(data.x.loc[missing, columns].reset_index(drop=True))
    x = pd.concat(xs, ignore_index=True)
    for c in x.select_dtypes(include="object").columns:
        x[c] = x[c].astype(str)
    return pd.concat(metas, ignore_index=True), x


def final(args):
    """Transfer variant on all of 2025; predictions for ranking departures without NM."""
    meta, x, y, imputed = load_pool()
    missing = meta.missing_nm.to_numpy(bool)
    weight = copied_block_weight(meta.delay.to_numpy(float))
    train_all = np.ones(len(meta), bool)
    months = sorted(meta.month.unique())[-INNER_MONTHS:]
    inner = meta.month.isin(months).to_numpy() & missing
    rmeta, rx = ranking_pool(list(x.columns))
    for seed in args.seeds:
        directory = args.output / f"s{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        taxi = fit("regressor", x, y, train_all & ~imputed, inner & ~imputed, seed, args.device)
        clf = fit("classifier", x, imputed.astype(int), train_all, inner, seed, args.device, weight=weight)
        taxi.save_model(str(directory / "transfer_taxi.cbm"))
        clf.save_model(str(directory / "transfer_classifier_weighted.cbm"))
        out = rmeta[[ID, CLOCK, "month", "airport", "delay"]].copy()
        out["transfer_taxi"] = np.maximum(taxi.predict(rx), 0)
        out["transfer_p"] = clf.predict_proba(rx)[:, 1]
        out["transfer_hybrid"] = combine(out.transfer_taxi, out.delay, out.transfer_p)
        out.to_parquet(directory / "ranking_predictions.parquet", index=False)
        (directory / "training.json").write_text(json.dumps({"seed": seed, "rows": len(meta), "inner_months": months,
            "trees": {"taxi": taxi.tree_count_, "classifier": clf.tree_count_}, "columns": list(x.columns)}, indent=2))
        print("no-NM final seed", seed, "trees", taxi.tree_count_, clf.tree_count_, "ranking rows", len(out), flush=True)


def current_predictions() -> pd.Series:
    """What each airport predicted before this expert for these flights (walk-forward OOF)."""
    from src.evaluation.specialist.policy import cap_by_delay
    from src.evaluation.specialist.summarize import load
    parts = []
    weights = json.loads(SPECIALIST_WEIGHTS.read_text())
    for airport, w in weights.items():
        d = load(SPECIALIST_ROOT, airport, SEEDS)
        parts.append(pd.Series(cap_by_delay(w * d.direct + (1 - w) * d.hybrid, d.delay), index=d[ID].to_numpy()))
    specialist = pd.concat(parts)
    old = pd.read_parquet(OLD_ARCHITECTURE_OOF, columns=[ID, "airport", "current"])
    lfpg_v36 = old[old.airport.eq("LFPG")].set_index(ID)["current"]
    structured = pd.read_parquet(STRUCTURED_OOF, columns=[ID, "ensemble"]).set_index(ID)["ensemble"]
    return specialist, lfpg_v36, structured


def evaluate(args):
    seeds = [s for s in SEEDS if (args.root / f"{VALIDATION_MONTHS[0]}_s{s}.parquet").exists()]
    frames = {s: pd.concat([pd.read_parquet(args.root / f"{m}_s{s}.parquet") for m in VALIDATION_MONTHS])
              .sort_values(ID).reset_index(drop=True) for s in seeds}
    d = frames[seeds[0]][[ID, CLOCK, "month", "airport", "delay", "imputed", TARGET]].copy()
    candidates = [c for c in frames[seeds[0]] if c.endswith(("_direct", "_hybrid"))]
    for c in candidates:
        d[c] = np.mean([f[c] for f in frames.values()], axis=0)
    from src.evaluation.specialist.policy import cap_by_delay
    specialist, lfpg_v36, structured = current_predictions()
    d["specialist"] = specialist.reindex(d[ID]).to_numpy()
    d["current"] = np.where(d.airport.eq(FCO), structured.reindex(d[ID]).to_numpy(),
                            np.where(d.airport.eq("LFPG"), lfpg_v36.reindex(d[ID]).to_numpy(), d.specialist))
    d["segment"] = np.where(d.airport.eq(FCO) & (d.delay > 43200), "FCO >12h",
                            np.where(d.airport.eq(FCO), "FCO <=12h", "other airports"))
    for c in candidates:
        d[c] = cap_by_delay(d[c], d.delay)
    rows = []
    selection = d.month.isin(VALIDATION_MONTHS[:2])
    for stage, keep in [("selection", selection), ("confirmation", ~selection)]:
        for seg, g in [("all", d[keep]), *d[keep].groupby("segment"), *d[keep & ~d.airport.eq(FCO)].groupby("airport")]:
            row = {"stage": stage, "group": seg, "n": len(g), "current": rmse(g[TARGET], g.current)}
            row.update({c: rmse(g[TARGET], g[c]) for c in candidates})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.root / "evaluation.csv", index=False)
    d.to_parquet(args.root / "paired_predictions.parquet", index=False)
    pd.set_option("display.width", 250)
    print(table.round(1).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--device", default="0")
    t.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    t.add_argument("--output", type=Path, required=True)
    e = commands.add_parser("evaluate")
    e.add_argument("--root", type=Path, required=True)
    f = commands.add_parser("final")
    f.add_argument("--device", default="0")
    f.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    f.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "final": final}[args.command](args)


if __name__ == "__main__":
    main()
