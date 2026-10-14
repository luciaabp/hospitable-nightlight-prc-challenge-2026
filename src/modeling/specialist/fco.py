"""FCO specialist: specialist features with a delay-weighted copied-block classifier.

Walk-forward validation on July, September, November and December 2025 with seeds 1 and 42,
trees chosen by early stopping on the last two training months. Per fold it writes
``<month>_s<seed>.parquet`` with:

    direct            regressor of the official label on all departures
    taxi              regressor on departures with a measured block time
    p_weighted, p_plain         copied-block probability, weighted and unweighted classifier
    hybrid_weighted, hybrid_plain   p·delay + (1 − p)·taxi when delay > taxi, else taxi

The submission uses ``hybrid_weighted`` for FCO departures with NM and delay <= 12 h.

    python -m src.modeling.specialist.fco train --device 0 --output outputs/experiments/specialist_fco
    python -m src.modeling.specialist.fco evaluate --root outputs/experiments/specialist_fco

    # Final models on all of 2025 and ranking predictions (one seed per GPU).
    python -m src.modeling.specialist.fco final --device 0 --seeds 1 --output outputs/final_models/specialist_fco
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLOCK, FCO, ID, INNER_MONTHS, SEEDS, TARGET, VALIDATION_MONTHS
from .features import load_airport
from .train import combine, copied_block_weight, fit, rmse

STRUCTURED_OOF = Path("outputs/diagnostics/structured_mixture_final/ensemble_oof.parquet")


def train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_airport(FCO, "training")
    imputed = data.imputed
    weight = copied_block_weight(data.meta.delay)
    for seed in args.seeds:
        for month in VALIDATION_MONTHS:
            target = args.output / f"{month}_s{seed}.parquet"
            if target.exists():
                continue
            begin = time.monotonic()
            train_mask = data.meta.month.lt(month).to_numpy()
            months = sorted(data.meta.month[train_mask].unique())[-INNER_MONTHS:]
            inner = train_mask & data.meta.month.isin(months).to_numpy()
            valid = data.meta.month.eq(month).to_numpy()
            xv = data.x.loc[valid]
            direct = fit("regressor", data.x, data.y, train_mask, inner, seed, args.device)
            taxi = fit("regressor", data.x, data.y, train_mask & ~imputed, inner & ~imputed, seed, args.device)
            weighted = fit("classifier", data.x, imputed.astype(int), train_mask, inner, seed, args.device,
                           weight=weight)
            plain = fit("classifier", data.x, imputed.astype(int), train_mask, inner, seed, args.device)
            out = data.meta.loc[valid, [ID, CLOCK, "month", "delay", "missing_nm", TARGET]].reset_index(drop=True)
            out["direct"] = np.maximum(direct.predict(xv), 0)
            out["taxi"] = np.maximum(taxi.predict(xv), 0)
            out["p_weighted"] = weighted.predict_proba(xv)[:, 1]
            out["p_plain"] = plain.predict_proba(xv)[:, 1]
            out["imputed"] = imputed[valid]
            for name in ("weighted", "plain"):
                out[f"hybrid_{name}"] = combine(out.taxi, out.delay, out[f"p_{name}"])
            out.to_parquet(target, index=False)
            print(FCO, seed, month, {c: round(rmse(out[TARGET], out[c]), 1) for c in
                  ["direct", "hybrid_weighted", "hybrid_plain"]}, round(time.monotonic() - begin), "s", flush=True)
    (args.output / "complete.json").write_text(json.dumps({"status": "complete"}))


def final(args):
    """Taxi regressor and weighted classifier on all of 2025; ranking predictions per seed."""
    data = load_airport(FCO, "training")
    ranking = load_airport(FCO, "ranking")
    if list(ranking.x.columns) != list(data.x.columns):
        raise ValueError("Ranking feature schema differs from training")
    imputed, weight = data.imputed, copied_block_weight(data.meta.delay)
    train_mask = np.ones(len(data.meta), bool)
    months = sorted(data.meta.month.unique())[-INNER_MONTHS:]
    inner = data.meta.month.isin(months).to_numpy()
    for seed in args.seeds:
        directory = args.output / f"s{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        taxi = fit("regressor", data.x, data.y, train_mask & ~imputed, inner & ~imputed, seed, args.device)
        classifier = fit("classifier", data.x, imputed.astype(int), train_mask, inner, seed, args.device,
                         weight=weight)
        taxi.save_model(str(directory / "taxi.cbm"))
        classifier.save_model(str(directory / "classifier_weighted.cbm"))
        out = ranking.meta[[ID, CLOCK, "month", "delay", "missing_nm"]].reset_index(drop=True)
        out["taxi"] = np.maximum(taxi.predict(ranking.x), 0)
        out["p_weighted"] = classifier.predict_proba(ranking.x)[:, 1]
        out["hybrid_weighted"] = combine(out.taxi, out.delay, out.p_weighted)
        out.to_parquet(directory / "ranking_predictions.parquet", index=False)
        (directory / "training.json").write_text(json.dumps({
            "seed": seed, "rows": len(data.meta), "inner_months": months,
            "trees": {"taxi": taxi.tree_count_, "classifier_weighted": classifier.tree_count_}}, indent=2))
        print(FCO, "final seed", seed, "trees", taxi.tree_count_, classifier.tree_count_, flush=True)


def segments(frame: pd.DataFrame) -> pd.Series:
    missing, delay = frame.missing_nm.to_numpy(bool), frame.delay.to_numpy(float)
    return pd.Series(np.select([missing & (delay > 43200), missing], ["no NM, delay>12h", "no NM, delay<=12h"],
                               "with NM"), index=frame.index)


def evaluate(args):
    frames = {s: pd.concat([pd.read_parquet(args.root / f"{m}_s{s}.parquet") for m in VALIDATION_MONTHS])
              .sort_values(ID).reset_index(drop=True) for s in SEEDS}
    d = frames[SEEDS[0]][[ID, "month", "delay", "missing_nm", "imputed", TARGET]].copy()
    for col in ["direct", "taxi", "p_weighted", "p_plain", "hybrid_weighted", "hybrid_plain"]:
        d[col] = np.mean([f[col] for f in frames.values()], axis=0)
    ref = pd.read_parquet(STRUCTURED_OOF).set_index(ID)["ensemble"]
    d["structured"] = ref.reindex(d[ID]).to_numpy()
    d["segment"] = segments(d)
    selection = d.month.isin(VALIDATION_MONTHS[:2])
    with_nm = ~d.missing_nm.to_numpy(bool)
    rows = []
    # Routed blend: flights with NM get (1 - a)·structured + a·candidate; flights without NM keep
    # the structured model. The weight a is chosen on July/September flights with NM only
    # (a = 1 is the candidate alone, as used by the submission).
    for name in ["direct", "hybrid_weighted", "hybrid_plain"]:
        grid = np.round(np.arange(0, 1.01, 0.1), 1)
        fit_rows = selection & with_nm
        a = min(grid, key=lambda a: rmse(d[TARGET][fit_rows], ((1 - a) * d.structured + a * d[name])[fit_rows]))
        d[f"blend_{name}"] = np.where(with_nm, (1 - a) * d.structured + a * d[name], d.structured)
        for stage, keep in [("selection", selection), ("confirmation", ~selection)]:
            f = d[keep]
            for seg, g in [("all", f), *f.groupby("segment")]:
                rows.append({"candidate": name, "weight_on_candidate": a, "stage": stage, "segment": seg, "n": len(g),
                             "structured": rmse(g[TARGET], g.structured), "candidate_alone": rmse(g[TARGET], g[name]),
                             "blend": rmse(g[TARGET], g[f"blend_{name}"])})
    table = pd.DataFrame(rows)
    table.to_csv(args.root / "evaluation.csv", index=False)
    d.to_parquet(args.root / "paired_predictions.parquet", index=False)
    monthly = d.groupby("month").apply(lambda g: pd.Series({"structured": rmse(g[TARGET], g.structured),
                                                            **{f"blend_{n}": rmse(g[TARGET], g[f"blend_{n}"])
                                                               for n in ["direct", "hybrid_weighted", "hybrid_plain"]}}))
    monthly.to_csv(args.root / "monthly.csv")
    print(monthly.round(1).to_string())
    from sklearn.metrics import roc_auc_score
    known = d[~d.missing_nm]
    auc = {name: float(roc_auc_score(known.imputed, known[f"p_{name}"])) for name in ("weighted", "plain")}
    pd.set_option("display.width", 220)
    print(table.round(1).to_string(index=False))
    print("AUC copied-block classifier on flights with NM:", {k: round(v, 4) for k, v in auc.items()})


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
