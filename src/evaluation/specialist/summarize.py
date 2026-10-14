"""Choose each airport's direct/hybrid weight on July/September; confirm on November/December.

    specialist = cap(w * direct + (1 - w) * hybrid),   w in {0, 0.1, ..., 1}

``cap`` is the delay bound of ``policy.cap_by_delay``, the same rule the submission uses.

``w`` is selected on the seed-averaged predictions of the selection months only. The
confirmation months report the specialist against the reference OOF of the previous
architecture (``current``), with daily bootstrap intervals and per-seed deltas.

    python -m src.evaluation.specialist.summarize --root outputs/experiments/specialist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.specialist.policy import cap_by_delay
from src.modeling.specialist.config import (AIRPORTS, CLOCK, CONFIRMATION_MONTHS, ID, SEEDS, SELECTION_MONTHS,
                                            TARGET, VALIDATION_MONTHS)

GRID = np.round(np.arange(0, 1.01, 0.1), 1)
# OOF of the previous architecture (0.6 surface_ensemble + 0.4 hybrid) for eight airports.
REFERENCE = Path("outputs/diagnostics/20260921_airport_blends_clean_seeds/paired_predictions.parquet")
# LSZH is missing there; approximate it with the same 0.6/0.4 blend of its surface OOF and
# the clean seed-42 hybrid specialist.
LSZH_REFERENCE = Path("outputs/diagnostics/20260921_lszh_surface_clean_seeds/paired_predictions.parquet")


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def reference_predictions() -> pd.Series:
    ref = pd.read_parquet(REFERENCE, columns=[ID, "current"]).set_index(ID)["current"]
    if LSZH_REFERENCE.exists():
        lszh = pd.read_parquet(LSZH_REFERENCE, columns=[ID, "surface", "s42"]).set_index(ID)
        ref = pd.concat([ref, 0.6 * lszh["surface"] + 0.4 * lszh["s42"]])
    return ref[~ref.index.duplicated()]


def bootstrap(frame, candidate, draws=4000, seed=20260924):
    """95 % interval of the RMSE change, resampling whole days within each month."""
    f = frame.assign(old=(frame[TARGET] - frame.reference) ** 2, new=(frame[TARGET] - frame[candidate]) ** 2,
                     day=frame[CLOCK].dt.strftime("%Y-%m-%d"))
    g = f.groupby(["month", "day"]).agg(n=(ID, "size"), old=("old", "sum"), new=("new", "sum"))
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws, 3))
    for _, monthly in g.groupby(level="month"):
        a = monthly[["n", "old", "new"]].to_numpy()
        totals += a[rng.integers(0, len(a), size=(draws, len(a)))].sum(axis=1)
    delta = np.sqrt(totals[:, 2] / totals[:, 0]) - np.sqrt(totals[:, 1] / totals[:, 0])
    return [float(v) for v in np.quantile(delta, [0.025, 0.975])]


def load(root: Path, airport: str, seeds) -> pd.DataFrame:
    frames = {s: pd.read_parquet([root / airport / f"{m}_s{s}.parquet" for m in VALIDATION_MONTHS])
              .sort_values(ID).reset_index(drop=True) for s in seeds}
    d = frames[seeds[0]][[ID, CLOCK, "month", "delay", "missing_nm", TARGET]].copy()
    for s, f in frames.items():
        np.testing.assert_array_equal(f[ID], d[ID])
        d[f"direct_s{s}"], d[f"hybrid_s{s}"] = f["direct"].to_numpy(), f["hybrid"].to_numpy()
    d["direct"] = d[[f"direct_s{s}" for s in seeds]].mean(axis=1)
    d["hybrid"] = d[[f"hybrid_s{s}" for s in seeds]].mean(axis=1)
    return d


def run(root: Path, seeds):
    reference = reference_predictions()
    weights, rows, monthly = {}, [], []
    airports = [a for a in AIRPORTS if (root / a / "complete.json").exists()]
    for airport in airports:
        d = load(root, airport, seeds)
        selection = d.month.isin(SELECTION_MONTHS).to_numpy()
        blend = lambda w, direct=d.direct, hybrid=d.hybrid: cap_by_delay(w * direct + (1 - w) * hybrid, d.delay)  # noqa: E731
        w = min(GRID, key=lambda w: rmse(d[TARGET][selection], blend(w)[selection]))
        weights[airport] = float(w)
        d["specialist"] = blend(w)
        d["reference"] = reference.reindex(d[ID]).to_numpy()
        has_reference = d.reference.notna().all()
        for stage, months in [("selection", SELECTION_MONTHS), ("confirmation", CONFIRMATION_MONTHS)]:
            f = d[d.month.isin(months)]
            row = {"airport": airport, "stage": stage, "w_direct": w, "n": len(f),
                   "specialist_rmse": rmse(f[TARGET], f.specialist)}
            if has_reference:
                row.update(reference_rmse=rmse(f[TARGET], f.reference),
                           delta=row["specialist_rmse"] - rmse(f[TARGET], f.reference))
                row["ci_low"], row["ci_high"] = bootstrap(f, "specialist")
                for s in seeds:
                    pred = cap_by_delay(w * f[f"direct_s{s}"] + (1 - w) * f[f"hybrid_s{s}"], f.delay)
                    row[f"delta_s{s}"] = rmse(f[TARGET], pred) - rmse(f[TARGET], f.reference)
            rows.append(row)
        for month, f in d.groupby("month"):
            monthly.append({"airport": airport, "month": month, "specialist": rmse(f[TARGET], f.specialist),
                            "capped_rows": int((f.specialist < w * f.direct + (1 - w) * f.hybrid).sum()),
                            "direct": rmse(f[TARGET], f.direct), "hybrid": rmse(f[TARGET], f.hybrid),
                            "reference": rmse(f[TARGET], f.reference) if has_reference else np.nan})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(root / "monthly.csv", index=False)
    (root / "weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    pd.set_option("display.width", 220)
    print(summary.round(2).to_string(index=False))
    print("weights:", weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    run(args.root, args.seeds)


if __name__ == "__main__":
    main()
