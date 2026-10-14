"""LOBT timing bounds: keep each prediction within (takeoff − LOBT) ± 3606 s.
``LOBT_flt`` is the last known off-block time filed with NM and is present in the ranking data;
 departure BLOCK and labels are not read. Rows without LOBT are unchanged, and the lower bound is floored at 0.

When the NM taxi proxy (takeoff − AOBT) is one hour or longer, the lower bound is capped at
proxy + 30 min. In 2025, every departure where the bound exceeded the proxy by more than 30 min
had a proxy under one hour (and a label above the bound), so a long proxy far below the bound is
an unobserved regime (January 2026 EHAM flights with a 1.5 h proxy and a LOBT 3–4 h before
takeoff). Short or missing proxies keep the full bound.

The submission applies the bounds to every airport except FCO. Walk-forward check on the
specialist OOF (July/September, November/December 2025):

    python -m src.evaluation.specialist.lobt_bounds --output outputs/diagnostics/specialist_lobt_bounds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.specialist.policy import cap_by_delay
from src.evaluation.specialist.summarize import bootstrap, load, rmse
from src.modeling.specialist.config import (CONFIRMATION_MONTHS, ID, RAW_DIR, SEEDS, SELECTION_MONTHS,
                                            TARGET)

MARGIN_SEC = 3606.0
LONG_PROXY_SEC = 3600.0
LONG_PROXY_SLACK_SEC = 1800.0
SPECIALIST_ROOT = Path("outputs/experiments/specialist")


def lobt_lead(frame: pd.DataFrame) -> np.ndarray:
    """Seconds from LOBT to takeoff (NaN when LOBT is missing)."""
    return (frame["MVT_TIME_UTC_mvt"] - frame["LOBT_flt"]).dt.total_seconds().to_numpy(float)


def nm_proxy(frame: pd.DataFrame) -> np.ndarray:
    """Seconds from NM AOBT to takeoff (NaN when AOBT is missing)."""
    return (frame["MVT_TIME_UTC_mvt"] - frame["AOBT_3_flt"]).dt.total_seconds().to_numpy(float)


def apply_bounds(prediction, lead, proxy=None, margin: float = MARGIN_SEC) -> np.ndarray:
    prediction, lead = np.asarray(prediction, float), np.asarray(lead, float)
    known = np.isfinite(lead)
    lower = np.where(known, np.maximum(lead - margin, 0), -np.inf)
    if proxy is not None:
        proxy = np.asarray(proxy, float)
        long_proxy = np.isfinite(proxy) & (proxy >= LONG_PROXY_SEC)
        lower = np.where(long_proxy, np.minimum(lower, proxy + LONG_PROXY_SLACK_SEC), lower)
    upper = np.where(known, lead + margin, np.inf)
    return np.clip(prediction, lower, upper)


def departures(dataset: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("training_*.parquet")) if dataset == "training" else [RAW_DIR / "ranking.parquet"]
    raw = pd.concat([pd.read_parquet(f, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt",
                                                 "AOBT_3_flt"])
                     for f in files])
    return raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates(ID).set_index(ID)


def backtest(args):
    args.output.mkdir(parents=True, exist_ok=True)
    weights = json.loads((SPECIALIST_ROOT / "weights.json").read_text())
    train = departures("training")
    lead = pd.Series(lobt_lead(train), index=train.index)
    proxy = pd.Series(nm_proxy(train), index=train.index)
    rows = []
    for airport, w in weights.items():
        d = load(SPECIALIST_ROOT, airport, SEEDS)
        d["reference"] = cap_by_delay(w * d.direct + (1 - w) * d.hybrid, d.delay)
        d["bounded"] = apply_bounds(d.reference, lead.reindex(d[ID]).to_numpy(), proxy.reindex(d[ID]).to_numpy())
        for stage, months in [("selection", SELECTION_MONTHS), ("confirmation", CONFIRMATION_MONTHS)]:
            f = d[d.month.isin(months)]
            ci = bootstrap(f, "bounded")
            rows.append({"airport": airport, "stage": stage, "n": len(f),
                         "rows_changed": int((f.bounded != f.reference).sum()),
                         "delta": rmse(f[TARGET], f.bounded) - rmse(f[TARGET], f.reference),
                         "delta_sse": float(((f[TARGET] - f.bounded) ** 2 - (f[TARGET] - f.reference) ** 2).sum()),
                         "ci_low": ci[0], "ci_high": ci[1]})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "backtest.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    backtest(parser.parse_args())


if __name__ == "__main__":
    main()
