"""Out-of-fold base predictions using consecutive two-month blocks."""
from __future__ import annotations
import pandas as pd
import movement_features as F
import runtime
from dataset_cache import TRUTH
from estimator import build_model, leaky_columns, prepare

TIME = 'MVT_TIME_UTC_mvt'

def month_blocks(months) -> dict[int, int]:
    uniq = sorted({int(m) for m in months})
    return {m: i // 2 for i, m in enumerate(uniq)}

def oof_base(cfg: dict, train: pd.DataFrame, blind: pd.DataFrame, ranking_cols_ref: pd.DataFrame, run=None) -> pd.DataFrame:
    train_month = train[TIME].dt.month
    blind_month = blind[TIME].dt.month
    blocks: dict[int, list[int]] = {}
    for month, k in month_blocks(train_month).items():
        blocks.setdefault(k, []).append(month)
    parts = []
    for k, months in sorted(blocks.items()):
        if run:
            run.log(f'block {k + 1}/{len(blocks)} months {months}')
        tr = train[~train_month.isin(months)]
        te = blind[blind_month.isin(months) & blind[TRUTH].notna()]
        cols = prepare(tr, [te])
        drop = leaky_columns(tr, ranking_cols_ref, cols)
        cols = [c for c in cols if c not in drop]
        model = build_model(cfg).fit(tr, cols, run=run)
        parts.append(pd.DataFrame({F.ID: te[F.ID].to_numpy(), 'dia': te[TIME].dt.strftime('%Y-%m-%d').to_numpy(), 'mes': te[TIME].dt.month.to_numpy(), TRUTH: te[TRUTH].to_numpy(), 'pred': model.predict(te)}))
        del tr, te, model
        runtime.release_memory()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(F.ID, ignore_index=True)
