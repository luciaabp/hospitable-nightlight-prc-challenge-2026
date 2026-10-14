"""Submission assembly in template order, strict checks and comparison with earlier submissions."""
import numpy as np
import pandas as pd

from .config import ID, PREDICTION


def assemble(template: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """``predictions``: one prediction per ID. Returns the submission with the template's IDs, order and types."""
    if predictions.index.duplicated().any():
        raise ValueError("More than one prediction for some ID")
    missing = ~template[ID].isin(predictions.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} template IDs without prediction")
    extra = ~predictions.index.isin(template[ID])
    if extra.any():
        raise ValueError(f"{int(extra.sum())} predictions with IDs outside the template")
    submission = template[[ID]].copy()
    submission[PREDICTION] = predictions.loc[template[ID]].to_numpy(dtype=float)
    check(submission, template)
    return submission


def check(submission: pd.DataFrame, template: pd.DataFrame) -> None:
    if list(submission.columns) != list(template.columns):
        raise ValueError(f"Columns {list(submission.columns)} differ from the template {list(template.columns)}")
    if not (submission.dtypes == template.dtypes).all():
        raise ValueError("Types differ from the template")
    if len(submission) != len(template) or not np.array_equal(submission[ID].to_numpy(), template[ID].to_numpy()):
        raise ValueError("IDs or order differ from the template")
    values = submission[PREDICTION].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Null or infinite predictions")
    if (values < 0).any():
        raise ValueError("Negative predictions")


def compare(submission: pd.DataFrame, reference: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Differences against another submission, total and by airport x month. ``meta``: ID, airport, month."""
    merged = submission.merge(reference, on=ID, suffixes=("", "_ref"), validate="1:1").merge(meta, on=ID)
    if len(merged) != len(submission):
        raise ValueError("The reference submission does not cover the same IDs")
    merged["delta"] = merged[PREDICTION] - merged[f"{PREDICTION}_ref"]

    def stats(part):
        d = part.delta
        return {"rows": int(len(d)), "changed": int((d != 0).sum()), "mean": float(d.mean()),
                "mean_abs": float(d.abs().mean()), "max_abs": float(d.abs().max()),
                "rows_abs_gt_60s": int((d.abs() > 60).sum()), "rows_abs_gt_600s": int((d.abs() > 600).sum())}

    return {"all": stats(merged),
            "by_airport_month": {f"{a}|{m}": stats(p) for (a, m), p in merged.groupby(["airport", "month"])}}
