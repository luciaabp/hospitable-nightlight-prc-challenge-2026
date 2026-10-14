"""First-class accounting of the day-shift and delay regimes, beside RMSE.

FCO's residual error is not spread over ordinary taxi: a few dozen movements
decide it. Reporting only RMSE hides which of the two failure directions moved,
and they respond to opposite changes of the gate, so they are reported apart.

Cohorts are defined with labels and are diagnostic only: they never select a
model and never become predictors. Masks overlap by design; percentages are
shares of the same total SSE and must not be added across cohorts.
"""
import numpy as np
import pandas as pd

from .structured_mixture import day_shift_mask

DAY = 86400.0
ORDINARY_LABEL = 1800.0    # an ordinary FCO departure does not taxi longer than this
COMMITTED_PREDICTION = 3600.0  # above this the gate has left the ordinary-taxi candidate
DAY_MISS = 21600.0         # a day-shift missed by more than six hours is a missed regime


def regime_cohorts(y, pred, imputed):
    """Label-defined masks for the two directions in which the gate fails."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    day_true = day_shift_mask(y, imputed)
    ordinary = y <= ORDINARY_LABEL
    return {
        "all": np.ones(len(y), bool),
        "ordinary": ordinary,
        # The false positives: an ordinary taxi predicted as a long regime.
        "delay_overcommit": ordinary & (pred > COMMITTED_PREDICTION),
        # The false negatives: the day-shift regime and the part of it that is missed.
        "day_true": day_true,
        "day_missed": day_true & (pred < y - DAY_MISS),
        # Predicting a whole day where the rule sees none.
        "day_overcommit": ~day_true & (pred > DAY),
        "long_not_day": ~day_true & (y > COMMITTED_PREDICTION),
    }


def regime_report(y, pred, imputed, variant="prediction"):
    """Per-cohort share of the total squared error, its RMSE and its direction."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    error = (y - pred) ** 2
    total = float(error.sum())
    rows = []
    for name, mask in regime_cohorts(y, pred, imputed).items():
        n = int(mask.sum())
        sse = float(error[mask].sum())
        rows.append({
            "variant": variant, "cohort": name, "rows": n,
            "sse": sse,
            # Share of the whole error budget, which is what a submission is scored on.
            "pct_sse": 100 * sse / total if total > 0 else np.nan,
            "rmse": float(np.sqrt(sse / n)) if n else np.nan,
            # Sign tells which way the gate leans: positive means over-prediction.
            "bias": float(np.mean(pred[mask] - y[mask])) if n else np.nan,
        })
    return pd.DataFrame(rows)


def label_partition(y, imputed):
    """A fixed, mutually exclusive partition, so two gates can be compared row for row.

    The cohorts above move with the prediction, which is what makes them useful as
    counts but useless for attributing a change: a variant can leave a cohort
    rather than improve inside it. This partition depends on labels only, so the
    per-cohort squared errors of any two variants subtract to the total change.
    """
    y = np.asarray(y, dtype=float)
    day = day_shift_mask(y, imputed)
    return {
        "ordinary": ~day & (y <= ORDINARY_LABEL),
        "long_under_hour": ~day & (y > ORDINARY_LABEL) & (y <= COMMITTED_PREDICTION),
        "long_over_hour": ~day & (y > COMMITTED_PREDICTION),
        "day_regime": day,
    }


def attribution(frame, variants, reference=None, y="y_true", imputed="imputed"):
    """Where a variant's squared error moved against a reference, cohort by cohort."""
    missing = [v for v in variants if v not in frame]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    reference = reference or variants[0]
    labels = frame[y].to_numpy(dtype=float)
    rows = []
    for name, mask in label_partition(labels, frame[imputed]).items():
        row = {"cohort": name, "rows": int(mask.sum())}
        for variant in variants:
            row[f"sse_{variant}"] = float(((labels[mask] - frame[variant].to_numpy()[mask]) ** 2).sum())
        for variant in variants:
            row[f"delta_{variant}"] = row[f"sse_{variant}"] - row[f"sse_{reference}"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_variants(frame, variants, y="y_true", imputed="imputed"):
    """Stack per-cohort reports so two gates can be read side by side."""
    missing = [v for v in variants if v not in frame]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    return pd.concat([regime_report(frame[y], frame[v], frame[imputed], v) for v in variants],
                     ignore_index=True)
