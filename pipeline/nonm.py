"""LIRF in summer: departures without NM delayed 1-3 h get the classifier blend instead of the gate.

    prediction = max(p d + (1 - p) r, 0)

``p``: imputed-block probability, ``d``: delay, ``r``: clean taxi (seed means). The imputed rate of these departures
drops in summer; the classifier captures it, the year-round linear gate does not.
"""
import numpy as np
import pandas as pd

from .config import ID, PipelineConfig

EOBT, AOBT = "EOBT_1_flt", "AOBT_3_flt"


def rows(frame: pd.DataFrame, delay, months, config: PipelineConfig) -> np.ndarray:
    """Departures without NM, ``lirf_mixture_delay[0] < d <= lirf_mixture_delay[1]``, in ``lirf_mixture_months``."""
    low, high = config.lirf_mixture_delay
    delay = np.asarray(delay, float)
    no_nm = ~(frame[AOBT].notna() & frame[EOBT].notna()).to_numpy()
    return no_nm & (delay > low) & (delay <= high) & np.isin(np.asarray(months), list(config.lirf_mixture_months))


def apply(frame: pd.DataFrame, parts: dict, previous, months, config: PipelineConfig) -> tuple[np.ndarray, dict]:
    """New LIRF prediction and a summary for the checks. ``parts``: ``structured.predict`` per seed."""
    first = next(iter(parts.values()))
    if not np.array_equal(first[ID].to_numpy(), frame[ID].to_numpy()):
        raise ValueError("The LIRF pieces do not follow the ranking order")
    previous = np.asarray(previous, float)
    delay = first.delay.to_numpy(float)
    selected = rows(frame, delay, months, config)
    if not selected.any():
        return previous, {"months": list(config.lirf_mixture_months), "rows": 0}
    taxi = np.mean([p.clean_taxi.to_numpy(float) for p in parts.values()], axis=0)
    history = np.isin(np.asarray(months), list(config.fnhist_months))
    column = np.where(history, "probability_fnhist", "probability")
    probability = np.mean([np.where(column == "probability_fnhist", p.probability_fnhist, p.probability)
                           for p in parts.values()], axis=0)
    mixture = np.maximum(probability * delay + (1 - probability) * taxi, 0)
    new = np.where(selected, mixture, previous)
    change = (new - previous)[selected]
    by_month = pd.Series(np.asarray(months)[selected]).value_counts().sort_index()
    return new, {"months": list(config.lirf_mixture_months), "delay_band_sec": list(config.lirf_mixture_delay),
                 "rows": int(selected.sum()), "rows_by_month": {m: int(n) for m, n in by_month.items()},
                 "mean_change": float(change.mean()), "mean_abs_change": float(np.abs(change).mean()),
                 "max_abs_change": float(np.abs(change).max())}
