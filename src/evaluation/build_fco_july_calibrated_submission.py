"""July-only FCO delay-weight calibration of the structured gate (used by the specialist submission)."""

import numpy as np
import pandas as pd

from src.modeling.data import ID_COLUMN

TARGET = "TAXITIME_SEC_mvt"
DELAY_THRESHOLD = 43_200.0
RANKING_MONTH = "2026-07"


def july_noncritical_mask(ranking: pd.DataFrame) -> pd.Series:
    delay = (ranking.MVT_TIME_UTC_mvt - ranking.SCHED_TIME_UTC_mvt).dt.total_seconds()
    month = ranking.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    return ranking.airport.eq("LIRF") & ranking.AOBT_3_flt.isna() & delay.le(DELAY_THRESHOLD) & month.eq(RANKING_MONTH)


def calibrated_components(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        raise ValueError("At least one structured prediction is required")
    indexed = []
    for prediction in predictions:
        if prediction[ID_COLUMN].isna().any() or prediction[ID_COLUMN].duplicated().any():
            raise ValueError("Structured predictions need unique non-null IDs")
        indexed.append(prediction.set_index(ID_COLUMN, verify_integrity=True))
    if any(set(part.index) != set(indexed[0].index) for part in indexed[1:]):
        raise ValueError("Structured components have different ID coverage")
    columns = ["clean_taxi", "probability", "delay", "gate3_quality"]
    ensemble = sum(part.loc[indexed[0].index, columns] for part in indexed) / len(indexed)
    delay_candidate = np.maximum(ensemble.clean_taxi, ensemble.delay)
    denominator = ensemble.delay - ensemble.clean_taxi
    effective_weight = np.where(
        denominator > 0,
        (ensemble.gate3_quality - ensemble.clean_taxi) / denominator,
        0.0,
    )
    capped_weight = np.minimum(effective_weight, ensemble.probability).clip(0.0, 1.0)
    ensemble["effective_weight"] = effective_weight
    ensemble["capped_weight"] = capped_weight
    ensemble["calibrated"] = ensemble.clean_taxi + capped_weight * (delay_candidate - ensemble.clean_taxi)
    return ensemble
