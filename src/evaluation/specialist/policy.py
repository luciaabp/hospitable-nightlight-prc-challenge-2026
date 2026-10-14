"""Submission policy on top of the specialist models.

Kept outside ``src.modeling.specialist`` so that changing it does not alter the model
code hash recorded by validation and final training.

* ``DELAY_CAP_SEC``: the label is apparent delay + (SCHED − BLOCK), and BLOCK rarely
  precedes SCHED by more than 30 minutes, so predictions are capped at
  ``max(delay, 0) + DELAY_CAP_SEC``. On the 2025 OOF it touches 6 of ~600,000 flights
  and slightly improves the selection months (218.15 → 217.55 s).

Which model predicts which rows is decided in ``submission.py``.
"""

import numpy as np

DELAY_CAP_SEC = 1800.0


def cap_by_delay(prediction, delay, cap=DELAY_CAP_SEC) -> np.ndarray:
    """``min(prediction, max(delay, 0) + cap)``; rows without a finite delay are not capped."""
    prediction, delay = np.asarray(prediction, float), np.asarray(delay, float)
    bound = np.where(np.isfinite(delay), np.maximum(delay, 0) + cap, np.inf)
    return np.minimum(prediction, bound)
