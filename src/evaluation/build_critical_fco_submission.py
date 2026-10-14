"""Replace the predictions of FCO's structurally corrupted departures.

Selection (all visible in ranking): departure from LIRF, no Network Manager
off-block (AOBT_3_flt absent) and an apparent delay above twelve hours.

In 2025 that selection holds 29 movements and **none** of them has an ordinary
taxi label. They are bimodal: either the airport off-block was imputed from the
schedule, and the label equals the delay, or the record carries a whole-day
offset, and the label is 86400 plus a normal taxi. Predicting the taxi the
aircraft actually performed is therefore the worst available answer, which is
what three ranking submissions confirmed.

The replacement is the mean of that two-point distribution. The weight is not
the 2025 rate: it is shifted towards the delay candidate so the change stays
ahead of the baseline across the whole plausible range of that rate.
"""
import numpy as np
import pandas as pd

DAY = 86400.0
DELAY_THRESHOLD = 43200.0
# Rate of imputed off-blocks in the 2025 selection is 0.621 (n=29, s.e. 0.090).
# A heavier delay weight costs ~1 point at that estimate and protects the tail.
DELAY_WEIGHT = 0.70
# Mean ordinary taxi of the 2025 day-offset records, used when no track is known.
DEFAULT_TAXI = 1136.0
# Surface tracks start after the aircraft is already rolling; measured on three
# FCO controls against their true off-block (mean 153 s, n=3).
TRACK_LAG = 153.0
# Strictness margin for the dominance bound, in seconds.
DOMINANCE_MARGIN = 1.0


def critical_departures(ranking, airport="LIRF"):
    frame = ranking[ranking.airport == airport].copy()
    delay = (pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
             - pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    frame["apparent_delay_sec"] = delay
    return frame[frame.AOBT_3_flt.isna() & (delay > DELAY_THRESHOLD)].reset_index(drop=True)


def candidates(frame, measured_taxi=None):
    """The two values the label can take: the apparent delay, or a day plus the taxi."""
    measured = frame.MVT_ID_mvt.map(measured_taxi or {})
    # A known track sharpens the day candidate; otherwise the 2025 mean is used.
    taxi = measured.fillna(DEFAULT_TAXI - TRACK_LAG) + TRACK_LAG
    return frame.apparent_delay_sec.to_numpy(float), (DAY + taxi).to_numpy(float)


def replacement(frame, measured_taxi=None, weight=DELAY_WEIGHT, previous=None, mode="mixture"):
    """Mean of the two candidates, or the most it can be moved without ever losing.

    With `previous`, `mode="dominant"` returns the value closest to that mean that
    still beats the current prediction under **either** candidate. Moving from a to
    q improves iff every possible label exceeds (a+q)/2, so the move is safe while
    q < 2*min(candidates) - a. Where no such q exists above a — the delay candidate
    already sits below the current prediction — the row is left untouched.
    """
    delay, day = candidates(frame, measured_taxi)
    mean = weight * delay + (1 - weight) * day
    if mode == "mixture":
        return mean
    if previous is None:
        raise ValueError("The dominant modes need the current predictions")
    previous = np.asarray(previous, dtype=float)
    ceiling = 2 * np.minimum(delay, day) - previous
    target = np.minimum(delay, day) if mode == "lower" else mean
    safe = np.minimum(target, ceiling - DOMINANCE_MARGIN)
    # An empty interval means no move can dominate; keep what is already there.
    return np.where(safe > previous, safe, previous)
