"""LIRF: predictions consistent with the regime where the recorded block cannot be an ordinary one.

The recorded block date is rebuilt as the one closest to an anchor (EOBT with NM, the schedule without). When the real
block falls more than 12 h after the anchor, the label can only be ``d = takeoff - scheduled`` (block imputed with the
schedule) or ``taxi + 86400 k`` (day shift), never an ordinary taxi. ``pi`` (probability of imputation) averages the
imputed-block classifier and an empirical Bayes rate per normalized flight number.

Modes: ``mixture`` = pi d + (1 - pi) alt; ``dominant`` moves from the previous prediction towards it only as far as
neither label gets worse; ``banded`` uses the mean in the delay bands where the shift is most likely and ``dominant``
elsewhere. The label is only supervision in training.
"""
import joblib
import numpy as np
import pandas as pd

from .config import FLIGHT, ID, ROME, SCHED, TIME, TIMELINE_FEATURES
from .data import imputed_block

DAY = 86400.0
WINDOW = 12 * 3600.0
LONG_DELAY = 6 * 3600.0
TYPICAL_TAXI = 1200.0
PRIOR_STRENGTH = 2.0
DOMINANCE_MARGIN = 1.0
# ``banded`` delay bands (hours): (12, 15.5], (15.5, 24] and > 24.
BAND_EDGES_H = (15.5, 24.0)
MODES = ("none", "dominant", "mixture", "banded")
EOBT, AOBT = "EOBT_1_flt", "AOBT_3_flt"
NM_TIMES = ["takeoff_minus_EOBT_1_flt", "AOBT_3_flt_minus_EOBT_1_flt", "takeoff_minus_AOBT_3_flt"]
# The flight's own time features derived from NM times: all but the takeoff seconds.
NM_TIMELINE = [c for c in TIMELINE_FEATURES if c != "tl_takeoff_sec"]


def anchor_offset(frame: pd.DataFrame) -> np.ndarray:
    """Takeoff - anchor in seconds: EOBT with NM, scheduled time without."""
    takeoff = pd.to_datetime(frame[TIME], utc=True)
    nm = frame[AOBT].notna() & frame[EOBT].notna()
    anchor = pd.to_datetime(frame[SCHED], utc=True).where(~nm, pd.to_datetime(frame[EOBT], utc=True))
    return (takeoff - anchor).dt.total_seconds().to_numpy(float)


def shifted_nm(frame: pd.DataFrame, taxi) -> np.ndarray:
    """Flights with NM whose EOBT is already more than 12 h off: their NM times are not that day's."""
    nm = (frame[AOBT].notna() & frame[EOBT].notna()).to_numpy()
    return nm & (anchor_offset(frame) - np.asarray(taxi, float) > WINDOW)


def day_candidate(taxi, gap) -> np.ndarray:
    """``r + 86400 k``: the days that bring the block within 12 h of the anchor."""
    k = np.maximum(np.round(np.asarray(gap, float) / DAY), 1)
    return np.asarray(taxi, float) + DAY * k


def flight_key(flight) -> str | None:
    """Flight number without the repeated prefix of anomalous records: TARAR753 -> TAR753."""
    if not isinstance(flight, str):
        return None
    prefix, rest = flight[:3], flight[3:]
    for k in (3, 2, 1):
        if len(rest) > k and rest.startswith(prefix[-k:]):
            return prefix + rest[k:]
    return flight


def flight_prior(training: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, float]:
    """Imputed / departures per flight number (LIRF, without NM, delay > 6 h) and the tail rate."""
    lirf = training[training.airport == ROME]
    delay = (pd.to_datetime(lirf[TIME], utc=True) - pd.to_datetime(lirf[SCHED], utc=True)).dt.total_seconds()
    imputed = pd.Series(imputed_block(lirf, tolerance), index=lirf.index)
    no_nm = lirf[AOBT].isna()
    long_ = lirf[no_nm & (delay > LONG_DELAY)]
    table = pd.DataFrame({"fn": long_[FLIGHT].map(flight_key), "imp": imputed[long_.index]})
    table = table.dropna(subset=["fn"]).groupby("fn").imp.agg(["sum", "size"])
    tail = imputed[no_nm & (delay - TYPICAL_TAXI > WINDOW)]
    return table, float((tail.sum() + 1) / (len(tail) + 2))


def band_prior(training: pd.DataFrame, tolerance: float) -> np.ndarray:
    """Imputed rate of the 2025 tail per delay band (``BAND_EDGES_H``), with a Beta(1, 1) prior."""
    lirf = training[training.airport == ROME]
    delay = (pd.to_datetime(lirf[TIME], utc=True) - pd.to_datetime(lirf[SCHED], utc=True)).dt.total_seconds()
    imputed = pd.Series(imputed_block(lirf, tolerance), index=lirf.index)
    keep = lirf[AOBT].isna() & (delay - TYPICAL_TAXI > WINDOW)
    band = delay_band(delay[keep])
    hits = np.bincount(band, weights=imputed[keep].to_numpy(float), minlength=len(BAND_EDGES_H) + 1)
    size = np.bincount(band, minlength=len(BAND_EDGES_H) + 1)
    return (hits + 1) / (size + 2)


def delay_band(delay) -> np.ndarray:
    """Delay band index: 0 up to 15.5 h, 1 up to 24 h, 2 above."""
    return np.digitize(np.asarray(delay, float) / 3600, BAND_EDGES_H, right=True)


def empirical_probability(flights, table: pd.DataFrame, prior: float) -> np.ndarray:
    """Empirical Bayes: long-delay history of the flight number, shrunk towards ``prior``."""
    fn = pd.Series(np.asarray(flights, dtype=object)).map(flight_key)  # no number: no history
    hits, size = fn.map(table["sum"]).fillna(0).to_numpy(float), fn.map(table["size"]).fillna(0).to_numpy(float)
    return (hits + PRIOR_STRENGTH * prior) / (size + PRIOR_STRENGTH)


def dominant_move(previous, target, *candidates, margin: float = DOMINANCE_MARGIN) -> np.ndarray:
    """As close to ``target`` as possible while improving on ``previous`` whichever candidate is true."""
    previous, target = np.asarray(previous, float), np.asarray(target, float)
    low, high = np.min(candidates, axis=0), np.max(candidates, axis=0)
    up = np.minimum(target, 2 * low - previous - margin)
    down = np.maximum(target, 2 * high - previous + margin)
    return np.where((previous < low) & (up > previous), up,
                    np.where((previous > high) & (down < previous), down, previous))


def clean_taxi_without_nm(model_dirs, X: pd.DataFrame, timeline: bool = False) -> np.ndarray:
    """Mean clean taxi of the seeds with the NM times emptied (and, with ``timeline``, the NM-derived ``tl_*``)."""
    clean = X.copy()
    clean[NM_TIMES] = np.nan
    if timeline:
        clean[[c for c in NM_TIMELINE if c in clean.columns]] = np.nan
    return np.mean([joblib.load(d / "final_base.joblib")["clean_taxi"].predict(clean) for d in model_dirs], axis=0)


def components(lirf: pd.DataFrame, X: pd.DataFrame, parts: dict, fnhist_months, model_dirs,
               timeline: bool = False) -> pd.DataFrame:
    """Per-departure LIRF pieces (seed mean of ``structured.predict``) and gap to the anchor."""
    first = next(iter(parts.values()))
    if not np.array_equal(first[ID].to_numpy(), lirf[ID].to_numpy()):
        raise ValueError("The LIRF pieces do not follow the ranking order")
    out = pd.DataFrame({ID: lirf[ID].to_numpy(), FLIGHT: lirf[FLIGHT].to_numpy(),
                        "delay": first.delay.to_numpy(float)})
    for column in ["clean_taxi", "probability", "probability_fnhist"]:
        out[column] = np.mean([p[column].to_numpy(float) for p in parts.values()], axis=0)
    dropped = shifted_nm(lirf, out.clean_taxi)
    out["nm_times_dropped"] = dropped
    if dropped.any():
        out.loc[dropped, "clean_taxi"] = clean_taxi_without_nm(model_dirs, X.loc[dropped], timeline)
    out["gap"] = anchor_offset(lirf) - out.clean_taxi.to_numpy()
    out["classifier_p"] = np.where(np.isin(first.month.to_numpy(), list(fnhist_months)),
                                   out.probability_fnhist, out.probability)
    return out


def tail(components: pd.DataFrame, previous, mode: str, table: pd.DataFrame, prior: float,
         bands=None) -> pd.DataFrame:
    """Tail rows with their new prediction.

    ``components`` (one row per LIRF departure): ``ID``, ``FLIGHT_mvt``, ``gap``, ``delay``, ``clean_taxi`` and
    ``classifier_p``; ``previous``: current prediction, same order; ``bands``: ``band_prior`` rates (``banded`` only).
    """
    if mode not in MODES[1:]:
        raise ValueError(f"Unknown mode: {mode}")
    rows = np.asarray(components.gap, float) > WINDOW
    out = components[rows].copy()
    out["previous"] = np.asarray(previous, float)[rows]
    out["alt"] = day_candidate(out.clean_taxi, out.gap)
    out["empirical_p"] = empirical_probability(out[FLIGHT], table, prior)
    out["pi"] = 0.5 * (out.classifier_p + out.empirical_p)
    out["new"] = out.pi * out.delay + (1 - out.pi) * out.alt
    if mode in ("dominant", "banded"):
        out["new"] = dominant_move(out.previous, out.new, out.delay, out.alt)
    if mode == "banded":
        if bands is None:
            raise ValueError("banded needs the per-band rates (band_prior)")
        out["band_p"] = np.asarray(bands, float)[delay_band(out.delay)]
        shift_likely = (out.band_p < 0.5).to_numpy()
        out.loc[shift_likely, "pi"] = 0.5 * (out.classifier_p + out.band_p)[shift_likely]
        mean = out.pi * out.delay + (1 - out.pi) * out.alt
        out.loc[shift_likely, "new"] = mean[shift_likely]
    return out
