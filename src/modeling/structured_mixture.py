"""Low-capacity convex gates optimized directly for final squared error."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder


def day_shift_mask(y, imputed):
    """Fixed supervision rule; schedule-proximity has precedence over day shift."""
    residual = np.asarray(y) - 86400
    return ~np.asarray(imputed, dtype=bool) & (residual >= 0) & (residual <= 7200)


DAY = 86400.0
DAY_BAND_COLUMN = "day_plausible"


def day_signal_frame(delay, flags, plausible_seconds=DAY/2):
    """Coordinates that let a linear gate express the day-shift band.

    The gate is linear on signed-log covariates, so it can only make the day
    candidate more likely as the delay grows. The day regime is not monotone in
    the delay: it lives in a band, and below the band the delay candidate alone
    is right. These four columns are that band, written out; all are visible in
    ranking and none reads a label.
    """
    delay = np.asarray(delay, dtype=float)
    # A schedule a whole day back cannot be less than half a day late. The
    # threshold bounds where the day candidate has any support; it is not fitted.
    plausible = (delay > plausible_seconds).astype(float)
    no_nm = np.asarray(flags.no_nm, dtype=float)
    return pd.DataFrame({
        # Signed distance from a whole day; near zero exactly for a day-back schedule.
        "day_residual": delay - DAY,
        DAY_BAND_COLUMN: plausible,
        # The interaction a linear logit cannot form: day-offset records concentrate
        # among movements with no Network Manager cross.
        "day_plausible_no_nm": plausible * np.nan_to_num(no_nm),
        # The flight's own mark; so far it only entered as a leave-one-out aggregate.
        "schedule_previous_day": np.nan_to_num(np.asarray(flags.schedule_previous_day, dtype=float)),
    })


def day_prior_estimate(plausible, shifted, smoothing=2.0, fallback=1e-5):
    """Rate of the day regime inside and outside the visible band, from history only.

    A single global rate is why the gate will not commit: the day regime is one in
    thirteen thousand overall but a sizeable minority inside the band, and the
    constant offset starts the day logit so low that reaching it costs more than
    the L2 penalty allows. Both cells are estimated on training rows only.
    """
    plausible = np.asarray(plausible, dtype=bool)
    shifted = np.asarray(shifted, dtype=bool)
    rates = {}
    for cell in (True, False):
        rows = plausible == cell
        # Smoothing keeps a cell with no observed event away from zero.
        rates[cell] = ((shifted[rows].sum() + smoothing * fallback) / (rows.sum() + smoothing)
                       if rows.any() else fallback)
    return rates


def experts(taxi, delay, three=False):
    taxi, delay = np.asarray(taxi), np.asarray(delay)
    delay = np.where(np.isfinite(delay), delay, taxi)
    second = delay if three else np.maximum(taxi, delay)
    return np.column_stack([taxi, second, taxi + 86400]) if three else np.column_stack([taxi, second])


def loss_gradient(flat, x, candidates, offset, y, penalty):
    """Exact squared error in hours + L2; no target or residual clipping."""
    beta = flat.reshape(x.shape[1], candidates.shape[1] - 1)
    logits = offset.copy()
    logits[:, 1:] += x @ beta
    weights = softmax(logits, axis=1)
    pred = (weights * candidates).sum(axis=1)
    residual = pred - y
    derivative = 2 * residual[:, None] * weights[:, 1:] * (candidates[:, 1:] - pred[:, None]) / len(y)
    gradient = x.T @ derivative + 2 * penalty * beta
    loss = np.mean(residual ** 2) + penalty * np.sum(beta ** 2)
    return loss, np.asarray(gradient).ravel()


class ConvexGate:
    # Class defaults, so gates pickled before the banded prior existed still load
    # and keep behaving exactly as they did when they were fitted.
    day_rates = None
    prior_cap = 0.01
    banded_prior_cap = 0.5

    def __init__(self, three=False, penalty=0.01, day_prior=1e-5, maxiter=250, day_rates=None,
                 prior_cap=0.01, banded_prior_cap=0.5):
        self.three, self.penalty, self.day_prior, self.maxiter = three, penalty, day_prior, maxiter
        # Rates per cell of the visible band, from day_prior_estimate; the gate keeps
        # them so a saved bundle rebuilds its own prior at inference time.
        self.day_rates = day_rates
        # The scalar cap stops a global prior handing the day candidate weight it never
        # earned. A banded prior is already conditioned, so it is allowed further.
        self.prior_cap, self.banded_prior_cap = prior_cap, banded_prior_cap

    def _row_prior(self, frame):
        if self.day_rates is None:
            return self.day_prior, self.prior_cap
        if DAY_BAND_COLUMN not in frame:
            raise ValueError(f"A banded prior needs the '{DAY_BAND_COLUMN}' column")
        band = np.asarray(frame[DAY_BAND_COLUMN], dtype=float) > 0.5
        return np.where(band, self.day_rates[True], self.day_rates[False]), self.banded_prior_cap

    def _matrix(self, frame, fit=False):
        if fit:
            self.cat = list(frame.select_dtypes(include=["object", "string", "category"]).columns)
            self.num = [c for c in frame if c not in self.cat]
            self.encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, max_categories=32)
            if self.cat:
                self.encoder.fit(frame[self.cat].fillna("NA").astype(str))
        numeric = frame[self.num].to_numpy(dtype=float, copy=True)
        numeric[~np.isfinite(numeric)] = np.nan
        # Signed log reduces leverage of extreme visible covariates, not the loss.
        numeric = np.sign(numeric) * np.log1p(np.abs(numeric))
        if fit:
            self.center = np.nan_to_num(pd.DataFrame(numeric).median().to_numpy())
            self.scale = np.nan_to_num(pd.DataFrame(numeric).std().to_numpy(), nan=1.)
            self.scale[self.scale < 1e-6] = 1.
        missing = np.isnan(numeric)
        numeric = (np.where(missing, self.center, numeric) - self.center) / self.scale
        pieces = [sparse.csr_matrix(np.column_stack([np.ones(len(frame)), numeric, missing]))]
        if self.cat:
            pieces.append(self.encoder.transform(frame[self.cat].fillna("NA").astype(str)))
        return sparse.hstack(pieces, format="csr")

    def _offset(self, probability, frame):
        p = np.clip(probability, 1e-6, 1-1e-6)
        if self.three:
            prior, cap = self._row_prior(frame)
            q = np.clip(np.broadcast_to(np.asarray(prior, dtype=float), p.shape), 1e-7, cap)
            return np.log(np.column_stack([(1-q)*(1-p), (1-q)*p, q]))
        return np.log(np.column_stack([1-p, p]))

    def fit(self, frame, taxi, delay, probability, y):
        x = self._matrix(frame, fit=True)
        candidates = experts(taxi, delay, self.three) / 3600
        offset = self._offset(probability, frame)
        result = minimize(loss_gradient, np.zeros(x.shape[1]*(candidates.shape[1]-1)),
                          args=(x, candidates, offset, np.asarray(y)/3600, self.penalty),
                          jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": 1e-10})
        self.coef = result.x.reshape(x.shape[1], -1)
        self.optimization = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "loss": float(result.fun)}
        return self

    def predict(self, frame, taxi, delay, probability):
        logits = self._offset(probability, frame)
        logits[:, 1:] += self._matrix(frame) @ self.coef
        weights = softmax(logits, axis=1)
        return (weights * experts(taxi, delay, self.three)).sum(axis=1)
