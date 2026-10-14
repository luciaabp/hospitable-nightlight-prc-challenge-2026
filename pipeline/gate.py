"""Convex three-candidate gate optimized for the final squared error.

    prediction = w1 r + w2 d + w3 (r + 86400),   w >= 0,  sum w = 1

Frozen copy of the ``gate3_quality`` variant of ``src/modeling/structured_mixture.py``.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder

DAY = 86400.0


def candidates(taxi, delay) -> np.ndarray:
    taxi, delay = np.asarray(taxi, dtype=float), np.asarray(delay, dtype=float)
    delay = np.where(np.isfinite(delay), delay, taxi)
    return np.column_stack([taxi, delay, taxi + DAY])


def loss_gradient(flat, x, cands, offset, y, penalty):
    """Exact squared error in hours + L2."""
    beta = flat.reshape(x.shape[1], cands.shape[1] - 1)
    logits = offset.copy()
    logits[:, 1:] += x @ beta
    weights = softmax(logits, axis=1)
    pred = (weights * cands).sum(axis=1)
    residual = pred - y
    derivative = 2 * residual[:, None] * weights[:, 1:] * (cands[:, 1:] - pred[:, None]) / len(y)
    gradient = x.T @ derivative + 2 * penalty * beta
    loss = np.mean(residual ** 2) + penalty * np.sum(beta ** 2)
    return loss, np.asarray(gradient).ravel()


class ConvexGate:
    """Linear logits on visible features; the offset starts from the imputed-block probability."""

    prior_cap = 0.01

    def __init__(self, penalty=0.01, day_prior=1e-5, maxiter=250):
        self.penalty, self.day_prior, self.maxiter = penalty, day_prior, maxiter

    def _matrix(self, frame, fit=False):
        if fit:
            self.cat = list(frame.select_dtypes(include=["object", "string", "category"]).columns)
            self.num = [c for c in frame if c not in self.cat]
            self.encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, max_categories=32)
            if self.cat:
                self.encoder.fit(frame[self.cat].fillna("NA").astype(str))
        numeric = np.array(frame[self.num], dtype=float)  # copy: to_numpy may be read-only
        numeric[~np.isfinite(numeric)] = np.nan
        # The signed log reduces the leverage of extreme covariates, not the loss.
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

    def _offset(self, probability):
        p = np.clip(probability, 1e-6, 1 - 1e-6)
        q = np.clip(np.broadcast_to(np.asarray(self.day_prior, dtype=float), p.shape), 1e-7, self.prior_cap)
        return np.log(np.column_stack([(1 - q) * (1 - p), (1 - q) * p, q]))

    def fit(self, frame, taxi, delay, probability, y):
        x = self._matrix(frame, fit=True)
        cands = candidates(taxi, delay) / 3600
        offset = self._offset(np.asarray(probability, dtype=float))
        result = minimize(loss_gradient, np.zeros(x.shape[1] * (cands.shape[1] - 1)),
                          args=(x, cands, offset, np.asarray(y) / 3600, self.penalty),
                          jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": 1e-10})
        self.coef = result.x.reshape(x.shape[1], -1)
        self.optimization = {"success": bool(result.success), "message": str(result.message),
                             "iterations": int(result.nit), "loss": float(result.fun)}
        return self

    def weights(self, frame, probability):
        logits = self._offset(np.asarray(probability, dtype=float))
        logits[:, 1:] += self._matrix(frame) @ self.coef
        return softmax(logits, axis=1)

    def predict(self, frame, taxi, delay, probability):
        return (self.weights(frame, probability) * candidates(taxi, delay)).sum(axis=1)


def gate_frame(X: pd.DataFrame, quality: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Gate inputs: 36 features, 24 quality features and the base pieces (``clean_taxi``, ``delay``, ``probability``)."""
    frame = pd.concat([X.reset_index(drop=True), quality.reset_index(drop=True)], axis=1)
    frame["base_taxi"] = np.asarray(base["clean_taxi"], dtype=float)
    frame["base_delay"] = np.asarray(base["delay"], dtype=float)
    frame["base_probability"] = np.asarray(base["probability"], dtype=float)
    frame["base_gap"] = frame.base_delay - frame.base_taxi
    return frame


def fit_gate(X, quality, base, y, penalty, day_prior) -> ConvexGate:
    frame = gate_frame(X, quality, base)
    return ConvexGate(penalty, day_prior).fit(frame, frame.base_taxi, frame.base_delay,
                                              frame.base_probability, y)


def predict_gate(gate: ConvexGate, X, quality, base) -> np.ndarray:
    frame = gate_frame(X, quality, base)
    return gate.predict(frame, frame.base_taxi, frame.base_delay, frame.base_probability)
