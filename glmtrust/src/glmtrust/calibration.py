"""Probability calibration for a 1-D variant-effect score.

A genomic language model emits a score whose ordering is informative but whose scale is arbitrary.
Calibration maps that score to a probability that a variant is deleterious. Two standard maps are
provided:

  * :class:`IsotonicCalibrator` -- monotone, non-parametric (``sklearn`` isotonic regression);
  * :class:`PlattCalibrator`    -- a two-parameter logistic (Platt scaling).

Neither is new. What the paper found, and what this module is careful about, is that the *isotonic*
map overfits at low prevalence unless it is fitted out-of-fold, so :func:`cross_conformal_calibrate`
returns cross-conformal (out-of-fold) probabilities and is what the pipeline uses by default.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

__all__ = ["Calibrator", "IsotonicCalibrator", "PlattCalibrator", "make_calibrator",
           "cross_conformal_calibrate"]


class Calibrator:
    """Base class: ``fit(scores, labels)`` then ``predict_proba(scores) -> P(deleterious)``."""

    def fit(self, scores, labels):
        raise NotImplementedError

    def predict_proba(self, scores):
        raise NotImplementedError

    def fit_predict(self, scores, labels):
        return self.fit(scores, labels).predict_proba(scores)


class IsotonicCalibrator(Calibrator):
    """Monotone non-parametric calibration. Clips out-of-range test scores to the fitted range."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores, labels):
        scores = np.asarray(scores, float).ravel()
        labels = np.asarray(labels, float).ravel()
        self._iso.fit(scores, labels)
        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, float).ravel()
        return np.clip(self._iso.predict(scores), 0.0, 1.0)


class PlattCalibrator(Calibrator):
    """Two-parameter logistic (Platt) scaling: sigmoid(a * score + b)."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")
        self._const = None                      # degenerate fallback if only one class is present

    def fit(self, scores, labels):
        scores = np.asarray(scores, float).reshape(-1, 1)
        labels = np.asarray(labels, int).ravel()
        if len(np.unique(labels)) < 2:
            self._const = float(labels.mean())
            return self
        self._const = None
        self._lr.fit(scores, labels)
        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, float).reshape(-1, 1)
        if self._const is not None:
            return np.full(len(scores), self._const)
        return self._lr.predict_proba(scores)[:, 1]


_REGISTRY = {"isotonic": IsotonicCalibrator, "platt": PlattCalibrator}


def make_calibrator(method: str = "isotonic") -> Calibrator:
    """Factory: ``make_calibrator('isotonic' | 'platt')``."""
    try:
        return _REGISTRY[method]()
    except KeyError:
        raise ValueError("method must be one of %s" % sorted(_REGISTRY)) from None


def cross_conformal_calibrate(scores, labels, method: str = "isotonic",
                              n_splits: int = 5, seed: int = 0):
    """Out-of-fold calibrated probabilities: for each fold, fit on the other folds and predict the
    held-out fold, so no variant is calibrated on itself. This is the low-variance, honest estimate
    the paper reports; the returned array aligns with the input order."""
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, int).ravel()
    out = np.full(len(labels), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(scores.reshape(-1, 1), labels):
        cal = make_calibrator(method).fit(scores[tr], labels[tr])
        out[te] = cal.predict_proba(scores[te])
    return out
