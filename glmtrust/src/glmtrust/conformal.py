"""Split conformal prediction for a binary deleterious/benign call.

Given calibrated probabilities on an exchangeable calibration set, conformal prediction returns, for
each new variant, a *set* of labels that, under exchangeability of the calibration and test data, contains the truth with a chosen frequency. For a
binary call the set is one of:

  {positive}            a confident deleterious call
  {benign}              a confident benign call
  {negative, positive}  the model is not certain -- abstain
  {}                    neither label conforms -- an outlier; also treated as abstain

:class:`MondrianConformal` computes a separate threshold per class, so the coverage guarantee (valid under exchangeability of calibration and test data)
targets coverage *within each class* -- the property that matters when the classes are as imbalanced as positive and
benign variants, and the reason a single marginal threshold (:class:`SplitConformal`) can silently
under-cover the rare class. The method is standard; the class-conditional form is the one the paper
argues for.

The nonconformity score of label ``c`` at a point with P(positive)=p is ``A = 1 - p(c)``:
``1 - p`` for the positive label, ``p`` for the negative label. A label is admitted to the set when
its nonconformity does not exceed the (finite-sample corrected) 1-alpha quantile of the calibration
nonconformities for that stratum.
"""
from __future__ import annotations

import numpy as np

__all__ = ["conformal_quantile", "SplitConformal", "MondrianConformal", "ABSTAIN"]

ABSTAIN = -1


def conformal_quantile(scores, alpha: float) -> float:
    """The finite-sample-valid conformal threshold: the k-th smallest nonconformity score with
    k = ceil((n + 1)(1 - alpha)). Returns +inf when that index exceeds n (admit the label always),
    which is the correct behaviour for a calibration set too small to certify level alpha."""
    scores = np.asarray(scores, float).ravel()
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


class _BaseConformal:
    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._fitted = False

    def predict_set(self, probs, alpha=None):
        """Return an (n, 2) boolean array; column 0 = negative in set, column 1 = positive in set."""
        raise NotImplementedError

    def predict(self, probs, alpha=None):
        """Decision per variant: 1 (positive), 0 (negative), or ABSTAIN for a non-singleton set."""
        s = self.predict_set(probs, alpha)
        out = np.full(len(s), ABSTAIN)
        out[(~s[:, 0]) & s[:, 1]] = 1
        out[s[:, 0] & (~s[:, 1])] = 0
        return out

    def evaluate(self, probs, labels, alpha=None):
        """Empirical coverage and abstention on a labelled set."""
        probs = np.asarray(probs, float).ravel()
        labels = np.asarray(labels, int).ravel()
        s = self.predict_set(probs, alpha)
        in_set = s[np.arange(len(labels)), labels]
        abstain = (s.sum(1) != 1)
        rep = {"coverage": float(in_set.mean()),
               "abstention_rate": float(abstain.mean())}
        for c in (0, 1):
            m = labels == c
            rep["coverage_class_%d" % c] = float(s[m, c].mean()) if m.any() else float("nan")
        return rep


class SplitConformal(_BaseConformal):
    """Marginal split conformal: one threshold pooled over both classes."""

    def fit(self, cal_probs, cal_labels):
        p = np.clip(np.asarray(cal_probs, float).ravel(), 0, 1)
        y = np.asarray(cal_labels, int).ravel()
        self._cal = np.where(y == 1, 1 - p, p)          # nonconformity of the true label
        self._fitted = True
        return self

    def predict_set(self, probs, alpha=None):
        if not self._fitted:
            raise RuntimeError("call fit() first")
        a = self.alpha if alpha is None else alpha
        p = np.clip(np.asarray(probs, float).ravel(), 0, 1)
        q = conformal_quantile(self._cal, a)
        return np.column_stack([p <= q, (1 - p) <= q])


class MondrianConformal(_BaseConformal):
    """Class-conditional split conformal: a separate threshold for each class, so coverage holds
    within the positive class and within the negative class rather than only on average."""

    def fit(self, cal_probs, cal_labels):
        p = np.clip(np.asarray(cal_probs, float).ravel(), 0, 1)
        y = np.asarray(cal_labels, int).ravel()
        self._cal = {0: p[y == 0], 1: 1 - p[y == 1]}    # per-class true-label nonconformities
        self._fitted = True
        return self

    def predict_set(self, probs, alpha=None):
        if not self._fitted:
            raise RuntimeError("call fit() first")
        a = self.alpha if alpha is None else alpha
        p = np.clip(np.asarray(probs, float).ravel(), 0, 1)
        q0 = conformal_quantile(self._cal[0], a)
        q1 = conformal_quantile(self._cal[1], a)
        return np.column_stack([p <= q0, (1 - p) <= q1])
