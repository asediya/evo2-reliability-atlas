"""Discrimination, calibration and selective-prediction metrics.

All functions take 1-D NumPy arrays. Labels are in {0, 1}; probabilities in [0, 1]; scores and
confidences are real-valued with higher = more deleterious / more confident respectively. Nothing
here is novel — these are the standard metrics — but they are collected so a caller never has to
reimplement expected calibration error or a risk-coverage curve, and so the trust layer and its
tests share one definition of each.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

__all__ = [
    "ece", "brier", "auroc", "auprc", "auroc_ci",
    "risk_coverage_curve", "capture_at_coverage", "selective_lift", "selective_error",
]


def _as1d(*arrs):
    return tuple(np.asarray(a).ravel() for a in arrs)


def ece(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: mean over bins of |mean label - mean probability|, weighted by
    bin population. ``strategy='uniform'`` uses equal-width bins; ``'quantile'`` uses equal-mass
    bins, which is the honest choice when probabilities pile up at one end (as they do at low
    prevalence)."""
    y_true, y_prob = _as1d(y_true, y_prob)
    y_true = y_true.astype(float)
    # The equal-width branch fixes its edges at [0, 1], so anything outside that range falls into no
    # bin and is dropped from the weighted sum without being subtracted from n. Percent-scale input
    # (0-100) therefore returned 0.0005 where its honest ECE is 0.1679, and partial NaN scaled the
    # answer by the finite fraction. The equal-mass branch is loud in both cases because its edges
    # are quantiles of the data. TrustLayer.fit already refuses non-finite scores at some length; a
    # public metric should not be quieter than the pipeline that calls it.
    _bad = int((~np.isfinite(y_prob)).sum())
    if _bad:
        raise ValueError("%d of %d probabilities are NaN or infinite; ece() will not drop them for "
                         "you, because dropping them scales the answer by the finite fraction"
                         % (_bad, y_prob.size))
    if y_prob.size and (y_prob.min() < 0.0 or y_prob.max() > 1.0):
        raise ValueError("probabilities must lie in [0, 1]; got [%g, %g]. Percent-scale input is the "
                         "usual cause: divide by 100." % (y_prob.min(), y_prob.max()))
    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    n = len(y_true)
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < len(edges) - 2 else (y_prob >= lo) & (y_prob <= hi)
        if m.any():
            total += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(total)


def brier(y_true, y_prob) -> float:
    """Brier score (mean squared error of the probability)."""
    y_true, y_prob = _as1d(y_true, y_prob)
    return float(brier_score_loss(y_true, y_prob))


def auroc(y_true, y_score, oriented: bool = False) -> float:
    """Area under the ROC curve. With ``oriented=True`` returns max(a, 1-a), i.e. the discriminative
    power regardless of the sign convention of the score."""
    y_true, y_score = _as1d(y_true, y_score)
    a = float(roc_auc_score(y_true, y_score))
    return max(a, 1 - a) if oriented else a


def auprc(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision)."""
    y_true, y_score = _as1d(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def auroc_ci(y_true, y_score, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for the AUROC. Returns (point, lo, hi)."""
    y_true, y_score = _as1d(y_true, y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    point = float(roc_auc_score(y_true, y_score))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def risk_coverage_curve(y_true, y_pred, confidence):
    """Selective risk as a function of coverage, ordering by decreasing confidence. Returns
    (coverage, risk) arrays, where coverage[k] is the fraction retained and risk[k] the error rate
    over that most-confident fraction."""
    y_true, y_pred, confidence = _as1d(y_true, y_pred, confidence)
    order = np.argsort(-confidence, kind="stable")
    err = (y_true[order] != y_pred[order]).astype(float)
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def capture_at_coverage(y_true, y_pred, confidence, coverage: float) -> float:
    """Fraction of all errors that fall in the refused (least-confident) set when the retained
    coverage is ``coverage``. This is the 'errors removed' quantity the paper reports."""
    y_true, y_pred, confidence = _as1d(y_true, y_pred, confidence)
    n = len(y_true)
    k_refuse = int(round((1 - coverage) * n))
    if k_refuse <= 0:
        return 0.0
    refused = np.argsort(confidence, kind="stable")[:k_refuse]           # least confident first
    err = y_true != y_pred
    total = int(err.sum())
    return float(err[refused].sum() / total) if total else 0.0


def selective_lift(y_true, y_pred, confidence, coverage: float) -> float:
    """Errors captured divided by the fraction refused: 1.0 is what random refusal achieves, >1 means
    the confidence ordering concentrates errors into the refused set."""
    frac_refused = 1 - coverage
    if frac_refused <= 0:
        return float("nan")
    return capture_at_coverage(y_true, y_pred, confidence, coverage) / frac_refused


def selective_error(y_true, y_pred, kept_mask) -> float:
    """Error rate over the retained (kept) subset."""
    y_true, y_pred, kept_mask = _as1d(y_true, y_pred, kept_mask)
    kept = kept_mask.astype(bool)
    if not kept.any():
        return float("nan")
    return float((y_true[kept] != y_pred[kept]).mean())
