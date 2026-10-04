"""Cross-group (e.g. cross-species) calibration transfer.

The setting the paper targets is a target species with *no labels of its own*. The only honest way to
calibrate it is to fit the map on other, labelled groups and apply it unchanged -- leave-one-group-
out. :func:`leave_one_group_out` returns the out-of-group calibrated probabilities together with a
per-group report, so the transfer can be judged group by group rather than pooled.

The paper's finding, carried here as a caution rather than a feature: a transferred *probability* is
no better than a single global two-parameter sigmoid, so the value of the trust layer is the
selective-prediction ordering, not the transferred probability. This module makes the transfer easy
to run and easy to check; it does not claim the probability transfers well.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .calibration import make_calibrator

__all__ = ["transfer_calibration", "leave_one_group_out"]


def transfer_calibration(source_scores, source_labels, target_scores, method: str = "isotonic"):
    """Fit a calibration map on a labelled source and apply it to an unlabelled target's scores."""
    cal = make_calibrator(method).fit(source_scores, source_labels)
    return cal.predict_proba(target_scores)


def leave_one_group_out(scores, labels, groups, method: str = "isotonic"):
    """For each group, fit the calibration map on every *other* group and predict the held-out group.

    Returns ``(probs, report)`` where ``probs`` are out-of-group calibrated probabilities aligned with
    the input order (NaN for a group that could not be predicted, e.g. its training pool held a single
    class), and ``report`` maps each group to its ``n``, AUROC and transferred-probability ECE, plus a
    ``__macro__`` entry averaging over groups.

    The reported AUROC is signed: a group on which the score runs the wrong way reads below 0.5 and
    is flagged with ``below_chance``. It is not oriented upward, because a group whose score is
    anti-predictive is the case a user most needs to see, and reporting max(a, 1-a) would hide it.
    """
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, int).ravel()
    groups = np.asarray(groups).ravel()
    probs = np.full(len(labels), np.nan)
    report = {}
    for g in _unique_stable(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
            report[_key(g)] = {"n": int(te.sum()), "note": "insufficient class diversity"}
            continue
        cal = make_calibrator(method).fit(scores[tr], labels[tr])
        p = cal.predict_proba(scores[te])
        probs[te] = p
        a = metrics.auroc(labels[te], scores[te])
        rec = {
            "n": int(te.sum()),
            "n_positive": int(labels[te].sum()),
            "auroc": a,
            "ece": metrics.ece(labels[te], p),
        }
        if a < 0.5:
            rec["below_chance"] = True
        report[_key(g)] = rec
    scored = [r for r in report.values() if "auroc" in r]
    if scored:
        report["__macro__"] = {
            "auroc": float(np.mean([r["auroc"] for r in scored])),
            "ece": float(np.mean([r["ece"] for r in scored])),
            "n_groups": len(scored),
        }
    return probs, report


def _unique_stable(a):
    """Unique values in first-appearance order (so a report reads in the caller's group order)."""
    seen, out = set(), []
    for v in a:
        k = _key(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _key(v):
    return v.item() if hasattr(v, "item") else v
