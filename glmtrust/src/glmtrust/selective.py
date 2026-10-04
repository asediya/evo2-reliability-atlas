"""Selective prediction: return a call only where the model is confident, abstain elsewhere.

Two operating points are provided, both standard:

  * :class:`SelectivePredictor` -- refuse the least-confident fraction to hit a target *coverage*
    (the confidence functional is the margin |2p - 1|). This is the layer the paper found *does*
    transfer to a label-free species, because the confidence *ordering* survives a shift of scale
    that the calibrated probability does not.

  * :func:`precision_operating_point` -- an RCPS-style rule that chooses the score threshold with the
    most calls whose Clopper-Pearson precision lower bound still meets a target, giving a distribution
    -free guarantee on the precision of the flagged set. Reported with recall, so the cost of the
    guarantee is visible.

Neither guards the missed-positive (false-negative) rate, and the paper is explicit that the
benefit is class-asymmetric; :func:`SelectivePredictor.evaluate` reports capture separately for the
two error types so that asymmetry is never hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta
from sklearn.metrics import roc_auc_score

from . import metrics
from .conformal import ABSTAIN

__all__ = ["margin_confidence", "SelectivePredictor",
           "precision_lower_bound", "precision_operating_point",
           "group_selective_report"]


def margin_confidence(probs):
    """Confidence functional |2p - 1|: 0 at the decision boundary, 1 at a certain call."""
    return np.abs(2 * np.asarray(probs, float).ravel() - 1)


class SelectivePredictor:
    """Retain the most-confident ``coverage`` fraction; abstain on the rest. The confidence threshold
    is learned on the data passed to :meth:`fit` and then applied unchanged, so it can be fitted on
    one species and deployed on another."""

    def __init__(self, coverage: float = 0.85, decision_threshold: float = 0.5):
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        self.coverage = coverage
        self.decision_threshold = decision_threshold
        self._conf_threshold = None

    def fit(self, probs, labels=None):
        """Learn the confidence threshold that retains ``coverage`` of the calibration set.

        This is a THRESHOLD rule and :func:`group_selective_report` uses a RANK rule; they agree on a
        continuous posterior and can disagree on a coarse one. ``np.quantile`` interpolates, so the
        threshold need not be an observed confidence, and ``>=`` keeps every member of a tie block
        that straddles it -- realised coverage can therefore exceed the target when the posterior
        takes few distinct values, which is the isotonic case. Check ``realised_coverage`` rather
        than assuming the target was hit.
        """
        conf = margin_confidence(probs)
        self._conf_threshold = float(np.quantile(conf, 1 - self.coverage))
        return self

    def realised_coverage(self, probs) -> float:
        """Fraction actually retained at the fitted threshold, which ties can push above ``coverage``."""
        return float(self.keep_mask(probs).mean())

    @property
    def confidence_threshold(self) -> float:
        if self._conf_threshold is None:
            raise RuntimeError("call fit() first")
        return self._conf_threshold

    def keep_mask(self, probs):
        return margin_confidence(probs) >= self.confidence_threshold

    def predict(self, probs):
        """Class call where retained, :data:`~glmtrust.conformal.ABSTAIN` where refused."""
        p = np.asarray(probs, float).ravel()
        pred = (p >= self.decision_threshold).astype(int)
        return np.where(self.keep_mask(p), pred, ABSTAIN)

    def evaluate(self, probs, labels):
        p = np.asarray(probs, float).ravel()
        y = _labels01(labels, "selective")
        pred = (p >= self.decision_threshold).astype(int)
        conf = margin_confidence(p)
        keep = self.keep_mask(p)
        err = pred != y
        # Every operating-point number below is derived from ONE decision mask: `keep`, the rule
        # this predictor actually applies. The previous version took coverage and selective_error
        # from `keep` but capture and lift from a RANK set sized by the TARGET coverage, so on a
        # tied posterior -- the isotonic case fit() warns about -- the same dict could report zero
        # observations refused alongside capture 1.0 and lift 2.0. The rank-rule figures are still
        # available, under names that say so, because the rank curve is a legitimate diagnostic;
        # it is simply not this predictor's operating point.
        refused = ~keep
        frac_refused = float(refused.mean())
        n_err = int(err.sum())
        capture = float(err[refused].sum() / n_err) if n_err else 0.0
        rep = {
            "coverage": float(keep.mean()),
            "target_coverage": float(self.coverage),
            "selective_error": metrics.selective_error(y, pred, keep),
            "full_error": float(err.mean()),
            "capture": capture,
            "lift": (capture / frac_refused) if frac_refused > 0 else float("nan"),
            "capture_rank_rule": metrics.capture_at_coverage(y, pred, conf, self.coverage),
            "lift_rank_rule": metrics.selective_lift(y, pred, conf, self.coverage),
        }
        # class-asymmetric capture: over-calls (fp) vs missed positives (fn), the honesty rail
        for name, mask in (("false_positive", (pred == 1) & (y == 0)),
                           ("false_negative", (pred == 0) & (y == 1))):
            tot = int(mask.sum())
            refused = ~keep
            rep["capture_%s" % name] = float((mask & refused).sum() / tot) if tot else float("nan")
        return rep


def precision_lower_bound(k_correct: int, n: int, confidence: float = 0.9) -> float:
    """Clopper-Pearson lower confidence bound on precision given ``k_correct`` of ``n`` flagged."""
    if n == 0 or k_correct <= 0:
        return 0.0                                 # no successes -> the lower bound is zero
    if k_correct >= n:
        return float((1 - confidence) ** (1.0 / n))
    return float(_beta.ppf(1 - confidence, k_correct, n - k_correct + 1))


def precision_operating_point(scores, labels, target_precision: float,
                              delta: float = 0.1, n_rep: int = 50,
                              test_frac: float = 0.5, seed: int = 0):
    """RCPS-style precision guarantee. On a calibration split, choose the lowest score threshold whose
    precision lower bound (at level 1-delta) still meets ``target_precision`` -- i.e. the most calls
    that remain certifiable -- then measure precision and recall on a held-out split. Averaged over
    ``n_rep`` random splits. Returns a dict with achieved precision, recall, mean call count and the
    fraction of splits for which any threshold was feasible."""
    from sklearn.model_selection import StratifiedShuffleSplit

    scores = np.asarray(scores, float).ravel()
    labels = _labels01(labels, "selective")
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_frac, random_state=seed)
    precs, recs, ncalls, feasible = [], [], [], []
    for cal, te in sss.split(scores.reshape(-1, 1), labels):
        sc, yc = scores[cal], labels[cal]
        order = np.argsort(-sc, kind="stable")
        cum = np.cumsum(yc[order])
        ks = np.arange(1, len(order) + 1)
        lcb = np.array([precision_lower_bound(int(cum[i]), int(ks[i]), 1 - delta)
                        for i in range(len(ks))])
        ok = np.where(lcb >= target_precision)[0]
        if len(ok) == 0:
            feasible.append(0.0)
            continue
        feasible.append(1.0)
        thr = sc[order][ok.max()]
        st, yt = scores[te], labels[te]
        flagged = st >= thr
        if flagged.any():
            precs.append(float(yt[flagged].mean()))
            recs.append(float(yt[flagged].sum() / max(1, yt.sum())))
            ncalls.append(int(flagged.sum()))
    return {
        "target_precision": target_precision,
        "achieved_precision": float(np.mean(precs)) if precs else float("nan"),
        "recall": float(np.mean(recs)) if recs else 0.0,
        "n_calls": float(np.mean(ncalls)) if ncalls else 0.0,
        "feasible_fraction": float(np.mean(feasible)),
    }


def _labels01(labels, where):
    """Labels as 0/1, refusing every other encoding by name.

    TrustLayer guards this at the front door; these lower-level helpers are public and did not, so
    ``_labels01(labels, "selective")`` coerced silently. The SVM convention {-1, +1} is the dangerous one:
    it does not raise, it changes the error count (23 becomes 196 on a 2,000-variant panel) and it
    reports a capture that looks plausible. An encoding this routine cannot interpret is a caller
    error, not something to guess at.
    """
    import numpy as _np
    raw = _np.asarray(labels).ravel()
    if raw.dtype.kind == "f":
        bad = int((~_np.isfinite(raw)).sum())
        if bad:
            raise ValueError("%s: %d of %d labels are NaN or infinite; an unadjudicated variant is "
                             "not a benign one" % (where, bad, raw.size))
    out = raw.astype(int)
    dom = _np.unique(out)
    if not _np.isin(dom, (0, 1)).all():
        raise ValueError("%s: labels must be 0 (negative) or 1 (positive); got %s. Map the SVM "
                         "convention with (labels > 0).astype(int)." % (where, dom.tolist()))
    return out


def group_selective_report(probs, labels, groups, coverage: float = 0.85,
                           decision_threshold: float = 0.5, ece_bins: int = 10,
                           tie_policy: str = "whole_block"):
    """Per-group selective prediction: each group refuses its own least-confident ``1 - coverage``
    fraction, so groups can carry different operating points -- the deployment pattern for a panel of
    species each with its own confidence distribution.

    For every group this reports ``n``, error count, the number refused, errors removed, capture, lift
    (errors removed / errors, divided by the fraction refused), an error-detection AUROC (does the
    confidence separate correct from incorrect calls), and equal-width ECE. Two aggregates follow:
    ``pooled`` (sum the per-group refusals and removals) and ``macro`` (average the per-group figures).

    This is the routine that reproduces the study's per-species trust-layer table; feed it
    leave-one-group-out calibrated probabilities (see :func:`glmtrust.leave_one_group_out`).
    """
    probs = np.asarray(probs, float).ravel()
    labels = _labels01(labels, "selective")
    groups = np.asarray(groups).ravel()
    refuse = 1.0 - coverage
    per = {}
    for g in dict.fromkeys(groups.tolist()):
        m = groups == g
        p, y = probs[m], labels[m]
        finite = np.isfinite(p)
        p, y = p[finite], y[finite]
        if len(y) == 0:
            continue
        conf = margin_confidence(p)
        pred = (p >= decision_threshold).astype(int)
        err = pred != y
        n = len(y)
        k = int(round(refuse * n))
        # A rank rule has to say what it does with ties, and a coarse posterior makes that decide the
        # answer rather than decorate it. An isotonic posterior on ~11k variants takes only a couple
        # of hundred distinct values, so hundreds of variants sit exactly on the coverage boundary --
        # 597 of them on the 8,192-bp panel this package ships a fixture for. `argsort` would then
        # break the tie by position in the input array, and permuting the rows alone moved pooled
        # capture over a range of 0.043 on that panel, twice the tolerance the reproduction benchmark
        # applies to the same quantity. Two users with identical data in different row order would
        # get different refusal sets.
        #
        # Any secondary sort key derived from position has the same defect, including a stable sort
        # or an arange tie-break: the position IS the row order. So the rule is defined on the data
        # only -- refuse whole tie blocks, never part of one. Blocks are taken in increasing
        # confidence while the next block still fits inside the coverage budget, which under-refuses
        # rather than over-refuses and is reported as `realised_refused` so the caller sees the
        # difference from the nominal k.
        # `tie_policy="rank"` reproduces the study's published figures, which were computed with a
        # positional tie-break in one fixed row order; it is offered for that purpose only and is
        # order-dependent by construction. `"whole_block"` is the default because a deployed tool
        # must give one answer per dataset.
        if tie_policy == "rank":
            order = np.argsort(conf, kind="stable")
            refused_mask = np.zeros(n, bool)
            refused_mask[order[:k]] = True
            n_tied = int((conf == conf[order[k - 1]]).sum()) if 0 < k <= n else 0
        elif tie_policy == "whole_block":
            refused_mask = np.zeros(n, bool)
            taken = 0
            n_tied = 0
            for c in np.unique(conf):               # ascending, data-only
                block = conf == c
                b = int(block.sum())
                if taken + b > k:
                    n_tied = b if taken < k else 0  # the block the budget lands inside
                    break
                refused_mask |= block
                taken += b
        else:
            raise ValueError("tie_policy must be 'whole_block' or 'rank'")
        removed = int(err[refused_mask].sum())
        k_realised = int(refused_mask.sum())
        total_err = int(err.sum())
        lift = ((removed / total_err) / (k_realised / n)
                if total_err and k_realised else float("nan"))
        ed = float(roc_auc_score(err.astype(int), -conf)) if 0 < total_err < n else float("nan")
        per[_key(g)] = {
            "n": n, "errors": total_err, "refused": k_realised, "nominal_refused": k,
            "removed": removed,
            "capture": float(removed / total_err) if total_err else 0.0,
            "lift": float(lift), "err_detect_auroc": ed,
            "n_tied_at_boundary": n_tied,
            "realised_coverage": float(1.0 - k_realised / n),
            "ece": metrics.ece(y, p, n_bins=ece_bins),
        }
    n_all = sum(r["n"] for r in per.values())
    n_err = sum(r["errors"] for r in per.values())
    tot_ref = sum(r["refused"] for r in per.values())
    tot_rem = sum(r["removed"] for r in per.values())
    finite_lifts = [r["lift"] for r in per.values() if np.isfinite(r["lift"])]
    return {
        "per_group": per,
        "pooled": {
            "n": n_all, "errors": n_err, "refused": tot_ref, "removed": tot_rem,
            "error_rate": float(n_err / n_all) if n_all else float("nan"),
            "capture": float(tot_rem / n_err) if n_err else float("nan"),
            "lift": float((tot_rem / n_err) / (tot_ref / n_all)) if n_err and tot_ref else float("nan"),
        },
        "macro": {
            "lift": float(np.mean(finite_lifts)) if finite_lifts else float("nan"),
            "ece": float(np.mean([r["ece"] for r in per.values()])),
            "err_detect_auroc": float(np.nanmean([r["err_detect_auroc"] for r in per.values()])),
        },
    }


def _key(v):
    return v.item() if hasattr(v, "item") else v
