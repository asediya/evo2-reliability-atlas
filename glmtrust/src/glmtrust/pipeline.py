"""The trust layer as one object: calibrate a score, then attach conformal and selective abstention.

:class:`TrustLayer` is the front door. It wires together the four standard components in this package
in the order the paper uses them:

    raw score  ->  calibrated probability  ->  conformal prediction set  ->  selective call / abstain

Two things it is careful about, because they are where naive glue code goes wrong:

  * the conformal layer is fitted on an INNER calibration split the calibrator never saw, so
    its quantile carries the split-conformal guarantee rather than an in-sample one; the
    selective layer is fitted on out-of-fold probabilities for the same reason;
  * :meth:`evaluate` reports cross-validated (or leave-one-group-out) numbers, so every reported
    figure is measured on variants the model was not fitted on.

:meth:`fit` then :meth:`predict` gives deployment behaviour (all data used to fit, applied to genuinely
new variants). :meth:`evaluate` gives an honest performance report on labelled data.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import metrics
from .calibration import cross_conformal_calibrate, make_calibrator
from .conformal import ABSTAIN, MondrianConformal, SplitConformal
from .selective import SelectivePredictor, margin_confidence
from .transfer import leave_one_group_out

__all__ = ["TrustLayer"]

_CONFORMAL = {"mondrian": MondrianConformal, "split": SplitConformal}


class TrustLayer:
    """Calibration + conformal + selective prediction for a variant-effect score.

    Parameters
    ----------
    calibration : {'platt', 'isotonic'}
        The probability map, defaulting to Platt.

        This default is deliberate: the class's
        own output is a per-variant selective call, and a transferred isotonic posterior is
        tie-heavy at that job -- across the paper's 11,130 variants it takes 248 distinct values
        against Platt's 8,792, so most variants land inside a tie block and the selective
        ordering is decided by sort order rather than by confidence. The paper reports isotonic
        for the ECE analysis and Platt for everything per-variant; the front door therefore
        defaults to the one that matches what :meth:`predict` returns. Pass
        ``calibration="isotonic"`` to reproduce the calibration-error arm.
    conformal : {'mondrian', 'split'}
        Class-conditional (Mondrian) or marginal split conformal.
    alpha : float
        Target conformal miscoverage (1 - alpha coverage per class for Mondrian).
    coverage : float
        Retained fraction for the selective predictor (e.g. 0.85 keeps the most-confident 85%).
    n_splits, seed : int
        Cross-validation folds and RNG seed used for out-of-fold calibration and for :meth:`evaluate`.
    """

    def __init__(self, calibration: str = "platt", conformal: str = "mondrian",
                 alpha: float = 0.1, coverage: float = 0.85, n_splits: int = 5, seed: int = 0):
        if conformal not in _CONFORMAL:
            raise ValueError("conformal must be 'mondrian' or 'split'")
        self.calibration = calibration
        self.conformal = conformal
        self.alpha = alpha
        self.coverage = coverage
        self.n_splits = n_splits
        self.seed = seed
        self._fitted = False

    # ---- deployment -------------------------------------------------------
    @staticmethod
    def _validate(scores, labels, where):
        """The input contract, shared by fit(), evaluate() and summary().

        These checks lived in fit() alone, which left evaluate() -- the method the paper documents
        -- unprotected: NaN scores through evaluate() raised the same scikit-learn message the
        guards were written to replace. Two failure modes are new here and both were silent:

        Labels are checked for finiteness BEFORE the int cast. ``np.asarray([np.nan], int)`` is
        platform-dependent and on most builds yields 0, so an unadjudicated variant became a benign
        one with nothing louder than a RuntimeWarning. An unadjudicated variant is a decision, not
        a benign call.

        Labels are then checked to be 0/1. The SVM convention {-1, +1} passed silently and changed
        12 of 21 reported fields, because -1 is a legal numpy column index: benign rows were read
        out of the positive column, the Mondrian ``y == 0`` stratum was empty, and a fitted layer
        could never call a variant positive. {0, 2} and {1, 2} already raised; only the negative
        encodings got through.
        """
        scores = np.asarray(scores, float).ravel()
        raw = np.asarray(labels).ravel()
        # this package targets species with no curated labels of their own, where an
        # empty panel, a length mismatch, or a calibration slice carrying only one class is a
        # plausible FIRST input. PlattCalibrator was hardened for the single-class case; the front
        # door was not, so a reader hitting it met an opaque failure from inside the calibrator
        # instead of a statement of what is wrong. Fail here, by name.
        if scores.size != raw.size:
            raise ValueError("scores and labels differ in length (%d vs %d)"
                             % (scores.size, raw.size))
        if scores.size == 0:
            raise ValueError("no variants supplied: TrustLayer.%s needs a non-empty panel" % where)
        if not np.isfinite(scores).any():
            raise ValueError("no finite scores supplied")
        # PARTIAL missingness used to fall straight through to scikit-learn, which answered
        # "Input X contains NaN. LogisticRegression does not accept missing values" -- an error
        # about somebody else's estimator, in a package whose whole subject is that missingness is
        # class-dependent and must be a stated decision rather than a silent drop.
        _bad = int((~np.isfinite(scores)).sum())
        if _bad:
            raise ValueError(
                "%d of %d scores are NaN or infinite. glmtrust will not drop them for you: which "
                "variants a score cannot reach is itself a result, and discarding them silently "
                "changes what the remaining accuracy means. Either restrict the panel yourself "
                "(scores[np.isfinite(scores)]) or impute explicitly, and state which you did."
                % (_bad, scores.size))
        if raw.dtype.kind == "f":
            _nan = int((~np.isfinite(raw)).sum())
            if _nan:
                raise ValueError(
                    "%d of %d labels are NaN or infinite. An unadjudicated variant is not a benign "
                    "one, and casting it to int would make it benign silently. Drop those rows or "
                    "adjudicate them, and say which you did." % (_nan, raw.size))
        labels = raw.astype(int)
        _dom = np.unique(labels)
        if not np.isin(_dom, (0, 1)).all():
            raise ValueError(
                "labels must be 0 (negative) or 1 (positive); got %s. The SVM convention {-1, +1} "
                "is the usual cause and is the dangerous one, because -1 indexes the last column of "
                "the conformal set rather than raising: it silently reads benign variants out of the "
                "positive column. Map with (labels > 0).astype(int)." % _dom.tolist())
        return scores, labels

    def fit(self, scores, labels, groups=None):
        """Fit the deployable trust layer. With ``groups`` the calibration used to set conformal and
        selective thresholds is leave-one-group-out (the label-free-target setting)."""
        scores, labels = self._validate(scores, labels, "fit")
        present = np.unique(labels)
        if present.size < 2:
            raise ValueError(
                "calibration needs both classes; got only label %s across %d variants. In a "
                "label-poor target this usually means the calibration slice was taken from the "
                "target species rather than from the label-rich donors -- fit on the donors and "
                "pass the target through predict()." % (present.tolist(), labels.size))
        self._calibrator = make_calibrator(self.calibration).fit(scores, labels)
        oof = self._oof_probabilities(scores, labels, groups)
        m = np.isfinite(oof)
        # Leave-one-group-out skips a group whose training pool or own slice holds a single class.
        # If EVERY group is skipped -- two groups, one a singleton, is enough -- the out-of-fold
        # vector is all NaN and the conformal quantile died on an empty array with
        # "IndexError: index -1 is out of bounds for axis 0 with size 0".
        if not m.any():
            raise ValueError(
                "no out-of-fold probability could be computed: every group's calibration pool "
                "carried a single class. With groups= this needs at least two groups that each "
                "hold both labels; check the group sizes before calling fit(groups=...).")
        self._conformal_model = _CONFORMAL[self.conformal](alpha=self.alpha).fit(oof[m], labels[m])
        self._selective_model = SelectivePredictor(coverage=self.coverage).fit(oof[m])
        self._fitted = True
        return self

    def predict(self, scores):
        """Apply the fitted layer to new scores. Returns a dict of aligned arrays: ``probability``,
        ``conformal_set`` (n x 2 bool: negative, positive), ``conformal_decision`` and
        ``selective_decision`` (0 negative / 1 positive / ABSTAIN)."""
        self._check_fitted()
        p = self._calibrator.predict_proba(np.asarray(scores, float).ravel())
        return {
            "probability": p,
            "conformal_set": self._conformal_model.predict_set(p),
            "conformal_decision": self._conformal_model.predict(p),
            "selective_decision": self._selective_model.predict(p),
        }

    # ---- honest evaluation ------------------------------------------------
    def evaluate(self, scores, labels, groups=None):
        """Cross-validated (or leave-one-group-out) performance. Every test variant is scored by a
        layer fitted without it. Returns a nested dict: discrimination, calibration, conformal and
        selective blocks.

        ON THE SELECTIVE BLOCK AND TIE POLICY. The capture and lift reported here rank variants by
        confidence within each group and cut at the coverage level, which splits a tie block when the
        cut falls inside one. **That is the policy the accompanying study publishes**: 253 of 415
        errors removed at 1,665 refusals on its 8,192-bp panel, a capture of 0.6096, and 316 of 893 at
        1,669 refusals on the 1,001-bp panel, 35.4% with a lift of 2.36. Those figures fit each
        calibration map on the whole training fold, as ``leave_one_group_out`` does; this method fits
        it on half of the fold and the conformal layer on the other half, and on the 8,192-bp fixture
        reads a capture of 0.6124.

        ``glmtrust.selective.group_selective_report`` refuses whole tie blocks by default and never part
        of one, which is deterministic under any row order but answers a different question and
        reports a different number. Reproduce a published selective figure with
        ``leave_one_group_out`` and ``group_selective_report(..., tie_policy="rank")``, as
        ``benchmarks/reproduce_paper_trust_layer.py`` does; reach for the default tie policy when you
        need order-independence rather than agreement with the paper.
        """
        scores, labels = self._validate(scores, labels, "evaluate")
        n = len(labels)
        prob = np.full(n, np.nan)
        cset = np.zeros((n, 2), bool)
        for tr, te in self._splits(scores, labels, groups):
            if len(np.unique(labels[tr])) < 2:
                continue
            # Split conformal needs a calibration set the CALIBRATOR HAS NOT SEEN. This used to
            # fit the calibrator on all of `tr` and then fit the conformal quantile on that same
            # fold's in-sample probabilities and the same outcomes, which is not split conformal
            # and carries none of its guarantee: in-sample conformity scores are optimistically
            # small, so the quantile is too tight. It also contradicted this module's own
            # docstring, which promises out-of-fold probabilities. An inner split fixes it.
            rs = np.random.default_rng(self.seed + 1)
            idx = rs.permutation(tr)
            cut = max(1, len(idx) // 2)
            fit_i, cal_i = idx[:cut], idx[cut:]
            if len(cal_i) == 0 or len(np.unique(labels[fit_i])) < 2 or len(np.unique(labels[cal_i])) < 2:
                fit_i = cal_i = tr          # too small to split: fall back, and say so
                self._conformal_split_degraded = True
            cal = make_calibrator(self.calibration).fit(scores[fit_i], labels[fit_i])
            p_cal, p_te = cal.predict_proba(scores[cal_i]), cal.predict_proba(scores[te])
            prob[te] = p_te
            conf = _CONFORMAL[self.conformal](alpha=self.alpha).fit(p_cal, labels[cal_i])
            cset[te] = conf.predict_set(p_te)

        m = np.isfinite(prob)
        y, p = labels[m], prob[m]
        pred = (p >= 0.5).astype(int)
        conf_val = margin_confidence(p)
        g_eval = np.asarray(groups).ravel()[m] if groups is not None else None
        auc, lo, hi = metrics.auroc_ci(y, p, seed=self.seed)
        in_set = cset[m][np.arange(m.sum()), y]
        abstain = cset[m].sum(1) != 1
        return {
            "n": int(m.sum()),
            "prevalence": float(y.mean()),
            "discrimination": {"auroc": auc, "auroc_ci": [lo, hi],
                               "auprc": metrics.auprc(y, p)},
            "calibration": {"ece": metrics.ece(y, p),
                            "ece_quantile": metrics.ece(y, p, strategy="quantile"),
                            "brier": metrics.brier(y, p)},
            "conformal": {"alpha": self.alpha, "coverage": float(in_set.mean()),
                          "split_degraded": bool(getattr(self, "_conformal_split_degraded", False)),
                          "coverage_benign": float(cset[m][y == 0, 0].mean()) if (y == 0).any() else float("nan"),
                          "coverage_pathogenic": float(cset[m][y == 1, 1].mean()) if (y == 1).any() else float("nan"),
                          "abstention_rate": float(abstain.mean())},
            # Runnability report T5, second observation. These were computed on the POOLED
            # confidence ordering, which is refusal applied GLOBALLY. The paper's deliverable is
            # each group refusing its own least-confident share, and the two differ materially on
            # the paper's own panel: within-species gives capture 0.354 and lift 2.35, global gives
            # 0.383 and 2.56. The tool was printing the global numbers under the within-species
            # heading. When groups are supplied the threshold is now applied within each group,
            # which is the shipped deliverable; the global figures remain available beside them.
            "selective": {"coverage": self.coverage,
                          "selective_error": _selective_error_at(
                              y, pred, conf_val, self.coverage, g_eval),
                          "full_error": float((pred != y).mean()),
                          "capture": metrics.capture_at_coverage(
                              y, pred, _within_group_rank(conf_val, g_eval), self.coverage),
                          "lift": metrics.selective_lift(
                              y, pred, _within_group_rank(conf_val, g_eval), self.coverage),
                          "refusal": "within-group" if g_eval is not None else "global",
                          "capture_global": metrics.capture_at_coverage(
                              y, pred, conf_val, self.coverage),
                          "lift_global": metrics.selective_lift(y, pred, conf_val, self.coverage)},
        }

    def summary(self, scores, labels, groups=None) -> str:
        """A one-block human-readable rendering of :meth:`evaluate`."""
        r = self.evaluate(scores, labels, groups)
        d, c, cf, s = r["discrimination"], r["calibration"], r["conformal"], r["selective"]
        return (
            "glmtrust report  (n={n}, prevalence={prev:.3f})\n"
            "  discrimination : AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]   AUPRC {ap:.3f}\n"
            # Runnability report T6: the first figure was unlabelled and is the equal-WIDTH
            # estimator, which collapses on a coarse posterior. It printed 0.0041 where the paper
            # reports ECE around 0.05: twelvefold apart, smaller first, so a reader quoting the tool
            # would conclude the layer is a hundredth as miscalibrated as the paper says. The paper's
            # own rule is to name the estimator beside each figure rather than nominate one as
            # canonical, so both are named and the equal-mass one leads.
            "  calibration    : ECE equal-mass {eceq:.4f}, equal-width {ece:.4f}   Brier {br:.4f}\n"
            "  conformal      : {conf} @ alpha={al:.2f}  coverage {cov:.3f} "
            "(negative {cb:.3f}, positive {cp:.3f})  abstain {ab:.3f}\n"
            "  selective      : keep {kcov:.0%}  selective-error {se:.4f} (vs full {fe:.4f})  "
            "capture {cap:.3f}  lift {lift:.2f}  ({rule} refusal)"
        ).format(
            n=r["n"], prev=r["prevalence"], auc=d["auroc"], lo=d["auroc_ci"][0], hi=d["auroc_ci"][1],
            ap=d["auprc"], ece=c["ece"], eceq=c["ece_quantile"], br=c["brier"], conf=self.conformal,
            al=cf["alpha"], cov=cf["coverage"], cb=cf["coverage_benign"], cp=cf["coverage_pathogenic"],
            ab=cf["abstention_rate"], kcov=s["coverage"], se=s["selective_error"], fe=s["full_error"],
            cap=s["capture"], lift=s["lift"], rule=s["refusal"],
        ) + self._report_warnings(d, s)

    @staticmethod
    def _report_warnings(d, s):
        """Runnability report T4.

        A constant score printed "capture 1.000, lift 6.67" at exit 0 with no warning: the best
        selective-prediction result the tool can produce, off a scorer carrying no information.
        Selective error ROSE at the same time (0.1000 -> 0.1176) while capture read 1.000, so the
        two numbers on one line pointed in opposite directions and nothing said so. The `audit`
        subcommand already prints a warnings block gated on effect size; `evaluate` printed none.
        """
        w = []
        lo, hi = d["auroc_ci"]
        if lo <= 0.5 <= hi:
            w.append("the AUROC interval [%.3f, %.3f] contains 0.5, so the score is not resolvably "
                     "better than chance on this panel and the selective line below should not be "
                     "read as a gain." % (lo, hi))
        if s["selective_error"] > s["full_error"]:
            w.append("selective error at %.0f%% coverage (%.4f) is HIGHER than answering everything "
                     "(%.4f): refusing is costing accuracy here, whatever the capture and lift read."
                     % (100 * s["coverage"], s["selective_error"], s["full_error"]))
        if not w:
            return ""
        return "\n  WARNINGS\n" + "".join("    - %s\n" % x for x in w).rstrip("\n")

    # ---- internals --------------------------------------------------------
    def _oof_probabilities(self, scores, labels, groups):
        if groups is not None:
            probs, _ = leave_one_group_out(scores, labels, groups, self.calibration)
            return probs
        return cross_conformal_calibrate(scores, labels, self.calibration, self.n_splits, self.seed)

    def _splits(self, scores, labels, groups):
        if groups is not None:
            groups = np.asarray(groups).ravel()
            # Runnability report T2: with one level, ~te is empty, the empty training fold reached
            # sklearn, and it raised "Found array with 0 sample(s)" — a message naming neither the
            # group column nor the cause. That is the paper's own first-use case: a user working in
            # ONE label-poor species, running the documented command with their single species in
            # --group-col. Refuse it here, in the caller's vocabulary.
            n_groups = len(dict.fromkeys(groups.tolist()))
            if n_groups < 2:
                raise ValueError(
                    "leave-one-group-out needs at least two groups; the group column has %d "
                    "(%r). Drop the group argument to use %d-fold cross-validation instead."
                    % (n_groups, list(dict.fromkeys(groups.tolist()))[:3], self.n_splits))
            for g in dict.fromkeys(groups.tolist()):
                te = groups == g
                yield np.where(~te)[0], np.where(te)[0]
        else:
            skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
            yield from skf.split(scores.reshape(-1, 1), labels)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")


def _within_group_rank(confidence, groups):
    """Confidence re-expressed so that a single global cut refuses the same SHARE of every group.

    Runnability report T5. The paper's deliverable refuses each species' own least-confident share;
    ranking within group and normalising to [0, 1] makes one global threshold do exactly that, so
    the existing pooled helpers implement the within-group rule unchanged. With no groups this is
    the identity on the ordering.
    """
    if groups is None:
        return confidence
    out = np.empty(len(confidence), float)
    for g in dict.fromkeys(np.asarray(groups).ravel().tolist()):
        m = np.asarray(groups).ravel() == g
        k = int(m.sum())
        r = np.argsort(np.argsort(confidence[m], kind="stable"), kind="stable")
        out[m] = (r + 0.5) / k if k else 0.0
    return out


def _selective_error_at(y, pred, confidence, coverage, groups=None):
    confidence = _within_group_rank(confidence, groups)
    n = len(y)
    k_keep = int(round(coverage * n))
    if k_keep <= 0:
        return float("nan")
    keep_idx = np.argsort(-confidence, kind="stable")[:k_keep]
    return float((pred[keep_idx] != y[keep_idx]).mean())
