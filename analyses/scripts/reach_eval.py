# -*- coding: utf-8 -*-
"""The reach-adjusted evaluation mode, as a reusable object.

This is the deliverable the field can adopt: given scores that may be missing and class labels, it
returns everything needed to compare two variant-effect predictors honestly under class-dependent
coverage — and refuses to return a single number, because a single number is what causes the
problem.

    report(scores, labels) -> {
        coverage:   n, k and r for each class, rho = r+ r-, and J = r+ - r-
        covered:    AUROC on the variants the method can score, the quantity the field publishes
        interval:   [rho*A_cov, rho*A_cov + (1-rho)] — every panel-wide AUROC consistent with the
                    data, both endpoints attained
        must_answer: the interval midpoint; equivalently CAFA's full-evaluation convention applied
                    to a ranking metric, and the minimax-regret point estimate under absolute loss
        class_gap:  r- minus r+, with a Newcombe hybrid-score interval
        youden:     the AUROC of the coverage indicator alone, which is exactly 0.5 + J/2 — the
                    coverage rule read as a classifier of the label it was supposed to be blind to
    }

    compare(a, b) -> whether "a beats b" is identified, unassumed and under monotone coverage

Written here rather than in `glmtrust/` so the frozen deposit stays untouched during revision. It
carries a self-test that reproduces the deposited 49-predictor audit.

    python analyses/scripts/reach_eval.py        # runs the self-test
"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------- building blocks
def _auroc(score, label):
    s = np.asarray(score, dtype=float)
    y = np.asarray(label).astype(int)
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    r = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j < len(s) - 1 and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _newcombe(k1, n1, k2, n2):
    """Difference of two independent proportions, hybrid-score interval."""
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    d = k1 / n1 - k2 / n2
    return d, (d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2),
               d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2))


# --------------------------------------------------------------------------------- the report
def report(scores, labels):
    """scores may contain NaN for variants the method cannot score. labels are 0/1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(int)
    if len(s) != len(y):
        raise ValueError("scores and labels differ in length")
    cov = np.isfinite(s)

    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    kpos, kneg = int(cov[y == 1].sum()), int(cov[y == 0].sum())
    if npos == 0 or nneg == 0:
        raise ValueError("both classes must be present")
    rpos, rneg = kpos / npos, kneg / nneg
    rho = rpos * rneg

    a_cov = _auroc(s[cov], y[cov]) if (kpos and kneg) else None
    if a_cov is None:
        return {"coverage": {"n_pos": npos, "n_neg": nneg, "k_pos": kpos, "k_neg": kneg,
                             "r_pos": rpos, "r_neg": rneg, "rho": rho, "J": rpos - rneg},
                "covered": None, "interval": None, "must_answer": None,
                "_note": "one class is entirely unscored; nothing is identified"}

    lo = rho * a_cov
    hi = lo + (1 - rho)
    gap, gap_ci = _newcombe(kneg, nneg, kpos, npos)

    return {
        "coverage": {"n_pos": npos, "n_neg": nneg, "k_pos": kpos, "k_neg": kneg,
                     "r_pos": rpos, "r_neg": rneg, "rho": rho, "J": rpos - rneg},
        "covered": a_cov,
        "interval": [lo, hi],
        "interval_width": hi - lo,
        "interval_monotone_coverage": [lo, a_cov],
        "must_answer": (lo + hi) / 2,
        "must_answer_shrinkage_form": 0.5 + rho * (a_cov - 0.5),
        "class_gap_neg_minus_pos": gap,
        "class_gap_ci95_newcombe": list(gap_ci),
        "class_gap_material": bool(gap_ci[0] > 0 or gap_ci[1] < 0),
        "coverage_indicator_auroc": 0.5 + (rpos - rneg) / 2,
        "_reading": "The interval is every panel-wide AUROC consistent with the data. Its width "
                    "is 1 - rho and does not depend on how accurate the method is on what it "
                    "covers. must_answer is the midpoint.",
    }


def compare(a, b, monotone=False):
    """Is 'a beats b' identified? a and b are report() outputs."""
    if a["interval"] is None or b["interval"] is None:
        return {"identified": False, "_why": "a method leaves one class entirely unscored"}
    ia = a["interval_monotone_coverage"] if monotone else a["interval"]
    ib = b["interval_monotone_coverage"] if monotone else b["interval"]
    if ia[0] > ib[1]:
        v = "a beats b, identified"
    elif ib[0] > ia[1]:
        v = "b beats a, identified"
    else:
        v = "NOT IDENTIFIED"
    return {"identified": v != "NOT IDENTIFIED", "verdict": v,
            "interval_a": list(ia), "interval_b": list(ib),
            "assumption": "monotone coverage" if monotone else "none",
            "point_difference_covered": a["covered"] - b["covered"],
            "point_difference_must_answer": a["must_answer"] - b["must_answer"]}


# ------------------------------------------------------------------------------- the self-test
def selftest():
    """Reproduce the deposited 49-predictor audit from its own reach and covered-AUROC values.

    The audit stores summaries rather than per-variant scores, so the check is of the algebra:
    given (A_cov, r+, r-), does report()'s must_answer match the deposited auroc_must_answer, and
    does the coverage-indicator AUROC match the deposited miss_auroc?
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(root, "reports", "dbnsfp_reach_audit.json")
    P = json.load(io.open(p, encoding="utf-8"))["predictors"]

    e_ma, e_mi, n = 0.0, 0.0, 0
    for q in P:
        rho = q["reach_pos"] * q["reach_neg"]
        lo = rho * q["auroc_covered"]
        ma = lo + (1 - rho) / 2
        e_ma = max(e_ma, abs(ma - q["auroc_must_answer"]))
        if q.get("miss_auroc") is not None and np.isfinite(q["miss_auroc"]):
            e_mi = max(e_mi, abs((0.5 + (q["reach_pos"] - q["reach_neg"]) / 2)
                                 - q["miss_auroc"]))
        n += 1
    print("  self-test against the deposited %d-predictor audit" % n)
    print("    max |must_answer - deposited|            %.3e   %s"
          % (e_ma, "OK" if e_ma < 1e-12 else "FAIL"))
    print("    max |coverage-indicator - deposited|     %.3e   %s"
          % (e_mi, "OK" if e_mi < 1e-12 else "FAIL"))

    # a synthetic end-to-end check of report() itself
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500, int), np.zeros(500, int)]
    s = rng.normal(y * 1.0, 1.0)
    s[rng.random(len(s)) < np.where(y == 1, 0.05, 0.40)] = np.nan   # benign lost more often
    r = report(s, y)
    assert r["interval"][0] <= r["must_answer"] <= r["interval"][1]
    assert abs(r["must_answer"] - r["must_answer_shrinkage_form"]) < 1e-12
    assert abs(r["interval_width"] - (1 - r["coverage"]["rho"])) < 1e-12
    print("    synthetic panel: rho %.3f, covered %.3f, interval [%.3f, %.3f], must-answer %.3f"
          % (r["coverage"]["rho"], r["covered"], r["interval"][0], r["interval"][1],
             r["must_answer"]))
    print("    midpoint identity, shrinkage identity and width identity all hold")

    full = report(np.where(np.isnan(s), rng.normal(0, 1, len(s)), s), y)
    c = compare(full, r)
    print("    compare(complete-coverage, partial-coverage): %s" % c["verdict"])
    print()
    print("  the object this returns is a triple, not a number:")
    print("    interval %s, midpoint %.4f, coverage (r+ %.3f, r- %.3f)"
          % ([round(x, 4) for x in r["interval"]], r["must_answer"],
             r["coverage"]["r_pos"], r["coverage"]["r_neg"]))


if __name__ == "__main__":
    selftest()
