# -*- coding: utf-8 -*-
"""Is the DeLong covariance scaled correctly? Check it against a bootstrap that shares no code.

The variance of an AUROC difference enters every closed-form interval the tool reports, and a
constant factor there is invisible: the point estimate is untouched, nothing raises, every unit test
that only checks shape or sign still passes, and the intervals come out uniformly too narrow. A
factor of 4 in the covariance is a factor of 2 in every interval width, which turns a result that
crosses zero into one that does not.

So the scaling is checked against an estimator that shares no code with it: an ordinary paired
bootstrap, resampling positives and negatives independently, standard error taken as the SD of the
draws. If DeLong's SE and the bootstrap SE agree to within Monte-Carlo error on several panels of
different size and separation, the scaling is right. If DeLong were off by 4x in the covariance, the
ratio below would sit near 0.5 on every row and could not be mistaken for noise.

The comparison is on the SE, not the interval, because the two methods differ slightly in shape
(normal-theory versus percentile) and only the scale is in question here.
"""
from __future__ import annotations

import os
import math
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "glmtrust/src")
from glmtrust.delong import delong_delta_ci, must_answer_delta_ci   # noqa: E402

Z = 1.959963984540054                                                # matches _norm_ppf(0.975)


def auroc(y, s):
    """Plain rank AUROC, written out rather than imported, so the check is independent."""
    p, n = s[y == 1], s[y == 0]
    if p.size == 0 or n.size == 0:
        return np.nan
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(allv.size, dtype=float)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j < sv.size and sv[j] == sv[i]:
            j += 1
        r[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return (r[:p.size].sum() - p.size * (p.size + 1) / 2.0) / (p.size * n.size)


def boot_se(y, a, b, n_boot, rng, stat):
    """Paired bootstrap SE of stat(a) - stat(b), resampling each class independently."""
    ip = np.flatnonzero(y == 1)
    ineg = np.flatnonzero(y == 0)
    out = np.empty(n_boot)
    for k in range(n_boot):
        s = np.concatenate([rng.choice(ip, ip.size, replace=True),
                            rng.choice(ineg, ineg.size, replace=True)])
        yy = y[s]
        out[k] = stat(yy, a[s]) - stat(yy, b[s])
    return float(out.std(ddof=1))


def ma_auroc(y, s):
    """Must-answer AUROC: unscorable variants place at exactly 1/2. Independent of delong.py."""
    fin = np.isfinite(s)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    kp, kn = int(((y == 1) & fin).sum()), int(((y == 0) & fin).sum())
    if kp == 0 or kn == 0:
        return 0.5
    cov = auroc(y[fin], s[fin])
    return (cov * kp * kn + 0.5 * (npos * nneg - kp * kn)) / (npos * nneg)


rng = np.random.default_rng(20260801)
print("  DeLong SE vs an independent paired bootstrap (2000 draws)")
print("  a 4x covariance error would put every ratio at ~0.50, not near 1.00")
print()
print("  %-30s %10s %10s %8s" % ("panel", "delong SE", "boot SE", "ratio"))

worst = 0.0
nonfinite = []
rows = 0
for npos, nneg, eff, miss in [(200, 200, 0.8, 0.0), (500, 1500, 0.5, 0.0),
                              (1000, 1000, 1.2, 0.0), (300, 900, 0.6, 0.35),
                              (800, 800, 0.9, 0.20)]:
    y = np.r_[np.ones(npos, int), np.zeros(nneg, int)]
    a = np.r_[rng.normal(eff, 1, npos), rng.normal(0, 1, nneg)]
    b = np.r_[rng.normal(eff * 0.7, 1, npos), rng.normal(0, 1, nneg)]
    if miss:                                    # class-dependent missingness, the case that matters
        m = rng.random(y.size) < (miss * np.where(y == 1, 1.6, 0.6))
        b = b.copy()
        b[m] = np.nan

    if miss:
        _, (lo, hi) = must_answer_delta_ci(y, a, b)
        d_se = (hi - lo) / (2 * Z)
        b_se = boot_se(y, a, b, 2000, rng, ma_auroc)
        tag = "must-answer, %d%% missing" % int(miss * 100)
    else:
        _, (lo, hi) = delong_delta_ci(y, a, b)
        d_se = (hi - lo) / (2 * Z)
        b_se = boot_se(y, a, b, 2000, rng, auroc)
        tag = "complete"
    ratio = d_se / b_se
    rows += 1
    # NON-FINITE MUST FAIL, NOT DISAPPEAR. `max(worst, abs(nan - 1))` returns `worst` unchanged, so
    # a panel whose closed-form SE is undefined would score as agreement: with every ratio NaN this
    # gate would print "worst deviation 0.0%" and "CLEAN ... agree on all 5 panels". A quantity that
    # cannot be computed is not a quantity that agrees.
    if not (math.isfinite(d_se) and math.isfinite(b_se) and math.isfinite(ratio)):
        nonfinite.append("%d/%d %s: d_se=%s b_se=%s ratio=%s"
                         % (npos, nneg, tag, d_se, b_se, ratio))
    else:
        worst = max(worst, abs(ratio - 1.0))
    print("  %-30s %10.5f %10.5f %8.3f" % ("%d/%d %s" % (npos, nneg, tag), d_se, b_se, ratio))

print()
print("  panels compared     : %d" % rows)
print("  worst deviation     : %.1f%%" % (100 * worst))
if nonfinite:
    print("  NON-FINITE          : %d panel(s) produced an undefined SE or ratio" % len(nonfinite))
    for _n in nonfinite:
        print("      %s" % _n)
    print("  RESULT: FAIL -- an undefined comparison cannot count as agreement")
    sys.exit(1)
if rows < 5:
    print("  RESULT: INCONCLUSIVE -- fewer panels ran than intended")
    sys.exit(2)
# 2000 bootstrap draws give the bootstrap SE its own ~1.6% relative Monte-Carlo error, so agreement
# is judged at 12%: loose enough not to flag noise, far tighter than the 50% a scaling bug produces.
if worst > 0.12:
    print("  RESULT: FAIL -- DeLong and the bootstrap disagree by more than Monte-Carlo error")
    sys.exit(1)
print("  RESULT: CLEAN -- closed-form and bootstrap standard errors agree on all %d panels" % rows)
