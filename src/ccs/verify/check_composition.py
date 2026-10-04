"""ADVERSARIAL CHECK 2 — is the composition matching real, or is the 0.97 a base-composition artifact?
If trinuc+GC matching worked, a classifier using ONLY composition (trinuc one-hot + local GC) should be
~0.5 on pos-vs-neg. If composition-only AUROC is high, the panel is NOT matched and Evo2/phyloP could be
exploiting composition. Also formally compare pos vs neg trinuc spectra (chi-square) and GC (Mann-Whitney).

NOTE: this script was written against an exploratory evo2+phylop ensemble (out-of-fold AUROC ~0.97) that is NOT a result in the manuscript, which reports Evo 2 alone. Run it via `src/ccs/verify/run_controls.py`, which records the panel-level composition and spatial controls that bear on the paper (deposited in reports/verify_controls.json) and labels the ensemble numbers as provenance.
"""
import os
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency, mannwhitneyu
import sys

# A referee on a cp437 console or with LC_ALL=C otherwise gets a traceback and a nonzero
# exit from a run that succeeded; build_tables.py even wrote its outputs first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = pl.read_parquet(f"{CCS_ROOT}/data/interim/verify_table.parquet")
y = d["label"].to_numpy().astype(int)
tri = d["tri"].to_list(); gc = d["gc"].to_numpy()
cats = sorted(set(tri)); idx = {c: i for i, c in enumerate(cats)}
T = np.zeros((len(y), len(cats)))
for i, t in enumerate(tri): T[i, idx[t]] = 1
Xc = np.column_stack([T, gc])
# composition-only out-of-fold AUROC (should be ~0.5 if matched)
ens = np.full(len(y), np.nan)
for trn, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, y):
    ens[te] = LogisticRegression(max_iter=2000).fit(Xc[trn], y[trn]).predict_proba(Xc[te])[:, 1]
comp_auc = roc_auc_score(y, ens)
# trinuc spectrum chi-square (pos vs neg)
pos_t = [tri[i] for i in range(len(y)) if y[i] == 1]; neg_t = [tri[i] for i in range(len(y)) if y[i] == 0]
tab = np.array([[pos_t.count(c) for c in cats], [neg_t.count(c) for c in cats]])
tab = tab[:, tab.sum(0) > 0]
chi2, pchi, _, _ = chi2_contingency(tab)
# GC pos vs neg
u, pgc = mannwhitneyu(gc[y == 1], gc[y == 0])
print(f"composition-ONLY (trinuc+GC) out-of-fold AUROC: {comp_auc:.4f}  (want ~0.50)")
print(f"trinuc spectrum chi2 p={pchi:.3f} (want >0.05 = matched)   GC MWU p={pgc:.3f}")
print(f"GC mean pos={gc[y==1].mean():.3f} neg={gc[y==0].mean():.3f}")
# TWO-SIDED verdict: an AUROC of 0.25 is as far from the matched
# value of 0.5 as 0.75 is, so a one-sided `< 0.60` test cannot detect an inverse composition signal.
# NB: this script runs on the small verify_table.parquet subset where the trinuc one-hot overfits;
# the panel-wide control is src/ccs/verify/atlas_controls.py (per-species, on the full 11,130 atlas).
print("VERDICT_HINT:", "matched (composition uninformative)" if abs(comp_auc - 0.5) < 0.10 else "COMPOSITION SIGNAL — |AUROC-0.5| >= 0.10 (matched value is 0.5)")
