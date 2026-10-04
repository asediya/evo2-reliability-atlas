"""ADVERSARIAL CHECK 4 — spatial-proximity leakage. Windows are 1 kb. If a negative sits within ~1 kb of
a positive, their windows overlap and Evo2/phyloP signal could bleed across, inflating AUROC. Confirm no
negative is within the window span of a positive, and that AUROC is unchanged after removing negatives
that are 'close' (<50 kb) to any positive.

NOTE: this script was written against an exploratory evo2+phylop ensemble (out-of-fold AUROC ~0.97) that is NOT a result in the manuscript, which reports Evo 2 alone. Run it via `src/ccs/verify/run_controls.py`, which records the panel-level composition and spatial controls that bear on the paper (deposited in reports/verify_controls.json) and labels the ensemble numbers as provenance.
"""
import os
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys

# A referee on a cp437 console or with LC_ALL=C otherwise gets a traceback and a nonzero
# exit from a run that succeeded; build_tables.py even wrote its outputs first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = pl.read_parquet(f"{CCS_ROOT}/data/interim/verify_table.parquet")
y = d["label"].to_numpy().astype(int); npd = d["nearest_pos_dist"].to_numpy()
evo = d["evo2_neg"].to_numpy(); phy = d["phylop"].to_numpy()
neg = y == 0
print(f"negatives within 1kb of a positive (window overlap): {int((neg & (npd < 1000)).sum())}")
print(f"negatives within 10kb: {int((neg & (npd < 10000)).sum())}   within 50kb: {int((neg & (npd < 50000)).sum())}")
def ens_auc(mask):
    yy = y[mask]; X = np.column_stack([evo[mask], phy[mask]]); e = np.full(mask.sum(), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, yy):
        e[te] = LogisticRegression(max_iter=1000).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(yy, e)
full = np.ones(len(y), bool)
keep = ~(neg & (npd < 50000))   # drop negatives closer than 50kb to any positive
print(f"AUROC full ensemble: {ens_auc(full):.4f}")
print(f"AUROC after dropping <50kb negatives (n_dropped={int((~keep).sum())}): {ens_auc(keep):.4f}")
print("VERDICT_HINT:", "no spatial leakage" if (neg & (npd < 1000)).sum() == 0 else "PROXIMITY ARTIFACT — overlapping windows")
