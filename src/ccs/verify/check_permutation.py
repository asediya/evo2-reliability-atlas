"""ADVERSARIAL CHECK 5 — permutation null control. Shuffle the labels and re-run the ENTIRE ensemble
pipeline (out-of-fold logistic) to build the null AUROC distribution. A correct pipeline yields a null
centered at 0.5; the real 0.971 must sit far outside it. This catches any structural inflation in the
cross-fold/ensembling machinery itself (independent of the biology).

NOTE: this script was written against an exploratory evo2+phylop ensemble (out-of-fold AUROC ~0.97) that is NOT a result in the manuscript, which reports Evo 2 alone. Run it via `src/ccs/verify/run_controls.py`, which records the panel-level composition and spatial controls that bear on the paper (deposited in reports/verify_controls.json) and labels the ensemble numbers as provenance.
"""
import os
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = pl.read_parquet(f"{CCS_ROOT}/data/interim/verify_table.parquet")
y = d["label"].to_numpy().astype(int); X = np.column_stack([d["evo2_neg"], d["phylop"]])
def oof_auc(yy, seed=0):
    e = np.full(len(yy), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
        e[te] = LogisticRegression(max_iter=1000).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(yy, e)
real = oof_auc(y)
rng = np.random.default_rng(0); null = []
for _ in range(300):
    yp = rng.permutation(y)
    null.append(oof_auc(yp, seed=int(rng.integers(0, 1e6))))
null = np.array(null)
p = (np.sum(null >= real) + 1) / (len(null) + 1)
print(f"real ensemble out-of-fold AUROC: {real:.4f}")
print(f"permutation null: mean={null.mean():.4f} sd={null.std():.4f} max={null.max():.4f} (n=300)")
print(f"permutation p-value: {p:.4g}")
print("VERDICT_HINT:", "null centered ~0.5, real far outside" if abs(null.mean()-0.5) < 0.03 and real > null.max() else "INSPECT pipeline")
