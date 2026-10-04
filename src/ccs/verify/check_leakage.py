"""ADVERSARIAL CHECK 1 — leakage audit of the ensemble AUROC (claimed 0.971).
The ensemble = logistic(evo2_neg, phylop). If the logistic ever sees test labels, AUROC inflates.
We compare: (a) strict out-of-fold (nested) AUROC, (b) in-sample refit-on-all AUROC (the leaky version),
(c) repeated with different seeds. If (a) ~ 0.97 and is stable, no leakage. If (a) << (b), the headline
number was leaky.

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
# (a) strict out-of-fold, 10 seeds
oof = []
for s in range(10):
    ens = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
        ens[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof.append(roc_auc_score(y, ens))
# (b) leaky in-sample refit-on-all
lr = LogisticRegression(max_iter=1000).fit(X, y)
insample = roc_auc_score(y, lr.predict_proba(X)[:, 1])
print(f"out-of-fold AUROC: mean={np.mean(oof):.4f} sd={np.std(oof):.4f} range=[{min(oof):.4f},{max(oof):.4f}]")
print(f"in-sample (leaky) AUROC: {insample:.4f}")
print(f"gap (in-sample - oof): {insample-np.mean(oof):.4f}")
print("VERDICT_HINT:", "no leakage (oof stable ~claimed)" if np.mean(oof) > 0.95 and insample-np.mean(oof) < 0.02 else "INSPECT")
