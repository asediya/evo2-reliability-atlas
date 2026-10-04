"""ADVERSARIAL CHECK 3 — is the ensemble's gain over the best SINGLE scorer statistically real?
Claim: ensemble 0.971 > phyloP 0.942 > Evo2 0.912. Paired bootstrap the AUROC deltas (same resampled
indices for both scorers) and report 95% CIs. If CI(ensemble - phylop) excludes 0, the FM adds signal
beyond conservation. Also report Evo2-vs-phyloP correlation (orthogonality of the added axis).

NOTE: this script was written against an exploratory evo2+phylop ensemble (out-of-fold AUROC ~0.97) that is NOT a result in the manuscript, which reports Evo 2 alone. Run it via `src/ccs/verify/run_controls.py`, which records the panel-level composition and spatial controls that bear on the paper (deposited in reports/verify_controls.json) and labels the ensemble numbers as provenance.
"""
import os
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
d = pl.read_parquet(f"{CCS_ROOT}/data/interim/verify_table.parquet")
y = d["label"].to_numpy().astype(int); evo = d["evo2_neg"].to_numpy(); phy = d["phylop"].to_numpy()
X = np.column_stack([evo, phy]); ens = np.full(len(y), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    ens[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
rng = np.random.default_rng(0); n = len(y)
de, dp = [], []   # ens-evo, ens-phy
base = {}
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if len(np.unique(y[idx])) < 2: continue
    ae = roc_auc_score(y[idx], evo[idx]); ap = roc_auc_score(y[idx], phy[idx]); an = roc_auc_score(y[idx], ens[idx])
    de.append(an - ae); dp.append(an - ap)
def ci(a): a = np.array(a); return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)
me, le, he = ci(de); mp, lp, hp = ci(dp)
r = spearmanr(evo, phy).statistic
print(f"AUROC  evo2={roc_auc_score(y,evo):.4f}  phylop={roc_auc_score(y,phy):.4f}  ensemble={roc_auc_score(y,ens):.4f}")
print(f"delta ensemble-evo2 : {me:+.4f} [{le:+.4f},{he:+.4f}]  {'SIG' if le>0 else 'ns'}")
print(f"delta ensemble-phylop: {mp:+.4f} [{lp:+.4f},{hp:+.4f}]  {'SIG' if lp>0 else 'ns'}")
print(f"Spearman(evo2,phylop) = {r:.3f}  (low => orthogonal added signal)")
print("VERDICT_HINT:", "ensemble gain over conservation is significant" if lp > 0 else "gain NOT significant")
