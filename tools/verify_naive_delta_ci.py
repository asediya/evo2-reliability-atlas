# -*- coding: utf-8 -*-
"""Does the naive (covered-subset) delta carry an interval, and does it exclude zero at each star floor?

The record shows a naive column with no interval anywhere, and immediately after the table asserts
"Every interval excludes zero". That is true of the four MATCHED intervals, which are the only ones
shown -- but the sentence follows a claim about the naive column, where at the 2-star floor the delta
is +0.0001 and cannot plausibly be distinguishable from zero.

Computing the missing intervals so the text can be corrected against numbers rather than reasoning.
Gene-clustered, because that is what this project learned to use.
"""
import gzip
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
# this hard-coded the author's drive, so the script could not run
# from a clean extraction of the deposit. Resolve the repository root from this file instead.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

panel = pl.read_parquet("data/processed/clinvar_panel.parquet")
for s in ("alphamissense", "revel"):
    panel = panel.join(pl.read_parquet("data/processed/human_scorers/%s.parquet" % s),
                       on="variant_id", how="left")
panel = panel.join(pl.read_parquet("data/interim/clinvar_gene.parquet"), on="variant_id", how="left")

print("  naive delta = AUROC(AlphaMissense on its own covered set) - AUROC(REVEL on its own)")
print("  intervals: gene-clustered bootstrap, 400 draws")
print()
print("  %-7s %11s %11s %22s %s" % ("stars", "naive d", "matched d", "naive 95% CI", "naive excludes 0?"))
for stars in (0, 1, 2, 3):
    p = panel.filter(pl.col("stars") >= stars)
    y = p["label"].to_numpy().astype(int)
    am = p["alphamissense"].to_numpy().astype(float)
    rv = p["revel"].to_numpy().astype(float)
    gene = np.array([g if g else "?" for g in p["gene"].to_list()])
    fa, fb = np.isfinite(am), np.isfinite(rv)
    if fa.sum() < 10 or fb.sum() < 10:
        continue
    naive = roc_auc_score(y[fa], am[fa]) - roc_auc_score(y[fb], rv[fb])
    both = fa & fb
    matched = roc_auc_score(y[both], am[both]) - roc_auc_score(y[both], rv[both])

    genes = np.array(sorted(set(gene)))
    idx = {g: np.flatnonzero(gene == g) for g in genes}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(400):
        pick = rng.choice(genes, genes.size, replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        yy, aa, bb = y[ii], am[ii], rv[ii]
        ga, gb = np.isfinite(aa), np.isfinite(bb)
        if len(set(yy[ga])) < 2 or len(set(yy[gb])) < 2:
            continue
        draws.append(roc_auc_score(yy[ga], aa[ga]) - roc_auc_score(yy[gb], bb[gb]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    excl = not (lo <= 0 <= hi)
    print("  >=%-6d %+11.4f %+11.4f   [%+.4f, %+.4f] %s"
          % (stars, naive, matched, lo, hi, "YES" if excl else "no -- SPANS ZERO"))
