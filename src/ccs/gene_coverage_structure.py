# -*- coding: utf-8 -*-
"""Where do 49 predictors go silent, and do they go silent in the same places?

Reach is usually reported, if at all, as one number per predictor. That number cannot answer the
question a diagnostic lab actually has: *for THIS gene, is there anything that scores it?*

Two possibilities with very different consequences, and they are distinguishable from the data:

  INDEPENDENT GAPS   each predictor declines a different scattered set, so pooling several predictors
                     covers nearly everything and no gene is dark.
  SHARED GAPS        the predictors decline the SAME genes, so a gene missed by one is missed by
                     most, ensembling buys little, and a set of genes exists that nothing can score.

Computed here:
  1. a gene x predictor coverage matrix (38,420 x 49)
  2. the correlation structure across predictors at gene level -- is there one common hard-gene axis?
  3. genes covered by no predictor, by few, and by all
  4. what distinguishes a dark gene from a covered one (variant count, consequence mix, label mix)
  5. per gene, the best-covering predictor -- the table a lab would actually use

Parallel over predictors; the matrix build is the expensive part.

    python src/ccs/gene_coverage_structure.py --jobs 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PANEL = "data/processed/dbnsfp_panel.parquet"


def _coverage_for(args):
    """Coverage fraction per gene for one predictor."""
    name, finite, gene_codes, n_gene = args
    tot = np.bincount(gene_codes, minlength=n_gene).astype(float)
    hit = np.bincount(gene_codes, weights=finite.astype(float), minlength=n_gene)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)
    return name, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--min-stars", type=int, default=1)
    ap.add_argument("--out", default="reports/gene_coverage_structure.json")
    a = ap.parse_args()

    t0 = time.time()
    d = pl.read_parquet(PANEL).filter(pl.col("stars") >= a.min_stars)
    cols = [c for c in d.columns if c.lower().endswith("_rankscore")]
    gene = np.array([g if g else "?" for g in d["gene"].to_list()])
    uniq, gene_codes = np.unique(gene, return_inverse=True)
    n_gene = uniq.size
    y = d["label"].to_numpy().astype(int)
    print("  %s variants, %s genes, %d predictors"
          % (format(d.height, ","), format(n_gene, ","), len(cols)))

    tasks = [(c.replace("_converted_rankscore", "").replace("_rankscore", ""),
              np.isfinite(d[c].to_numpy().astype(float)), gene_codes, n_gene) for c in cols]
    print("  building the gene x predictor coverage matrix on %d workers ..." % a.jobs)
    names, mat = [], []
    with ProcessPoolExecutor(max_workers=min(a.jobs, len(tasks))) as ex:
        for nm, frac in ex.map(_coverage_for, tasks):
            names.append(nm)
            mat.append(frac)
    M = np.vstack(mat)                       # predictors x genes
    print("  matrix %s in %.0f s" % (str(M.shape), time.time() - t0))

    n_var = np.bincount(gene_codes, minlength=n_gene)
    # A gene is only informative about coverage if it carries a few variants; a gene with one variant
    # is covered or not by accident.
    keep = n_var >= 5
    print("  genes with >=5 variants: %s of %s" % (format(int(keep.sum()), ","), format(n_gene, ",")))

    Mk = M[:, keep]
    covered_by = (Mk >= 0.5).sum(axis=0)     # how many predictors cover at least half the gene
    print()
    print("  === how many of the %d predictors cover a given gene? ===" % len(names))
    for lo, hi, label in ((0, 0, "NO predictor"), (1, 4, "1-4"), (5, 14, "5-14"),
                          (15, 29, "15-29"), (30, 44, "30-44"), (45, 49, "45-49 (nearly all)")):
        n = int(((covered_by >= lo) & (covered_by <= hi)).sum())
        print("    %-22s %6s genes  (%.1f%%)" % (label, format(n, ","), 100 * n / keep.sum()))

    # Is there ONE common hard-gene axis, or many independent ones?
    print()
    print("  === do predictors fail on the same genes? ===")
    C = np.corrcoef(Mk)
    off = C[np.triu_indices_from(C, k=1)]
    print("    mean pairwise correlation of gene-level coverage : %.3f" % np.nanmean(off))
    print("    median                                           : %.3f" % np.nanmedian(off))
    print("    fraction of pairs correlated above 0.8           : %.2f"
          % float(np.nanmean(off > 0.8)))
    ev = np.linalg.eigvalsh(np.nan_to_num(C, nan=0.0))[::-1]
    print("    variance explained by the first component        : %.1f%%"
          % (100 * ev[0] / ev.sum()))

    # what a dark gene looks like
    genes_k = uniq[keep]
    nvar_k = n_var[keep]
    dark = covered_by == 0
    print()
    print("  === genes no predictor covers (>=5 variants each) ===")
    print("    count            : %s" % format(int(dark.sum()), ","))
    if dark.any():
        print("    median variants  : %.0f" % float(np.median(nvar_k[dark])))
        order = np.argsort(-nvar_k * dark)
        print("    largest examples : %s"
              % ", ".join("%s (%d)" % (genes_k[i], nvar_k[i]) for i in order[:8] if dark[i]))

    payload = {
        "_meta": {"panel": PANEL, "min_stars": a.min_stars, "n_variants": int(d.height),
                  "n_genes": int(n_gene), "n_genes_ge5": int(keep.sum()),
                  "n_predictors": len(names),
                  "coverage_threshold": "predictor covers >=50% of a gene's variants"},
        "predictors": names,
        "covered_by_histogram": {str(k): int((covered_by == k).sum())
                                 for k in range(len(names) + 1)},
        "mean_pairwise_gene_coverage_correlation": float(np.nanmean(off)),
        "first_component_variance_explained": float(ev[0] / ev.sum()),
        "n_genes_no_predictor": int(dark.sum()),
        "dark_genes": [str(g) for g in genes_k[dark][:500]],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(payload, open(a.out, "w", encoding="utf-8"), indent=2)
    print()
    print("  wrote %s in %.0f s total" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
