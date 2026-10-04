# -*- coding: utf-8 -*-
"""How much did taking MAX over REVEL's transcripts flatter REVEL?

The two scorers were aggregated differently and the asymmetry was invisible in the code:

  AlphaMissense_hg38.tsv.gz   one row per variant, the CANONICAL transcript only
  revel_with_transcript_ids   one row per (variant, transcript)

so `group_by(variant_id).max()` is a no-op for AlphaMissense and a best-of-N pick for REVEL. 5.1% of
REVEL variants carry more than one transcript, with a mean max-minus-min spread of 0.1441, so this is
not a rounding-level choice.

Recompute the missense comparison under max, mean and min over REVEL's transcripts. The spread
across those three is the size of a decision nobody declared -- which is this project's own thesis,
turned on its own analysis.
"""
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
panel = panel.join(pl.read_parquet("data/processed/human_scorers/alphamissense.parquet"),
                   on="variant_id", how="left")

print("  re-reading REVEL under three aggregations ...")
rv = pl.read_csv("data/external/human_panel/revel_with_transcript_ids", separator=",",
                 columns=["chr", "grch38_pos", "ref", "alt", "REVEL"],
                 schema_overrides={"chr": pl.Utf8, "grch38_pos": pl.Utf8, "ref": pl.Utf8,
                                   "alt": pl.Utf8, "REVEL": pl.Float64})
rv = rv.filter(pl.col("grch38_pos") != ".")
rv = rv.with_columns((pl.col("chr") + "-" + pl.col("grch38_pos") + "-"
                      + pl.col("ref") + "-" + pl.col("alt")).alias("variant_id"))
agg = rv.group_by("variant_id").agg(pl.col("REVEL").max().alias("revel_max"),
                                    pl.col("REVEL").mean().alias("revel_mean"),
                                    pl.col("REVEL").min().alias("revel_min"))
panel = panel.join(agg, on="variant_id", how="left")

p = panel.filter((pl.col("stars") >= 1) & (pl.col("consequence") == "missense_variant"))
y = p["label"].to_numpy().astype(int)
am = p["alphamissense"].to_numpy().astype(float)
print("  missense panel %s variants\n" % format(len(y), ","))


def must_answer(yy, ss):
    fin = np.isfinite(ss)
    pos, neg = yy == 1, yy == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    kp, kn = int(fin[pos].sum()), int(fin[neg].sum())
    if kp < 1 or kn < 1:
        return np.nan
    cov = roc_auc_score(yy[fin], ss[fin])
    return (cov * kp * kn + 0.5 * (npos * nneg - kp * kn)) / (npos * nneg)


print("  %-10s %10s %12s %14s" % ("REVEL agg", "covered", "matched d", "must-answer d"))
res = {}
for how in ("max", "mean", "min"):
    rvv = p["revel_%s" % how].to_numpy().astype(float)
    both = np.isfinite(am) & np.isfinite(rvv)
    d_mt = roc_auc_score(y[both], am[both]) - roc_auc_score(y[both], rvv[both])
    d_ma = must_answer(y, am) - must_answer(y, rvv)
    cov = roc_auc_score(y[np.isfinite(rvv)], rvv[np.isfinite(rvv)])
    res[how] = (d_mt, d_ma)
    print("  %-10s %10.4f %12.4f %14.4f" % (how, cov, d_mt, d_ma))

print()
mt = [res[h][0] for h in res]
ma = [res[h][1] for h in res]
print("  spread across aggregations:")
print("    matched delta      %+.4f to %+.4f   (range %.4f)" % (min(mt), max(mt), max(mt) - min(mt)))
print("    must-answer delta  %+.4f to %+.4f   (range %.4f)" % (min(ma), max(ma), max(ma) - min(ma)))
print()
print("  For reference, the retracted headline quoted matched -0.0096 and must-answer -0.0357,")
print("  both computed under 'max'. An undeclared aggregation choice moves them by the amounts")
print("  above -- the same class of defect this project exists to document.")
