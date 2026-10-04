# -*- coding: utf-8 -*-
"""A2 step 1 — build the smallest panel that can answer the reverse-complement question.

The objection: a genome has no preferred strand, so a score that changes when you flip the window
is measuring the model's strand bias rather than the variant. The paper's 1,001-bp comparison is a
near-tie with a margin of **+0.0024**, so if strand-flipping moves scores by more than that, the
near-tie is noise.

Scoring the whole co-scorable panel twice at 40B is hours. It is also more than the question needs.
Two quantities answer it, and both are estimable from a few hundred variants:

  1. the per-variant difference between the forward score and the strand-averaged score, against
     the spread of the scores themselves — if the shift is small relative to the spread, AUROC
     cannot move far
  2. the AUROC computed both ways on the same subset, which bounds the movement directly

This selects a stratified subset: equal numbers per species, balanced on label, so no species or
class dominates the estimate.

    python analyses/scripts/strand_prepare.py --per-species 40
"""
import argparse
import glob
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
RNG = np.random.default_rng(20260804)
OUT = "analyses/data/strand"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=40,
                    help="variants per species, split evenly between classes")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    frames = []
    for f in sorted(glob.glob("data/interim/*_scoring_windows.parquet")):
        sp = os.path.basename(f).replace("_scoring_windows.parquet", "")
        d = pl.read_parquet(f)
        if "label" not in d.columns:
            continue
        keep = []
        for lab in (0, 1):
            s = d.filter(pl.col("label") == lab)
            if len(s) == 0:
                continue
            n = min(a.per_species // 2, len(s))
            idx = RNG.choice(len(s), size=n, replace=False)
            keep.append(s[idx.tolist()])
        if keep:
            frames.append(pl.concat(keep).with_columns(pl.lit(sp).alias("species")))
            print("  %-28s %5d rows available, %d selected" % (sp, len(d), sum(len(k) for k in keep)))

    panel = pl.concat(frames)
    # The scorer reads variant_id / ref_seq / alt_seq / var_off; species and label ride along for
    # the analysis and are harmless to it.
    p = os.path.join(OUT, "strand_subset_windows.parquet")
    panel.write_parquet(p)
    print()
    print("  wrote %s: %s variants across %d species, %d positive, window %d bp"
          % (p, "{:,}".format(len(panel)), panel["species"].n_unique(),
             int(panel["label"].sum()), len(panel["ref_seq"][0])))
    print()
    print("  next, inside the container:")
    print("    python src/ccs/score_evo2_meanll.py --panel %s \\" % p)
    print("      --model-size 40b --batch 4 --chunk 64 --out %s/fwd.parquet" % OUT)
    print("    python src/ccs/score_evo2_meanll.py --panel %s \\" % p)
    print("      --model-size 40b --batch 4 --chunk 64 --strand --out %s/avg.parquet" % OUT)


if __name__ == "__main__":
    main()
