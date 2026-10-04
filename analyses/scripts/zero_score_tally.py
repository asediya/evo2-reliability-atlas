# -*- coding: utf-8 -*-
"""How many deposited scores are exactly zero, and why.

A single-position variant delta is `logP(alt) - logP(ref)` at one offset. Exactly 0.0 in float means
the model expressed no preference at all, and there are only two ways to get it:

  the two windows are identical -- the assembly base at that position IS the alternate allele, so
  substituting alt reproduces the reference window. This is a property of the record, flagged
  `ref_ok = False` where the panel carries that column.

  a numerical tie -- the two logits are equal after reduced-precision inference. The model really
  did read the two alleles as equally likely at that site.

Worth counting for two reasons. A whole panel of zeros is the signature of a var_off fault, which
cost seven hours of 40B time once already (`tools/check_var_off.py` now guards it). And a score of
exactly zero sits at the centre of the score distribution, so under a rank-based abstention rule
these variants land in a tie block, which the paper's selective layer has to say something about.

    python analyses/scripts/zero_score_tally.py

Writes analyses/results/zero_scores.json.
"""
import io
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SPECIES = ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]
OUT = "analyses/results/zero_scores.json"


def main():
    per, n_tot, n_zero, n_ident = {}, 0, 0, 0
    for sp in SPECIES:
        f = "data/processed/scores/%s_evo2_40b_local_scores.parquet" % sp
        w = "data/interim/%s_scoring_windows.parquet" % sp
        if not os.path.exists(f):
            continue
        d = pl.read_parquet(f)
        col = [c for c in d.columns if "evo2" in c.lower()][0]
        s = d[col].to_numpy().astype(float)
        zero_ids = d.filter(pl.col(col) == 0)["variant_id"].to_list()
        ident = 0
        if os.path.exists(w) and zero_ids:
            ww = pl.read_parquet(w)
            same = set(ww["variant_id"][i] for i in range(ww.height)
                       if ww["ref_seq"][i] == ww["alt_seq"][i])
            ident = sum(1 for v in zero_ids if v in same)
        per[sp] = {"n": int(len(s)), "n_zero": len(zero_ids),
                   "n_zero_identical_window": ident,
                   "n_zero_numerical_tie": len(zero_ids) - ident}
        n_tot += len(s)
        n_zero += len(zero_ids)
        n_ident += ident

    out = {
        "_generated_by": "analyses/scripts/zero_score_tally.py",
        "_readout": "1,001-bp single-position variant delta, Evo 2-40B, nine-species atlas",
        "_why": "An exactly-zero delta means no expressed preference. Two causes: an identical "
                "ref/alt window (the assembly carries the alternate allele) or a numerical tie in "
                "reduced-precision inference. Separating them matters because a whole panel of "
                "zeros is instead the signature of a var_off fault.",
        "n": n_tot,
        "n_zero": n_zero,
        "n_zero_identical_window": n_ident,
        "n_zero_numerical_tie": n_zero - n_ident,
        "pct_zero": round(100.0 * n_zero / n_tot, 4) if n_tot else None,
        "per_species": per,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print("  %-9s %7s %7s %11s %8s" % ("species", "n", "zeros", "identical", "tie"))
    for sp, v in per.items():
        print("  %-9s %7d %7d %11d %8d"
              % (sp, v["n"], v["n_zero"], v["n_zero_identical_window"],
                 v["n_zero_numerical_tie"]))
    print("\n  %d of %d scores exactly zero (%.3f%%): %d identical window, %d numerical tie"
          % (n_zero, n_tot, out["pct_zero"], n_ident, n_zero - n_ident))
    print("  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
