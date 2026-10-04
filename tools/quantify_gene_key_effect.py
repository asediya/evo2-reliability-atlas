# -*- coding: utf-8 -*-
"""How much did the fragmented gene key distort the gene-clustered intervals?

The tracked dbNSFP audit was built before gene canonicalisation and clustered the bootstrap on
38,420 keys standing for 15,606 real genes -- a 2.46x fragmentation. Fragmentation splits a gene
into several clusters that are then resampled independently, which is exactly the assumption
clustering exists to avoid, so it understates the within-gene correlation and returns intervals that
are TOO NARROW. Too narrow is the dangerous direction: it manufactures significance.

This compares the two artifacts predictor by predictor and answers the only two questions that
matter. How much wider are the honest intervals? And did any published verdict depend on the
difference -- that is, did an interval that excluded zero stop excluding it?

Run against the committed version:
    git show HEAD:reports/dbnsfp_reach_audit.json > <old>
    python tools/quantify_gene_key_effect.py <old> reports/dbnsfp_reach_audit.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

old = json.load(open(sys.argv[1], encoding="utf-8"))
new = json.load(open(sys.argv[2], encoding="utf-8"))

print("  gene keys : %s -> %s  (%.2fx fragmentation removed)"
      % (f"{old['_meta']['n_genes']:,}", f"{new['_meta']['n_genes']:,}",
         old["_meta"]["n_genes"] / new["_meta"]["n_genes"]))
print("  draws     : %d vs %d %s" % (old["_meta"]["n_boot"], new["_meta"]["n_boot"],
                                     "(matched)" if old["_meta"]["n_boot"] ==
                                     new["_meta"]["n_boot"] else "(NOT MATCHED -- not comparable)"))
if old["_meta"]["n_boot"] != new["_meta"]["n_boot"]:
    print("  ABORT: the draw counts differ, so width changes confound the gene key with Monte-Carlo")
    sys.exit(2)

o = {p["predictor"]: p for p in old["predictors"]}
n = {p["predictor"]: p for p in new["predictors"]}
shared = [k for k in n if k in o]

CI_FIELDS = ["auroc_covered_ci", "class_gap_ci"]
ratios, moved, flips, pt_moved = [], 0, [], 0
for k in shared:
    for f in CI_FIELDS:
        a, b = o[k].get(f), n[k].get(f)
        if not a or not b or any(x is None for x in list(a) + list(b)):
            continue
        wa, wb = a[1] - a[0], b[1] - b[0]
        if wa > 0:
            ratios.append(wb / wa)
        # a verdict flip: excluded zero before, does not now (or the reverse)
        ex_a = (a[0] > 0) or (a[1] < 0)
        ex_b = (b[0] > 0) or (b[1] < 0)
        if ex_a != ex_b:
            flips.append("%s %s: %s -> %s  [%.4f,%.4f] -> [%.4f,%.4f]"
                         % (k, f, "sig" if ex_a else "ns", "sig" if ex_b else "ns",
                            a[0], a[1], b[0], b[1]))
    # point estimates must NOT move: clustering changes intervals, never the estimate
    for f in ["auroc_covered", "class_gap", "auroc_must_answer"]:
        if f in o[k] and f in n[k] and o[k][f] is not None and n[k][f] is not None:
            if abs(float(o[k][f]) - float(n[k][f])) > 1e-12:
                pt_moved += 1

r = np.array(ratios)
print("  predictors compared : %d" % len(shared))
print("  intervals compared  : %d" % r.size)
print()
print("  width ratio (honest / fragmented)")
print("    median  %.3f      mean %.3f" % (float(np.median(r)), float(r.mean())))
print("    min     %.3f      max  %.3f" % (float(r.min()), float(r.max())))
print("    wider in %d of %d intervals (%.0f%%)"
      % (int((r > 1).sum()), r.size, 100 * (r > 1).mean()))
print()
print("  point estimates that moved : %d  (must be 0 -- clustering changes only intervals)" % pt_moved)
print("  verdicts that flipped      : %d" % len(flips))
for f in flips:
    print("    %s" % f)

if r.size < 50:
    print()
    print("  RESULT: INCONCLUSIVE -- too few intervals compared to characterise the effect")
    sys.exit(2)
if pt_moved:
    print()
    print("  RESULT: FAIL -- point estimates moved, so something other than the gene key changed")
    sys.exit(1)
print()
print("  RESULT: characterised. The fragmented key was anti-conservative by a median %.1f%%."
      % (100 * (float(np.median(r)) - 1)))
