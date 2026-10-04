# -*- coding: utf-8 -*-
"""Is the species arm exposed to the same between-gene artefact that retracted the human comparison?

The human defect had a specific shape: ClinVar puts pathogenic AND benign variants in the SAME
disease genes, so which genes a scorer covers determines which variants it sees, and pooling across
genes let coverage masquerade as skill. Holding gene fixed flipped the sign.

The species panels are built differently. Positives are OMIA causal variants -- 879 across 474
genes, median one per gene -- and negatives are trinucleotide-matched population variants sampled
genome-wide. If the two classes barely share genes, then there is no within-gene comparison to make,
and the human artefact cannot arise in the same form. That is not automatically reassuring: it means
the comparison is BETWEEN genic and non-genic sequence by construction, which is a composition
statement the paper already makes.

Measured here through variant consequence, which is available for both classes, rather than through
gene, which is available only for positives.
"""
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
# this hard-coded the author's drive, so the script could not run
# from a clean extraction of the deposit. Resolve the repository root from this file instead.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "src/ccs")
from fig5_stats import SP                                              # noqa: E402

GENIC = ("missense", "synonymous", "stop", "start", "splice", "coding", "frameshift",
         "initiator", "intron", "UTR", "intragenic")

print("  %-9s %8s %8s %26s %26s" % ("species", "n_pos", "n_neg",
                                    "positives in genic terms", "negatives in genic terms"))
rows = []
for sp in SP:
    f = "data/processed/consequence_%s.parquet" % sp
    if not os.path.exists(f):
        print("  %-9s  (no consequence table)" % sp)
        continue
    cons, ev = SP[sp]
    g = pl.read_parquet("data/processed/conservation/%s_gerp.parquet" % cons).select(["variant_id"])
    c = pl.read_parquet(f).select(["variant_id", "consequence"])
    d = g.join(c, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in vid])
    term = np.array([str(t) for t in d["consequence"].to_list()])
    genic = np.array([any(k in t for k in GENIC) for t in term])
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        continue
    pg, ng = genic[y == 1].mean(), genic[y == 0].mean()
    rows.append((sp, npos, nneg, pg, ng))
    print("  %-9s %8d %8d %25.1f%% %25.1f%%" % (sp, npos, nneg, 100 * pg, 100 * ng))

print()
print("  === do the two classes occupy the same consequence terms? ===")
print("  %-9s %-34s %10s %10s" % ("species", "term", "n_pos", "n_neg"))
shared_any = 0
for sp in SP:
    f = "data/processed/consequence_%s.parquet" % sp
    if not os.path.exists(f):
        continue
    cons, ev = SP[sp]
    g = pl.read_parquet("data/processed/conservation/%s_gerp.parquet" % cons).select(["variant_id"])
    c = pl.read_parquet(f).select(["variant_id", "consequence"])
    d = g.join(c, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in vid])
    term = np.array([str(t) for t in d["consequence"].to_list()])
    shared = 0
    for t in sorted(set(term)):
        m = term == t
        a, b = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if a >= 10 and b >= 10:
            shared += 1
            print("  %-9s %-34s %10d %10d" % (sp, t[:34], a, b))
    shared_any += shared > 0

if not rows:
    print()
    print("  NODATA: no per-species consequence table found under data/processed/. This gate needs")
    print("  the undeposited data tree. It examined nothing, so it must not be recorded as a pass.")
    sys.exit(3)

print()
print("  species with at least one consequence term carrying >=10 of BOTH classes: %d of %d"
      % (shared_any, len(rows)))
print()
print("  Reading: where the two classes barely co-occupy a stratum, a within-stratum comparison is")
print("  not identified -- there is nothing to hold fixed. That is a different situation from the")
print("  human panel, where both classes sat in the same genes and pooling hid a sign flip.")
