# -*- coding: utf-8 -*-
"""Extract hg19 windows for the saturation splicing MPRA variants so Evo 2 can join that panel.

The manuscript's regulatory evidence rests on panels where Evo 2 is the only sequence model and the
comparators are conservation scores. Splicing is the one regulatory task with strong, published,
specialised competitors on the same variants: SpliceAI reads 0.799 here and Pangolin 0.803. Scoring
Evo 2 on exactly these variants is the sharpest available test of whether a general genome language
model reaches task-specific supervised models on regulatory variation.

The panel is hg19 and only hg38 is on disk, so rather than download a genome for six genes, the six
loci are fetched as slabs from the UCSC REST API and cached. Every window is reference-verified: if
the hg19 base at the variant position does not match the table's REF, the variant is dropped, since
a mismatch would invert the sign of the score.

    python analyses/scripts/build_mfass_payload.py
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SRC = "analyses/data/mfass/S2.xlsx"
OUT = "analyses/data/mfass"
CACHE = os.path.join(OUT, "hg19_slabs")
W, HALF = 1001, 500
SHEETS = ["FAS exon 6", "RON exon 11", "POU1F1 exon 2", "WT1 exon 9", "BRCA1", "MLH1"]
API = "https://api.genome.ucsc.edu/getData/sequence?genome=hg19;chrom=%s;start=%d;end=%d"


def slab(chrom, start, end):
    """Fetch [start, end) as 0-based hg19 sequence, cached on disk so a re-run costs nothing."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "%s_%d_%d.txt" % (chrom, start, end))
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(API % (chrom, start, end), timeout=120) as r:
                seq = json.load(r)["dna"].upper()
            open(p, "w", encoding="utf-8").write(seq)
            return seq
        except Exception as e:
            if attempt == 3:
                raise
            print("    retry %d for %s:%d-%d (%s)" % (attempt + 1, chrom, start, end, type(e).__name__))
            time.sleep(5)


def main():
    x = pd.ExcelFile(SRC)
    rows = []
    for s in SHEETS:
        d = x.parse(s)
        lab = [c for c in d.columns if "disrupt" in c.lower()][0]
        for _, r in d.iterrows():
            rows.append({"assay": s, "chrom": str(r["Chrom"]), "pos": int(r["HG19 pos"]),
                         "ref": str(r["ref"]).upper(), "alt": str(r["alt"]).upper(),
                         "variant_class": r["Variant class"], "label": int(bool(r[lab]))})
    t = pd.DataFrame(rows)
    t["chrom"] = t["chrom"].apply(lambda c: c if c.startswith("chr") else "chr" + c)
    print("  %s variants across %d assays" % ("{:,}".format(len(t)), t["assay"].nunique()))

    seqs = {}
    for (assay, chrom), g in t.groupby(["assay", "chrom"]):
        lo = int(g["pos"].min()) - HALF - 10
        hi = int(g["pos"].max()) + HALF + 10
        print("  fetching %-18s %s:%d-%d  (%.1f kb)" % (assay, chrom, lo, hi, (hi - lo) / 1e3))
        seqs[(assay, chrom)] = (lo, slab(chrom, lo, hi))

    vid, refs, alts, offs, labs, cls, asy = [], [], [], [], [], [], []
    bad_ref = short = 0
    for r in t.itertuples(index=False):
        lo, sq = seqs[(r.assay, r.chrom)]
        i = r.pos - 1 - lo                          # 0-based index of the variant inside the slab
        a, b = i - HALF, i + HALF + 1
        if a < 0 or b > len(sq):
            short += 1
            continue
        win = sq[a:b]
        if win[HALF] != r.ref:                      # a mismatch would invert the score's sign
            bad_ref += 1
            continue
        vid.append("%s:%s:%d:%s>%s" % (r.assay, r.chrom, r.pos, r.ref, r.alt))
        refs.append(win)
        alts.append(win[:HALF] + r.alt + win[HALF + 1:])
        # HALF, not HALF + 1. This read "1-based, matching the atlas convention", and the atlas
        # convention does not transfer: the atlas scoring windows are 1,002 bp with the variant at
        # index 501, while these are genuinely 1,001 bp (HALF left, the variant, HALF right) with
        # the variant at index HALF = 500. Carrying 501 over made var_off point one base to the
        # RIGHT of the substitution, so the scorer compared a flanking base against itself and
        # every delta came out exactly zero. The two window constructions share the label
        # "1,001 bp" and differ by one base; that is the whole bug.
        offs.append(HALF)                           # 0-based index of the variant in the window
        labs.append(r.label)
        cls.append(r.variant_class)
        asy.append(r.assay)

    out = pl.DataFrame({"variant_id": vid, "ref_seq": refs, "alt_seq": alts, "var_off": offs,
                        "label": labs, "variant_class": cls, "assay": asy})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "mfass_w1001.parquet")
    out.write_parquet(p)
    print("\n  dropped: %d reference mismatch, %d truncated window" % (bad_ref, short))
    print("  wrote %s: %s variants, %d bp, %.1f MB"
          % (p, "{:,}".format(len(out)), W, os.path.getsize(p) / 1e6))
    for row in out.group_by("assay").agg(pl.len(), pl.col("label").sum()).sort("assay").iter_rows():
        print("    %-18s %5d variants, %4d disruptive" % row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
