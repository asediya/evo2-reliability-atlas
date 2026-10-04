# -*- coding: utf-8 -*-
"""Build a consequence-stratified ClinVar panel and extract 1,001-bp windows for Evo 2 scoring.

Two things this fixes, and the first matters more than the second.

**Evo 2 is currently the auditor and never a participant.** The paper charges CADD, REVEL and
AlphaMissense a must-answer penalty on 1.4 M ClinVar variants and never charges itself on the same
panel. A referee sees that asymmetry immediately. Scoring Evo 2 here puts it into the league table
it critiques, with its own reach, its own class gap and its own identified interval beside the
other 49.

**The regulatory rung of the consequence gradient is n = 18**, with a minimum detectable difference
of 0.28 — that rung could never have resolved anything smaller than a quarter of the AUROC scale.
ClinVar carries tens of thousands of variants in each regulatory class.

Sampling is balanced by (consequence x label) so no cell can dominate, and every window is
verified: the reference base at the variant offset must match the ClinVar REF, or the variant is
dropped. A silent reference mismatch would invert the sign of the score.

    python analyses/scripts/build_clinvar_payload.py --per-cell 250
"""
import argparse
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analyses/data/clinvar"
FASTA = "data/raw/genomes/human/human.fa"
W, HALF = 1001, 500
RNG = np.random.default_rng(20260805)

# The classes that carry the gradient. Regulatory classes are listed separately rather than
# lumped, because lumping them is what makes a region look uniformly hard when a class is not.
CLASSES = ["nonsense", "missense_variant", "splice_donor_variant", "splice_acceptor_variant",
           "5_prime_UTR_variant", "3_prime_UTR_variant", "non-coding_transcript_variant",
           "synonymous_variant"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=250,
                    help="variants per (consequence x label) cell")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d = pl.read_parquet("data/processed/clinvar_panel.parquet")
    print("  ClinVar panel: %s variants" % "{:,}".format(len(d)))

    keep = []
    for c in CLASSES:
        for lab in (1, 0):
            cell = d.filter((pl.col("consequence") == c) & (pl.col("label") == lab))
            if len(cell) == 0:
                print("  %-32s label=%d   EMPTY" % (c, lab))
                continue
            n = min(a.per_cell, len(cell))
            idx = RNG.choice(len(cell), size=n, replace=False)
            keep.append(cell[idx.tolist()])
            print("  %-32s label=%d   %6d available -> %d" % (c, lab, len(cell), n))
    panel = pl.concat(keep)
    print("  sampled %s variants across %d cells" % ("{:,}".format(len(panel)), len(keep)))

    from pyfaidx import Fasta
    fa = Fasta(FASTA, as_raw=True, sequence_always_upper=True)
    names = set(fa.keys())

    vid, refs, alts, offs, labs, cons, chrom_used = [], [], [], [], [], [], []
    bad_chrom = bad_ref = 0
    for r in panel.iter_rows(named=True):
        c = str(r["chrom"])
        name = c if c in names else ("chr" + c if "chr" + c in names else None)
        if name is None:
            bad_chrom += 1
            continue
        pos = int(r["pos"])
        start, end = pos - 1 - HALF, pos - 1 + HALF + 1
        if start < 0:
            continue
        try:
            seq = str(fa[name][start:end]).upper()
        except Exception:
            continue
        if len(seq) != W:
            continue
        off = HALF                                  # 0-based index of the variant in the window
        if seq[off] != str(r["ref"]).upper():
            bad_ref += 1                            # a mismatch would invert the score's sign
            continue
        alt = seq[:off] + str(r["alt"]).upper() + seq[off + 1:]
        vid.append(r["variant_id"])
        refs.append(seq)
        alts.append(alt)
        # `off`, not `off + 1`. Two lines above, `off = HALF` is the index the window was actually
        # built at, and `alt` is constructed by substituting there. Recording off + 1 contradicted
        # the construction on the very next line: the scorer then read a flanking base for both ref
        # and alt and returned exactly zero for every variant. The "atlas convention" this claimed
        # to match belongs to the 1,002-bp atlas windows, not to these 1,001-bp ones.
        offs.append(off)                            # 0-based index of the variant in the window
        labs.append(int(r["label"]))
        cons.append(r["consequence"])
        chrom_used.append(name)
    fa.close()

    out = pl.DataFrame({"variant_id": vid, "ref_seq": refs, "alt_seq": alts,
                        "var_off": offs, "label": labs, "consequence": cons,
                        "chrom": chrom_used})
    p = os.path.join(OUT, "clinvar_w1001.parquet")
    out.write_parquet(p)
    print()
    print("  dropped: %d unmapped chromosome, %d reference mismatch" % (bad_chrom, bad_ref))
    print("  wrote %s: %s variants, %d bp windows, %.1f MB"
          % (p, "{:,}".format(len(out)), W, os.path.getsize(p) / 1e6))
    print("  class mix:")
    for row in (out.group_by(["consequence", "label"]).len()
                .sort(["consequence", "label"]).iter_rows(named=True)):
        print("    %-32s label=%d  %5d" % (row["consequence"], row["label"], row["len"]))
    print()
    print("  passes needed: %s variants x 2 (ref, alt) = %s sequences"
          % ("{:,}".format(len(out)), "{:,}".format(2 * len(out))))


if __name__ == "__main__":
    main()
