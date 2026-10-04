"""Genome-anchored, strand-aware window builder for dog (post-liftOver).
After CanFam3.1->ROS_Cfam liftOver, OMIA ref/alt can be strand-flipped and/or allele-swapped
relative to the genome. We anchor to the genome: read the reference base, resolve which strand
the OMIA alleles are on, then build explicit wild-type (ref_seq) and disease (alt_seq) windows.
delta = logL(disease) - logL(wildtype) is thus always the correct deleteriousness direction.

Drops variants where the genome base matches neither allele nor its complement (bad coordinate).
Output: dog_windows.parquet [variant_id, ref_seq, alt_seq, var_off, ref_ok]
"""
import argparse, sys
import polars as pl
from pathlib import Path
try:                                     # package form: python -m src.ccs.<name>
    from .extract_windows_local import load_fai, fetch, W, HALF
except ImportError:                      # file form: python src/ccs/<name>.py
    from extract_windows_local import load_fai, fetch, W, HALF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/interim/dog_omia_pos.parquet")
    ap.add_argument("--genome", default="data/raw/genomes/dog/dog_ROS_Cfam.fa")
    ap.add_argument("--out", default="data/interim/dog_windows.parquet")
    a = ap.parse_args()

    idx = load_fai(a.genome + ".fai")
    df = pl.read_parquet(a.inp).unique(subset=["variant_id"])
    rows = df.select(["variant_id", "chrom", "pos", "ref", "alt"]).rows()

    vid, refs, alts, ok = [], [], [], []
    dropped = 0
    swapped = 0
    with open(a.genome, "rb") as fh:
        for v, c, p, r, al in rows:
            s = fetch(fh, idx, str(c), p - HALF, p + (W - HALF - 1))
            if s is None or len(s) != W:
                dropped += 1; continue
            g = s[HALF]; r, al = r.upper(), al.upper()
            # Deep-layer review, finding D3. The previous form of this branch tested `g in (r, al)`
            # and then wrote `r` over the genome base regardless of which of the two matched. When
            # the genome carries OMIA's ALTERNATE allele -- a reference/alternate swap relative to
            # CanFam3.1 -- that produced a "wild-type" window that does not occur in the assembly
            # and scored the substitution in the wrong direction, while `ok.append(True)` recorded
            # the build as verified. Swaps are now detected, counted and dropped, and ref_ok records
            # the genuine result instead of a constant.
            #
            # Blast radius on the deposited panel: at most one variant. build_atlas8192.py rejects
            # reference mismatches independently, and dog loses exactly one variant between the
            # 1,001-bp panel (2,497) and the 8,192-bp panel (2,496). The 8,192-bp arm, which carries
            # every headline number, cannot contain a swapped record at all.
            if g == r:                              # genome carries the OMIA reference
                wt, dis = r, al
            elif g == COMP.get(r):                  # same, opposite strand
                wt, dis = COMP[r], COMP[al]
            elif g in (al, COMP.get(al)):           # genome carries the OMIA ALTERNATE -> swapped
                swapped += 1; continue
            else:
                dropped += 1; continue              # genome matches neither -> bad coord
            # build wild-type + disease windows anchored on genome flanks
            wt_win = s[:HALF] + wt + s[HALF + 1:]
            dis_win = s[:HALF] + dis + s[HALF + 1:]
            vid.append(v); refs.append(wt_win); alts.append(dis_win); ok.append(wt_win[HALF] == g)

    out = pl.DataFrame({"variant_id": vid, "ref_seq": refs, "alt_seq": alts}).with_columns(
        [pl.lit(HALF).alias("var_off"), pl.Series("ref_ok", ok)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"dog windows: {out.height} recovered (genome-anchored) | dropped {dropped} bad-coord/edge "
          f"| {swapped} dropped as reference/alternate swapped against the assembly")


if __name__ == "__main__":
    main()
