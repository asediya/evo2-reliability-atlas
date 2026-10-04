"""Build a composition-matched labeled scoring set for pig, genome-only (no genotype pool).

Positives = genome-verified OMIA pig SNVs (ref_ok). Negatives = random genome sites whose
reference trinucleotide context (pos-1,pos,pos+1) EXACTLY matches a positive's, with the
positive's alt allele assigned — i.e. trinucleotide-context + substitution-type matched, the
same composition control used in CP1. Removes the GC/mutation-rate shortcut.

Output: pig_scoring_windows.parquet [variant_id, ref_seq, alt_seq, var_off, label]
"""
import argparse, sys, random
import polars as pl
from pathlib import Path
try:                                     # package form: python -m src.ccs.<name>
    from .extract_windows_local import load_fai, fetch, W, HALF
except ImportError:                      # file form: python src/ccs/<name>.py
    from extract_windows_local import load_fai, fetch, W, HALF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="data/interim/pig_omia_pos.parquet")
    ap.add_argument("--win", default="data/interim/pig_windows.parquet")
    ap.add_argument("--genome", default="data/raw/genomes/pig/pig.fa")
    ap.add_argument("--out", default="data/interim/pig_scoring_windows.parquet")
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    idx = load_fai(a.genome + ".fai")
    win = pl.read_parquet(a.win).filter(pl.col("ref_ok"))          # 36 verified positives
    pos = pl.read_parquet(a.pos).join(win.select("variant_id"), on="variant_id", how="inner")
    pos = pos.join(win, on="variant_id", how="inner")

    # positive rows carry ref_seq -> trinucleotide context = ref_seq[HALF-1:HALF+2]
    prows = pos.select(["variant_id", "chrom", "pos", "ref", "alt", "ref_seq", "alt_seq", "var_off"]).rows(named=True)
    pos_positions = {(r["chrom"], r["pos"]) for r in prows}

    # demand: how many negatives per (trimer, alt)
    from collections import Counter, defaultdict
    need = Counter()
    for r in prows:
        tri = r["ref_seq"][HALF - 1:HALF + 2]
        need[(tri, r["alt"].upper())] += a.neg_per_pos
    total_need = sum(need.values())

    # main chromosomes only (1..18, X) — avoid scaffolds for clean negatives
    main_chroms = [c for c in idx if c.isdigit() or c in ("X", "Y")]
    weights = [idx[c][0] for c in main_chroms]

    neg = []
    got = defaultdict(int)
    fh = open(a.genome, "rb")
    draws, cap = 0, total_need * 20000
    while sum(got.values()) < total_need and draws < cap:
        draws += 1
        c = rng.choices(main_chroms, weights=weights, k=1)[0]
        p = rng.randint(HALF + 2, idx[c][0] - (W - HALF) - 1)
        if (c, p) in pos_positions:
            continue
        s = fetch(fh, idx, c, p - HALF, p + (W - HALF - 1))
        if s is None or len(s) != W or "N" in s[HALF - 1:HALF + 2]:
            continue
        tri = s[HALF - 1:HALF + 2]
        # try to satisfy any needed (tri, alt) with this ref base = tri[1]
        for (ntri, nalt), q in need.items():
            if ntri == tri and got[(ntri, nalt)] < q and nalt != tri[1]:
                alt_seq = s[:HALF] + nalt + s[HALF + 1:]
                neg.append({"variant_id": f"neg_{c}_{p}_{tri[1]}_{nalt}",
                            "ref_seq": s, "alt_seq": alt_seq, "var_off": HALF, "label": 0})
                got[(ntri, nalt)] += 1
                break
    fh.close()

    posdf = pl.DataFrame([{"variant_id": r["variant_id"], "ref_seq": r["ref_seq"],
                           "alt_seq": r["alt_seq"], "var_off": HALF, "label": 1} for r in prows])
    negdf = pl.DataFrame(neg)
    out = pl.concat([posdf, negdf]).unique(subset=["variant_id"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"pig scoring set: {out.filter(pl.col('label')==1).height} pos + "
          f"{out.filter(pl.col('label')==0).height} neg = {out.height} "
          f"(needed {total_need} neg, draws {draws})")


if __name__ == "__main__":
    main()
