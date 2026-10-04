"""Dog matched negatives from the Ensembl dog-variation SNV pool (29.9M real SNVs, ROS_Cfam_1.0),
trinucleotide+substitution matched to the genome-anchored dog positives. Genome-anchored (handles
strand). Output: dog_scoring_windows.parquet [variant_id, ref_seq, alt_seq, var_off, label]
"""
import argparse, sys, random
from collections import Counter, defaultdict
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
    ap.add_argument("--win", default="data/interim/dog_windows.parquet")
    ap.add_argument("--pos", default="data/interim/dog_omia_pos.parquet")
    ap.add_argument("--pool", default="data/interim/dog_snv_pool.tsv")
    ap.add_argument("--genome", default="data/raw/genomes/dog/dog_ROS_Cfam.fa")
    ap.add_argument("--out", default="data/interim/dog_scoring_windows.parquet")
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    idx = load_fai(a.genome + ".fai")
    win = pl.read_parquet(a.win)                       # recovered positives (wt=ref_seq, dis=alt_seq)
    prows = win.select(["variant_id", "ref_seq", "alt_seq"]).rows(named=True)
    posp = pl.read_parquet(a.pos).join(win.select("variant_id"), on="variant_id", how="inner")
    pos_positions = {(str(c), int(p)) for c, p in posp.select(["chrom", "pos"]).rows()}

    need = Counter()
    for r in prows:
        tri = r["ref_seq"][HALF - 1:HALF + 2]
        need[(tri, r["alt_seq"][HALF])] += a.neg_per_pos
    total_need = sum(need.values())

    pool = pl.read_csv(a.pool, separator="\t", has_header=False,
                       new_columns=["chrom", "pos", "ref", "alt"],
                       schema_overrides={"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8})
    pool = pool.filter(pl.col("chrom").is_in(list(idx.keys())))
    cand = pool.sample(fraction=1.0, shuffle=True, seed=a.seed)

    neg, got = [], defaultdict(int)
    fh = open(a.genome, "rb"); seen = 0
    for c, p, r, al in cand.iter_rows():
        if sum(got.values()) >= total_need:
            break
        seen += 1
        if (str(c), int(p)) in pos_positions:
            continue
        s = fetch(fh, idx, str(c), p - HALF, p + (W - HALF - 1))
        if s is None or len(s) != W or "N" in s[HALF - 1:HALF + 2]:
            continue
        g = s[HALF]; r, al = r.upper(), al.upper()
        if g == r:   altb = al
        elif g == al: altb = r
        elif g == COMP.get(r): altb = COMP[al]
        elif g == COMP.get(al): altb = COMP[r]
        else: continue
        if altb == g:
            continue
        tri = s[HALF - 1:HALF + 2]
        key = (tri, altb)
        if key in need and got[key] < need[key]:
            neg.append({"variant_id": f"neg_{c}_{p}_{g}_{altb}",
                        "ref_seq": s, "alt_seq": s[:HALF] + altb + s[HALF + 1:],
                        "var_off": HALF, "label": 0})
            got[key] += 1
    fh.close()

    posdf = pl.DataFrame([{"variant_id": r["variant_id"], "ref_seq": r["ref_seq"],
                           "alt_seq": r["alt_seq"], "var_off": HALF, "label": 1} for r in prows])
    out = pl.concat([posdf, pl.DataFrame(neg)]).unique(subset=["variant_id"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"dog scoring set: {out.filter(pl.col('label')==1).height} pos + "
          f"{out.filter(pl.col('label')==0).height} neg = {out.height} (need {total_need}, scanned {seen})")


if __name__ == "__main__":
    main()
