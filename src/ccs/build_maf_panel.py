"""Large-N constraint benchmark from the in-hand cattle genotype panel (no new download).
Positives = RARE SNVs (MAF in [rare_lo,rare_hi], enriched for purifying selection = proxy-deleterious);
negatives = COMMON SNVs (MAF>=common_lo, proxy-benign), matched to positives on trinucleotide context
(controls local mutation rate so the signal isn't composition). A noisy but freely-scaling second
ground-truth that well-powers the calibration/coverage curves (OMIA has only ~180 positives).

  micromamba run -p ccs-bio python build_maf_panel.py --afreq-dir geno_pgen --genome cattle.fa \
     --out maf_panel.parquet --n-pos 2500 --neg-per-pos 1
"""
import argparse, glob, os, subprocess
import numpy as np
import polars as pl

W, C = 101, 50


def extract_feats(genome, chroms, poss, tag):
    reg = "\n".join(f"{c}:{p-C}-{p+C}" for c, p in zip(chroms, poss)) + "\n"
    rp = f"/tmp/maf_{tag}.regions"; open(rp, "w").write(reg)
    out = subprocess.run(["samtools", "faidx", genome, "-r", rp], capture_output=True, text=True).stdout
    rows, cur_hdr, cur = [], None, []

    def flush():
        if cur_hdr is not None:
            s = "".join(cur).upper(); chrom, rng = cur_hdr[1:].rsplit(":", 1)
            pos = int(rng.split("-")[0]) + C
            if len(s) == W:
                rows.append((chrom, pos, s[C - 1:C + 2]))
    for ln in out.splitlines():
        if ln.startswith(">"): flush(); cur_hdr = ln; cur = []
        else: cur.append(ln.strip())
    flush(); os.remove(rp)
    return pl.DataFrame(rows, schema=["chrom", "pos", "tri"], orient="row").with_columns(pl.col("pos").cast(pl.Int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afreq-dir", required=True); ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n-pos", type=int, default=2500)
    ap.add_argument("--neg-per-pos", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rare-lo", type=float, default=0.005); ap.add_argument("--rare-hi", type=float, default=0.02)
    ap.add_argument("--common-lo", type=float, default=0.2)
    a = ap.parse_args()

    af = []
    for f in sorted(glob.glob(os.path.join(a.afreq_dir, "*.afreq"))):
        d = pl.read_csv(f, separator="\t")
        cc = "#CHROM" if "#CHROM" in d.columns else d.columns[0]
        d = d.select([pl.col("ID").alias("variant_id"), pl.col("ALT_FREQS").cast(pl.Float64).alias("altf")])
        d = d.with_columns(pl.min_horizontal(pl.col("altf"), 1 - pl.col("altf")).alias("maf"))
        # parse variant_id = chrom_pos_ref_alt
        d = d.with_columns(pl.col("variant_id").str.split("_").alias("p")).with_columns([
            pl.col("p").list.get(0).alias("chrom"), pl.col("p").list.get(1).cast(pl.Int64, strict=False).alias("pos"),
            pl.col("p").list.get(2).alias("ref"), pl.col("p").list.get(3).alias("alt")]).drop("p")
        d = d.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1) & pl.col("pos").is_not_null())
        af.append(d)
    af = pl.concat(af)
    rare = af.filter((pl.col("maf") >= a.rare_lo) & (pl.col("maf") <= a.rare_hi))
    common = af.filter(pl.col("maf") >= a.common_lo)
    print(f"pool: rare={rare.height} common={common.height}")
    rare = rare.sample(n=min(a.n_pos, rare.height), seed=a.seed)
    common = common.sample(n=min(a.n_pos * a.neg_per_pos * 6, common.height), seed=a.seed)

    # trinucleotide context
    rare = rare.join(extract_feats(a.genome, rare["chrom"].to_list(), rare["pos"].to_list(), "rare"), on=["chrom", "pos"], how="inner")
    common = common.join(extract_feats(a.genome, common["chrom"].to_list(), common["pos"].to_list(), "com"), on=["chrom", "pos"], how="inner")
    rare = rare.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))
    common = common.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))

    pool_by = {(k[0] if isinstance(k, tuple) else k): sub for k, sub in common.group_by("mkey")}
    used, neg_rows, matched = set(), [], 0
    for r in rare.iter_rows(named=True):
        cand = pool_by.get(r["mkey"])
        if cand is None: continue
        picked = 0
        for vid, ch, ps, rf, al in zip(cand["variant_id"], cand["chrom"], cand["pos"], cand["ref"], cand["alt"]):
            if (ch, ps) in used: continue
            used.add((ch, ps)); neg_rows.append({"variant_id": vid, "chrom": ch, "pos": ps, "ref": rf, "alt": al})
            picked += 1
            if picked >= a.neg_per_pos: break
        if picked: matched += 1

    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label"]
    posn = rare.with_columns(pl.lit(1).alias("label")).select(cols)
    negn = pl.DataFrame(neg_rows).with_columns(pl.lit(0).alias("label")).select(cols)
    panel = pl.concat([posn, negn]).unique(subset=["variant_id"])
    panel.write_parquet(a.out)
    print(f"matched {matched}/{rare.height} rare | panel {panel.height} "
          f"(rare/pos={panel.filter(pl.col('label')==1).height} common/neg={panel.filter(pl.col('label')==0).height})")


if __name__ == "__main__":
    main()
