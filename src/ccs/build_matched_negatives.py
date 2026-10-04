"""Build composition-MATCHED negatives for the OMIA deleterious-variant panel (runs in ccs-bio).
Each OMIA positive is matched to K background common SNVs sharing the same MUTATION TYPE
(reference trinucleotide context + alt allele) and nearest local GC. This removes the
'coding-vs-intergenic composition' confound so AUROC reflects harmful-vs-benign, not location.

Output: omia_matched_panel.parquet [variant_id, chrom, pos, ref, alt, label]

  micromamba run -p ccs-bio python build_matched_negatives.py --omia omia_cattle_snvs.csv \
     --pvar-dir geno_pgen --genome cattle.fa --out omia_matched_panel.parquet --neg-per-pos 10
"""
import argparse, subprocess, glob, os
import numpy as np
import polars as pl

W, C = 101, 50  # 101bp window, variant at index 50


def extract_feats(genome, chroms, poss, tag):
    """Return DataFrame [chrom, pos, tri, gc] (robust to samtools skipping bad regions:
    parse each header '>chrom:start-end' -> pos=start+C, join back by coordinate)."""
    reg = "\n".join(f"{c}:{p-C}-{p+C}" for c, p in zip(chroms, poss)) + "\n"
    rp = f"/tmp/mn_{tag}.regions"; open(rp, "w").write(reg)
    out = subprocess.run(["samtools", "faidx", genome, "-r", rp], capture_output=True, text=True).stdout
    rows, cur_hdr, cur = [], None, []

    def flush():
        if cur_hdr is not None:
            s = "".join(cur).upper()
            h = cur_hdr[1:]
            chrom, rng = h.rsplit(":", 1)
            start = int(rng.split("-")[0]); pos = start + C
            if len(s) == W:
                rows.append((chrom, pos, s[C - 1:C + 2], (s.count("G") + s.count("C")) / W))
    for ln in out.splitlines():
        if ln.startswith(">"):
            flush(); cur_hdr = ln; cur = []
        else:
            cur.append(ln.strip())
    flush()
    os.remove(rp)
    print(f"[extract {tag}] out_bytes={len(out)} parsed_rows={len(rows)}", flush=True)
    if not rows:
        return pl.DataFrame(schema={"chrom": pl.Utf8, "pos": pl.Int64, "tri": pl.Utf8, "gc": pl.Float64})
    return pl.DataFrame(rows, schema=["chrom", "pos", "tri", "gc"], orient="row").with_columns(
        pl.col("pos").cast(pl.Int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omia", required=True); ap.add_argument("--pvar-dir", required=True)
    ap.add_argument("--genome", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=400000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos = pl.read_csv(a.omia)
    if "on_ARS_UCD1_2" in pos.columns:
        pos = pos.filter(pl.col("on_ARS_UCD1_2").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"]))
    pos = pos.with_columns([pl.col("chrom").cast(pl.Utf8), pl.col("pos").cast(pl.Int64, strict=False),
                            pl.col("ref").cast(pl.Utf8), pl.col("alt").cast(pl.Utf8)]).filter(
        (pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1) & pl.col("pos").is_not_null()
    ).unique(subset=["variant_id"])

    # candidate pool: common SNVs from pgen .pvar
    pv = []
    for f in sorted(glob.glob(os.path.join(a.pvar_dir, "*.pvar"))):
        d = pl.read_csv(f, separator="\t", comment_prefix="##")
        cc = "#CHROM" if "#CHROM" in d.columns else d.columns[0]
        d = d.select([pl.col(cc).cast(pl.Utf8).alias("chrom"), pl.col("POS").cast(pl.Int64).alias("pos"),
                      pl.col("REF").alias("ref"), pl.col("ALT").alias("alt")]).filter(
            (pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
        pv.append(d.sample(n=min(d.height, a.pool_size // 25), seed=a.seed))
    pool = pl.concat(pv).join(pos.select(["chrom", "pos"]), on=["chrom", "pos"], how="anti")

    # features (join by coordinate — robust to skipped regions)
    pos = pos.join(extract_feats(a.genome, pos["chrom"].to_list(), pos["pos"].to_list(), "pos"),
                   on=["chrom", "pos"], how="inner")
    pool = pool.join(extract_feats(a.genome, pool["chrom"].to_list(), pool["pos"].to_list(), "pool"),
                     on=["chrom", "pos"], how="inner")
    # mutation-type key = ref-trinuc>alt
    pos = pos.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))
    pool = pool.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))

    # match: per positive, K nearest-GC pool variants of the SAME mutation type, no reuse
    rng = np.random.default_rng(a.seed)
    pool_by = {(k[0] if isinstance(k, tuple) else k): sub.sort("gc")
               for k, sub in pool.group_by("mkey")}
    used = set()
    neg_rows, matched_pos = [], 0
    for r in pos.iter_rows(named=True):
        cand = pool_by.get(r["mkey"])
        if cand is None: continue
        cg_arr = cand["gc"].to_numpy(); vids = cand["variant_id"] if "variant_id" in cand.columns else None
        chr_ = cand["chrom"].to_list(); ps = cand["pos"].to_list(); rf = cand["ref"].to_list(); al = cand["alt"].to_list()
        idx = np.argsort(np.abs(cg_arr - r["gc"]))
        picked = 0
        for j in idx:
            key = (chr_[j], ps[j])
            if key in used: continue
            used.add(key)
            neg_rows.append({"chrom": chr_[j], "pos": ps[j], "ref": rf[j], "alt": al[j]})
            picked += 1
            if picked >= a.neg_per_pos: break
        if picked: matched_pos += 1

    neg = pl.DataFrame(neg_rows).with_columns([
        (pl.col("chrom") + "_" + pl.col("pos").cast(pl.Utf8) + "_" + pl.col("ref") + "_" + pl.col("alt")).alias("variant_id"),
        pl.lit(0).alias("label")])
    posn = pos.with_columns(pl.lit(1).alias("label"))
    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label"]
    panel = pl.concat([posn.select(cols), neg.select(cols)]).unique(subset=["variant_id"])
    panel.write_parquet(a.out)
    print(f"matched {matched_pos}/{pos.height} positives | panel {panel.height} "
          f"(pos={panel.filter(pl.col('label')==1).height} neg={panel.filter(pl.col('label')==0).height})")
    print(f"neg-per-pos achieved: {neg.height / max(1, matched_pos):.1f} | mean pos GC {pos['gc'].mean():.3f}")


if __name__ == "__main__":
    main()
