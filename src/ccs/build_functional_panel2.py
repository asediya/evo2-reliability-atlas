"""From the cached snpEff class table (_allclasses), build TWO matched functional panels sharing a
synonymous negative pool, plus the deduplicated union to score once with Evo2/phyloP:

  - LOF vs synonymous     : clean strong-signal (confirms transfer at >>OMIA N)
  - missense vs synonymous: graded constraint signal, large-N -> a RICH calibration curve across [0,1]

Both trinucleotide+alt matched (removes the composition shortcut).
  python build_functional_panel2.py --allclasses functional_panel_allclasses.parquet \
     --genome cattle.fa --outdir data/interim --cap-mis 2000
"""
import argparse, os, subprocess
import polars as pl


def extract_tri(genome, chroms, poss, tag):
    reg = "\n".join(f"{c}:{p-1}-{p+1}" for c, p in zip(chroms, poss)) + "\n"
    rp = f"/tmp/ftri_{tag}.regions"; open(rp, "w").write(reg)
    out = subprocess.run(["samtools", "faidx", genome, "-r", rp], capture_output=True, text=True).stdout
    rows, hdr, cur = [], None, []

    def flush():
        if hdr is not None:
            s = "".join(cur).upper(); chrom, rng = hdr[1:].rsplit(":", 1)
            pos = int(rng.split("-")[0]) + 1
            if len(s) == 3:
                rows.append((chrom, pos, s))
    for ln in out.splitlines():
        if ln.startswith(">"): flush(); hdr = ln; cur = []
        else: cur.append(ln.strip())
    flush(); os.remove(rp)
    return pl.DataFrame(rows, schema=["chrom", "pos", "tri"], orient="row").with_columns(pl.col("pos").cast(pl.Int64))


def add_mkey(d, genome, tag):
    d = d.join(extract_tri(genome, d["chrom"].to_list(), d["pos"].to_list(), tag), on=["chrom", "pos"], how="inner")
    return d.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))


def match_negs(pos, synpool, used):
    pool = {(k[0] if isinstance(k, tuple) else k): sub for k, sub in synpool.group_by("mkey")}
    neg_rows, matched = [], 0
    for r in pos.iter_rows(named=True):
        cand = pool.get(r["mkey"])
        if cand is None:
            continue
        for vid, ch, ps, rf, al in zip(cand["variant_id"], cand["chrom"], cand["pos"], cand["ref"], cand["alt"]):
            if (ch, ps) in used:
                continue
            used.add((ch, ps)); neg_rows.append({"variant_id": vid, "chrom": ch, "pos": ps, "ref": rf, "alt": al})
            matched += 1; break
    return pl.DataFrame(neg_rows), matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allclasses", required=True); ap.add_argument("--genome", required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--cap-mis", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pl.read_parquet(a.allclasses)
    lof = df.filter(pl.col("cls") == "del")
    mis = df.filter(pl.col("cls") == "missense")
    syn = df.filter(pl.col("cls") == "syn")
    if mis.height > a.cap_mis:
        mis = mis.sample(n=a.cap_mis, seed=a.seed)
    print(f"LOF={lof.height} missense(sampled)={mis.height} syn_pool={syn.height}", flush=True)

    lof = add_mkey(lof, a.genome, "lof"); mis = add_mkey(mis, a.genome, "mis"); syn = add_mkey(syn, a.genome, "syn")
    used = set()
    lof_neg, m1 = match_negs(lof, syn, used)
    mis_neg, m2 = match_negs(mis, syn, used)   # disjoint syn negatives (shared `used`)
    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label"]

    def panel(pos, neg):
        p = pos.with_columns(pl.lit(1).alias("label")).select(cols)
        n = neg.with_columns(pl.lit(0).alias("label")).select(cols)
        return pl.concat([p, n]).unique(subset=["variant_id"])

    lofp = panel(lof, lof_neg); misp = panel(mis, mis_neg)
    lofp.write_parquet(os.path.join(a.outdir, "func_lof_panel.parquet"))
    misp.write_parquet(os.path.join(a.outdir, "func_mis_panel.parquet"))
    union = pl.concat([lofp.select(["variant_id", "chrom", "pos", "ref", "alt"]),
                       misp.select(["variant_id", "chrom", "pos", "ref", "alt"])]).unique(subset=["variant_id"])
    union.write_parquet(os.path.join(a.outdir, "func_scoring_set.parquet"))
    print(f"LOF panel: {lofp.height} (pos={lofp.filter(pl.col('label')==1).height}) matched={m1}", flush=True)
    print(f"missense panel: {misp.height} (pos={misp.filter(pl.col('label')==1).height}) matched={m2}", flush=True)
    print(f"union scoring set: {union.height} variants -> {2*union.height} Evo2 sequences", flush=True)


if __name__ == "__main__":
    main()
