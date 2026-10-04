"""Large-N functional-consequence panel = the signal-carrying replication that well-powers the
calibration curves (OMIA has only ~180 positives).

Positives  = LOF SNVs (HIGH impact: stop_gained, splice_acceptor/donor, start_lost, stop_lost)
             -> genuine strong-effect, the class Evo2 excels on (OMIA stop-gain AUROC ~0.97).
Negatives  = synonymous SNVs (LOW impact, in the SAME coding regions -> composition-comparable,
             isolating functional disruption from base composition), trinucleotide + alt matched.
Also emits a `missense` class column for stratified analysis.

Input: snpEff-annotated (and pre-grepped) VCF + the sample_vars parquet (for af) + genome (trinuc).
  python build_functional_panel.py --ann annotated.vcf --vars sample_vars.parquet \
     --genome cattle.fa --out functional_panel.parquet --cap-pos 4000
"""
import argparse, os, subprocess
import numpy as np
import polars as pl

C = 50  # trinuc flank for samtools (window 101, center base + 1 each side)
HIGH = ["stop_gained", "splice_acceptor_variant", "splice_donor_variant", "start_lost", "stop_lost"]


def classify(info):
    """Most-severe class from a snpEff INFO string. Returns 'del' | 'missense' | 'syn' | None."""
    ann = ""
    for kv in info.split(";"):
        if kv.startswith("ANN="):
            ann = kv[4:]; break
    if not ann:
        return None
    toks = ann  # scanning the whole ANN blob for effect tokens is sufficient for classing
    if any(h in toks for h in HIGH):
        return "del"
    if "missense_variant" in toks:
        return "missense"
    if "synonymous_variant" in toks:
        return "syn"
    return None


def extract_tri(genome, chroms, poss, tag):
    """Trinucleotide context (ref-centered) via one samtools faidx call; join by coordinate."""
    reg = "\n".join(f"{c}:{p-1}-{p+1}" for c, p in zip(chroms, poss)) + "\n"
    rp = f"/tmp/func_{tag}.regions"; open(rp, "w").write(reg)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True); ap.add_argument("--vars", required=True)
    ap.add_argument("--genome", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cap-pos", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # parse the (pre-filtered) annotated VCF
    recs = []
    with open(a.ann) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            chrom, pos, vid, ref, alt, info = p[0], int(p[1]), p[2], p[3], p[4], p[7]
            cls = classify(info)
            if cls:
                recs.append((vid, chrom, pos, ref, alt, cls))
    df = pl.DataFrame(recs, schema=["variant_id", "chrom", "pos", "ref", "alt", "cls"], orient="row")
    df = df.unique(subset=["variant_id"])
    n = df.group_by("cls").len().sort("cls")
    print("class counts:", n.to_dicts(), flush=True)

    dele = df.filter(pl.col("cls") == "del")
    syn = df.filter(pl.col("cls") == "syn")
    if dele.height > a.cap_pos:
        dele = dele.sample(n=a.cap_pos, seed=a.seed)
    print(f"LOF positives={dele.height}  synonymous pool={syn.height}", flush=True)

    # trinucleotide context for matching
    dele = dele.join(extract_tri(a.genome, dele["chrom"].to_list(), dele["pos"].to_list(), "del"),
                     on=["chrom", "pos"], how="inner")
    syn = syn.join(extract_tri(a.genome, syn["chrom"].to_list(), syn["pos"].to_list(), "syn"),
                   on=["chrom", "pos"], how="inner")
    dele = dele.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))
    syn = syn.with_columns((pl.col("tri") + ">" + pl.col("alt")).alias("mkey"))

    pool = {(k[0] if isinstance(k, tuple) else k): sub for k, sub in syn.group_by("mkey")}
    used, neg_rows, matched = set(), [], 0
    for r in dele.iter_rows(named=True):
        cand = pool.get(r["mkey"])
        if cand is None:
            continue
        for vid, ch, ps, rf, al in zip(cand["variant_id"], cand["chrom"], cand["pos"], cand["ref"], cand["alt"]):
            if (ch, ps) in used:
                continue
            used.add((ch, ps)); neg_rows.append({"variant_id": vid, "chrom": ch, "pos": ps, "ref": rf, "alt": al})
            matched += 1; break

    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label"]
    posn = dele.with_columns(pl.lit(1).alias("label")).select(cols)
    negn = pl.DataFrame(neg_rows).with_columns(pl.lit(0).alias("label")).select(cols)
    panel = pl.concat([posn, negn]).unique(subset=["variant_id"])
    panel.write_parquet(a.out)
    print(f"matched {matched}/{dele.height} LOF | panel {panel.height} "
          f"(LOF/pos={panel.filter(pl.col('label')==1).height} syn/neg={panel.filter(pl.col('label')==0).height})", flush=True)
    # also stash the full classified table (for missense stratification later)
    df.write_parquet(a.out.replace(".parquet", "_allclasses.parquet"))


if __name__ == "__main__":
    main()
