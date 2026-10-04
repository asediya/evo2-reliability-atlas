"""Extract ref/alt sequence windows for each variant from the cattle genome (WSL, samtools).
Input: labeled_variants.parquet [variant_id, chrom, pos, ref, alt, ...]
Output: windows.parquet [variant_id, gene, tissue, label, ref_seq, alt_seq, var_off, ref_ok]

Window W=1002 bp centred so the variant base sits at 0-based offset 501.
Batched via a single `samtools faidx -r regions.txt` call. Run inside ccs-bio:
  micromamba run -p ~/micromamba/envs/ccs-bio python extract_windows.py \
     --in labeled.parquet --genome cattle.fa --out windows.parquet
"""
import argparse, subprocess, os
import polars as pl

W = 1002
HALF = 501  # variant at 0-based offset 501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fai", default=None)
    a = ap.parse_args()

    df = pl.read_parquet(a.inp).unique(subset=["variant_id"])
    fai = a.fai or (a.genome + ".fai")
    clen = {r[0]: int(r[1]) for r in (ln.split("\t")[:2] for ln in open(fai))}

    df = df.filter(
        pl.col("chrom").is_in(list(clen.keys())) &
        (pl.col("pos") > HALF + 1) &
        pl.col("chrom").map_elements(lambda c: clen.get(c, 0), return_dtype=pl.Int64).gt(0)
    )
    # drop variants whose window would run off the contig end
    df = df.with_columns(
        pl.struct(["chrom", "pos"]).map_elements(
            lambda s: s["pos"] + (W - HALF) <= clen[s["chrom"]], return_dtype=pl.Boolean
        ).alias("_fits")
    ).filter(pl.col("_fits")).drop("_fits")

    rows = df.select(["variant_id", "chrom", "pos", "ref", "alt"]).rows()
    regions = "\n".join(f"{c}:{p-HALF}-{p+(W-HALF-1)}" for (_, c, p, _, _) in rows) + "\n"
    reg_path = a.out + ".regions.txt"
    open(reg_path, "w").write(regions)

    print(f"extracting {len(rows)} windows via samtools faidx ...", flush=True)
    out = subprocess.run(["samtools", "faidx", a.genome, "-r", reg_path],
                         capture_output=True, text=True, check=True).stdout

    # parse fasta (same order as regions)
    seqs, cur = [], []
    for line in out.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    assert len(seqs) == len(rows), f"{len(seqs)} seqs vs {len(rows)} rows"

    vid, ref_seq, alt_seq, ref_ok = [], [], [], []
    for (v, c, p, rf, al), s in zip(rows, seqs):
        s = s.upper()
        if len(s) != W:
            continue
        ok = s[HALF] == rf.upper()
        alt = s[:HALF] + al.upper() + s[HALF + 1:]
        vid.append(v); ref_seq.append(s); alt_seq.append(alt); ref_ok.append(ok)

    win = pl.DataFrame({"variant_id": vid, "ref_seq": ref_seq, "alt_seq": alt_seq})
    win = win.with_columns([pl.lit(HALF).alias("var_off"), pl.Series("ref_ok", ref_ok)])
    win.write_parquet(a.out)
    n_ok = int(sum(ref_ok))
    print(f"wrote {a.out}: {win.height} windows | ref-allele match: {n_ok}/{len(ref_ok)} "
          f"({100*n_ok/max(1,len(ref_ok)):.1f}%)")
    os.remove(reg_path)


if __name__ == "__main__":
    main()
