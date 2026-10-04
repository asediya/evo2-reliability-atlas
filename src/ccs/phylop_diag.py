"""Diagnose the phyloP signal: distribution for pos vs neg, value range sanity, and bigWig
coordinate sanity (chrom names, stats). Run in ccs-bio (pyBigWig)."""
import sys, os
import numpy as np
import polars as pl
import pyBigWig

panel = pl.read_parquet(sys.argv[1])
ph = pl.read_parquet(sys.argv[2])
bwdir = sys.argv[3]

df = panel.join(ph, on="variant_id", how="inner").filter(pl.col("phylop").is_not_nan())
pos = df.filter(pl.col("label") == 1)["phylop"].to_numpy()
neg = df.filter(pl.col("label") == 0)["phylop"].to_numpy()
allv = df["phylop"].to_numpy()
print(f"non-nan variants: {len(allv)}  (pos={len(pos)} neg={len(neg)})")
print(f"phyloP POS: mean={pos.mean():.3f} median={np.median(pos):.3f}")
print(f"phyloP NEG: mean={neg.mean():.3f} median={np.median(neg):.3f}")
print(f"phyloP overall pctiles [0,1,25,50,75,99,100]: {np.percentile(allv,[0,1,25,50,75,99,100]).round(3)}")
print(f"fraction phyloP>2 (conserved) pos={np.mean(pos>2):.3f} neg={np.mean(neg>2):.3f}")

print("\n=== bigWig sanity (chr1) ===")
bw = pyBigWig.open(os.path.join(bwdir, "phyloP_bosTau9_chr1.bw"))
ch = bw.chroms()
print("chroms in file:", list(ch.items())[:5])
name = "chr1" if "chr1" in ch else list(ch.keys())[0]
print(f"{name} length={ch[name]} mean={bw.stats(name,type='mean')} max={bw.stats(name,type='max')} min={bw.stats(name,type='min')}")
# sample 5 positive variants on chr1 and show their phyloP directly from the bigWig
p1 = panel.filter((pl.col("label")==1) & (pl.col("chrom")=="1")).head(5)
for r in p1.iter_rows(named=True):
    v = bw.values(name, int(r["pos"])-1, int(r["pos"]))[0]
    print(f"  chr1:{r['pos']} ({r['variant_id']}) -> phyloP={v}")
bw.close()
