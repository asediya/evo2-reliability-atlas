"""Build the OMIA deleterious-variant test panel.
Positives = OMIA cattle causal SNVs (on ARS-UCD1.2). Negatives = random common SNVs sampled
from the imputed genotype panel (.pvar), 'variant-type-blind' controls.

  python build_omia_panel.py --omia omia_cattle_snvs.csv --pvar-dir geno_pgen \
    --out omia_panel.parquet --neg-per-pos 10
"""
import argparse, glob, os
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omia", required=True)
    ap.add_argument("--pvar-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos = pl.read_csv(a.omia)
    # keep SNVs on ARS-UCD1.2 with clean alleles
    if "on_ARS_UCD1_2" in pos.columns:
        pos = pos.filter(pl.col("on_ARS_UCD1_2").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"]))
    pos = pos.with_columns([pl.col("chrom").cast(pl.Utf8), pl.col("pos").cast(pl.Int64, strict=False),
                            pl.col("ref").cast(pl.Utf8), pl.col("alt").cast(pl.Utf8)])
    pos = pos.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1) &
                     pl.col("pos").is_not_null())
    pos = pos.unique(subset=["variant_id"]).with_columns(pl.lit(1).alias("label"))
    npos = pos.height
    print(f"OMIA positives (SNV, ARS-UCD1.2): {npos}")

    # sample negatives from .pvar (plink2: ## comments, header '#CHROM POS ID REF ALT ...')
    pv = []
    for f in sorted(glob.glob(os.path.join(a.pvar_dir, "*.pvar"))):
        df = pl.read_csv(f, separator="\t", comment_prefix="##")
        cols = df.columns
        cc = "#CHROM" if "#CHROM" in cols else cols[0]
        df = df.select([pl.col(cc).cast(pl.Utf8).alias("chrom"), pl.col("POS").cast(pl.Int64).alias("pos"),
                        pl.col("REF").alias("ref"), pl.col("ALT").alias("alt")])
        df = df.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
        pv.append(df.sample(n=min(df.height, 60000), seed=a.seed))
    pool = pl.concat(pv)
    # exclude any positive positions
    pool = pool.join(pos.select(["chrom", "pos"]), on=["chrom", "pos"], how="anti")
    n_neg = min(pool.height, npos * a.neg_per_pos)
    neg = pool.sample(n=n_neg, seed=a.seed).with_columns([
        (pl.col("chrom") + "_" + pl.col("pos").cast(pl.Utf8) + "_" + pl.col("ref") + "_" + pl.col("alt")).alias("variant_id"),
        pl.lit(0).alias("label"),
    ])
    print(f"negatives (random common SNVs): {neg.height}")

    cols = ["variant_id", "chrom", "pos", "ref", "alt", "label"]
    panel = pl.concat([pos.select(cols), neg.select(cols)]).unique(subset=["variant_id"])
    panel.write_parquet(a.out)
    print(f"wrote {a.out}: {panel.height} rows (pos={panel.filter(pl.col('label')==1).height} "
          f"neg={panel.filter(pl.col('label')==0).height})")


if __name__ == "__main__":
    main()
