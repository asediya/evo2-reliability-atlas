"""Fine-mapping input, part 1: derive per-eGene z-scores from CattleGTEx nominals for SuSiE-RSS.
For every significant eGene (from the panel positives), pull ALL its cis variants' nominal
association stats and compute z = sign(slope) * Phi^{-1}(1 - p/2). DuckDB scans all nominals
across 52 threads (CPU-heavy). Output one parquet per tissue for the SuSiE driver.

  python extract_finemap_z.py --eqtl-dir <dir> --panel labeled.parquet --out-dir finemap_z --threads 52
"""
import argparse, glob, os
import numpy as np
import duckdb
import polars as pl
from scipy.special import ndtri  # inverse standard-normal CDF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqtl-dir", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=52)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    con = duckdb.connect(); con.execute(f"PRAGMA threads={a.threads}")
    panel = pl.read_parquet(a.panel)
    sig = panel.filter(pl.col("label") == 1).select(["tissue", "gene"]).unique()

    total = 0
    for tissue in sig["tissue"].unique().to_list():
        genes = sig.filter(pl.col("tissue") == tissue)["gene"].to_list()
        f = os.path.join(a.eqtl_dir, f"{tissue}.nominals.2rd.txt.gz").replace(os.sep, "/")
        if not os.path.exists(f):
            print(f"  {tissue}: nominals missing, skip"); continue
        gl = "','".join(genes)
        df = con.execute(f"""
            SELECT gene, varid, TRY_CAST(dist AS BIGINT) AS dist,
                   TRY_CAST(p AS DOUBLE) AS p, TRY_CAST(slope AS DOUBLE) AS slope
            FROM read_csv('{f}', delim=' ', header=false, null_padding=true, ignore_errors=true,
                 names=['gene','varid','dist','p','slope'])
            WHERE gene IN ('{gl}') AND p IS NOT NULL
        """).pl()
        if df.height == 0:
            continue
        p = np.clip(df["p"].to_numpy().astype(float), 1e-300, 1.0)
        slope = df["slope"].to_numpy().astype(float)
        z = np.sign(slope) * ndtri(1.0 - p / 2.0)
        df = df.with_columns([pl.Series("z", z), pl.lit(tissue).alias("tissue")])
        # split varid -> chrom,pos,ref,alt for LD matching
        df = df.with_columns(pl.col("varid").str.split("_").alias("pp")).with_columns([
            pl.col("pp").list.get(0).alias("chrom"),
            pl.col("pp").list.get(1).cast(pl.Int64, strict=False).alias("pos"),
        ]).drop("pp")
        out = os.path.join(a.out_dir, f"{tissue}.finemap_z.parquet")
        df.write_parquet(out)
        total += df.height
        print(f"  {tissue:18s} genes={len(genes)} variants={df.height}", flush=True)
    print(f"TOTAL fine-map z rows: {total}")


if __name__ == "__main__":
    main()
