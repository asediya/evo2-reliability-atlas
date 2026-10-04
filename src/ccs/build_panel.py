"""Build the Week-1 cattle variant panel from CattleGTEx QTLtools output.

Positives: lead cis-eQTL of BH-FDR<thr significant eGenes (from *.permutations files, col11 beta perm-p).
Negatives: gene- and TSS-distance-matched non-associated variants (from *.nominals, p>0.5).
Provisional (permutation lead, not fine-mapped; MAF-matching deferred to the genotype stage).

Output: labeled_variants.parquet [variant_id, chrom, pos, ref, alt, gene, tissue, label, dist, source_p]

Runs in the Windows uv venv (duckdb multithreaded + polars):
  python build_panel.py --eqtl-dir <dir> --out labeled.parquet --neg-per-pos 4 --fdr 0.05
"""
import argparse, glob, os
import numpy as np
import duckdb
import polars as pl


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n); ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    qs = q[order]
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(qs, 0, 1)
    return out


def tissue_of(path, kind):
    return os.path.basename(path).replace(f".{kind}.2rd.txt.gz", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqtl-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--max-per-egene", type=int, default=1)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")

    perm_files = sorted(glob.glob(os.path.join(a.eqtl_dir, "*.permutations.2rd.txt.gz")))
    print(f"{len(perm_files)} permutation files")

    # ---- POSITIVES: significant eGene lead variants, per tissue BH-FDR ----
    pos_parts = []
    for f in perm_files:
        t = tissue_of(f, "permutations"); f = f.replace(os.sep, "/")
        df = con.execute(f"""
            SELECT gene, varid, TRY_CAST(dist AS BIGINT) AS dist, TRY_CAST(perm_beta AS DOUBLE) AS perm_p
            FROM read_csv('{f}', delim=' ', header=false, null_padding=true, ignore_errors=true,
                 names=['gene','nvar','sh1','sh2','dummy','varid','dist','nom_p','slope','perm_dir','perm_beta'])
            WHERE perm_beta IS NOT NULL
        """).pl()
        if df.height == 0:
            continue
        q = bh_fdr(df["perm_p"].to_numpy())
        df = df.with_columns([pl.Series("qval", q), pl.lit(t).alias("tissue")])
        df = df.filter(pl.col("qval") < a.fdr)
        pos_parts.append(df)
        print(f"  {t:18s} eGenes(FDR<{a.fdr}): {df.height}")
    pos = pl.concat(pos_parts)
    # parse varid chrom_pos_ref_alt; keep SNVs only
    pos = pos.with_columns([
        pl.col("varid").str.split("_").alias("p"),
    ]).with_columns([
        pl.col("p").list.get(0).alias("chrom"),
        pl.col("p").list.get(1).cast(pl.Int64, strict=False).alias("pos"),
        pl.col("p").list.get(2).alias("ref"),
        pl.col("p").list.get(3).alias("alt"),
    ]).filter(
        (pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1) &
        pl.col("pos").is_not_null()
    ).drop("p")
    # one positive per eGene (already the lead); cap per (tissue,gene)
    pos = pos.unique(subset=["tissue", "gene"], keep="first")
    pos = pos.with_columns([pl.lit(1).alias("label"),
                            pl.col("perm_p").alias("source_p")])
    print(f"TOTAL positives (SNV leads, FDR<{a.fdr}): {pos.height}")

    # ---- NEGATIVES: CROSS-LOCUS, |TSS-distance|-matched background (clearly non-associated) ----
    # Background = variants tested but far from significant (p>0.9), sampled genome-wide (any gene),
    # then matched to the positives' |TSS-distance| distribution per tissue via quantile bins.
    pos = pos.with_columns(pl.col("dist").abs().alias("abs_dist"))
    neg_parts = []
    for f in sorted(glob.glob(os.path.join(a.eqtl_dir, "*.nominals.2rd.txt.gz"))):
        t = tissue_of(f, "nominals"); f = f.replace(os.sep, "/")
        pos_t = pos.filter(pl.col("tissue") == t)
        if pos_t.height == 0:
            continue
        cand = con.execute(f"""
            SELECT varid, TRY_CAST(dist AS BIGINT) AS dist
            FROM read_csv('{f}', delim=' ', header=false, null_padding=true, ignore_errors=true,
                 names=['gene','varid','dist','p','slope'])
            WHERE TRY_CAST(p AS DOUBLE) > 0.9
            USING SAMPLE 400000 ROWS
        """).pl()
        if cand.height == 0:
            continue
        cand = cand.with_columns(pl.col("dist").abs().alias("abs_dist"))
        cand = cand.filter(~pl.col("varid").is_in(pos_t["varid"]))
        ad = pos_t["abs_dist"].to_numpy()
        edges = np.unique(np.quantile(ad, np.linspace(0, 1, 21)))
        picked = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            last = (i == len(edges) - 2)
            pmask = ((pl.col("abs_dist") >= lo) & (pl.col("abs_dist") <= hi)) if last \
                else ((pl.col("abs_dist") >= lo) & (pl.col("abs_dist") < hi))
            npos_b = pos_t.filter(pmask).height
            if npos_b == 0:
                continue
            pool_b = cand.filter(pmask)
            take = min(pool_b.height, npos_b * a.neg_per_pos)
            if take > 0:
                picked.append(pool_b.sample(n=take, seed=i + 1))
        if picked:
            neg_t = pl.concat(picked).with_columns(pl.lit(t).alias("tissue"))
            neg_parts.append(neg_t)
            print(f"  {t:18s} neg (cross-locus bg): {neg_t.height}")
    neg = pl.concat(neg_parts).unique(subset=["varid", "tissue"])
    neg = neg.with_columns(pl.col("varid").str.split("_").alias("pp")).with_columns([
        pl.col("pp").list.get(0).alias("chrom"),
        pl.col("pp").list.get(1).cast(pl.Int64, strict=False).alias("pos"),
        pl.col("pp").list.get(2).alias("ref"),
        pl.col("pp").list.get(3).alias("alt"),
    ]).filter(
        (pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1) &
        pl.col("pos").is_not_null()
    ).drop("pp")
    neg = neg.with_columns([pl.lit(0).alias("label"), pl.lit(1.0).alias("source_p"),
                            pl.lit("NA").alias("gene")])
    print(f"TOTAL negatives (SNV, cross-locus background): {neg.height}")

    cols = ["variant_id", "chrom", "pos", "ref", "alt", "gene", "tissue", "label", "dist", "source_p"]
    pos2 = pos.rename({"varid": "variant_id"}).select(cols)
    neg2 = neg.rename({"varid": "variant_id"}).select(cols)
    panel = pl.concat([pos2, neg2]).unique(subset=["variant_id", "gene", "tissue"], keep="first")
    panel.write_parquet(a.out)
    print(f"\nwrote {a.out}: {panel.height} rows | pos={panel.filter(pl.col('label')==1).height} "
          f"neg={panel.filter(pl.col('label')==0).height} | "
          f"unique variants={panel['variant_id'].n_unique()}")


if __name__ == "__main__":
    main()
