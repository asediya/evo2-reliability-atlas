"""Parallel SuSiE-RSS fine-mapping across eGenes (WSL ccs-bio; the all-cores compute job).
For each (tissue, eGene): extract cis-variant dosages from the chr pgen (plink2 --export A),
compute LD in R, run susie_rss -> per-variant PIP + credible-set id. Pool of workers pegs cores.

  micromamba run -p ~/micromamba/envs/ccs-bio python finemap_driver.py \
    --z-dir data/interim/finemap_z --pgen-dir data/interim/geno_pgen \
    --susie-r src/ccs/susie_one.R --out data/processed/cattle_finemapped.parquet \
    --tmpdir /tmp/fm --workers 48
"""
import argparse, os, subprocess, glob
import polars as pl
from multiprocessing import Pool


def worker(t):
    tissue, gene, chrom, varids, zs, pgen, tmpdir, susie_r = t
    base = os.path.join(tmpdir, f"{tissue}__{gene}")
    varf, zf, outf, raw = base + ".vars", base + ".z", base + ".pip", base + ".raw"
    try:
        with open(varf, "w") as fh:
            fh.write("\n".join(varids) + "\n")
        with open(zf, "w") as fh:
            fh.write("variant_id\tz\n")
            for v, zz in zip(varids, zs):
                fh.write(f"{v}\t{zz}\n")
        r = subprocess.run(["plink2", "--pfile", pgen, "--extract", varf, "--export", "A",
                            "--out", base], capture_output=True, text=True, timeout=600)
        if not os.path.exists(raw):
            return None
        subprocess.run(["Rscript", susie_r, raw, zf, outf], capture_output=True, text=True, timeout=600)
        if not os.path.exists(outf):
            return None
        df = pl.read_csv(outf)
        if df.height == 0:
            return None
        return df.with_columns([pl.lit(gene).alias("gene"), pl.lit(tissue).alias("tissue")])
    except Exception:
        return None
    finally:
        for f in [varf, zf, raw, outf, base + ".log"]:
            try:
                os.remove(f)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-dir", required=True)
    ap.add_argument("--pgen-dir", required=True)
    ap.add_argument("--susie-r", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmpdir", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-genes", type=int, default=0)
    ap.add_argument("--cap-variants", type=int, default=2000)
    ap.add_argument("--tissues", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.tmpdir, exist_ok=True)

    tasks = []
    for zfp in sorted(glob.glob(os.path.join(a.z_dir, "*.finemap_z.parquet"))):
        tissue = os.path.basename(zfp).replace(".finemap_z.parquet", "")
        if a.tissues and tissue not in a.tissues:
            continue
        z = pl.read_parquet(zfp)
        vcol = "varid" if "varid" in z.columns else "variant_id"
        for key, sub in z.group_by(["gene", "chrom"]):
            gene, chrom = key
            pgen = os.path.join(a.pgen_dir, f"Chr{chrom}")
            if not os.path.exists(pgen + ".pgen"):
                continue
            sub = sub.drop_nulls(subset=[vcol, "z"])
            if sub.height > a.cap_variants:
                sub = sub.with_columns(pl.col("z").abs().alias("_az")).sort("_az", descending=True).head(a.cap_variants)
            if sub.height < 5:
                continue
            tasks.append((tissue, str(gene), str(chrom), sub[vcol].to_list(),
                          sub["z"].to_list(), pgen, a.tmpdir, a.susie_r))
            if a.max_genes and len(tasks) >= a.max_genes:
                break
        if a.max_genes and len(tasks) >= a.max_genes:
            break

    print(f"{len(tasks)} fine-mapping tasks; workers={a.workers}", flush=True)
    results, done = [], 0
    with Pool(a.workers) as p:
        for r in p.imap_unordered(worker, tasks, chunksize=4):
            done += 1
            if r is not None:
                results.append(r)
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} done, {len(results)} ok", flush=True)

    if results:
        out = pl.concat(results)
        out.write_parquet(a.out)
        cred = out.filter(pl.col("pip") >= 0.9)
        print(f"wrote {a.out}: {out.height} rows, {out['gene'].n_unique()} genes | "
              f"high-PIP(>=0.9) credible variants: {cred.height}")
    else:
        print("no fine-mapping results produced")


if __name__ == "__main__":
    main()
