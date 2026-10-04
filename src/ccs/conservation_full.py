"""Full-panel phyloP conservation for ALL ~44.5M cattle variants, parallelized to saturate every core.
Cattle phyloP bigWigs are bosTau9 = ARS-UCD1.2 (same build as the panel). Each chromosome is split into
NCHUNK slices so we run ~75 workers across the 52 cores. Output: [variant_id, phylop] for the whole panel
-> the genome-scale conservation-vs-selection baseline that the GPU-limited Evo2 (28k) can't match.
  python src/ccs/conservation_full.py
"""
import glob, os, re, time
from multiprocessing import Pool
import polars as pl
import pyBigWig

AF = "data/interim/geno_pgen"
BW = "data/raw/conservation/cattle/phyloP"
OUT = "data/interim/full_phylop.parquet"
NCHUNK = 3


def one(args):
    afreq, k, n = args
    chrom = re.search(r"Chr(\w+)\.afreq", os.path.basename(afreq)).group(1)
    d = pl.read_csv(afreq, separator="\t", schema_overrides={"#CHROM": pl.String, "ID": pl.String})
    ids = d["ID"].to_list()[k::n]
    bwf = f"{BW}/phyloP_bosTau9_chr{chrom}.bw"
    if not os.path.exists(bwf):
        return [(i, float("nan")) for i in ids]
    bw = pyBigWig.open(bwf)
    chroms = bw.chroms()
    name = f"chr{chrom}" if f"chr{chrom}" in chroms else (chrom if chrom in chroms else list(chroms)[0])
    out = []
    for i in ids:
        p = int(i.split("_")[1])
        try:
            v = bw.values(name, p - 1, p)[0]
        except Exception:
            v = None
        out.append((i, float(v) if v is not None else float("nan")))
    bw.close()
    return out


if __name__ == "__main__":
    t0 = time.time()
    files = sorted(glob.glob(f"{AF}/*.afreq"))
    tasks = [(f, k, NCHUNK) for f in files for k in range(NCHUNK)]
    print(f"{len(files)} chroms x {NCHUNK} = {len(tasks)} parallel tasks", flush=True)
    with Pool(processes=min(len(tasks), os.cpu_count())) as pool:
        results = pool.map(one, tasks)
    rows = [r for sub in results for r in sub]
    pl.DataFrame(rows, schema=["variant_id", "phylop"], orient="row").write_parquet(OUT)
    ok = sum(1 for _, v in rows if v == v)
    print(f"wrote {OUT}: {len(rows)} variants, non-nan={ok} in {time.time()-t0:.0f}s", flush=True)
