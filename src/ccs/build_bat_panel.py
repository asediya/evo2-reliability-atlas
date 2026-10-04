"""BAT One-Health arm — build the polymorphism/SFS-lite panel from the 10-sample Myoluc2.0 VCF.

With only 10 bats (coarse allele counts, AN<=20) we do NOT chase fine frequencies. Instead the SFS-lite
test (analog of the cattle Idea-1 selection-spectrum, but coarse): among REAL segregating bat SNVs,
bin by minor-allele COUNT and ask whether Evo2 deleteriousness FALLS as the allele becomes more common
(singletons = recently arisen / less filtered = more deleterious; common = tolerated). A negative
Spearman = Evo2 tracks purifying selection in a wild zoonotic-host bat, ~65 My from the farm animals.

CPU-only. Reads the joint-call VCF (data/interim/bat/bat.vcf.gz) via the bat-map container's bcftools,
samples SNVs per allele-count bin, then reuses extract_windows_local.py for ref/alt windows + ref-match
against Myoluc2.0. Waits gracefully if the VCF isn't ready yet.
  python src/ccs/build_bat_panel.py --per-bin 6000
"""
import os
import argparse, os, subprocess, sys, time
import polars as pl

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VCF = "data/interim/bat/bat.vcf.gz"
TSV = "data/interim/bat/bat_af.tsv"
CAND = "data/interim/bat_candidates.parquet"
WIN = "data/interim/bat_windows.parquet"
GENOME = "data/raw/genomes/bat/bat.fa"
EV = "logs/status/events.log"; ST = "logs/status/bat_panel.status"
# coarse bins for AN<=20: singleton, rare(2-3), low(4-6), common(>=7 up to major)
BINS = [("singleton", 1, 2), ("rare", 2, 4), ("low", 4, 7), ("common", 7, 11)]


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] bat-panel: {m}\n")
def st(m):
    with open(ST, "w", encoding="utf-8") as f: f.write(m + "\n")


def export_tsv():
    """Make a [chrom,pos,ref,alt,AC,AN] TSV from the VCF using the bat-map container's bcftools."""
    q = r"%CHROM\t%POS\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\n"
    cmd = ["docker", "run", "--rm", "-v", f"{CCS_ROOT}:/workspace", "-w", "/workspace",
           "bat-map", "bash", "-c",
           f"export PATH=/opt/conda/bin:$PATH; bcftools view -v snps -m2 -M2 {VCF} | "
           f"bcftools query -f '{q}' > {TSV}"]
    subprocess.run(cmd, check=True, timeout=3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(VCF):
        st("WAIT | bat VCF not ready yet (mapping/calling in progress)"); return
    if not os.path.exists(TSV) or os.path.getsize(TSV) == 0:
        st("RUNNING | exporting biallelic-SNV AF table from VCF (bcftools)")
        ev("exporting biallelic SNV AC/AN table from the 10-sample VCF")
        export_tsv()

    d = pl.read_csv(TSV, separator="\t", has_header=False,
                    new_columns=["chrom", "pos", "ref", "alt", "ac", "an"],
                    schema_overrides={"chrom": pl.String, "pos": pl.Int64, "ref": pl.String,
                                      "alt": pl.String, "ac": pl.Int64, "an": pl.Int64})
    d = (d.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
         .filter((pl.col("an") >= 6) & (pl.col("ac") >= 1) & (pl.col("ac") < pl.col("an")))  # real, well-called
         .with_columns([
             pl.min_horizontal("ac", pl.col("an") - pl.col("ac")).alias("mac"),   # minor-allele count
             (pl.col("chrom") + "_" + pl.col("pos").cast(pl.String) + "_" + pl.col("ref") + "_" + pl.col("alt")).alias("variant_id"),
         ])
         .with_columns((pl.col("mac") / pl.col("an")).alias("maf")))
    ev(f"{d.height} biallelic segregating bat SNVs (AN>=6)")

    parts = []
    for name, lo, hi in BINS:
        b = d.filter((pl.col("mac") >= lo) & (pl.col("mac") < hi))
        n = min(a.per_bin, b.height)
        if n > 0:
            parts.append(b.sample(n=n, seed=a.seed).with_columns(pl.lit(name).alias("freq_bin")))
        ev(f"bin {name} (mac {lo}-{hi-1}): {b.height} available -> {n}")
    out = pl.concat(parts)
    out.write_parquet(CAND)
    st(f"RUNNING | {out.height} SNVs sampled across {len(parts)} allele-count bins -> extracting windows")

    # reuse the standard window extractor + ref-match against Myoluc2.0
    r = subprocess.run([sys.executable, "src/ccs/extract_windows_local.py",
                        "--in", CAND, "--genome", GENOME, "--out", WIN],
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()[-500:] if r.stderr else "", file=sys.stderr)
    if os.path.exists(WIN):
        w = pl.read_parquet(WIN); ok = int(w["ref_ok"].sum())
        st(f"DONE | bat panel: {out.height} SNVs, {w.height} windows, ref-match {ok}/{w.height} "
           f"({100*ok/max(1,w.height):.0f}%) -> ready for Evo2 scoring (queued after bench)")
        ev(f"DONE: bat panel built - {w.height} windows, ref-match {100*ok/max(1,w.height):.0f}% (Myoluc2.0)")
    else:
        st("ERROR | window extraction produced no output"); ev("window extraction failed")


if __name__ == "__main__":
    main()
