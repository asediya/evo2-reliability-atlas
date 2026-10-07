"""Idea 1 result: does Evo2-40B deleteriousness track PURIFYING SELECTION? On the cattle ARS-UCD1.2
frequency panel (28k variants stratified across the site-frequency spectrum), test whether predicted
deleteriousness RISES as allele frequency FALLS -- the genome-scale, label-free validation that
escapes the 800-label OMIA ceiling. Runs after selection scoring; writes to the JARVIS dashboard.
  python src/ccs/build_selection_analysis.py
"""
import os, time
import numpy as np
import polars as pl
from scipy.stats import spearmanr

EV = "logs/status/events.log"; ST = "logs/status/selection.status"
ORDER = ["singleton", "rare", "low", "common", "major"]


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] selection: {m}\n")
def st(m):
    with open(ST, "w") as f: f.write(m + "\n")


def main():
    sc = "data/processed/scores/selection_evo2_40b.parquet"
    if not os.path.exists(sc):
        st("WAIT | selection scores not written yet"); return
    cand = pl.read_parquet("data/interim/selection_candidates.parquet").select(["variant_id", "maf", "freq_bin"])
    s = pl.read_parquet(sc).select(["variant_id", pl.col("evo2_40b_neg").alias("del")])
    d = cand.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 100:
        st(f"WAIT | only {d.height} scored"); return

    agg = (d.group_by("freq_bin").agg(pl.len().alias("n"), pl.col("del").mean().alias("mean_del"),
                                      pl.col("maf").mean().alias("mean_maf"))
           .sort(pl.col("freq_bin").replace_strict({b: i for i, b in enumerate(ORDER)}, default=9)))
    rho, p = spearmanr(d["maf"].to_numpy(), d["del"].to_numpy())   # expect NEGATIVE (rare -> deleterious)
    means = [r["mean_del"] for r in agg.rows(named=True)]
    monotone = all(means[i] >= means[i+1] for i in range(len(means)-1))   # singleton highest -> major lowest

    lines = ["# Idea 1: Evo2-40B deleteriousness vs purifying selection (cattle, 28k variants)", "",
             "| freq bin | n | mean MAF | mean Evo2 deleteriousness |", "|---|---|---|---|"]
    for r in agg.rows(named=True):
        lines.append(f"| {({'singleton': 'ultra-rare'}).get(r['freq_bin'], r['freq_bin'])} | {r['n']} | {r['mean_maf']:.4f} | {r['mean_del']:.3f} |")
    verdict = "PASS" if (rho < 0 and p < 1e-3) else "WEAK/FAIL"
    lines += ["", f"Spearman(MAF, deleteriousness) = {rho:+.3f} (p={p:.1e}); expected NEGATIVE.",
              f"Monotone rise toward rare: {monotone}.",
              f"**{verdict}** — selection-spectrum validation {'holds: Evo2 captures deleteriousness genome-wide, label-free' if verdict=='PASS' else 'is weak — investigate'}."]
    open("logs/selection_spectrum.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    st(f"DONE | selection-spectrum {verdict}: Spearman {rho:+.3f} (rare=more deleterious); logs/selection_spectrum.md")
    ev(f"DONE: selection-spectrum {verdict} - Spearman(MAF,del)={rho:+.3f} p={p:.1e}, monotone={monotone}")


if __name__ == "__main__":
    main()
