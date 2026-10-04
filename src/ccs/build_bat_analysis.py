"""BAT One-Health result — does Evo2-40B deleteriousness track purifying selection in a wild
zoonotic-host bat (Myotis lucifugus)? SFS-lite on the real 10-sample segregating SNVs: bin by minor
allele count, test Spearman(MAF, deleteriousness) < 0 (rarer = more deleterious). Coarse (10 samples)
but a genuine cross-clade generalisation check for the cattle Idea-1 finding, ~65 My away.
Runs after bat Evo2 scoring. Writes logs/bat_selection.md + dashboard.
  python src/ccs/build_bat_analysis.py
"""
import os, sys, time
import numpy as np
import polars as pl
from scipy.stats import spearmanr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SC = "data/processed/scores/bat_evo2_40b.parquet"
CAND = "data/interim/bat_candidates.parquet"
EV = "logs/status/events.log"; ST = "logs/status/bat_panel.status"; MD = "logs/bat_selection.md"
ORDER = ["singleton", "rare", "low", "common"]


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] bat: {m}\n")
def st(m):
    with open(ST, "w", encoding="utf-8") as f: f.write(m + "\n")


def main():
    if not os.path.exists(SC):
        st("WAIT | bat Evo2 scores not written yet"); return
    cand = pl.read_parquet(CAND).select(["variant_id", "maf", "mac", "freq_bin"])
    s = pl.read_parquet(SC).select(["variant_id", pl.col("evo2_40b_neg").alias("del")])
    d = cand.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 200:
        st(f"WAIT | only {d.height} bat SNVs scored"); return

    agg = (d.group_by("freq_bin").agg(pl.len().alias("n"), pl.col("del").mean().alias("mean_del"),
                                      pl.col("maf").mean().alias("mean_maf"))
           .sort(pl.col("freq_bin").replace_strict({b: i for i, b in enumerate(ORDER)}, default=9)))
    rho, p = spearmanr(d["maf"].to_numpy(), d["del"].to_numpy())
    means = [r["mean_del"] for r in agg.rows(named=True)]
    monotone = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    lines = ["# BAT One-Health: Evo2-40B deleteriousness vs purifying selection (Myotis lucifugus, 10 samples)",
             "", "SFS-lite (coarse allele counts, AN<=20) — PILOT-grade cross-clade check of the cattle result.", "",
             "| allele-count bin | n | mean MAF | mean Evo2 deleteriousness |", "|---|---|---|---|"]
    for r in agg.rows(named=True):
        lines.append(f"| {r['freq_bin']} | {r['n']} | {r['mean_maf']:.3f} | {r['mean_del']:.3f} |")
    verdict = "PASS" if (rho < 0 and p < 1e-3) else "WEAK/NULL"
    lines += ["", f"Spearman(MAF, deleteriousness) = {rho:+.3f} (p={p:.1e}); expected NEGATIVE.",
              f"Monotone singleton->common: {monotone}.",
              f"**{verdict}** — " + ("Evo2's selection-tracking GENERALISES to a wild zoonotic-host bat ~65 My "
              "from the farm animals (pilot-grade, 10 samples)." if verdict == "PASS" else
              "no clear selection signal at this (coarse, 10-sample) resolution — report honestly as an "
              "underpowered pilot, not evidence of absence.")]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    st(f"DONE | bat SFS-lite {verdict}: Spearman {rho:+.3f} (rare=more deleterious); {MD}")
    ev(f"DONE: bat selection {verdict} - Spearman(MAF,del)={rho:+.3f} p={p:.1e}, monotone={monotone}")


if __name__ == "__main__":
    main()
