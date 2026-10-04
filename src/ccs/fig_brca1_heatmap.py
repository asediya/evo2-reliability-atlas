"""BRCA1 saturation-mutagenesis heatmap -- Evo2's signature panel, built from OUR data.
3,893 SGE-measured substitutions across BRCA1: Evo2 predicted deleteriousness as a dense
per-position x per-alternate-base quilt, with the EXPERIMENTAL functional truth (Findlay SGE
LOF/FUNC) as a track directly above on shared coordinates. Agreement AUROC 0.874.

This is the 'texture' panel a reviewer remembers -- and it's honest: predicted quilt vs measured truth."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import figstyle as fs


def main():
    w = pl.read_parquet("data/interim/brca1_windows.parquet")
    s = pl.read_parquet("data/processed/scores_cloud/brca1_evo2_40b_meanll_8192.parquet")  # field-standard 8192, AUROC 0.874
    d = w.join(s, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    pos = np.array([int(v.split("_")[1]) for v in vid])
    alt = np.array([v.split("_")[3] for v in vid])
    score = (-d["evo2_meanll_delta"].to_numpy()).astype(float)      # deleteriousness = -delta
    fclass = np.array(d["func_class"].to_list())

    # order + collapse introns: rank of unique position.
    # BRCA1 is MINUS-strand (genomic pos anti-correlates with residue), so DESCENDING genomic coord
    # = transcript 5'->3' = protein N->C = exon 2 -> exon 23 left to right.
    upos = np.array(sorted(set(pos))[::-1])
    xr = {p: i for i, p in enumerate(upos)}
    N = len(upos)
    ALT = {"A": 0, "C": 1, "G": 2, "T": 3}

    M = np.full((4, N), np.nan)                       # Evo2 quilt
    for p, a, sc in zip(pos, alt, score):
        M[ALT[a], xr[p]] = sc
    # functional truth track: worst class seen at each position (LOF>INT>FUNC)
    rank = {"FUNC": 0, "INT": 1, "LOF": 2}
    truth = np.zeros(N)
    for p, fc in zip(pos, fclass):
        truth[xr[p]] = max(truth[xr[p]], rank[fc])
    # Evo2 column deleteriousness (mean over alts) as a matching track
    coltrack = np.nanmean(M, axis=0)

    # color: diverging, centered on the benign (FUNC) median
    benign_med = np.nanmedian(score[fclass == "FUNC"])
    lo, hi = np.nanpercentile(score, [3, 97])
    norm = TwoSlopeNorm(vmin=lo, vcenter=benign_med, vmax=hi)

    fig = plt.figure(figsize=(9.2, 3.15))
    gs = GridSpec(3, 1, height_ratios=[0.5, 0.5, 2.4], hspace=0.18,
                  left=0.075, right=0.9, top=0.86, bottom=0.16)

    # --- experimental truth strip ---
    axT = fig.add_subplot(gs[0]);  truthcm = ListedColormap(["#2E9E5B", "#E6B33A", "#C0392B"])
    axT.imshow(truth[None, :], aspect="auto", cmap=truthcm, vmin=0, vmax=2, interpolation="nearest")
    axT.set_yticks([0]); axT.set_yticklabels(["SGE\ntruth"], fontsize=6.8); axT.set_xticks([])
    for sp in axT.spines.values(): sp.set_visible(False)
    axT.tick_params(length=0)

    # --- Evo2 column-mean track (line over the quilt) ---
    axL = fig.add_subplot(gs[1]); axL.plot(np.arange(N), coltrack, color=fs.EVO2, lw=0.6)
    axL.fill_between(np.arange(N), np.nanmin(coltrack), coltrack, color=fs.EVO2, alpha=0.18, lw=0)
    axL.set_xlim(0, N); axL.set_xticks([]); axL.set_yticks([])
    axL.set_ylabel("Evo2\nΣ", fontsize=6.8, rotation=0, ha="right", va="center")
    for sp in axL.spines.values(): sp.set_visible(False)

    # --- the saturation quilt ---
    axH = fig.add_subplot(gs[2])
    im = axH.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    axH.set_yticks(range(4)); axH.set_yticklabels(list("ACGT"), fontsize=8)
    axH.set_ylabel("alternate base", fontsize=8)
    axH.set_xlabel("BRCA1 substitution position  (5′ → 3′, introns collapsed)", fontsize=8)
    axH.set_xticks([0, N - 1]); axH.set_xticklabels(["exon 2", "exon 23"], fontsize=7)
    for sp in axH.spines.values(): sp.set_edgecolor("#BBBBBB")

    cax = fig.add_axes([0.915, 0.16, 0.014, 0.42])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Evo2 predicted\ndeleteriousness", fontsize=6.8)
    cb.ax.tick_params(labelsize=6, length=2)
    # truth legend
    for i, (lab, c) in enumerate([("loss-of-fn", "#C0392B"), ("intermediate", "#E6B33A"), ("functional", "#2E9E5B")]):
        fig.text(0.915, 0.80 - i * 0.045, "■", color=c, fontsize=9, va="center")
        fig.text(0.935, 0.80 - i * 0.045, lab, fontsize=6.3, va="center", color=fs.INK)

    fig.suptitle("Evo 2 reconstructs the BRCA1 functional map it never saw", x=0.075, y=0.965,
                 ha="left", fontsize=11, fontweight="bold")
    fig.text(0.075, 0.905, f"3,893 saturation-genome-editing substitutions  ·  zero-shot  ·  "
             f"predicted vs measured AUROC 0.874", fontsize=7.2, color=fs.MUTED, style="italic", ha="left")
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/BRCA1_heatmap.png", bbox_inches="tight")
    print(f"wrote reports/figures/BRCA1_heatmap.png  ({N} positions x 4 bases, {len(vid)} variants)")


if __name__ == "__main__":
    main()
