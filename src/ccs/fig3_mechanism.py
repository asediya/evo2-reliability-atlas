"""Figure 3 MECHANISM CORE (Beat 2 · WHY, molecular) — structure vs mechanism, with molecules carrying the
MEASURED number. Centre = the |Δ mean-LL| the model actually produces: a spike for coding positives,
but for causal cis-eQTLs it barely flinches (1.9e-4 ≈ a harmless variant). Molecules flank each pole.
Real deposited structures, illustrative + attributed; the confound (disease-coding vs pig-common-eQTL) is stated."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import fig3_style as S

RNG = np.random.default_rng(7)


def _mol(name, zoom):
    return OffsetImage(mpimg.imread(f"assets/mol/{name}.png"), zoom=zoom)


def _deltas():
    pv = pl.read_parquet("reports/fig2_pervariant.parquet")
    cp = np.abs(pv.filter(pl.col("label") == 1)["deleteriousness"].to_numpy())
    cb = np.abs(pv.filter(pl.col("label") == 0)["deleteriousness"].to_numpy())
    lab = pl.read_parquet("data/interim/ablation/eqtl_abl_sample.parquet")
    ev = pl.read_parquet("data/processed/scores_cloud/eqtl_abl_8192_ll.parquet")
    d = lab.join(ev, on="variant_id", how="inner")
    eq = np.abs(d.filter(pl.col("label") == 1)["evo2_meanll_delta"].to_numpy())
    return cp, cb, eq


def build(ax):
    cp, cb, eq = _deltas()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- the measured substance: |Δ mean-LL| beeswarm on a log axis (centre) ----
    def _clip(v):
        return np.log10(np.clip(v, 1e-5, 1e-1))
    y0, y1 = -5.0, -1.0                                             # log10 axis window
    def Y(logv):
        return 0.12 + 0.76 * (logv - y0) / (y1 - y0)
    cols = [("coding\npathogenic", cp, S.CODING, 0.30), ("causal\neQTL", eq, S._shade(S.EQTL, -0.15), 0.52),
            ("benign\ncoding", cb, "#B7B7B7", 0.70)]
    for name, v, c, xc in cols:
        lv = _clip(v); samp = RNG.choice(lv, size=min(320, len(lv)), replace=False)
        jx = xc + RNG.uniform(-0.045, 0.045, len(samp))
        ax.scatter(jx, [Y(z) for z in samp], s=3.0, color=c, alpha=0.45, zorder=3, linewidths=0)
        med = np.median(lv)
        ax.plot([xc - 0.055, xc + 0.055], [Y(med)] * 2, color=S._shade(c, -0.35), lw=2.2, zorder=5)
        ax.text(xc, 0.05, name, ha="center", va="top", fontsize=5.8, color=S._shade(c, -0.2), fontweight="bold", linespacing=1.0)
        ax.text(xc + 0.075, Y(med), f"{10 ** med:.1e}", ha="left", va="center", fontsize=5.8, color=S._shade(c, -0.35),
                fontweight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for lg in [-4, -3, -2]:
        ax.plot([0.24, 0.76], [Y(lg)] * 2, color="#EEE", lw=0.6, zorder=0)
        ax.text(0.225, Y(lg), f"1e{lg}", ha="right", va="center", fontsize=4.8, color=S.MUTED)
    ax.text(0.5, 0.95, "the number Evo 2 actually produces:  |Δ mean-LL| per variant", ha="center", fontsize=6.8,
            color=S.INK, fontweight="bold")
    ax.annotate("", xy=(0.52, Y(np.median(_clip(eq))) + 0.03), xytext=(0.30, Y(np.median(_clip(cp))) - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=S.FAIL, lw=1.6, connectionstyle="arc3,rad=0.25"), zorder=6)
    ax.text(0.41, 0.30, "14× smaller —\n= a HARMLESS variant", ha="center", va="center", fontsize=5.8, color=S.FAIL,
            fontweight="bold", linespacing=1.15, zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=S.FAIL, lw=0.8, alpha=0.95))

    # ---- STRUCTURE pole (readable) : protein + number ----
    ax.add_patch(FancyBboxPatch((0.005, 0.55), 0.20, 0.42, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#EAF2F8", ec="none", zorder=0))
    ax.add_artist(AnnotationBbox(_mol("protein_p53_crop", 0.058), (0.105, 0.80), frameon=False, zorder=4))
    ax.text(0.105, 0.965, "Evo 2 SEES  ·  STRUCTURE", ha="center", fontsize=6.2, color=S.CODING, fontweight="bold")
    ax.text(0.105, 0.60, "coding → protein fold\nAUROC 0.94  (Fig 2 ref)", ha="center", fontsize=5.4, color=S.CODING, linespacing=1.15)

    # ---- MECHANISM pole (blind) : non-locality ladder + number ----
    ax.add_patch(FancyBboxPatch((0.795, 0.03), 0.20, 0.94, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#F1F1F1", ec="none", zorder=0))
    ax.text(0.895, 0.965, "Evo 2 BLIND  ·  MECHANISM", ha="center", fontsize=6.2, color=S._shade(S.EQTL, -0.4), fontweight="bold")
    ladder = [("tf_grip_crop", "TF grip"), ("ctcf_crop", "insulator (CTCF)"), ("nucleosome_crop", "chromatin"), ("cohesin_crop", "3-D loop (Mb)")]
    yy = [0.80, 0.60, 0.40, 0.20]
    for (nm, lab), y in zip(ladder, yy):
        ax.add_artist(AnnotationBbox(_mol(nm, 0.045), (0.86, y), frameon=False, zorder=4))
        ax.text(0.995, y, lab, ha="right", va="center", fontsize=5.2, color=S._shade(S.EQTL, -0.35))
    ax.annotate("", xy=(0.9, 0.11), xytext=(0.9, 0.89), arrowprops=dict(arrowstyle="-|>", color=S._shade(S.EQTL, -0.3), lw=1.4), zorder=3)
    ax.text(0.80, 0.5, "escalating\nnon-locality", ha="center", va="center", fontsize=4.8, color=S.MUTED, rotation=90, style="italic")
    ax.text(0.895, 0.005, "AUROC 0.49 (chance)", ha="center", va="bottom", fontsize=5.4, color=S.FAIL, fontweight="bold")

    ax.text(0.5, -0.02, "confound stated: coding = cross-species DISEASE variants (Fig 2); eQTL = pig COMMON variants — the gap is consequence/regime, "
            "not purely structure-vs-mechanism · structures illustrative, real PDB, not the assayed variants · 'blind' = zero-shot likelihood readout.",
            ha="center", va="top", fontsize=4.4, color=S.MUTED, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    build(ax)
    S.finding_title(ax, "2", "WHY, molecularly — the model registers a fold it can read,\nbut not the regulatory machinery that a causal eQTL acts through")
    fig.savefig("reports/figures/_fig3_mechanism.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_mechanism.png")


if __name__ == "__main__":
    main()
