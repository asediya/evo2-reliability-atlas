"""Figure 3 Panel D — ROBUSTNESS. Every readout x window lands on the chance bedrock; coding floats far above.
Reviewer-proofing that the null is not a scoring artifact. Honest ghost for the untested 16384 window."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import fig3_style as S

SHAPE = {"mean-LL": "^", "single-pos": "o"}


def panel_D(ax):
    R = json.load(open("reports/fig3_data.json"))["D_robustness"]
    cells = R["cells"]; cod = R["coding_ref"]
    ax.set_xlim(10.4, 14.7); ax.set_ylim(0.44, 0.99)
    ax.axhspan(cod - 0.018, cod + 0.018, color=S.CODING, alpha=0.13, zorder=0)          # coding reference band
    ax.text(14.6, cod, f"coding (any recipe)\n~{cod:.2f}", ha="right", va="center", fontsize=5.5, color=S.CODING, fontweight="bold")
    S.bedrock(ax, 0.5, 10.4, 14.7, label=False)
    ax.text(10.5, 0.507, "chance bedrock 0.5", ha="left", va="bottom", fontsize=5.4, color=S.BEDROCK, fontweight="bold", style="italic")
    for cell in cells:
        x = np.log2(cell["window"])
        ax.scatter([x], [cell["auroc"]], s=40, marker=SHAPE[cell["readout"]], color=S._shade(S.EQTL, -0.22),
                   edgecolors="white", lw=0.7, zorder=4)
    ax.scatter([14], [0.5], s=40, marker="o", facecolors="none", edgecolors="#CBCBCB", lw=1.1, zorder=3)  # ghost untested
    ax.text(14, 0.47, "16384\nnot scored\n(compute retired)", ha="center", va="top", fontsize=4.3, color="#B0B0B0", style="italic")
    ax.text(11.9, 0.545, "every readout × window lands on the bedrock", ha="center", fontsize=5.2, color=S.MUTED, style="italic")
    # readout shape key
    ax.scatter([10.62], [0.62], s=30, marker="^", color=S._shade(S.EQTL, -0.22), zorder=4); ax.text(10.75, 0.62, "mean-LL", fontsize=5.2, va="center", color=S.MUTED)
    ax.scatter([10.62], [0.58], s=30, marker="o", color=S._shade(S.EQTL, -0.22), zorder=4); ax.text(10.75, 0.58, "single-pos", fontsize=5.2, va="center", color=S.MUTED)
    ax.set_xticks([11, 12, 13, 14]); ax.set_xticklabels(["2048", "4096", "8192", "16384"], fontsize=6.0)
    ax.set_xlabel("context window (bp)", fontsize=7.2, color=S.CAP)
    ax.set_ylabel("AUROC", fontsize=8, color=S.CAP); ax.tick_params(labelsize=6.2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(3.6, 3.6)); panel_D(ax)
    S.finding_title(ax, "D", "Not an artifact: flat across\nevery readout and window")
    fig.savefig("reports/figures/_fig3_D.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_D.png")


if __name__ == "__main__":
    main()
