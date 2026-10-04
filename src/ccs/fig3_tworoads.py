"""Figure 3 molecular centerpiece — TWO ROADS FROM A VARIANT. The molecular WHY of the blind spot:
a coding variant reshapes a protein STRUCTURE (local, sequence-determined) that Evo 2 reads; a regulatory
eQTL acts through CHROMATIN / 3-D looping with no local signature, so Evo 2 is blind. Real deposited
structures (p53 core 2OCJ, nucleosome 1KX5) as illustrative mechanism anchors — not computed here."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import fig3_style as S


def _mol(name, zoom):
    return OffsetImage(mpimg.imread(f"assets/mol/{name}.png"), zoom=zoom)


def _dna(ax, x0, x1, y, amp=0.075, turns=3.0, vfrac=0.62):
    t = np.linspace(0, turns * 2 * np.pi, 400); x = np.linspace(x0, x1, 400)
    y1 = y + amp * np.sin(t); y2 = y + amp * np.sin(t + np.pi)
    for i in range(0, 400, 9):                                    # base-pair rungs behind
        ax.plot([x[i], x[i]], [y1[i], y2[i]], color="#CDC6BA", lw=1.0, zorder=2)
    ax.plot(x, y1, color=S.CODING, lw=2.6, zorder=3, solid_capstyle="round")
    ax.plot(x, y2, color=S._shade(S.CODING, -0.35), lw=2.6, zorder=3, solid_capstyle="round")
    vi = int(vfrac * 400)
    ax.plot([x[vi], x[vi]], [y1[vi], y2[vi]], color=S.FAIL, lw=3.0, zorder=4)   # the variant base pair
    ax.scatter([x[vi]], [y], s=52, color=S.FAIL, zorder=5, edgecolors="white", lw=1.2)
    return x[vi], y


def build(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.40, 0.525), 0.585, 0.45, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#EAF2F8", ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((0.40, 0.03), 0.585, 0.45, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#F1F1F1", ec="none", zorder=0))

    vx, vy = _dna(ax, 0.03, 0.29, 0.5)
    ax.text(0.16, 0.66, "one DNA variant", ha="center", fontsize=8.5, color=S.INK, fontweight="bold")
    ax.text(0.16, 0.345, "Evo 2 reads ~1–8 kb\nof local sequence", ha="center", fontsize=6.0, color=S.MUTED,
            style="italic", linespacing=1.15)

    ax.add_patch(FancyArrowPatch((vx + 0.005, vy + 0.02), (0.485, 0.74), arrowstyle="-|>", mutation_scale=18,
                                 lw=2.4, color=S.CODING, connectionstyle="arc3,rad=-0.28", zorder=3))
    ax.add_patch(FancyArrowPatch((vx + 0.005, vy - 0.02), (0.485, 0.26), arrowstyle="-|>", mutation_scale=18,
                                 lw=2.4, color=S._shade(S.EQTL, -0.25), connectionstyle="arc3,rad=0.28", zorder=3))

    # ---- CODING road ----
    ax.add_artist(AnnotationBbox(_mol("protein_p53_crop", 0.115), (0.575, 0.75), frameon=False, zorder=4))
    ax.text(0.70, 0.90, "CODING variant", fontsize=9, color=S.CODING, fontweight="bold")
    ax.text(0.70, 0.79, "→ amino-acid change\n→ protein STRUCTURE\n    (local, sequence-determined)",
            fontsize=6.6, color=S.INK, va="center", linespacing=1.3)
    ax.text(0.70, 0.635, "Evo 2 + conservation read it", fontsize=7.0, color=S.CODING, fontweight="bold")
    ax.text(0.968, 0.90, "0.94", fontsize=15, color=S.CODING, fontweight="bold", ha="right", va="top")
    ax.text(0.968, 0.845, "AUROC", fontsize=5.6, color=S.CODING, ha="right", va="top")

    # ---- REGULATORY road ----
    ax.add_artist(AnnotationBbox(_mol("nucleosome_crop", 0.105), (0.575, 0.25), frameon=False, zorder=4))
    ax.text(0.70, 0.44, "REGULATORY eQTL", fontsize=9, color=S._shade(S.EQTL, -0.4), fontweight="bold")
    ax.text(0.70, 0.335, "→ enhancer / TF site\n→ CHROMATIN & 3-D looping\n    (no local structural signature)",
            fontsize=6.6, color=S.INK, va="center", linespacing=1.3)
    ax.text(0.70, 0.175, "Evo 2 is blind", fontsize=7.0, color=S.FAIL, fontweight="bold")
    ax.text(0.968, 0.445, "0.49", fontsize=15, color=S._shade(S.EQTL, -0.4), fontweight="bold", ha="right", va="top")
    ax.text(0.968, 0.393, "AUROC (chance)", fontsize=5.6, color=S.FAIL, ha="right", va="top")

    ax.text(0.40, 0.006, "structures illustrative — p53 core PDB 2OCJ · nucleosome PDB 1KX5 "
            "(real deposited structures, not predicted here)", fontsize=4.7, color=S.MUTED, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    build(ax)
    S.finding_title(ax, "•", "Two roads from a variant — the model can read structure, not regulation")
    fig.savefig("reports/figures/_fig3_tworoads.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_tworoads.png")


if __name__ == "__main__":
    main()
