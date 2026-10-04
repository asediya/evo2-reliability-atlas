"""Figure 3 two-hero dichotomy (research-backed centrepiece): LEFT = large p53 fold with red hotspots
(STRUCTURE the model reads, 0.94); RIGHT = a molecular+schematic 3-D enhancer-promoter chromatin loop with
the eQTL and the model's tiny receptive window (MECHANISM it can't see, 0.49). Big, legible, message-first."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
import fig3_style as S


def _mol(name, zoom):
    return OffsetImage(mpimg.imread(f"assets/mol/{name}.png"), zoom=zoom)


def left_hero(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97, boxstyle="round,pad=0.008,rounding_size=0.03", fc="#EAF2F8", ec="none", zorder=0))
    ax.text(0.5, 0.94, "STRUCTURE", ha="center", fontsize=15, fontweight="bold", color=S.CODING)
    ax.text(0.5, 0.885, "a CODING variant reshapes a protein fold", ha="center", fontsize=8, color=S.INK, style="italic")
    ax.add_artist(AnnotationBbox(_mol("p53_hero_crop", 0.245), (0.5, 0.525), frameon=False, zorder=4))
    ax.annotate("pathogenic\nhotspot residue", xy=(0.585, 0.50), xytext=(0.80, 0.72), fontsize=6.6, color=S.FAIL,
                ha="center", fontweight="bold", arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=0.9))
    ax.text(0.5, 0.15, "Evo 2 reads it", ha="center", fontsize=10.5, color=S.CODING, fontweight="bold")
    ax.text(0.5, 0.065, "AUROC 0.94", ha="center", fontsize=19, color=S.CODING, fontweight="bold")
    ax.text(0.5, 0.022, "p53 DNA-binding domain · PDB 2OCJ · hotspots R175/R248/R273", ha="center", fontsize=4.6, color=S.MUTED, style="italic")


def right_hero(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.015), 0.97, 0.97, boxstyle="round,pad=0.008,rounding_size=0.03", fc="#F1F1F1", ec="none", zorder=0))
    ax.text(0.5, 0.94, "MECHANISM", ha="center", fontsize=15, fontweight="bold", color=S._shade(S.EQTL, -0.4))
    ax.text(0.5, 0.885, "a REGULATORY variant acts through 3-D chromatin looping", ha="center", fontsize=8, color=S.INK, style="italic")

    # ---- the 3-D chromatin loop (upper) ----
    cx, cy, rx, ry = 0.5, 0.635, 0.25, 0.145
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color="#B0A288", lw=2.4, zorder=1)   # looped DNA
    for a in [0.9, 1.9, 4.4, 5.4]:                                                            # nucleosome beads on the loop
        ax.add_artist(AnnotationBbox(_mol("nucleosome_crop", 0.03), (cx + rx * np.cos(a), cy + ry * np.sin(a)), frameon=False, zorder=3))
    ax.add_artist(AnnotationBbox(_mol("cohesin_crop", 0.058), (cx, cy - ry - 0.01), frameon=False, zorder=4))     # loop anchor
    ax.text(cx + 0.155, cy - ry - 0.01, "cohesin / CTCF\nloop anchor", ha="left", va="center", fontsize=5.4, color=S.MUTED, style="italic")
    ax.add_artist(AnnotationBbox(_mol("tf_grip_crop", 0.034), (cx - rx - 0.005, cy + 0.02), frameon=False, zorder=4))  # TF at enhancer
    ax.scatter([cx - rx - 0.005], [cy - 0.045], s=44, color=S.FAIL, zorder=6, edgecolors="white", lw=1)           # eQTL on the enhancer
    ax.text(cx - 0.155, cy + 0.09, "enhancer +\nTF grip", ha="center", fontsize=5.4, color=S.MUTED, style="italic")
    ax.text(cx, cy + 0.005, "in 3-D the enhancer\ntouches the promoter", ha="center", va="center", fontsize=6.0, color=S.INK, fontweight="bold", linespacing=1.1)

    # ---- linear genome ruler (the sequence the model actually reads) ----
    gy = 0.33
    ax.plot([0.10, 0.90], [gy, gy], color="#8C7E63", lw=2.4, zorder=2, solid_capstyle="round")
    ax.text(0.13, gy - 0.045, "promoter", ha="center", fontsize=6.0, color=S.INK, fontweight="bold")
    ax.scatter([0.85], [gy], s=44, color=S.FAIL, zorder=5, edgecolors="white", lw=0.8)
    ax.text(0.85, gy - 0.045, "causal eQTL\n(enhancer)", ha="center", fontsize=6.0, color=S.FAIL, fontweight="bold")
    ax.annotate("", xy=(0.85, gy + 0.035), xytext=(0.13, gy + 0.035), arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=0.7))
    ax.text(0.49, gy + 0.052, "~100 kb – 1 Mb apart in linear sequence", ha="center", fontsize=5.8, color=S.MUTED, style="italic")
    ax.plot([0.11, 0.175], [gy, gy], color=S.CODING, lw=6.0, zorder=6, solid_capstyle="butt")                     # model window
    ax.text(0.185, gy - 0.09, "Evo 2 reads only ~1–8 kb of local sequence — never the 3-D contact", ha="left", fontsize=5.6, color=S.CODING, fontweight="bold")

    ax.text(0.5, 0.15, "Evo 2 is blind", ha="center", fontsize=10.5, color=S.FAIL, fontweight="bold")
    ax.text(0.5, 0.065, "AUROC 0.49", ha="center", fontsize=19, color=S._shade(S.EQTL, -0.4), fontweight="bold")
    ax.text(0.5, 0.022, "chromatin / cohesin / TF structures illustrative (PDB 1KX5, 6WG3, 1GLU) · distances order-of-magnitude, cited from Enformer-class models", ha="center", fontsize=4.2, color=S.MUTED, style="italic")


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6))
    left_hero(a); right_hero(b)
    fig.savefig("reports/figures/_fig3_heroes.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_heroes.png")


if __name__ == "__main__":
    main()
