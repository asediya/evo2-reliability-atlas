"""Figure 3 Panel A (hero) — THE CLIFF IS ONE RUNG TOO FAR. Ordered consequence rungs
MISSENSE(0.824)→CODING(0.925)→non-coding-Mendelian(0.879)→causal-eQTL(0.488) as a geological terrace
cross-section: three high terraces (a plateau), then a sheer cliff plunging through the chance sea-level
only at the quantitative-regulatory terminus. A ghost line marks where a reader naively predicts failure
(the coding/non-coding boundary) — the real cliff sits one terrace to its right. Numbers verified live."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon
import fig3_style as S

SEA = "#3A6B8C"


def _ladder():
    tm = pl.read_parquet("reports/type_matched_atlas.parquet")
    g = {r["stratum"]: r["auroc"] for r in tm.iter_rows(named=True) if str(r["readout"]).startswith("8192")}
    eq = json.load(open("reports/fig3_data.json"))["C_null"]
    return [("MISSENSE", g["MISSENSE-only (type-matched)"]), ("CODING", g["CODING-only"]),
            ("non-coding\nMendelian", g["non-coding only"]), ("causal\neQTL", eq["global_auroc"])]


def _mol(name, zoom):
    return OffsetImage(mpimg.imread(f"assets/mol/{name}.png"), zoom=zoom)


def build(ax, mols=True):
    L = _ladder(); names = [r[0] for r in L]; ys = [r[1] for r in L]
    xs = [0.55, 1.5, 2.45, 3.75]; w = 0.42; base = 0.44
    ax.set_xlim(0.0, 4.42); ax.set_ylim(base, 1.02)

    # ---- terraced landmass ----
    surf = []
    for x, y in zip(xs, ys):
        surf += [(x - w, y), (x + w, y)]
    ax.add_patch(Polygon(surf + [(xs[-1] + w, base), (xs[0] - w, base)], closed=True,
                         facecolor="#DACFBA", edgecolor="none", zorder=2))
    for i in range(len(xs) - 1):                                            # terrace tops + connecting faces
        ax.plot([xs[i] - w, xs[i] + w], [ys[i], ys[i]], color="#8C7E63", lw=1.8, zorder=3, solid_capstyle="round")
        ax.plot([xs[i] + w, xs[i + 1] - w], [ys[i], ys[i + 1]], color="#8C7E63", lw=1.3, zorder=3)
    ax.plot([xs[-1] - w, xs[-1] + w], [ys[-1], ys[-1]], color="#8C7E63", lw=1.8, zorder=3)

    # ---- chance sea-level + submerged underworld ----
    ax.axhspan(base, 0.5, color="#A9C4D8", alpha=0.40, zorder=4)             # water tints the submerged eQTL terrace
    ax.axhline(0.5, color=SEA, lw=1.6, ls=(0, (5, 2)), zorder=5)
    ax.text(0.06, 0.505, "chance sea-level 0.5", ha="left", va="bottom", fontsize=5.8, color=SEA, fontweight="bold", style="italic")

    # ---- AUROC labels + rung names ----
    for x, y, nm in zip(xs, ys, names):
        ax.text(x, y + 0.013, f"{y:.3f}", ha="center", va="bottom", fontsize=7.0, color=S.INK, fontweight="bold", zorder=6)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=6.6)

    # ---- 'established plateau (Fig 2)' bracket vs 'this work' ----
    ax.annotate("", xy=(xs[0] - w, 0.965), xytext=(xs[2] + w, 0.965), arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.8))
    ax.text((xs[0] + xs[2]) / 2, 0.97, "high plateau — all disease-variant classes stay detectable  (coding = Fig 2 reference)",
            ha="center", va="bottom", fontsize=5.6, color=S.MUTED, style="italic")

    # ---- the ghost line: where intuition predicts the cliff ----
    gx = (xs[1] + xs[2]) / 2
    ax.axvline(gx, color="#B24E4E", lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=3)
    ax.annotate("you'd predict the\ncliff HERE\n(coding→non-coding)", xy=(gx, 0.70), xytext=(gx - 0.02, 0.64),
                fontsize=5.4, color="#B24E4E", ha="center", va="center", style="italic")

    # ---- the real cliff face + terminal drop ----
    cx = (xs[2] + w + xs[3] - w) / 2
    ax.annotate("", xy=(cx, ys[3] + 0.01), xytext=(cx, ys[2] - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=S.FAIL, lw=2.4, mutation_scale=15), zorder=6)
    ax.text(cx + 0.04, (ys[2] + ys[3]) / 2 + 0.03, "−0.391", ha="left", va="center", fontsize=8.0, color=S.FAIL, fontweight="bold")
    ax.text(cx + 0.04, (ys[2] + ys[3]) / 2 - 0.01, "~4× any coding step\n— the real cliff,\none rung further", ha="left", va="center",
            fontsize=5.4, color=S.FAIL, style="italic", linespacing=1.15)

    # ---- molecular anchors: protein on the plateau, nucleosome submerged at the eQTL terrace ----
    if mols:
        try:
            ax.add_artist(AnnotationBbox(_mol("protein_p53_crop", 0.052), (xs[1], 0.905), frameon=False, zorder=6))
            ax.add_artist(AnnotationBbox(_mol("nucleosome_crop", 0.04), (xs[3] + 0.02, 0.478), frameon=False, zorder=6))
        except Exception:
            pass
    ax.text(xs[3] - 0.02, 0.548, "regulatory =\nchromatin / 3-D", ha="center", va="bottom", fontsize=5.0,
            color=S._shade(S.EQTL, -0.35), style="italic", linespacing=1.05)

    ax.set_ylabel("AUROC", fontsize=8, color=S.CAP); ax.tick_params(labelsize=6.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    build(ax)
    S.finding_title(ax, "A", "The cliff is one rung too far — Evo 2 fails only\nat the quantitative-regulatory frontier")
    fig.savefig("reports/figures/_fig3_A_cliff.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_A_cliff.png")


if __name__ == "__main__":
    main()
