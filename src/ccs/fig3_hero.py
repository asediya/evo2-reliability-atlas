"""Figure 3 Panel A — DIVERGENCE CANYON. The dissociation as one physical mark: coding is the ascending
upper rim, causal cis-eQTL is a floor pinned to the chance bedrock, and the tinted canyon between them IS
the coding-minus-eQTL competence gap, widening across the 40x parameter ladder. All numbers = fig3_data.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import fig3_style as S

MODELS = ["1B", "7B", "40B"]


def build(ax):
    D = json.load(open("reports/fig3_data.json"))["A_scale_ladder"]
    xs = np.log10([1, 7, 40])
    cod = [D["coding"][m] for m in MODELS]
    eq = {e["model"]: e for e in D["eqtl"]}
    eqa = [eq[m]["auroc"] for m in MODELS]; eqlo = [eq[m]["lo"] for m in MODELS]; eqhi = [eq[m]["hi"] for m in MODELS]
    x0, x1 = xs[0] - 0.30, xs[-1] + 0.50
    ax.set_xlim(x0, x1); ax.set_ylim(0.455, 0.98)

    S.bedrock(ax, 0.5, x0, x1, label=False)                                     # shared chance datum
    ax.text(x0 + 0.02, 0.507, "chance bedrock 0.5", ha="left", va="bottom", fontsize=5.6, color=S.BEDROCK,
            fontweight="bold", style="italic")

    # ---- the canyon: filled gap between the rising coding rim and the flat eQTL floor ----
    xf = np.linspace(xs[0], xs[-1], 240)
    codf = np.interp(xf, xs, cod); eqf = np.interp(xf, xs, eqa)
    ax.fill_between(xf, eqf, codf, color=S.CANYON, zorder=1, edgecolor="none")
    for i in range(7):                                                          # deepen toward the wide 40B end
        ax.fill_between(xf, eqf, codf, where=(xf >= xs[0] + (xs[-1] - xs[0]) * i / 7),
                        color=S.CANYON, alpha=0.07, zorder=1, edgecolor="none")

    # ---- eQTL floor: bootstrap-CI haze + line + solid markers (rides the bedrock) ----
    ax.fill_between(xs, eqlo, eqhi, color=S.EQTL, alpha=0.30, zorder=2, edgecolor="none")
    ax.plot(xs, eqa, color=S._shade(S.EQTL, -0.15), lw=1.8, zorder=3, solid_capstyle="round")
    ax.scatter(xs, eqa, s=26, color=S._shade(S.EQTL, -0.15), edgecolors="white", lw=0.8, zorder=4)

    # ---- coding rim: rising line + HOLLOW markers (point estimates, no CI — honest) ----
    ax.plot(xs, cod, color=S.CODING, lw=2.0, zorder=3, solid_capstyle="round")
    ax.scatter(xs, cod, s=34, facecolors="white", edgecolors=S.CODING, lw=1.7, zorder=4)

    # ---- Δ gap ticks inside the canyon (the gap the reader should feel) ----
    for x, c, e in zip(xs, cod, eqa):
        ax.annotate("", xy=(x, c - 0.008), xytext=(x, e + 0.008),
                    arrowprops=dict(arrowstyle="<->", color="#B0A288", lw=0.7), zorder=3)
    ax.text(np.mean(xs), 0.71, "competence gap", ha="center", va="center", fontsize=8.4, color="#B29B71",
            style="italic", fontweight="bold")
    ax.text(np.mean(xs), 0.672, "scaling 40× never closes it  (0.41 → 0.44)", ha="center", va="center",
            fontsize=5.8, color="#A8946F", style="italic")

    # ---- endpoint value labels ----
    ax.text(xs[-1] + 0.06, cod[-1], f"coding {cod[-1]:.3f}", ha="left", va="center", fontsize=6.8, color=S.CODING, fontweight="bold")
    ax.text(xs[-1] + 0.06, eqa[-1] + 0.004, f"eQTL {eqa[-1]:.3f}", ha="left", va="bottom", fontsize=6.8,
            color=S._shade(S.EQTL, -0.4), fontweight="bold")

    # ---- silhouettes: pig on the eQTL floor, cattle on the coding rim ----
    try:
        ax.add_artist(AnnotationBbox(OffsetImage(S.load_silhouette("pig", S._shade(S.EQTL, -0.25), longest=90), zoom=0.075),
                                     (np.mean(xs[1:]) + 0.05, eqa[-1] - 0.028), frameon=False, zorder=5))
        ax.add_artist(AnnotationBbox(OffsetImage(S.load_silhouette("cattle", S.CODING, longest=90), zoom=0.07),
                                     (xs[0] + 0.02, cod[0] + 0.033), frameon=False, zorder=5))
    except Exception:
        pass

    # ---- honesty notes (kept apart from the crowded bedrock zone) ----
    ax.text(x0 + 0.02, cod[0] + 0.052, "coding rim = point estimates\n(n=9 species; no stored CI)",
            fontsize=4.7, color=S.CODING, va="bottom", linespacing=1.1)
    ax.text(x1 - 0.02, 0.468, "shaded floor = variant-bootstrap 95% CI (straddles chance)",
            fontsize=4.6, color=S._shade(S.EQTL, -0.4), va="center", ha="right", style="italic")

    # ---- axes ----
    ax.set_xticks(xs); ax.set_xticklabels(["Evo 2-1B", "Evo 2-7B", "Evo 2-40B\n(40× params →)"], fontsize=6.6)
    ax.set_ylabel("AUROC", fontsize=8, color=S.CAP); ax.tick_params(labelsize=6.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(4.7, 3.8))
    build(ax)
    S.finding_title(ax, "A", "Scaling 40× buys coding skill —\nand nothing regulatory")
    fig.savefig("reports/figures/_fig3_A.png", bbox_inches="tight", dpi=300)
    print("wrote reports/figures/_fig3_A.png")


if __name__ == "__main__":
    main()
