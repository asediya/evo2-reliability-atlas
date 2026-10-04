"""Figure 3 Panels B (Competence Cliffs) and C (Camouflage Swarm). Chance-bedrock family.
B: each scorer plunges from its coding clifftop to the shared chance sea-level; drop = the collapse; GERP flips.
C: ~600 permutation-null AUROCs as identical ticks with the observed value camouflaged among them = the null is genuine."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import fig3_style as S

ORDER_B = ["Evo2-40B", "NT-500M", "GERP"]
COL_B = {"Evo2-40B": S.CODING, "NT-500M": S.NT_C, "GERP": S.GERP_C}


def panel_B(ax):
    D = json.load(open("reports/fig3_data.json"))["B_class_level"]
    eq = {e["scorer"]: e for e in D["eqtl"]}
    xs = [0, 1, 2]
    ax.set_xlim(-0.62, 2.7); ax.set_ylim(0.44, 0.99)
    ax.axhspan(0.44, 0.5, color="#DCE6EC", alpha=0.7, zorder=0)                 # the 'sea' below chance
    S.bedrock(ax, 0.5, -0.62, 2.7, label=False)
    ax.text(-0.58, 0.507, "chance 0.5 (sea level)", ha="left", va="bottom", fontsize=5.4, color=S.BEDROCK,
            fontweight="bold", style="italic")
    for x, sc in zip(xs, ORDER_B):
        c = COL_B[sc]; ch = D["coding_ref"][sc]; e = eq[sc]
        ax.plot([x - 0.17, x + 0.17], [ch, ch], color=c, lw=1.8, zorder=3)      # coding clifftop (reference)
        ax.scatter([x], [ch], s=24, facecolors="white", edgecolors=c, lw=1.4, zorder=4)
        ax.text(x, ch + 0.013, f"coding {ch:.2f}", ha="center", va="bottom", fontsize=5.6, color=c, fontweight="bold")
        ax.annotate("", xy=(x, e["auroc"] + 0.008), xytext=(x, ch - 0.008),     # the plunge
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.8, alpha=0.85, mutation_scale=14), zorder=3)
        ax.plot([x, x], [e["lo"], e["hi"]], color=S._shade(c, -0.25), lw=1.3, zorder=4)   # eQTL CI in the water
        ax.scatter([x], [e["auroc"]], s=30, color=S._shade(c, -0.25), edgecolors="white", lw=0.8, zorder=5)
        ax.text(x + 0.21, (ch + e["auroc"]) / 2, f"−{D['drop'][sc]:.2f}", ha="left", va="center", fontsize=6.6,
                color=c, fontweight="bold")
        ax.text(x, 0.452, sc, ha="center", va="top", fontsize=6.6, fontweight="bold", color=S.INK)
    ax.annotate("conservation FLIPS:\ncoding-winner → chance", xy=(2, 0.66), xytext=(1.5, 0.86),
                fontsize=5.8, color=S._shade(S.GERP_C, -0.15), fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=S.GERP_C, lw=0.9, connectionstyle="arc3,rad=-0.2"))
    ax.text(1, 0.63, "NT barely falls —\nweak on coding too", ha="center", va="center", fontsize=5.0,
            color=S._shade(S.NT_C, -0.25), style="italic")
    ax.text(2.62, 0.50, "all three land\nat chance", ha="right", va="center", fontsize=5.2, color=S.MUTED, style="italic")
    ax.set_xticks([]); ax.set_ylabel("AUROC", fontsize=8, color=S.CAP); ax.tick_params(labelsize=6.4)
    for s in ("top", "right", "bottom"): ax.spines[s].set_visible(False)


def panel_C(ax):
    C = json.load(open("reports/fig3_data.json"))["C_null"]
    perm = np.array(C["perm_null_sample"]); obs = C["global_auroc"]
    ax.set_xlim(0.40, 1.03); ax.set_ylim(0, 1)
    ax.axvline(0.5, color=S.BEDROCK, lw=2.0, zorder=3)                          # vertical chance datum
    ax.text(0.5, 0.99, "chance 0.5", ha="center", va="top", fontsize=5.8, color=S.BEDROCK, fontweight="bold")
    ax.axvspan(0.80, 1.03, color=S.CODING, alpha=0.08, zorder=0)               # empty signal zone
    ax.text(0.915, 0.60, "a usable causal\npredictor lands\nhere →", ha="center", va="center", fontsize=5.8,
            color=S._shade(S.CODING, -0.05), style="italic")
    ax.text(0.915, 0.30, "(empty)", ha="center", va="center", fontsize=6.0, color=S._shade(S.CODING, -0.05), fontweight="bold")
    rng = np.random.default_rng(3)
    yj = rng.uniform(0.28, 0.74, len(perm))
    ax.scatter(perm, yj, s=7, color="#BFBFBF", alpha=0.55, marker="|", linewidths=0.7, zorder=2)  # 600 permutation nulls
    ax.scatter([obs], [0.505], s=9, color=S.FAIL, alpha=0.75, marker="|", linewidths=1.1, zorder=3)  # observed, camouflaged
    ax.annotate("the observed AUROC (0.488) is\none of these ticks — you can't tell\nwhich. That is the null.",
                xy=(obs, 0.55), xytext=(0.405, 0.90), fontsize=5.4, color=S.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=0.6, alpha=0.6, connectionstyle="arc3,rad=0.2"))
    ax.text(0.405, 0.12, f"within each eGene (LD-matched): the PIP≥0.9 causal variant is top-scored only "
            f"{C['within_egene']['precision_at_1']*100:.0f}% of the time (base rate {C['within_egene']['base_rate']*100:.0f}%);"
            f" mean causal rank-percentile {C['within_egene']['mean_causal_rank_pctile']:.2f} (0.5 = chance)",
            fontsize=5.0, color=S.MUTED, va="center", ha="left")
    ax.set_yticks([]); ax.set_xlabel("AUROC  (causal cis-eQTL vs LD-matched controls)", fontsize=7.2, color=S.CAP)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.tick_params(labelsize=6.2)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    panel_B(a); panel_C(b)
    S.finding_title(a, "B", "The blind spot is class-wide —\neven conservation flips to chance")
    S.finding_title(b, "C", "Genuine chance: the observed AUROC\nhides among its own permutations")
    fig.savefig("reports/figures/_fig3_BC.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig3_BC.png")


if __name__ == "__main__":
    main()
