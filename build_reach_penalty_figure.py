#!/usr/bin/env python3
"""Build the reach-versus-must-answer-penalty figure and its 49-row table.

Exploratory: no submitted figure uses it (FIGURES.md). The paper prints the must-answer penalty
(median 0.284) beside the gene-level circularity correction (0.031) but not their ratio, since the
correction has no fixed estimand; this figure divides one by the other, and the ratio falls BELOW
one for predictors that reach most of the panel.

Panel (a) plots each predictor's penalty against its reach. Panel (b) gives the median penalty of
each reach group as a multiple of that correction.

Source of truth is the deposited reach audit; nothing here is typed by hand.

    python3 build_reach_penalty_figure.py
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
AUDIT = os.path.join(ROOT, "reports", "dbnsfp_reach_audit.json")

# The gene-level circularity correction the field does make, against which the must-answer penalty
# is compared. Stated in the manuscript; not derivable from this audit file.
GENE_LEVEL_DROP = 0.031

MM = 1 / 25.4
W_MM, H_MM = 170, 88

# Okabe-Ito, the palette used by the other figures in this paper.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#666666"
LIGHT = "#BBBBBB"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

STRATA = [(0.00, "all 49"), (0.75, "reach ≥ 0.75"), (0.90, "reach ≥ 0.90"), (0.99, "reach ≥ 0.99")]

# Named in the manuscript's own discussion of this result, so a reader can find them here.
# Offsets are hand-set: the low-reach cluster is dense and default placement overlapped it.
LABEL = {
    "CADD_raw":      ("CADD", (5, -7)),          # reach 1.000, penalty 0.000
    "GERP++_RS":     ("GERP++", (5, 5)),         # reach 0.996, penalty 0.002 — sits on top of CADD
    "AlphaMissense": ("AlphaMissense", (7, -4)), # reach 0.626, penalty 0.313
    "REVEL":         ("REVEL", (-34, 3)),        # reach 0.598, penalty 0.330 — left, to clear the above
    "popEVE":        ("popEVE", (-30, -2)),      # reach 0.481, penalty 0.338
}


def load():
    with open(AUDIT) as fh:
        d = json.load(fh)
    return d["_meta"], d["predictors"]


def strata_medians(preds):
    out = []
    for thr, name in STRATA:
        sub = [p for p in preds if p["reach"] >= thr]
        med = st.median([p["penalty"] for p in sub])
        out.append((name, len(sub), med, med / GENE_LEVEL_DROP))
    return out


def panel_a(ax, preds):
    over = [p for p in preds if p["penalty"] > GENE_LEVEL_DROP]
    under = [p for p in preds if p["penalty"] <= GENE_LEVEL_DROP]

    ax.axhline(GENE_LEVEL_DROP, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.text(0.012, GENE_LEVEL_DROP * 1.35, "gene-level circularity correction, 0.031",
            fontsize=6.3, color=GREY, va="bottom")

    for thr, _ in STRATA[1:]:
        ax.axvline(thr, color=LIGHT, lw=0.7, zorder=0)
        ax.text(thr, 1.02, f"{thr:.2f}", fontsize=6, color=GREY, ha="center",
                transform=ax.get_xaxis_transform())

    ax.scatter([p["reach"] for p in over], [p["penalty"] for p in over],
               s=17, facecolor=VERMILLION, edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter([p["reach"] for p in under], [p["penalty"] for p in under],
               s=17, facecolor=BLUE, edgecolor="white", linewidth=0.4, zorder=3)

    for p in preds:
        if p["predictor"] in LABEL:
            name, offset = LABEL[p["predictor"]]
            ax.annotate(name, (p["reach"], p["penalty"]), fontsize=6, color="#222222",
                        xytext=offset, textcoords="offset points", zorder=4)

    ax.set_xlabel("Reach — fraction of the 328,328-variant panel the predictor scores")
    ax.set_ylabel("Must-answer penalty (AUROC)")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 0.62)
    ax.set_title("a", loc="left", fontweight="bold")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=VERMILLION, markeredgecolor="white",
               markersize=4.6, label=f"penalty exceeds 0.031  (n = {len(over)})"),
        Line2D([], [], marker="o", ls="", markerfacecolor=BLUE, markeredgecolor="white",
               markersize=4.6, label=f"penalty at or below 0.031  (n = {len(under)})"),
    ], loc="upper right", frameon=False, fontsize=6.3, handletextpad=0.3)


def panel_b(ax, rows):
    ax.axvline(1.0, color=GREY, lw=0.9, ls="--", zorder=1)
    # in the gap between the first and second bar, so it clears the panel letter above
    ax.text(1.15, len(rows) - 1.5, "parity", fontsize=6.3, color=GREY, ha="left", va="center")

    ys = range(len(rows))[::-1]
    for y, (name, n, med, ratio) in zip(ys, rows):
        colour = VERMILLION if ratio > 1 else BLUE
        ax.barh(y, ratio, height=0.52, color=colour, zorder=2)
        # white bbox so a short bar's label stays readable where it crosses the parity line
        ax.text(ratio + 0.28, y, f"{ratio:.2f}×", va="center", fontsize=6.6, color="#222222",
                zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        ax.text(-0.35, y, f"{name}\nn = {n}", va="center", ha="right", fontsize=6.3, color="#222222")

    ax.set_xlabel("Median penalty ÷ gene-level correction")
    ax.set_xlim(0, 11.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("b", loc="left", fontweight="bold")


def main():
    meta, preds = load()
    rows = strata_medians(preds)

    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30,
                          left=0.078, right=0.965, top=0.90, bottom=0.145)
    panel_a(fig.add_subplot(gs[0, 0]), preds)
    panel_b(fig.add_subplot(gs[0, 1]), rows)

    out_pdf = os.path.join(HERE, "Figure_reach_penalty.pdf")
    fig.savefig(out_pdf)
    fig.savefig(os.path.join(HERE, "Figure_reach_penalty.png"), dpi=300)
    plt.close(fig)

    # ---- the 49-row table ----
    order = sorted(preds, key=lambda p: (-p["reach"], p["predictor"]))
    tsv = os.path.join(HERE, "Table_S3_reach_penalty.tsv")
    with open(tsv, "w") as fh:
        fh.write("predictor\treach\treach_positives\treach_negatives\t"
                 "auroc_covered\tauroc_must_answer\tmust_answer_penalty\n")
        for p in order:
            fh.write(f"{p['predictor']}\t{p['reach']:.4f}\t{p['reach_pos']:.4f}\t"
                     f"{p['reach_neg']:.4f}\t{p['auroc_covered']:.4f}\t"
                     f"{p['auroc_must_answer']:.4f}\t{p['penalty']:.4f}\n")

    med = st.median([p["penalty"] for p in preds])
    print(f"panel: {meta['n']:,} variants, {meta['n_genes']:,} genes, {len(preds)} predictors")
    print(f"median must-answer penalty {med:.4f}  ({med / GENE_LEVEL_DROP:.1f}× the 0.031 correction)")
    for name, n, m, r in rows:
        print(f"  {name:<14} n={n:>2}  median penalty {m:.4f}  ratio {r:.2f}")
    print(f"\nwrote {out_pdf}\nwrote {tsv}")


if __name__ == "__main__":
    main()
