"""Figure 1 - "No control rescues regulation" (the thesis figure).
Elimination matrix: coding vs regulatory held fixed across ascertainment-matching, a 40x scale
sweep, and a swapped model family. Everything as distance-above-chance (AUROC-0.5), null at origin.
Numbers from reports/COMPILED_RESULTS.md + type_matched_atlas + compiled_results (all CI-backed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs

# ---------- data (AUROC, lo, hi) ----------
# Panel B elimination matrix rows (top->bottom within blocks)
ASCERT = [("unmatched", 0.962, .951, .972, True),
          ("coding-only", 0.925, .903, .945, True),
          ("missense type-matched", 0.824, .761, .877, True)]
SCALE_COD = [("40B", .942, None, None, True), ("7B", .927, None, None, True), ("1B", .898, None, None, True)]
SCALE_REG = [("40B", .498, .473, .524, False), ("7B", .488, .463, .513, False), ("1B", .492, .466, .517, False)]
FAM = [("Evo2 coding", .878, None, None, True), ("NT coding", .595, None, None, True),
       ("Evo2 eQTL", .498, .473, .524, False), ("NT eQTL", .486, .460, .511, False)]
# Panel C trajectories
SC = ["1B", "7B", "40B"]
COD = [.898, .927, .942]
REG = [.492, .488, .498]; REG_LO = [.466, .463, .473]; REG_HI = [.517, .513, .524]
COD_SPREAD_40 = [.926, .885, .841, .804, .743, .727]   # per-species missense@8192 (the real n=spread)


# vertebrate topology; traversal order gives tips top->bottom
TREE = ("root", [
    ("Mammalia", [
        ("human", []),
        ("Laurasiatheria", [
            ("Carnivora", [("cat", []), ("dog", [])]),
            ("Ungulata", [
                ("horse", []),
                ("Artiodactyla", [
                    ("pig", []),
                    ("Ruminantia", [("cattle", []), ("Caprinae", [("sheep", []), ("goat", [])])]),
                ]),
            ]),
        ]),
    ]),
    ("chicken", []),
])


def _layout(node, c=None):
    # The counter is made per call. A mutable default is created once at def time and never
    # reset, so a second call in one process would return leaf y values 9..17 instead of 0..8. Its
    # consumer _draw_tree maps y through sy(y) = y1 - (y/8)*(y1-y0), which assumes [0, 8], so the
    # nine species labels would land outside the panel with no exception raised.
    if c is None:
        c = [0]
    name, kids = node
    if not kids:
        y = c[0]; c[0] += 1
        return {"n": name, "x": 0.0, "y": y, "k": [], "leaf": True}
    K = [_layout(k, c) for k in kids]
    return {"n": name, "x": min(k["x"] for k in K) - 1.0,
            "y": sum(k["y"] for k in K) / len(K), "k": K, "leaf": False}


def _draw_tree(ax, node, x0, x1, y0, y1, xmin):
    def sx(x): return x0 + (x - xmin) / (0 - xmin) * (x1 - x0)   # tree x (neg->0) into [x0,x1]
    def sy(y): return y1 - (y / 8.0) * (y1 - y0)                 # tip 0 at top

    def rec(nd):
        if nd["leaf"]:
            ax.plot([sx(nd["x"]), x1 + 0.02], [sy(nd["y"])] * 2, color="#B9C2CB", lw=1.0, zorder=2)
            ax.text(x1 + 0.03, sy(nd["y"]), nd["n"], va="center", ha="left", fontsize=7.0, color=fs.INK)
            return
        ys = [sy(k["y"]) for k in nd["k"]]
        ax.plot([sx(nd["x"])] * 2, [min(ys), max(ys)], color="#B9C2CB", lw=1.0, zorder=2)  # vertical
        for k in nd["k"]:
            ax.plot([sx(nd["x"]), sx(k["x"])], [sy(k["y"])] * 2, color="#B9C2CB", lw=1.0, zorder=2)
            rec(k)
    rec(node)
    return sx, sy


def schematic(ax):
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    T = _layout(TREE); xmin = T["x"]
    sx, sy = _draw_tree(ax, T, x0=0.15, x1=1.9, y0=0.15, y1=2.55, xmin=xmin)
    ax.text(0.15, 2.85, "9 vertebrates · ~320 My", fontsize=7.4, fontweight="bold", color=fs.INK)

    def box(x, y, w, h, t, fc, ec, tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.09",
                                    fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.1, color=tc or fs.INK, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                     lw=1.2, color="#9AA3AB", zorder=2, shrinkA=1, shrinkB=1))
    arrow(3.05, 1.5, 3.75, 1.5)
    box(3.8, 1.02, 1.75, 0.96, "Evo 2 – 40B\nzero-shot\nmean log-lik.", "#EAF1F7", fs.EVO2, fs.EVO2)
    arrow(5.65, 1.5, 6.3, 2.05); arrow(5.65, 1.5, 6.3, 0.95)
    box(6.35, 1.72, 2.05, 0.78, "coding disease variants\n(OMIA · type-matched)", "#EAF1F7", fs.EVO2)
    box(6.35, 0.55, 2.05, 0.78, "causal cis-eQTLs\n(PigGTEx · LD-matched)", "#FBEEE6", fs.NT)
    arrow(8.5, 2.11, 9.05, 1.65); arrow(8.5, 0.94, 9.05, 1.4)
    box(9.1, 1.05, 0.82, 0.9, "matched\nboth\narms", "#F3F3F3", "#9AA3AB", fs.MUTED)
    ax.text(3.05, 2.82, "held fixed: same variants · same 8192-bp readout · same class balance —"
            " only functional class toggled", fontsize=6.6, style="italic", color=fs.MUTED)


def elimination(ax):
    blocks = [
        ("ascertainment", [("unmatched", .962, .951, .972, True), ("coding-only", .925, .903, .945, True),
                           ("missense-matched", .824, .761, .877, True)]),
        ("scale  1B→40B", [("40B", .942, None, None, True), ("7B", .927, None, None, True), ("1B", .898, None, None, True),
                           ("40B", .498, .473, .524, False), ("7B", .488, .463, .513, False), ("1B", .492, .466, .517, False)]),
        ("model family", [("Evo2", .878, None, None, True), ("NT", .595, None, None, True),
                          ("Evo2", .498, .473, .524, False), ("NT", .486, .460, .511, False)]),
    ]
    y = 0.0; spans = []
    for name, rows in blocks:
        y0 = y - 0.5
        for (lab, a, lo, hi, cod) in rows:
            lo = lo if lo is not None else a; hi = hi if hi is not None else a
            col = fs.NT if lab == "NT" else fs.EVO2
            fs.cbar_point(ax, fs.dac(a), y, fs.dac(lo), fs.dac(hi), col, cod, label=f"{lab}  {a:.3f}")
            y += 1
        spans.append((name, y0, y - 0.5)); y += 0.7
    total = y - 0.7
    for i, (name, a0, a1) in enumerate(spans):          # faint band + block label in the dead zone
        if i % 2 == 0:
            ax.axhspan(a0, a1, color="#F5F5F5", zorder=0)
        ax.text(0.205, (a0 + a1) / 2, name, va="center", ha="left", fontsize=7.4,
                fontweight="bold", color=fs.MUTED, zorder=5)
    ax.axvspan(0.27, 0.62, color=fs.EVO2, alpha=0.045, lw=0, zorder=0)   # coding territory
    fs.null_axis(ax, lo=-0.06, hi=0.62)
    ax.set_ylim(-0.7, total); ax.set_yticks([]); ax.invert_yaxis()
    ax.text(0.0, total + 0.15, "at chance", fontsize=6.6, color=fs.MUTED, ha="center", va="top")
    ax.text(0.445, total + 0.15, "coding grammar", fontsize=6.6, color=fs.EVO2, ha="center", va="top", fontweight="bold")
    ax.set_title("Every control holds coding above chance; none rescues regulation", pad=14)


def trajectories(ax):
    x = np.arange(3)
    # coding: mean trajectory (clean; per-species dots deferred to a same-readout pass)
    ax.plot(x, [fs.dac(v) for v in COD], "-", color=fs.EVO2, lw=2.2, zorder=3)
    ax.plot(x, [fs.dac(v) for v in COD], "o", color=fs.EVO2, ms=6.5, zorder=4)
    for xi, v in zip(x, COD):
        ax.text(xi, fs.dac(v) + 0.018, f"{v:.3f}", ha="center", fontsize=6.5, color=fs.EVO2)
    # regulatory: eQTL with CI ribbon, open markers
    lo = [fs.dac(v) for v in REG_LO]; hi = [fs.dac(v) for v in REG_HI]
    ax.fill_between(x, lo, hi, color=fs.NT, alpha=0.15, lw=0)
    ax.plot(x, [fs.dac(v) for v in REG], "--", color=fs.NT, lw=1.6, zorder=3)
    ax.plot(x, [fs.dac(v) for v in REG], "o", mfc="white", mec=fs.NT, mew=1.4, ms=6, zorder=4)
    ax.axhline(0, color=fs.NULLC, lw=0.9, alpha=0.45)
    ax.axhspan(-0.02, 0.02, color=fs.NULLC, alpha=0.05, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(SC)
    ax.set_xlabel("model scale  (params, log)")
    ax.set_ylabel("AUROC − 0.5")
    ax.set_ylim(-0.06, 0.50)
    ax.text(1.5, fs.dac(COD[1]) + 0.028, "coding", color=fs.EVO2, fontsize=8, fontweight="bold", ha="center")
    ax.text(1.5, fs.dac(REG[1]) - 0.035, "causal eQTL", color=fs.NT, fontsize=8, fontweight="bold", ha="center")
    ax.set_title("Scale buys coding, not regulation")


def dumbbell(ax):
    rows = [("coding", .878, .595, True), ("eQTL", .498, .486, False)]
    for i, (lab, ev, nt, cod) in enumerate(rows):
        y = i
        ax.plot([fs.dac(nt), fs.dac(ev)], [y, y], color="#CCCCCC", lw=2, zorder=2)
        ax.plot(fs.dac(ev), y, "o", mfc=(fs.EVO2 if cod else "white"), mec=fs.EVO2, mew=1.4, ms=7.5, zorder=4)
        ax.plot(fs.dac(nt), y, "o", mfc=(fs.NT if cod else "white"), mec=fs.NT, mew=1.4, ms=7.5, zorder=4)
    ax.axvline(0, color=fs.NULLC, lw=0.9, alpha=0.45)
    ax.axvspan(-0.02, 0.02, color=fs.NULLC, alpha=0.05, lw=0)
    ax.set_ylim(-0.7, 1.7); ax.set_yticks([0, 1]); ax.set_yticklabels(["coding", "eQTL"], fontweight="bold")
    ax.tick_params(axis="y", length=0); ax.invert_yaxis()
    ax.set_xlim(-0.06, 0.47); ax.set_xlabel("AUROC − 0.5")
    ax.text(fs.dac(.878), 0.34, "Evo2", color=fs.EVO2, fontsize=7, ha="center", fontweight="bold")
    ax.text(fs.dac(.595), -0.34, "NT", color=fs.NT, fontsize=7, ha="center", fontweight="bold")
    ax.set_title("Ceiling is the model class", pad=8)


def main():
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.82, 1.5, 0.98], width_ratios=[1, 1, 1],
                  hspace=0.42, wspace=0.5, left=0.10, right=0.97, top=0.935, bottom=0.075)
    axA = fig.add_subplot(gs[0, :]); schematic(axA); fs.panel_letter(axA, "a", dx=-0.01, dy=1.10)
    axB = fig.add_subplot(gs[1, :]); elimination(axB); fs.panel_letter(axB, "b", dx=-0.055, dy=1.06)
    axC = fig.add_subplot(gs[2, :2]); trajectories(axC); fs.panel_letter(axC, "c", dx=-0.09)
    axD = fig.add_subplot(gs[2, 2]); dumbbell(axD); fs.panel_letter(axD, "d", dx=-0.06)
    fig.text(0.11, 0.985, "coding = filled marker   ·   regulatory = open marker   ·   null band shaded at the origin",
             fontsize=6.8, style="italic", color=fs.MUTED, ha="left")
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/Figure1.png", bbox_inches="tight")
    fig.savefig("reports/figures/Figure1.pdf", bbox_inches="tight")
    print("wrote reports/figures/Figure1.png / .pdf")


if __name__ == "__main__":
    main()
