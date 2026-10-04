"""Figure 4 — THE LEAK.  A decision pipeline drawn as a physical system, with the real variants flowing.

Not a grid of charts: a left-to-right schematic in which 11,130 real variants enter, pass two gates, and
arrive somewhere. Stream widths are area-true counts; the particles inside each stream are the actual
variants, positioned by their measured log-odds. The argument is spatial — you watch where the pathogenic
mass leaks out, and you watch the trust gate fail to catch it.

  gate 1  the model calls at p = 0.5      -> 731 positives cross into the "negative" channel
  gate 2  the trust layer refuses the least-confident 15%
          it removes 77% of the false alarms and 26% of the misses
  arrive  540 missed positives reach the clinician, inside a stream labelled "benign"

Values from reports/fig4_leak.json. Run: python -m src.ccs.fig4_leak
"""
import os, json, textwrap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch

MM = 1 / 25.4
W_MM, H_MM = 180.0, 168.0
TITLE, PANEL, AXIS, TICK, FOOT = 9.5, 7.0, 6.5, 6.0, 5.5

MISS = "#A8201A"      # a pathogenic variant called benign — the costly error
ALARM = "#E69F00"     # a benign variant called pathogenic — the affordable error
PATH = "#8C3B36"      # pathogenic truth
BEN = "#AFB6BD"       # benign truth
INK = "#1A1A1A"
MUTED = "#63666A"
RULE = "#3A3A3A"
GOOD = "#0072B2"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})
G = json.load(open("reports/fig4_leak.json", encoding="utf-8"))
N = G["n"]


def _wrap(t, mm, pt):
    return textwrap.fill(" ".join(t.split()), width=max(20, int((mm * 72 / 25.4) / (0.52 * pt))))


def _stream(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha, z=2):
    """Area-true flow band with an S-curve spine."""
    m0, m1 = x0 + (x1 - x0) * 0.42, x0 + (x1 - x0) * 0.58
    v = [(x0, y0a), (m0, y0a), (m1, y1a), (x1, y1a), (x1, y1b), (m1, y1b), (m0, y0b), (x0, y0b), (x0, y0a)]
    c = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), fc=color, ec="none", alpha=alpha, zorder=z))


def _particles(ax, key, x0, x1, ya, yb, color, size, rng, n_max=900):
    """Real variants inside a stream: x from their measured log-odds, y filling the band."""
    v = np.array(G[key], float)
    if len(v) == 0:
        return
    if len(v) > n_max:
        v = rng.choice(v, n_max, replace=False)
    lo, hi = -9.5, 4.5
    fx = np.clip((v - lo) / (hi - lo), 0, 1)
    xs = x0 + (x1 - x0) * fx
    ys = yb + (ya - yb) * rng.random(len(v))
    ax.scatter(xs, ys, s=size, color=color, alpha=0.75, linewidths=0, zorder=6, rasterized=True)


# ---------------------------------------------------------------- the pipeline
def panel_leak(ax):
    """Four classes tracked through four stations. Every band width is an exact count; the particles
    inside the mid-section are the real variants at their measured log-odds."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.default_rng(7)
    TOP, H, GAPU = 0.885, 0.735, 0.016
    u = H / N
    xs = {"in": 0.115, "g1": 0.335, "g2": 0.625, "out": 0.845}

    # class order within each channel, and the channel each class lands in per station
    #   station 1 TRUTH        : channel 0 = pathogenic, channel 1 = benign
    #   station 2/3/4 THE CALL : channel 0 = called pathogenic, channel 1 = called benign
    CLS = ["tp", "miss", "fa", "tn"]
    ORD1 = {0: ["tp", "miss"], 1: ["fa", "tn"]}
    ORD2 = {0: ["tp", "fa"], 1: ["miss", "tn"]}
    surv = {"tp": G["tp"] - G["tp_refused"], "miss": G["miss_survive"],
            "fa": G["fa_survive"], "tn": G["tn"] - G["tn_refused"]}

    def spans(order, counts):
        """y-spans per class, two channels stacked from the top with a gap between them."""
        out = {}; y = TOP
        for ch in (0, 1):
            for c in order[ch]:
                h = counts[c] * u
                out[c] = (y, y - h); y -= h
            y -= GAPU
        return out

    full = {c: G[c] for c in CLS}
    sA = spans(ORD1, full)      # truth
    sB = spans(ORD2, full)      # after the call
    sC = sB                     # the call channels persist up to gate 2
    sD = spans(ORD2, surv)      # after refusal

    COL = {"tp": (PATH, 0.40), "miss": (MISS, 0.90), "fa": (ALARM, 0.90), "tn": (BEN, 0.42)}

    # entry blocks
    for ch, lab, col, cnt in [(0, f"pathogenic\n{G['path']:,}", PATH, G["path"]),
                              (1, f"benign\n{G['benign']:,}", BEN, G["benign"])]:
        t = sA[ORD1[ch][0]][0]; bm = sA[ORD1[ch][-1]][1]
        ax.add_patch(Rectangle((xs["in"] - 0.030, bm), 0.024, t - bm, fc=col, ec="none", alpha=0.9))
        ax.text(xs["in"] - 0.038, (t + bm) / 2, lab, ha="right", va="center", fontsize=FOOT,
                color=(PATH if ch == 0 else MUTED), fontweight="bold", linespacing=1.2)

    for c in CLS:
        col, al = COL[c]
        _stream(ax, xs["in"], xs["g1"], sA[c][0], sA[c][1], sB[c][0], sB[c][1], col, al)   # the crossing
        _stream(ax, xs["g1"], xs["g2"], sB[c][0], sB[c][1], sC[c][0], sC[c][1], col, al)   # carry through
        _stream(ax, xs["g2"], xs["out"], sC[c][0], sC[c][0] - surv[c] * u,
                sD[c][0], sD[c][1], col, al)                                               # survivors

    # real variants inside the carry-through section
    for c, key, sz in [("miss", "logit_miss", 1.2), ("fa", "logit_fa", 1.2)]:
        _particles(ax, key, xs["g1"] + 0.02, xs["g2"] - 0.02, sC[c][0] - 0.002, sC[c][1] + 0.002,
                   COL[c][0], sz, rng)

    # gates
    for k, lab, sub in [("g1", "GATE 1", "the model calls at p = 0.5"),
                        ("g2", "GATE 2", "the trust layer refuses the least-confident 15%")]:
        x = xs[k]
        ax.plot([x, x], [TOP - H - GAPU - 0.02, TOP + 0.028], color=RULE, lw=0.9, zorder=8)
        ax.text(x, TOP + 0.040, lab, ha="center", va="bottom", fontsize=FOOT - 0.2, color=RULE,
                fontweight="bold")
        ax.text(x, TOP - H - GAPU - 0.032, sub, ha="center", va="top", fontsize=FOOT - 0.4,
                color=MUTED, linespacing=1.2)
    ax.text(xs["in"] - 0.030, TOP + 0.040, "TRUTH", ha="left", va="bottom", fontsize=FOOT - 0.4, color=MUTED)
    ax.text(xs["out"], TOP + 0.040, "WHAT ARRIVES", ha="left", va="bottom", fontsize=FOOT - 0.4, color=MUTED)

    # channel labels after the call
    ax.text(xs["g1"] + 0.010, sB["tp"][0] - 0.012, "called pathogenic", ha="left", va="top",
            fontsize=FOOT - 0.3, color=INK)
    ax.text(xs["g1"] + 0.010, sB["miss"][0] - 0.012, "called benign", ha="left", va="top",
            fontsize=FOOT - 0.3, color=INK)

    # THE LEAK
    ax.annotate("", xy=(xs["g1"] - 0.012, (sB["miss"][0] + sB["miss"][1]) / 2),
                xytext=(xs["in"] + 0.045, sA["miss"][1] - 0.030),
                arrowprops=dict(arrowstyle="-|>", color=MISS, lw=1.1, mutation_scale=7,
                                connectionstyle="arc3,rad=-0.28"), zorder=9)
    ax.text(xs["in"] + 0.040, sA["miss"][1] - 0.036,
            f"{G['miss']} positives\ncross into “negative”",
            ha="left", va="top", fontsize=FOOT + 0.2, color=MISS, fontweight="bold",
            linespacing=1.25, zorder=9)

    # what the refusal gate removes, drawn peeling off below
    ax.text(xs["g2"], TOP - H - GAPU - 0.070,
            f"removes {G['refused']:,} calls:  {G['fa_refused']} of {G['fa']} false alarms, "
            f"only {G['miss_refused']} of {G['miss']} misses",
            ha="center", va="top", fontsize=FOOT - 0.4, color=RULE, fontweight="bold")

    # arrival
    ax.text(xs["out"] + 0.028, (sD["tp"][0] + sD["tp"][1]) / 2, f"{surv['tp']:,} correctly flagged",
            ha="left", va="center", fontsize=FOOT - 0.3, color=MUTED)
    ax.text(xs["out"] + 0.028, (sD["miss"][0] + sD["miss"][1]) / 2,
            f"{G['miss_survive']} missed positives\nreach the clinician",
            ha="left", va="center", fontsize=FOOT + 0.3, color=MISS, fontweight="bold", linespacing=1.25)


# ---------------------------------------------------------------- supporting quantification
def panel_conf(ax):
    rng = np.random.default_rng(11)
    lanes = [("correct calls", "logit_tp", "#B9BFC6", 0.5, 0.30),
             ("missed positives", "logit_miss", MISS, 1.9, 0.75),
             ("false alarms", "logit_fa", ALARM, 1.9, 0.85)]
    ax.set_xlim(-9.5, 4.5); ax.set_ylim(-0.6, 2.6)
    ax.axvline(0, color=RULE, lw=0.7, zorder=3)
    for i, (nm, key, col, sz, al) in enumerate(lanes[::-1]):
        v = np.array(G[key], float)
        ax.scatter(v, i + (rng.random(len(v)) - 0.5) * 0.5, s=sz, color=col, alpha=al,
                   linewidths=0, rasterized=True, zorder=2)
        ax.plot([np.median(v)] * 2, [i - 0.30, i + 0.30], color=col, lw=1.5, zorder=5)
        ax.text(-9.3, i + 0.34, nm, fontsize=FOOT - 0.2, color=INK, ha="left", va="center",
                fontweight="bold")
    ax.set_yticks([]); ax.set_xticks([-8, -4, 0, 2, 4]); ax.tick_params(labelsize=FOOT - 0.4)
    ax.set_xlabel("signed log-odds     ←  called benign      called pathogenic  →", fontsize=FOOT)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9A9A9A")
    ax.text(0.0, -0.40, _wrap(f"The misses sit where the CORRECT calls sit (median |2p−1| "
                              f"{G['miss_conf_median']:.2f} vs {G['tp_conf_median']:.2f}), not where the "
                              f"false alarms sit ({G['fa_conf_median']:.2f}). That is why gate 2 cannot "
                              f"find them.", 82, FOOT - 0.3),
            transform=ax.transAxes, fontsize=FOOT - 0.3, color=INK, ha="left", va="top", linespacing=1.3)


def panel_catch(ax):
    cats = [("false alarms\n(affordable)", G["pct_fa_caught"], ALARM, G["fa_refused"], G["fa"]),
            ("missed positives\n(costly)", G["pct_miss_caught"], MISS, G["miss_refused"], G["miss"])]
    ys = [1, 0]
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.55, 1.55)
    for y, (nm, v, col, k, tot) in zip(ys, cats):
        ax.add_patch(Rectangle((0, y - 0.20), 1.0, 0.40, fc="#F1F2F4", ec="none", zorder=1))
        ax.add_patch(Rectangle((0, y - 0.20), v, 0.40, fc=col, ec="none", alpha=0.85, zorder=2))
        ax.text(0.0, y + 0.31, nm, fontsize=FOOT - 0.2, color=INK, ha="left", va="bottom",
                fontweight="bold", linespacing=1.15)
        ax.text(v + 0.014, y, f"{v:.0%}   ({k} of {tot})", fontsize=FOOT - 0.2, color=col,
                ha="left", va="center", fontweight="bold")
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=FOOT - 0.4)
    ax.set_xlabel("share of that error class the trust layer removes", fontsize=FOOT)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9A9A9A")


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    gs = GridSpec(2, 2, height_ratios=[1.62, 0.72], hspace=0.60, wspace=0.26,
                  left=0.085, right=0.965, top=0.845, bottom=0.085)
    axL = fig.add_subplot(gs[0, :]); panel_leak(axL)
    ax1 = fig.add_subplot(gs[1, 0]); panel_conf(ax1)
    ax2 = fig.add_subplot(gs[1, 1]); panel_catch(ax2)
    for ax, L, t in [(axL, "a", "Where the positives leak out"),
                     (ax1, "b", "The misses are confident"),
                     (ax2, "c", "So the gate catches the wrong class")]:
        ax.text(-0.02 if L == "a" else -0.10, 1.04, L, transform=ax.transAxes,
                fontsize=PANEL + 1.0, fontweight="bold", va="bottom", ha="right", color=INK)
        ax.text(0.01 if L == "a" else -0.05, 1.04, t, transform=ax.transAxes, fontsize=PANEL,
                fontweight="bold", va="bottom", ha="left", color=INK)

    fig.text(0.085, 0.968, "The trust layer catches the errors you could afford",
             fontsize=TITLE, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.085, 0.936,
             _wrap(f"All {N:,} cross-species disease variants, flowing through the decision pipeline. "
                   f"Stream widths are counts; the particles inside them are the actual variants, placed "
                   f"by their measured log-odds.", 170, PANEL),
             fontsize=PANEL, color=MUTED, ha="left", va="top", style="italic", linespacing=1.35)
    fig.text(0.085, 0.038,
             _wrap("Probabilities from Platt scaling on a leave-one-species-out pool of label-rich species; "
                   "calls taken at p = 0.5; confidence is |2p−1|; the refusal gate removes each species' own "
                   "least-confident 15%. Particle counts are subsampled for rendering; every width, share "
                   "and count is exact.", 170, FOOT - 0.5),
             fontsize=FOOT - 0.5, color=MUTED, ha="left", va="top", linespacing=1.35)

    os.makedirs("reports/figures", exist_ok=True)
    b = "reports/figures/Figure4_leak"
    fig.savefig(b + ".png", dpi=600, facecolor="white")
    fig.savefig(b + ".pdf", dpi=600, facecolor="white")
    fig.savefig(b + "_preview.png", dpi=200, facecolor="white")
    print(f"wrote {b} at {W_MM}x{H_MM} mm")
    return fig


if __name__ == "__main__":
    main()
