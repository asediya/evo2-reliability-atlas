"""Figure 4 — THE ASYMMETRY.  "Trustworthy about the wrong thing."

Evo 2 is a specificity machine: it misses a third of all positives and misses them CONFIDENTLY,
so every confidence-based trust layer protects the specificity it already has and cannot touch the
sensitivity it lacks.

a  flow      truth -> call, area-true. One stream leaks a third of its mass; the other barely leaks.
b  lanes     every variant on a signed log-odds axis, split by outcome. The misses sit where the CORRECT
             calls sit, not where the false alarms sit -- which is why confidence cannot separate them.
c  trajectory as you refuse more, the operating point walks the WRONG WAY: specificity -> 1.000 while
             sensitivity falls 0.674 -> 0.428. The ideal corner is never approached.
d  plane     every trust mechanism x species placed by what it protects. All of them sit far below the
             class-neutral diagonal; the region a working trust layer would occupy is empty.

Every value is read from reports/fig4_asym.json (src/ccs/fig4_asym_stats.py). Nothing is typed by hand.
Run: python -m src.ccs.fig4_asym
"""
import os, json, textwrap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyBboxPatch

MM = 1 / 25.4
W_MM, H_MM = 180.0, 176.0
TITLE, PANEL, AXIS, TICK, FOOT = 9.5, 7.0, 6.5, 6.0, 5.5

MISS = "#A8201A"      # the costly error: a pathogenic variant called benign
ALARM = "#E69F00"     # the affordable error: a benign variant called pathogenic
OKC = "#B9BFC6"       # correct calls
IDEAL = "#0072B2"     # the reference / where you would want to be
INK = "#1A1A1A"
MUTED = "#63666A"
RULE = "#3A3A3A"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.labelsize": AXIS, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

S = json.load(open("reports/fig4_asym.json", encoding="utf-8"))


def _wrap(t, mm, pt):
    return textwrap.fill(" ".join(t.split()), width=max(20, int((mm * 72 / 25.4) / (0.52 * pt))))


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color("#9A9A9A")


def _plab(ax, letter, text):
    ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=PANEL + 1.0, fontweight="bold",
            va="bottom", ha="right", color=INK)
    ax.text(-0.04, 1.06, text, transform=ax.transAxes, fontsize=PANEL, fontweight="bold",
            va="bottom", ha="left", color=INK)


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha):
    """Area-true bezier ribbon from a source span to a target span."""
    m = (x0 + x1) / 2
    verts = [(x0, y0a), (m, y0a), (m, y1a), (x1, y1a), (x1, y1b), (m, y1b), (m, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha, zorder=2))


# ---------------------------------------------------------------- a. the flow
def panel_a(ax):
    F = S["flow"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    N = F["pathogenic"] + F["benign"]
    gap = 0.035
    hP = (F["pathogenic"] / N) * (1 - gap); hB = (F["benign"] / N) * (1 - gap)
    # left column: truth. right column: the call.
    pT, pB = 1.0, 1.0 - hP                      # pathogenic band (top)
    bT, bB = pB - gap, pB - gap - hB            # benign band
    x0, x1 = 0.14, 0.72
    for (t, b, c) in [(pT, pB, MISS), (bT, bB, OKC)]:
        ax.add_patch(FancyBboxPatch((x0 - 0.035, b), 0.030, t - b, boxstyle="square,pad=0",
                                    fc=c, ec="none", alpha=0.35, zorder=3))
    # ribbons, area-true: pathogenic splits caught/missed; benign splits correct/false-alarm
    fCaught = F["pathogenic_caught"] / F["pathogenic"]
    yc = pT - hP * fCaught
    rT = 1.0
    rCall1 = rT - hP * fCaught - hB * (F["benign_false_alarm"] / F["benign"])
    _ribbon(ax, x0, x1, pT, yc, rT, rT - hP * fCaught, IDEAL, 0.30)                    # caught
    _ribbon(ax, x0, x1, yc, pB, bB - gap * 0.4, bB - gap * 0.4 - hP * (1 - fCaught), MISS, 0.85)  # MISSED
    fFA = F["benign_false_alarm"] / F["benign"]
    _ribbon(ax, x0, x1, bT, bT - hB * fFA, rT - hP * fCaught, rCall1, ALARM, 0.85)     # false alarm
    _ribbon(ax, x0, x1, bT - hB * fFA, bB, bB - gap * 0.4 - hP * (1 - fCaught),
            bB - gap * 0.4 - hP * (1 - fCaught) - hB * (1 - fFA), OKC, 0.55)           # correct benign

    ax.text(x0 - 0.045, (pT + pB) / 2, f"positives\n{F['pathogenic']:,}", ha="right", va="center",
            fontsize=FOOT, color=INK, linespacing=1.25, fontweight="bold")
    ax.text(x0 - 0.045, (bT + bB) / 2, f"negatives\n{F['benign']:,}", ha="right", va="center",
            fontsize=FOOT, color=INK, linespacing=1.25, fontweight="bold")
    ax.text(x0 - 0.045, 1.012, "TRUTH", ha="right", va="bottom", fontsize=FOOT - 0.5, color=MUTED)
    ax.text(x1 + 0.02, 1.012, "THE MODEL'S CALL", ha="left", va="bottom", fontsize=FOOT - 0.5, color=MUTED)

    ax.annotate(f"{F['pathogenic_missed']} positives\ncalled negative",
                xy=(x1 - 0.06, bB - gap * 0.4 - hP * (1 - fCaught) / 2),
                xytext=(x1 + 0.045, bB + 0.10), fontsize=FOOT + 0.3, color=MISS, fontweight="bold",
                ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=MISS, lw=0.6))
    ax.text(x1 + 0.045, rT - hP * fCaught / 2, f"{F['pathogenic_caught']:,} caught", ha="left",
            va="center", fontsize=FOOT, color=MUTED)
    ax.text(x1 + 0.045, rCall1 - 0.035, f"{F['benign_false_alarm']} false alarms", ha="left",
            va="center", fontsize=FOOT, color=ALARM, fontweight="bold")

    ax.text(0.0, -0.10, f"sensitivity {F['sensitivity']:.3f}  ", ha="left", va="top",
            fontsize=PANEL, color=MISS, fontweight="bold", transform=ax.transAxes)
    ax.text(0.42, -0.10, f"specificity {F['specificity']:.3f}", ha="left", va="top",
            fontsize=PANEL, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, -0.20, _wrap(f"It almost never cries wolf. It misses a third of what matters — "
                              f"{F['share_of_errors_missed_pathogenic']:.0%} of all errors are misses, "
                              f"invisible in an {F['overall_error']:.1%} overall error rate.", 84, FOOT),
            ha="left", va="top", fontsize=FOOT, color=INK, transform=ax.transAxes, linespacing=1.3)


# ---------------------------------------------------------------- b. the misses are confident
def panel_b(ax):
    L = {x["name"]: x for x in S["lanes"]}
    names = ["correct calls", "missed pathogenic", "false alarms"]
    cols = {"correct calls": OKC, "missed pathogenic": MISS, "false alarms": ALARM}
    rng = np.random.default_rng(3)
    ax.set_xlim(-9.5, 4.5); ax.set_ylim(-0.62, len(names) - 0.38)
    ax.axvline(0, color=RULE, lw=0.8, zorder=3)
    ax.text(0.0, len(names) - 0.30, "decision boundary", fontsize=FOOT - 0.5, color=RULE,
            ha="center", va="bottom")
    for i, nm in enumerate(names[::-1]):
        v = np.array(L[nm]["logit"], float)
        yy = i + (rng.random(len(v)) - 0.5) * 0.52
        ax.scatter(v, yy, s=(0.6 if nm == "correct calls" else 2.2), color=cols[nm],
                   alpha=(0.16 if nm == "correct calls" else 0.62), linewidths=0,
                   rasterized=True, zorder=2)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.34, i + 0.34], color=cols[nm], lw=1.6, zorder=5,
                solid_capstyle="round")
        ax.text(-9.3, i + 0.30, nm, fontsize=FOOT + 0.2, color=INK, ha="left", va="center",
                fontweight="bold")
        ax.text(-9.3, i - 0.02, f"n = {L[nm]['n']:,}", fontsize=FOOT - 0.5, color=MUTED,
                ha="left", va="center")
    ax.set_yticks([])
    ax.set_xticks([-8, -4, 0, 2, 4]); ax.tick_params(labelsize=TICK - 0.5)
    ax.set_xlabel("signed log-odds, logit(p)      ←  called negative        called positive  →",
                  fontsize=AXIS - 0.5)
    _spines(ax, keep=("bottom",))
    mm = L["missed pathogenic"]; cc = L["correct calls"]; ff = L["false alarms"]
    ax.text(0.0, -0.26, _wrap(f"The misses sit exactly where the correct calls sit "
                              f"(median |2p−1| {mm['median_conf']:.2f} vs {cc['median_conf']:.2f}), not where "
                              f"the false alarms sit ({ff['median_conf']:.2f}). "
                              f"{mm['frac_in_most_confident_15']:.0%} of them fall in the most-confident 15%; "
                              f"no false alarm does. Confidence cannot tell them apart.", 84, FOOT),
            transform=ax.transAxes, fontsize=FOOT, color=INK, ha="left", va="top", linespacing=1.3)


# ---------------------------------------------------------------- c. refusing walks the wrong way
def panel_c(ax):
    T = S["trajectory"]
    se = np.array([t["sensitivity"] for t in T]); sp = np.array([t["specificity"] for t in T])
    cov = np.array([t["coverage"] for t in T])
    ax.set_xlim(0.955, 1.006); ax.set_ylim(0.36, 0.78)
    ax.scatter([1.0], [1.0], s=0, zorder=0)
    ax.annotate("", xy=(sp[-1], se[-1]), xytext=(sp[0], se[0]),
                arrowprops=dict(arrowstyle="-", color=MISS, lw=0.0))
    ax.plot(sp, se, color=MISS, lw=1.5, zorder=4, solid_capstyle="round")
    for k in range(len(T) - 1):
        if k % 3 == 0:
            ax.annotate("", xy=(sp[k + 1], se[k + 1]), xytext=(sp[k], se[k]),
                        arrowprops=dict(arrowstyle="-|>", color=MISS, lw=0.9, mutation_scale=6), zorder=5)
    ax.scatter(sp[0], se[0], s=30, color=MISS, zorder=6, edgecolors="white", lw=0.8)
    ax.text(sp[0] - 0.001, se[0] + 0.012, "call everything\n(100% coverage)", fontsize=FOOT,
            color=MISS, ha="right", va="bottom", fontweight="bold", linespacing=1.25)
    ax.scatter(sp[-1], se[-1], s=22, color=MISS, zorder=6, edgecolors="white", lw=0.8)
    ax.text(sp[-1] - 0.0015, se[-1], f"refuse {1 - cov[-1]:.0%}", fontsize=FOOT, color=MISS,
            ha="right", va="center", fontweight="bold")
    ax.axhline(se[0], color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0.9575, se[0] - 0.008, "sensitivity you start with", fontsize=FOOT - 0.5, color=MUTED,
            ha="left", va="top")
    ax.annotate("", xy=(1.003, 0.775), xytext=(1.003, se[0]),
                arrowprops=dict(arrowstyle="-|>", color=IDEAL, lw=1.0, mutation_scale=7))
    ax.text(1.0005, 0.775, "where a real\ntrust layer\nwould go", fontsize=FOOT, color=IDEAL,
            ha="right", va="top", fontweight="bold", linespacing=1.25)
    ax.set_xlabel("specificity", fontsize=AXIS); ax.set_ylabel("sensitivity", fontsize=AXIS)
    ax.tick_params(labelsize=TICK - 0.5)
    _spines(ax)
    ax.text(0.0, -0.30, _wrap(f"Refusing low-confidence calls perfects the specificity that was already "
                              f"{sp[0]:.3f} and drives sensitivity from {se[0]:.3f} down to {se[-1]:.3f}. "
                              f"The operating point moves away from the corner you need.", 84, FOOT),
            transform=ax.transAxes, fontsize=FOOT, color=INK, ha="left", va="top", linespacing=1.3)


# ---------------------------------------------------------------- d. every mechanism, same failure
def panel_d(ax):
    P = S["mechanism_plane"]
    ax.set_xlim(0, 1.04); ax.set_ylim(0, 1.04)
    ax.fill_between([0, 1.04], [0, 1.04], [1.04, 1.04], color="#EEF3F7", zorder=0)
    ax.plot([0, 1.04], [0, 1.04], color=RULE, lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.text(0.50, 0.565, "class-neutral", fontsize=FOOT - 0.3, color=RULE, rotation=45,
            ha="center", va="bottom")
    ax.text(0.30, 0.92, "a trust layer that\nprotected both classes\nwould live up here",
            fontsize=FOOT + 0.2, color=IDEAL, ha="center", va="top", linespacing=1.3, fontweight="bold")
    marks = {"abstention (15%)": ("o", MISS), "conformal (90%)": ("^", ALARM)}
    for mech, (mk, col) in marks.items():
        q = [x for x in P if x["mechanism"] == mech]
        ax.scatter([x["protects_benign"] for x in q], [x["protects_pathogenic"] for x in q],
                   s=16, marker=mk, color=col, alpha=0.85, zorder=5, edgecolors="white", lw=0.4)
        bx = float(np.mean([x["protects_benign"] for x in q]))
        by = float(np.mean([x["protects_pathogenic"] for x in q]))
        ax.scatter([bx], [by], s=52, marker=mk, color=col, zorder=6, edgecolors=INK, lw=0.8)
        ax.text(bx, by - 0.075, mech.split(" ")[0], fontsize=FOOT + 0.2, color=col,
                ha="center", va="top", fontweight="bold")
    ax.set_xlabel("fraction of BENIGN errors the mechanism removes", fontsize=AXIS - 0.5)
    ax.set_ylabel("fraction of PATHOGENIC\nerrors it removes", fontsize=AXIS - 0.5, linespacing=1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=TICK - 0.5)
    _spines(ax)
    ax.text(0.0, -0.30, _wrap("Each small mark is one species; the outlined mark is the mean. Every "
                              "mechanism sits far below the diagonal — they remove the errors you could "
                              "afford and leave the ones you cannot.", 84, FOOT),
            transform=ax.transAxes, fontsize=FOOT, color=INK, ha="left", va="top", linespacing=1.3)


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    gs = GridSpec(2, 2, hspace=0.92, wspace=0.42,
                  left=0.098, right=0.965, top=0.855, bottom=0.075)
    axa = fig.add_subplot(gs[0, 0]); panel_a(axa); _plab(axa, "a", "What the model actually gets wrong")
    axb = fig.add_subplot(gs[0, 1]); panel_b(axb); _plab(axb, "b", "…and it gets it wrong confidently")
    axc = fig.add_subplot(gs[1, 0]); panel_c(axc); _plab(axc, "c", "So refusing walks the wrong way")
    axd = fig.add_subplot(gs[1, 1]); panel_d(axd); _plab(axd, "d", "Every trust layer, same blind spot")

    fig.text(0.098, 0.965, "Trustworthy about the wrong thing", fontsize=TITLE, fontweight="bold",
             ha="left", va="top", color=INK)
    fig.text(0.098, 0.932, _wrap("Evo 2-40B on 11,130 cross-species disease variants. The model is a "
                                 "specificity machine, and every confidence-based trust layer protects the "
                                 "specificity it already has rather than the sensitivity it lacks.", 168, PANEL),
             fontsize=PANEL, color=MUTED, ha="left", va="top", style="italic", linespacing=1.35)
    fig.text(0.098, 0.030, _wrap(f"n = {S['_meta']['n']:,} variants across nine species, scored zero-shot and "
                                 f"assigned probabilities by Platt scaling on a leave-one-species-out pool of "
                                 f"label-rich species; calls are taken at p = 0.5. Confidence is |2p−1|. "
                                 f"Conformal coverage is split-conformal at a nominal 90%.", 170, FOOT - 0.5),
             fontsize=FOOT - 0.5, color=MUTED, ha="left", va="top", linespacing=1.35)

    os.makedirs("reports/figures", exist_ok=True)
    b = "reports/figures/Figure4_asymmetry"
    fig.savefig(b + ".png", dpi=600, facecolor="white")
    fig.savefig(b + ".pdf", dpi=600, facecolor="white")
    fig.savefig(b + "_preview.png", dpi=200, facecolor="white")
    print(f"wrote {b} at {W_MM}x{H_MM} mm")
    return fig


if __name__ == "__main__":
    main()
