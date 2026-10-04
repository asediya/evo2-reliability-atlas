"""Figure 4 — THE TRUST LEDGER. A dense per-species benchmark matrix for the whole trust layer.

Nine species x fifteen measured columns, grouped into six blocks that walk the argument:
  cohort -> what the model does -> calibration error -> can confidence help -> conformal -> mechanism.
Each cell is a bar-in-cell: length is the value on that column's own scale, the number is printed, and
the fill direction encodes whether high or low is good. Column scales are never shared across columns,
so no false comparison is implied.

Margins carry what a matrix alone cannot:
  top    per-column spread across species, so a reader sees which columns actually vary
  right  sensitivity with its bootstrap interval - the column the whole figure turns on
  below  every one of the 11,130 variants on a shared signed log-odds axis, one strip per species,
         split into correct / missed-positive / false-alarm, aligned to the matrix rows

Values from reports/fig4_matrix.json (src/ccs/fig4_matrix_stats.py). Nothing typed by hand.
Run: python -m src.ccs.fig4_matrix
"""
import os, json, textwrap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

MM = 1 / 25.4
W_MM, H_MM = 180.0, 205.0
TITLE, PANEL, AXIS, TICK, FOOT = 9.5, 7.0, 6.5, 6.0, 5.5

MISS = "#A8201A"
ALARM = "#E69F00"
OKC = "#C3C8CE"
GOOD = "#0072B2"
INK = "#1A1A1A"
MUTED = "#63666A"
RULE = "#3A3A3A"
GRID = "#DCDEE1"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})

S = json.load(open("reports/fig4_matrix.json", encoding="utf-8"))
ROWS, COLS = S["rows"], S["cols"]
NR, NC = len(ROWS), len(COLS)

GROUPS = []
for c in COLS:
    if not GROUPS or GROUPS[-1][0] != c["group"]:
        GROUPS.append([c["group"], 0])
    GROUPS[-1][1] += 1


def _wrap(t, mm, pt):
    return textwrap.fill(" ".join(t.split()), width=max(20, int((mm * 72 / 25.4) / (0.52 * pt))))


def _vals(key):
    return np.array([r[key] if r[key] == r[key] else np.nan for r in ROWS], float)


def _norm(key, log=False):
    v = _vals(key)
    if log:
        v = np.log10(np.maximum(v, 1e-9))
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


# ---------------------------------------------------------------- the matrix
def panel_matrix(ax, ax_top, ax_right):
    ax.set_xlim(0, NC); ax.set_ylim(0, NR); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    # block headers + separators
    x = 0
    for name, k in GROUPS:
        ax.add_patch(Rectangle((x + 0.06, -0.92), k - 0.12, 0.52, fc="#EFF1F3", ec="none",
                               clip_on=False, zorder=1))
        ax.text(x + k / 2, -0.66, name.upper(), ha="center", va="center", fontsize=FOOT - 1.0,
                color=RULE, fontweight="bold", clip_on=False, zorder=3)
        if x:
            ax.plot([x, x], [-0.4, NR], color="#B9BEC4", lw=0.7, zorder=4, clip_on=False)
        x += k
    for j, c in enumerate(COLS):
        ax.text(j + 0.5, -0.16, c["label"], ha="center", va="bottom", fontsize=FOOT - 0.4,
                color=INK, clip_on=False, rotation=0)

    for i, r in enumerate(ROWS):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, i), NC, 1, fc="#F7F8F9", ec="none", zorder=0))
        lab = r["species"] + (" *" if r["target_only"] else "")
        ax.text(-0.18, i + 0.5, lab, ha="right", va="center", fontsize=TICK - 0.3,
                color=(MISS if r["target_only"] else INK),
                fontweight=("bold" if r["target_only"] else "normal"), clip_on=False)

    for j, c in enumerate(COLS):
        nv = _norm(c["key"], c.get("log", False))
        for i, r in enumerate(ROWS):
            v = r[c["key"]]
            if v != v:
                ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center", fontsize=FOOT, color=MUTED)
                continue
            f = 0.0 if nv[i] != nv[i] else float(nv[i])
            if c["good"] == "high":
                col, frac = (GOOD, f)
            elif c["good"] == "low":
                col, frac = (MISS, 1 - f)
            else:
                col, frac = (OKC, f)
            ax.add_patch(Rectangle((j + 0.08, i + 0.56), (NC and 0.84) * max(frac, 0.015), 0.30,
                                   fc=col, ec="none", alpha=0.55, zorder=2))
            bold = c["key"] in ("sensitivity", "conf_path")
            ax.text(j + 0.5, i + 0.40, c["fmt"].format(v), ha="center", va="center",
                    fontsize=FOOT - 0.2, color=INK, fontweight=("bold" if bold else "normal"), zorder=3)

    # ---- top margin: how much does each column actually vary across species? ----
    ax_top.set_xlim(0, NC); ax_top.set_ylim(-0.08, 1.08)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax_top.spines[s].set_visible(False)
    for j, c in enumerate(COLS):
        nv = _norm(c["key"], c.get("log", False))
        ax_top.plot([j + 0.5, j + 0.5], [0, 1], color=GRID, lw=0.6, zorder=1)
        ax_top.scatter([j + 0.5] * NR, nv, s=3.0, color=RULE, alpha=0.55, zorder=3, linewidths=0)
    ax_top.text(-0.18, 0.5, "spread\nacross species", ha="right", va="center", fontsize=FOOT - 0.8,
                color=MUTED, linespacing=1.15, clip_on=False)

    # ---- right margin: sensitivity with its bootstrap interval ----
    ax_right.set_ylim(0, NR); ax_right.invert_yaxis()
    ax_right.set_xlim(0, 1.02)
    for i, r in enumerate(ROWS):
        ax_right.plot([r["sens_lo"], r["sens_hi"]], [i + 0.5] * 2, color=MISS, lw=1.0,
                      solid_capstyle="round", zorder=3)
        ax_right.scatter([r["sensitivity"]], [i + 0.5], s=13, color=MISS, zorder=4,
                         edgecolors="white", lw=0.5)
        ax_right.scatter([r["specificity"]], [i + 0.5], s=13, color=OKC, marker="s", zorder=4,
                         edgecolors="white", lw=0.5)
    ax_right.axvline(1.0, color=GRID, lw=0.6, zorder=1)
    ax_right.set_yticks([])
    ax_right.set_xticks([0, 0.5, 1.0]); ax_right.tick_params(labelsize=FOOT - 0.5)
    ax_right.set_xlabel("rate", fontsize=FOOT - 0.2, labelpad=1)
    for s in ("top", "right", "left"):
        ax_right.spines[s].set_visible(False)
    ax_right.spines["bottom"].set_color("#9A9A9A")
    ax_right.text(0.5, -0.9, "sensitivity ● vs specificity ■", ha="center", va="bottom",
                  fontsize=FOOT - 0.4, color=INK, fontweight="bold", clip_on=False)


# ---------------------------------------------------------------- per-variant strips
def panel_strips(fig, gs_slice):
    inner = gs_slice.subgridspec(NR, 1, hspace=0.0)
    LO, HI = -9.5, 4.5
    rng = np.random.default_rng(5)
    for i, r in enumerate(ROWS):
        ax = fig.add_subplot(inner[i, 0])
        ax.set_xlim(LO, HI); ax.set_ylim(-1, 1)
        t = r["refuse_logit_abs"]
        ax.axvspan(-t, t, color="#EFE9E7", lw=0, zorder=0)
        ax.axvline(0, color=RULE, lw=0.5, zorder=2)
        for key, col, sz, al in [("logit_ok", OKC, 0.45, 0.30),
                                 ("logit_miss", MISS, 1.7, 0.75),
                                 ("logit_fa", ALARM, 1.7, 0.85)]:
            v = np.array(r[key], float)
            if len(v):
                ax.scatter(v, (rng.random(len(v)) - 0.5) * 1.5, s=sz, color=col, alpha=al,
                           linewidths=0, rasterized=True, zorder=3)
        ax.set_yticks([])
        ax.text(LO - 0.25, 0, r["species"] + (" *" if r["target_only"] else ""), ha="right",
                va="center", fontsize=FOOT - 0.4,
                color=(MISS if r["target_only"] else INK), clip_on=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if i == NR - 1:
            ax.set_xticks([-8, -4, 0, 2, 4]); ax.tick_params(labelsize=FOOT - 0.3)
            ax.spines["bottom"].set_color("#9A9A9A")
            ax.set_xlabel("signed log-odds, logit(p)      ←  called benign          "
                          "called pathogenic  →", fontsize=AXIS - 0.5, labelpad=1)
        else:
            ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        if i == 0:
            ax.text(LO + 0.2, 1.9, "every variant, one mark", fontsize=FOOT - 0.3, color=INK,
                    ha="left", va="bottom", fontweight="bold", clip_on=False)
            ax.text(1.2, 1.9, "● missed positives", fontsize=FOOT - 0.3, color=MISS,
                    ha="left", va="bottom", fontweight="bold", clip_on=False)
            ax.text(-2.2, 1.9, "● false alarm", fontsize=FOOT - 0.3, color=ALARM,
                    ha="left", va="bottom", fontweight="bold", clip_on=False)


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    gs = GridSpec(3, 2, height_ratios=[0.20, 1.34, 1.16], width_ratios=[1.0, 0.20],
                  hspace=0.30, wspace=0.045, left=0.105, right=0.965, top=0.878, bottom=0.072)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    panel_matrix(ax_m, ax_top, ax_r)
    panel_strips(fig, gs[2, 0])

    F = S["_meta"]["flow"]
    fig.text(0.105, 0.972, "The trust ledger: what every layer protects, species by species",
             fontsize=TITLE, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.105, 0.941,
             _wrap(f"Evo 2-40B on {S['_meta']['n']:,} cross-species disease variants. Discrimination is "
                   f"good (AUROC 0.82–0.95) yet at the decision threshold the model misses "
                   f"{F['pathogenic_missed']:,} of {F['pathogenic']:,} positives "
                   f"(sensitivity {F['sensitivity']:.2f}) while almost never raising a false alarm "
                   f"(specificity {F['specificity']:.2f}). Every trust layer below protects the second "
                   f"number, not the first.", 168, PANEL),
             fontsize=PANEL, color=MUTED, ha="left", va="top", style="italic", linespacing=1.35)
    fig.text(0.105, 0.030,
             _wrap("Bars run on each column's own scale — never compare lengths across columns; blue = "
                   "higher is better, red = lower is better, grey = descriptive. * marks species never "
                   "present in any calibration training pool. Probabilities come from Platt scaling on a "
                   "leave-one-species-out pool of label-rich species; calls are taken at p = 0.5; "
                   "confidence is |2p−1|; conformal coverage is split-conformal at a nominal 90%. "
                   "Sensitivity intervals are 2,000 stratified bootstrap resamples.", 170, FOOT - 0.5),
             fontsize=FOOT - 0.5, color=MUTED, ha="left", va="top", linespacing=1.35)

    os.makedirs("reports/figures", exist_ok=True)
    b = "reports/figures/Figure4_ledger"
    fig.savefig(b + ".png", dpi=600, facecolor="white")
    fig.savefig(b + ".pdf", dpi=600, facecolor="white")
    fig.savefig(b + "_preview.png", dpi=200, facecolor="white")
    print(f"wrote {b} at {W_MM}x{H_MM} mm  ({NR} species x {NC} columns + {S['_meta']['n']:,} variant marks)")
    return fig


if __name__ == "__main__":
    main()
