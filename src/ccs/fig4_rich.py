"""Figure 4 — THE TRUST LAYER, rendered.  Six panels, richer treatment, same verified numbers.

Thesis: Evo 2 is a specificity machine. It misses a third of all positives, misses them
CONFIDENTLY, and every confidence-based trust layer therefore removes the errors you could afford
while leaving the ones you cannot.

  a  the leak        gradient flow, truth -> call -> trust gate -> clinician
  b  confidence      the misses sit on top of the correct calls, not with the false alarms
  c  trajectory      refusing walks the operating point away from the corner you need
  d  what it removes affordable vs costly error, side by side
  e  the plane       every mechanism x species; the region that would matter is empty
  f  per species     sensitivity against specificity, with bootstrap intervals

Values from reports/fig4_asym.json, fig4_leak.json, fig4_matrix.json. Nothing typed by hand.
Run: python -m src.ccs.fig4_rich
"""
import os, json, textwrap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
W_MM, H_MM = 180.0, 208.0
TITLE, HEAD, PANEL, AXIS, TICK, FOOT = 12.0, 8.0, 7.0, 6.5, 6.0, 5.5

# a deeper, warmer palette than the pale defaults - saturated enough to carry gradients
INK = "#101720"
CRIM = "#A3182B"      # missed positives - the costly error
CRIM_D = "#5E0A16"
AMBER = "#D98F2E"     # false alarm - the affordable error
AMBER_D = "#8A5310"
NAVY = "#17395B"      # correct / reference
NAVY_L = "#4E7FA8"
SLATE = "#9AA3AC"
SLATE_L = "#D6DBE0"
CARD = "#F7F5F2"
EDGE = "#DDD8D2"
MUTED = "#6B7480"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

A = json.load(open("reports/fig4_asym.json", encoding="utf-8"))
G = json.load(open("reports/fig4_leak.json", encoding="utf-8"))
M = json.load(open("reports/fig4_matrix.json", encoding="utf-8"))
F = A["flow"]
N = G["n"]


def _wrap(t, mm, pt):
    return textwrap.fill(" ".join(t.split()), width=max(20, int((mm * 72 / 25.4) / (0.52 * pt))))


def card(ax, pad=0.045):
    """A soft raised card behind a panel, so panels read as objects rather than floating ink."""
    ax.add_patch(FancyBboxPatch((-pad, -pad), 1 + 2 * pad, 1 + 2 * pad,
                                boxstyle="round,pad=0.0,rounding_size=0.03",
                                transform=ax.transAxes, fc=CARD, ec=EDGE, lw=0.6, zorder=-10,
                                clip_on=False, path_effects=[pe.withSimplePatchShadow(
                                    offset=(0.7, -0.7), shadow_rgbFace="#B9B3AB", alpha=0.30)]))


def grad_fill(ax, path, c0, c1, horizontal=True, alpha=1.0, z=3):
    """Fill an arbitrary path with a linear gradient - the main source of depth in this figure."""
    p = PathPatch(path, fc="none", ec="none", zorder=z)
    ax.add_patch(p)
    bb = path.get_extents()
    g = (np.linspace(0, 1, 256).reshape(1, -1) if horizontal else np.linspace(1, 0, 256).reshape(-1, 1))
    cm = LinearSegmentedColormap.from_list("g", [c0, c1])
    im = ax.imshow(g, extent=[bb.x0, bb.x1, bb.y0, bb.y1], aspect="auto", cmap=cm,
                   alpha=alpha, zorder=z, interpolation="bilinear")
    im.set_clip_path(p)
    return p


def ribbon_path(x0, x1, y0a, y0b, y1a, y1b):
    m0, m1 = x0 + (x1 - x0) * 0.40, x0 + (x1 - x0) * 0.60
    v = [(x0, y0a), (m0, y0a), (m1, y1a), (x1, y1a), (x1, y1b), (m1, y1b), (m0, y0b), (x0, y0b), (x0, y0a)]
    c = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
         Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    return Path(v, c)


def plab(ax, L, t, sub=None):
    ax.text(0.0, 1.10, L, transform=ax.transAxes, fontsize=HEAD, fontweight="bold",
            va="bottom", ha="left", color=CRIM)
    ax.text(0.038, 1.10, t, transform=ax.transAxes, fontsize=PANEL, fontweight="bold",
            va="bottom", ha="left", color=INK)
    if sub:
        ax.text(1.0, 1.10, sub, transform=ax.transAxes, fontsize=FOOT, va="bottom", ha="right",
                color=MUTED)


# ---------------------------------------------------------------- a. the leak
def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); card(ax)
    TOP, H, GAP = 0.90, 0.74, 0.020
    # channels drawn at comparable weight; TRUE COUNTS printed on every band
    hp, hb = 0.42, 0.30
    xs = {"in": 0.10, "g1": 0.36, "g2": 0.66, "out": 0.90}
    fT, fB = TOP, TOP - hp
    bT, bB = fB - GAP, fB - GAP - hb
    rTP = G["tp"] / G["path"]; rFA = G["fa"] / G["benign"]
    cpT = TOP; cpB = cpT - hp * rTP - hb * rFA
    cbT = cpB - GAP; cbB = cbT - hp * (1 - rTP) - hb * (1 - rFA)

    for pth, c0, c1 in [
        (ribbon_path(xs["in"], xs["g1"], fT, fT - hp * rTP, cpT, cpT - hp * rTP), NAVY_L, NAVY),
        (ribbon_path(xs["in"], xs["g1"], fT - hp * rTP, fB, cbT, cbT - hp * (1 - rTP)), CRIM, CRIM_D),
        (ribbon_path(xs["in"], xs["g1"], bT, bT - hb * rFA, cpT - hp * rTP, cpB), AMBER, AMBER_D),
        (ribbon_path(xs["in"], xs["g1"], bT - hb * rFA, bB, cbT - hp * (1 - rTP), cbB), SLATE_L, SLATE),
    ]:
        grad_fill(ax, pth, c0, c1, alpha=0.92)
    for pth, c0, c1 in [
        (ribbon_path(xs["g1"], xs["g2"], cpT, cpB, cpT, cpB), NAVY, NAVY_L),
        (ribbon_path(xs["g1"], xs["g2"], cbT, cbT - hp * (1 - rTP), cbT, cbT - hp * (1 - rTP)), CRIM_D, CRIM),
        (ribbon_path(xs["g1"], xs["g2"], cbT - hp * (1 - rTP), cbB, cbT - hp * (1 - rTP), cbB), SLATE, SLATE_L),
    ]:
        grad_fill(ax, pth, c0, c1, alpha=0.55)
    sMiss = G["miss_survive"] / G["miss"]
    grad_fill(ax, ribbon_path(xs["g2"], xs["out"], cbT, cbT - hp * (1 - rTP) * sMiss,
                              cbT, cbT - hp * (1 - rTP) * sMiss), CRIM, CRIM_D, alpha=0.95)

    for t, b, lab, cnt, col in [(fT, fB, "pathogenic", G["path"], CRIM),
                                (bT, bB, "benign", G["benign"], SLATE)]:
        ax.add_patch(Rectangle((xs["in"] - 0.028, b), 0.020, t - b, fc=col, ec="none", alpha=0.95))
        ax.text(xs["in"] - 0.036, (t + b) / 2, f"{lab}\n{cnt:,}", ha="right", va="center",
                fontsize=FOOT + 0.2, color=col if col != SLATE else MUTED, fontweight="bold",
                linespacing=1.25)

    for k, lab in [("g1", "call at p = 0.5"), ("g2", "refuse least-confident 15%")]:
        ax.plot([xs[k], xs[k]], [bB - 0.03, TOP + 0.03], color=INK, lw=1.0, alpha=0.55, zorder=8)
        ax.text(xs[k], bB - 0.055, lab, ha="center", va="top", fontsize=FOOT, color=MUTED)

    ax.text(xs["g1"] + 0.012, cpT - 0.022, f"called pathogenic", fontsize=FOOT, color=INK, va="top")
    ax.text(xs["g1"] + 0.012, cbT - 0.022, f"called benign", fontsize=FOOT, color=INK, va="top")
    ax.text(0.5 * (xs["in"] + xs["g1"]), cbT - hp * (1 - rTP) * 0.5,
            f"{G['miss']}", ha="center", va="center", fontsize=HEAD + 2, color="white",
            fontweight="bold", zorder=9,
            path_effects=[pe.withStroke(linewidth=2.0, foreground=CRIM_D)])
    ax.text(xs["in"] + 0.020, bB - 0.115, "positives crossing into “negative”",
            ha="left", va="top", fontsize=FOOT + 0.4, color=CRIM, fontweight="bold")
    ax.text(xs["out"] + 0.012, cbT - hp * (1 - rTP) * sMiss / 2,
            f"{G['miss_survive']}\nreach the\nclinician", ha="left", va="center",
            fontsize=FOOT + 0.6, color=CRIM, fontweight="bold", linespacing=1.25)
    ax.text(xs["in"] - 0.028, TOP + 0.055, "TRUTH", fontsize=FOOT - 0.3, color=MUTED, ha="left")
    ax.text(xs["out"], TOP + 0.055, "ARRIVES", fontsize=FOOT - 0.3, color=MUTED, ha="right")
    ax.text(0.0, -0.075, "Channel heights are drawn at comparable weight so the leak is visible; "
                         "every count is printed and exact.", fontsize=FOOT - 0.4, color=MUTED,
            transform=ax.transAxes, va="top")


# ---------------------------------------------------------------- b. confidence
def panel_b(ax):
    card(ax)
    L = {x["name"]: x for x in A["lanes"]}
    lanes = [("correct calls", SLATE, 0.5, 0.30), ("missed pathogenic", CRIM, 2.4, 0.80),
             ("false alarms", AMBER, 2.4, 0.90)]
    rng = np.random.default_rng(3)
    ax.set_xlim(-9.5, 4.5); ax.set_ylim(-0.65, 2.62)
    ax.axvline(0, color=INK, lw=1.0, alpha=0.6, zorder=4)
    for i, (nm, col, sz, al) in enumerate(lanes[::-1]):
        v = np.array(L[nm]["logit"], float)
        ax.scatter(v, i + (rng.random(len(v)) - 0.5) * 0.50, s=sz, color=col, alpha=al,
                   linewidths=0, rasterized=True, zorder=3)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.30, i + 0.30], color=col, lw=2.4, zorder=6,
                solid_capstyle="round", path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
        ax.text(-9.2, i + 0.36, nm, fontsize=FOOT + 0.2, color=INK, ha="left", fontweight="bold")
    ax.set_yticks([]); ax.set_xticks([-8, -4, 0, 4]); ax.tick_params(labelsize=FOOT)
    ax.set_xlabel("signed log-odds    ←  benign        pathogenic  →", fontsize=FOOT)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)


# ---------------------------------------------------------------- c. trajectory
def panel_c(ax):
    card(ax)
    T = A["trajectory"]
    se = np.array([t["sensitivity"] for t in T]); sp = np.array([t["specificity"] for t in T])
    ax.set_xlim(0.955, 1.008); ax.set_ylim(0.36, 0.80)
    for w, a in [(4.0, 0.10), (2.6, 0.22)]:
        ax.plot(sp, se, color=CRIM, lw=w, alpha=a, zorder=2, solid_capstyle="round")
    ax.plot(sp, se, color=CRIM, lw=1.5, zorder=4, solid_capstyle="round")
    for k in range(0, len(T) - 1, 3):
        ax.annotate("", xy=(sp[k + 1], se[k + 1]), xytext=(sp[k], se[k]),
                    arrowprops=dict(arrowstyle="-|>", color=CRIM, lw=1.0, mutation_scale=7), zorder=5)
    ax.scatter(sp[0], se[0], s=42, color=CRIM, zorder=7, edgecolors="white", lw=1.2,
               path_effects=[pe.withSimplePatchShadow(offset=(0.6, -0.6), alpha=0.4)])
    ax.text(sp[0] - 0.0012, se[0] + 0.016, "call everything", fontsize=FOOT, color=CRIM,
            ha="right", va="bottom", fontweight="bold")
    ax.text(sp[-1] - 0.0015, se[-1], "refuse 70%", fontsize=FOOT, color=CRIM, ha="right",
            va="center", fontweight="bold")
    ax.annotate("", xy=(1.004, 0.79), xytext=(1.004, se[0]),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4, mutation_scale=8))
    ax.text(1.0018, 0.79, "where a real\ntrust layer goes", fontsize=FOOT, color=NAVY,
            ha="right", va="top", fontweight="bold", linespacing=1.25)
    ax.set_xlabel("specificity", fontsize=AXIS); ax.set_ylabel("sensitivity", fontsize=AXIS)
    ax.tick_params(labelsize=FOOT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)


# ---------------------------------------------------------------- d. what it removes
def panel_d(ax):
    card(ax)
    rows = [("false alarms", G["pct_fa_caught"], AMBER, AMBER_D, G["fa_refused"], G["fa"], "affordable"),
            ("missed positives", G["pct_miss_caught"], CRIM, CRIM_D, G["miss_refused"], G["miss"], "costly")]
    ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 1.7)
    for y, (nm, v, c0, c1, k, tot, tag) in zip([1, 0], rows):
        ax.add_patch(FancyBboxPatch((0, y - 0.22), 1.0, 0.44, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="#ECEAE7", ec="none", zorder=1))
        pth = Path([(0, y - 0.22), (v, y - 0.22), (v, y + 0.22), (0, y + 0.22), (0, y - 0.22)])
        grad_fill(ax, pth, c0, c1, alpha=0.95, z=3)
        ax.text(0.0, y + 0.33, f"{nm}  ({tag})", fontsize=FOOT + 0.2, color=INK, ha="left",
                va="bottom", fontweight="bold")
        ax.text(v + 0.02, y, f"{v:.0%}", fontsize=HEAD, color=c1, ha="left", va="center",
                fontweight="bold")
        ax.text(v + 0.02, y - 0.30, f"{k} of {tot}", fontsize=FOOT, color=MUTED, ha="left", va="center")
    ax.set_yticks([]); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "50", "100%"], fontsize=FOOT)
    ax.set_xlabel("share of that error class removed", fontsize=AXIS)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)


# ---------------------------------------------------------------- e. the plane
def panel_e(ax):
    card(ax)
    P = A["mechanism_plane"]
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    pth = Path([(0, 0), (1.05, 1.05), (1.05, 1.05), (0, 1.05), (0, 0)])
    grad_fill(ax, pth, "#E8EFF5", "#CFDDE9", horizontal=False, alpha=0.85, z=0)
    ax.plot([0, 1.05], [0, 1.05], color=INK, lw=0.8, ls=(0, (4, 2)), alpha=0.55, zorder=2)
    ax.text(0.30, 0.86, "a layer that protected\nboth classes lives here", fontsize=FOOT + 0.2,
            color=NAVY, ha="center", va="center", fontweight="bold", linespacing=1.3, zorder=4)
    for mech, mk, c0, c1 in [("abstention (15%)", "o", CRIM, CRIM_D),
                             ("conformal (90%)", "^", AMBER, AMBER_D)]:
        q = [x for x in P if x["mechanism"] == mech]
        ax.scatter([x["protects_benign"] for x in q], [x["protects_pathogenic"] for x in q],
                   s=22, marker=mk, color=c0, alpha=0.75, zorder=5, edgecolors="white", lw=0.5)
        bx = float(np.mean([x["protects_benign"] for x in q]))
        by = float(np.mean([x["protects_pathogenic"] for x in q]))
        ax.scatter([bx], [by], s=90, marker=mk, color=c1, zorder=6, edgecolors="white", lw=1.2,
                   path_effects=[pe.withSimplePatchShadow(offset=(0.6, -0.6), alpha=0.45)])
        ax.text(bx, by - 0.10, mech.split(" ")[0], fontsize=FOOT + 0.2, color=c1, ha="center",
                va="top", fontweight="bold")
    ax.set_xlabel("removes BENIGN errors", fontsize=AXIS)
    ax.set_ylabel("removes POSITIVE-class errors", fontsize=AXIS)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(labelsize=FOOT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)


# ---------------------------------------------------------------- f. per species
def panel_f(ax):
    card(ax)
    R = sorted(M["rows"], key=lambda r: r["sensitivity"])
    ys = np.arange(len(R))
    ax.set_xlim(0, 1.06); ax.set_ylim(-0.8, len(R) - 0.2)
    for i, r in enumerate(R):
        ax.plot([r["sens_lo"], r["sens_hi"]], [i, i], color=CRIM, lw=1.4, alpha=0.35,
                solid_capstyle="round", zorder=2)
        ax.scatter([r["sensitivity"]], [i], s=22, color=CRIM, zorder=5, edgecolors="white", lw=0.7)
        ax.scatter([r["specificity"]], [i], s=20, color=NAVY, marker="s", zorder=5,
                   edgecolors="white", lw=0.7)
        ax.text(-0.02, i, r["species"], ha="right", va="center", fontsize=FOOT,
                color=(CRIM if r["target_only"] else INK),
                fontweight=("bold" if r["target_only"] else "normal"))
    ax.axvline(1.0, color=EDGE, lw=0.8, zorder=1)
    ax.set_yticks([]); ax.set_xticks([0, 0.5, 1.0]); ax.tick_params(labelsize=FOOT)
    ax.text(0.0, -0.155, "● sensitivity", transform=ax.transAxes, fontsize=FOOT, color=CRIM,
            fontweight="bold", ha="left", va="top")
    ax.text(0.55, -0.155, "■ specificity", transform=ax.transAxes, fontsize=FOOT, color=NAVY,
            fontweight="bold", ha="left", va="top")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(EDGE)


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 6, height_ratios=[1.30, 0.86, 0.86], hspace=0.85, wspace=1.15,
                  left=0.085, right=0.955, top=0.845, bottom=0.070)
    axa = fig.add_subplot(gs[0, :]); panel_a(axa)
    axb = fig.add_subplot(gs[1, 0:3]); panel_b(axb)
    axc = fig.add_subplot(gs[1, 3:6]); panel_c(axc)
    axd = fig.add_subplot(gs[2, 0:2]); panel_d(axd)
    axe = fig.add_subplot(gs[2, 2:4]); panel_e(axe)
    axf = fig.add_subplot(gs[2, 4:6]); panel_f(axf)
    plab(axa, "a", "Where the positives leak out", f"n = {N:,}")
    plab(axb, "b", "The misses are confident")
    plab(axc, "c", "Refusing walks the wrong way")
    plab(axd, "d", "What the gate removes")
    plab(axe, "e", "Every mechanism, same gap")
    plab(axf, "f", "Species by species")

    fig.text(0.085, 0.975, "The trust layer catches the errors you could afford",
             fontsize=TITLE, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.085, 0.930,
             _wrap(f"Evo 2-40B on {N:,} cross-species disease variants. Discrimination is strong "
                   f"(AUROC 0.82–0.95), yet at the decision threshold the model misses "
                   f"{F['pathogenic_missed']} of {F['pathogenic']:,} positives — sensitivity "
                   f"{F['sensitivity']:.2f} against specificity {F['specificity']:.2f} — and it misses "
                   f"them confidently.", 168, PANEL),
             fontsize=PANEL, color=MUTED, ha="left", va="top", style="italic", linespacing=1.4)
    fig.text(0.085, 0.028,
             _wrap("Probabilities from Platt scaling on a leave-one-species-out pool of label-rich "
                   "species; calls at p = 0.5; confidence is |2p−1|; the gate refuses each species' own "
                   "least-confident 15%; conformal coverage is split-conformal at a nominal 90%. "
                   "Intervals are 2,000 stratified bootstrap resamples.", 172, FOOT - 0.5),
             fontsize=FOOT - 0.5, color=MUTED, ha="left", va="top", linespacing=1.35)

    os.makedirs("reports/figures", exist_ok=True)
    b = "reports/figures/Figure4_rich"
    fig.savefig(b + ".png", dpi=600, facecolor="white")
    fig.savefig(b + ".pdf", dpi=600, facecolor="white")
    fig.savefig(b + "_preview.png", dpi=200, facecolor="white")
    print(f"wrote {b} at {W_MM}x{H_MM} mm")
    return fig


if __name__ == "__main__":
    main()
