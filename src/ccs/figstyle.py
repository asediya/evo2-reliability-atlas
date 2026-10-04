"""Shared visual language for all 6 main figures (from reports/FIGURE_BIBLE.md).
Okabe-Ito derived, colorblind-safe. Evo2 = protagonist blue; GERP = un-heroic gray;
NT = vermillion; ESM = green; from-scratch = purple. Coding = solid fill; regulatory = open.
Null lives at the ORIGIN via distance-above-chance geometry (AUROC - 0.5)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette by job ----
EVO2 = "#0072B2"     # protagonist
GERP = "#8A8A8A"     # incumbent baseline, deliberately un-heroic
NT = "#D55E00"       # cross-family
ESM = "#009E73"      # protein specialist
SCRATCH = "#CC79A7"  # trained-from-scratch
INK = "#1A1A1A"
MUTED = "#6B6B6B"
NULLC = "#000000"
CODING_FILL = True   # coding = solid; regulatory = open

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    # Embed real TrueType. Without these matplotlib emits Type 3 (outlined) text, which is not
    # selectable/searchable and is rejected by many journals — Figure 1's PDF carried 5 Type 3
    # fonts and zero TrueType until this was set. fig2_style.py already had it; figstyle.py did not.
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.solid_capstyle": "round",
})


def dac(auroc):
    """distance above chance"""
    return auroc - 0.5


def panel_letter(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right", color=INK)


def null_axis(ax, lo=-0.08, hi=0.50, band=0.02):
    """distance-above-chance x-axis with the null at the origin + faint null band."""
    ax.axvspan(-band, band, color=NULLC, alpha=0.05, lw=0, zorder=0)
    ax.axvline(0, color=NULLC, lw=0.9, alpha=0.45, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("discrimination above chance  (AUROC − 0.5)")


def marker_for(is_coding):
    return dict(marker="o", markersize=6, markerfacecolor="none") if not is_coding else dict(marker="o", markersize=6)


def cbar_point(ax, x, y, lo, hi, color, is_coding, label=None, weight=1.0):
    """a single distance-above-chance point with a horizontal CI whisker."""
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, alpha=0.55 * weight + 0.15, zorder=3,
            solid_capstyle="round")
    mfc = color if is_coding else "white"
    ax.plot([x], [y], marker="o", ms=6.5, mfc=mfc, mec=color, mew=1.4, zorder=4)
    if label:
        ax.text(hi + 0.008, y, label, va="center", ha="left", fontsize=7, color=INK)
