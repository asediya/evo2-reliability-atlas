"""Shared visual system for Figure 3 (The Regulatory Blind Spot). Cohesive 'chance-bedrock' family:
every panel has one horizontal chance datum and encodes value as area/length/texture, never a bare axis point.
Reuses the Fig-2 silhouette loader + palette primitives so the two figures read as one paper."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import fig2_style as F2

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.8,
})

INK = F2.INK; MUTED = F2.MUTED; CAP = F2.CAP; GRID = F2.GRID
CODING = "#0072B2"     # Evo2 coding competence (blue, rising)
EQTL   = "#9AA0AA"     # regulatory eQTL: dead / flat (cool grey)
CANYON = "#E7E2DA"     # the competence gap (warm rock-grey fill)
BEDROCK = "#3A3A3A"    # the chance floor (0.5) — a physical datum
FAIL   = "#C1443E"     # ember accent for the failure / drop
GERP_C = "#6E8B7B"     # conservation (slate-green)
NT_C   = "#E69F00"     # nucleotide transformer (amber)

load_silhouette = F2.load_silhouette
_shade = F2._shade
SEA = "#3A6B8C"


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s); P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def bedrock(ax, y=0.5, x0=0.0, x1=1.0, label=True, lw=2.4, fs=6.2):
    """Draw the shared chance datum as a physical floor with a thin sub-chance shadow."""
    ax.axhline(y, color=BEDROCK, lw=lw, zorder=4, solid_capstyle="round")
    ax.axhspan(y - 0.5, y, color=BEDROCK, alpha=0.05, zorder=0)          # 'below chance' underworld
    if label:
        ax.text(x1, y - 0.006, "chance 0.5", ha="right", va="top", fontsize=fs, color=BEDROCK,
                fontweight="bold", style="italic")


def finding_title(ax, letter, text, fs=7.4):
    ax.text(-0.01, 1.045, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
    ax.text(0.04, 1.045, text, transform=ax.transAxes, fontsize=fs, fontweight="bold", va="bottom",
            ha="left", color=CODING, linespacing=1.12)
