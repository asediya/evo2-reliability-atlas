"""Additional file 1's Figure S2 (file Figure3_blindspot) — "The Regulatory Blind Spot", authored 1:1 at Genome Biology
double-column width.

Type scale, canvas cap, line-width floor, ink and rcParams come from style_gb rather than
being restated here, so a contract change reaches this figure without an edit.

Palette is Okabe-Ito throughout. Colour encodes series identity only:
"candidate causal eQTL" is vermillion in every panel it appears in, Mendelian/coding competence is
blue, the positive control is orange, chance is neutral ink. Never red-vs-green.

EVERY number is read from reports/fig3_stats.json (computed by src/ccs/fig3_stats.py). No float is
typed by hand in this module.

Run:  python -m src.ccs.fig3_rebuild
"""
import os
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D

# The house style module. Imported the same way as figtext: relative when this file is
# run as part of the package, flat when src/ccs is on sys.path directly.
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG

# ---------------------------------------------------------------- page
MM = 1 / 25.4
# Full page width is 170 mm, NOT 180.
# Authored at 180 the figure is silently downscaled at typesetting, shrinking
# every font ~5.6% with no author review.
#
# Height 173 mm is within style_gb.H_MM_MAX (graphic only; the rest of the journal's 225 mm
# belongs to the legend the typesetter sets beneath it).
W_MM, H_MM = SG.W_MM, 173
assert H_MM <= SG.H_MM_MAX and W_MM <= SG.W_MM

# The type scale is the contract's, not this module's. Nothing here is
# set below SG.FLOOR.
AXIS, TICK, FOOT = SG.AXIS, SG.TICK, SG.ANNOT

# ---------------------------------------------------------------- palette (Okabe-Ito)
# Colour encodes SERIES IDENTITY and nothing else. It never marks emphasis, and no discrimination
# anywhere in this figure rests on red vs green.
CODING = "#0072B2"    # Okabe-Ito blue      — Mendelian / coding competence (matches Fig 2's Evo2)
FAIL = "#D55E00"      # Okabe-Ito vermillion — the candidate causal eQTL series, in a, b and c
EQTL = "#8C929B"      # neutral grey        — non-causal control, and the non-Evo2 scorers in d
# Text-safe renditions. FAIL and EQTL name their own series in the canvas keys, and as
# TYPE on white they measured 3.87:1 and 3.13:1 -- below WCAG 4.5:1 for small text. Markers and fills keep
# the full-chroma originals; only the words that name them are darkened (style_gb.text_safe).
FAIL_T = SG.text_safe(FAIL)
EQTL_T = SG.text_safe(EQTL)
POS = "#E69F00"       # Okabe-Ito orange    — positive control only (|z|, panel d)
BEDROCK = "#3A3A3A"   # the single chance datum, drawn identically everywhere
RULE = SG.RULE        # light structure: the group separators in a and d
INK = SG.INK
MUTED = SG.MUTED

# Every line on this canvas is at or above SG.LINE_MIN (0.3 pt): 0.5 pt structure, 0.6 pt axes,
# 0.7 pt brackets and break marks, 0.9-1.0 pt data: the journal's ceiling for a line is 1 pt.
#
# SG.FAMILY is a preference order, not a mandate, and its head is Helvetica. Taking it as written
# builds this figure in Helvetica while Figures 1, 2, 5, 6 and 7 as submitted are all ArialMT --
# one figure in a different face, which is the exact defect ("seven figures looking like seven
# different papers") the style module exists to prevent. Arial is second in SG.FAMILY, so this
# stays inside the contract's own list and is pinned here rather than silently inherited.
mpl.rcParams.update(SG.rc({"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}))

S = json.load(open("reports/fig3_stats.json", encoding="utf-8"))


# ---------------------------------------------------------------- layout, all in millimetres
# Explicit mm geometry rather than a GridSpec. Panels a and d carry row labels in their left
# gutters and panels b and c do not, so a single grid either starves the labelled panels or wastes
# 20 mm on the unlabelled ones. Axes RIGHT edges align per column, which is what reads as a grid;
# left edges sit where each panel's own labels put them.
class Box:
    """An axes rectangle in mm from the page's top-left, plus the fig-fraction rect."""

    def __init__(self, x0, x1, ytop, ybot):
        self.x0, self.x1, self.ytop, self.ybot = x0, x1, ytop, ybot
        self.w, self.h = x1 - x0, ybot - ytop

    @property
    def rect(self):
        return [self.x0 / W_MM, 1 - self.ybot / H_MM, self.w / W_MM, self.h / H_MM]


# Budget, top to bottom: 5 margin + 6 letter band + A axes + 9 ticks-and-x-label + 6 gap
# + 11 letter/provenance band + A + 9 + 3 margin, so H = 2A + 49. A = 62.
# Panel a is a plane, not a ledger, and it needs the width; the two score distributions are
# panel a's own right-hand marginal.
CGAP = 4.5                     # the axis break in panel c
# Laid out for the 173 mm canvas (style_gb.H_MM_MAX: a 300-word legend
# measures 47 mm, and graphic plus legend must fit the journal's 225 mm ceiling).
# These are ABSOLUTE millimetres from the top edge, so changing the page height alone does not move
# them, and a box that outruns the page prints off the paper. A span that falls fully off the page is
# absent from PyMuPDF extraction entirely, so a gate would report overhanging labels rather than a
# truncated panel, and the PNG would still look clean.
# Height is NOT shared uniformly. Panel d is a ledger: 10 rows plus a header and a key on a
# 38 mm axis is already a 2.9 mm pitch for SG.ANNOT type, so it is not compressed.
# The two planes are compressed instead: their content is a scatter and rescales with its
# box. The distance marginal keeps its 8 mm because its two class labels are SG.ANNOT type
# stacked inside it.
BOX_AT = Box(30.0, 140.0, 13.0, 21.0)        # distance marginal
BOX_A = Box(30.0, 140.0, 23.0, 58.0)         # the scale plane
BOX_AR = Box(142.0, 162.0, 23.0, 58.0)       # score marginal
BOX_B = Box(30.0, 86.0, 76.0, 108.0)         # the matched-pair plane
BOX_CT = Box(115.0, 166.0, 76.0, (76.0 + 108.0 - CGAP) / 2)
BOX_CB = Box(115.0, 166.0, (76.0 + 108.0 + CGAP) / 2, 108.0)
BOX_D = Box(42.0, 158.0, 125.0, 163.0)       # the scorer / other-signal ledger
# 3.0 mm -> 2.05 mm: 2.05 mm IS style_gb.LETTER_X_PT (5.8 pt), the one left margin the whole set
# now uses. 88.0 is this plate's right column and stays -- that letter marks its own panel corner.
LETTER_X = (2.05, 88.0)        # panel letters align in two columns, at the panel bounding box


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color("#8C8C8C")


def _letter(fig, col, y_mm, ch):
    """The only panel marker: a bold lower-case letter and nothing else (src/ccs/style_gb.py)."""
    fig.text(LETTER_X[col] / W_MM, 1 - y_mm / H_MM, ch, fontsize=SG.PANEL, fontweight="bold",
             color=INK, ha="left", va="baseline")


def _prov(ax, box, lines):
    """A small right-aligned provenance note above the panel: readouts, n, what the bars are.

    Anchored va='bottom' so the block grows upward out of the panel and never eats plot area.
    """
    ax.text(1.0, 1.0 + 1.6 / box.h, "\n".join(lines), transform=ax.transAxes, fontsize=FOOT,
            color=MUTED, ha="right", va="bottom", linespacing=1.30)


def _chance_v(ax):
    ax.axvline(0.5, color=BEDROCK, lw=0.9, ls=(0, (4, 2)), zorder=2)


def _interval_row(ax, y, v, lo, hi, colour, marker="o", size=17):
    """One dot-and-interval row: point estimate, 95% CI bar, serifed ends."""
    ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=3)
    for e in (lo, hi):
        ax.plot([e, e], [y - 0.15, y + 0.15], color=colour, lw=0.9, zorder=3)
    ax.scatter([v], [y], s=size, color=colour, zorder=4, marker=marker,
               edgecolors="white", lw=0.6)


# ------------------------------------------------- a. the scale plane: where the labels are organised
def panel_a(ax, axtop, axright, box):
    """Every one of the 20,000 variants placed at its distance from its own eGene's lead variant.

    This arm's null is usually reported as a number. The number is not the finding; the SCALE
    MISMATCH is. Fine-mapped credible sets are organised at 10^5 bp -- a candidate causal variant
    sits a median 89,784 bp from the lead variant of its own eGene, a control 345,717 bp -- and the
    readout being tested conditions on 501 bases of left context. Only 4.7 % of the 1,840 non-lead candidate causals
    lie within 1,001 bp of their lead -- 2.7 % within 500 bp, which is the half-width the window
    actually reaches from a variant at its centre. Both are stated because "inside that window"
    was ambiguous between the two and the difference is a factor of 1.7. Drawn on the base-pair axis, "Evo 2 does not read these variants"
    stops being an AUROC and becomes a geometric fact about what was in the receptive field.

    The right-hand marginal is the score distribution by class, so the reader sees the two
    densities and the coordinate they are indexed on in one object.
    """
    import pandas as pd
    d = pd.read_parquet("reports/eqtl_pervariant.parquet")
    lead = d.loc[d.groupby("gene_id").pip.idxmax(), ["gene_id", "pos"]].rename(
        columns={"pos": "lead_pos"})
    m = d.merge(lead, on="gene_id", how="left")
    m["dist"] = (m.pos - m.lead_pos).abs()
    m = m[m.dist > 0]
    lx = np.log10(m.dist.to_numpy())
    sc = m.evo2_40b_score.to_numpy()
    y = m.label.to_numpy()
    XLO, XHI, YLO, YHI = 2.0, 6.4, -5.0, 5.0
    keep = (lx >= XLO) & (lx <= XHI) & (sc >= YLO) & (sc <= YHI)

    ax.hexbin(lx[keep & (y == 0)], sc[keep & (y == 0)], gridsize=(46, 22),
              extent=(XLO, XHI, YLO, YHI), cmap="Greys", bins="log", mincnt=1, linewidths=0,
              zorder=2)
    ax.scatter(lx[keep & (y == 1)], sc[keep & (y == 1)], s=.9, marker="o", color=FAIL, alpha=.30,
               lw=0, zorder=3)
    for w, lab in ((1001, "1,001 bp"), (8192, "8,192 bp")):
        ax.axvline(np.log10(w), color=CODING, lw=.8, ls=(0, (3, 2)), zorder=5)
        ax.text(np.log10(w) - .04, YHI - .25, lab, fontsize=FOOT, rotation=90, ha="right",
                va="top", color=CODING)
    # The denominator matters and is stated -- in the LEGEND. A variant that IS its eGene's lead sits
    # at distance zero, cannot be placed on a log axis, and is not in this panel at all; how many
    # candidate causals remain (1,840) and what share of them lies inside the 1,001-bp window (4.7 %)
    # are findings about the panel, and set on the canvas they sat across the hexbin with the
    # 1,001-bp guide line running through them. They are printed here for the legend to quote.
    n_off = int((y == 1).sum())
    inside = float((m.dist.to_numpy()[y == 1] <= 1001).mean())
    print("  panel a: %s candidate causals off their lead; %.1f%% inside 1,001 bp; %s of 20,000 "
          "variants at non-zero distance across %s eGenes"
          % ("{:,}".format(n_off), 100 * inside, "{:,}".format(len(m)),
             "{:,}".format(int(m.gene_id.nunique()))))
    ax.set_xlim(XLO, XHI); ax.set_ylim(YLO, YHI)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(["100 bp", "1 kb", "10 kb", "100 kb", "1 Mb"], fontsize=TICK)
    ax.set_xlabel("distance from the variant to its own eGene's lead credible-set variant",
                  fontsize=AXIS)
    ax.set_ylabel("Evo 2-40B score", fontsize=AXIS)
    _spines(ax)

    b = np.linspace(XLO, XHI, 70)
    med = {}
    for lab, msk, col, ls in (("candidate causal", y == 1, FAIL, "-"),
                              ("non-causal control", y == 0, EQTL, (0, (3, 1.6)))):
        h, _ = np.histogram(lx[msk & keep], bins=b, density=True)
        axtop.plot((b[:-1] + b[1:]) / 2, h, color=col, lw=1.0, ls=ls, zorder=3)
        med[lab] = float(np.median(m.dist.to_numpy()[msk]))
        axtop.axvline(np.log10(med[lab]), color=col, lw=.6, ls=(0, (1.5, 1.5)), zorder=2)
    axtop.set_xlim(XLO, XHI); axtop.set_xticks([]); axtop.set_yticks([])
    for s_ in axtop.spines.values():
        s_.set_visible(False)
    # Stacked in POINTS, not at data fractions of the marginal's height. At y = 0.99 and 0.58 of
    # the ylim the gap is 41% of the box, which is 7.1 pt once the box is 8 mm -- under the 7.6 pt
    # bbox of a 6.8 pt line, so the two class labels overlapped. An offset transform fixes the
    # leading at 8.6 pt whatever height this marginal is given.
    _t = axtop.transAxes
    _drop = ScaledTranslation(0.0, -8.6 / 72.0, axtop.figure.dpi_scale_trans)
    # The medians are formatted from the values the dotted lines are drawn at, so the label and its
    # line cannot disagree. The candidate causal median of 1,840 distances is a half-integer,
    # 89,783.5 bp, and prints rounded to the nearest base pair.
    axtop.text(0.045, 0.98, "candidate causal, median {:,.0f} bp".format(med["candidate causal"]), transform=_t,
               fontsize=FOOT, ha="left", va="top", color=FAIL_T)
    axtop.text(0.045, 0.98, "non-causal control, median {:,.0f} bp".format(med["non-causal control"]),
               transform=_t + _drop,
               fontsize=FOOT, ha="left", va="top", color=EQTL_T)

    bs = np.linspace(YLO, YHI, 60)
    for msk, col, ls in ((y == 1, FAIL, "-"), (y == 0, EQTL, (0, (3, 1.6)))):
        h, _ = np.histogram(sc[msk & keep], bins=bs, density=True)
        axright.plot(h, (bs[:-1] + bs[1:]) / 2, color=col, lw=1.0, ls=ls, zorder=3)
    axright.set_ylim(YLO, YHI); axright.set_xticks([]); axright.set_yticks([])
    for s_ in axright.spines.values():
        s_.set_visible(False)
    axright.text(0.5, -0.012, "by score",
                 transform=axright.transAxes, fontsize=FOOT, ha="center", va="top", color=MUTED)


# ------------------------------------------------- b. the matched design, drawn as matched pairs
def panel_b(ax, box):
    """The design promises a within-eGene comparison. This draws the comparison it actually made.

    Each candidate causal variant is placed at its own score against the MEDIAN score of the
    non-causal controls drawn inside the same eGene. Under the design's own logic the cloud should
    sit above the diagonal; it straddles it, and the sign test is 46.4 % -- if anything the wrong
    side. The panel also states what the pooled AUROC cannot: only 2,783 of the 5,000 candidate
    causal variants (55.7 %) ever acquired an in-gene comparator at all, so 44.3 % of the positive
    class is not in a matched pair.
    """
    import pandas as pd
    d = pd.read_parquet("reports/eqtl_pervariant.parquet")
    ctrl = d[d.label == 0].groupby("gene_id").evo2_40b_score.median()
    cau = d[d.label == 1].copy()
    cau["ctrl"] = cau.gene_id.map(ctrl)
    ok = cau.ctrl.notna()
    x = cau.evo2_40b_score.to_numpy()[ok.to_numpy()]
    yv = cau.ctrl.to_numpy()[ok.to_numpy()]
    LO, HI = -5.0, 5.0
    k = (x >= LO) & (x <= HI) & (yv >= LO) & (yv <= HI)
    ax.hexbin(x[k], yv[k], gridsize=26, extent=(LO, HI, LO, HI), cmap="Greys", bins="log",
              mincnt=1, linewidths=0, zorder=2)
    ax.plot([LO, HI], [LO, HI], color=BEDROCK, lw=.8, ls=(0, (4, 2)), zorder=4)
    # Not on the canvas: the sign-test share ("scores above its own gene's controls in
    # 46.4 % of the 2,783 matched pairs") is a finding, the Results states it, and the cloud
    # straddling the drawn diagonal shows it. The red note is kept because it is not a finding
    # about what is drawn -- it says what is ABSENT from the panel, which nothing else here can.
    # Top left, above the diagonal, where the plane holds no pair: at the bottom right the dashed
    # diagonal ran through the note's first line.
    ax.text(LO + .3, HI - .3, "%s of the %s candidate causals\nhave no in-gene comparator:\n"
            "not plotted" % ("{:,}".format(int((~ok).sum())), "{:,}".format(len(cau))),
            fontsize=FOOT, ha="left", va="top", color=FAIL_T, linespacing=1.4)
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
    ax.set_xlabel("the candidate causal variant's own score", fontsize=AXIS)
    ax.set_ylabel("median score of the controls\nin the same eGene", fontsize=AXIS,
                  linespacing=1.35)
    _spines(ax)
    _prov(ax, box, ["one hexagon = a count of matched pairs, log-scaled"])


# ---------------------------------------------------------------- d. scale, on a broken axis
def panel_d_scale(axt, axb, boxt):
    """Two sub-axes sharing the model-scale axis, each spanning the SAME 0.120 of AUROC over the
    SAME height, so the two slopes can be compared by eye without the two thirds of empty axis that
    a single 0.5-1.0 scale forced. The change over the 40x range is a bracket beside each series,
    measured against the axis, instead of the two sentences that floated in that void.
    """
    sc = S["scale"]
    models = ["1B", "7B", "40B"]
    xs = np.arange(3)
    cod = [sc["coding"][m] for m in models]
    eq = [sc["eqtl"][m]["auroc"] for m in models]
    lo = [sc["eqtl"][m]["lo"] for m in models]
    hi = [sc["eqtl"][m]["hi"] for m in models]

    SPAN = 0.120                                   # identical data range in both sub-axes
    XLO, XHI = -0.62, 2.80
    BR, BRLAB = 2.15, 2.26                         # bracket stem and its label

    def bracket(ax, a, b, colour, text):
        ax.plot([BR, BR], [a, b], color=colour, lw=0.7, zorder=5)
        for e in (a, b):
            ax.plot([BR - 0.06, BR + 0.06], [e, e], color=colour, lw=0.7, zorder=5)
        ax.text(BRLAB, (a + b) / 2, text, fontsize=FOOT, color=colour, ha="left", va="center")

    # ---- upper: whole-atlas macro, the competence reference
    ctr = (min(cod) + max(cod)) / 2
    axt.set_ylim(ctr - SPAN / 2, ctr + SPAN / 2)
    axt.plot(xs, cod, color=CODING, lw=1.0, marker="o", ms=3.4, mfc="white", mew=1.0, zorder=4)
    bracket(axt, cod[0], cod[2], CODING, f"+{cod[2] - cod[0]:.3f}")
    axt.text(XLO + 0.04, ctr + SPAN / 2 - 0.004, "whole-atlas macro", fontsize=FOOT,
             color=CODING, ha="left", va="top")
    axt.set_yticks([0.86, 0.90, 0.94])
    axt.set_xticks([])
    _spines(axt, keep=("left",))

    # ---- lower: the eQTL series, its bootstrap band, and chance
    axb.set_ylim(0.5 - SPAN / 2, 0.5 + SPAN / 2)
    axb.fill_between(xs, lo, hi, color=FAIL, alpha=0.16, lw=0, zorder=2)
    axb.plot(xs, eq, color=FAIL, lw=1.0, marker="s", ms=3.2, zorder=4)
    # bracket() colours its rule AND its number with one value; the number is type, so both take
    # the text-safe rendition and stay matched.
    bracket(axb, eq[0], eq[2], FAIL_T, f"{eq[2] - eq[0]:+.3f}")
    axb.text(XLO + 0.04, 0.5 + SPAN / 2 - 0.004, "candidate causal eQTL (band: 95% CI)", fontsize=FOOT,
             color=FAIL_T, ha="left", va="top")
    # the chance rule stops before the bracket rail. Run full width (axhline) it passes 0.05 mm
    # above the "+0.006" label, which reads as a collision at reproduction size.
    axb.plot([XLO, BR - 0.16], [0.5, 0.5], color=BEDROCK, lw=0.9, ls=(0, (4, 2)), zorder=2)
    axb.text(XLO + 0.04, 0.5015, "chance", fontsize=FOOT, color=BEDROCK, ha="left", va="bottom")
    axb.set_yticks([0.46, 0.50, 0.54])
    axb.set_xticks(xs)
    axb.set_xticklabels(models, fontsize=TICK)
    axb.set_xlabel("model scale", fontsize=AXIS)
    _spines(axb)

    for ax in (axt, axb):
        ax.set_xlim(XLO, XHI)
    # named "eQTL": the n belongs to the lower series. Set bare above a two-series panel whose
    # upper series is the whole atlas, it reads as the atlas's n, which it is not.
    _prov(axt, boxt, ["whole-atlas macro: point estimate only; eQTL n = %s/%s" % (f"{sc['eqtl']['40B']['n_pos']:,}",
                                        f"{sc['eqtl']['40B']['n_neg']:,}")])


def _break_marks(fig, x_mm, y_mms, w=1.2, h=1.7):
    """The two diagonals that say an axis is broken. Drawn in figure mm so the slope does not
    depend on which sub-axis they sit on."""
    for y in y_mms:
        fig.add_artist(Line2D([(x_mm - w / 2) / W_MM, (x_mm + w / 2) / W_MM],
                              [1 - (y + h / 2) / H_MM, 1 - (y - h / 2) / H_MM],
                              transform=fig.transFigure, color="#8C8C8C", lw=0.7,
                              clip_on=False, zorder=6))


# ------------------------------------------------------- e. one axis: every scorer, every signal
def panel_e(ax, box):
    """Two groups on one AUROC axis: the scores this paper tests, and everything else that was
    measured on the same 20,000 variants.

    The first group alone would read as "nothing separates these variants", which is not what was
    found and not what the paper says: a
    TSS-distance baseline reads 0.651 and a gene-grouped linear probe of Evo 2-1B embeddings reads
    0.519, both deposited (`eqtl_tss_baseline.json`, `eqtl_blindspot_probe.json`). Put on one axis,
    the two groups say something the first group alone cannot: the panel
    is separable, and the sequence scores are the things not separating it.

    The open blue markers are each scorer's own coding-panel reference from `coding_ref`, so every
    row in the first group carries the comparison the claim needs: what it does elsewhere, and what
    it does here.
    """
    sc = S["scorers"]
    probe = json.load(open("reports/eqtl_blindspot_probe.json", encoding="utf-8"))
    tss = json.load(open("reports/eqtl_tss_baseline.json", encoding="utf-8"))
    cref = S["coding_ref"]

    seq = [("Evo 2-40B", sc["Evo2-40B"], cref["Evo2-40B"]),
           ("Nucleotide Transformer", sc["NT-500M"], cref["NT-500M"]),
           ("GERP (absmax25)", sc["GERP:gerp_absmax25"], cref["GERP"]),
           ("GERP (exact)", sc["GERP:gerp_exact"], cref["GERP"]),
           ("GERP (mean25)", sc["GERP:gerp_mean25"], cref["GERP"])]
    seq.sort(key=lambda r: -r[1]["auroc"])
    oth = [("PIP (circular by construction)", {"auroc": sc["PIP (positive control)"]["auroc"],
                                               "lo": sc["PIP (positive control)"]["lo"],
                                               "hi": sc["PIP (positive control)"]["hi"]}, None),
           ("distance to the TSS", {"auroc": tss["auroc"], "lo": tss["ci"][0],
                                    "hi": tss["ci"][1]}, None),
           ("|z| effect size (circular)", sc["|z| effect size (positive control)"], None),
           ("Evo 2-1B probe, concat", {"auroc": probe["probe"]["concat (ref|alt|delta)"]["auroc_gene_grouped"],
                                       "lo": probe["probe"]["concat (ref|alt|delta)"]["ci"][0],
                                       "hi": probe["probe"]["concat (ref|alt|delta)"]["ci"][1]}, None),
           ("Evo 2-1B probe, delta", {"auroc": probe["probe"]["delta (alt-ref)"]["auroc_gene_grouped"],
                                      "lo": probe["probe"]["delta (alt-ref)"]["ci"][0],
                                      "hi": probe["probe"]["delta (alt-ref)"]["ci"][1]}, None)]
    oth.sort(key=lambda r: -r[1]["auroc"])

    ys_seq = list(range(len(seq) + len(oth), len(oth), -1))
    ys_oth = list(range(len(oth) - 1, -1, -1))
    rows = list(zip(ys_seq, seq)) + list(zip(ys_oth, oth))

    XLO, XDATA, XHI = 0.42, 1.02, 1.19
    for y, (lab, v, ref) in rows:
        pos = lab.startswith(("PIP", "|z|"))
        c = POS if pos else EQTL
        _interval_row(ax, y, v["auroc"], v["lo"], v["hi"], c, marker=("D" if pos else "o"),
                      size=15)
        if ref is not None:
            ax.scatter([ref], [y], s=15, marker="o", facecolor="white", edgecolor=CODING,
                       lw=0.8, zorder=4)
        # At 0.872 the value column's right edge would sit 0.32 pt from the CI column's left edge on every
        # one of the ten rows -- 15% of a space at 7.4 pt Arial (~2.06 pt), so the two columns would read
        # as one string. x here is an AXES FRACTION, not data (get_yaxis_transform), and this axes
        # is ~320 pt wide, so 0.006 of it is ~1.9 pt: enough to clear a space. Measured on the
        # rebuilt PDF, not asserted.
        ax.text(0.866, y, "%.3f" % v["auroc"], transform=ax.get_yaxis_transform(), fontsize=FOOT,
                color=INK, va="center", ha="right", clip_on=False)
        ax.text(0.998, y, "%.3f–%.3f" % (v["lo"], v["hi"]), transform=ax.get_yaxis_transform(),
                fontsize=FOOT, color=MUTED, va="center", ha="right", clip_on=False)

    ytop = len(rows) + 2.0
    gy = float(len(oth))
    # The rule sits 0.45 of a row above the gap's centre: at the centre it ran through the top of the
    # two group labels hung under it.
    ax.plot([XLO, XDATA], [gy + 0.45, gy + 0.45], color=RULE, lw=0.5, zorder=1)
    ax.text(XDATA, ys_seq[0] + 0.52, "scores tested here", fontsize=SG.FLOOR, color=MUTED,
            ha="right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.9))
    ax.text(XDATA, gy - 0.20, "other signals on the same panel", fontsize=FOOT, color=MUTED,
            ha="right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.9))
    ax.plot([XLO + 0.012], [gy - 0.20], marker="D", ms=3.4, color=POS, clip_on=False, zorder=5)
    # White plate, like the two section headers above: the chance rule at x = 0.5 runs straight
    # through this label (14.1% edge density under it, tools/inkunder.py).
    ax.text(XLO + 0.028, gy - 0.20, "circular by construction", fontsize=FOOT, color=MUTED,
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.9))
    ax.text(0.872, ytop, "AUROC", transform=ax.get_yaxis_transform(), fontsize=FOOT, color=MUTED,
            va="top", ha="right", clip_on=False)
    ax.text(0.998, ytop, "95% CI", transform=ax.get_yaxis_transform(), fontsize=FOOT,
            color=MUTED, va="top", ha="right", clip_on=False)

    _chance_v(ax)
    ax.text(0.5, ytop, " chance", fontsize=FOOT, color=BEDROCK, ha="left", va="top")
    # On the header row, with "chance", "AUROC" and "95% CI": one row lower it ran into the
    # "scores tested here" head.
    ax.scatter([0.60], [ytop - 0.45], s=15, marker="o", facecolor="white", edgecolor=CODING,
               lw=0.8, clip_on=False, zorder=6)
    ax.text(0.612, ytop - 0.45, "the same score on the coding panel", fontsize=FOOT,
            color=CODING, ha="left", va="center")

    ax.set_yticks([y for y, _ in rows])
    ax.set_yticklabels([r[0] for _, r in rows], fontsize=FOOT)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-0.75, ytop)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    _spines(ax, keep=("left", "bottom"))
    ax.spines["bottom"].set_bounds(XLO, XDATA)
    ax.spines["left"].set_bounds(-0.4, ys_seq[0] + 0.4)
    ax.set_xlabel("AUROC (bars: 95% CI)", fontsize=AXIS)
    ax.xaxis.set_label_coords(((XLO + XDATA) / 2 - XLO) / (XHI - XLO), -0.165)
    ax.tick_params(axis="y", length=0)
    gn = sorted(v["n_pos"] + v["n_neg"] for k, v in sc.items() if k.startswith("GERP:"))
    ez = sc["Evo2-40B"]
    # The comparison ("vs within-eGene non-causal control") is the legend's: at the head of this
    # line it sat under panel b's axis title and read as its second line.
    _prov(ax, box, ["sequence n = %s/%s    GERP n = %s–%s (reach)    probe gene-grouped over %s eGenes"
                    % (f"{ez['n_pos']:,}", f"{ez['n_neg']:,}", f"{gn[0]:,}", f"{gn[-1]:,}",
                       f"{probe['n_egenes']:,}")])


# ---------------------------------------------------------------- composite
def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    axat = fig.add_axes(BOX_AT.rect)
    axa = fig.add_axes(BOX_A.rect)
    axar = fig.add_axes(BOX_AR.rect)
    panel_a(axa, axat, axar, BOX_A)

    axb = fig.add_axes(BOX_B.rect); panel_b(axb, BOX_B)

    axct = fig.add_axes(BOX_CT.rect)
    axcb = fig.add_axes(BOX_CB.rect)
    panel_d_scale(axct, axcb, BOX_CT)
    _break_marks(fig, BOX_CT.x0, (BOX_CT.ybot, BOX_CB.ytop))

    axd = fig.add_axes(BOX_D.rect); panel_e(axd, BOX_D)

    fig.text((BOX_CT.x0 - 13.0) / W_MM, 1 - (BOX_CT.ytop + BOX_CB.ybot) / 2 / H_MM, "AUROC",
             fontsize=AXIS, color=INK, ha="center", va="center", rotation=90)

    _letter(fig, 0, BOX_AT.ytop - 3.0, "a")
    _letter(fig, 0, BOX_B.ytop - 3.0, "b")
    _letter(fig, 1, BOX_B.ytop - 3.0, "c")
    _letter(fig, 0, BOX_D.ytop - 3.0, "d")

    os.makedirs("reports/figures", exist_ok=True)
    base = "reports/figures/Figure3_blindspot"
    fig.savefig(base + ".png", dpi=600, facecolor="white")
    fig.savefig(base + ".pdf", dpi=600, facecolor="white", metadata={"CreationDate": None})
    print(f"wrote {base} at {W_MM}x{H_MM} mm (authored 1:1)")


if __name__ == "__main__":
    main()
