# Module-level import of the house style, so the scale below is DERIVED, not copied.
# Same dual form src/ccs/fig3_rebuild.py uses: package import when run as -m, flat as a file.
try:
    from . import style_gb as _SG
except ImportError:
    import style_gb as _SG

"""FIGURE 6 — "Signal without a curator."

Every result in Figs 1-6 rests on a CURATED DISEASE LABEL (OMIA, ClinVar, SGE). This figure is the
evidence that does not depend on a curator, and the test of whether it is conservation in disguise.

  a  SELECTION FIELD    28,006 cattle variants, MAF x deleteriousness   [2-D density field]
  b  CONDITIONING LATTICE  the same gradient inside phyloP quintiles    [small-multiple mini-axes]
  c  LIFT PLANE          conservation-only vs combined AUROC, 9 species [2-D plane, meaningful coords]
  d  (reserved -- ClinVar, NOT YET VERIFIED, see COMPILED_RESULTS 10e)
  e  (reserved -- bat cross-clade, PILOT grade)

SKELETON BUDGET (checked against figure-skeleton-ban-list BEFORE building, not after)
Fig 5 closed four primitives that were free when it was designed: area-partition waffle (5a),
step/staircase over a count axis (5b), categorical x categorical matrix (5d), plus the ledger (5e) and
the slope gnomon (5c). What remained free: small-multiple lattice, alluvial/Sankey pooled flow,
**2-D density field**, beeswarm, network topology.
  * panel a takes the 2-D DENSITY FIELD -- the only free primitive whose unit is the VARIANT rather
    than the species, which is exactly what the ban list prescribes as the way out of the species-rail
    trap. No scatter anywhere in Figs 1-6 renders a continuous joint density.
  * panel b takes the SMALL-MULTIPLE LATTICE, faceted on conservation stratum. The lattice entry
    explicitly requires the facet variable NOT be the nine species; it is not.
  * panel c is the one per-species panel the gates allow, and only under all four constraints:
    both axes carry data (a meaningful coordinate, not a slot index), NO silhouettes (Fig 2's ink),
    NO left name margin, NO right value column, NO sort by n or by the plotted delta.

Run: python -m src.ccs.fig6_final
"""
import json
import textwrap
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

MM = 1 / 25.4
# Figure limits: 170 mm wide, 225 mm tall INCLUDING legend.
W_MM = 170.0
# Type floor: MINPT is the hard floor for every in-panel annotation. The methods block,
# the bat caveat paragraph and other prose strings sit in the manuscript legend, not on the
# canvas, and panel b's log ticks are plain log10 exponents, so no mathtext superscript is
# rendered at 70% of its base. Scaling all type up at once is avoided: it multiplies collisions.
MINPT = _SG.FLOOR   # FLOOR, not ANNOT: this module's annotation tier is dense
                    # multi-line blocks inside fixed panel reservations; at a larger size the
                    # stratum rows and the mean/CI trio outgrow them.
# DERIVED from style_gb's scale, not copied: a literal duplicate would not follow a change to
# the house scale, and nothing would report it. TITLE stays literal: style_gb has no TITLE
# rung because titles live in the manuscript legend.
TITLE = 9.5
PANEL, AXIS, TICK, FOOT = _SG.PANEL, _SG.AXIS, _SG.TICK, _SG.ANNOT

CODING = "#0072B2"
FAIL = "#A8201A"
BEDROCK = "#3A3A3A"
SUB = "#8C929B"
POS = "#E69F00"
INK = "#1A1A1A"
MUTED = _SG.MUTED

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # This was the only figure of the seven embedding DejaVuSans alongside
    # Arial, and the cause is not a missing glyph in Arial: it is matplotlib's mathtext engine,
    # which renders $10^{-4}$ in its own DejaVu font whatever the text family is. Panel b's log
    # tick labels are the only mathtext in the figure, so five spans were arriving in a second
    # family. Point mathtext at the same face as everything else.
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # Ink, axes and ticks take the house tokens (style_gb.rc): type in INK, axes and ticks in #8C8C8C.
    "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.edgecolor": "#8C8C8C", "xtick.color": "#8C8C8C", "ytick.color": "#8C8C8C",
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.labelsize": AXIS, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

J = json.load(open(ROOT / "reports" / "fig6_free.json", encoding="utf-8"))

# white -> project blue, for count density
DENS = LinearSegmentedColormap.from_list("dens", ["#FFFFFF", "#DCE9F2", "#8FC0DC", "#3E8FBE", "#0A4F77"])



def _edge(v):
    """Format a bin edge so a surviving sign is always justified by the digits shown.

    The 4/7 edge of panel a's equal-count ladder is -0.0215. At "%+.1f" that prints as "-0.0":
    a signed zero, which reads as a typesetting fault and would be the only tick in the ten plates
    carrying a sign its precision could not support. Printing an unsigned "0.0" instead would be
    worse -- it would assert the edge sits exactly at zero, which it does not -- and widening the
    whole axis to two decimals moves every right-aligned label into the rotated axis title. So
    one extra decimal is given to the single label that needs it, and only when rounding would
    otherwise destroy the sign.
    """
    return ("%+.2f" % v) if (v != 0 and abs(v) < 0.05) else ("%+.1f" % v)


def _pitched_ticks(ax, lo, hi, size=None, gap_pt=2.4, steps=(0.05, 0.10, 0.20, 0.25, 0.50)):
    """Y ticks at the finest listed step whose LABELS still clear each other on this axes.

    Tick spacing is a layout quantity, not a taste: a step is legible only if the distance it
    occupies on the page exceeds the height of the label set in it. Hard-coding the step couples
    the figure to one type size, which is exactly how six colliding label pairs appeared in
    panel b the moment the type scale was raised to the journal's measured norm. Given the axes'
    rendered height, this returns the densest step that leaves `gap_pt` of clear air.
    """
    fig = ax.figure
    fig.canvas.draw()
    h_pt = ax.get_window_extent(fig.canvas.get_renderer()).height * 72.0 / fig.dpi
    lab_pt = (size if size is not None else MINPT) * 1.15          # cap-to-descender box
    need = (lab_pt + gap_pt) / max(1e-9, h_pt) * (hi - lo)         # in data units
    step = next((s for s in steps if s >= need), steps[-1])
    first = np.ceil(lo / step) * step
    return np.round(np.arange(first, hi + step * 1e-6, step), 10)


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color("#8C8C8C")


def load_selection():
    """The 28,006 cattle variants. MAF_COL is pinned in fig6_stats for a reason: `af` (unfolded) gives
    rho -0.1145 against the published -0.0993 on `maf` (folded), and still passes a sign check."""
    sc = pl.read_parquet(ROOT / "data/processed/scores/selection_evo2_40b.parquet")
    ca = pl.read_parquet(ROOT / "data/interim/selection_candidates.parquet")
    j = sc.join(ca, on="variant_id", how="inner")
    maf = j["maf"].to_numpy().astype(float)
    dele = j["evo2_40b_neg"].to_numpy().astype(float)
    ok = np.isfinite(maf) & np.isfinite(dele) & (maf > 0)
    return maf[ok], dele[ok]


ENRICH = LinearSegmentedColormap.from_list(
    "enrich", ["#2166AC", "#8FB8D8", "#EAF0F4", "#FFFFFF", "#F5E3DE", "#D98E7C", "#A8201A"])

# ---------------------------------------------------------------- the SIGN channel
# A blue-red diverging ramp carries sign in HUE alone. Print it in greyscale, or read it with
# deuteranopia, and the two arms collapse onto each other: r = -5.5 and r = +5.3 are the same grey
# (both ~100/255 on this ramp, measured), so the two corners of panel a that carry the whole
# association become indistinguishable. The magnitude survives -- lightness is monotone away from
# the midpoint -- but the direction, which is the entire claim, does not.
#
# Sign therefore gets a redundant channel rather than a new ramp: the ramp is kept and every negative cell is
# additionally HATCHED. Hatched = observed below expected. It costs nothing in colour, it is
# invariant under any colour transform including a photocopy, and it needs no key beyond hatching
# the negative half of the colour bar, which is done in panel_a_resid.
HATCH = "//"
mpl.rcParams["hatch.linewidth"] = 0.4          # >= style_gb.LINE_MIN (0.3 pt)


def _lum(rgb):
    """WCAG relative luminance, the same test panel d of Figure 1 uses to pick text ink."""
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2])


def _hatch_ink(t):
    """Hatch colour for a cell at ramp position t.

    One fixed hatch colour cannot work: the negative arm runs from a saturated blue (luminance
    0.13) to white (0.997), so a dark hatch vanishes at one end and a white hatch at the other.
    Chosen by measured luminance instead, which is the rule this figure already uses for the
    residual numerals printed on the same cells.
    """
    return "white" if _lum(ENRICH(t)[:3]) < 0.35 else BEDROCK


def _hatch_split():
    """The ramp position at which the hatch ink flips, for hatching the colour bar in two pieces."""
    ts = np.linspace(0.0, 0.5, 501)
    for t in ts:
        if _lum(ENRICH(t)[:3]) >= 0.35:
            return float(t)
    return 0.5


def _check_under_diagonal(ax, lines, fontsize, min_pt=1.0):
    """Refuse to ship a stats block that the identity diagonal cuts through.

    The panel plots conservation+Evo 2 against conservation alone on equal square axes, so every
    stick runs from (x, x) upward: the region strictly BELOW the diagonal is empty by construction
    and is the only place this block can sit. A right-aligned line ending at axes-x 0.99 occupies
    [0.99 - w, 0.99] over [h, h + lineheight], and the diagonal y = x cuts it unless the line
    STARTS right of the diagonal at the line's TOP edge.

    It is an assertion, not a comment, because a comment goes stale the moment the type size
    moves: the next scale change fails loudly at build time instead of printing a
    label with a black line through it. Same reason fig2_split reads its own PDF back.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axw = ax.get_window_extent(r).width * 72.0 / fig.dpi
    axh = ax.get_window_extent(r).height * 72.0 / fig.dpi
    bad = []
    heights = {}
    for h, s, _c, wt in lines:
        t = fig.text(0, 0, s, fontsize=fontsize, fontweight=wt)
        e = t.get_window_extent(r)
        w_pt = e.width * 72.0 / fig.dpi
        # MEASURE THE HEIGHT, DO NOT ASSUME ONE LINE. fontsize*1.15 is the box of a SINGLE line; a
        # two-line label renders 24 pt against that 11.5 pt allowance, so its top corner can cross
        # the diagonal while a check using that allowance passes. Measured, it cannot.
        h_pt = e.height * 72.0 / fig.dpi
        heights[h] = h_pt
        t.remove()
        lh = h_pt / axh
        if not (math.isfinite(h) and math.isfinite(w_pt) and math.isfinite(h_pt)):
            bad.append(f"{s!r} at h={h}: non-finite geometry (h={h}, w={w_pt}, height={h_pt}) -- "
                       f"a position that cannot be evaluated is not a position that clears")
            continue
        margin = (0.99 - w_pt / axw - lh - h) * axw
        if margin < min_pt:
            bad.append(f"{s!r} at h={h}: clears the diagonal by only {margin:.1f} pt "
                       f"({w_pt:.1f} pt wide, {(0.99 - lh - h) * axw:.1f} pt of room at that height)")
    # The stack is set line by line at explicit heights rather than as one multi-line string, so
    # nothing but this stops two of them landing on top of each other when a line is added or a
    # height nudged. The box is 1.118 em (style_gb.LETTER_BOX_EM): va="bottom" anchors the DESCENDER
    # line, so the whole box stands above h.
    hs = sorted(h for h, _s, _c, _wt in lines if math.isfinite(h))
    for lo_h, hi_h in zip(hs, hs[1:]):
        # the LOWER line's own measured height is what must fit in the gap, for the same reason
        gap = (hi_h - lo_h) * axh - heights.get(lo_h, fontsize * 1.118)
        if gap < 0:
            bad.append(f"lines at h={lo_h} and h={hi_h} overlap by {-gap:.1f} pt")
    # RAISE, DO NOT ASSERT. `python -O` and PYTHONOPTIMIZE=1 strip the assert STATEMENT, so this
    # gate would silently become a no-op under an optimised interpreter. An explicit raise cannot
    # be optimised away.
    if bad:
        raise AssertionError(
            "stats block does not fit the triangle under the identity diagonal -- shorten a "
            "line, drop one, or lower the stack:\n  " + "\n  ".join(bad))


def _ink_on(rgba):
    """style_gb.ink_on, imported lazily -- this module is run both as ccs.fig6_final and bare."""
    try:
        from .style_gb import ink_on
    except ImportError:
        from style_gb import ink_on
    return ink_on(rgba)


def _neg_hatch(ax, R, NX, NY, v):
    """Hatch every cell whose residual is negative. Drawn under the white cell separators."""
    n = 0
    for i in range(NX):
        for k in range(NY):
            r = R[i, k]
            if not np.isfinite(r) or r >= 0:
                continue
            ax.add_patch(plt.Rectangle((i - 0.5, k - 0.5), 1.0, 1.0, fill=False, lw=0.0,
                                       hatch=HATCH, edgecolor=_hatch_ink((r + v) / (2 * v)),
                                       zorder=2.5))
            n += 1
    return n


def _resid_grid(lx, dv, nx, ny):
    """Pearson standardised residuals of a 2-D contingency table on EQUAL-COUNT bins.

    r = (O - E) / sqrt(E) with E the exact independence product (row x column / total). No density
    estimate, no bandwidth, no smoothing: every cell is a count. Under independence r is ~N(0,1), so
    |r| > 2 is the familiar cell-level significance cut and the numbers on the page mean something
    a reader already knows how to read.
    """
    xe = np.quantile(lx, np.linspace(0, 1, nx + 1))
    ye = np.quantile(dv, np.linspace(0, 1, ny + 1))
    H, _, _ = np.histogram2d(lx, dv, bins=[xe, ye])
    E = H.sum(1)[:, None] * H.sum(0)[None, :] / H.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (H - E) / np.sqrt(E)
    return H, E, R, xe, ye


def panel_a_resid(ax, ax_cb=None):
    """HERO — the dependence no curator drew, as an ASSOCIATION RASTER.

    THIS REPLACES A REJECTED CONTOUR SURFACE. The first build drew log2(observed/expected) as a filled
    contour field; the user's verdict was that it "doesn't look scientific" and that is correct --
    contouring a 10x8 table interpolates between cells that have no intermediate values, so it implies
    a continuity the data does not have, renders as a pale wash, shows no counts, and carries no
    uncertainty. See the kill list in reports/fig6_plan.md.

    What is drawn now is the same contingency table with NO smoothing: discrete cells, Pearson
    standardised residuals, and the residual printed wherever |r| > 2. That reports the STRENGTH of the
    evidence, not just its direction -- residuals reaching |r| ~ 9 say the dependence is decisively
    detectable at n = 28,006 even though the effect size (rho = -0.099) is small. Both facts are true
    and the panel now shows both.
    """
    S = J["selection"]
    maf, dele = load_selection()
    lx = np.log10(maf)
    NX, NY = 10, 7
    H, E, R, xe, ye = _resid_grid(lx, dele, NX, NY)
    v = float(np.nanmax(np.abs(R)))

    im = ax.imshow(R.T, origin="lower", cmap=ENRICH, vmin=-v, vmax=v, aspect="auto",
                   interpolation="nearest", zorder=2)
    _neg_hatch(ax, R, NX, NY, v)               # redundant sign channel; see _neg_hatch
    # hairline separators: this is a table of counts, and the cell boundaries are real
    for i in range(1, NX):
        ax.axvline(i - 0.5, color="white", lw=0.5, zorder=3)
    for k in range(1, NY):
        ax.axhline(k - 0.5, color="white", lw=0.5, zorder=3)
    for i in range(NX):
        for k in range(NY):
            if abs(R[i, k]) >= 2:
                ax.text(i, k, f"{R[i, k]:+.1f}", ha="center", va="center", fontsize=MINPT,
                        # the cell's own colour on the ENRICH ramp, normalised the same way
                        # imshow does it above (vmin=-v, vmax=v)
                        color=_ink_on(ENRICH((R[i, k] + v) / (2.0 * v))),
                        fontweight="bold", zorder=5)

    keep = [0, 2, 4, 6, 8, NX]
    ax.set_xticks([k - 0.5 for k in keep])
    # Three significant figures on every edge, in plain decimals: the lowest edge is not set in
    # exponent notation beside five decimals, and the 0.5 ceiling prints as 0.500 like its neighbours.
    ax.set_xticklabels(["%.*f" % (2 - math.floor(math.log10(10 ** xe[k])), 10 ** xe[k]) for k in keep],
                       fontsize=MINPT)
    # The first tick sits ON the axes' left edge, so a centred label hangs half its width out to
    # the left and lands on the bottom y tick label, so it is re-anchored to start there.
    ax.get_xticklabels()[0].set_ha("left")
    # The last tick sits ON the right edge; anchored the same way, it ends there.
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_yticks(np.arange(NY + 1) - 0.5)
    ax.set_yticklabels([_edge(e) for e in ye], fontsize=MINPT)
    ax.set_xlabel("minor allele frequency in cattle   (folded MAF; both axes equal-count bins)",
                  fontsize=AXIS)
    # ONE line, not two. "(variant-delta ~1 kb, equal-count bins)" moves to the
    # legend and to the x-label: rotated, two lines of that length are both
    # taller than the axes they label and wide enough to sit on the y ticks.
    ax.set_ylabel("Evo 2 deleteriousness", fontsize=AXIS)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=1.5, width=0.4)


    if ax_cb is not None:
        cb = plt.colorbar(im, cax=ax_cb, orientation="horizontal",
                          ticks=[-math.floor(v), -2, 0, 2, math.floor(v)])
        # the "(observed − expected) ⁄ √expected" gloss is legend material; the
        # key only has to name the quantity and the printing rule.
        # THE KEY STATES THE HATCH RULE BY BEING HATCHED. The negative half of the bar carries
        # the same hatch the negative cells do, so "hatched = below expected" is read off the key
        # rather than spelled out in a caption line this bar has no room for. Two pieces, split
        # where the ramp's luminance crosses the ink threshold, for the reason in _hatch_ink.
        _sp = _hatch_split()
        for _x0, _x1, _t in ((0.0, _sp, 0.0), (_sp, 0.5, 0.5)):
            cb.ax.add_patch(plt.Rectangle((_x0, 0.0), _x1 - _x0, 1.0, transform=cb.ax.transAxes,
                                          fill=False, lw=0.0, hatch=HATCH,
                                          edgecolor=_hatch_ink(_t), clip_on=False, zorder=3))
        cb.set_label("Pearson standardised residual   ·   |r| > 2 printed",
                     fontsize=MINPT, labelpad=3.0)
        cb.ax.tick_params(labelsize=MINPT, length=1.5, width=0.4, pad=1.5)
        cb.outline.set_linewidth(0.4)
    return im, v


def panel_a(ax, ax_cb=None):
    """SUPERSEDED — the rejected contour surface. Kept only so the kill list has a referent.

    WHY NOT A RAW SCATTER OR A COUNT-DENSITY FIELD. Both were built and both were rejected on the
    render. The deleteriousness score spans -7 to +8 while the whole effect is a 0.36-unit shift in the
    conditional median (0.414 at singletons -> 0.059 at major alleles, a 7x ratio but only ~5% of any
    axis that can hold the distribution). A count-density hexbin therefore draws one large blob with a
    sub-pixel tilt: it is honest and it is unreadable, and worse, an earlier draft of this panel put a
    MEAN-derived annotation ("singletons 0.772") on a MEDIAN curve, which is a different statistic.

    WHAT THIS PLOTS INSTEAD. log2(observed / expected) over a grid whose bins are QUANTILES of each
    margin, so every row and every column holds the same number of variants and the expectation is the
    exact independence product (row_sum * col_sum / total) -- no density estimation, no bandwidth, no
    area approximation. A KDE-product version on hexagonal geometry was tried and is noisy (sd 0.60,
    range -2.96..+1.69, corners wrong); the exact-marginal version is clean and is what ships.

    This is the dependence that rho = -0.099 measures, rendered so a real-but-small marginal effect is
    visible as STRUCTURE rather than as a tilt. It is drawn as a filled contour surface, not as cells:
    a celled grid would read as Fig 5d's categorical matrix, and both axes here are continuous.
    """
    S = J["selection"]
    maf, dele = load_selection()
    lx = np.log10(maf)

    # 10 x 8 not 14 x 12: at ~2,800 variants per column the coarser grid is where the corner signal is
    # stable. The finer grid renders the same corners plus a field of mid-plane speckle that is
    # sampling noise, and contourf interpolates that noise into shapes a reader will try to interpret.
    NX, NY = 10, 8
    xe = np.quantile(lx, np.linspace(0, 1, NX + 1))
    ye = np.quantile(dele, np.linspace(0, 1, NY + 1))
    H, _, _ = np.histogram2d(lx, dele, bins=[xe, ye])
    E = H.sum(1)[:, None] * H.sum(0)[None, :] / H.sum()
    L = np.log2(np.where(H > 0, H, np.nan) / E)

    # rank axes: cells are equal-count, so plot on 0..1 and label ticks with the real values
    xc = (np.arange(NX) + 0.5) / NX
    yc = (np.arange(NY) + 0.5) / NY
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    lim = float(np.nanmax(np.abs(L)))
    levels = np.linspace(-lim, lim, 25)
    cf = ax.contourf(XX, YY, L, levels=levels, cmap=ENRICH, extend="both", zorder=2)
    # kill the hairline seams contourf leaves in vector output; in matplotlib >=3.8 the ContourSet is
    # itself a Collection and no longer exposes .collections
    try:
        cf.set_edgecolor("face")
    except AttributeError:
        for c in cf.collections:
            c.set_edgecolor("face")
    ax.contour(XX, YY, L, levels=[-0.4, -0.2, 0.2, 0.4], colors="#00000055", linewidths=0.35, zorder=3)

    ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])

    def _fmt_maf(v):
        return f"{v:.4f}".rstrip("0") if v >= 0.001 else f"{v:.0e}".replace("e-0", "e−")

    xt = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(xt)
    ax.set_xticklabels([_fmt_maf(float(10 ** np.quantile(lx, t))) for t in xt])
    ax.set_yticks(xt)
    ax.set_yticklabels([f"{np.quantile(dele, t):+.1f}" for t in xt])
    ax.set_xlabel("minor allele frequency in cattle   (folded MAF, equal-count bins)", fontsize=AXIS)
    ax.set_ylabel("Evo 2 deleteriousness\n(variant-delta ~1 kb, equal-count bins)",
                  fontsize=AXIS, linespacing=1.25)
    _spines(ax)

    # ---- readability annotations. BOTH deleterious corners are named, because the two that matter
    # are the TOP ones: bottom-right is common-and-tolerated, which is also enriched and is the
    # mirror of the argument, not a counterexample. An earlier draft put "under-represented" on the
    # bottom-right, which is red -- the label contradicted the ink it sat on.
    ax.text(0.025, 0.955, "RARE  ·  DELETERIOUS", transform=ax.transAxes, fontsize=FOOT - 0.4,
            color=FAIL, fontweight="bold", ha="left", va="top")
    ax.text(0.025, 0.900, "over-represented — the signature\nof purifying selection",
            transform=ax.transAxes, fontsize=FOOT - 0.9, color=FAIL, ha="left", va="top", linespacing=1.25)
    ax.text(0.975, 0.955, "COMMON  ·  DELETERIOUS", transform=ax.transAxes, fontsize=FOOT - 0.4,
            color=CODING, fontweight="bold", ha="right", va="top")
    ax.text(0.975, 0.900, "under-represented", transform=ax.transAxes, fontsize=FOOT - 0.9,
            color=CODING, ha="right", va="top")
    ax.text(0.975, 0.045, "common · tolerated — enriched, the mirror image",
            transform=ax.transAxes, fontsize=FOOT - 1.0, color=MUTED, ha="right", va="bottom")

    if ax_cb is not None:
        cb = plt.colorbar(cf, cax=ax_cb, orientation="horizontal",
                          ticks=[-lim, -lim / 2, 0, lim / 2, lim])
        cb.ax.set_xticklabels([f"{2 ** v:.2f}×" for v in [-lim, -lim / 2, 0, lim / 2, lim]])
        cb.set_label("observed ÷ expected under independence", fontsize=FOOT - 0.4, labelpad=1.5)
        cb.ax.tick_params(labelsize=FOOT - 1.0, length=1.5, width=0.4)
        cb.outline.set_linewidth(0.4)
    return cf, lim


def panel_b(fig, gs_slice):
    """SMALL-MULTIPLE LATTICE — is the label-free gradient just conservation?

    Six complete mini-axes, each with its own frame: one pooled reference cell plus the five phyloP
    quintiles. The lattice gate requires >=6 cells and requires the facet variable NOT be the nine
    species; the facet here is conservation stratum. Cell titles sit INSIDE the cell, per the same gate.

    WHAT EACH CELL PLOTS. Within a stratum, every variant is scored against THAT STRATUM'S OWN median
    deleteriousness, so under independence P(del > median) is exactly 0.5 at every frequency BY
    CONSTRUCTION. The null is therefore exact rather than eyeballed, all six cells share one
    interpretable axis, and a small effect is legible as a departure from flat. Plotting the raw
    conditional median instead would repeat the hero's problem -- the shift is ~5% of the score range.

    THE RESULT THIS PANEL CARRIES. The gradient survives at fixed conservation in 4 of 5 quintiles, so
    it is not a conservation artifact -- but it STRENGTHENS monotonically with conservation and goes
    flat in the least-conserved quintile. That is Fig 4's regulatory blind spot, reproduced from
    population genetics instead of eQTLs, with no labels of any kind.
    """
    C = J["selection_conditioned"]
    cells = C["cells"]
    # THE QUINTILE INTERVALS ARE TABLE S6'S. The cells carry their own rho_ci from a second bootstrap,
    # which differed from Table S6's intervals in the third decimal in every stratum; the strata block
    # of the same JSON is what Table S6 prints, so the plate reads it. rho is identical in both.
    _strata = {"q%d" % st["stratum"]: st for st in C["strata"]}
    for _c in cells:
        if _c["key"] in _strata:
            assert abs(_c["rho"] - _strata[_c["key"]]["rho"]) < 1e-12, _c["key"]
    inner = gs_slice.subgridspec(2, 3, wspace=0.28, hspace=0.52)
    axes0 = None
    for k, c in enumerate(cells[:6]):
        ax = fig.add_subplot(inner[k // 3, k % 3])
        if k == 0:
            axes0 = ax
        pooled = c["key"] == "all"
        flat = (c["key"] == "q0")
        col = BEDROCK if pooled else (CODING if flat else FAIL)
        x = np.log10(np.array(c["maf"], float))
        y = np.array(c["p_above"], float)
        ax.axhline(0.5, color="#9A9A9A", lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.fill_between(x, c["lo"], c["hi"], facecolor=col, alpha=0.16, lw=0, zorder=3)
        ax.plot(x, y, color=col, lw=1.0, zorder=5)
        ax.plot(x, y, "o", ms=1.9, color=col, zorder=6)

        ax.set_xlim(x.min() - 0.15, x.max() + 0.15)
        # ylim from the DATA, not a guess: a hardcoded (0.40, 0.70) clipped the Wilson bands flat
        # against the floor in exactly the two most-conserved quintiles — the cells that carry the
        # panel's headline.
        _lo = min(min(cc["lo"]) for cc in cells[:6]); _hi = max(max(cc["hi"]) for cc in cells[:6])
        # headroom raised 0.075 -> 0.105 so the three-line stat block in the
        # top-right corner clears the curve's left shoulder at 6 pt.
        ax.set_ylim(_lo - 0.012, _hi + 0.105)
        ax.set_xticks([-4, -3, -2, -1])
        # TICK DENSITY IS SOLVED, NOT TYPED. The hard-coded [0.45, 0.50, 0.55, 0.60] put four
        # labels on an 18 mm cell: at the corrected type scale that is a 7.05 pt pitch under a
        # 7.3 pt label, so EVERY adjacent pair overlapped -- six of this figure's seven collisions.
        # The step is now chosen from the cell's own rendered height so the labels cannot touch,
        # and it widens by itself if the cell is ever made shorter. Shrinking the labels instead
        # would put them under the 6.5 pt floor.
        _yt = _pitched_ticks(ax, *ax.get_ylim())
        ax.set_yticks(_yt)
        # two decimals kept explicitly: the default formatter drops the trailing zero once the
        # step is 0.10, and "0.5" beside a panel whose whole claim is a departure from 0.50 reads
        # as a different quantity from the 0.50 the dashed null line marks.
        ax.set_yticklabels([f"{v:.2f}" for v in _yt], fontsize=MINPT)
        # tick_params BEFORE set_xticklabels: it rewrites labelsize on every tick label, so calling
        # it afterwards silently undid the size set below.
        ax.tick_params(labelsize=MINPT, length=1.5)
        # NO MATHTEXT. $10^{-4}$ renders the exponent at ~70% of the base, so
        # even a 5.9 pt tick would put "-4" on the page at 4.13 pt. The axis is
        # labelled as log10 MAF and the ticks are plain integers at full size, as in
        # Figure 5. The shared axis title under the lattice says "log10".
        ax.set_xticklabels(["−4", "−3", "−2", "−1"], fontsize=MINPT)
        if k % 3:
            ax.set_yticklabels([])
        if k // 3 == 0:
            ax.set_xticklabels([])
        _spines(ax)

        # cell title INSIDE the cell (lattice gate), typographically differentiated from panel titles
        # TOP-RIGHT, not top-left. These curves start high on the left and fall to the right, so the
        # left shoulder is exactly where the first data marker lands — it was drawn on top of its own
        # caption in the q3 cell. Top-right is the clear quadrant for a descending series.
        # "ALL QUINTILES", not "ALL VARIANTS": this cell pools the five strata, the 27,234 variants
        # phyloP reaches, not the 28,006 of panel a and Table S6's first row.
        ax.text(0.955, 0.955, ("ALL QUINTILES" if pooled else c["label"].replace("phyloP ", "phyloP ")),
                transform=ax.transAxes, fontsize=MINPT, color=col, fontweight="bold",
                ha="right", va="top")
        # Print the statistic the VERDICT is made on. The previous version printed only the endpoint
        # "drop", which ranks q0 (drawn as flat) ABOVE q1 (counted as surviving) — so the panel's own
        # numbers contradicted its own colouring. rho and its CI are what decide it.
        rc = (_strata[c["key"]]["ci"] if c["key"] in _strata else c.get("rho_ci")) or [float("nan"), float("nan")]
        # 0.955 / 0.807 / 0.659 is a 6 pt stack on an 18 mm cell: the pitch is
        # set from the type size, not guessed, so the three lines cannot touch.
        ax.text(0.955, 0.807, f"n = {c['n']:,}   ρ = {c['rho']:+.3f}", transform=ax.transAxes,
                fontsize=MINPT, color=MUTED, ha="right", va="top")
        ax.text(0.955, 0.659, f"[{rc[0]:+.3f}, {rc[1]:+.3f}]", transform=ax.transAxes,
                fontsize=MINPT, color=(CODING if flat else MUTED), ha="right", va="top",
                fontweight=("bold" if flat else "normal"))
        # The q0 gloss "CI spans zero — no gradient detectable" is dropped from
        # the canvas: the interval printed directly above it already spans zero
        # and is set in the CODING colour, so the sentence restated the datum it
        # sat under. It is legend material.
    axes0.set_ylabel("P(deleteriousness > stratum median)", fontsize=AXIS)
    return axes0


def panel_c(ax):
    """LIFT PLANE — the same question on the labelled disease panels.

    Each species is placed at its CONSERVATION-ONLY AUROC (x) and rises to its COMBINED
    conservation+Evo2 AUROC (y). The diagonal is 'conservation alone', so the STICK HEIGHT is the lift
    the foundation model adds, and the x-position asks whether the lift depends on how strong
    conservation already is.

    GATE COMPLIANCE (this is the one per-species panel the ban list allows, and only under all four):
    both axes carry data, so species sit at a meaningful coordinate rather than a slot index; NO
    PhyloPic silhouettes (Fig 2's signature ink); no left name margin; no right value column; no sort
    by n or by the plotted delta. The mark is a segment terminating in a dot -- deliberately NOT a
    dot-with-CI-whisker (ban 4): the segment is the estimate itself, and the only interval in the panel
    is the POOLED one, drawn once as a band rather than nine times as whiskers.
    """
    D = J["decomposition"]; P = D["pooled"]
    rows = D["per_species"]
    xs = np.array([r["a_c"] for r in rows], float)
    ys = np.array([r["a_cf"] for r in rows], float)

    lo = min(xs.min(), ys.min()) - 0.03
    hi = max(xs.max(), ys.max()) + 0.175
    # The reference geometry is drawn over the DATA's range, not the axes' corners.
    # `hi` carries 0.175 of headroom that exists solely to hold the rotated species names, and
    # running the identity line and the mean ribbon corner-to-corner would send both straight through
    # that band and the names in it.
    # No text-vs-text collision check can see it, because the thing the name collides with is a
    # line. Beyond the last species these two carry no information anyway — they exist to be read
    # against the nine sticks, and the dashed 1.00 rule still spans the full width to close the top.
    grid = np.linspace(lo, max(xs.max(), ys.max()) + 0.012, 50)
    # The band must be COMMENSURABLE with the sticks. The cross-species pooled AUROC (+0.020) ranks
    # variants ACROSS species, so cross-species positive/negative pairs count — a different estimand,
    # and drawing it here would make 7 of 8 sticks overshoot a band a reader takes for their average.
    # So the band is built from the very lifts drawn, with species as the resampling unit, because the
    # claim is cross-species generality.
    # THE BAND IS THE SPECIES MEAN, the paper's primary aggregation everywhere else: the unweighted
    # mean of the nine lifts with its t interval on 8 degrees of freedom (+0.071 [+0.029, +0.113] in
    # Note S15 and the Analyses). The n-weighted mean (+0.054, species-clustered interval) lets human's
    # and dog's panels carry half the weight; it is kept in the legend, not on the plate.
    from scipy import stats as _st
    _v = np.array([r["a_cf"] - r["a_c"] for r in rows], float)
    _m = float(_v.mean())
    _hw = float(_st.t.ppf(0.975, len(_v) - 1) * _v.std(ddof=1) / np.sqrt(len(_v)))
    SM = {"mean": _m, "lo": _m - _hw, "hi": _m + _hw}
    # An AUROC cannot exceed 1, so the ribbon and its mean line stop at the 1.00 rule.
    _below1 = plt.Rectangle((lo, lo), hi - lo, 1.0 - lo, transform=ax.transData)
    _rib = ax.fill_between(grid, grid + SM["lo"], grid + SM["hi"], facecolor=POS, alpha=0.20, lw=0, zorder=2)
    _mean, = ax.plot(grid, grid + SM["mean"], color=POS, lw=0.9, ls=(0, (4, 2)), zorder=3)
    for _a in (_rib, _mean):
        _a.set_clip_path(_below1)
    ax.plot(grid, grid, color=BEDROCK, lw=1.0, zorder=4)

    for r in rows:
        x, y0, y1 = r["a_c"], r["a_c"], r["a_cf"]
        ax.plot([x, x], [y0, y1], color=FAIL, lw=1.0, alpha=0.85, zorder=5, solid_capstyle="round")
        ax.plot([x], [y1], "o", ms=2.6, color=FAIL, zorder=6)

    # Species names pushed apart along x by a 1-D solver, then leadered back to their own dot. Five of
    # the nine sit within 0.045 AUROC of each other (cat/goat/sheep/human/dog), so hand-placing collides
    # -- the first render printed "goat" over "cat" as an unreadable overstrike.
    lab = sorted(rows, key=lambda r: r["a_c"])
    lx_ = np.array([r["a_c"] for r in lab], float)
    span = hi - lo
    # The minimum separation is the ROTATED label's own width plus clear air, measured on this
    # axes -- not the hand-set 0.052 of the range, which was solved once against a 56 mm box and
    # silently became too small when the box or the type size changed. Nine names at 6.5 pt need
    # 9.6 pt each; the solver below spreads them to exactly that and leaders each one back.
    fig_ = ax.figure
    fig_.canvas.draw()
    _axw_pt = ax.get_window_extent(fig_.canvas.get_renderer()).width * 72.0 / fig_.dpi
    minsep = max(span * 0.052, span * (MINPT * 1.15 + 2.3) / max(1e-9, _axw_pt))
    for _ in range(400):
        moved = False
        for i in range(len(lx_) - 1):
            gap = lx_[i + 1] - lx_[i]
            if gap < minsep:
                shift = (minsep - gap) / 2
                lx_[i] -= shift; lx_[i + 1] += shift
                moved = True
        lx_ = np.clip(lx_, lo + span * 0.02, hi - span * 0.02)
        if not moved:
            break
    y_lab = hi - span * 0.215
    for r, xl in zip(lab, lx_):
        ax.plot([r["a_c"], xl], [r["a_cf"] + span * 0.012, y_lab - span * 0.012],
                color="#B9B4AD", lw=0.35, zorder=4)
        ax.text(xl, y_lab, r["sp"], fontsize=MINPT, color=INK, ha="center", va="bottom",
                rotation=90)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # The rule alone, with no label: it lands exactly on the existing 1.00 y tick, so the tick
    # names it. Every text placement tried in this band collided with the rotated species names
    # or their leaders, which own the space above 1.0.
    ax.axhline(1.0, color=BEDROCK, lw=0.5, ls=(0, (2, 2)), zorder=1)
    # explicit ticks: the auto locator emitted 1.05, whose label ran 0.5 mm off the page edge.
    # x steps by 0.10, y by 0.05. Seven "0.xx" labels are 88.9 pt of glyph on a 37 mm axis,
    # so at 0.05 they would overlap by
    # 1.2 pt each and the extractor would return them as ONE 28-character span,
    # "0.700.750.800.850.900.951.00" -- which a Text-vs-Text collision audit scores as clean,
    # because it is a single object. The y axis is unaffected: its labels stack, they do not abut.
    tk = np.arange(0.70, 1.001, 0.05)
    ax.set_xticks(np.arange(0.70, 1.001, 0.10)); ax.set_yticks(tk)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("conservation alone — AUROC", fontsize=AXIS)
    ax.set_ylabel("conservation + Evo 2\nAUROC", fontsize=AXIS, linespacing=1.2)
    ax.tick_params(labelsize=MINPT)
    _spines(ax)
    # the lower-right, below the diagonal, is the only region with no sticks and no label leaders --
    # in the upper-left the leaders would run straight through this text.
    # Everything here must fit UNDER the diagonal, which on square axes means a right-aligned line at
    # height h may not start left of x = h. Short lines, stacked low.
    # THE UNDER-DIAGONAL RULE IS MEASURED, NOT STATED: _check_under_diagonal measures every line
    # on the rendered axes, so a change of type size or box fails the build instead of shipping
    # a label that the diagonal runs through. Nothing else catches it:
    # no text-vs-text pair overlaps (the tick labels are the only neighbours and they are a
    # separate artist), and tools/inkunder.py averages edge density over the whole span box, so three
    # thin strokes across a wide label dilute below its 10% bar. "by species" (22 pt narrower than
    # "species-clustered", and still the resampling unit) and the four-line split below make it fit.
    # The budget is arithmetic, not taste: on square axes the diagonal runs corner to corner, so a
    # right-aligned line whose top edge sits at axes-y t may be at most (0.99 - t) * 104.9 pt wide.
    # Four lines therefore get 95.3 / 87.2 / 79.1 / 71.0 pt from the bottom up, and the block has to
    # be ordered WIDEST AT THE BOTTOM -- the triangle is the constraint, so the stack is a staircase.
    # No three-line arrangement carries all four claims: "n-weighted mean +0.054" is 85.7 pt and the
    # top slot of a three-line stack is 79.1. Splitting the key off the count is what makes it fit,
    # and the reading order still runs encoding -> count -> estimate -> interval.
    _lines = [(0.236, "stick = Evo 2's lift", MUTED, "normal"),
              (0.159, f"{D['n_positive']}/{D['n_species']} species positive", MUTED, "normal"),
              (0.082, f"species mean {SM['mean']:+.3f}", "#8A6A00", "bold"),
              (0.005, f"[{SM['lo']:+.3f}, {SM['hi']:+.3f}], t on {len(_v) - 1} df", "#8A6A00", "normal")]
    for _h, _s, _c, _wt in _lines:
        ax.text(0.99, _h, _s, transform=ax.transAxes, fontsize=MINPT, color=_c,
                fontweight=_wt, ha="right", va="bottom")
    _check_under_diagonal(ax, _lines, MINPT)


def panel_e(ax):
    """SUPERSEDED — the contour version of the bat replication.

    Deliberately the SAME encoding as panel a. A replication panel is supposed to look like the thing it
    replicates: the reader's question is "does the anti-diagonal appear again?", and any other idiom
    would make that comparison harder rather than easier. It is drawn at a quarter of the hero's size and
    subordinated to it, so it reads as a repeat measurement, not a second finding.

    PILOT GRADE, and the panel says so. Ten Myotis lucifugus samples cap AN at 20, so the frequency axis
    carries only 32 distinct values and collapses to ~4 usable bins -- a property of the panel, not of
    the scoring, and not fixable with more compute.
    """
    B = J["bat"]
    sc = pl.read_parquet(ROOT / "data/processed/scores/bat_evo2_40b.parquet")
    ca = pl.read_parquet(ROOT / "data/interim/bat_candidates.parquet")
    j = sc.join(ca, on="variant_id", how="inner")
    x = j["maf"].to_numpy().astype(float)
    d = j["evo2_40b_neg"].to_numpy().astype(float)
    ok = np.isfinite(x) & np.isfinite(d) & (x > 0)
    lx, dd = np.log10(x[ok]), d[ok]

    xe = np.unique(np.quantile(lx, np.linspace(0, 1, 6)))
    ye = np.unique(np.quantile(dd, np.linspace(0, 1, 9)))
    H, _, _ = np.histogram2d(lx, dd, bins=[xe, ye])
    E = H.sum(1)[:, None] * H.sum(0)[None, :] / H.sum()
    L = np.log2(np.where(H > 0, H, np.nan) / E)
    nx, ny = H.shape
    xc = (np.arange(nx) + 0.5) / nx
    yc = (np.arange(ny) + 0.5) / ny
    XX, YY = np.meshgrid(xc, yc, indexing="ij")
    lim = float(np.nanmax(np.abs(L)))
    cf = ax.contourf(XX, YY, L, levels=np.linspace(-lim, lim, 21), cmap=ENRICH, extend="both", zorder=2)
    try:
        cf.set_edgecolor("face")
    except AttributeError:
        for c in cf.collections:
            c.set_edgecolor("face")

    ax.set_xlim(xc[0], xc[-1]); ax.set_ylim(yc[0], yc[-1])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels([f"{10 ** np.quantile(lx, t):.2f}" for t in [0, 0.5, 1.0]], fontsize=TICK - 1.2)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([f"{np.quantile(dd, t):+.1f}" for t in [0, 0.5, 1.0]], fontsize=TICK - 1.2)
    ax.set_xlabel("minor allele frequency", fontsize=AXIS)
    ax.set_ylabel("deleteriousness", fontsize=AXIS)
    _spines(ax)
    ax.text(0.04, 0.945, "rare · deleterious", transform=ax.transAxes, fontsize=FOOT - 1.0,
            color=FAIL, fontweight="bold", ha="left", va="top")
    ax.text(0.96, 0.055, "common · tolerated", transform=ax.transAxes, fontsize=FOOT - 1.0,
            color=FAIL, ha="right", va="bottom")
    lo_, hi_ = B["spearman_ci"]
    ax.text(0.5, -0.26, f"ρ = {B['spearman_rho']:+.3f} [{lo_:+.3f}, {hi_:+.3f}]  ·  n = {B['n']:,}  ·  "
                        f"PILOT: 10 samples, AN ≤ 20",
            transform=ax.transAxes, fontsize=FOOT - 1.0, color=MUTED, ha="center", va="top")


def hero_preview():
    """Panel a alone, at the width it will ship at, for eyeball review before the full compose."""
    H = 118.0
    fig = plt.figure(figsize=(W_MM * MM, H * MM))
    # explicit placement: the gridspec version left ~15 mm of dead paper between the x-label and the
    # colour key, and the key's own label landed on the methods paragraph
    ax = fig.add_axes([0.112, 0.295, 0.860, 0.520])
    cax = fig.add_axes([0.345, 0.185, 0.400, 0.016])
    _cf, lim = panel_a(ax, cax)

    S = J["selection"]; lo, hi = S["spearman_ci"]
    fig.text(0.112, 0.030, textwrap.fill(
        f"ρ = {S['spearman_rho']:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {S['spearman_p']:.0e}, "
        f"n = {S['n']:,}, cattle ARS-UCD1.2, scored with the block-streaming variant-delta harness at ~1 kb — "
        f"NOT the 8192 bp mean-LL readout behind the atlas (Fig 2), which is worth +0.071 on its own. "
        f"The dependence is real but small: at most {2 ** lim:.2f}× over or under the independence "
        f"expectation. Normalising is what makes it visible — on the raw score axis the conditional "
        f"median moves 0.414 → 0.059, about 5% of the range the distribution spans.", width=150),
        fontsize=FOOT - 0.7, color=MUTED, ha="left", va="bottom", linespacing=1.35)

    fig.text(0.112, 0.848, "a", fontsize=PANEL, fontweight="bold", ha="center", va="bottom",
             color=INK)
    fig.text(0.972, 0.850, f"cattle · ARS-UCD1.2 · n = {J['selection']['n']:,}",
             fontsize=FOOT - 0.4, color=MUTED, ha="right", va="bottom")

    os.makedirs(ROOT / "reports" / "figures", exist_ok=True)
    p = str(ROOT / "reports" / "figures" / "_fig6_hero_preview")
    fig.savefig(p + ".png", dpi=600)
    return fig


def bc_preview():
    """Panels b and c alone, at shipping width, for review before the five-panel compose."""
    H = 104.0
    fig = plt.figure(figsize=(W_MM * MM, H * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.34,
                          left=0.088, right=0.975, top=0.845, bottom=0.190)

    axb = panel_b(fig, gs[0, 0])
    fig.text(0.088, 0.895, "b", fontsize=PANEL, fontweight="bold", ha="center", va="bottom",
             color=INK)

    C = J["selection_conditioned"]
    axc = fig.add_subplot(gs[0, 1]); panel_c(axc)
    fig.text(0.638, 0.895, "c", fontsize=PANEL, fontweight="bold", ha="center", va="bottom",
             color=INK)

    fig.text(0.088, 0.030, textwrap.fill(
        f"b: cattle, {C['n_scored']:,} of {C['n_total']:,} variants carry a phyloP value "
        f"({C['reach']:.1%} reach; missingness is NaN, not null). Within a stratum every variant is "
        f"scored against that stratum's own median, so 0.5 is an exact null. Bands are Wilson 95%. "
        f"The gradient survives at fixed conservation in {C['n_strata_ci_excludes_zero']} of "
        f"{len(C['strata'])} quintiles — it is NOT a conservation artifact — but it strengthens "
        f"monotonically with conservation and its CI crosses zero in the least-conserved quintile. "
        f"That is the same competence boundary, reached from population genetics instead of eQTLs, "
        f"with no labels. cattle phyloP is the one genuine phyloP track (the "
        f"pig/sheep/horse/dog tracks are GERP copies). "
        f"c: 5-fold CV logistic on within-species z-scored scores, n = "
        f"{J['decomposition']['pooled']['n']:,} over "
        f"{J['decomposition']['pooled']['n_species_pooled']} species; the pooled lift is given "
        f"here with a CI.", width=176),
        fontsize=FOOT - 0.9, color=MUTED, ha="left", va="bottom", linespacing=1.32)

    os.makedirs(ROOT / "reports" / "figures", exist_ok=True)
    fig.savefig(str(ROOT / "reports" / "figures" / "_fig6_bc_preview.png"), dpi=600)
    return fig


def main():
    """The composed figure. A ClinVar panel is deliberately ABSENT: its
    'abstention concentrates in noncoding' headline rests on a stratum that is 0.3% of the panel with
    81 negatives at 94.6% positive, so the miscalibration there is a base-rate artifact. The defensible
    ClinVar result is a labelled-benchmark generality claim and belongs with the trust layer, not in a
    figure whose subject is evidence that needs no curator. See COMPILED_RESULTS 10e."""
    # The mm budget below is authored from the ORIGINAL top edge, which carried a title at
    # 4 mm and a standfirst at 10-20 mm. Both moved to the legend (house style), so the whole
    # stack lifts by HEAD_MM and the canvas loses the same amount: every panel keeps its
    # authored size in millimetres, and the plate no longer opens with 20 mm of white paper.
    HEAD_MM = 16.0
    # Canvas height (src/ccs/style_gb.py): the journal's 225 mm ceiling covers the figure AND its
    # legend, and a 300-word legend sets to 25-35 mm, so a full-width graphic has to stop near
    # 195 mm. A taller plate leaves the typesetter too little room for the legend, and
    # production would scale the whole figure down -- taking every label with it.
    H_MM = 173.0
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    # Budget solved in MILLIMETRES of the canvas, not by eye. Bands, top edge downwards:
    #   title 4 · subtitle 10-20 · a header 24.5 · a axes 26-56 · a ticks+xlabel
    #   56-64 · key 66-74.5 · b header 78 · b lattice 79.5-127.5 · b ticks to 131.5 ·
    #   b shared xlabel 132-135 · c/d header 140.5 · c axes 142-183 (square) ·
    #   c xlabel to 185 · d axes 142-174 · d xlabel to 182
    #
    # Bands are NOT scaled uniformly with the canvas. Panel b's six lattice cells carry a stratum
    # label and an "n = ... rho = ..." line that are sized in POINTS, so shrinking the cell removes
    # gap the type still needs; the lattice therefore keeps its full 48 mm. Height is saved instead in the
    # three bands whose content is a raster or a scatter and so rescales with its box: panel a's
    # raster (30 mm), panel c's square (its species names are spread by a solver that measures
    # its own type) and panel d's lattice (32 mm).
    def T(y_mm):            # figure fraction of a y measured DOWN from the ORIGINAL top edge
        return 1.0 - (y_mm - HEAD_MM) / H_MM

    def box(x0, y_top, w_mm, h_mm):
        return [x0, T(y_top + h_mm), w_mm / W_MM, h_mm / H_MM]

    ax_a = fig.add_axes(box(0.112, 26.0, 146.2, 30.0))
    cax = fig.add_axes(box(0.330, 66.0, 73.1, 2.0))
    _im, lim = panel_a_resid(ax_a, cax)

    gs_b = fig.add_gridspec(1, 1, left=0.112, right=0.972, top=T(79.5), bottom=T(127.5))
    axb0 = panel_b(fig, gs_b[0, 0])
    # The lattice y-label belongs to the LATTICE, not to its top-left cell. On a
    # 18 mm cell a 42 mm rotated string overhangs the cell by more than its own
    # height; centred on the 44 mm band it fits with room to spare.
    axb0.set_ylabel("")
    fig.text(0.038, T(103.5), "P(deleteriousness > stratum median)", fontsize=AXIS,
             color=INK, ha="center", va="center", rotation=90)
    # panel b had NO x-axis label anywhere — on the panel COMPILED_RESULTS §10b calls the strongest
    # result in the figure. One shared label under the lattice. The ticks are now
    # plain log10 exponents, so the label names the log10 axis.
    fig.text(0.542, T(132.9), "log10 minor allele frequency in cattle   (equal-count bins)",
             fontsize=AXIS, color=INK, ha="center", va="top")

    # panel c holds aspect='equal' (the diagonal must read as 45 degrees), so its BOX must be square in
    # millimetres or matplotlib shrinks the axes inside it and leaves dead paper.
    # The species names on it are spread by a solver that measures its own type, so the square
    # box does not crowd them.
    ax_c = fig.add_axes(box(0.112, 142.0, 37.0, 37.0))
    panel_c(ax_c)
    ax_e = fig.add_axes(box(0.4000, 142.0, 32.0, 32.0))
    panel_e_lattice(ax_e)

    S = J["selection"]; C = J["selection_conditioned"]; D = J["decomposition"]
    # TITLE AND STANDFIRST MOVED TO THE MANUSCRIPT LEGEND (house style, src/ccs/style_gb.py).
    # "Signal without a curator" and the three-line italic paragraph beneath it were the
    # figure arguing its own claim on the plate. The legend makes that argument, in the
    # journal's face, and already opens "Score is associated with allele frequency without
    # curated labels, and the limits of that signal."

    for x_, y_mm, letter, title, note in (
            (None, 24.5, "a", "Evo 2 score is associated with allele frequency genome-wide, label-free",
             f"cattle · ARS-UCD1.2 · n = {S['n']:,}"),
            (None, 78.0, "b", "…and it is not conservation in disguise — except where conservation ends",
             f"phyloP quintiles · reach {C['reach']:.1%}"),
            (None, 140.5, "c", "The same, on the labelled panels", None),
            (0.4000, 140.5, "d", "A second clade — but not a matched window", None)):
        y_ = T(y_mm)
        # House style: a bold lower-case letter, no reversed-out chip, no heading. `title`
        # is kept in the table above as documentation of what each panel is; the legend
        # states it for the reader.
        # x_ is None for the left column: those letters take the SHARED margin (style_gb.
        # letter_x) rather than a figure fraction typed to this plate's own rail, which put them
        # at 50.9 pt while half the set put theirs at 5.8 pt.
        from .style_gb import letter_x as _letter_x
        _lx, _ha = (_letter_x(fig), "left") if x_ is None else (x_, "center")
        fig.text(_lx, y_, letter, fontsize=PANEL, fontweight="bold", ha=_ha, va="bottom",
                 color=INK)
        if note:
            fig.text(0.972, y_ + 0.002, note, fontsize=MINPT, color=MUTED, ha="right",
                     va="bottom")

    # NO METHODS FOOTER ON THE CANVAS: a methods paragraph is not a figure element, so its
    # prose is in the manuscript legend for Figure 6.

    os.makedirs(ROOT / "reports" / "figures", exist_ok=True)
    base = str(ROOT / "reports" / "figures" / "Figure6_curator")
    from .style_gb import tidy_minus as _tm
    _tm(fig)   # ASCII hyphen -> U+2212 in numeric labels
    fig.savefig(base + ".pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(base + ".png", dpi=600)
    return fig


# --------------------------------------------- d : the bat's observability lattice
BAT_PQ = None        # resolved at draw time by fig2_measure._deposit_table


def panel_e_lattice(ax):
    """Draws the constraint that governs the bat panel.

    A sentence such as "the frequency floor is 1/AN, 0.05 only where
    all ten are called and 0.167 at AN = 6" is the wrong instrument for a structural
    fact. Ten Myotis lucifugus samples cap the allele number at 20, so an allele frequency is not a
    continuous quantity here at all -- it is a ratio of two small integers, and most of the plane it
    appears to live on cannot be occupied.

    Rows are the allele number actually called, columns the minor allele count. Cell shade is how
    many variants sit there. The 28 hatched cells are not empty: they are IMPOSSIBLE, because a
    minor allele count cannot exceed half the allele number. Reading the lattice, 5,103 of the
    23,888 variants -- more than a fifth -- sit in the single cell AN = 20, MAC = 1, whose frequency
    is 0.05, and no variant anywhere in the panel can have a frequency below 1/20.
    """
    import pandas as pd
    from pathlib import Path as _P
    try:
        from .fig2_measure import _deposit_table
    except ImportError:
        from fig2_measure import _deposit_table
    pq = _deposit_table("fig6_bat_selection.parquet")
    b = pd.read_parquet(pq)
    ans = sorted(b.an.unique())
    macs = list(range(1, int(b.mac.max()) + 1))
    G = np.full((len(ans), len(macs)), np.nan)
    for i, an in enumerate(ans):
        for j, mc in enumerate(macs):
            if mc > an // 2:
                continue
            G[i, j] = int(((b.an == an) & (b.mac == mc)).sum())
    with np.errstate(divide="ignore"):
        L = np.log10(np.where(G > 0, G, np.nan))
    im = ax.imshow(L, aspect="auto", cmap="Greys", vmin=0, vmax=np.nanmax(L),
                   interpolation="nearest", extent=[0, len(macs), len(ans), 0], zorder=3)
    n_imp = 0
    for i, an in enumerate(ans):
        for j, mc in enumerate(macs):
            if mc > an // 2:
                n_imp += 1
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="none", edgecolor="#C6CCD2",
                                           # "xxx", not "///": panel a of this same figure uses a
                                           # "//" diagonal to mean NEGATIVE RESIDUAL, and at print
                                           # size "///" is the same texture. One mark cannot carry
                                           # two meanings on one plate, so the impossible region
                                           # takes a cross-hatch instead.
                                           lw=0, hatch="xxx", zorder=4))
            # No in-cell numerals. The bottom row's ten counts need 123 pt of advance in a 105 pt
            # row, so they overprinted into one unreadable 34-digit run and the tenth was not drawn
            # at all -- a collision the span-overlap check scores as clean because it fuses them
            # into a single span.

    ax.set_xticks(np.arange(len(macs)) + .5); ax.set_xticklabels(macs, fontsize=FOOT)
    ax.set_yticks(np.arange(len(ans)) + .5); ax.set_yticklabels([int(a) for a in ans], fontsize=FOOT)
    ax.set_xlabel("minor allele count", fontsize=AXIS)
    ax.set_ylabel("allele number called", fontsize=AXIS)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    # The three notes are anchored in AXES FRACTION, not in data units. Anchoring them at
    # heights in data units would tie a stack of POINT-sized lines to a box whose height is in
    # millimetres: when the box shrinks, the anchors close up while the type does not, and the
    # blocks print on top of each other. Axes fractions with the blocks wrapped to the width the
    # narrower lattice leaves keeps the stack solvable: 3 + 2 + 2 lines at 1.35 leading is 74 pt
    # in a 90.7 pt box, so each gap is ~8 pt whatever the box height becomes.
    top = int(np.nanmax(G))
    ti, tj = [int(v[0]) for v in np.where(G == top)]
    ax.text(1.06, 1.00, "darkest cell: allele number %d, count %d,\nholding %s variants"
            % (ans[ti], macs[tj], "{:,}".format(top)), transform=ax.transAxes, fontsize=FOOT,
            ha="left", va="top", color=INK, linespacing=1.35, clip_on=False)
    ax.text(1.06, 0.70, "so the frequency floor is 1/AN: 0.05\nwith all ten called, 0.167 at AN = 6",
            transform=ax.transAxes, fontsize=FOOT, ha="left", va="top",
            color=INK, linespacing=1.35, clip_on=False)
    ax.text(1.06, 0.40, "%d cells are not empty but impossible:\na minor allele count cannot exceed\n"
            "half the allele number" % n_imp, transform=ax.transAxes, fontsize=FOOT,
            ha="left", va="top", color=MUTED, linespacing=1.35, clip_on=False)

    # Shade key ON THE CANVAS, not in the legend. The grey ramp is this panel's only
    # quantitative encoding and it is log10(count) -- a fact no sentence states. Anchored at
    # x = 1.06 in AXES FRACTION, the same column the three notes above use (1.06 lands at PDF
    # x 288.8, exactly where those notes start), so the key travels with the box instead of
    # shrinking away from the type when the canvas height changes.
    cax_d = ax.inset_axes([1.06, -0.16, 0.90, 0.035], transform=ax.transAxes)
    cb_d = ax.figure.colorbar(im, cax=cax_d, orientation="horizontal",
                              ticks=[0.0, 1.0, 2.0, 3.0])
    cb_d.ax.set_xticklabels(["1", "10", "100", "1,000"])
    cb_d.set_label("variants per cell  (log scale)", fontsize=MINPT, labelpad=2.0)
    cb_d.ax.tick_params(labelsize=MINPT, length=1.5, width=0.4, pad=1.5)
    cb_d.outline.set_linewidth(0.4)


if __name__ == "__main__":
    main()


