# Module-level import of the house style, so the scale below is derived, not copied.
# Same dual form src/ccs/fig3_rebuild.py uses: package import when run as -m, flat as a file.
try:
    from . import style_gb as _SG
except ImportError:
    import style_gb as _SG

"""Figure S3 (Additional file 1) — "Score is associated with allele frequency without curated labels, and its limits."

Every other result in the paper rests on a curated or assayed label (ClinVar, OMIA, SGE, splicing assays). This figure is the
evidence that does not depend on a curator, and the test of whether it is conservation in disguise.

  a  selection field    28,006 cattle variants, MAF x deleteriousness   [2-D density field]
  b  conditioning lattice  the same gradient inside phyloP quintiles    [small-multiple mini-axes]
  c  lift plane          conservation-only vs combined AUROC, 9 species [2-D plane, meaningful coords]
  d  bat lattice      allele number x minor allele count, bat pilot [observability lattice]

  * panel a is a 2-D table of counts over MAF x deleteriousness whose unit is the variant rather
    than the species, so the plate does not repeat a species rail.
  * panel b is a small-multiple lattice faceted on conservation stratum, not on the nine species.
  * panel c is the plate's one per-species panel, under these constraints: both axes carry data (a
    meaningful coordinate, not a slot index), no silhouettes (Figure 9's ink), no left name margin,
    no right value column, no sort by n or by the plotted delta.

Run: python -m src.ccs.fig6_final
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib as mpl
# Agg is pinned, as in fig4_trust.py: the active backend sets the page size and the text layout.
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

MM = 1 / 25.4
# Figure limits: 170 mm wide, 225 mm tall including legend.
W_MM = 170.0
# Type floor: MINPT is the hard floor for every in-panel annotation. The methods block,
# the bat caveat paragraph and other prose strings sit in the manuscript legend, not on the
# canvas, and panel b's log ticks are plain log10 exponents, so no mathtext superscript is
# rendered at 70% of its base. Scaling all type up at once is avoided: it multiplies collisions.
MINPT = _SG.FLOOR   # FLOOR, not ANNOT: this module's annotation tier is dense
                    # multi-line blocks inside fixed panel reservations; at a larger size the
                    # stratum rows and the mean/CI trio outgrow them.
# Derived from style_gb's scale, not copied: a literal duplicate would not follow a change to
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
    # Without this the figure embeds DejaVuSans alongside
    # Arial, and the cause is matplotlib's mathtext engine rather than a missing glyph in Arial:
    # it renders $10^{-4}$ in its own DejaVu font whatever the text family is. No label on
    # this plate is set as mathtext (panel b's log ticks are plain text; see panel_b), so this is a
    # guard: any mathtext is pointed at the same face as everything else.
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
    """Y ticks at the finest listed step whose labels still clear each other on this axes.

    Tick spacing is a layout quantity: a step is legible only if the distance it
    occupies on the page exceeds the height of the label set in it. Hard-coding the step couples
    the figure to one type size: at the journal's measured type scale a hard-coded step puts six
    colliding label pairs in panel b. Given the axes'
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

# ---------------------------------------------------------------- the sign channel
# A blue-red diverging ramp carries sign in hue alone. Print it in greyscale, or read it with
# deuteranopia, and the two arms collapse onto each other: r = -5.5 and r = +5.3 are the same grey
# (both ~100/255 on this ramp, measured), so the two corners of panel a that carry the whole
# association become indistinguishable. The magnitude survives -- lightness is monotone away from
# the midpoint -- but the direction, which is the entire claim, does not.
#
# Sign therefore gets a redundant channel rather than a new ramp: the ramp is kept and every negative cell is
# additionally hatched. Hatched = observed below expected. It costs nothing in colour, it is
# invariant under any colour transform including a photocopy, and it needs no key beyond hatching
# the negative half of the colour bar, which is done in panel_a_resid.
HATCH = "//"
mpl.rcParams["hatch.linewidth"] = 0.4          # >= style_gb.LINE_MIN (0.3 pt)


def _lum(rgb):
    """WCAG relative luminance, the same test fig5_final.py's panel_d (Figure S5c) uses to pick text ink."""
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
    stick runs from (x, x) upward: the region strictly below the diagonal is empty by construction
    and is the only place this block can sit. A right-aligned line ending at axes-x 0.99 occupies
    [0.99 - w, 0.99] over [h, h + lineheight], and the diagonal y = x cuts it unless the line
    starts right of the diagonal at the line's top edge.

    It is an assertion because a comment goes stale the moment the type size
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
        # The height is measured, not assumed to be one line. fontsize*1.15 is the box of a single line; a
        # two-line label renders 24 pt against that 11.5 pt allowance, so its top corner can cross
        # the diagonal while a check using that allowance passes; the measured height catches it.
        h_pt = e.height * 72.0 / fig.dpi
        heights[h] = h_pt
        t.remove()
        lh = h_pt / axh
        if not (math.isfinite(h) and math.isfinite(w_pt) and math.isfinite(h_pt)):
            bad.append(f"{s!r} at h={h}: non-finite geometry (h={h}, w={w_pt}, height={h_pt}) -- "
                       f"a position that cannot be evaluated is counted as not clearing")
            continue
        margin = (0.99 - w_pt / axw - lh - h) * axw
        if margin < min_pt:
            bad.append(f"{s!r} at h={h}: clears the diagonal by only {margin:.1f} pt "
                       f"({w_pt:.1f} pt wide, {(0.99 - lh - h) * axw:.1f} pt of room at that height)")
    # The stack is set line by line at explicit heights rather than as one multi-line string, so
    # nothing but this stops two of them landing on top of each other when a line is added or a
    # height nudged. The box is 1.118 em (style_gb.LETTER_BOX_EM): va="bottom" anchors the descender
    # line, so the whole box stands above h.
    hs = sorted(h for h, _s, _c, _wt in lines if math.isfinite(h))
    for lo_h, hi_h in zip(hs, hs[1:]):
        # the lower line's own measured height is what must fit in the gap, for the same reason
        gap = (hi_h - lo_h) * axh - heights.get(lo_h, fontsize * 1.118)
        if gap < 0:
            bad.append(f"lines at h={lo_h} and h={hi_h} overlap by {-gap:.1f} pt")
    # Raised, not asserted: `python -O` and PYTHONOPTIMIZE=1 strip the assert statement, so this
    # gate would silently become a no-op under an optimised interpreter. An explicit raise cannot
    # be optimised away.
    if bad:
        raise AssertionError(
            "stats block does not fit the triangle under the identity diagonal -- shorten a "
            "line, drop one, or lower the stack:\n  " + "\n  ".join(bad))


def _segments_cross(p1, p2, q1, q2):
    """True when the closed segments p1-p2 and q1-q2 share a point (orientation test)."""
    def _o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def _on(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
                and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)
    o1, o2, o3, o4 = _o(p1, p2, q1), _o(p1, p2, q2), _o(q1, q2, p1), _o(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on(p1, p2, q1)) or (o2 == 0 and _on(p1, p2, q2))
            or (o3 == 0 and _on(q1, q2, p1)) or (o4 == 0 and _on(q1, q2, p2)))


def _check_leaders_do_not_cross(segs):
    """Refuse to ship panel c with two species' leaders touching anywhere.

    `segs` is [(species, [point, point, ...]), ...]: each leader as a polyline in data units. Raised,
    not asserted, for the reason _check_under_diagonal gives.
    """
    bad = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (si, pi), (sj, pj) = segs[i], segs[j]
            for a, b in zip(pi, pi[1:]):
                for c, d in zip(pj, pj[1:]):
                    if _segments_cross(a, b, c, d):
                        bad.append(f"{si} x {sj}")
    if bad:
        raise AssertionError("panel c: species leaders cross: " + ", ".join(sorted(set(bad))))


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
    """Pearson standardised residuals of a 2-D contingency table on equal-count bins.

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
    """Panel a: the label-free MAF x deleteriousness dependence, as an association raster.

    Not a filled contour surface. Contouring log2(observed/expected) over a 10x8 table
    interpolates between cells that have no intermediate values, so it implies
    a continuity the data does not have, renders as a pale wash, shows no counts, and carries no
    uncertainty.

    What is drawn is the contingency table with no smoothing: discrete cells, Pearson
    standardised residuals, and the residual printed wherever |r| > 2. That reports the strength of the
    evidence, not just its direction -- residuals reaching |r| ~ 9 say the dependence is decisively
    detectable at n = 28,006 even though the effect size (rho = -0.099) is small; the panel shows both.
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
                        fontweight="bold", zorder=5,
                        # a hatched (negative) cell keeps its hatch, knocked out behind the numeral by a
                        # box in the cell's own colour
                        bbox=(dict(boxstyle="square,pad=0.22", ec="none",
                                   fc=ENRICH((R[i, k] + v) / (2.0 * v))) if R[i, k] < 0 else None))

    keep = [0, 2, 4, 6, 8, NX]
    ax.set_xticks([k - 0.5 for k in keep])
    # Three significant figures on every edge, in plain decimals: the lowest edge is not set in
    # exponent notation beside five decimals, and the 0.5 ceiling prints as 0.500 like its neighbours.
    ax.set_xticklabels(["%.*f" % (2 - math.floor(math.log10(10 ** xe[k])), 10 ** xe[k]) for k in keep],
                       fontsize=MINPT)
    # The first tick sits on the axes' left edge, so a centred label hangs half its width out to
    # the left and lands on the bottom y tick label, so it is re-anchored to start there.
    ax.get_xticklabels()[0].set_ha("left")
    # The last tick sits on the right edge; anchored the same way, it ends there.
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_yticks(np.arange(NY + 1) - 0.5)
    ax.set_yticklabels([_edge(e) for e in ye], fontsize=MINPT)
    ax.set_xlabel("Minor allele frequency in cattle (folded MAF; both axes equal-count bins)",
                  fontsize=AXIS)
    # One line. A second line such as "(variant-delta ~1 kb, equal-count bins)" is not set: rotated,
    # two lines of that length would be taller than the axes they label and wide enough to sit on
    # the y ticks. The x-label names the equal-count bins.
    ax.set_ylabel("Evo 2 deleteriousness", fontsize=AXIS)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=1.5, width=0.4)


    if ax_cb is not None:
        cb = plt.colorbar(im, cax=ax_cb, orientation="horizontal",
                          ticks=[-math.floor(v), -2, 0, 2, math.floor(v)])
        # the "(observed − expected) ⁄ √expected" gloss is legend material; the
        # key only has to name the quantity and the printing rule.
        # The key states the hatch rule by being hatched. The negative half of the bar carries
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


def panel_b(fig, gs_slice):
    """Small-multiple lattice — is the label-free gradient just conservation?

    Six complete mini-axes, each with its own frame: one pooled reference cell plus the five phyloP
    quintiles. A lattice here has >=6 cells and a facet variable other than the nine
    species; the facet is conservation stratum. Cell titles sit inside the cell.

    Within a stratum, every variant is scored against that stratum's own median
    deleteriousness, so under independence P(del > median) is exactly 0.5 at every frequency by
    construction. The null is therefore exact rather than eyeballed, all six cells share one
    interpretable axis, and a small effect is legible as a departure from flat. Plotting the raw
    conditional median instead would repeat the raw-axis problem of panel a -- the shift is ~5% of the score range.

    The gradient survives at fixed conservation in 4 of 5 quintiles, so
    it is not a conservation artefact. Its coefficient orders with conservation, but adjacent quintiles
    are not separable, so no monotone trend is claimed; its interval crosses zero in the least-conserved
    quintile, the boundary the expression panels also mark (Note S39), and that quintile is mostly neutral
    sequence, so its null says nothing by itself about regulatory variants.
    """
    C = J["selection_conditioned"]
    cells = C["cells"]
    # The quintile intervals are Table S6's. The cells carry their own rho_ci from a second bootstrap,
    # which differs from Table S6's intervals in the third decimal in every stratum; the strata block
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
        # ylim from the data, not a guess: a hard-coded (0.40, 0.70) would clip the Wilson bands flat
        # against the floor in the two most-conserved quintiles — the cells that carry the
        # panel's headline.
        _lo = min(min(cc["lo"]) for cc in cells[:6]); _hi = max(max(cc["hi"]) for cc in cells[:6])
        # headroom of 0.105 so the three-line stat block in the
        # top-right corner clears the curve's left shoulder at MINPT.
        ax.set_ylim(_lo - 0.012, _hi + 0.105)
        ax.set_xticks([-4, -3, -2, -1])
        # Tick density is computed, not typed. A hard-coded [0.45, 0.50, 0.55, 0.60] puts four
        # labels on an 18 mm cell: at this type scale that is a 7.05 pt pitch under a
        # 7.3 pt label, so every adjacent pair would overlap.
        # The step is chosen from the cell's own rendered height so the labels cannot touch,
        # and it widens by itself if the cell is ever made shorter. Shrinking the labels instead
        # would put them under style_gb.FLOOR.
        _yt = _pitched_ticks(ax, *ax.get_ylim())
        ax.set_yticks(_yt)
        # two decimals kept explicitly: the default formatter drops the trailing zero once the
        # step is 0.10, and "0.5" beside a panel whose whole claim is a departure from 0.50 reads
        # as a different quantity from the 0.50 the dashed null line marks.
        ax.set_yticklabels([f"{v:.2f}" for v in _yt], fontsize=MINPT)
        # tick_params before set_xticklabels: it rewrites labelsize on every tick label, so calling
        # it afterwards would undo the size set below.
        ax.tick_params(labelsize=MINPT, length=1.5)
        # No mathtext. $10^{-4}$ renders the exponent at ~70% of the base, so
        # even a 5.9 pt tick would put "-4" on the page at 4.13 pt. The axis is
        # labelled as log10 MAF and the ticks are plain integers at full size. The shared axis title under the lattice says "log10".
        ax.set_xticklabels(["−4", "−3", "−2", "−1"], fontsize=MINPT)
        if k % 3:
            ax.set_yticklabels([])
        if k // 3 == 0:
            ax.set_xticklabels([])
        _spines(ax)

        # cell title inside the cell, typographically differentiated from panel titles;
        # top-right, not top-left. These curves start high on the left and fall to the right, so the
        # left shoulder is exactly where the first data marker lands, which in the q3 cell is on top
        # of a top-left caption. Top-right is the clear quadrant for a descending series.
        # "All quintiles", not "All variants": this cell pools the five strata, the 27,234 variants
        # phyloP reaches, not the 28,006 of panel a and Table S6's first row.
        # Quintile bounds from the strata block, at the two decimals Table S6 prints: at one decimal
        # the boundary between the two middle quintiles rounds to a signed zero.
        if pooled:
            _title = "All quintiles"
        else:
            _b = (_strata[c["key"]]["phylop_lo"], _strata[c["key"]]["phylop_hi"])
            assert min(abs(v) for v in _b) >= 0.005, (c["key"], _b)
            _title = "phyloP %s to %s" % tuple(("%+.2f" % v).replace("-", "\u2212") for v in _b)
        ax.text(0.955, 0.955, _title,
                transform=ax.transAxes, fontsize=MINPT, color=col, fontweight="bold",
                ha="right", va="top")
        # Print the statistic each cell's call is made on. The endpoint "drop" alone
        # ranks q0 (drawn as flat) above q1 (counted as surviving) -- so the panel's own
        # numbers would contradict its own colouring. rho and its CI are what decide it.
        rc = (_strata[c["key"]]["ci"] if c["key"] in _strata else c.get("rho_ci")) or [float("nan"), float("nan")]
        # 0.955 / 0.807 / 0.659 is a MINPT stack on an 18 mm cell: the pitch is
        # set from the type size, not guessed, so the three lines cannot touch.
        # Spearman's correlation, named in words: ρ is the paper's pair coverage, and an r with a subscript s
        # would print the s under the 7.0 pt floor. The legend and the Notes write it rₛ.
        ax.text(0.955, 0.807, f"n = {c['n']:,}   Spearman {c['rho']:+.3f}", transform=ax.transAxes,
                fontsize=MINPT, color=MUTED, ha="right", va="top")
        ax.text(0.955, 0.659, f"[{rc[0]:+.3f}, {rc[1]:+.3f}]", transform=ax.transAxes,
                fontsize=MINPT, color=(CODING if flat else MUTED), ha="right", va="top",
                fontweight=("bold" if flat else "normal"))
        # No q0 gloss ("CI spans zero — no gradient detectable") on
        # the canvas: the interval printed directly above already spans zero
        # and is set in the CODING colour, so the sentence would restate the datum it
        # sits under. It is legend material.
    axes0.set_ylabel("P(deleteriousness > stratum median)", fontsize=AXIS)
    return axes0


def panel_c(ax):
    """Lift plane — the same question on the labelled panels.

    Each species is placed at its conservation-only AUROC (x; the conservation score is per-base GERP)
    and rises to its combined GERP+Evo2 AUROC (y). The diagonal is 'GERP alone', so the stick height is the lift
    the foundation model adds, and the x-position asks whether the lift depends on how strong
    conservation already is.

    This is the plate's one per-species panel, and it meets each of these conditions: both axes
    carry data, so species sit at a meaningful
    coordinate rather than a slot index; no PhyloPic silhouettes (Figure 9's signature ink); no left
    name margin; no right value column; no sort by n or by the plotted delta. The mark is a segment
    terminating in a dot, not a dot-with-CI-whisker: the segment is the estimate
    itself, and the only interval in the panel is the species mean's, drawn once as a band rather than
    nine times as whiskers.
    """
    D = J["decomposition"]; P = D["pooled"]
    rows = D["per_species"]
    xs = np.array([r["a_c"] for r in rows], float)
    ys = np.array([r["a_cf"] for r in rows], float)

    lo = min(xs.min(), ys.min()) - 0.03
    hi = max(xs.max(), ys.max()) + 0.175
    # The reference geometry is drawn over the data's range, not the axes' corners.
    # `hi` carries 0.175 of headroom that exists solely to hold the rotated species names, and
    # running the identity line and the mean ribbon corner-to-corner would send both straight through
    # that band and the names in it.
    # No text-vs-text collision check can see it, because the thing the name collides with is a
    # line. Beyond the last species these two carry no information anyway — they exist to be read
    # against the nine sticks, and the dashed 1.00 rule still spans the full width to close the top.
    grid = np.linspace(lo, max(xs.max(), ys.max()) + 0.012, 50)
    # The band must be commensurable with the sticks. The cross-species pooled AUROC (+0.020) ranks
    # variants across species, so cross-species positive/negative pairs count — a different estimand,
    # and drawing it here would make 8 of 9 sticks overshoot a band a reader takes for their average.
    # So the band is built from the very lifts drawn, with species as the resampling unit, because the
    # claim is cross-species generality.
    # The band is the species mean, the paper's primary aggregation everywhere else: the unweighted
    # mean of the nine lifts with its t interval on 8 degrees of freedom (+0.071 [+0.029, +0.113] in
    # Note S17 and the Analyses). The n-weighted mean (+0.054, species-clustered interval) lets human's
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

    # Species names pushed apart along x by a 1-D solver, then leadered back to their own dot. Four of
    # the nine sit within 0.045 AUROC of each other on x (cat, sheep, human and dog, 0.844 to 0.885), so
    # hand-placed names overstrike.
    lab = sorted(rows, key=lambda r: r["a_c"])
    lx_ = np.array([r["a_c"] for r in lab], float)
    span = hi - lo
    # The minimum separation is the rotated label's own width plus clear air, measured on this
    # axes, floored at 0.052 of the range; a fixed fraction alone would fall short whenever the box or
    # the type size changes. Nine names at MINPT need
    # (MINPT * 1.15 + 2.3) pt each; the solver below spreads them to exactly that and leaders each one back.
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
    # Elbow leaders. A straight leader from each dot to its spread-out name can cross another
    # species' leader just above that species' dot, so a name reads onto the wrong dot. Each leader
    # therefore rises vertically from its own dot to one common height just above the highest dot,
    # then slants to its name. Every slanted part runs between the same two heights, and the solver
    # above keeps the names in the order of their dots' x, so no two slanted parts can cross; the
    # vertical parts sit below that height, where no slanted part goes.
    y_elb = max(r["a_cf"] for r in lab) + span * 0.024
    if not y_elb < y_lab - span * 0.012:
        raise AssertionError("panel c: no room between the highest dot and the species names")
    segs = []
    for r, xl in zip(lab, lx_):
        pts = [(r["a_c"], r["a_cf"] + span * 0.012), (r["a_c"], y_elb), (xl, y_lab - span * 0.012)]
        ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], color="#B9B4AD", lw=0.35, zorder=4,
                solid_joinstyle="miter")
        segs.append((r["sp"], pts))
        ax.text(xl, y_lab, r["sp"][:1].upper() + r["sp"][1:], fontsize=MINPT, color=INK, ha="center", va="bottom",
                rotation=90)
    _check_leaders_do_not_cross(segs)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # The rule alone, with no label: it lands exactly on the existing 1.00 y tick, so the tick
    # names it. The band above 1.0 belongs to the rotated species names and their leaders, so no
    # text is set in it.
    ax.axhline(1.0, color=BEDROCK, lw=0.5, ls=(0, (2, 2)), zorder=1)
    # explicit ticks: the auto locator would emit 1.05, whose label runs 0.5 mm off the page edge.
    # x steps by 0.10, y by 0.05. Seven "0.xx" labels are 88.9 pt of glyph on a 37 mm axis,
    # so at 0.05 they would overlap by
    # 1.2 pt each and the extractor would return them as one 28-character span,
    # "0.700.750.800.850.900.951.00" -- which a Text-vs-Text collision audit scores as clean,
    # because it is a single object. The y axis is unaffected: its labels stack, they do not abut.
    tk = np.arange(0.70, 1.001, 0.05)
    ax.set_xticks(np.arange(0.70, 1.001, 0.10)); ax.set_yticks(tk)
    # Two decimals on both axes: they carry the same quantity.
    ax.set_xticklabels([f"{v:.2f}" for v in np.arange(0.70, 1.001, 0.10)])
    ax.set_aspect("equal", adjustable="box")
    # The conservation score here is per-base GERP (Note S17), named as such: the only conservation
    # track named elsewhere on this plate is panel b's phyloP, and "conservation" alone would be
    # read as that one.
    ax.set_xlabel("GERP alone, AUROC", fontsize=AXIS)
    ax.set_ylabel("GERP + Evo 2,\nAUROC", fontsize=AXIS, linespacing=1.2)
    ax.tick_params(labelsize=MINPT)
    _spines(ax)
    # the y axis stops at the 1.00 rule: above it is the band of rotated species names
    ax.spines["left"].set_bounds(lo, 1.0)
    # and the x axis stops at AUROC 1.0 too: the headroom right of it holds the stats block only
    ax.spines["bottom"].set_bounds(lo, 1.0)
    # the lower-right, below the diagonal, is the only region with no sticks and no label leaders --
    # in the upper-left the leaders would run straight through this text.
    # Everything here must fit under the diagonal, which on square axes means a right-aligned line at
    # height h may not start left of x = h. Short lines, stacked low.
    # The under-diagonal rule is measured, not stated: _check_under_diagonal measures every line
    # on the rendered axes, so a change of type size or box fails the build instead of shipping
    # a label that the diagonal runs through. Nothing else catches it:
    # no text-vs-text pair overlaps (the tick labels are the only neighbours and they are a
    # separate artist), and an edge-density average over the whole span box would let three
    # thin strokes across a wide label dilute below a 10% bar.
    # The four-line split below makes it fit.
    # The budget is arithmetic: on square axes the diagonal runs corner to corner, so a
    # right-aligned line whose top edge sits at axes-y t may be at most (0.99 - t) * 104.9 pt wide.
    # Four lines therefore get 95.3 / 87.2 / 79.1 / 71.0 pt from the bottom up, and the block has to
    # be ordered widest at the bottom -- the triangle is the constraint, so the stack is a staircase.
    # No three-line arrangement carries all four claims: the
    # top slot of a three-line stack is 79.1. Splitting the key off the count is what makes it fit,
    # and the reading order still runs encoding -> count -> estimate -> interval.
    _lines = [(0.236, "Stick = Evo 2's lift", MUTED, "normal"),
              (0.159, f"{D['n_positive']}/{D['n_species']} species positive", MUTED, "normal"),
              (0.082, f"Species mean {SM['mean']:+.3f}", "#8A6A00", "bold"),
              (0.005, f"[{SM['lo']:+.3f}, {SM['hi']:+.3f}], t on {len(_v) - 1} df", "#8A6A00", "normal")]
    for _h, _s, _c, _wt in _lines:
        ax.text(0.99, _h, _s, transform=ax.transAxes, fontsize=MINPT, color=_c,
                fontweight=_wt, ha="right", va="bottom")
    _check_under_diagonal(ax, _lines, MINPT)


def main():
    """The composed figure. A ClinVar panel is deliberately absent: its
    'abstention concentrates in noncoding' headline rests on a stratum that is 0.3% of the panel with
    81 negatives at 94.6% positive, so the miscalibration there is a base-rate artefact. The defensible
    ClinVar result is a labelled-benchmark generality claim and belongs with the trust layer, not in a
    figure whose subject is evidence that needs no curator."""
    # The mm budget below is authored from a top edge that reserves 4 mm for a title and
    # 10-20 mm for a standfirst. House style sets both in the legend, so the whole
    # stack lifts by HEAD_MM and the canvas loses the same amount: every panel keeps its
    # authored size in millimetres, and the plate does not open with 20 mm of white paper.
    HEAD_MM = 16.0
    # Canvas height (src/ccs/style_gb.py): the journal's 225 mm ceiling covers the figure and its
    # legend, and a 300-word legend sets to about 47 mm, so a full-width graphic stops at
    # style_gb.H_MM_MAX, 173 mm. A taller plate leaves the typesetter too little room for the legend, and
    # production would scale the whole figure down -- taking every label with it.
    H_MM = 173.0
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    # Budget solved in millimetres of the canvas, not by eye. Bands, top edge downwards:
    #   title 4 · subtitle 10-20 · a header 24.5 · a axes 26-56 · a ticks+xlabel
    #   56-64 · key 66-74.5 · b header 78 · b lattice 79.5-127.5 · b ticks to 131.5 ·
    #   b shared xlabel 132-135 · c/d header 140.5 · c axes 142-179 (square) ·
    #   c xlabel to 185 · d axes 142-174 · d xlabel to 182
    #
    # Bands are not scaled uniformly with the canvas. Panel b's six lattice cells carry a stratum
    # label and an "n = ... Spearman ..." line that are sized in points, so shrinking the cell removes
    # gap the type still needs; the lattice therefore keeps its full 48 mm. Height is saved instead in the
    # three bands whose content is a raster or a scatter and so rescales with its box: panel a's
    # raster (30 mm), panel c's square (its species names are spread by a solver that measures
    # its own type) and panel d's lattice (32 mm).
    def T(y_mm):            # figure fraction of a y measured down from the original top edge
        return 1.0 - (y_mm - HEAD_MM) / H_MM

    def box(x0, y_top, w_mm, h_mm):
        return [x0, T(y_top + h_mm), w_mm / W_MM, h_mm / H_MM]

    ax_a = fig.add_axes(box(0.112, 26.0, 146.2, 30.0))
    cax = fig.add_axes(box(0.330, 66.0, 73.1, 2.0))
    _im, lim = panel_a_resid(ax_a, cax)

    gs_b = fig.add_gridspec(1, 1, left=0.112, right=0.972, top=T(79.5), bottom=T(127.5))
    axb0 = panel_b(fig, gs_b[0, 0])
    # The lattice y-label belongs to the lattice, not to its top-left cell. On a
    # 18 mm cell a 42 mm rotated string overhangs the cell by more than its own
    # height. It sits 1.5 mm below the band's centre so its top end clears panel b's letter.
    axb0.set_ylabel("")
    fig.text(0.038, T(105.0), "P(deleteriousness > stratum median)", fontsize=AXIS,
             color=INK, ha="center", va="center", rotation=90)
    # panel b needs an x-axis label -- it carries the strongest
    # result in the figure. One shared label under the lattice. The ticks are
    # plain log10 exponents, so the label names the log10 axis.
    fig.text(0.542, T(132.9), "Minor allele frequency in cattle (log10; equal-count bins)",
             fontsize=AXIS, color=INK, ha="center", va="top")

    # panel c holds aspect='equal' (the diagonal must read as 45 degrees), so its box must be square in
    # millimetres or matplotlib shrinks the axes inside it and leaves dead paper.
    # The species names on it are spread by a solver that measures its own type, so the square
    # box does not crowd them.
    ax_c = fig.add_axes(box(0.112, 142.0, 37.0, 37.0))
    panel_c(ax_c)
    ax_e = fig.add_axes(box(0.4000, 142.0, 32.0, 32.0))
    panel_e_lattice(ax_e)

    S = J["selection"]; C = J["selection_conditioned"]; D = J["decomposition"]
    # The title and standfirst are in Additional file 1's Figure S3 legend (house style, src/ccs/style_gb.py).
    # A title and an italic paragraph on the plate would be the
    # figure arguing its own claim. The legend makes that argument, in the
    # journal's face, and opens "Score is associated with allele frequency without
    # curated labels, and its limits."

    for x_, y_mm, letter, title, note in (
            (None, 24.5, "a", "Evo 2 score is associated with allele frequency genome-wide, label-free",
             f"Cattle · ARS-UCD1.2 · n = {S['n']:,}"),
            (None, 78.0, "b", "…and it is not conservation in disguise — except where conservation ends",
             f"phyloP quintiles · reach {C['reach']:.1%}"),
            (None, 140.5, "c", "The same, on the labelled panels", None),
            (0.3465, 140.5, "d", "A second clade — but not a matched window", "bat")):
        y_ = T(y_mm)
        # House style: a bold lower-case letter, no reversed-out chip, no heading. `title`
        # is kept in the table above as documentation of what each panel is; the legend
        # states it for the reader.
        # x_ is None for the left column: those letters take the shared margin (style_gb.
        # letter_x) rather than a figure fraction typed to this plate's own rail, which would put
        # them at 50.9 pt, far from the margin the other figures in the set use.
        _letter_x = _SG.letter_x
        # A right-column letter starts at its panel's bounding-box left edge (panel d's y title).
        _lx, _ha = (_letter_x(fig), "left") if x_ is None else (x_, "left")
        fig.text(_lx, y_, letter, fontsize=PANEL, fontweight="bold", ha=_ha, va="bottom",
                 color=INK)
        if note == "bat":
            # Panel d's own tag, on its header line as panel a's is: the lattice is the bat pilot,
            # not cattle, and the pilot status belongs on the panel (fig6_free.json bat._note).
            # The binomial is set in italic, so the tag is two runs: the roman tail right-aligned
            # at the margin, the italic name ending where the tail begins. The tail's width is the
            # font's own advance (TextToPath, unhinted), which is what the PDF sets; a width read
            # off the 100-dpi raster canvas is hinted and runs about 7% wide at this size.
            from matplotlib.textpath import TextToPath as _T2P
            B = J["bat"]
            _tail = fig.text(0.972, y_ + 0.002, f" · {B['build']} · n = {B['n']:,} · pilot",
                             fontsize=MINPT, color=MUTED, ha="right", va="bottom")
            _w_pt = _T2P().get_text_width_height_descent(_tail.get_text(),
                                                         _tail.get_fontproperties(), ismath=False)[0]
            _x0 = 0.972 - _w_pt / (W_MM * 72.0 / 25.4)
            fig.text(_x0, y_ + 0.002, B["species"], fontsize=MINPT, color=MUTED, ha="right",
                     va="bottom", style="italic")
        elif note:
            fig.text(0.972, y_ + 0.002, note, fontsize=MINPT, color=MUTED, ha="right",
                     va="bottom")

    # No methods footer on the canvas: a methods paragraph is not a figure element, so its
    # prose is in the Additional file 1 legend for Figure S3.

    os.makedirs(ROOT / "reports" / "figures", exist_ok=True)
    base = str(ROOT / "reports" / "figures" / "Figure6_curator")
    _tm = _SG.tidy_minus
    _tm(fig)   # ASCII hyphen -> U+2212 in numeric labels
    fig.savefig(base + ".pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(base + ".png", dpi=600)
    return fig


# --------------------------------------------- d : the bat's observability lattice
BAT_PQ = None        # resolved at draw time by fig2_measure._deposit_table


def panel_e_lattice(ax):
    """Draws the constraint that governs the bat panel.

    A sentence such as "the frequency floor is 1/AN, 0.05 only where
    all ten are called and 0.167 at AN = 6" states this structural fact without
    showing it. Ten Myotis lucifugus samples cap the allele number at 20, so an allele frequency is not a
    continuous quantity here at all -- it is a ratio of two small integers, and most of the plane it
    appears to live on cannot be occupied.

    Rows are the allele number actually called, columns the minor allele count. Cell shade is how
    many variants sit there. The 28 hatched cells are not empty: they are impossible, because a
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
                                           # "//" diagonal to mean negative residual, and at print
                                           # size "///" is the same texture. One mark cannot carry
                                           # two meanings on one plate, so the impossible region
                                           # takes a cross-hatch instead.
                                           lw=0, hatch="xxx", zorder=4))
            # No in-cell numerals. The bottom row's ten counts need 123 pt of advance in a 105 pt
            # row, so they would overprint into one unreadable 34-digit run and the tenth would not be
            # drawn at all -- a collision the span-overlap check scores as clean because it fuses them
            # into a single span.

    ax.set_xticks(np.arange(len(macs)) + .5); ax.set_xticklabels(macs, fontsize=MINPT)
    ax.set_yticks(np.arange(len(ans)) + .5); ax.set_yticklabels([int(a) for a in ans], fontsize=MINPT)
    ax.set_xlabel("Minor allele count", fontsize=AXIS)
    ax.set_ylabel("Allele number called", fontsize=AXIS)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    # The three notes are anchored in axes fraction, not in data units. Anchoring them at
    # heights in data units would tie a stack of point-sized lines to a box whose height is in
    # millimetres: when the box shrinks, the anchors close up while the type does not, and the
    # blocks print on top of each other. Axes fractions with the blocks wrapped to the width the
    # narrower lattice leaves keeps the stack solvable: 3 + 2 + 2 lines at 1.35 leading is 74 pt
    # in a 90.7 pt box, so each gap is ~8 pt whatever the box height becomes.
    top = int(np.nanmax(G))
    ti, tj = [int(v[0]) for v in np.where(G == top)]
    ax.text(1.06, 1.00, "Darkest cell: allele number %d, count %d,\nholding %s variants"
            % (ans[ti], macs[tj], "{:,}".format(top)), transform=ax.transAxes, fontsize=FOOT,
            ha="left", va="top", color=INK, linespacing=1.35, clip_on=False)
    ax.text(1.06, 0.70, "So the frequency floor is 1/AN: 0.05\nwith all ten called, 0.167 at AN = 6",
            transform=ax.transAxes, fontsize=FOOT, ha="left", va="top",
            color=INK, linespacing=1.35, clip_on=False)
    ax.text(1.06, 0.40, "%d cells are impossible rather than empty:\na minor allele count cannot exceed\n"
            "half the allele number" % n_imp, transform=ax.transAxes, fontsize=FOOT,
            ha="left", va="top", color=MUTED, linespacing=1.35, clip_on=False)

    # Shade key on the canvas, not in the legend. The grey ramp is this panel's only
    # quantitative encoding and it is log10(count) -- a fact no sentence states. Anchored at
    # x = 1.06 in axes fraction, the same column the three notes above use (1.06 lands at PDF
    # x 288.8, exactly where those notes start), so the key travels with the box instead of
    # shrinking away from the type when the canvas height changes.
    cax_d = ax.inset_axes([1.06, -0.16, 0.90, 0.035], transform=ax.transAxes)
    cb_d = ax.figure.colorbar(im, cax=cax_d, orientation="horizontal",
                              ticks=[0.0, 1.0, 2.0, 3.0])
    cb_d.ax.set_xticklabels(["1", "10", "100", "1,000"])
    cb_d.set_label("Variants per cell (log scale)", fontsize=MINPT, labelpad=2.0)
    cb_d.ax.tick_params(labelsize=MINPT, length=1.5, width=0.4, pad=1.5)
    cb_d.outline.set_linewidth(0.4)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        import sys
        # An input this archive does not carry: name it and stop with exit 3, the archive's
        # "stopped at an undeposited path" code, rather than with a traceback.
        sys.stderr.write("MISSING INPUT: %s\nIt is not part of Additional file 2: "
                         "reports/DATA_MANIFEST.md lists every data/ path and its public source, "
                         "and docs/REPRODUCING.md says which scripts need data/ or Additional "
                         "file 3's tables (CCS_TABLES). Stopped (exit 3).\n" % (e.filename or e))
        sys.exit(3)


