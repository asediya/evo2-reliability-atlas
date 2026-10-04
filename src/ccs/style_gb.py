"""Genome Biology house style, shared by every figure builder in this paper.

Why this module exists. Each figure grew its own header block, panel-title voice and
accent colour, so seven figures reached the submission looking like seven different
papers: on-canvas titles and italic subtitle paragraphs, editorial panel headings in
blue or red, pill badges, filled letter chips, and commentary paragraphs set on the
canvas in bold colour. A journal figure does none of that. The title and the reading
instructions belong to the legend, which the typesetter sets in the journal's own face;
the canvas carries data, the axes that measure it, and the labels a reader needs to
identify a series.

The rules, applied in every builder:

  1. No figure title and no subtitle on the canvas. The legend carries both.
  2. Panels are marked with a bold lower-case letter and nothing else. No chips, no
     sentence-length headings, no colour. What a panel shows is the legend's job.
  3. Colour encodes data. It never marks emphasis. Commentary is not set in red.
  4. One type scale, below, with a 6.0 pt floor at final size.
  5. No decoration: no drop shadows, no rounded cards, no stroke effects behind text.
  6. Text that states a finding rather than labelling a mark belongs in the legend or
     in a supplementary note, not on the canvas.

Numbers a reader needs in order to read a panel -- an n, an effect size printed beside
the mark it belongs to, an axis rule at chance -- stay on the canvas. Removing those
would cost information; removing a sentence that argues does not.
"""

# ---------------------------------------------------------------- type scale (points)
# Calibrated to what Genome Biology actually prints, not to a floor. Glyph-ink measurement
# of ten published GB figures at the 170 mm full-page width gives tick labels 5-9.5 pt,
# axis titles 7-12.5 pt and panel letters 11-16 pt, in a ratio of about 1.6 : 1.25 : 1.
# Nature Portfolio's 5-7 pt guidance is calibrated to a narrower column and does not apply at 170 mm.
#
# WHY FLOOR DOES NOT TRACK ANNOT. Figure 2 panel a has 386.4 pt of height for 49 predictor
# names. At 7.0 pt exactly 49 fit; at 7.6 pt only 45 do, and the plate cannot grow because
# graphic + legend is capped at 225 mm. Cutting four predictors would change what the figure
# claims. So the dense blocks -- Figure 2's name and reach columns, Figure 4's provenance line,
# Figure 5's SGE key, Figure 7's nine-species list, Figure 9's whole annotation tier, Figure 10's
# b/c legend -- are set at FLOOR, and ANNOT rises past them. FLOOR is the size the house style
# provides for exactly this case.
#
# MEASURE PENETRATION, NOT BOX HEIGHT. A value and its CI on one line grazing by 0.8 pt have a
# bounding-box intersection 8.5 pt tall. Reporting the height makes 0.8 pt grazes look like 8.5 pt
# collisions. min(width, height).
PANEL = 12.0    # the bold panel letter
AXIS = 9.0      # axis labels
TICK = 8.4      # tick labels
ANNOT = 7.4     # in-panel annotation, series labels, n counts
FLOOR = 7.0     # nothing may render below this

# ---------------------------------------------------------------- the panel letter's left margin
# A letter placed relative to whatever a builder has to hand -- a gridspec cell's left edge, an axes
# fraction, a figure fraction -- steps in and out as a reader turns the pages. Panel letters do not
# mark different things in different figures, so they belong at one margin: 5.8 pt (2.05 mm) from
# the page edge.
#
# This is the LEFT COLUMN's margin. A letter over a right-hand column still sits at its own column's
# left edge -- that is the panel's corner, which is what the letter marks.
LETTER_X_PT = 5.8


def letter_x(fig):
    """Figure-fraction x of the shared panel-letter margin, for this figure's width."""
    return LETTER_X_PT / (fig.get_size_inches()[0] * 72.0)


# A panel letter drawn with va="bottom" is anchored on its DESCENDER line, not its baseline, so
# the whole em box -- 1.118 em for the Arial/Helvetica metrics this house style pins -- stands
# ABOVE the anchor. Measured against the PDF span bbox, not assumed: a 12 pt letter
# reaches 13.42 pt up. The letters therefore have to be clamped, and the clamp has to be in
# POINTS: the pad they are seated with is a figure fraction, so it shrinks with the canvas while
# the glyph does not. Unclamped at PANEL 12, the atlas plate's "a" and "b" would stand 0.16 pt
# off the top of its 173 mm page -- a clipped figure. Nothing above 12 pt is safe there by
# inspection; measure instead.
LETTER_BOX_EM = 1.118


def letter_top(fig, size=None, margin_pt=0.0):
    """Highest anchor y (figure fraction) that still keeps a va="bottom" letter on the canvas."""
    h_pt = fig.get_size_inches()[1] * 72.0
    return 1.0 - ((size or PANEL) * LETTER_BOX_EM + margin_pt) / h_pt


def letter_tf(ax):
    """Blended transform for a panel letter: x in FIGURE coords, y in the axes' own fraction.

    Use with letter_x(ax.figure) so the letter keeps whatever vertical clearance its panel needs
    while its left margin is the one every other figure uses.
    """
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.figure.transFigure, ax.transAxes)

# ---------------------------------------------------------------- canvas
# "width of 170 mm for full page width figure / maximum height of 225 mm for figure and
# legend". The cap covers the caption too: a figure drawn to the full 225 mm leaves
# the typesetter no room to place its own legend beneath it.
W_MM = 170.0
# MEASURED. The rule is 225 mm for FIGURE PLUS LEGEND. Set with real Arial advances at 8 pt on
# 9.6 pt leading across the full 481.89 pt measure, a 300-word legend is 14 lines = 47.4 mm.
# 173 mm leaves room for the longest legend here plus a 3 mm gap. Re-measure if the legends change.
H_MM_MAX = 173.0          # graphic only; the rest of the 225 mm belongs to the legend

# ---------------------------------------------------------------- the deleteriousness ramp
# TOL_DEL replaces RdBu_r wherever a value is drawn diverging about a baseline (Figure 5's BRCA1
# quilt, its four structure renders and their shared colour bar).
#
# RdBu_r is ColorBrewer-safe for colour-vision deficiency -- its extremes sit 0.399 (deuteranopia)
# and 0.356 (protanopia) apart in sRGB -- but both extremes have relative luminance 0.030, a WCAG
# contrast ratio of exactly 1.00. In greyscale a maximally tolerated substitution and a maximally
# deleterious one are the SAME GREY, so a printed or photocopied quilt carries no signal.
#
# TOL_DEL keeps the diverging structure and the blue/red convention and pins its lightest anchor
# to t = 0.5, so it still lands on the TwoSlopeNorm centre. What changes is that the tolerated arm
# ends at a MEDIUM blue rather than a dark one, which makes the luminance run monotonically down
# from the centre to the deleterious end and down again, more gently, to the tolerated end:
#
#     ends contrast        4.00   (RdBu_r 1.00)      centre vs tolerated  3.23
#     deuteranopia ends    0.675  (RdBu_r 0.399)     centre vs deleterious 12.89
#     protanopia ends      0.705  (RdBu_r 0.356)
#
# so it is better in greyscale AND under both simulated dichromacies. The four structure PNGs are
# rendered from an undeposited PDB tree and cannot be re-rendered here; tools/recolour_brca1_
# structures.py re-maps them pixelwise onto this ramp so the panel does not disagree with itself.
TOL_DEL_NODES = [(0.00, "#5C8DBB"), (0.25, "#A8C8E0"), (0.50, "#F7F6F3"),
                 (0.68, "#E9A183"), (0.85, "#BC3A2E"), (1.00, "#5A0D0A")]


def tol_del_cmap():
    """The deleteriousness ramp as a matplotlib colormap. See TOL_DEL_NODES for why it exists."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tol_del", TOL_DEL_NODES)

LINE_MIN = 0.3            # "All lines should be wider than 0.25 pt"

# ---------------------------------------------------------------- colour
# Okabe-Ito, ordered so that any prefix stays maximally separated under dichromacy
# (minimum pairwise CIE-Lab dE >= 17 deuteranope, >= 20 protanope at every n).
# ---------------------------------------------------------------- the text-safe ochre
# MEASURED. As LABEL TEXT, the set's three different ochres all
# fail WCAG's 4.5:1 for small text on white: #E8B15C at 1.93 (Figure 5's BRCT), #C08A2E at
# 3.04 (Figures 2 and 4's middle reach regime) and #B37C00 at 3.62 (a clade shade in 6 and 7).
# 1.93:1 is a label a reader has to hunt for. OCHRE_TEXT is one colour at the same hue (40 deg,
# the family all three sit in) taken down to 4.60:1, so the swatch and the word that names
# it stay the same colour and the word is readable. Use it wherever an ochre also sets type.
OCHRE_TEXT = "#9F6B02"


def declutter_1d(desired, widths, lo, hi, gap):
    """Nudge a row of labels apart along one axis, moving each as little as possible.

    `desired` are the positions each label wants (the centre of the thing it names), `widths` their
    extents in the same units. Returns positions whose boxes no longer overlap and which stay
    inside [lo, hi]. Order is preserved, so a label never crosses its neighbour and its leader
    never crosses another's.

    Relaxation rather than an exact solve: adjacent overlaps are split evenly and the row is then
    pushed back inside the bounds, repeated to convergence. For nine labels it settles in a few
    passes and the residual displacement is far below a point.
    """
    n = len(desired)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: desired[i])
    x = [float(desired[i]) for i in order]
    w = [float(widths[i]) for i in order]
    # If the row cannot fit -- sum of widths plus gaps wider than hi - lo -- the `lo` shift and the
    # `hi` shift below push against each other and this loop runs all 400 iterations without
    # converging, then returns positions that still breach a boundary, silently. Say so. The
    # positions are unchanged either way: this only reports, because a row that currently fits must
    # keep the exact geometry it has.
    need_total = sum(widths) + gap * (n - 1)
    converged = False
    for _ in range(400):
        moved = False
        for k in range(n - 1):
            need = (w[k] + w[k + 1]) / 2.0 + gap
            d = x[k + 1] - x[k]
            if d < need - 1e-12:
                push = (need - d) / 2.0
                x[k] -= push
                x[k + 1] += push
                moved = True
        left = lo - (x[0] - w[0] / 2.0)
        if left > 0:
            x = [v + left for v in x]; moved = True
        right = (x[-1] + w[-1] / 2.0) - hi
        if right > 0:
            x = [v - right for v in x]; moved = True
        if not moved:
            converged = True
            break
    if not converged:
        import sys as _sys
        over = need_total - (hi - lo)
        _sys.stderr.write(
            "style_gb.label_row: %d labels did not converge in 400 passes; they need %.4g of a "
            "span %.4g wide (%s by %.4g). The row still breaches a boundary by about "
            "%.4g; widen the axes, shorten the labels, or reduce gap_pt.\n"
            % (n, need_total, hi - lo, "over" if over > 0 else "within", abs(over),
               max(0.0, lo - (x[0] - w[0] / 2.0), (x[-1] + w[-1] / 2.0) - hi)))
    out = [0.0] * n
    for k, i in enumerate(order):
        out[i] = x[k]
    return out


def label_row(ax, items, y, fontsize, y_from, rotation=0, gap_pt=2.5, va="top", ha="center",
              lead_color="#B4B4B4", lead_lw=0.4, zorder=3):
    """Seat a row of labels on ONE baseline under the columns they name.

    Replaces the two-rank stagger both mosaic panels used. A stagger is what you fall back on when
    labels are placed at their columns' centres and left to collide; it costs a second rank of
    height, doubles the leader lengths, and makes the reader's eye zig-zag to read nine names in
    order. Measuring each label and nudging it the minimum distance keeps one baseline.

    `items` is [(x_centre, text, colour), ...] in the axes' own x units. Every label gets a leader
    from (x_centre, y_from), vertical where the label did not have to move and slanted where it
    did: one rule for all nine reads as a tick row, whereas drawing leaders only for the displaced
    ones makes the displacement itself look like an encoding. Returns the placed x positions.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_window_extent(r)
    x0, x1 = ax.get_xlim()
    per_unit = bb.width / float(x1 - x0)          # pixels per axes-x unit

    texts, widths = [], []
    for cx, s, col in items:
        # For rotation=90 pass ha="right", va="center": with rotation_mode="anchor" that hangs the
        # label below the anchor reading upward, and the HORIZONTAL centring then comes from va --
        # which is what the declutter below assumes, since it treats each desired x as a centre.
        t = ax.text(cx, y, s, fontsize=fontsize, ha=ha, va=va, color=col,
                    rotation=rotation, rotation_mode="anchor", zorder=zorder + 1)
        texts.append(t)
        widths.append(t.get_window_extent(r).width / per_unit)

    gap = gap_pt * fig.dpi / 72.0 / per_unit
    xs = declutter_1d([it[0] for it in items], widths, x0, x1, gap)
    for t, nx in zip(texts, xs):
        t.set_x(nx)

    for (cx, _s, _c), nx in zip(items, xs):
        ax.plot([cx, nx], [y_from, y], color=lead_color, lw=lead_lw, zorder=zorder,
                solid_capstyle="butt", clip_on=False)
    return xs


_MINUS_RE = None


def tidy_minus(fig):
    """Replace the ASCII hyphen with a true minus sign (U+2212) in every numeric label.

    MEASURED. matplotlib sets its OWN tick labels with U+2212 (axes.unicode_minus is
    on by default), but every number a builder formats by hand arrives with an ASCII hyphen. The
    two are different widths and sit at different heights, so a hand-formatted "-2.4" inside a
    cell can sit directly above "\u22124" on its axis with a mismatched sign.

    The hyphen is replaced ONLY where it is immediately before a digit and NOT preceded by a letter
    or digit, so a real hyphen survives: "Evo 2-1B", "8,192-bp", "non-causal", "log-count".

    Call once on the finished figure, immediately before savefig.
    """
    global _MINUS_RE
    if _MINUS_RE is None:
        import re as _re
        # The second alternative catches the minus of a scientific-notation exponent, which the
        # first cannot see because a digit precedes the "e". Without it "7e-05" and "p = 7e-24"
        # keep an ASCII hyphen while every other numeric minus sign is U+2212. Requiring a DIGIT
        # before the e/E keeps "Evo 2-1B", "e-value", "non-coding" and "8,192-bp" untouched.
        _MINUS_RE = _re.compile(r"(?<![0-9A-Za-z])-(?=[0-9])|(?<=[0-9][eE])-(?=[0-9])")
    # FixedFormatter first. set_xticklabels/set_yticklabels install a formatter that RE-SUPPLIES
    # its original strings on every draw, so editing the Text objects alone is undone by savefig's
    # own draw. MEASURED, not assumed: on this matplotlib, set_xticklabels installs a FUNCFormatter,
    # not a FixedFormatter. Both are handled; the FuncFormatter is wrapped so the
    # substitution happens at draw time, after the formatter has produced its string.
    from matplotlib.ticker import FixedFormatter as _FF, FuncFormatter as _FnF
    for _ax in fig.axes:
        for _axis in (_ax.xaxis, _ax.yaxis):
            _fm = _axis.get_major_formatter()
            if isinstance(_fm, _FF):
                _fm.seq = [_MINUS_RE.sub("\u2212", _s) for _s in _fm.seq]
            elif isinstance(_fm, _FnF):
                _axis.set_major_formatter(_FnF(
                    lambda _v, _p, _inner=_fm.func: _MINUS_RE.sub("\u2212", str(_inner(_v, _p)))))
    import matplotlib.text as _mtext
    n = 0
    for t in fig.findobj(_mtext.Text):
        s0 = t.get_text()
        if s0 and "-" in s0:
            s1 = _MINUS_RE.sub("\u2212", s0)
            if s1 != s0:
                t.set_text(s1)
                n += 1
    return n


def over(fg, alpha, bg="#FFFFFF"):
    """The colour a reader ACTUALLY SEES when `fg` is painted at `alpha` over `bg`.

    ink_on() answers a question about the ink behind the type, so it has to be handed the
    composited colour, not the nominal one. Figure 6's wedges are filled with desat(col, 0.35) at
    alpha 0.96. For horse, INK clears the nominal #D55E00 at exactly 4.50:1 but reads only 4.20:1
    on the #B56A30 the reader actually sees.
    """
    from matplotlib.colors import to_hex
    f = [int(to_hex(fg)[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(to_hex(bg)[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(alpha * f[i] + (1 - alpha) * b[i])) for i in range(3))


def ink_on(bg, target=4.5):
    """White, the house INK, or pure black -- whichever a reader can actually read on `bg`.

    MEASURED. A value printed inside a filled mark is routinely set white because the
    fill is "a colour", but a fill chosen to be distinguishable as a MARK is often light: Figure 6's
    wedge numerals set white come to 1.76:1 on cat's amber and Figure 1's cells to 2.73:1 on mid viridis.

    White and a PURE BLACK dark candidate cross over at luminance 0.179, so whichever wins always
    clears WCAG's 4.5. The house INK is #1A1A1A, whose own luminance 0.0103 moves the crossover
    to luminance 0.202, where both candidates sit at 4.18:1 -- short of 4.5 for mid-tone fills.

    Pure black gives a real floor -- worst case 4.58:1 at luminance 0.179 -- so it is the
    fallback wherever neither house candidate reaches `target`. That band is narrow, and #1A1A1A
    against #000000 is invisible outside it, so the house ink survives everywhere it works.

    Pass the fill as a hex string or any matplotlib colour spec.
    """
    from matplotlib.colors import to_hex
    h = to_hex(bg)

    def _l(hx):
        v = [int(hx[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        f = lambda c_: c_ / 12.92 if c_ <= 0.03928 else ((c_ + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(v[0]) + 0.7152 * f(v[1]) + 0.0722 * f(v[2])

    def _r(a, b):
        la, lb = _l(a), _l(b)
        hi, lo = max(la, lb), min(la, lb)
        return (hi + 0.05) / (lo + 0.05)

    best = "#FFFFFF" if _r("#FFFFFF", h) >= _r(INK, h) else INK
    if _r(best, h) >= target:
        return best
    return "#FFFFFF" if _r("#FFFFFF", h) >= _r("#000000", h) else "#000000"


def text_safe(c, target=4.5):
    """Darken `c` along its OWN hue until it clears `target`:1 against white.

    A category's colour is often set as type as well as ink -- the key label beside its swatch,
    a coloured tick label, a species name. Measured across the ten plates, that produced label
    text at 1.63:1 (#FDBE85), 1.79 (#EDBA47), 3.06 (#CC79A7) and 3.42 (#009E73): washed out at
    6.8 pt, which is most of what reads as unfinished in a figure. Hue and saturation are held,
    so the word still matches the swatch it names; only lightness moves, and only downward.
    Returns `c` unchanged when it already clears the target.
    """
    import colorsys
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    def _cr(rgb):
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        return 1.05 / (0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2]) + 0.05)

    if _cr((r, g, b)) >= target:
        return c
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    while l > 0.0:
        l -= 0.005
        rgb = colorsys.hls_to_rgb(h, max(l, 0.0), s)
        if _cr(rgb) >= target:
            return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgb)
    return "#000000"

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#000000", "#F0E442"]
# Paul Tol high-contrast: use for two or three series, or whenever greyscale must survive
# (lightness spread dL* 20.3).
TOL_HC = ["#004488", "#DDAA33", "#BB5566"]
# Highlight against context.
HILITE, CONTEXT = "#D55E00", "#BBBBBB"
SEQ = "viridis"           # monotone in L*; cividis if greyscale printing is possible
DIV = "RdBu"              # only with a real midpoint, always centred

# ---------------------------------------------------------------- neutral ink
INK = "#1A1A1A"      # primary text
MUTED = "#5C5C5C"    # secondary text: n counts, units, provenance
RULE = "#B4B4B4"     # reference rules, chance lines, light structure
GRID = "#E4E4E4"

# Arial first, deliberately: every figure in this paper as submitted is ArialMT, and
# Helvetica resolves on macOS as a .ttc collection. Leaving Helvetica at the head made
# any builder that trusted this list alone render in a different face from its siblings.
FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]


def rc(extra=None):
    """rcParams every builder sets before it draws."""
    p = {
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": FAMILY,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": "#8C8C8C", "axes.linewidth": 0.6,
        "xtick.color": "#8C8C8C", "ytick.color": "#8C8C8C",
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "axes.labelsize": AXIS, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.fontsize": ANNOT, "font.size": ANNOT,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "lines.linewidth": 1.0,
        "figure.dpi": 600, "savefig.dpi": 600,
    }
    if extra:
        p.update(extra)
    return p


def panel(ax, letter, x=-0.085, y=1.045):
    """The only panel marker: a bold lower-case letter above the axes' top-left.

    No accompanying text. A panel heading that says what the panel shows duplicates
    the legend, and a heading that says what the panel *means* is an argument the
    legend has to make in the journal's own voice.
    """
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL,
            fontweight="bold", color=INK, ha="left", va="baseline")
    return ax


def count(ax, text, x=1.0, y=1.045):
    """A small right-aligned provenance note (an n, a readout) above the panel."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=ANNOT, color=MUTED,
            ha="right", va="baseline", linespacing=1.28)
    return ax


def spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    return ax
