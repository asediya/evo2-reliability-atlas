"""Additional file 1's Figure S4 — THE TRUST LAYER. "The ordering transfers; the probability does not."

Built only from reports/fig4_reconciliation.json + reports/fig4_pervariant.parquet
(src/ccs/fig4_reconcile.py). No float is typed by hand in this module.

WHY THIS FIGURE LOOKS THE WAY IT DOES
A 50-agent fleet generated 14 candidate designs and killed all 14, because the deposited calibration
numbers flipped verdict with the estimator/aggregation choice. The reconciliation settled it:
  * MACRO aggregation only. Pooled ECE is unstable (pooled width10 said transfer won by 10x) because
    human (n=3,000, 50% prevalence) dominates eight species at ~9% and per-species errors cancel.
  * The verdict is a PAIRED bootstrap on delta-ECE, not a comparison of point estimates. Under macro,
    all four estimators straddle zero -> "statistically indistinguishable" is the robust claim.
  * The deposited "abstention halves error" was measured on IN-COHORT (oracle) posteriors
    (analyze_trust_layer.py:151). Panel c uses honest LOSO posteriors and a random control instead.
  * The per-variant hero uses the PLATT posterior (8,792 distinct confidences / 11,130 variants).
    Isotonic gives only 248, so a per-variant panel on isotonic would place >99% of marks by sort
    tie-break -- decoration wearing the costume of data.

Run: python -m src.ccs.fig4_trust
"""
import os, json, textwrap
import numpy as np, polars as pl
import matplotlib as mpl
# Caption text is wrapped by measuring against fig.canvas.get_renderer(), so the line breaks
# depend on the ACTIVE backend: built under macosx the breaks differ from the same build under
# Agg, which is what any headless Linux rebuild (a referee, CI, a Zenodo re-run) gets. Pinning
# Agg here makes the figure reproduce identically everywhere. fig2_split.py already does this.
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

try:
    from .figtext import restore_text_layer
except ImportError:
    from figtext import restore_text_layer

MM = 1 / 25.4
# Full page width is 170 mm and the 225 mm height ceiling INCLUDES the legend; a canvas over
# either limit would be silently downscaled in production.
W_MM, H_MM = 170.0, 173.0
# No TITLE size: src/ccs/style_gb.py rule 1 puts the title and the subtitle in the
# manuscript legend, so nothing on this canvas is set above the panel letter.
# Module-level import of the house style, so the scale below can be derived from it.
# Same dual form src/ccs/fig3_rebuild.py uses: package import when run as -m, flat when run as a file.
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG

# DERIVED from style_gb, not copied, so a change to the house style reaches this figure too.
PANEL, AXIS, TICK, FOOT = SG.PANEL, SG.AXIS, SG.TICK, SG.ANNOT

CODING = "#0072B2"     # the transferable/deployable arm
FAIL = "#A8201A"       # errors, and the failure accent
BEDROCK = "#3A3A3A"    # reference datum, drawn identically everywhere
SUB = "#8C929B"        # substrate / control
POS = "#E69F00"        # the unattainable oracle
INK = "#1A1A1A"
MIN_N = 10             # no cell or profile point is drawn on thinner evidence than this
MUTED = SG.MUTED

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # Ink, axes and ticks take the house tokens (style_gb.rc): type in INK, axes and ticks in #8C8C8C.
    "text.color": INK, "axes.labelcolor": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.edgecolor": "#8C8C8C", "xtick.color": "#8C8C8C", "ytick.color": "#8C8C8C",
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.labelsize": AXIS, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
})

R = json.load(open("reports/fig4_reconciliation.json", encoding="utf-8"))
PV = pl.read_parquet("reports/fig4_pervariant.parquet")
RICH = set(R["_meta"]["rich_pool"])
TARGET_ONLY = R["_meta"]["target_only_never_trained_on"]
ORDER = [s for s, _ in sorted(((s, PV.filter(pl.col("species") == s).height)
                               for s in PV["species"].unique().to_list()), key=lambda t: t[1])]


def _wrap(t, mm, pt):
    return textwrap.fill(" ".join(t.split()), width=max(20, int((mm * 72 / 25.4) / (0.52 * pt))))


def _axfrac(ax, px):
    """Convert a height in display pixels to a fraction of this axes' height."""
    return px / max(1e-9, ax.get_window_extent(ax.figure.canvas.get_renderer()).height)


def _below(ax, y_guess):
    """First free y (axes fraction) under an axes, measured below its x-label instead of guessed."""
    fig = ax.figure; fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = ax.get_window_extent(r).y0
    lo = top
    for o in [ax.xaxis.label] + list(ax.get_xticklabels()):
        if o.get_visible() and (o.get_text() or "").strip():
            lo = min(lo, o.get_window_extent(r).y0)
    return min(y_guess, -_axfrac(ax, top - lo) - _axfrac(ax, 3.4 / 72.0 * fig.dpi))


def _fit(ax, txt, width=1.0, **kw):
    """Wrap txt to the axes width by MEASURING it. The old character-width estimate (0.52 em/char) was
    wrong often enough to push captions off the canvas, so measure the real string instead of guessing.

    The measurement is now CHECKED. textwrap.fill takes a character count, and the count derived from
    one measurement is still an estimate: it left panel e's caption 7 mm past the grid's right edge
    and, before the red paragraph under panel a was deleted, 0.5 mm past the page edge itself. Narrow
    the wrap until the RENDERED box is inside the budget."""
    fig = ax.figure; r = fig.canvas.get_renderer()
    avail = ax.get_window_extent(r).width * width

    def _w(s_):
        t = ax.text(0.0, 0.0, s_, transform=ax.transAxes, **kw)
        fig.canvas.draw()
        w_ = t.get_window_extent(r).width
        t.remove()
        return w_

    w = _w(txt)
    if w <= avail:
        return txt
    flat = " ".join(txt.split())
    n = max(24, int(len(flat) * avail / w * 0.97))
    out = textwrap.fill(flat, width=n)
    while n > 24 and _w(out) > avail:
        n -= 2
        out = textwrap.fill(flat, width=n)
    return out


def _line(ax, y, txt, gap_pt=2.6, width=1.0, x=0.0, **kw):
    """Draw one caption line at y and return the next free y, spaced by its RENDERED height."""
    kw.setdefault("linespacing", 1.40)
    fig = ax.figure
    t = ax.text(x, y, _fit(ax, txt, width, **kw), transform=ax.transAxes, ha="left", va="top", **kw)
    fig.canvas.draw()
    h = t.get_window_extent(fig.canvas.get_renderer()).height
    return y - _axfrac(ax, h) - _axfrac(ax, gap_pt / 72.0 * fig.dpi)


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color("#8C8C8C")


def _mark(place, x_letter, y, letter, right=None, x_right=1.0, va="baseline"):
    """The ONLY panel marker (src/ccs/style_gb.py rule 2): a bold lower-case letter in black.

    `place` is ax.text or fig.text with the matching transform already bound, so panel a
    (figure coords) and b-e (axes coords) render the same marker rather than two lookalikes
    that drift apart. Gone with the old header: the reversed white-on-black letter chip, the
    bold sentence-length panel title, and the hairline rule that underlined them. `right` is
    the one thing that may keep the letter company -- a small grey provenance note (an n, a
    readout), which is style_gb.count(), not a heading.
    """
    place(x_letter, y, letter, fontsize=PANEL, fontweight="bold", va=va, ha="left",
          color=INK)
    if right:
        place(x_right, y, right, fontsize=FOOT, va=va, ha="right", color=MUTED)


def _plab_top(ax):
    """House style: a bold lower-case letter and nothing else.

    The `text` argument is accepted and IGNORED -- see src/ccs/fig3_rebuild.py:_plab, the
    worked example this follows. Every panel used to carry a sentence-length heading in bold
    ("No robust gain over a pooled sigmoid", "Why it fails, and what it would cost to fix"),
    which either repeated the legend's per-panel sentence or stated the finding the legend is
    there to state in the journal's own voice. The call sites keep their strings as
    documentation of what each panel is; the canvas no longer prints them.
    """
    # WHERE THE LETTER SITS IS MEASURED, NOT TYPED. It was pinned at y = 1.075 -- a fraction of the
    # axes height, so it means a different number of points in every row, and nothing in it knows
    # how tall the letter is. At the corrected 11 pt panel size the letter's box grew past what
    # 0.075 of a 66 pt panel buys and landed on the two labels that already own the axes' top-left
    # corner: panel c's top y tick ("0.08") and panel d's "Δlogit π" column header. The letter is
    # now lifted until its own RENDERED box clears whatever occupies that corner, with the letter's
    # BOTTOM (not its baseline) as the anchor, so the clearance is the one the PDF will show.
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_window_extent(r)
    top = bb.y1
    for o in list(ax.get_yticklabels()) + list(ax.texts):
        if not o.get_visible() or not (o.get_text() or "").strip():
            continue
        ob = o.get_window_extent(r)
        # only the top-left corner: a label further right cannot be under the letter
        if ob.y1 > bb.y1 - 2.0 and ob.x0 < bb.x0 + 0.45 * bb.width:
            top = max(top, ob.y1)
    return top + 2.6 * fig.dpi / 72.0


def _plab(ax, letter, text=None, right=None, y_page=None, left=False):
    """Place one panel letter. `y_page` overrides the per-panel clearance with a shared one.

    `left` puts the letter at the shared margin every figure in the set uses (style_gb.letter_x)
    instead of this axes' own -0.115 fraction, which is only correct for a right-hand column.
    """
    fig = ax.figure
    if y_page is None:
        y_page = _plab_top(ax)
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    y = 1.0 + (y_page - bb.y1) / bb.height
    if left:
        from .style_gb import letter_x, letter_tf
        _mark(lambda x, y_, s, **kw: ax.text(x, y_, s, transform=letter_tf(ax), **kw),
              letter_x(fig), y, letter, right, va="bottom")
    else:
        _mark(lambda x, y_, s, **kw: ax.text(x, y_, s, transform=ax.transAxes, **kw),
              -0.115, y, letter, right, va="bottom")


def _plab_row(items):
    """One shared baseline for every letter in a row.

    _plab lifts each letter until its OWN panel's top-left corner is clear, which is correct
    per panel and ragged across a row, because panels differ in what sits in that corner. Taking
    the row maximum keeps every panel's clearance and puts the letters on one line.

    `items` is [(ax, letter), ...] left to right. Every top is measured BEFORE any letter is
    drawn, so a letter already placed cannot inflate its neighbour's clearance.
    """
    y_page = max(_plab_top(ax) for ax, _ in items)
    for i, (ax, letter) in enumerate(items):
        _plab(ax, letter, y_page=y_page, left=(i == 0))


# ---------------------------------------------------------------- a. the risk field
FLD = json.load(open("reports/fig4_field.json", encoding="utf-8"))


def panel_a(fig, gs_slice):
    """LAYERED RISK FIELD. Local error rate over (signed log-odds x species) as a raster whose COLOUR is
    the measured rate and whose OPACITY is the denominator behind it, so thinly-evidenced regions fade
    rather than asserting structure -- the failure of the staircase this replaces, which drew goat's six
    errors as seven crisp steps. Every one of the 893 individual errors is drawn on top, and the refusal
    corridor each species would actually apply is outlined."""
    from matplotlib.colors import LinearSegmentedColormap
    # matplotlib.patheffects is no longer imported here: nothing on this canvas is haloed.
    M = FLD["_meta"]; RW = FLD["rows"]
    LO, HI, NB = M["lo"], M["hi"], M["nbins"]
    inner = gs_slice.subgridspec(2, 1, height_ratios=[0.78, 3.30], hspace=0.14)
    cmap = LinearSegmentedColormap.from_list("risk", ["#F4F1EE", "#EBC9C0", "#D07B6C", "#A8201A", "#5E0A16"])
    VMAX = 0.60
    rng = np.random.default_rng(4)

    # ---- marginal: the pooled error-rate profile, so the field has a reference curve ----
    axm = fig.add_subplot(inner[0, 0])
    pr = np.array([np.nan if v is None else v for v in FLD["pooled"]["rate"]], float)
    pc = np.array(FLD["pooled"]["count"], float)
    # A bin holding ONE variant that happened to be an error reads 1.000 and would set the whole strip's
    # scale, drawing 1.9x taller than the genuinely-evidenced peak -- the exact crime the field below is
    # designed to avoid. Mask, do not clip: clipping turns those spikes into a saturated flat shelf.
    pr = np.where(pc >= MIN_N, pr, np.nan)
    ctr = (np.array(M["edges"][:-1]) + np.array(M["edges"][1:])) / 2
    # Fill only where the rate is DEFINED. nan_to_num dropped the fill to zero at the mask boundary and
    # drew a spurious spike where the masked bins ended.
    axm.fill_between(ctr, 0, pr, where=np.isfinite(pr), color="#A8201A", alpha=0.22, lw=0, zorder=2)
    axm.plot(ctr, pr, color="#A8201A", lw=0.9, zorder=3)
    # Headroom for the band label, and no more. The old 2.70x was sized for a three-line red
    # paragraph; the neutral two-line label needs less, and the profile curve gets the difference.
    axm.set_xlim(LO, HI); axm.set_ylim(0, np.nanmax(pr) * 2.20)
    axm.set_xticks([]); axm.set_yticks([])
    for sname in ("top", "right", "left", "bottom"):
        axm.spines[sname].set_visible(False)
    # The strip is 2.4 mm tall and cannot carry both its own caption and the hot-zone callout; the
    # caption moves to the block below, where there is room.
    # Mark where risk actually concentrates: the LONGEST CONTIGUOUS run above twice the cohort rate.
    # Taking min/max of every qualifying bin spanned almost the whole populated axis and said nothing.
    base = M["n_err"] / M["n"]
    over = (pc >= MIN_N) & (pr > 2 * base)
    runs, st = [], None
    for k, v in enumerate(over):
        if v and st is None:
            st = k
        elif not v and st is not None:
            runs.append((st, k)); st = None
    if st is not None:
        runs.append((st, len(over)))
    if runs:
        a_, b_ = max(runs, key=lambda t: t[1] - t[0])
        h0, h1 = M["edges"][a_], M["edges"][b_]
        axm.axvspan(h0, h1, color=FAIL, alpha=0.11, lw=0, zorder=1)
        # Report the MEASURED rate in the band, not `2 * base` -- that is the inequality used to SELECT
        # the bins, so printing it asserted only that bins chosen for exceeding 16% exceed 16%, and
        # understated the real figure by 2.07x.
        _p = np.clip(PV["p"].to_numpy(), 1e-6, 1 - 1e-6)
        _lg = np.log(_p) - np.log(1 - _p)
        _y, _pd = PV["label"].to_numpy(), PV["pred"].to_numpy()
        _in = (_lg >= h0) & (_lg < h1)
        _rate = float((_pd != _y)[_in].mean())
        _miss_out = int(((_y == 1) & (_pd == 0) & (_lg < h0)).sum())
        _miss_all = int(((_y == 1) & (_pd == 0)).sum())
        # A NEUTRAL LABEL for the shaded band, not an argument about it. What this used to say --
        # "highest local error ... BUT 82% of missed positives sit LEFT of it, out in
        # confidently-negative territory" -- was two claims in red bold: it named the band and then
        # argued with the reader about what the band means. The naming stays, with every number it
        # carried; the argument goes to the legend and Additional file 1 Note S45.
        # Grey, unbolded, and set in the strip's own headroom rather than haloed over the profile
        # curve: the halo was there only because the caption was sitting on the line it described.
        axm.annotate(f"band: logit {h0:+.1f} to {h1:+.1f} — {_rate:.0%} local error, "
                     f"{_rate / base:.1f}× the cohort rate",
                     xy=(h0, np.nanmax(pr) * 0.55), xytext=(LO + 0.2, np.nanmax(pr) * 2.13),
                     fontsize=FOOT, color=MUTED, ha="left", va="top",
                     linespacing=1.40, arrowprops=dict(arrowstyle="->", color="#9A9A9A", lw=0.5),
                     zorder=8)

    # ---- the field ----
    ax = fig.add_subplot(inner[1, 0])
    # Panel a gets its OWN left margin, wider than the shared one, so the per-row evidence rail
    # ("N calls, M errors") can live outside the data area. Inside, it sat on top of the thin-cell
    # underscores and on real error dots at extreme logits -- and those dots cannot be masked, because
    # they are errors.
    DX = 0.190
    for a_ in (axm, ax):
        bx = a_.get_position()
        a_.set_position([bx.x0 + DX, bx.y0, bx.width - DX, bx.height])
    ax.set_xlim(LO, HI); ax.set_ylim(len(RW), 0)
    for i, r in enumerate(RW):
        cnt = np.array(r["count"], float)
        rate = np.array([np.nan if v is None else v for v in r["rate"]], float)
        alpha = np.clip(np.log1p(cnt) / np.log1p(60.0), 0.0, 1.0)      # opacity = evidence
        rgba = cmap(np.clip(np.nan_to_num(rate) / VMAX, 0, 1))
        # Opacity alone does NOT retire a thin cell: alpha(1) = 0.293, so a single variant that happened to
        # be an error composited darker than 86% of the field. Cells under MIN_N are drawn as an outline
        # with no fill -- a grey fill would read as "moderate risk" beside the near-white low-rate cells.
        thin = (cnt > 0) & (cnt < MIN_N)
        rgba[..., 3] = np.where(cnt >= MIN_N, 0.15 + 0.85 * alpha, 0.0)
        ax.imshow(rgba.reshape(1, NB, 4), extent=[LO, HI, i + 0.92, i + 0.08],
                  aspect="auto", interpolation="nearest", zorder=2)
        bw = (HI - LO) / NB
        # Thin cells are marked by a hairline UNDERSCORE, not a boxed outline. Outlining 135 of 293 cells
        # drew a full grid across the sparse left half and read as content.
        # It sits at i+0.62, NOT i+0.90: at 0.90 it was 0.34 pt from the refusal corridor's bottom edge
        # against 0.485 pt of summed stroke, so 51 of 135 underscores FUSED with the corridor and the
        # mark that distinguishes "too few to rate" from "nothing sampled" was destroyed.
        for j in np.flatnonzero(thin):
            ax.plot([LO + j * bw + bw * 0.13, LO + (j + 1) * bw - bw * 0.13], [i + 0.62, i + 0.62],
                    color="#BDB7AF", lw=0.42, solid_capstyle="butt", zorder=3)
        # above VMAX the ramp is exhausted, so darkness stops tracking rate; hatch rather than let it lie
        for j in np.flatnonzero((cnt >= MIN_N) & (np.nan_to_num(rate) > VMAX)):
            ax.add_patch(Rectangle((LO + j * bw, i + 0.08), bw, 0.84, fc="none", ec="white",
                                   lw=0.0, hatch="////", zorder=4))
        t = r["refuse_logit_abs"]
        ax.add_patch(Rectangle((-t, i + 0.06), 2 * t, 0.88, fc="none", ec=BEDROCK, lw=0.55,
                               ls=(0, (2.4, 1.6)), zorder=6))
        # SHAPE AS WELL AS COLOUR. The two error classes are not two shades of one
        # quantity: a missed positive is a pathogenic variant called benign and an over-call is the
        # reverse, and which of the two a corridor catches is the panel's finding. Encoding that in
        # hue alone puts it beyond a greyscale print and beyond a deuteranope reader -- dark red and
        # orange are the classic pair to lose -- and both classes are drawn on a red field, so even
        # in colour the dark-red dots sit on cells of their own hue. Circle vs triangle survives all
        # of that. The marks are sized for a shape to read, at
        # 3.0/4.4 pt^2; the triangle carries the larger number because a triangle of equal AREA
        # reads smaller than a disc.
        for key, col, mk, sz in [("miss_logit", "#4A0910", "o", 3.0),
                                 ("fa_logit", POS, "^", 4.4)]:
            v = np.array(r[key], float)
            if len(v):
                ax.scatter(v, i + 0.50 + (rng.random(len(v)) - 0.5) * 0.46, s=sz, marker=mk,
                           color=col, alpha=0.85, linewidths=0.12, edgecolors='white', zorder=7,
                           rasterized=True)
        lab = r["species"] + (" *" if r["target_only"] else "")
        # The asterisk and the colour carry "never trained on"; the bold was emphasis on top of an
        # encoding that already worked, so it goes (style_gb rule: bold is for the panel letter).
        ax.text(-0.371, i + 0.50, lab, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=TICK,
                color=(FAIL if r["target_only"] else INK))
        # A RATE CHANNEL, because the shading cannot carry one. Rendered ink tracks sample size
        # (Spearman 0.983) far better than error rate (0.20): goat has the second-highest error rate of
        # the nine species and, being the smallest, draws the lightest row on the panel. Redirecting the
        # reader to "the printed counts" did not help either -- counts reproduce the same n-ordering.
        # The bar below is length-proportional to the species error rate and is the only mark here that
        # is comparable across rows.
        rmax = max(x["base_rate"] for x in RW)
        # The bar's KEY is emitted once, with the other two column headers, in the clear band above
        # the panel -- see the header block below. A second key was drafted here on the top row and
        # is not kept: the two overlapped by 62 pt^2, and one key per mark is the point.
        ax.barh(i + 0.50, 0.0675 * r["base_rate"] / rmax, left=-0.350, height=0.36,
                transform=ax.get_yaxis_transform(), color=FAIL, alpha=0.85, lw=0, zorder=8,
                clip_on=False)
        # the raw error count is derivable from n and the rate; the RATE is the comparable quantity
        ax.text(-0.013, i + 0.50, f"{r['n']:,} calls · {r['base_rate']:.1%} err",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=FOOT, color=MUTED, zorder=9)
        res = r.get("lift_resolved", True)
        ax.text(HI - 0.12, i + 0.50, (f"{r['lift']:.1f}×" if res else f"{r['lift']:.1f}× n.s."),
                ha="right", va="center", fontsize=FOOT,
                color=(CODING if res else MUTED), zorder=9)
    ax.axvline(0, color="#8E8880", lw=0.5, zorder=5)

    # ---- COLUMN HEADERS. Three marks in this panel carried no key anywhere -- not on the plate and
    # not in the legend -- and the journal's rule puts keys in the graphic. They were: the red bar
    # beside each species name (its error rate, the only mark here comparable across rows), the
    # right-hand "x" column (the selective lift), and the "n.s." that appears beside some of those
    # multipliers. A significance marker with no stated test is the worst of the three. They sit in
    # the empty band above row 0, left of and right of the existing "band: logit ..." guide, which
    # occupies x 168-355 pt and is not touched.
    # TWO LINES, both ending short of the marginal profile. As one line the key ran on under the
    # profile curve, which rises from the left edge of the field at this height, and the white plate
    # behind the words cut the curve in two.
    ax.text(-0.350, -2.35, "bar = that species' error rate", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=FOOT, color=MUTED, clip_on=False, zorder=9)
    ax.text(-0.350, -1.15, "* = never in the 1,001-bp training pool",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FOOT, color=MUTED,
            clip_on=False, zorder=9)
    # NO transform on these two: HI is a DATA x, and the lift column above is placed in data
    # coordinates too. Passing get_yaxis_transform() here would read HI - 0.12 as an AXES FRACTION
    # -- roughly 800 % of the axes width -- and put both lines off the right edge of the page,
    # where they vanish from the PDF entirely rather than failing any bounds check.
    # ONE line, in axes fractions, in the only band above this panel that is clear of ink. The
    # first attempt used two lines right-aligned over the column itself; both sat under the marginal
    # density curve, which is a PATH, so the span-collision gate scored them clean while the render
    # showed the curve running through the type. Measured on the plate, the band y 19-28 pt is free
    # of ink from x 60 to 351 pt -- 291 pt, and this line sets to 266 pt at 6.8 pt.
    # x = -0.350, the same axes fraction the error-rate bar and its key already use, so the header
    # starts at page x 60 pt and the 268 pt line ends at 328 -- inside the clear 60-351 run. (An
    # earlier -0.03 put it at 155-423 and straight through the "band: logit ..." guide: this axes
    # has a left rail, so its fraction 0 is at page x 164, not at the gridspec's left edge.)
    # The asterisk on goat, chicken and pig was printed with no key anywhere on the plate, so the
    # legend had to spend eighteen words saying what it meant -- the same rule violation as the
    # three marks above. It goes at the right end of the same clear band (x 335-470 pt is free
    # there; the "n = ..." subtitle sits a line higher, at y 11-19).
    ax.text(-0.350, 1.368, "\u00d7 = errors caught by refusing that species' least-confident 15%, "
            "vs random; n.s. spans 1\u00d7", transform=ax.transAxes,
            ha="left", va="center", fontsize=FOOT, color=MUTED, clip_on=False, zorder=9)

    # ---- reading guides. Every number quoted here is computed from the field, not typed. ----
    # No white stroke behind the guide any more (style_gb: no path effects behind text). It sits in
    # the empty left third of the field, where nothing is drawn for it to fight with.
    lab_kw = dict(fontsize=FOOT, zorder=10)
    # The empty left third of the field (0.19% of variants sit left of logit -6.5) is the only place on
    # this panel with room for guides, so it earns its width by carrying them.
    # The "where errors concentrate" claim is marked on the POOLED PROFILE, not inside the field: an
    # in-field callout needed a leader across the entire panel to reach the zone it named.
    # name the corridor on the row where it is widest, so the leader stays short
    iw = int(np.argmax([r["refuse_logit_abs"] for r in RW]))
    ax.annotate("dashed box =\nwhat this species refuses",
                xy=(-RW[iw]["refuse_logit_abs"], iw + 0.50), xytext=(LO + 0.25, iw + 0.10),
                color=BEDROCK, ha="left", va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="->", color=BEDROCK, lw=0.55,
                                connectionstyle="arc3,rad=0.10"), **lab_kw)
    ax.set_yticks([])
    # the boundary is named on its own tick rather than floated as a label -- anywhere it floated
    # it landed on the profile line or on the field
    ax.set_xticks([-8, -4, 0, 2, 4])
    ax.set_xticklabels(["-8", "-4", "0\ndecision boundary, p = 0.5", "2", "4"], linespacing=1.3)
    ax.tick_params(labelsize=FOOT)
    ax.set_xlabel("signed log-odds, logit(p)      ←  confidently negative        "
                  "confidently positive  →", fontsize=AXIS, labelpad=1)
    for sname in ("top", "right", "left"):
        ax.spines[sname].set_visible(False)
    ax.spines["bottom"].set_color("#8C8C8C")

    # Everything below the axis is laid out by MEASURED height, not by hand-tuned offsets -- the offsets
    # collided every time a caption's wording changed length.
    n_over = sum(1 for r in RW for c, v in zip(r["count"], r["rate"])
                 if c >= MIN_N and v is not None and v > VMAX)
    n_thin = sum(1 for r in RW for c in r["count"] if 0 < c < MIN_N)
    C = R["staircase"]["census"]
    y = _below(ax, -0.05)                                     # clears the x-axis label

    # The colour-key sentence, the mark key ("dark dots = ...") and the ramp swatch were all
    # legend prose: they defined the encoding rather than reporting a measurement, and they carried
    # the smallest type on the canvas. They move to the manuscript legend verbatim. What stays here
    # is the pair of MEASURED statements -- every number below is read from the artefact.
    del n_over, n_thin
    # The red bold paragraph that stood second here ("but refusal is not class-neutral: 731 of 893
    # errors are MISSED positives and only 26% of those are caught, against 77% of the 162 harmless
    # over-calls") argued the finding, in colour, in bold, and -- at 170 mm -- ran 0.5 mm PAST the
    # right page edge. The argument belongs to the legend and Additional file 1 Note S45. Its five
    # numbers do not: they are the census of the dots this panel draws, so they stay as one plain
    # grey line that counts the marks rather than interpreting them.
    # The wrap width is DERIVED from the axes box so the block cannot reach past the grid's right
    # edge again; it used to be the hand-set 1.310 that caused the overflow.
    _bx = ax.get_position()
    # Wrapped at the LEFT COLUMN's right edge, not at the grid's. Panel a is full width but the
    # block that hangs under it is not: panel c's letter sits in the middle gutter at x = 0.5715,
    # and a single 94-character line reached 0.5865, so the caption's last line and that letter
    # overlapped by 2.7 pt on the 173 mm canvas. Wrapping at 0.494 puts 37 pt of clear paper
    # between the block and the letter whatever the two rows' heights become; the line it costs
    # is paid for by the tightened leading below.
    _wmax = (0.494 - _bx.x0) / _bx.width + 0.310

    # ---- MARK KEY. Three marks on this panel had no key anywhere on the page: the two error
    # classes (whose mapping was moved to the manuscript legend) and the hairline underscore that
    # marks a thinly-sampled bin. With the error classes now carrying shape as well as hue the
    # mapping has to be legible from the figure itself, and the underscore -- which is the only
    # thing distinguishing "sampled, too few to rate" from "nothing here at all" -- has never been
    # named. A key that labels a mark is canvas text under style_gb; it is not commentary.
    _pb = ax.text(0.0, 0.0, "Hg", transform=ax.transAxes, fontsize=FOOT)
    fig.canvas.draw()
    _lh = _axfrac(ax, _pb.get_window_extent(fig.canvas.get_renderer()).height)
    _pb.remove()
    _yk = y - _lh * 0.5
    _bw_mm = ax.get_position().width * W_MM
    _kx = lambda mm: -0.310 + mm / _bw_mm
    for _xm, _col, _mk, _sz, _lb in ((0.0, "#4A0910", "o", 7.0, "missed positive"),
                                     (27.0, POS, "^", 9.0, "over-call")):
        ax.scatter([_kx(_xm)], [_yk], s=_sz, marker=_mk, color=_col, transform=ax.transAxes,
                   clip_on=False, zorder=9, edgecolors="white", linewidths=0.3)
        ax.text(_kx(_xm + 1.7), _yk, _lb, transform=ax.transAxes, ha="left", va="center",
                fontsize=FOOT, color=INK, clip_on=False, zorder=9)
    ax.plot([_kx(45.0), _kx(48.6)], [_yk, _yk], transform=ax.transAxes, color="#BDB7AF", lw=0.42,
            solid_capstyle="butt", clip_on=False, zorder=9)
    ax.text(_kx(50.3), _yk, f"bin holds 1\u2013{MIN_N - 1} calls: too few to rate",
            transform=ax.transAxes, ha="left", va="center", fontsize=FOOT, color=INK,
            clip_on=False, zorder=9)
    # 2.6 -> 1.0 pt here and 2.6 -> 1.4 pt on the two caption lines below. On the 173 mm canvas
    # this block's last line sat 2.7 pt into panel c's letter: the block hangs from panel a's foot
    # in POINTS while the row beneath it moved up in millimetres. hspace is not the lever -- raising
    # it shrinks the axes as fast as it opens the gap, so the letters barely move (1.06 -> 1.22
    # bought 0.5 pt and cost enough axes height to collide eight of panel a's own cell labels).
    # The 4 pt comes out of leading that was never needed: at 2.6 pt the pitch was 1.38x the type
    # size, and 1.4 pt still leaves 1.32x.
    y = y - _lh - _axfrac(ax, 1.0 / 72.0 * fig.dpi)

    for txt, kw in [
        # Two decimals, not one: the deposited lift is 2.3542 and ".1f" printed it as "2.4×" inside
        # the image while the legend and the Results both said 2.35×, so the figure disagreed with
        # its own caption on the same page. The value is read from the artefact either way.
    ]:
        y = _line(ax, y, txt, x=-0.310, width=_wmax, gap_pt=1.4, **kw)


# ---------------------------------------------------------------- b. the tie
def panel_b(ax):
    """The tie DRAWN, not asserted: the four bootstrap posteriors of delta-ECE, superimposed.
    If a cross-species map were better than two parameters, these would sit left of zero. They pile on it."""
    from scipy.stats import gaussian_kde
    order = ["width10", "width15", "mass10", "mass15"]
    lab = {"width10": "equal-width, 10 bins", "width15": "equal-width, 15 bins",
           "mass10": "equal-mass, 10 bins", "mass15": "equal-mass, 15 bins"}
    keys = [k for k in R["paired_verdicts"] if k.endswith("__macro")]
    keys = sorted(keys, key=lambda k: order.index(k.split("__")[1]))
    allx = np.concatenate([R["paired_verdicts"][k]["draws"] for k in keys])
    lo, hi = np.percentile(allx, [0.3, 99.7]); pad = 0.12 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 400)
    shades = ["#7FB3D5", "#4E8FC0", CODING, "#004E7C"]
    # DASH PATTERN AS WELL AS SHADE. The four shades are steps of one blue ramp, which alone
    # is a SEQUENTIAL encoding on a categorical variable: the four estimators are not ordered
    # and are not more or less of anything. Worse, four tints of one hue is the encoding that fails
    # first under greyscale, and the panel's whole point is which of the four posteriors is the one
    # that leaves the null -- a reader who cannot tell mass15 from width15 cannot check the claim.
    # Each curve therefore carries a dash pattern too, and the legend below the axes draws the same
    # handle, so the key and the curve match by pattern and not only by tint.
    styles = ["-", (0, (4.0, 2.0)), (0, (1.2, 1.6)), (0, (5.0, 1.6, 1.2, 1.6))]
    peak = 0
    for k, c, st in zip(keys, shades, styles):
        d = np.array(R["paired_verdicts"][k]["draws"])
        dens = gaussian_kde(d)(xs); peak = max(peak, dens.max())
        ax.fill_between(xs, 0, dens, color=c, alpha=0.26, lw=0, zorder=2)
        ax.plot(xs, dens, color=c, lw=1.0, ls=st, zorder=3, label=lab[k.split("__")[1]])
    ax.axvline(0, color=BEDROCK, lw=1.0, zorder=5)
    # left of the rule, so it cannot touch the legend that occupies the upper right
    ax.text(-0.0004, peak * 1.24, "no difference", fontsize=FOOT, color=BEDROCK, ha="right",
            va="bottom")
    frac = float(np.mean(allx < 0))
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, peak * 1.48)
    ax.set_yticks([])
    # SHORTENED, NOT WRAPPED. _below() places the legend under whatever height the x-label takes,
    # so a second line would push the b/c key down into panel e's letter. One shorter line keeps
    # the key high. The full wording is in the manuscript legend; the axis needs the quantity, not
    # the sentence.
    ax.set_xlabel("Δ ECE   (isotonic − sigmoid), macro over species", fontsize=AXIS)
    _spines(ax, keep=("bottom",))
    # Legend below the axes, not inside it: in the upper right it lay across the two widest posteriors.
    y_lg = _below(ax, -0.05)
    # handlelength 2.4, not 1.3: at 1.3 the handle is 8.8 pt and the dash-dot cycle is 9.4 pt, so the
    # key would draw a pattern the curve does not have.
    # FLOOR, not ANNOT. This is a dense multi-line key in a fixed reservation, and FLOOR is the
    # size the house style provides for exactly this case -- the same decoupling that lets
    # Figure 2 keep 49 rows.
    # Centred at 0.40, not 0.5: at 0.5 its right-hand column touched the first entry of panel c's
    # key, which shares this row.
    lgb = ax.legend(fontsize=SG.FLOOR, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.40, y_lg), ncol=2, handlelength=2.4,
                    columnspacing=1.2, labelspacing=0.24, borderpad=0.0)
    fig_ = ax.figure; fig_.canvas.draw()
    y_cap = (lgb.get_window_extent(fig_.canvas.get_renderer())
             .transformed(ax.transAxes.inverted()).y0) - 0.035
    # Emitted last, wrapped to the panel's own width. COUNTED, not asserted: "all four straddle zero" was
    # true only while ece_mass split tied scores on row order. With ties binned by value, mass15 leaves
    # the null and favours transfer -- state it rather than let a referee find it.
    #
    # What this line no longer does is argue. It read "... — 88% of the pooled mass favours transfer,
    # A COIN-FLIP RATHER THAN AN EFFECT", in bold: the reading of the posteriors, not the posteriors.
    # That clause is the legend's and Note S48's to make. The three counts stay, because they measure
    # the four densities drawn above and appear nowhere else on the canvas.
    NR = R["null_robustness"]
    n_ind, n_est = NR["n_indistinguishable"], NR["n_macro_estimators"]
    # Name the four objects the way the legend and Table S22 name them, not "posteriors":
    # everywhere else in the paper -- including this figure's own legend -- "posterior" means the
    # calibration map, so "4 posteriors" here reads as four calibration maps rather than four
    # bootstrap densities of one contrast. And spell the dissenter with the key's own label:
    # "mass15" is a raw dict key matching none of the four labels drawn in the key below the axes.
    dis = ", ".join(lab[k] for k in NR["dissenting"]) or "none"
    _ = (n_ind, n_est, frac, dis)   # stated in the Results; no longer set on the canvas


# ---------------------------------------------------------------- c. what abstention buys
def panel_c(ax):
    A = R["abstention"]
    cov = np.array(A["isotonic_LOSO"]["coverage"])
    hon = np.array(A["isotonic_LOSO"]["macro_error"])
    rnd = np.array(A["isotonic_LOSO"]["macro_random"])
    orc = np.array(A["oracle_isotonic"]["macro_error"])
    # the value refusal actually delivers = the area between the honest curve and the random control
    ax.fill_between(cov, hon, rnd, where=(hon <= rnd), color=CODING, alpha=0.16, lw=0, zorder=1,
                    interpolate=True)
    # Three curves identified by a legend, not by inline labels. The inline labels had to sit ON the
    # curves they named, and at low coverage the curves crowd together, so no placement cleared both
    # its neighbours and the callouts. A legend in the empty corner is collision-free by construction.
    ax.plot(cov, rnd, color=SUB, lw=1.0, ls=(0, (3, 1.6)), zorder=3, label="random (control)")
    ax.plot(cov, orc, color=POS, lw=1.0, ls=(0, (1, 1.4)), zorder=3,
            label="oracle (needs labels)")
    ax.plot(cov, hon, color=CODING, lw=1.0, zorder=5, label="transferred (honest)")
    i = int(np.argmin(hon))
    ax.scatter([cov[i]], [hon[i]], s=22, color=CODING, zorder=6, edgecolors="white", lw=0.7)
    # The key sits below the axes: inside them it would cross the control and oracle curves.
    # No callouts: the minimum is marked on the curve, and the manuscript legend says what it is.
    j = list(cov).index(0.45)
    _ = (i, j)
    _P = R["staircase"]["per_species"]
    _wk = min(_P, key=lambda k: _P[k]["self_knowledge_auroc"])
    _fr = _P[_wk]["n_err"] / R["staircase"]["census"]["n_errors"]
    ax.set_xticks([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    ax.set_xlim(1.03, 0.27); ax.set_ylim(0.026, 0.0800)
    ax.set_xlabel("coverage  (fraction still called)", fontsize=AXIS)
    ax.set_ylabel("macro error", fontsize=AXIS)
    y_lg = _below(ax, -0.05)                     # measured: below the x-label, whatever height it took
    # Handles and column gaps tightened (1.7/0.45/1.0 -> 1.2/0.35/0.8) and the anchor clamped
    # below. At the corrected type scale the three-column key measured 232 pt against 166 pt of
    # axes and, centred at 0.48, its last entry ran PAST THE PAGE EDGE: "transferred (honest)"
    # was cropped to "transferred (hones" in the PDF. Narrowing the furniture rather than the
    # words keeps all three series named at 6.8 pt on one line.
    _lg_anchor = 0.48
    lg = ax.legend(loc="upper center", bbox_to_anchor=(_lg_anchor, y_lg), ncol=3, frameon=False,
                   fontsize=SG.FLOOR, handlelength=1.2, handletextpad=0.35, columnspacing=0.8,
                   borderpad=0.0)
    # The legend's HANDLES carry the series colour; its text does not need to repeat it, and
    # coloured label text is emphasis rather than encoding. Set the words in ink.
    for t_ in lg.get_texts():
        t_.set_color(INK)
    fig_ = ax.figure; fig_.canvas.draw()
    _r = fig_.canvas.get_renderer()
    _lim = 0.972 * fig_.get_window_extent(_r).width      # the plate's own right margin
    _gb = lg.get_window_extent(_r)
    if _gb.x1 > _lim:
        lg.set_bbox_to_anchor(
            (_lg_anchor - (_gb.x1 - _lim) / ax.get_window_extent(_r).width, y_lg),
            transform=ax.transAxes)
        fig_.canvas.draw()
    y_cap = (lg.get_window_extent(fig_.canvas.get_renderer())
             .transformed(ax.transAxes.inverted()).y0) - 0.035
    # Was a red italic "Caveat: ... and there confidence does not predict error" -- a warning about
    # how to read the panel, which is legend copy. What stays is the plain arithmetic behind it: the
    # weight, the error share and the self-knowledge AUROC, none of which is printed anywhere else.
    _line(ax, y_cap, f"macro average over {len(_P)} species", fontsize=FOOT, color=MUTED)
    _spines(ax)


# ---------------------------------------------------------------- d. per-species price
def panel_dm(ax):
    """The transfer, drawn per variant rather than as nine ECE rows.

    Nine species x three estimators on an ECE axis would be 27 marks summarising 11,130
    calls, and would not answer the question the section actually asks -- does recalibrating
    change what gets CALLED? Here every variant is placed at its probability under the global
    sigmoid against its probability under leave-one-species-out isotonic transfer.

    The two disagree on the probability scale (mean absolute difference 0.052), the cloud lies along
    the identity, and only 68 of 11,130 variants -- 0.61 % -- cross the decision boundary, every
    one of them from sigmoid-positive to transfer-negative.

    NO ECE IS PRINTED ON THIS PANEL, and that is deliberate, for two reasons.

    First, a POOLED ECE misleads. Pooling nine species whose base rates run from 4 % to 30 % lets one
    species' over-confidence cancel another's under-confidence inside a shared reliability bin, so
    the transfer appears to cut calibration error tenfold. This module's own header rejects pooled
    ECE for exactly that reason. The pooled gap is also three times the upper bound the Results
    place on any advantage of the transfer, and printing it would contradict panel b, which plots
    the paired macro delta-ECE entirely between -0.015 and +0.005.

    Second, the macro pair cannot be computed here in a form that agrees with the supplement.
    Table S23 reports macro trivial 0.0551 against isotonic
    0.0532 on the equal-width-10 estimator. Recomputing that estimator on this panel's own source,
    reports/_recon_pervariant_trust.parquet, gives 0.0506/0.0501 from the global_sigmoid column and
    0.0541/0.0501 from platt_LOSO, and matches Table S23 to four decimals in none of the nine
    species under either. reports/calibration_transfer.parquet, which tabulates the same quantity,
    disagrees with both and with S23 (goat isotonic 0.0555 against S23's 0.0564). The isotonic
    column IS the right one -- it carries the 248 distinct values Note S44 attributes to it -- so
    this is a lineage gap in the trivial arm, not a wrong column. Printing any of the three pairs
    on the plate would put a fourth number into circulation that no table backs.

    What this panel is FOR is the decision boundary, and that it computes from its own data: 68 of
    11,130. The calibration comparison belongs to panel b, which draws it as the paired posterior
    it is, and to Table S23. The mean absolute shift on the probability scale, 0.052, is also from
    this panel's own data. Those 68 are drawn individually. That is the whole of "no robust transfer gain":
    not that the transfer does nothing, but that what it does is rescale confidence rather than
    change the call.
    """
    import pandas as pd
    from pathlib import Path as _P
    d = pd.read_parquet(_P("reports") / "_recon_pervariant_trust.parquet")
    a = d.global_sigmoid.to_numpy(); b = d.isotonic_LOSO.to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, lab = a[ok], b[ok], d.label.to_numpy()[ok]


    flip = (a >= .5) != (b >= .5)
    ax.axhspan(0.5, 1.0, xmin=0, xmax=0.5, color="#C1443E", alpha=.05, lw=0, zorder=1)
    ax.axvspan(0.5, 1.0, ymin=0, ymax=0.5, color="#C1443E", alpha=.05, lw=0, zorder=1)
    ax.hexbin(a[~flip], b[~flip], gridsize=44, extent=(0, 1, 0, 1), cmap="Greys", bins="log",
              mincnt=1, linewidths=0, zorder=2)
    ax.scatter(a[flip], b[flip], s=4.5, marker="o", facecolor="#C1443E", edgecolor="none",
               lw=0, zorder=5)
    ax.plot([0, 1], [0, 1], color=INK, lw=.7, ls=(0, (4, 2)), zorder=4)
    ax.axhline(.5, color=MUTED, lw=.5, ls=(0, (2, 2)), zorder=3)
    ax.axvline(.5, color=MUTED, lw=.5, ls=(0, (2, 2)), zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1.0]); ax.set_yticks([0, .25, .5, .75, 1.0])
    # "0", not "0.00", at the origin, where the two axes' "0.00" labels would meet. The other ticks follow the
    # same shortest form (0.25, 0.5, 0.75, 1), the form the probability axes of the main figures use,
    # so no axis mixes a bare 0 with two-decimal labels.
    from matplotlib.ticker import FuncFormatter as _FF
    for _axis in (ax.xaxis, ax.yaxis):
        _axis.set_major_formatter(_FF(lambda v, _: "0" if abs(v) < 1e-9 else "%g" % v))
    ax.set_xlabel("probability under one global sigmoid", fontsize=AXIS)
    ax.set_ylabel("probability under leave-one-species-out\nisotonic transfer", fontsize=AXIS,
                  linespacing=1.35)
    _spines(ax)
    # Three short lines that end left of the diagonal: the longer wording ran through it.
    ax.text(0.035, 0.965, "shaded: the two disagree\nred: a call crossing the\ndecision boundary",
            transform=ax.transAxes, fontsize=FOOT, ha="left", va="top", color=INK,
            linespacing=1.4)
    ax.text(0.965, 0.035, "one hexagon =\na count of calls", transform=ax.transAxes,
            fontsize=FOOT, ha="right", va="bottom", color=MUTED, linespacing=1.3)


def panel_e(ax):
    """POSITIVE-CLASS coverage under BOTH split-conformal constructions, nominal 90%.

    The earlier version plotted only the MARGINAL arm and titled its pathogenic collapse as the finding.
    That is indefensible: build_conformal.py states "Mondrian is reported because marginal coverage skews
    under class imbalance", so the collapse is the documented artifact of the construction the pipeline
    itself rejects -- and abstain_marg is 0.000 for all nine species, i.e. the plotted predictor never
    abstains at all, inside a figure whose thesis is that abstention is the deliverable.

    The real finding is the TRADE, and it is the stronger result: the class-conditional construction does
    reach nominal on the class that matters, but only by declining to answer on most of the cohort."""
    C = R["conformal"]
    idx = sorted(range(len(C["sp"])), key=lambda i: C["n"][i])
    sp = [C["sp"][i] for i in idx]
    p_marg = np.array([C["cov_path_marg"][i] for i in idx])
    p_mond = np.array([C["cov_path"][i] for i in idx])
    ab_mond = np.array([C["abstain"][i] for i in idx])
    ab_marg = np.array([C["abstain_marg"][i] for i in idx])
    xs = np.arange(len(sp))
    ax.axhspan(0, 0.90, color="#F6EDEC", zorder=0)
    for x, a_, b_ in zip(xs, p_marg, p_mond):
        ax.plot([x, x], [a_, b_], color="#C9C4BC", lw=0.9, zorder=2, solid_capstyle="round")
    ax.scatter(xs, p_marg, s=20, color=FAIL, marker="v", zorder=5, edgecolors="white", lw=0.4,
               # "— abstains on 0%" is dropped from the KEY: it is a finding, it reads like a
               # bug beside a series name, and the legend now states it. Every class-conditional
               # point already prints its own abstention rate along the bottom of the panel.
               label="marginal")
    ax.scatter(xs, p_mond, s=18, color=CODING, zorder=5, edgecolors="white", lw=0.4,
               # No range in the label. Every Mondrian point now carries its OWN abstention rate
               # along the bottom of the panel, which is the whole point of finding 39; a min-max
               # in the legend re-summarises what is already priced per species, and at 113 pt it
               # made the legend wide enough to start at x 343 and run into the "nominal 90%
               # target" annotation at 354. Without it the widest entry is the marginal one, 83 pt,
               # and the legend starts at 373.
               label="class-conditional")
    # KEYED, NOT CAPTIONED. The rule is a legend entry like the two series rather than a floating
    # caption beside the key, which would read as one run of words with it, and the whole key
    # sits above the data, where the dashed rule cannot strike through it.
    ax.axhline(0.90, color=BEDROCK, lw=1.0, ls=(0, (4, 2)), zorder=6, label="nominal 90%")
    n_ok = int((p_mond >= 0.90).sum())
    # Counting all nine flatters the arm: six are the calibration species, where coverage at or
    # above nominal follows by construction and is not evidence of transfer. The out-of-sample
    # evidence is the three held-out species, which is what Results reports, so lead with it.
    HELD_OUT = ("goat", "chicken", "pig")
    ho = [i for i, s_ in enumerate(sp) if s_.lower() in HELD_OUT]
    n_ho_ok = int((p_mond[ho] >= 0.90).sum())
    # rotation_mode="anchor": without it matplotlib rotates each label about its centre and
    # then right-aligns the ROTATED box, which slid "goat" and "chicken" into each other by
    # 15 pt^2 -- the one text collision this figure carried, in the shipped render as well.
    # Anchoring rotates about the alignment point instead, so each label hangs from its own tick.
    ax.set_xticks(xs)
    # 90 degrees, not 42. Nine species names under a half-width panel do not fit on a 42-degree
    # rake: "goat" and "chicken" overlapped by 15 pt^2 in the shipped render and still touched
    # once anchored. Set vertical they hang cleanly from their own ticks at any panel width.
    # rotation_mode="anchor" applies the alignment BEFORE the rotation, which pushed the
    # vertical labels up into the axes and over the panel's shaded field. Default mode
    # rotates first and then aligns the rotated box, so the labels hang below the axis.
    _tl = ax.set_xticklabels([s_ + (" *" if s_.lower() in HELD_OUT else "") for s_ in sp],
                             fontsize=TICK, rotation=90, ha="center", va="top")
    for _t, _s in zip(_tl, sp):
        _t.set_color(FAIL if _s.lower() in HELD_OUT else INK)
    ax.set_ylim(0, 1.19); ax.set_xlim(-0.5, len(sp) - 0.5)
    # Coverage is bought with silence and the price is not the same in every cell: 43% in human,
    # 70% in cat. One pooled figure charges all nine the same 61%, which is the accounting error
    # this paper exists to name, so every Mondrian point carries its own rate.
    # get_xaxis_transform: x in DATA, y in AXES FRACTION -- get_yaxis_transform is the one whose x
    # is an axes fraction. y = 0.02 clears every mark (the lowest is goat's marginal triangle at
    # 0.222) and the 0.90 shading boundary. Must run AFTER set_ylim/set_xlim: the blended
    # transform reads the limits.
    for x_, ab_ in zip(xs, ab_mond):
        ax.text(x_, 0.02, f"{ab_:.0%}", transform=ax.get_xaxis_transform(),
                fontsize=FOOT, color=CODING, ha="center", va="bottom", zorder=7)
    # The row of percentages is keyed where it is read: over its right-hand end, in the empty band
    # under the lowest triangle of those columns.
    ax.text(len(xs) - 0.55, 0.150, "class-conditional abstention", transform=ax.get_xaxis_transform(),
            fontsize=FOOT, color=CODING, ha="right", va="bottom", zorder=7)
    # y=0.33, not 0.40: at 170 mm the rotated label's top reached into panel e's own letter chip --
    # the same chip-eats-ylabel defect Fig 5 shipped once (memory figure-qa-tooling blind spot 4).
    ax.set_ylabel("positive-class coverage", fontsize=AXIS, y=0.33)
    # UPPER right, not lower. The per-species abstention rates added above are printed along the
    # bottom of this panel at y = 0.02, which is exactly where a lower-right legend sits: the
    # class-conditional entry overlapped seven of the nine rates. Measured on the plate, the only
    # band inside panel e with no ink at all is its top, y 300-320 pt, which holds both entries.
    _h, _l = ax.get_legend_handles_labels()
    _o = [_l.index(k) for k in ("nominal 90%", "marginal", "class-conditional") if k in _l]
    ax.legend([_h[k] for k in _o], [_l[k] for k in _o], fontsize=FOOT, frameon=False,
              loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, handletextpad=0.3,
              columnspacing=1.1, borderaxespad=0.0, labelspacing=0.2)
    _spines(ax)
    # No caption under this panel. Its numbers are readings of the marks -- the spread of the red
    # triangles and how many blue dots clear the dashed rule -- and the manuscript legend states
    # them; each point carries its own abstention rate.
    _ = (p_marg, n_ok, n_ho_ok)     # stated in the manuscript legend


# ---------------------------------------------------------------- footer
# panel_foot() DELETED, together with the bottom-anchored methods paragraph it returned.
# The "Claim" / "Caveats" card and the methods sentence were nine and three lines of
# explanatory prose respectively -- legend copy printed on the canvas at the smallest type in
# the figure. They are reproduced verbatim in the manuscript legend; deleting the whole footer
# row is what buys panels a-e the vertical room to be legible at 6 pt.


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    # Three rows, not four: the footer prose row is gone. Positions below are set in MILLIMETRES
    # off the canvas edges so the panel marks keep their printed size on the taller page.
    def _fy(mm_from_top):
        return 1.0 - mm_from_top / H_MM

    # Grid geometry:
    #   top 9.8 mm. There is no on-canvas title, so the grid starts just under the panel-a letter.
    #   hspace 1.06. The gap between rows has to hold the block under panel a AND the next row's
    #     panel letter; a smaller gap puts panel c's letter on the last line of that block.
    #     hspace is a fraction of the AVERAGE AXES HEIGHT, so the gap comes out of the axes rows
    #     and has to be rescaled whenever the canvas height changes.
    #   bottom 0.125 and row 0 at 1.22. Panel a's nine species rows get a 9.45 pt pitch, +0.50 pt
    #     clear of the label glyph box, and the cropped plate is 168.1 mm, inside the 173 mm
    #     graphic budget with 4.9 mm to spare.
    gs = GridSpec(3, 2, height_ratios=[1.22, 0.48, 0.85], hspace=1.06, wspace=0.34,
                  left=0.150, right=0.955, top=_fy(9.8), bottom=0.125)

    panel_a(fig, gs[0, :])
    # Panel a's marker is placed in FIGURE coordinates because the panel is two stacked axes with
    # its own left rail; b-e use _plab and the axes transform. Same marker either way.
    # x = 0.111, the figure-coordinate position of the -0.115 axes offset _plab uses for b-e, so
    # all five letters sit in one column instead of a alone hanging over the grid's left edge.
    from .style_gb import letter_x as _letter_x
    _mark(fig.text, _letter_x(fig), _fy(6.2), "a",
          f"n = {PV.height:,}  ·  cells ordered by sample size", x_right=0.955)
    axb = fig.add_subplot(gs[1, 0]); panel_b(axb)
    axc = fig.add_subplot(gs[1, 1]); panel_c(axc)
    axdm = fig.add_subplot(gs[2, 0]); panel_dm(axdm)
    axe = fig.add_subplot(gs[2, 1]); panel_e(axe)
    # Both panels of a row must be DRAWN before either letter is placed: _plab_row measures the
    # clearance each panel needs and takes the row maximum.
    _plab_row([(axb, "b"), (axc, "c")])   # "No robust gain over a pooled sigmoid" / "…but refusing does"
    _plab_row([(axdm, "d"), (axe, "e")])  # "Why it fails…" / "Nominal coverage is bought with silence"

    # NO on-canvas title and NO subtitle (src/ccs/style_gb.py rule 1). The manuscript legend
    # carries the title ("Flexible calibration shows no robust transfer gain; confidence has
    # limited selective utility") and the cohort sentence ("Evo 2-40B zero-shot on 11,130
    # cross-species disease variants..."), set by the typesetter in the journal's own face.
    os.makedirs("reports/figures", exist_ok=True)
    base = "reports/figures/Figure4_trust"
    # Nothing on this canvas carries a path effect any more (style_gb: no strokes behind text), so
    # every string is already in the PDF text layer and this call now adds zero twins. Kept as the
    # guard it is: if a halo is ever reintroduced, the string does not silently leave the text layer.
    restore_text_layer(fig)
    fig.savefig(base + ".png", dpi=600, facecolor="white")
    # GB: figures are "closely cropped to minimise the amount of white space surrounding the
    # illustration". Crop the PAGE at the save, to the drawn extent plus 3 mm, and only ever
    # upward from the BOTTOM edge: the width stays the declared 170 mm, the top edge does not
    # move, and no artist is re-placed. This is NOT the H_MM change FIGURES.md warns about --
    # nothing authored in millimetres from the top is touched. A plate already tighter than
    # 3 mm keeps its own margin (the clamp at 0).
    _y0 = min(max(0.0, fig.get_tightbbox(fig.canvas.get_renderer()).y0 - 3.0 * MM),
              fig.get_figheight())
    from .style_gb import tidy_minus as _tm
    _tm(fig)   # ASCII hyphen -> U+2212 in numeric labels
    fig.savefig(base + ".pdf", dpi=600, facecolor="white", bbox_inches=mpl.transforms.Bbox(
        [[0.0, _y0], [fig.get_figwidth(), fig.get_figheight()]]), metadata={"CreationDate": None})
    fig.savefig(base + "_preview.png", dpi=200, facecolor="white")
    print(f"wrote {base} at {W_MM}x{H_MM} mm (authored 1:1)")
    return fig


if __name__ == "__main__":
    main()
