"""Panels D, H, G, F, E. All values read from the verified sources; nothing is typed.

 D  composition: AUROC over four nested strata of one cohort, ordered categories + 95% CI
 H  readout: 1,001 -> 8,192 bp slope chart over the same four strata, with a delta column
 G  separation-vs-discrimination plane: AUROC vs Cohen's d  (Figure 2/atlas, not Figure 3)
 F  conservation frontier: Evo 2 vs GERP dumbbells on one AUROC axis, sorted by delta
 E  distance-null: per-species AUROC against divergence from human

D, H, E and F draw quantities as positions and lengths a reader can measure, not as areas or
line widths, and the canvas carries marks and keys rather than sentences arguing the paper's
case. See src/ccs/style_gb.py for the contract.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import fig2_style as S
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG

S_ALL, S_COD, S_NC, S_MIS = ("ALL (unmatched, as published)", "CODING-only", "non-coding only",
                             "MISSENSE-only (type-matched)")

# The four strata of panels a and b are the same four objects, so they carry the same four hues and
# a reader can carry an identity across the two panels. Okabe-Ito: blue, bluish green, orange,
# reddish purple -- distinguishable under every common colour-vision deficiency. Colour is never
# the only cue here: each stratum is also named, on its category tick in a and beside its line in b.
STRATA = ["ALL", "coding", "non-coding", "missense"]
STRAT_KEY = {"ALL": S_ALL, "coding": S_COD, "non-coding": S_NC, "missense": S_MIS}
STRAT_COL = {"ALL": "#0072B2", "coding": "#009E73",
             "non-coding": "#E69F00", "missense": "#CC79A7"}


def _anno(ax, txt, xy, xytext, col=S.CAP, fs=SG.ANNOT):
    ax.annotate(txt, xy=xy, xytext=xytext, fontsize=fs, color=col, ha="left", va="center", zorder=9,
                linespacing=1.45,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.7, connectionstyle="arc3,rad=0.2"))


def _seat(placed, box):
    """True if `box` (x0, y0, x1, y1, data units) misses every box already committed."""
    return not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                   for q in placed)


# ---------------------------------------------------------------- D : COMPOSITION
def panel_D(ax):
    """AUROC over four nested strata of one cohort: ordered categories, dot and 95% interval.

    The four strata are subsets of one 9,544-variant cohort -- coding and non-coding partition it,
    missense sits inside coding -- so they are not points on a continuum, and an x-axis such as
    "% positive -> consequence-matched" would invite a reader to interpolate between them. They
    are ordered categories, in the legend's order. The composition each category stands for is
    printed under it, as an n and a positive rate, rather than encoded as a position or a radius.
    """
    tm = pl.read_parquet("reports/type_matched_atlas.parquet")
    g = {r["stratum"]: r for r in tm.iter_rows(named=True) if r["readout"].startswith("8192")}
    rows = [(k, g[STRAT_KEY[k]]) for k in STRATA]
    FLOOR = g[S_MIS]["auroc"]
    # Bottom sits below every plotted bound: the missense interval runs to 0.788, and a limit above
    # that crops a real interval rather than dead space.
    ax.set_xlim(-0.62, 3.62); ax.set_ylim(0.770, 1.008)
    for yv in (0.80, 0.85, 0.90, 0.95, 1.00):
        ax.axhline(yv, color=SG.GRID, lw=0.5, zorder=0)
    # The missense value carried back across the other three categories as a light reference rule.
    # It starts at the missense dot and stops there rather than running the full width, so it does
    # not strike through the value printed beside that dot; and it is not labelled with its own
    # number, which is printed once, on the mark the rule comes from.
    ax.plot([-0.62, 2.90], [FLOOR, FLOOR], color=STRAT_COL["missense"], ls=(0, (4, 2)), lw=0.7,
            zorder=1)
    # Seated at the far left, the one stretch of the rule that no interval crosses: the
    # non-coding and missense intervals both run through 0.819 at their own columns.
    ax.text(-0.58, FLOOR + 0.005, "missense floor", fontsize=SG.ANNOT, color=SG.MUTED,
            ha="left", va="bottom", zorder=6)
    for i, (name, r) in enumerate(rows):
        col = STRAT_COL[name]
        lo, hi, au, n, pos = r["lo"], r["hi"], r["auroc"], r["n"], r["pos"]
        ax.plot([i, i], [lo, hi], color=col, lw=1.0, zorder=3)
        for yy in (lo, hi):
            ax.plot([i - 0.085, i + 0.085], [yy, yy], color=col, lw=1.0, zorder=3)
        ax.scatter([i], [au], s=24, color=col, edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(i + 0.14, au, f"{au:.3f}", fontsize=6.8, color=SG.INK, ha="left", va="center",
                zorder=6)
        # Composition, under the category it describes. Printing the positive rate here keeps
        # the number without implying a continuum on the x-axis.
        tr = ax.get_xaxis_transform()
        ax.text(i, -0.068, f"n = {n:,}", fontsize=SG.ANNOT, color=SG.MUTED, ha="center", va="top",
                transform=tr, clip_on=False)
        ax.text(i, -0.108, f"{100 * pos / n:.0f}% positive", fontsize=SG.ANNOT, color=SG.MUTED,
                ha="center", va="top", transform=tr, clip_on=False)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([n for n, _ in rows], fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("AUROC", fontsize=8.0, color=S.CAP); ax.tick_params(axis="y", labelsize=7.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # THE AXIS IS TRUNCATED AND SAYS SO. It starts at 0.770 -- not at 0, and not at the 0.5 a
    # reader expects an AUROC axis to be anchored to -- which makes the 0.978-to-0.819 drop look
    # like a fall to the floor of the measurement. The convention for a broken axis is the glyph,
    # not a footnote: the spine is interrupted just above the bottom and two slanted strokes mark
    # the break. Drawn in axes coordinates so it stays put at any panel size, and unclipped so it
    # can straddle the spine.
    _brk(ax)


def _brk(ax, y0=0.034, gap=0.026, dx=0.011, dy=0.009):
    """The broken-axis glyph: interrupt the left spine and cross it with two slanted strokes.

    Drawn on the spine itself rather than beside it, because a break mark that does not interrupt
    the rule it marks reads as a decoration. y0/gap are axes fractions, so the glyph keeps its
    proportions whatever box the panel is given.
    """
    col = ax.spines["left"].get_edgecolor()
    lw = ax.spines["left"].get_linewidth()
    ax.add_patch(plt.Rectangle((-dx * 0.55, y0 - 0.004), dx * 1.10, gap + 0.008,
                               transform=ax.transAxes, facecolor="white", edgecolor="none",
                               zorder=7, clip_on=False))
    for yy in (y0, y0 + gap):
        ax.plot([-dx, dx], [yy - dy, yy + dy], transform=ax.transAxes, color=col, lw=lw,
                solid_capstyle="butt", zorder=8, clip_on=False)


# ---------------------------------------------------------------- H : READOUT
def panel_H(ax):
    """The same four strata across the two measured readouts, as a slope chart.

    Only two readouts are measured, so the panel plots two endpoints per stratum and joins them at
    one uniform weight: a line width scaled with log(n) would make each stratum a tapered wedge
    whose thickness a reader could not compare. Panel a prints n for these strata, and the delta
    column on the right states each stratum's change in AUROC.
    """
    tm = pl.read_parquet("reports/type_matched_atlas.parquet")
    g8 = {r["stratum"]: (r["auroc"], r["n"]) for r in tm.iter_rows(named=True)
          if r["readout"].startswith("8192")}
    g1 = {r["stratum"]: r["auroc"] for r in tm.iter_rows(named=True)
          if r["readout"].startswith("1001")}
    x1, x2 = np.log2(1001), np.log2(8192)
    x_name = x1 - 0.14                      # stratum name + n, right-aligned, left of the 1,001 end
    x_del = x2 + 0.92                       # delta column, right-aligned, right of the 8,192 end
    # Derived, so no plotted value can be clipped away silently.
    _ymin = min(min(g1.values()), min(v[0] for v in g8.values())) - 0.02
    ax.set_xlim(x1 - 1.30, x2 + 1.00); ax.set_ylim(_ymin, 1.012)
    for yv in np.arange(0.65, 1.001, 0.05):
        ax.axhline(yv, color=SG.GRID, lw=0.5, zorder=0)
    for xv in (x1, x2):
        ax.axvline(xv, color=SG.GRID, lw=0.5, zorder=0)
    # No bedrock rule here. A horizontal line at the missense value would lie along the missense series
    # itself -- the one stratum that does not move -- in the same hue, and would run on through the delta
    # printed beside it. The flat line and its -0.006 say it; panel a carries the value.
    for k in STRATA:
        key = STRAT_KEY[k]; col = STRAT_COL[k]
        a1 = g1[key]; a8, _n = g8[key]
        # DOTTED, NOT SOLID. Only two readouts are measured, so the segment between
        # the endpoints is interpolation, not data: a solid rule invites a reader to read an AUROC
        # off it at some intermediate window length that was never scored. A dotted connector still
        # ties each stratum's two endpoints together and still shows the sign and steepness of the
        # change, while saying that nothing was measured in between.
        ax.plot([x1, x2], [a1, a8], color=col, lw=1.2, ls=(0, (1.1, 1.7)),
                dash_capstyle="round", zorder=3)
        ax.scatter([x1, x2], [a1, a8], s=18, color=col, edgecolors="white", lw=0.6, zorder=4)
        # One line per stratum, at its own 1,001-bp endpoint. The n is not repeated here: panel a
        # prints it for these same four strata, from these same rows.
        ax.text(x_name, a1, k, ha="right", va="center", fontsize=6.8, color=col, zorder=5)
        gain = a8 - a1
        ax.text(x_del, a8, f"{'+' if gain >= 0 else ''}{gain:.3f}", ha="right", va="center",
                fontsize=SG.ANNOT, color=col, zorder=5)
    ax.text(x_del, 1.008, "Δ AUROC", ha="right", va="top", fontsize=SG.ANNOT, color=SG.MUTED)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["1001 bp\nsingle-pos", "8192 bp\nmean-LL"], fontsize=7.0, linespacing=1.45)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("AUROC", fontsize=8.0, color=S.CAP); ax.tick_params(axis="y", labelsize=7.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- G : SEPARATION x DISCRIMINATION
# Label placement for panel G works in POINTS on the page, not in data units.
#
# The previous seat search offered eight seats around each point at offsets expressed as
# fractions of the axis span -- 0.11 * xr * 0.14 = 0.075 in Cohen's d, about 3 pt on the page --
# and accepted the first that missed the label boxes already committed. Two things followed. The
# offset was smaller than the marker itself (the discs run to 3.7 pt of radius, and the biggest
# species carry the biggest discs), so a seat that cleared every other LABEL still sat on its own
# DOT: "horse", "cat", "pig", "dog" and "human" were each printed with their own marker struck
# through the word. And a bbox test over the PDF's text spans cannot see it, because the marker is
# a path, not a span -- the figure passed a collision gate while five of its nine labels were
# illegible. The markers are obstacles here, in the same coordinate system as the text.
_T2P = None


def _text_wh(s, fs):
    """(width, height) of a single-line string in POINTS, in the figure's own face."""
    global _T2P
    if _T2P is None:
        from matplotlib.textpath import TextToPath
        _T2P = TextToPath()                      # measures at 72 dpi, so it returns points
    from matplotlib.font_manager import FontProperties
    w, h, d = _T2P.get_text_width_height_descent(s, FontProperties(family=SG.FAMILY, size=fs), False)
    return w, max(h + d, fs * 0.72)


def _hits_rect(a, b, pad=0.0):
    return (a[0] - pad < b[2] and b[0] < a[2] + pad and a[1] - pad < b[3] and b[1] < a[3] + pad)


def _hits_disc(box, c, pad=0.0):
    """True if the disc (cx, cy, r) reaches into `box`. Closest-point test, exact for a circle."""
    dx = max(box[0] - c[0], 0.0, c[0] - box[2])
    dy = max(box[1] - c[1], 0.0, c[1] - box[3])
    return dx * dx + dy * dy < (c[2] + pad) ** 2


def _seat_label(text, c, discs, hard, soft, frame, fs):
    """Find a seat for `text` beside marker `c` = (cx, cy, r), all in points.

    Twelve directions at four radii, nearest first, and diagonals before the horizontal: a label
    due east of a point lands squarely on that point's own horizontal confidence bar, which is how
    the first repair of this panel produced nine labels with a line ruled through them.

    `discs` and `hard` must be missed. `soft` -- here, the pooled band, whose ink is a reference
    rather than this species' own measurement -- is missed if any seat can, and given up if none
    can, rather than pushing a label somewhere worse. Returns (x, y, ha, va, box, lead, u): `lead`
    is non-zero when the seat is far enough out that the caller must draw a leader line, because
    past that distance proximity no longer binds the label to its point.
    """
    w, h = _text_wh(text, fs)
    w += 1.2                                              # side bearing, so boxes never kiss
    h += 1.2
    for rects in ((hard + soft), hard):
        for rad in (c[2] + 2.6, c[2] + 6.0, c[2] + 10.0, c[2] + 16.0):
            for deg in (26, -26, 154, -154, 52, -52, 128, -128, 0, 180, 90, -90):
                t = np.deg2rad(deg); ux, uy = np.cos(t), np.sin(t)
                ax_, ay_ = c[0] + rad * ux, c[1] + rad * uy
                ha = "left" if ux > 0.30 else ("right" if ux < -0.30 else "center")
                va = "bottom" if uy > 0.30 else ("top" if uy < -0.30 else "center")
                x0 = ax_ if ha == "left" else (ax_ - w if ha == "right" else ax_ - w / 2)
                y0 = ay_ if va == "bottom" else (ay_ - h if va == "top" else ay_ - h / 2)
                box = (x0, y0, x0 + w, y0 + h)
                if not (frame[0] < box[0] and box[2] < frame[2]
                        and frame[1] < box[1] and box[3] < frame[3]):
                    continue
                if any(_hits_disc(box, d, 1.0) for d in discs):
                    continue
                if any(_hits_rect(box, b, 0.8) for b in rects):
                    continue
                return ax_, ay_, ha, va, box, (rad if rad > c[2] + 4 else 0.0), (ux, uy)
    return None


def panel_G(ax):
    """Figure 2 (the atlas), panel d: AUROC against Cohen's d, one point per species."""
    summ = json.load(open("reports/fig2b_summary.json"))
    D = json.load(open("reports/fig2_data.json")); pooled = D["pooled"]
    # x must hold every d CI: chicken reaches +5.06 on the full panel and was clipped at 4.7.
    ax.set_xlim(0.55, 5.45); ax.set_ylim(0.83, 1.008)
    ax.axhspan(pooled["lo"], pooled["hi"], color=S.EVO2, alpha=0.06, zorder=0)
    ax.axhline(pooled["auroc"], color=S.EVO2, lw=0.8, ls=(0, (5, 3)), alpha=0.5, zorder=1)
    # Label the line from the same value the line is drawn at. It used to read a hardcoded 0.956
    # while the line sat at pooled["auroc"], which the recompute layer had moved to 0.973 -- the
    # figure disagreed with itself, and with the manuscript.
    ax.text(4.68, pooled["auroc"], f"pooled\n{pooled['auroc']:.3f}", fontsize=SG.ANNOT, color=S.EVO2,
            va="center", ha="right", linespacing=1.45)
    # No "small-n mirage" band any more. It shaded d > 2.2 on the 3,506-variant capped subset,
    # where only the underpowered species reached that far. On the full panel every species except
    # human sits above 2.2 -- including cattle (188 positives) and dog (227) -- so the band would
    # brand well-powered estimates a mirage. The small-n species are already marked by their
    # dashed, thinner CI bars, and they interleave with the well-powered ones (small n: 2.35-3.65;
    # well-powered: 1.09-3.41), so there is no region to shade.

    # ---- page geometry: this axes' box in points, and the maps between data and points --------
    fig = ax.figure
    pos = ax.get_position()
    Wp, Hp = pos.width * fig.get_figwidth() * 72.0, pos.height * fig.get_figheight() * 72.0
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    to_x = lambda v: (v - xa) / (xb - xa) * Wp
    to_y = lambda v: (v - ya) / (yb - ya) * Hp
    un_x = lambda p: xa + p / Wp * (xb - xa)
    un_y = lambda p: ya + p / Hp * (yb - ya)
    frame = (1.5, 1.5, Wp - 1.5, Hp - 1.5)

    # ---- the intervals -----------------------------------------------------------------------
    # (drawn after the point<->page maps above are in scope: each bar is also an obstacle)
    # Nine horizontal d intervals crossing nine vertical AUROC intervals wove a plaid in which no
    # bar could be followed to its own point. Three changes, none of which shortens an interval:
    # the bars are drawn in a lighter tint of the clade colour so the markers sit clearly on top of
    # them; each carries a white halo, so a crossing reads as one bar passing over another rather
    # than as a lattice; and each interval ends in a cap, so its extent is bounded and countable.
    # Small-n species keep their dashed, thinner bars and are drawn first, i.e. underneath.
    import matplotlib.patheffects as pe
    cap_x, cap_y = (xb - xa) * 0.0075, (yb - ya) * 0.014            # cap half-lengths, ~2 pt
    bars = []                                                       # the intervals, as obstacles
    for k, sp in enumerate(sorted(S.LEAF_ORDER, key=lambda s: summ[s]["n_pos"])):
        s = summ[sp]; col = S.species_color(sp); small = s["n_pos"] < 60
        tint = S._shade(col, 0.34 if small else 0.18)
        halo = [pe.withStroke(linewidth=(1.9 if small else 2.4), foreground="white")]
        z = 2 + k * 0.01
        lw = 0.7 if small else 1.0
        ax.plot([s["d_lo"], s["d_hi"]], [s["auroc"]] * 2, color=tint, lw=lw,
                ls=((0, (2.2, 1.6)) if small else "-"), zorder=z, path_effects=halo,
                solid_capstyle="butt")
        ax.plot([s["cohens_d"]] * 2, [s["au_lo"], s["au_hi"]], color=tint, lw=lw,
                ls=((0, (2.2, 1.6)) if small else "-"), zorder=z, path_effects=halo,
                solid_capstyle="butt")
        for e in (s["d_lo"], s["d_hi"]):                                     # caps on the d interval
            ax.plot([e, e], [s["auroc"] - cap_y, s["auroc"] + cap_y], color=tint, lw=lw,
                    zorder=z, solid_capstyle="butt")
        for e in (s["au_lo"], s["au_hi"]):                                   # caps on the AUROC interval
            ax.plot([s["cohens_d"] - cap_x, s["cohens_d"] + cap_x], [e, e], color=tint, lw=lw,
                    zorder=z, solid_capstyle="butt")
        t_ = 1.7                                                             # bar + halo half-width, pt
        bars.append((to_x(s["d_lo"]) - t_, to_y(s["auroc"]) - t_,
                     to_x(s["d_hi"]) + t_, to_y(s["auroc"]) + t_))
        bars.append((to_x(s["cohens_d"]) - t_, to_y(s["au_lo"]) - t_,
                     to_x(s["cohens_d"]) + t_, to_y(s["au_hi"]) + t_))

    # ---- the markers -------------------------------------------------------------------------
    discs = {}
    for sp in S.LEAF_ORDER:
        s = summ[sp]
        area = 16 + 26 * (np.log(s["n"]) - 4.4) / 4.0                        # pt^2, area not radius
        ax.scatter([s["cohens_d"]], [s["auroc"]], s=area, color=S.species_color(sp), zorder=5,
                   edgecolors="white", linewidths=0.7)
        discs[sp] = (to_x(s["cohens_d"]), to_y(s["auroc"]), np.sqrt(area / np.pi) + 0.35)

    # ---- the two fixed annotations, seated first so the species labels route around them ------
    # Both numbers read from the same summary the marker is plotted from. Typed, they said
    # ".970 / d 0.9" -- the values from the superseded 3,506-variant capped subset -- while the
    # marker beside them sat at the full panel's 0.974 / 1.09. This annotation names the human
    # point, so human does not also get a bare "human" seat: the figure used to print the species
    # twice, once on the leader and once beside the dot.
    _h = summ["human"]
    _htxt = f"human  AUROC {_h['auroc']:.3f}, d {_h['cohens_d']:.1f}"
    _anno(ax, _htxt, (_h["cohens_d"] - 0.10, _h["auroc"] + 0.003), (0.80, 1.001),
          col=S.species_color("human"), fs=SG.ANNOT)
    hw, hh = _text_wh(_htxt, SG.ANNOT)
    boxes = [(to_x(0.80), to_y(1.001) - hh / 2, to_x(0.80) + hw, to_y(1.001) + hh / 2)]
    pw = max(_text_wh("pooled", SG.ANNOT)[0], _text_wh(f"{pooled['auroc']:.3f}", SG.ANNOT)[0])
    ph = 2 * SG.ANNOT * 1.45
    boxes.append((to_x(4.68) - pw, to_y(pooled["auroc"]) - ph / 2, to_x(4.68), to_y(pooled["auroc"]) + ph / 2))

    # ---- the species labels ------------------------------------------------------------------
    # The pooled band is a soft obstacle: it is a reference, not any one species' measurement, so a
    # label may cross it if nothing else will do -- but not while a seat clear of it exists.
    soft = [(frame[0], to_y(pooled["lo"]) - 0.8, frame[2], to_y(pooled["hi"]) + 0.8)]
    obst = [discs[s] for s in S.LEAF_ORDER]
    placed = {}
    for sp in sorted([s for s in S.LEAF_ORDER if s != "human"], key=lambda s: -summ[s]["n"]):
        c = discs[sp]
        seat = _seat_label(sp, c, obst, boxes + bars, soft, frame, SG.ANNOT)
        if seat is None:                                                     # never on this data
            raise RuntimeError(f"panel G: no seat for {sp!r}")
        axp, ayp, ha, va, box, lead, u = seat
        boxes.append(box); placed[sp] = box
        if lead:                                                             # long offset: draw the tie
            x0, y0 = c[0] + (c[2] + 1.0) * u[0], c[1] + (c[2] + 1.0) * u[1]
            x1, y1 = axp - 1.4 * u[0], ayp - 1.4 * u[1]
            ax.plot([un_x(x0), un_x(x1)], [un_y(y0), un_y(y1)], color=S._shade(S.species_color(sp), 0.30),
                    lw=0.5, zorder=4, solid_capstyle="butt")
        ax.text(un_x(axp), un_y(ayp), sp, fontsize=SG.ANNOT, color=S.species_color(sp),
                ha=ha, va=va, zorder=6)

    # ---- the check the old bbox gate could not make ------------------------------------------
    # Assert, in this panel's own coordinates, that no label box reaches any marker. This is the
    # defect the PDF text-span test is blind to, so it is asserted where the geometry is known.
    worst = min((np.hypot(max(b[0] - d[0], 0, d[0] - b[2]), max(b[1] - d[1], 0, d[1] - b[3])) - d[2]
                 for sp, b in placed.items() for d in obst), default=99.0)
    for sp, b in placed.items():
        for d in obst:
            assert not _hits_disc(b, d), f"panel G: label {sp!r} sits on a marker"
    print(f"    panel c: 8 species labels seated, min label-to-marker clearance {worst:.2f} pt")

    ax.set_xlabel("Cohen's d  (positive − negative)", fontsize=SG.AXIS, color=S.CAP)
    ax.set_ylabel("AUROC", fontsize=SG.AXIS, color=S.CAP); ax.tick_params(labelsize=SG.TICK)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- F : CONSERVATION FRONTIER
def panel_F(ax):
    D = json.load(open("reports/fig2_data.json")); g = D["gerp"]
    order = sorted([s for s in S.LEAF_ORDER if s in g], key=lambda s: -g[s]["delta"])
    # Each row is a dumbbell -- the two values and the distance between them -- with the deltas
    # aligned in their own column where they can be read down.
    DX = 1.045                                                       # delta column, fixed x
    ax.set_xlim(0.46, 1.16); ax.set_ylim(-0.7, len(order) + 0.95)
    ax.axvline(0.5, color="#DDD", lw=0.6)
    for i, sp in enumerate(order):
        y = len(order) - 1 - i; gg = g[sp]; ev, ge = gg["evo2"], gg["gerp"]; d = gg["delta"]
        sig = gg["verdict"].strip() == "***"
        lo, hi = sorted([ev, ge]); win = S.PATHO if d > 0 else S.BENIGN
        ax.plot([lo, hi], [y, y], color=win, lw=(3.4 if sig else 1.4),
                alpha=(0.85 if sig else 0.45), ls=("-" if sig else (0, (1.5, 1.5))),
                solid_capstyle="round", zorder=2)
        ax.scatter([ge], [y], s=26, color=S.BENIGN, zorder=4, edgecolors="white", linewidths=0.7)
        ax.scatter([ev], [y], s=26, color=S.PATHO, zorder=4, edgecolors="white", linewidths=0.7)
        # values outboard of their own dot, so they never sit over the connector between them
        ax.text(min(ev, ge) - 0.012, y, f"{ge if ge < ev else ev:.2f}", ha="right", va="center",
                fontsize=SG.ANNOT, color=(S.BENIGN if ge < ev else S.PATHO), zorder=5)
        ax.text(max(ev, ge) + 0.012, y, f"{ev if ev > ge else ge:.2f}", ha="left", va="center",
                fontsize=SG.ANNOT, color=(S.PATHO if ev > ge else S.BENIGN), zorder=5)
        ax.text(0.485, y + 0.20, f"{sp}", ha="right", va="center", fontsize=6.6,
                color=(win if sig else S.INK))
        ax.text(0.485, y - 0.30, f"n={gg['n']}", ha="right", va="center", fontsize=SG.ANNOT, color=S.MUTED)
        ax.text(DX, y, f"{'+' if d > 0 else ''}{d:.3f}{'***' if sig else ' ns'}",
                ha="left", va="center", fontsize=SG.ANNOT, color=(win if sig else S.MUTED))
    ax.text(DX, len(order) - 0.06, "Evo 2 − GERP", ha="left", va="center", fontsize=SG.ANNOT, color=S.CAP)
    # A key, not a direction. Arrow headers pointing left for GERP and right for Evo 2 would
    # claim GERP is always the left-hand dot; on horse and human the order reverses. The two dots
    # are identified by their own marks instead, drawn once above the column they label.
    ky = len(order) + 0.52
    for xk, col, lab in ((0.525, S.BENIGN, "GERP (conservation)"), (0.80, S.PATHO, "Evo 2")):
        ax.scatter([xk], [ky], s=26, color=col, zorder=5, edgecolors="white", linewidths=0.7)
        ax.text(xk + 0.018, ky, lab, ha="left", va="center", fontsize=SG.ANNOT, color=col)
    ax.set_xlabel("AUROC  (co-scorable set: both methods defined; ≠ atlas)", fontsize=7.2, color=S.CAP)
    # The x-limit was widened to seat the delta column, which pushed the scale past 1.0 -- an AUROC
    # cannot exceed 1, so a tick at 1.1 invites a reader to place a value that cannot exist. Ticks
    # and the spine stop where the measurement does; the delta column lives outside the scale.
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["bottom"].set_bounds(0.5, 1.0)
    ax.set_yticks([]); ax.tick_params(labelsize=7.0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- E : DISTANCE NULL
def panel_E(ax):
    """Per-species AUROC against divergence from human.

    One uniform mark per species, named in text, with a leader line only where a name cannot sit
    beside its mark: a silhouette of unequal visual weight would put the datum somewhere inside a
    picture rather than at a point.
    """
    D = json.load(open("reports/fig2_data.json")); fr = D["forest"]; dd = D["distance"]
    ax.set_xlim(-18, 360); ax.set_ylim(0.83, 1.0)
    m = float(np.mean([r["auroc"] for r in fr]))
    # The line is the mean of the nine plotted points and is labelled as such. No envelope is
    # shaded around it, because a band here would not be an interval of anything.
    ax.axhline(m, color=SG.RULE, lw=0.8, ls=(0, (5, 3)), zorder=1)
    ax.text(-14, m + 0.003, f"species mean {m:.3f}", fontsize=SG.ANNOT, color=SG.MUTED,
            ha="left", va="bottom", zorder=6)
    # EVERY MARK AT ITS TRUE DIVERGENCE. The seven non-primate mammals all diverge
    # from human at 94 My, and the x axis is a measured quantity in millions of years: fanning them
    # across it so that their names resolve would show a reader separations that do not exist, in
    # the one panel whose entire claim is that divergence explains nothing. The seven stack in a
    # single column at x = 94 -- true x -- and are separated only VERTICALLY, by their own AUROC,
    # with the names seated left and right of the column. Their AUROCs are NOT jittered: y is the
    # measurement this panel exists to report, and the seven values (0.863-0.975) already
    # separate the marks.
    _n94 = sum(1 for r in fr if r["divergence_my"] == 94)
    ax.text(94 + 7, 0.9925, f"{_n94} mammals at 94 My", fontsize=SG.ANNOT, color=SG.MUTED,
            ha="left", va="bottom")
    pts = [(float(r["divergence_my"]), r["auroc"], r["species"]) for r in fr]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=20, color=S.EVO2, zorder=5,
               edgecolors="white", linewidths=0.7)
    # Name each mark by trying seats around it and taking the first that collides with nothing
    # already committed; the 94-My note and the chicken callout are seeded as occupied.
    XR, YR = 378.0, 0.17                                   # axis spans, for pt -> data conversion
    # THE MARKS ARE OBSTACLES TOO. Stacked in one true-x column, the seat above one mammal is the
    # seat its neighbour's marker occupies -- the seat above cattle is dog's marker -- and a name
    # printed across its neighbour's dot is exactly the defect the collision audit cannot see,
    # because it compares text with text. Every marker is a committed box, and the lateral seats
    # are tried FIRST so a column of names resolves sideways.
    MW, MH = 0.017 * XR, 0.019 * YR                        # marker half-box (s=20 + white edge)
    placed = [(94.0, 0.9885, 235.0, 1.0),                  # the "7 mammals at 94 My" note
              (228.0, 0.9745, 350.0, 0.9865),              # the chicken callout seat, below
              (-16.0, m, 95.0, m + 0.011)]                 # the species-mean label
    placed += [(px - MW, py - MH, px + MW, py + MH) for px, py, _ in pts]
    for x, au, sp in sorted(pts, key=lambda p: -p[1]):
        if sp == "chicken":
            continue                                       # carries its own datum label
        # h is sized to one line of label type, with a small pad: a taller half-box makes the
        # lateral seats of one column collide with each other.
        w = 0.040 * XR * (len(sp) ** 0.60); h = 0.021 * YR
        for ddx, ddy, ha, va in ((0.026, 0.000, "left", "center"),
                                 (-0.026, 0.000, "right", "center"),
                                 (0.024, 0.021, "left", "bottom"),
                                 (-0.024, 0.021, "right", "bottom"),
                                 (0.024, -0.021, "left", "top"),
                                 (-0.024, -0.021, "right", "top"),
                                 (0.000, 0.034, "center", "bottom"),
                                 (0.000, -0.034, "center", "top")):
            cx, cy = x + ddx * XR, au + ddy * YR
            bx = (cx - (w / 2 if ha == "center" else (w if ha == "right" else 0)),
                  cy - (h if va != "bottom" else 0),
                  cx + (w / 2 if ha == "center" else (w if ha == "left" else 0)),
                  cy + (h if va != "top" else 0))
            if bx[0] > -16 and bx[2] < 358 and bx[1] > 0.834 and bx[3] < 0.998 and _seat(placed, bx):
                break
        else:                                              # no seat cleared everything
            raise RuntimeError(f"panel E: no seat for {sp!r}")
        placed.append(bx)
        ax.text(cx, cy, sp, fontsize=SG.ANNOT, color=SG.INK, ha=ha, va=va, zorder=6)
    # The out-group anchor keeps its coordinates: neither 319 My nor 0.954 is printed anywhere else.
    ax.annotate("chicken  319 My, 0.954", xy=(319, 0.954), xytext=(232, 0.980),
                fontsize=SG.ANNOT, color=SG.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=SG.MUTED, lw=0.6,
                                shrinkA=1.5, shrinkB=3.0))
    ax.set_xlabel("divergence from human  (My, TimeTree)", fontsize=7.2, color=S.CAP)
    ax.set_ylabel("AUROC", fontsize=8.0, color=S.CAP); ax.tick_params(labelsize=7.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(352, 0.834, f"ρ = {dd['rho']:+.2f}, p = {dd['p']:.2f} (n=9)",
            ha="right", va="bottom", fontsize=6.6, color=SG.MUTED)


def main():
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    panel_D(ax[0, 0]); panel_H(ax[0, 1]); panel_G(ax[0, 2]); panel_F(ax[1, 0]); panel_E(ax[1, 1])
    ax[1, 2].axis("off")
    fig.savefig("reports/figures/_fig2_DHGFE.png", bbox_inches="tight", dpi=190)
    print("wrote reports/figures/_fig2_DHGFE.png")


if __name__ == "__main__":
    main()
