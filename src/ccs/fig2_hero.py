"""Panel a of the atlas plate (Figure 7a): build_dotplot, a per-species dot plot on the species tree.

build() below is the radial "corona" this panel used to be -- per-species wedges on a radial tree.
It is no longer drawn in any submitted figure: its wedge encoding was decorative, and its AUROC-only
view let the plate read as Evo 2 skill when a one-bit coding flag scores the same panels as well. It
is kept, unchanged, because the exploratory fig2_atlas_v2.py still calls it; its numbers come from
reports/fig2_data.json. build_dotplot() computes its own from Additional file 3's per-variant table
(fig2_measure.atlas_skill) and Table 2's interval table (fig2_measure.table2_intervals).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
import fig2_style as S
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG
try:
    from . import fig2_measure as MEAS
except ImportError:
    import fig2_measure as MEAS

R_LEAF = 3.0          # tree leaves / AUROC 0.5 base
R_TOP = 6.2           # AUROC 1.0
R_SIL = 7.52          # silhouette ring (was 7.15: several silhouettes point inward -- sheep,
                      # goat and cattle lead with head or horns -- and the autocropped image
                      # reached the AUROC value label sitting at the arc tip)


def A(r): return R_LEAF + (r - 0.5) / 0.5 * (R_TOP - R_LEAF)   # AUROC -> radius
def xy(r, t): return r * np.cos(t), r * np.sin(t)


def sector(ax, r0, r1, tc, wrad, color, alpha=1.0, z=3, ec="none", lw=0):
    th = np.linspace(tc - wrad / 2, tc + wrad / 2, 40)
    outer = [(r1 * np.cos(a), r1 * np.sin(a)) for a in th]
    inner = [(r0 * np.cos(a), r0 * np.sin(a)) for a in th[::-1]]
    ax.add_patch(Polygon(outer + inner, closed=True, facecolor=color, alpha=alpha,
                         edgecolor=ec, lw=lw, zorder=z, joinstyle="round"))


NO_CLUSTER_INTERVAL = {"goat"}   # see the comment where it is used, in the wedge loop


def ring(ax, r, color, lw=0.8, ls="-", alpha=1.0, z=1):
    ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=color, lw=lw, ls=ls, alpha=alpha, zorder=z))


def node_angle(nd, ang):
    if not nd["kids"]:
        return ang[nd["name"]]
    return float(np.mean([node_angle(c, ang) for c in nd["kids"]]))


def draw_tree(ax, nd, ang, amax=319.0):
    r = R_LEAF * (1 - nd["age"] / amax)
    ta = node_angle(nd, ang)
    if nd["kids"]:
        cas = [node_angle(c, ang) for c in nd["kids"]]
        aa = np.linspace(min(cas), max(cas), 30)                      # connecting arc at this node's radius
        ax.plot(r * np.cos(aa), r * np.sin(aa), color="#5A5A5A", lw=1.1, solid_capstyle="round", zorder=2)
        for c, ca in zip(nd["kids"], cas):
            cr = R_LEAF * (1 - c["age"] / amax)
            ax.plot([r * np.cos(ca), cr * np.cos(ca)], [r * np.sin(ca), cr * np.sin(ca)],
                    color="#5A5A5A", lw=1.1, solid_capstyle="round", zorder=2)
            draw_tree(ax, c, ang, amax)


def build(ax):
    D = json.load(open("reports/fig2_data.json"))
    frow = {r["species"]: r for r in D["forest"]}
    summ = json.load(open("reports/fig2b_summary.json"))          # per-species n_pos for the small-n flag
    # Missense floor read from the recompute layer. Typed, it was 0.824 -- the value from the
    # 3,506-variant capped subset that type_matched_atlas.parquet held while nothing re-ran it.
    pooled = D["pooled"]
    miss = next(r["auroc"] for r in D["ascert"] if r["stratum"].startswith("MISSENSE"))
    ang = S.radial_positions()
    lay = S.layout_tree()
    ns = [frow[s]["n"] for s in S.LEAF_ORDER]
    lnmin, lnmax = np.log(min(ns)), np.log(max(ns))

    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-8.2, 8.2); ax.set_ylim(-9.9, 8.2)  # headroom for bottom legend

    # time-ring graticule (faint concentric circles at a few divergence ages)
    for age in [319, 200, 94, 0]:
        ring(ax, R_LEAF * (1 - age / 319), "#ECECEC", lw=0.7, z=0)
    # AUROC gridlines + reference rings
    for a in [0.6, 0.7, 0.8, 0.9, 1.0]:
        ring(ax, A(a), "#EDEDED", lw=0.6, ls=":", z=0)
    # The two reference rings are DOTTED and LONG-DASHED, not two four-and-five dashes that look
    # identical at 170 mm. They used to differ only in hue, and the key below wrote both of them as
    # the same neutral-grey em-dash pair, so nothing on the plate said which ring was which: a
    # reader had to guess, and in greyscale there was nothing to guess from. The key now names the
    # pattern in words, so the distinction survives with no colour at all.
    ring(ax, A(miss), S.PATHO, lw=1.0, ls=(0, (1.2, 2.0)), alpha=0.75, z=1)        # missense floor
    ring(ax, A(pooled["auroc"]), S.EVO2, lw=1.0, ls=(0, (7, 2.5)), alpha=0.65, z=1)  # pooled
    sector(ax, A(pooled["lo"]), A(pooled["hi"]), 0, 2 * np.pi, S.EVO2, alpha=0.05, z=0)  # pooled band (full ring)

    draw_tree(ax, lay, ang)

    import matplotlib.patheffects as pe
    for s in S.LEAF_ORDER:
        r = frow[s]; t = ang[s]; col = S.species_color(s)
        w = np.deg2rad(13 + 15 * (np.log(r["n"]) - lnmin) / (lnmax - lnmin))     # width ~ log n
        a1 = max(0.5, r["auroc_1001"])
        npos = summ[s]["n_pos"]; small = npos < 30                               # goat n_pos=9, chicken n_pos=28
        # Goat is drawn WITHOUT an interval. Its nine positives occupy two loci, eight of them
        # inside a 570-bp span, giving Kish's effective count of 1.2 independent positives; the
        # Methods therefore call goat's AUROC "a descriptive point estimate" and Table 2 prints it
        # as "0.959 (no interval)". Drawing it the same 95 % CI whisker as human's 3,000 positives
        # asserted on the plate the certainty the Methods elsewhere refuse. The asterisk that was
        # here already flags "fewer than 30 positives", which is a different and weaker statement,
        # and it covers chicken too -- chicken's 28 positives do support an interval. Hard-coded
        # rather than derived because no deposited artefact carries the per-species cluster count;
        # this list and the one in fig2_measure.panel_d are the two places to change together.
        no_ci = s in NO_CLUSTER_INTERVAL
        if not no_ci:
            sector(ax, A(r["lo"]), A(r["hi"]), t, w * 1.16, col, alpha=0.20, z=3)           # CI halo (faint)
        # The wedge keeps its species hue but at 35% less chroma, which is what
        # brings this plate into the set's range. ink_on() below picks the label colour from the
        # muted fill, so the AUROC label follows automatically.
        sector(ax, R_LEAF, A(a1), t, w, S.desat(col, 0.35), alpha=0.96, z=4)               # 1001bp base
        sector(ax, A(a1), A(r["auroc"]), t, w, S._shade(S.desat(col, 0.35), 0.45), alpha=0.96, z=4)  # readout-gain tip (lighter)
        aa = np.linspace(t - w / 2, t + w / 2, 20)
        ax.plot(A(a1) * np.cos(aa), A(a1) * np.sin(aa), color="white", lw=0.6, zorder=5)    # 1001bp demarcation
        # EXPLICIT 95% CI whisker at the arc tip (radial) — makes goat n=9 visibly less certain than human n=300
        rl, rh = A(r["lo"]), A(r["hi"]); wc = S.PATHO if small else S._shade(col, -0.4)
        if no_ci:
            # point estimate only: an open ring at the value, no bar and no caps
            rp = A(r["auroc"])
            ax.plot([rp * np.cos(t)], [rp * np.sin(t)], marker="o", ms=3.4, mfc="white",
                    mec=S.PATHO, mew=1.0, zorder=6)
        else:
            ax.plot([rl * np.cos(t), rh * np.cos(t)], [rl * np.sin(t), rh * np.sin(t)], color=wc, lw=1.2, zorder=6,
                    solid_capstyle="round", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            for rr in (rl, rh):                                                            # tangential end caps
                ca = np.array([t - 0.9 / rr, t + 0.9 / rr])
                ax.plot(rr * np.cos(ca), rr * np.sin(ca), color=wc, lw=1.1, zorder=6)
        # silhouette at rim, tinted clade colour
        im = OffsetImage(S.load_silhouette(s, S.SIL_INK), zoom=0.115)
        ab = AnnotationBbox(im, xy(R_SIL, t), frameon=False, box_alignment=(0.5, 0.5), zorder=6)
        ax.add_artist(ab)
        # AUROC value label inside its own wedge (radial, upright).
        # House style forbids a path-effect stroke behind text, so the darker outline that used to
        # make white type readable is gone -- and with it the two grounds it was compensating for:
        # the LIGHT readout tip, and the CI whisker, which is drawn along this same radius and used
        # to run straight through the digits. The label is seated at the mid-point of the saturated
        # 1,001-bp band instead, which is solid species colour on every wedge and lies well inside
        # the whisker's inner end. Bold is reserved for the panel letter.
        lx, ly = xy((R_LEAF + A(a1)) / 2, t)
        rot = np.rad2deg(t) % 360; rot = rot - 180 if 90 < rot < 270 else rot   # keep upright (fix negative-angle flip)
        _flag = "\u2020" if no_ci else ("*" if small else "")
        # Not "white". The wedge is the species colour, and several of those are light: measured
        # on the shipped plate this numeral ran to 1.76:1 on cat's amber, 2.92 on human's pink and
        # 3.00 on goat's blue. SG.ink_on picks whichever ink the fill can carry -- and it is handed
        # the fill the reader SEES, desat(col, 0.35) composited at the sector's own alpha over
        # white, not the nominal species colour. Passing the nominal one left horse's numeral at
        # 4.20:1 against a 4.5 bar while this comment claimed it had been re-picked from the fill.
        ax.text(lx, ly, f"{r['auroc']:.2f}{_flag}", ha="center", va="center", fontsize=SG.TICK,
                color=SG.ink_on(SG.over(S.desat(col, 0.35), 0.96)), rotation=rot,
                rotation_mode="anchor", zorder=7)

    # centre + reference labels (pooled value lives in the chip strip / panel C / the labelled blue ring —
    # dropped here so it no longer piles on the 6 o'clock silhouette; legend moved below the silhouette feet)
    # A white knockout with NO frame and no rounding: the branches converge on this point, so the
    # label needs the ink behind it cleared, but the rounded grey-edged card it used to sit in was
    # decoration (house style: no rounded card boxes, no annotation frames that are not data).
    ax.text(0, 0, "Amniota\n~319 My", ha="center", va="center", fontsize=SG.ANNOT, color=S.MUTED,
            linespacing=1.45, zorder=8,
            bbox=dict(boxstyle="square,pad=0.28", fc="white", ec="none"))
    # ONE text object with point-based line spacing, not three placed 0.32-0.35 DATA units apart.
    # This axes is aspect-equal, so a fixed data offset shrinks with the panel: at the split-figure
    # height the three lines would print on top of each other. Line spacing in points is size-invariant.
    # Prose about the construction of the corona belongs in the manuscript legend; what stays here
    # is the ring/whisker key, which names the two plotted reference values.
    # The key's first line states two facts separately (where the wedge starts, and what the
    # saturated band measures); the block's compact width is what buys the corona its extra
    # radius. Anchored in axes fractions rather than in data units: this axes is aspect-equal,
    # so a data offset moves under the panel when the row height changes.
    ax.text(0.5, 0.072,
            "wedge: base 0.5 = chance · solid = 1,001 bp · light tip = 8,192-bp gain\n"
            # This block is CENTRED on panel a, so its widest line sets how far right the key reaches,
            # and panel b's opaque background starts at ~300 pt and paints out anything beyond it, so
            # every line here has to end short of that edge.
            f"long dash = pooled {pooled['auroc']:.3f}   dotted = missense floor {miss:.3f}   "
            "whisker = 95% CI\n"
            # The asterisk names no species: chicken and goat (n_pos = 9) both meet
            # "n_pos < 30"; the builder simply suppresses the asterisk where the dagger applies.
            # Naming chicken alone would imply goat is NOT small-n, when it is the smallest positive
            # class in the atlas. Leaving the species out avoids that false exclusivity
            # without widening the line -- adding ", goat" would push the centred
            # block's right edge from 293.7 pt to about 303 pt, into panel b's column.
            "* n_pos < 30   \u2020 point estimate only, no cluster-robust interval (goat)",
            transform=ax.transAxes, ha="center", va="top", fontsize=SG.ANNOT, color=S.MUTED,
            linespacing=1.5)


# ================================================================== Figure 7a: the dot plot
# Four estimates per species, each on its own sub-row so that no marker or whisker hides another:
# Evo 2 at 1,001 bp (open) and at 8,192 bp (filled), then the two label-free baselines of Table 2,
# the one-bit coding flag and snpEff's four-level impact ordering. Offsets are fractions of a row.
DOT_OFF = {"e1": -0.33, "e8": -0.11, "coding": +0.11, "impact": +0.33}
# Colour is the PREDICTOR, and each predictor also has its own shape, so no pair relies on hue:
# Evo 2 blue circles (open/filled = readout), the coding flag black squares, snpEff impact orange
# triangles. Greyscale contrast of the three fills: blue 0.152 vs black 0.010 is 3.4:1, blue vs
# orange 0.416 is 2.3:1, black vs orange 7.8:1 (WCAG relative luminance).
IMPACT_C, IMPACT_EDGE = "#E69F00", "#9F6B02"
DOT_STYLE = {
    "e1": dict(marker="o", ms=4.6, mfc="white", mec=S.EVO2, mew=.8),
    "e8": dict(marker="o", ms=4.6, mfc=S.EVO2, mec=S.EVO2, mew=.6),
    "coding": dict(marker="s", ms=3.9, mfc=SG.INK, mec=SG.INK, mew=.5),
    "impact": dict(marker="^", ms=4.9, mfc=IMPACT_C, mec=IMPACT_EDGE, mew=.5),
}
# The two readouts are named by what they compute as well as by their window: the difference between
# them is mostly the averaging over the window, not its length (Table S9).
DOT_LABEL = {"e1": "Evo 2, 1,001-bp single token", "e8": "Evo 2, 8,192-bp window mean", "coding": "coding flag",
             "impact": "snpEff impact"}
SIL_PT = 17.0          # silhouette's longest side, in points, sized for a 25-pt species row


def _pt_x(fig, pt):
    """`pt` points as a figure-fraction width."""
    return pt / (fig.get_size_inches()[0] * 72.0)


def _rect_tree(ax, nd, color="#5A5A5A", lw=0.9):
    """A rectangular, time-scaled cladogram: x = age (My), y = the leaf rows of layout_tree."""
    if not nd["kids"]:
        return
    ys = [c["y"] for c in nd["kids"]]
    ax.plot([nd["age"], nd["age"]], [min(ys), max(ys)], color=color, lw=lw,
            solid_capstyle="projecting", zorder=2)
    for c in nd["kids"]:
        ax.plot([nd["age"], c["age"]], [c["y"], c["y"]], color=color, lw=lw,
                solid_capstyle="projecting", zorder=2)
        _rect_tree(ax, c, color, lw)


def build_dotplot(fig, ax):
    """Figure 7a on `ax` (already positioned): per species, Evo 2 at both readouts with Table 2's
    95% intervals, and the coding-flag and snpEff-impact AUROCs; the unweighted nine-species mean
    as a last row. The species rows follow the species tree, which is drawn to their left with a
    PhyloPic silhouette at each tip. Returns the tree axes.

    WHY A DOT PLOT. The corona drew one number per species as a coloured wedge whose area, arc and
    tip carried no quantity, and it drew Evo 2 alone -- so a plate whose point is that Evo 2
    separates the atlas classes could not show that annotation alone separates them as well. Here
    every estimate is a position on one common AUROC scale.

    Whiskers are Table 2's intervals exactly as printed (table2_intervals: the wider of the
    variant-level and locus-clustered endpoints). Goat has none: Table 2 prints it without one
    (fig2_measure.NO_CLUSTER_INTERVAL). The mean row's Evo 2 whiskers are Table 2's macro-row t
    intervals on 8 df; the two baselines carry no interval in Table 2 and carry none here.

    No text on the canvas beyond the axis, the row names, one key and the tree's scale bar: the
    decoding of the whiskers, the dagger and the mean row is the legend's.
    """
    V, M = MEAS.atlas_skill()
    T2 = MEAS.table2_intervals()
    order = S.LEAF_ORDER
    YM = len(order) + 0.75                       # the mean row, set off from goat by a rule
    ax.set_ylim(YM + 0.62, -0.55)                # inverted: the tree's first leaf at the top
    lo = min(min(v["coding"], v["e1"]) for v in V.values())
    lo = min(lo, min(T2[(sp, r)][0] for sp in order for r in ("1001bp", "8192bp")
                     if sp not in MEAS.NO_CLUSTER_INTERVAL))
    ax.set_xlim(np.floor((lo - 0.015) / 0.05) * 0.05, 1.006)
    for x in np.arange(0.7, 1.0001, 0.1):
        ax.axvline(x, color=SG.GRID, lw=.5, zorder=0)
    cap = MEAS._dy(ax, 1.5)                      # whisker cap half-height, in points

    def _dot(y, v, key, ci=None):
        if ci is not None:
            c = DOT_STYLE[key]["mec"]
            ax.plot([ci[0], ci[1]], [y, y], color=c, lw=.8, zorder=3, solid_capstyle="butt")
            for xx in ci:
                ax.plot([xx, xx], [y - cap, y + cap], color=c, lw=.8, zorder=3)
        ax.plot([v], [y], linestyle="none", zorder=5, **DOT_STYLE[key])

    for i, sp in enumerate(order):
        v = V[sp]
        weak = sp in MEAS.NO_CLUSTER_INTERVAL
        for key, readout in (("e1", "1001bp"), ("e8", "8192bp")):
            _dot(i + DOT_OFF[key], v[key], key, None if weak else T2[(sp, readout)][:2])
        for key in ("coding", "impact"):
            _dot(i + DOT_OFF[key], v[key], key)
    for key in ("e1", "e8", "coding", "impact"):
        m, l_, h_ = M[key]
        _dot(YM + DOT_OFF[key], m, key, (l_, h_) if key in ("e1", "e8") else None)
    ax.axhline(len(order) - 0.5 + 0.24, color=SG.RULE, lw=.5, zorder=1)
    ax.set_yticks(list(range(len(order))) + [YM])
    ax.set_yticklabels([sp + (MEAS.DAGGER if sp in MEAS.NO_CLUSTER_INTERVAL else "")
                        for sp in order] + ["species mean"], fontsize=SG.TICK)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.set_xticks(np.arange(0.7, 1.0001, 0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "%.1f" % x))
    ax.set_xlabel("AUROC", fontsize=SG.AXIS)
    SG.spines(ax, keep=("bottom",))

    # ---- the tree and the silhouettes, seated against the MEASURED row names
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw = fig.bbox.width
    names_x0 = min(t.get_window_extent(rend).x0 for t, sp in
                   zip(ax.get_yticklabels(), order + ["mean"]) if sp in order) / fw
    pos = ax.get_position()
    sil_c = names_x0 - _pt_x(fig, 3.0 + SIL_PT / 2.0)            # silhouette centre, 3 pt clear
    tree_x1 = sil_c - _pt_x(fig, SIL_PT / 2.0 + 3.0)
    tree_x0 = SG.letter_x(fig) + _pt_x(fig, 6.0)
    axT = fig.add_axes([tree_x0, pos.y0, tree_x1 - tree_x0, pos.height], sharey=ax)
    axT.axis("off")
    lay = S.layout_tree()
    root = lay["age"]
    axT.set_xlim(root * 1.04, 0.0)               # time runs right: tips at the right edge
    _rect_tree(axT, lay)
    axT.plot([root * 1.04, root], [lay["y"], lay["y"]], color="#5A5A5A", lw=.9, zorder=2)
    # Scale bar, 100 My, under the ROOT, on the mean row's height, where the tree has no leaf. At
    # the tips' end it ran into the mean row's name, which is right-aligned against the dot axes
    # and reaches back over the silhouette column.
    ysb = YM
    axT.plot([root, root - 100.0], [ysb, ysb], color=SG.MUTED, lw=.8, solid_capstyle="butt",
             zorder=2)
    axT.annotate("100 My", xy=(root - 50.0, ysb), xytext=(0, -2.0), textcoords="offset points",
                 ha="center", va="top", fontsize=SG.ANNOT, color=SG.MUTED, annotation_clip=False)
    tf = blended_transform_factory(fig.transFigure, ax.transData)
    for i, sp in enumerate(order):
        im = OffsetImage(S.load_silhouette(sp, S.SIL_INK), zoom=SIL_PT / 256.0)
        ax.add_artist(AnnotationBbox(im, (sil_c, i), xycoords=tf, frameon=False,
                                     box_alignment=(0.5, 0.5), zorder=4, annotation_clip=False))

    # ---- the key: which mark is which, in the order the sub-rows run
    h = [Line2D([0], [0], linestyle="none", label=DOT_LABEL[k], **DOT_STYLE[k])
         for k in ("e1", "e8", "coding", "impact")]
    # Two columns, Evo 2's readouts in the first and the two baselines in the second; seated 2 pt
    # above the dot axes, from the tree's left edge, so it ends well short of panel b's column.
    lg = fig.legend(handles=h, loc="lower left", ncol=2, frameon=False, fontsize=SG.ANNOT,
                    bbox_to_anchor=(tree_x0, pos.y1 + 2.0 / (fig.get_size_inches()[1] * 72.0)),
                    bbox_transform=fig.transFigure, handletextpad=0.35, columnspacing=1.4,
                    labelspacing=0.35, borderaxespad=0.0, borderpad=0.0)
    for t in lg.get_texts():
        t.set_color(SG.INK)
    return axT


def main():
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    build(ax)
    fig.savefig("reports/figures/_fig2_hero.png", bbox_inches="tight", dpi=200)
    print("wrote reports/figures/_fig2_hero.png")


if __name__ == "__main__":
    main()
