"""The measurement figure, its supplement, and the two atlas panels that read the per-variant table.

fig2_split draws Figure S7 from panel_c and panel_d, and fig7_atlas and fig8_readout read the atlas
through this module; panel_a, panel_b, panel_headtohead and fig2_defgh.panel_D/E/F/H are not called.

  measurement plate (panel_a, panel_b, panel_headtohead)
    a  discrimination melts to the missense floor -- drawn against the panel's own
       composition, to scale, so the reader sees what each stratum is made of.
    b  context lifts every stratum, the missense bedrock least -- 13 slopes: the four
       strata and all nine species, so the stratum result and the species result are read off
       one pair of axes.
    c  the head-to-head the section heading claims: per species, Evo 2 minus GERP on the
       variants all three scores cover, at both readouts, each with a two-sample locus-bootstrap
       interval, and the species mean with its t interval.
  supplementary plate (panel_c, panel_d) -- the two panels the measurement plate used to carry
    a  no detectable grading with divergence -- with the interval on every point and the
       pooled band behind them.
    b  the conservation plane -- GERP against Evo 2 at 1,001 bp -- with the paired interval on
       every per-species difference beside it.
  atlas plate (panel_composition; its panel a is fig2_hero.build_dotplot, which reads
  atlas_skill() and table2_intervals() below)
    c  what each species' panel is made of: the coding share of positives and of negatives.

Every number is read from the recompute layer (`reports/fig2_data.json`,
`reports/type_matched_atlas.parquet`, `reports/fig2_pervariant.parquet`) or computed at build time
from Additional file 3's tables (`fig1_atlas_pervariant.parquet`, `table2_cluster_aware_ci.tsv`,
`refseq_chromosome_map.csv`). Nothing is typed.
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG
try:
    from . import fig2_style as S
except ImportError:
    import fig2_style as S

R = "reports"


def _deposit_table(name):
    """Locate a reduced table that ships in Additional file 3, not in this repository.

    Figures 9 and 10 and Additional file 1's Figures S3, S5 and S7 read per-variant tables from the
    recompute deposit. Resolving one as a bare relative path silently depended on the maintainer's
    own directory layout, so the builders failed for anyone who unpacked only this archive. Look in
    the places a reader could reasonably have put it, then say exactly what is missing and where it
    comes from.
    """
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    cands = [os.environ.get("CCS_TABLES") and os.path.join(os.environ["CCS_TABLES"], name),
             os.path.join(root, "tables", name),
             os.path.join(root, os.pardir, "final10", "tables", name),
             os.path.join(root, "reports", name)]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "%s ships in Additional file 3 (Recompute tables and scripts), under tables/. "
        "Copy it to %s, or set CCS_TABLES to the directory holding it."
        % (name, os.path.join(root, "tables")))


FLOOR = SG.FLOOR
# Annotations in this module are set at ANNOT, the tier the rest of the figure set uses;
# FLOOR is the floor the checker enforces, not a style.
ANNOT = SG.ANNOT
PATHO, BENIGN = S.PATHO, S.BENIGN
GERP_C = "#5C5C5C"

STRATA = [("ALL (unmatched, as published)", "all variants", "unmatched"),
          ("CODING-only", "coding only", "coding"),
          ("non-coding only", "non-coding only", "non-coding"),
          ("MISSENSE-only (type-matched)", "missense, type-matched (within coding)", "missense")]


def _load():
    d = json.load(open(os.path.join(R, "fig2_data.json"), encoding="utf-8"))
    asc = {r["stratum"]: r for r in d["ascert"]}
    return d, asc, d["ascert_ghost1001"]


def _seat(ys, lo, hi, gap):
    """Push labels apart to a minimum gap, keeping their order and staying inside [lo, hi].

    A slopegraph's right-hand labels collide wherever the series converge, which is exactly
    where the reader most needs to tell them apart. Ordering is preserved, so a label never
    crosses its neighbour's line.
    """
    o = np.argsort(ys)
    v = np.array(ys, dtype=float)[o]
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + gap)
    over = v[-1] - hi
    if over > 0:
        v -= over
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], v[i + 1] - gap)
    v = np.maximum(v, lo)
    out = np.empty_like(v)
    out[o] = v
    return out


# ---------------------------------------------------------------- a : composition and the melt
def panel_a(axC, axR):
    """Left: what the panel is made of, to scale. Right: what each stratum scores, both readouts.

    Laid out as a forest: the stratum names are tick labels, the counts are an annotation column
    between the two axes, and the readout gain is an annotation column to the right of the second.
    Putting any of that inside the composition axes does not fit at 170 mm -- the longest row label
    plus its counts sets to 58 mm against a 38 mm panel.
    """
    _, asc, ghost = _load()
    ntot = asc["ALL (unmatched, as published)"]["n"]
    start = {"ALL (unmatched, as published)": 0.0, "CODING-only": 0.0,
             "non-coding only": asc["CODING-only"]["n"] / ntot,
             "MISSENSE-only (type-matched)": 0.0}
    for i, (k, lab, _short) in enumerate(STRATA):
        r = asc[k]
        x0 = start[k]; w = r["n"] / ntot; fpos = r["pos"] / r["n"]
        axC.add_patch(plt.Rectangle((x0, i - .17), w * fpos, .34, facecolor=PATHO,
                                    edgecolor="none", zorder=4))
        axC.add_patch(plt.Rectangle((x0 + w * fpos, i - .17), w * (1 - fpos), .34,
                                    facecolor=BENIGN, alpha=.55, edgecolor="none", zorder=3))
        pct = ("%.0f%%" if 100 * fpos >= 10 else "%.1f%%") % (100 * fpos)
        axC.text(1.22, i, "{:,}".format(r["n"]), transform=axC.get_yaxis_transform(),
                 fontsize=ANNOT, ha="right", va="center", color=SG.MUTED, clip_on=False)
        axC.text(1.50, i, pct, transform=axC.get_yaxis_transform(), fontsize=ANNOT,
                 ha="right", va="center", color=PATHO, clip_on=False)
    # "positive" and "% pos.", not "pathogenic": the atlas positives are catalogued alleles, and 203 of
    # the 737 scored OMIA positives are trait rather than disease alleles (Table S13), so the class
    # names follow the atlas plate's positives/negatives.
    for x, h in ((1.22, "n"), (1.50, "% pos.")):
        axC.text(x, -0.72, h, transform=axC.get_yaxis_transform(), fontsize=ANNOT,
                 ha="right", va="bottom", color=SG.MUTED, clip_on=False)
    # the one containment the widths cannot show: missense is a subset of coding, not a fifth slice
    wm = asc["MISSENSE-only (type-matched)"]["n"] / ntot
    axC.plot([wm, wm], [1.20, 2.80], color=SG.RULE, lw=.5, ls=(0, (2, 1.6)), zorder=2)
    axC.set_xlim(0, 1.005); axC.set_ylim(-0.85, 3.45); axC.invert_yaxis()
    axC.set_yticks(range(4))
    # "unmatched", not "all": the same word the legend and panel b use for this stratum. Three of
    # the four strata already agreed; only the top one had two names, and "all" is the one that
    # loses the point -- the row exists to be the NOT-consequence-matched comparison the other
    # three are matched against.
    axC.set_yticklabels(["unmatched\n(all variants)", "coding\nonly", "non-coding\nonly",
                         "missense\n(type-matched)"], fontsize=ANNOT, linespacing=1.35)
    axC.set_xticks([0, .25, .5, .75, 1.0])
    axC.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=ANNOT)
    axC.set_xlabel("share of the %s-variant panel (%%)" % "{:,}".format(ntot), fontsize=SG.AXIS)
    SG.spines(axC, keep=("bottom",))
    axC.tick_params(axis="y", length=0)
    # The key sits in the header row, level with "n" and "% pos.". Inside the bars' rows it shared the
    # missense row with that row's own count, and read as part of it.
    for x, fc, al, txt in ((0.0, PATHO, 1.0, "positive"), (0.47, BENIGN, .55, "negative")):
        axC.add_patch(plt.Rectangle((x, -0.83), .035, .16, facecolor=fc, alpha=al,
                                    edgecolor="none", zorder=4, clip_on=False))
        axC.text(x + .05, -0.75, txt, fontsize=ANNOT, ha="left", va="center", color=SG.INK,
                 clip_on=False)

    for i, (k, _, _short) in enumerate(STRATA):
        r = asc[k]; g = ghost[k]
        axR.plot([g, r["auroc"]], [i, i], lw=1.1, color=SG.INK, alpha=.45, zorder=3)
        axR.plot([r["lo"], r["hi"]], [i, i], lw=.7, color=SG.INK, zorder=4)
        for xx in (r["lo"], r["hi"]):
            axR.plot([xx, xx], [i - .12, i + .12], lw=.7, color=SG.INK, zorder=4)
        axR.scatter(g, i, s=17, marker="o", facecolor="white", edgecolor=SG.INK, lw=.7, zorder=5)
        axR.scatter(r["auroc"], i, s=19, marker="o", facecolor=SG.INK, edgecolor="white",
                    lw=.45, zorder=6)
        axR.text(1.03, i, "%+.3f" % (r["auroc"] - g), transform=axR.get_yaxis_transform(),
                 fontsize=ANNOT, va="center", ha="left", color=SG.MUTED, clip_on=False)
    floor = asc["MISSENSE-only (type-matched)"]["auroc"]
    # The rule stops under its own label: drawn full height it ran through "pooled missense".
    axR.plot([floor, floor], [-0.50, 3.45], color=S.PATHO, lw=.6, ls=(0, (2.5, 2)), zorder=2)
    axR.text(floor, -0.72, "pooled missense %.3f" % floor, fontsize=ANNOT, ha="center",
             va="bottom", color=S.PATHO)
    axR.text(1.03, -0.72, "readout gain", transform=axR.get_yaxis_transform(),
             fontsize=ANNOT, va="bottom", ha="left", color=SG.MUTED, clip_on=False)
    axR.set_ylim(-0.85, 3.45); axR.invert_yaxis(); axR.set_yticks([])
    axR.set_xlim(.60, 1.0); axR.set_xticks([.6, .7, .8, .9, 1.0])
    # "pooled": each stratum's AUROC ranks its variants across all nine species together, where the
    # paper's primary aggregation is the species mean.
    # The whiskers' meaning (95% CI) is the legend's: with it, the label ran off the page.
    axR.set_xlabel("pooled AUROC   (open, single token;  filled, window mean)", fontsize=SG.AXIS)
    SG.spines(axR, keep=("bottom",))
    axR.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- b : the readout, 13 slopes
def panel_b(ax1, ax2):
    d, asc, ghost = _load()
    YLO, YHI = .60, 1.0
    for ax in (ax1, ax2):
        ax.set_xlim(-.30, 1.34); ax.set_ylim(YLO, YHI)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["1,001", "8,192"], fontsize=ANNOT)
        SG.spines(ax, keep=("left", "bottom"))
        ax.tick_params(axis="x", length=0)
    ax2.set_yticks([]); SG.spines(ax2, keep=("bottom",))
    ax1.set_yticks([.6, .7, .8, .9, 1.0]); ax1.set_ylabel("AUROC", fontsize=SG.AXIS)

    lab1 = [(STRATA[i][2], ghost[STRATA[i][0]], asc[STRATA[i][0]]["auroc"]) for i in range(4)]
    seat1 = _seat([b for _, _, b in lab1], YLO + .02, YHI - .01, .030)
    for (nm, a, b), ys in zip(lab1, seat1):
        ax1.plot([0, 1], [a, b], lw=1.1, color=SG.INK, alpha=.75, zorder=3)
        ax1.scatter([0], [a], s=15, facecolor="white", edgecolor=SG.INK, lw=.7, zorder=5)
        ax1.scatter([1], [b], s=17, facecolor=SG.INK, edgecolor="white", lw=.45, zorder=5)
        ax1.plot([1.02, 1.10], [b, ys], lw=.4, color=SG.RULE, zorder=2)
        ax1.text(1.12, ys, "%s  %.3f" % (nm, b), fontsize=ANNOT, va="center", ha="left",
                 color=SG.INK)

    F = sorted(d["forest"], key=lambda r: -r["auroc"])
    seat2 = _seat([r["auroc"] for r in F], YLO + .02, YHI - .01, .030)
    for r, ys in zip(F, seat2):
        c = S.species_color(r["species"])
        ax2.plot([0, 1], [r["auroc_1001"], r["auroc"]], lw=.9, color=c, alpha=.85, zorder=3)
        ax2.scatter([0], [r["auroc_1001"]], s=13, marker="o", facecolor="white", edgecolor=c,
                    lw=.65, zorder=5)
        ax2.scatter([1], [r["auroc"]], s=15, marker="o", facecolor=c, edgecolor="white",
                    lw=.4, zorder=5)
        ax2.plot([1.02, 1.10], [r["auroc"], ys], lw=.4, color=SG.RULE, zorder=2)
        # The leader line and markers keep the full-chroma species colour; the WORD takes the
        # text-safe rendition -- in full chroma, cat's label is 1.79:1 on white.
        # FLOOR: a nine-row key beside the panel; at ANNOT it reaches the rotated
        # y-label of the panel to its right, which is already at labelpad 1.0. A plate that sets
        # panel c to its right buys the clearance there, shifting panel c's left edge right into its
        # own unused margin, rather than here: this column is nine rows of data at the floor and
        # cannot give.
        ax2.text(1.12, ys, "%s  %.3f→%.3f" % (r["species"], r["auroc_1001"], r["auroc"]),
                 fontsize=FLOOR, va="center", ha="left", color=S.species_text_color(r["species"]))
    # The two headers sit in the empty band at the foot of each panel. At the top they shared the
    # rows where the highest lines end and their labels begin, and both were overdrawn there.
    ax1.text(0.5, YLO + .012, "four strata", fontsize=SG.ANNOT, ha="center", va="bottom",
             color=SG.MUTED)
    ax2.text(0.5, YLO + .012, "nine species", fontsize=SG.ANNOT, ha="center", va="bottom",
             color=SG.MUTED)


# ---------------------------------------------------------------- c : divergence, with intervals
def panel_c(ax):
    d, _, _ = _load()
    P = d["pooled"]
    ax.axhspan(P["lo"], P["hi"], color=S.EVO2, alpha=.10, zorder=1)
    ax.axhline(P["auroc"], color=S.EVO2, lw=.6, ls=(0, (3, 2)), zorder=2)
    F = d["forest"]
    grp = {}
    for r in F:
        grp.setdefault(r["divergence_my"], []).append(r)
    # CATEGORICAL SLOTS, NOT A NUMERIC AXIS. The nine species sit at three divergences only -- human
    # 0 My, seven mammals 94 My, chicken 319 My -- and on a numeric divergence axis the seven had to be
    # dodged across +/-45 My of x that carried no divergence, which read as seven different ages. Each
    # divergence is now a slot of its own, the seven mammals spread evenly inside theirs, and the axis
    # line is broken between slots so that no distance along it can be read as time.
    slots = sorted(grp)
    SLOT_W = {my: (0.0 if len(grp[my]) == 1 else 0.42 * (len(grp[my]) - 1)) for my in slots}
    slot_x, x0 = {}, 0.0
    for j, my in enumerate(slots):
        slot_x[my] = x0 + SLOT_W[my] / 2
        x0 += SLOT_W[my] + 1.0
    for my, rows in grp.items():
        # ALPHABETICAL, not by descending AUROC. Seven of the nine species -- cattle, dog, sheep,
        # goat, horse, cat and pig -- share one divergence, 94 My. They have to be dodged apart or
        # their intervals overlap into one bar, but the dodge ORDER is arbitrary and the old one
        # was -auroc, which laid them out 0.974, 0.970, 0.962, 0.959, 0.941, 0.890, 0.863 from left
        # to right across +/-45 My of fabricated x. The panel therefore DREW a clean monotone fall
        # of discrimination with divergence -- which is exactly the grading its own Spearman test
        # reports as absent (rho = -0.365, p = 0.334, n = 9), stated in the legend two lines later.
        # A reader who looks at the picture and a reader who reads the test got opposite answers.
        # Alphabetical gives 0.890, 0.974, 0.970, 0.959, 0.941, 0.863, 0.962: visibly non-monotone,
        # so no trend can be read off an ordering that carries no information.
        rows = sorted(rows, key=lambda r: r["species"])
        span = SLOT_W[my]
        xs = np.linspace(-span / 2, span / 2, len(rows)) if len(rows) > 1 else [0.0]
        for r, dx in zip(rows, xs):
            c = S.species_color(r["species"])
            x = slot_x[my] + dx
            ax.plot([x, x], [r["lo"], r["hi"]], lw=.7, color=S._shade(c, -0.55), alpha=1.0,
                    zorder=6)
            ax.scatter(x, r["auroc"], s=4 + 34 * (r["n"] / 3000.0), marker="o",
                       facecolor=c, edgecolor="white", lw=.4, zorder=5)
    XLO, XHI = -0.6, x0 - 1.0 + 0.6
    ax.set_xlim(XLO, XHI); ax.set_ylim(.72, 1.005)
    # The ticks are the divergences the forest carries, and the names under them are computed from
    # it too: a lone species is named, a shared divergence is "<k> mammals". "7 mammals", not
    # "mammals": the tick has to say that seven marks share this one x value, or the horizontal
    # spread still reads as seven different divergences. The names are a row of their own under the
    # numbers, the first set to the left of its tick and the second to the right of its own:
    # centred, the two ticks sit 94 My apart and the names met as one phrase.
    ticks = slots
    ax.set_xticks([slot_x[t] for t in ticks])
    ax.set_xticklabels(["%d My" % t for t in ticks], fontsize=ANNOT)
    # 16.5 pt below the axis line, in points. At 12.5 pt this row collided with the tick labels
    # above it by 1.47 pt (tools/pdfcheck.py): a tick label at ANNOT ends 2.2 pt of tick + 3.5 pt of
    # pad + one 8.5 pt line box below the axis, 14.2 pt, and the row must start under that.
    _names = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-16.5,
                                     units="points")
    for j, t in enumerate(ticks):
        txt = grp[t][0]["species"] if len(grp[t]) == 1 else "%d mammals" % len(grp[t])
        ax.text(slot_x[t], 0, txt, transform=_names, fontsize=ANNOT, ha="center", va="top",
                color=SG.INK, clip_on=False)
    ax.set_xlabel("divergence from human (TimeTree), one slot per divergence", fontsize=SG.AXIS,
                  labelpad=16.0)
    # The default labelpad: this panel was pulled in to 1.0 on the measurement plate, where panel
    # b's nine-species key ran up to its y-label. On the supplementary plate nothing sits to its left.
    ax.set_ylabel("AUROC, 8,192 bp", fontsize=SG.AXIS)
    SG.spines(ax)
    # The broken x axis: one spine segment under each slot, none between them.
    ax.spines["bottom"].set_visible(False)
    for t in ticks:
        lo_, hi_ = slot_x[t] - SLOT_W[t] / 2 - 0.25, slot_x[t] + SLOT_W[t] / 2 + 0.25
        ax.plot([lo_, hi_], [0, 0], transform=ax.get_xaxis_transform(), color=SG.INK, lw=.5,
                clip_on=False, solid_capstyle="butt", zorder=2)
    # "pooled" in the empty stretch between the mammals' slot and chicken's
    _mid = (slot_x[ticks[-2]] + SLOT_W[ticks[-2]] / 2 + slot_x[ticks[-1]]) / 2
    ax.text(_mid, P["auroc"] + .004, "pooled", fontsize=ANNOT, color=S.EVO2, ha="center",
            va="bottom")
    # The seven non-human mammals all sit at 94 My and are spread across x only so their intervals
    # do not overlap; the tick says "7 mammals" and the legend says the spread carries no
    # divergence. A two-line note saying so on the canvas was wider than the panel and was crossed by
    # pig's interval, so the note lives in the legend, with the marker-area key.
    return P, d["distance"], len(F)


# ---------------------------------------------------------------- d : the conservation plane
def panel_d(axP, axS):
    """The GERP comparison, drawn on the two scores rather than as nine subtracted numbers.

    Every one of the 9,532 variants both methods can score at the 1,001-bp readout sits at its GERP
    value against its Evo 2 value. 9,532, not 9,529: the latter is the co-scorable set at 8,192 bp
    (gerp x evo2_meanll_8192), and this plane and the strip beside it are both at 1,001 bp.
    The plane shows WHY the two agree: the agreement is the shape of the cloud, the
    disagreement is where the two classes separate on one
    axis and not the other, and the nine per-species differences are drawn as the strip on the right.
    """
    import pandas as pd
    d = pd.read_parquet(os.path.join(ROOT, "fig1_atlas_pervariant.parquet")) \
        if False else pd.read_parquet(_deposit_table("fig1_atlas_pervariant.parquet"))
    m = d[d.gerp.notna() & d.evo2_1001.notna()]
    # WITHIN-SPECIES PERCENTILES, NOT RAW SCORES. The nine GERP tracks come from different alignments
    # and are not on one scale, so a plane of raw GERP pools scores that cannot be pooled; each score
    # is therefore ranked within its own species (midranks for ties) and drawn as a percentile. The
    # two classes are compared within species throughout the paper, and so they are here.
    x = 100.0 * m.groupby("species").gerp.rank(method="average", pct=True).to_numpy()
    # THE READOUT MUST MATCH THE COMPARISON. The deposited GERP block in fig2_data.json is computed
    # at the 1,001-bp single-position readout -- verified against every one of the nine species to
    # four decimals -- so the strip beside this plane is a 1,001-bp comparison. Drawing the cloud at
    # 8,192 bp put two different readouts in one panel under one axis label.
    y = 100.0 * m.groupby("species").evo2_1001.rank(method="average", pct=True).to_numpy()
    lab = m.label.to_numpy()
    XL, XH, YL, YH = 0.0, 100.0, 0.0, 100.0
    k = (x >= XL) & (x <= XH) & (y >= YL) & (y <= YH)
    # The hexbin's greys are TRUNCATED at [0.08, 0.45] of the ramp instead of running to black.
    # The catalogued alleles drawn on top are S.PATHO, whose relative luminance is 0.158 -- which
    # sat INSIDE the full ramp's range, 1.04 against Greys(0.65). In greyscale a single catalogued
    # allele was invisible over any mid-density cell, which is the one comparison this panel exists
    # to make. Truncating is necessary but NOT sufficient: the marks carry alpha .40, so a lone
    # allele composites over white to (230,180,178), L 0.527 -- still inside the hexbin's rendered
    # 0.918..0.439 band: 1.67 against a count-1 cell and 1.18 against the densest. The 2.05 this
    # comment used to claim was measured on opaque #C1443E, which is not what prints. Shape is the
    # fix -- an open ring is never a filled hexagon -- and dropping alpha takes the worst pair to
    # 2.35 and the count-1 pair to 4.63 without adding a hue.
    _greys = plt.get_cmap("Greys")
    _hexcm = LinearSegmentedColormap.from_list(
        "greys_trunc", [_greys(0.08 + 0.37 * t) for t in np.linspace(0, 1, 64)])
    axP.hexbin(x[k & (lab == 0)], y[k & (lab == 0)], gridsize=(36, 26), extent=(XL, XH, YL, YH),
               cmap=_hexcm, bins="log", mincnt=1, linewidths=0, zorder=2)
    axP.scatter(x[k & (lab == 1)], y[k & (lab == 1)], s=1.6, marker="o", facecolor="none",
                edgecolor=S.PATHO, linewidths=.3, zorder=3)
    axP.set_xlim(XL, XH); axP.set_ylim(YL, YH)
    axP.set_xticks([0, 25, 50, 75, 100]); axP.set_yticks([0, 25, 50, 75, 100])
    axP.set_xlabel("GERP, percentile within species", fontsize=SG.AXIS)
    axP.set_ylabel("Evo 2, single token,\npercentile within species", fontsize=SG.AXIS,
                   linespacing=1.35)
    SG.spines(axP)
    # TEXT OFF THE DATA. Text set across the point cloud is unreadable and hides the data it
    # describes (see tools/inkunder.py).
    # The key carries the one thing a reader cannot get anywhere else: WHICH MARK IS WHICH. It is
    # a real legend with proxy handles rather than a sentence, so the key cannot drift from the
    # marks it names, and it sits in the panel's empty lower-left corner on a white patch.
    from matplotlib.lines import Line2D
    _h = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                 markeredgecolor=S.PATHO, markeredgewidth=.5, markersize=3.0,
                 label="catalogued allele"),
          Line2D([0], [0], marker="h", linestyle="none", markerfacecolor=_greys(0.30),
                 markeredgecolor="none", markersize=4.2, label="population variant")]
    _lg = axP.legend(handles=_h, loc="lower left", frameon=True, framealpha=0.88,
                     facecolor="white", edgecolor="none", borderpad=0.25, handletextpad=0.5,
                     labelspacing=0.35, borderaxespad=0.4, fontsize=ANNOT)
    for _t in _lg.get_texts():
        _t.set_color(SG.MUTED)
    _lg.set_zorder(6)

    d_ = _load()[0]
    G = d_["gerp"]
    rows = sorted(G.items(), key=lambda kv: -kv[1]["delta"])
    # Species whose positive class does not support a cluster-robust interval, so the variant-level
    # bootstrap this strip draws is NOT an honest uncertainty for them. Goat's nine positives occupy
    # two loci, eight of them inside a 570-bp span, giving Kish's effective count of 1.2 independent
    # positives (Methods; Additional file 1: Note S29); Table 2 accordingly prints goat's AUROC as
    # "0.959 (no interval)" and calls it a descriptive point estimate. Drawing goat the same solid
    # interval and the same *** as cattle asserted, on this plate, the significance the Methods
    # elsewhere refuse -- and half of the panel's own "2 of 9" count rested on it. The estimate is
    # still drawn, because it is the paper's estimate; what is withdrawn is the claim that its
    # interval means what the other eight intervals mean. The set is hard-coded rather than derived
    # because no deposited artefact carries the per-species cluster count; the citation above is the
    # audit trail, and if that changes this list is the one place to change.
    NO_CLUSTER_INTERVAL = {"goat"}
    for i, (sp, r) in enumerate(rows):
        c = S.species_color(sp)
        weak = sp in NO_CLUSTER_INTERVAL
        # No interval at all for these species, not a dotted one: the text says goat carries none,
        # and a dotted whisker is still an interval.
        if not weak:
            axS.plot([r["lo"], r["hi"]], [i, i], lw=.7, color=c, zorder=3)
            for xx in (r["lo"], r["hi"]):
                axS.plot([xx, xx], [i - .18, i + .18], lw=.7, color=c, zorder=3)
        axS.scatter(r["delta"], i, s=13, marker="D",
                    facecolor=("white" if weak else c), edgecolor=(c if weak else "white"),
                    lw=(0.7 if weak else .4), zorder=5)
        strong = r["verdict"] == "***" and not weak
        # Seated past the WIDEST interval, not at a fixed 1.06 of the axes width. At 1.06 the
        # column's left edge fell at data 0.206 on a (-.16, .32) axis, inside chicken's upper
        # limit of 0.293, and the whisker and its cap printed through the digits of goat's,
        # chicken's and pig's differences. The 3 pt gap is in POINTS (see _dy below).
        axS.annotate("%+.3f %s" % (r["delta"], "†" if weak else r["verdict"]),
                     xy=(max(rr["hi"] for _, rr in rows), i), xycoords="data",
                     xytext=(3, 0), textcoords="offset points",
                     fontsize=ANNOT, va="center", ha="left", annotation_clip=False,
                     color=(SG.INK if strong else SG.MUTED))
    axS.axvline(0, color=SG.RULE, lw=.6, ls=(0, (2.5, 2)), zorder=2)
    axS.set_yticks(range(len(rows)))
    axS.set_yticklabels([sp for sp, _ in rows], fontsize=ANNOT)
    for tick, (sp, _) in zip(axS.get_yticklabels(), rows):
        tick.set_color(S.species_text_color(sp))
    axS.set_ylim(-.8, len(rows) - .3); axS.invert_yaxis()
    axS.set_xlim(-.16, .32); axS.set_xticks([-.1, 0, .1, .2, .3])
    axS.set_xlabel("Evo 2 − GERP, paired\n(AUROC, 95% CI)", fontsize=SG.AXIS, linespacing=1.35)
    SG.spines(axS, keep=("bottom",))
    axS.tick_params(axis="y", length=0)
    axS.annotate("difference", xy=(max(rr["hi"] for _, rr in rows), -0.70), xycoords="data",
                 xytext=(3, 0), textcoords="offset points", fontsize=ANNOT, va="bottom",
                 ha="left", color=SG.MUTED, annotation_clip=False)
    return rows


def _dy(ax, pt):
    """`pt` points expressed in this axes' data units on the y axis.

    Layout distances that exist to clear a GLYPH have to be authored in points: a fraction of the
    ylim shrinks when the canvas or the box does, while the type in it does not. This paper lost
    an afternoon to that class of bug when the plates went from 195 mm to 173 mm.

    abs(): an inverted axis (the row plots below count rows downward) has hi < lo, and a distance
    is a distance.
    """
    h_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72.0
    lo, hi = ax.get_ylim()
    return abs(hi - lo) * pt / h_pt


# ------------------------------------------------ the per-variant atlas table, read once
# Figures 9a, 9c, 10a and 10c are computed at build time from ONE deposited table, Additional
# file 3's tables/fig1_atlas_pervariant.parquet, so no value they draw is carried by a JSON that
# could lag it.
#
# Orientation, identical to Additional file 3's recompute scripts: higher = more pathogenic.
#   e1 = evo2_1001                 as stored
#   e8 = -evo2_meanll_8192         the negated mean-log-likelihood delta (21 NULL rows)
#   g  = gerp                      as stored
# NaN means the score did not answer; every mask below uses np.isfinite.
#
# The two label-free baselines of Table 2. These dict literals are DEFINITIONS, copied from
# Additional file 3's scripts/recompute_macro_ci.py, which asserts every Table 2 cell they produce:
# the FIRST term of consequence.split("&") decides; CODING is a set of Sequence Ontology terms;
# IMPACT is snpEff's four-level ordering, and every term it does not list is MODIFIER, 0. CODING is
# the same nine terms as CODING_SO below. If either changes there it has to change here.
CODING = {"missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost",
          "frameshift_variant", "initiator_codon_variant", "splice_donor_variant",
          "splice_acceptor_variant"}
IMPACT = {"stop_gained": 3, "stop_lost": 3, "start_lost": 3, "splice_donor_variant": 3,
          "splice_acceptor_variant": 3, "frameshift_variant": 3, "missense_variant": 2,
          "synonymous_variant": 1, "splice_region_variant": 1,
          "5_prime_UTR_premature_start_codon_gain_variant": 1}      # every other term: MODIFIER, 0

# Species whose positives do not support a cluster-robust interval. Goat's nine positives occupy two
# 100-kb loci, eight of them inside a 570-bp span (Kish's effective count 1.25); Table 2 prints its
# AUROC with no interval and the Methods call it a descriptive point estimate. On the variants all
# three scores cover, its eight positives sit in ONE locus, so a locus bootstrap cannot resample
# them at all. Every panel that draws an interval draws none for these species. The set is checked
# against the deposited cluster counts on every build (table2_intervals, headtohead), so a change in
# the data cannot leave it stale in silence.
NO_CLUSTER_INTERVAL = {"goat"}
LOCUS_BP = 100_000          # the locus definition of table2_cluster_aware_ci.py
DAGGER = "†"           # marks a species drawn without an interval; the legend says why

_ATLAS = None


def _auroc(y, s):
    """AUROC as the Mann-Whitney statistic with midranks for ties, as the recompute scripts compute it."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _parse_vid(vid):
    """'[neg_]<chrom>_<pos>_<ref>_<alt>' -> (chrom, pos), RIGHT-anchored.

    A copy of Additional file 3's table2_cluster_aware_ci.parse_vid, which is not importable from
    this archive. A naive split('_') breaks on the 488 ids that sit on RefSeq accessions
    (NC_030833.1_46369302_G_A), whose contig name itself contains an underscore.
    """
    s = vid[4:] if vid.startswith("neg_") else vid
    head, ref, alt = s.rsplit("_", 2)
    chrom, pos = head.rsplit("_", 1)
    if not (set(ref) <= set("ACGTN") and set(alt) <= set("ACGTN") and pos.isdigit() and chrom):
        raise ValueError("variant_id parse failure: %r" % vid)
    return chrom, int(pos)


def _atlas():
    """The atlas table with the derived columns every panel below reads. Cached: three panels share it.

    Loci are keyed on the PHYSICAL chromosome through refseq_chromosome_map.csv, as
    table2_cluster_aware_ci.py keys them: seven species carry some positives under a RefSeq
    accession and others under a bare chromosome name, and keying on the label would count one
    100-kb locus twice wherever both occur on it.
    """
    global _ATLAS
    if _ATLAS is None:
        import pandas as pd
        d = pd.read_parquet(_deposit_table("fig1_atlas_pervariant.parquet"))
        if not np.array_equal(d.variant_id.str.startswith("neg_").to_numpy(), d.label.to_numpy() == 0):
            raise SystemExit("fig1_atlas_pervariant: the neg_ prefix disagrees with label")
        cm = pd.read_csv(_deposit_table("refseq_chromosome_map.csv"), dtype=str)
        r2c = {(a, b): c for a, b, c in zip(cm.species, cm.refseq_accession, cm.chromosome)}
        pv = [_parse_vid(v) for v in d.variant_id]
        d["locus"] = [r2c.get((sp, c), c) + ":" + str(p // LOCUS_BP)
                      for sp, (c, p) in zip(d.species, pv)]
        d["e1"] = d.evo2_1001.astype(float)
        d["e8"] = -d.evo2_meanll_8192.astype(float)
        d["g"] = d.gerp.astype(float)
        first = d.consequence.astype("string").str.split("&").str[0]
        # All nine panels are annotated (every variant carries a consequence). The coding flag of a
        # variant with no consequence is not 0 -- it is unknown -- and the flag's AUROC and the
        # composition panel would both silently treat it as non-coding. Refuse instead.
        if first.isna().any():
            raise SystemExit("fig1_atlas_pervariant: %d variants carry no consequence; the coding "
                             "flag and the composition panel are defined on annotated panels only"
                             % int(first.isna().sum()))
        d["coding"] = first.isin(CODING).astype(float)
        d["impact"] = first.map(IMPACT).fillna(0).astype(float)
        _ATLAS = d
    return _ATLAS


def _tint(v):
    """Mean over species with its two-sided 95% t interval on k - 1 degrees of freedom.

    The paper's rule for every unweighted mean over the nine species: a percentile bootstrap over
    nine clusters runs about a fifth too narrow (Methods; Table 2's macro row uses the same t).
    """
    from scipy import stats
    v = np.asarray(v, float)
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return float(v.mean()), float(v.mean() - tc * se), float(v.mean() + tc * se)


def atlas_skill():
    """The values of Figures 9a and 10a: per species, Evo 2 at both readouts and the two label-free
    baselines.

    Each on its own full panel, as Table 2 computes them: 1,001 bp and the two baselines on every
    variant, 8,192 bp on the variants that readout scores. Also the composition that DECIDES the
    coding flag's AUROC. A one-bit score ties every coding positive with every coding negative, so
    AUROC(coding) = (1 + P(coding | positive) - P(coding | negative)) / 2 exactly; that identity is
    asserted, which ties Figure 9a's coding flag to Figure 9c's two bars for every species.
    """
    d = _atlas()
    out = {}
    for sp in S.LEAF_ORDER:
        g = d[d.species == sp]
        a1 = g[np.isfinite(g.e1)]
        a8 = g[np.isfinite(g.e8)]
        r = {"e1": _auroc(a1.label, a1.e1), "e8": _auroc(a8.label, a8.e8),
             "coding": _auroc(g.label, g.coding), "impact": _auroc(g.label, g.impact),
             "pos_coding": float(g[g.label == 1].coding.mean()),
             "neg_coding": float(g[g.label == 0].coding.mean()),
             "n_pos": int((g.label == 1).sum()), "n_neg": int((g.label == 0).sum())}
        ident = 0.5 * (1.0 + r["pos_coding"] - r["neg_coding"])
        assert abs(ident - r["coding"]) < 1e-12, (sp, ident, r["coding"])
        out[sp] = r
    mean = {k: _tint([out[s][k] for s in S.LEAF_ORDER]) for k in ("e1", "e8", "coding", "impact")}
    return out, mean


def table2_intervals():
    """Table 2's per-species 95% intervals, as PRINTED: the `env_lo` / `env_hi` columns of Additional
    file 3's tables/table2_cluster_aware_ci.tsv -- the wider of the variant-level and the two-sample
    locus-bootstrap endpoints (its README), which table2_cluster_aware_ci.py gates against the
    printed cells. Keyed (species, "1001bp" | "8192bp").

    Two checks on every build. The TSV's own AUROC must equal the one atlas_skill() computes from the
    per-variant table to its four printed decimals, so the whisker and the dot come from the same
    panel. And the species drawn without an interval must be exactly the ones whose positives occupy
    fewer than three loci.
    """
    import pandas as pd
    t = pd.read_csv(_deposit_table("table2_cluster_aware_ci.tsv"), sep="\t")
    skill, _ = atlas_skill()
    out = {}
    for r in t.itertuples():
        mine = skill[r.species]["e1" if r.readout == "1001bp" else "e8"]
        assert abs(round(mine, 4) - r.auroc) < 1e-9, (r.species, r.readout, mine, r.auroc)
        out[(r.species, r.readout)] = (float(r.env_lo), float(r.env_hi), int(r.pos_loci))
    few = {sp for (sp, _), v in out.items() if v[2] < 3}
    assert few == NO_CLUSTER_INTERVAL, (few, NO_CLUSTER_INTERVAL)
    return out


_H2H = None


def headtohead(B=2000, seed=61):
    """Figure 10c's values: per species, Evo 2 minus GERP on the variants evo2_1001, evo2_meanll_8192
    and gerp ALL score, at both readouts, with a two-sample locus-bootstrap 95% interval.

    The bootstrap is the one Table 2's clustered intervals use, restricted to one species at a time:
    the positive-bearing 100-kb loci are drawn whole with replacement, then the negatives are drawn
    individually with replacement, from a FRESH numpy default_rng(seed) per species (so no species'
    interval depends on which species were drawn before it). All three AUROCs are computed on the
    SAME resampled variants, so the 1,001-bp and 8,192-bp margins of a replicate share one draw.
    Percentile interval, B = 2,000. The species mean carries the paper's t interval on 8 df.
    """
    global _H2H
    if _H2H is not None:
        return _H2H
    import pandas as pd
    d = _atlas()
    co = d[np.isfinite(d.e1) & np.isfinite(d.e8) & np.isfinite(d.g)]
    rows = {}
    for sp in S.LEAF_ORDER:
        g = co[co.species == sp]
        y = g.label.to_numpy()
        e1, e8, gg = g.e1.to_numpy(), g.e8.to_numpy(), g.g.to_numpy()
        ip = np.flatnonzero(y == 1)
        ineg = np.flatnonzero(y == 0)
        codes, _ = pd.factorize(g.locus.to_numpy()[ip])          # loci in order of first appearance
        groups = [ip[codes == k] for k in range(codes.max() + 1)]
        rng = np.random.default_rng(seed)
        m = np.empty((B, 2))
        for b in range(B):
            pidx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            idx = np.concatenate([pidx, ineg[rng.integers(0, len(ineg), len(ineg))]])
            yy = y[idx]
            ag = _auroc(yy, gg[idx])
            m[b] = (_auroc(yy, e1[idx]) - ag, _auroc(yy, e8[idx]) - ag)
        ag = _auroc(y, gg)
        rows[sp] = {"m1": _auroc(y, e1) - ag, "m8": _auroc(y, e8) - ag,
                    "ci1": tuple(np.percentile(m[:, 0], [2.5, 97.5])),
                    "ci8": tuple(np.percentile(m[:, 1], [2.5, 97.5])),
                    "n": len(g), "n_pos": int(len(ip)), "pos_loci": len(groups)}
    few = {sp for sp, r in rows.items() if r["pos_loci"] < 3}
    assert few == NO_CLUSTER_INTERVAL, (few, NO_CLUSTER_INTERVAL)
    mean = {k: _tint([rows[s][k] for s in S.LEAF_ORDER]) for k in ("m1", "m8")}
    _H2H = (rows, mean, len(co))
    return _H2H


# ------------------------------------------- Figure 9c : what each species' panel is made of
# type_matched_atlas.py keeps its own copy of this set and names this one as its twin; keep them equal.
CODING_SO = {"missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost",
             "frameshift_variant", "initiator_codon_variant", "splice_donor_variant",
             "splice_acceptor_variant"}


def _pct(v):
    """A share printed as the plates print shares: whole percent from 10 %, one decimal below it.

    A non-zero share under 0.05 % would print as "0.0%", a false statement about a bar that is
    drawn because it is NOT empty; it prints as "<0.1%" instead.
    """
    p = 100.0 * v
    if 0.0 < p < 0.05:
        return "<0.1%"
    return ("%.0f%%" if p >= 10 else "%.1f%%") % p


def panel_composition(ax):
    """The coding share of each species' positives and of its negatives, side by side.

    WHAT IT SHOWS AND WHY IT IS DRAWN THIS WAY. The atlas contrasts catalogued positives -- 85 to 100 %
    coding in every species -- with negatives that are almost entirely non-coding everywhere except
    human, whose ClinVar benign negatives are mostly coding. That contrast is what panel a's
    one-bit coding flag scores: AUROC(coding) = (1 + share among positives - share among negatives)
    / 2, which atlas_skill() asserts, so the gap between each pair of bars here IS the black square
    in panel a, read in another unit.

    This panel used to be an area-true mosaic, a column per species with width proportional to its
    panel size and the coding share as a solid band inside it. Width made the three smallest panels
    -- goat (99 variants), chicken (308) and pig (396) -- 0.9 to 3.6 % of the axis, too narrow to
    carry a label, so the plate stated the headline contrast for six species and left the reader to
    guess it for three. Panel size is printed in panel b ("positives/negatives" per row) and belongs
    there; here every species gets the same width and every share is printed.

    Colour is the CLASS, in the pair panel b uses (positives ember, negatives slate). The two measure
    1.15:1 in greyscale as opaque fills, so the negatives are drawn at 0.55 alpha, 2.4:1 apart
    composited over white, and the class is also carried by
    position (positives always the left bar) and by the key.

    The coding definition is the coding flag's own (atlas_skill): the first term of the consequence
    in CODING. The rule this panel used before also counted a protein-altering term in any later
    position. On this table the two agree variant for variant, and that is asserted, so the panel
    cannot drift from the flag it decomposes.
    """
    d = _atlas()
    parts = d.consequence.astype("string").str.split("&")
    _PROTEIN_ALTERING = {"frameshift_variant", "stop_gained", "stop_lost", "start_lost",
                         "missense_variant"}
    old_rule = (parts.str[0].isin(CODING_SO).fillna(False)
                | parts.apply(lambda ps: isinstance(ps, list) and any(p in _PROTEIN_ALTERING
                                                                      for p in ps)))
    assert np.array_equal(old_rule.to_numpy(bool), d.coding.to_numpy() == 1.0), \
        "the two coding definitions disagree on the atlas; panel c no longer decomposes panel a"
    skill, _ = atlas_skill()
    order = S.LEAF_ORDER
    W, DX = 0.34, 0.19                      # bar width and half-separation, in species units
    NEG_ALPHA = 0.55
    YTOP = 100.0
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylim(0, YTOP)
    for i, sp in enumerate(order):
        r = skill[sp]
        for x, v, c, a in ((i - DX, r["pos_coding"], PATHO, 1.0),
                           (i + DX, r["neg_coding"], BENIGN, NEG_ALPHA)):
            ax.bar(x, 100.0 * v, width=W, color=c, alpha=a, edgecolor="none", zorder=3)
            # 1.5 pt above the bar top, in points, so the gap survives any change of panel height.
            ax.annotate(_pct(v), xy=(x, 100.0 * v), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=ANNOT, color=SG.INK, zorder=5,
                        annotation_clip=False)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=SG.TICK)
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("coding (%)", fontsize=SG.AXIS)
    SG.spines(ax)
    # The key names the two classes in the words panel b uses. A real legend with proxy handles, so
    # it cannot drift from the bars it names. It sits 12 pt above the axes: the printed shares of
    # the tallest bars (98 %, 96 %, 100 %) stand up to 10 pt above the top of the axes, so a key
    # seated on the axes' top edge collided with them wherever it was put along the row.
    from matplotlib.patches import Patch
    lg = ax.legend(handles=[Patch(facecolor=PATHO, edgecolor="none", label="positives"),
                            Patch(facecolor=BENIGN, alpha=NEG_ALPHA, edgecolor="none",
                                  label="negatives")],
                   loc="lower left", bbox_to_anchor=(0.0, 1.0),
                   bbox_transform=mtransforms.offset_copy(ax.transAxes, fig=ax.figure, y=12.0,
                                                          units="points"),
                   ncol=2, frameon=False,
                   fontsize=ANNOT, handlelength=1.0, handleheight=0.8, handletextpad=0.4,
                   columnspacing=1.2, borderaxespad=0.0, borderpad=0.0)
    for t in lg.get_texts():
        t.set_color(SG.INK)
    return skill


# ------------------------------------------- Figure 10c : Evo 2 against GERP, species by species
def panel_headtohead(ax):
    """The head-to-head the section heading claims, one species per row, both readouts.

    Each row carries the Evo 2 minus GERP AUROC margin at 1,001 bp (open) and at 8,192 bp (filled),
    measured on the variants all three scores cover, with its two-sample locus-bootstrap 95%
    interval (headtohead()). GERP is the same score at both readouts, so the horizontal step from
    open to filled within a row is Evo 2's own readout gain with the variant set held fixed.
    The bottom row is the species mean with its t interval on 8 df.

    Rows follow the species tree, as in Figure 9, not the size of the margin: an order sorted by the
    8,192-bp margin would be a different order at 1,001 bp, and the reader of the pair of figures
    would have to re-find every species.

    Goat is drawn without an interval (NO_CLUSTER_INTERVAL): on these variants its eight positives
    sit in one 100-kb locus, the locus bootstrap cannot resample them, and the interval it returns
    reflects the negatives alone.

    The two species-mean values are printed beside their own whiskers. That row gets a wider
    sub-row spacing than the species rows so the two lines of type, set in points, clear each other.
    """
    rows, mean, n_co = headtohead()
    order = S.LEAF_ORDER
    YM = len(order) + 0.75                 # the mean row, set off from goat by a rule
    OFF = {"m1": -0.17, "m8": +0.17}       # 1,001 bp above, 8,192 bp below, as in the key
    # Inverted: first species at the top. The bottom limit leaves the mean row's lower line of type
    # (4.6 pt below the row, half a line tall) clear of the axis line under it.
    ax.set_ylim(YM + 0.95, -0.55)
    lo = min(min(r["ci1"][0], r["ci8"][0]) for sp, r in rows.items() if sp not in NO_CLUSTER_INTERVAL)
    hi = max(max(r["ci1"][1], r["ci8"][1]) for sp, r in rows.items() if sp not in NO_CLUSTER_INTERVAL)
    lo = min(lo, mean["m1"][1], mean["m8"][1])
    hi = max(hi, mean["m1"][2], mean["m8"][2])
    # The data's own range plus 0.03 either side, labelled every 0.1: ticks at every 0.1 over a
    # range padded out to whole tenths set eight labels in 39 mm and ran them into one another.
    ax.set_xlim(lo - 0.03, hi + 0.03)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "0" if abs(x) < 1e-9 else "%.1f" % x))
    ax.axvline(0, color=SG.RULE, lw=.6, ls=(0, (2.5, 2)), zorder=1)
    cap = _dy(ax, 1.6)                     # whisker cap half-height, in points
    ms = {"m1": dict(marker="o", ms=4.4, mfc="white", mec=SG.INK, mew=.8),
          "m8": dict(marker="o", ms=4.4, mfc=SG.INK, mec=SG.INK, mew=.6)}

    def _row(y, v, ci, key):
        if ci is not None:
            ax.plot([ci[0], ci[1]], [y, y], color=SG.INK, lw=.8, zorder=3, solid_capstyle="butt")
            for xx in ci:
                ax.plot([xx, xx], [y - cap, y + cap], color=SG.INK, lw=.8, zorder=3)
        ax.plot([v], [y], linestyle="none", zorder=5, **ms[key])

    for i, sp in enumerate(order):
        r = rows[sp]
        weak = sp in NO_CLUSTER_INTERVAL
        _row(i + OFF["m1"], r["m1"], None if weak else r["ci1"], "m1")
        _row(i + OFF["m8"], r["m8"], None if weak else r["ci8"], "m8")
    # the species mean: its sub-rows 9 pt apart, so the two printed values clear each other
    moff = _dy(ax, 4.6)
    for key, dy in (("m1", -moff), ("m8", +moff)):
        m, l_, h_ = mean[key]
        _row(YM + dy, m, (l_, h_), key)
        ax.annotate("%+.3f [%+.3f, %+.3f]" % (m, l_, h_), xy=(h_, YM + dy), xycoords="data",
                    xytext=(4, 0), textcoords="offset points", fontsize=ANNOT, color=SG.INK,
                    ha="left", va="center", annotation_clip=False)
    ax.axhline(len(order) - 0.5 + 0.24, color=SG.RULE, lw=.5, zorder=1)
    ax.set_yticks(list(range(len(order))) + [YM])
    ax.set_yticklabels([sp + (DAGGER if sp in NO_CLUSTER_INTERVAL else "") for sp in order]
                       + ["species mean"], fontsize=ANNOT)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_xlabel("Evo 2 − GERP (AUROC)", fontsize=SG.AXIS)
    SG.spines(ax, keep=("bottom",))
    # How many positive-bearing 100-kb loci each species' interval resamples: the clusters of the
    # locus bootstrap. A percentile interval over a few tens of clusters is approximate, and the
    # reader can see which species that is rather than being told. Right of every interval, in points.
    _lx = 1.0 + 3.0 / (ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0)
    ax.text(_lx, -0.62, "loci", transform=ax.get_yaxis_transform(), fontsize=ANNOT, color=SG.MUTED,
            ha="left", va="bottom", clip_on=False)
    for i, sp in enumerate(order):
        ax.text(_lx, i, "{:,}".format(rows[sp]["pos_loci"]), transform=ax.get_yaxis_transform(),
                fontsize=ANNOT, color=SG.MUTED, ha="left", va="center")
    from matplotlib.lines import Line2D
    # The readouts' names, as panel b's axis title defines them against their windows.
    lg = ax.legend(handles=[Line2D([0], [0], linestyle="none", label="single token", **ms["m1"]),
                            Line2D([0], [0], linestyle="none", label="window mean", **ms["m8"])],
                   # Starting over the row-name column, not at the axes' edge, so that the key ends
                   # short of the "loci" head over the right-hand column.
                   loc="lower left", bbox_to_anchor=(-0.30, 1.0), ncol=2, frameon=False,
                   fontsize=ANNOT, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2,
                   borderpad=0.0)
    for t in lg.get_texts():
        t.set_color(SG.INK)
    return rows, mean, n_co
