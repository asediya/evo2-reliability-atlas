# Module-level import of the house style, so the scale below is DERIVED, not copied.
# Same dual form src/ccs/fig3_rebuild.py uses: package import when run as -m, flat as a file.
try:
    from . import style_gb as _SG
except ImportError:
    import style_gb as _SG

"""FIGURE 5 — "Reach is not competence."

Composes the punched-plate sieve hero with four supports, each in a DIFFERENT idiom and each checked
against the skeleton ban list derived from Figs 1-4 (memory: figure-skeleton-ban-list, `wf_25bf26fa-27b`).

  a  SIEVE            ten variant-resolution plates, holes = no-calls   [area partition + lattice]
  b  SUBSIDENCE       two skylines over a no-call COUNT axis            [step/staircase family]
  c  GNOMON           one slope triangle used as an instrument          [angular, not a series]
  d  CONDITION MATRIX 8 species x 3 conditions, both axes categorical   [ban-9 carve-out]
  e  LEDGER           typographic set-membership, mostly struck out     [document, not a plot]

BAN CHECKS PERFORMED BEFORE BUILDING, NOT AFTER
  * The originally specified panel d -- a three-rail strand bundle, eight series across three
    categorical stations -- is BAN 7 (few-station slopegraph/trajectory: Fig2 H, Fig3 b, Fig3 d).
    Ban 1 establishes that changing the mark does not rescue a skeleton, so eight series instead of
    four is the same gestalt. Replaced with a categorical x categorical value matrix, which ban 9
    explicitly leaves free: "a raster whose BOTH axes are categorical, and only if it carries no
    marginal profile track". No marginal track is drawn.
  * No panel gets a right-margin per-entity value column (ban 2), no panel is sorted by a derived
    scalar (ban 3), no dot-with-CI-whisker appears anywhere (ban 4) -- uncertainty is carried as a
    species-resampled fan (c), as grain (a), and as printed ranges (a, e).
  * No beige rounded prose card (ban 12): the caveats live in the panel captions.

WHAT THE FIGURE DOES AND DOES NOT CLAIM
  * The must-call cost is taken under fig5_stats' closed-form 0.5-pair abstention
    null, never median imputation.
  * Runs-test z values are the per-species values in the JSON, never the POOLED-sequence values the
    stats layer explicitly disowns.
  * THE CODING CONFOUND: a coding-stratum "survives in 6 of 7" tally is only a SIGN COUNT.
    Matched properly on consequence category, the class gap is not separable from zero in ANY of the four
    species where matching is possible, and human -- the only well-powered match -- is a clean null. The
    figure does not claim the confound is defeated. It claims what panel b shows: whatever
    causes the holes, they cost you AUROC at deployment.

INPUTS (all produced by src/ccs/fig5_stats.py unless noted)
  reports/fig5_reach.json                          <- fig5_stats.main()
  data/processed/conservation/{build}_gerp.parquet <- deposited GERP tracks
  reports/cnn_baseline.parquet                     <- src/ccs/cnn_baseline.py
  reports/fig2_data.json                           <- Fig 2, reused for verdict consistency only

Run: python -m src.ccs.fig5_final
"""
import os, json, textwrap
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

try:
    from . import fig5_stats as S
except ImportError:
    import fig5_stats as S

ROOT = Path(__file__).resolve().parents[2]

MM = 1 / 25.4
# Page limits: 170 mm full page width, 225 mm MAX HEIGHT INCLUDING THE LEGEND. A larger figure is
# downscaled in production, pulling every small label below legibility with no author review, so
# any height saved comes out of panel a's label bands, never out of type size.
W_MM, H_MM = 170.0, 173.0
# DERIVED from style_gb's scale, not copied, so a change to the house scale reaches this figure.
# TITLE stays literal: style_gb has no TITLE rung because titles live in the manuscript legend.
TITLE = 9.5
PANEL, AXIS, TICK, FOOT = _SG.PANEL, _SG.AXIS, _SG.TICK, _SG.ANNOT

CODING = "#0072B2"
FAIL = "#A8201A"
BEDROCK = "#3A3A3A"
from .style_gb import text_safe as SG_text_safe, label_row as SG_label_row

SUB = "#8C929B"
# SUB names its own series in type as well as drawing it; on white that grey is 3.13:1, below
# WCAG 4.5 at 6.8 pt. The bars keep SUB; the words take the darkened rendition.
SUB_T = SG_text_safe(SUB)
POS = "#E69F00"
ATLAS_PV = str(Path(__file__).resolve().parents[4] / "final10" / "tables"
                / "fig1_atlas_pervariant.parquet")
try:                                   # `S` is already this module's own namespace, so alias it
    from . import fig2_style as CLADE
except ImportError:
    import fig2_style as CLADE
PATHO_C = "#C1443E"
INK = "#1A1A1A"
MUTED = _SG.MUTED
SOCKET = "#E8E4DF"

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

BUILD = {"goat": "goat", "chicken": "chicken", "pig": "pig", "sheep": "sheep", "horse": "horse",
         "cat": "cat", "cattle": "cattle_ensvar", "dog": "dog_cf3", "human": "human"}
# amniote tree topology -- NOT n, NOT class gap (ban 3)
TREE = ["chicken", "human", "dog", "cat", "horse", "pig", "cattle", "sheep", "goat"]

J = json.load(open(ROOT / "reports" / "fig5_reach.json", encoding="utf-8"))
R = {r["species"]: r for r in J["per_species"]}


def _sig_gap(r):
    """A plate is flagged only when the class gap's CI EXCLUDES zero.

    The previous rule was `class_gap > 0.02`, a bare threshold on a point estimate. It painted goat
    (+0.089 on 9 positives, CI [-0.24,+0.22]) in the same alarm red as horse (+0.489, CI [+0.40,+0.54]),
    so a reader counted six red species and reported the phenomenon as near-universal. It is four.
    """
    ci = r.get("class_gap_ci")
    return bool(ci and ci[0] is not None and np.isfinite(ci[0]) and ci[0] > 0)


def _fit(ax, txt, width=1.0, **kw):
    fig = ax.figure; r = fig.canvas.get_renderer()
    avail = ax.get_window_extent(r).width * width
    t = ax.text(0.0, 0.0, txt, transform=ax.transAxes, **kw)
    fig.canvas.draw(); w = t.get_window_extent(r).width; t.remove()
    if w <= avail:
        return txt
    return textwrap.fill(" ".join(txt.split()), width=max(24, int(len(txt) * avail / w * 0.97)))


def _cap(ax, y, txt, gap_pt=2.2, width=1.0, **kw):
    fig = ax.figure
    t = ax.text(0.0, y, _fit(ax, txt, width, **kw), transform=ax.transAxes, ha="left", va="top", **kw)
    fig.canvas.draw()
    h = t.get_window_extent(fig.canvas.get_renderer()).height
    H = ax.get_window_extent(fig.canvas.get_renderer()).height
    return y - h / H - (gap_pt / 72.0 * fig.dpi) / H


def _spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color("#8C8C8C")


from .style_gb import letter_x as SG_letter_x


def _plab(ax, letter, text=None, left=False):
    """House style (src/ccs/style_gb.py): a bold lower-case letter, no chip, no heading.

    The reversed-out letter in a filled square was deck styling, and the chip carried a
    second defect: bbox chips are invisible to get_window_extent, so the collision checker
    scored zero overlaps while the chip painted over panel c's rotated y-axis label
    (QA blind spot #4). A plain letter is measurable by the same checker that guards the
    rest of the figure. `text` is accepted and ignored -- the call sites keep their strings
    as documentation of what each panel is, and the legend states it for the reader."""
    if left:
        from .style_gb import letter_x, letter_tf
        ax.text(letter_x(ax.figure), 1.085, letter, transform=letter_tf(ax), fontsize=PANEL,
                fontweight="bold", va="bottom", ha="left", color=INK, zorder=8)
    else:
        ax.text(-0.115, 1.085, letter, transform=ax.transAxes, fontsize=PANEL, fontweight="bold",
                va="bottom", ha="center", color=INK, zorder=8)
    ax.plot([-0.040, 1.0], [1.062, 1.062], transform=ax.transAxes, color="#C4BFB8", lw=0.6,
            clip_on=False, zorder=5)


# ---------------------------------------------------------------- a. the sieve
def plate_cells(sp):
    """1 = material, 0 = hole, ordered by GENOMIC POSITION.

    Uses fig5_stats.order_by_position, which parses the id FROM THE RIGHT. The left-anchored regex this
    replaced could not cross the underscore inside a RefSeq accession (NC_058368.1_...), so it returned
    None and assigned one constant key to every unparsed variant -- 100% of cat's and horse's pathogenic
    cells and ~89% of dog's were then positioned by numpy's internal sort state. A literal dither, on
    exactly the half the thesis rests on.
    """
    d = pl.read_parquet(ROOT / "data" / "processed" / "conservation" / f"{BUILD[sp]}_gerp.parquet")
    vid = np.array(d["variant_id"].to_list()); g = d["gerp"].to_numpy().astype(float)
    isneg = np.array([v.startswith("neg_") for v in vid])
    return [np.isfinite(g[m])[S.order_by_position(vid[m])].astype(float) for m in (isneg, ~isneg)]


def plate_grid(ax, n_max, w, ylim_span):
    """One cell size for BOTH halves of a plate, derived from the rendered box.

    THE GRAIN CONFOUND THIS FIXES. The halves used to be EQUAL AREA while holding ~10:1 different
    counts, so a cell on the pathogenic side was ~10x larger than one on the benign side. Identical
    missingness then rendered as ~10x fewer, chunkier holes on the right, and the reader compared two
    textures that differ for a reason that is not the data. The proof is sheep: its class gap is exactly
    +0.000 and both halves are 73% reachable, yet the old plate drew fine speckle on the left against
    coarse blocks on the right, so the one species that should have looked self-identical looked split.

    Now the cell is square and the SAME SIZE everywhere in a plate, so a half's HEIGHT is proportional to
    its variant count and hole texture is directly comparable across the divider. Sheep must now look
    self-similar; that is the acceptance test.

    Returns (ncol, row_h) in data units, sized so the LARGER half exactly fills the plate's 0..1 band.
    """
    bb = ax.get_window_extent(ax.figure.canvas.get_renderer())
    xspan = 2 * 0.47 + 0.06 + 0.02                                    # matches set_xlim in panel_a
    half_w_px = bb.width * w / xspan
    plate_h_px = bb.height / ylim_span                                # the 0..1 band only
    ncol = max(1, int(round(np.sqrt(half_w_px * n_max / plate_h_px))))
    nrow_max = int(np.ceil(n_max / ncol))
    return ncol, 1.0 / nrow_max


def draw_half(ax, v, x0, w, colour, ncol, row_h):
    """Draw one half BOTTOM-ALIGNED at the shared cell size. Returns the height used, in data units.

    NO PADDING. The grid used to be padded up to nrow*ncol -- first with NaN + set_bad(alpha=0), which
    composited to page white inside the frame and read as punched holes, then with material, which
    invented reachable variants. Neither is acceptable once ncol is shared between the halves, because
    the smaller half can then be up to ncol-1 cells short: goat's 9 positives would take 3
    invented material cells and print 8/12 = 67% against a true 89%. The remainder row is drawn as its
    own short strip spanning only the columns it actually fills, so every drawn cell is exactly one
    variant and no cell is invented in either direction.
    """
    n = len(v)
    full, rem = divmod(n, ncol)
    cmap = ListedColormap([SOCKET, colour])
    h = (full + (1 if rem else 0)) * row_h
    if full:
        ax.imshow(v[:full * ncol].reshape(full, ncol), extent=[x0, x0 + w, h - full * row_h, h],
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto", zorder=3)
    if rem:
        ax.imshow(v[full * ncol:].reshape(1, rem), extent=[x0, x0 + w * rem / ncol, 0, row_h],
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto", zorder=3)
    # frame hugs what is actually drawn, so unfilled plate area is never enclosed
    if full:
        ax.add_patch(plt.Rectangle((x0, h - full * row_h), w, full * row_h, fill=False, ec="#B8B3AC",
                                   lw=0.4, zorder=5))
    if rem:
        ax.add_patch(plt.Rectangle((x0, 0), w * rem / ncol, row_h, fill=False, ec="#B8B3AC",
                                   lw=0.4, zorder=5))
    return h


def panel_a(fig, gs_slice):
    """Ten plates. ylim was (-0.33, 1.17) -- 30% of every plate axes reserved below the plate for two
    thin text rows, which printed as 38.4 mm of full-width blank paper across the panel while b-e were
    squeezed to 18.5 mm of plot height each. Tightened to (-0.20, 1.17); the reclaimed height is what
    brings the figure under the 225 mm ceiling without shrinking any type."""
    inner = gs_slice.subgridspec(2, 5, wspace=0.115, hspace=0.145)
    HW, GAP = 0.47, 0.06
    # THE LABEL BANDS ARE SIZED IN POINTS, NOT IN PLATE FRACTIONS. A band holds two lines of
    # 6.8-7.0 pt type, a constant number of points at any plate height; a fixed fraction of the
    # plate band would grow with the plate and hand its extra height to blank paper. Sizing the
    # bands in points turns plate height into cells.
    TOP_PT, BOT_PT = 18.5, 17.5          # label+CI above the plate; reach% + class counts below
    _probe = fig.add_subplot(inner[0, 0])
    _h_pt = _probe.get_window_extent(fig.canvas.get_renderer()).height * 72.0 / fig.dpi
    _probe.remove()
    B = max(24.0, _h_pt - TOP_PT - BOT_PT)          # the 0..1 plate band, in points
    YLO, YHI = -BOT_PT / B, 1.0 + TOP_PT / B
    Y_LAB, Y_CI = 1.0 + (TOP_PT - 0.7) / B, 1.0 + 9.2 / B
    Y_PCT, Y_CELLS = -1.5 / B, -(BOT_PT - 1.0) / B
    span = YHI - YLO
    tot = sum(R[s]["n"] for s in TREE)
    axes0 = None
    for k, sp in enumerate(TREE + ["Evo2"]):
        ax = fig.add_subplot(inner[k // 5, k % 5])
        if k == 0:
            axes0 = ax
        if sp == "Evo2":
            # The real 11,130 at the true 8,890 | 2,240 class split, all material. This used to be
            # np.ones(600) per half -- a 1,200-cell placeholder at a 1:1 split that no panel has,
            # captioned "11,130 cells". Drawing the real counts makes the label true and lets the
            # reference plate's class-size ratio be read like every other plate's.
            nb = sum(R[s]["n_neg"] for s in TREE); npo = sum(R[s]["n_pos"] for s in TREE)
            ncol, row_h = plate_grid(ax, max(nb, npo), HW, span)
            draw_half(ax, np.ones(nb), 0.0, HW, CODING, ncol, row_h)
            draw_half(ax, np.ones(npo), HW + GAP, HW, CODING, ncol, row_h)
            lab, gap_s, col = "Evo 2", "Gap +0.000", CODING
            l_pct, r_pct = "100%", "100%"
            ci_s = "Nothing punched out"
            cells = f"{nb + npo:,} cells"
        else:
            r = R[sp]; ben, pat = plate_cells(sp)
            ncol, row_h = plate_grid(ax, max(len(ben), len(pat)), HW, span)
            draw_half(ax, ben, 0.0, HW, SUB, ncol, row_h)
            draw_half(ax, pat, HW + GAP, HW, FAIL, ncol, row_h)
            lab = sp[:1].upper() + sp[1:]; gap_s = f"Gap {r['class_gap']:+.3f}"
            col = FAIL if _sig_gap(r) else MUTED
            lo_, hi_ = r["class_gap_ci"]
            ci_s = f"[{lo_:+.3f}, {hi_:+.3f}]"
            l_pct, r_pct = f"{r['reach_neg']:.0%}", f"{r['reach_pos']:.0%}"
            cells = f"{r['n_neg']:,} | {r['n_pos']:,}"
        ax.plot([HW + GAP / 2] * 2, [0, 1], color=BEDROCK, lw=0.5, zorder=4)
        ax.set_xlim(-0.01, 2 * HW + GAP + 0.01); ax.set_ylim(YLO, YHI); ax.axis("off")
        # BOLD MEANS ONE THING: "this species' gap CI excludes zero", which the key states. Evo 2
        # is already set in CODING blue, and its gap note is blue too, so its name takes no weight.
        # Bold is a redundant channel beside the red gap, and survives a
        # photocopy, the same reasoning as the hatch in Figure 9's panel a.
        ax.text(0.0, Y_LAB, lab, fontsize=TICK, color=(CODING if sp == "Evo2" else INK),
                ha="left", va="top",
                fontweight=("bold" if _sig_gap(R.get(sp, {})) else "normal"))
        ax.text(2 * HW + GAP, Y_LAB, gap_s, fontsize=FOOT, color=col, ha="right", va="top",
                fontweight="bold")
        # The Wilson/Newcombe range is PRINTED, not whiskered (ban 4). It was computed and assigned to a
        # dead variable in the previous version while the docstring claimed panel a carried "printed
        # ranges" -- goat's 89% came from 8 of 9 variants with no interval anywhere on the plate.
        # No italic on the Evo 2 row: it is already set in CODING blue here and in its species
        # name and gap, so the slant was a fourth channel for a fact already carried three times.
        # Italic is reserved in this set for a taxon name (Figure 6's clade), not for emphasis.
        ax.text(2 * HW + GAP, Y_CI, ci_s, fontsize=FOOT,
                color=(CODING if sp == "Evo2" else MUTED), ha="right", va="top")
        ax.text(HW / 2, Y_PCT, l_pct, fontsize=FOOT, color=(CODING if sp == "Evo2" else SUB_T),
                ha="center", va="top", fontweight="bold")
        ax.text(HW + GAP + HW / 2, Y_PCT, r_pct, fontsize=FOOT,
                color=(CODING if sp == "Evo2" else FAIL), ha="center", va="top", fontweight="bold")
        ax.text(HW + GAP / 2, Y_CELLS, cells, fontsize=FOOT, color=MUTED, ha="center", va="bottom")
    return axes0


# ---------------------------------------------------------------- b. the set GERP declines
def panel_b(ax):
    """What the conservation track's silence actually IS, drawn as a set rather than a staircase.

    The old panel was a nine-step cumulative staircase of no-calls against AUROC: nine steps for
    11,130 variants, and it could not say WHICH variants were missing or whether missing them
    mattered. Here the whole atlas is one area-true block, split by species (column width is the
    species' share) and by whether GERP returns a value (row height is the share of that species
    it scores). Cell fill is the share of the cell that is a catalogued disease allele.

    Three facts the staircase could not carry come out of the geometry. GERP is silent on 1,598 of
    the 11,130. That silence is not evenly spread -- horse, cat and pig lose 402, 328 and 228 -- and
    it is not random with respect to the outcome: the silent set is 7.0 % catalogued alleles against
    22.3 % among the variants it does score. And Evo 2 still separates the silent set, which is why
    the must-answer correction is a correction to the COMPARISON rather than a statement that the
    variants are unreadable.
    """
    import pandas as pd
    from scipy.stats import rankdata
    try:
        from .fig2_measure import _deposit_table
    except ImportError:
        from fig2_measure import _deposit_table
    d = pd.read_parquet(_deposit_table("fig1_atlas_pervariant.parquet"))
    d["silent"] = d.gerp.isna()
    order = [sp for sp in CLADE.LEAF_ORDER if sp in set(d.species)]
    tot = len(d)
    x = 0.0
    seats = []
    silent_n = []
    for sp in order:
        g = d[d.species == sp]
        w = len(g) / tot
        h_sil = float(g.silent.mean())
        for sil, y0, h in ((True, 1 - h_sil, h_sil), (False, 0.0, 1 - h_sil)):
            gg = g[g.silent == sil]
            if not len(gg):
                continue
            f = float((gg.label == 1).mean())
            ax.add_patch(plt.Rectangle((x, y0), w, h, facecolor=PATHO_C, alpha=max(f, 0.03),
                                       edgecolor="white", lw=.4, zorder=3))
        # EVERY column's silent count, in one row over the block (placed below, once the limits are
        # set). Printed inside the cells, a count needs a column wider than its digits, so chicken,
        # pig and goat -- the three narrowest -- carried none.
        silent_n.append((x + w / 2, int(g.silent.sum())))
        seats.append((x + w / 2, sp, w))
        x += w
    ax.axhline(1 - float(d.silent.mean()), color=INK, lw=.7, ls=(0, (3, 2)), zorder=7)
    YTOP = 1.70            # headroom over the block for the count row and the header line
    ax.set_xlim(0, 1); ax.set_ylim(-0.66, YTOP)
    # ONE BASELINE. Staggering the nine names into two ranks by INDEX PARITY (i % 2) is not a
    # property of the data: a wide column drops to the second rank purely
    # because of where it falls in the list, and the reader's eye zig-zags to read the row in
    # order. Nine horizontal names need ~169 pt and this axis is ~159 pt, so they genuinely do not
    # fit flat -- but rotated 90 deg each name is ~7 pt wide, the row needs ~85 pt, and one
    # baseline fits easily. This is also the treatment Figure 9c and Figure 10e already use for
    # species names, so the set gains a convention instead of a third variant.
    # xlim/ylim are set FIRST: label_row measures in the axes' own units.
    _span = YTOP - (-0.66)
    _axh = ax.get_position().height * ax.figure.get_figheight() * 72.0
    _dyp = _span / _axh                      # data units per point on this axis
    SG_label_row(ax, [(xc, sp[:1].upper() + sp[1:], INK) for xc, sp, _w in seats], -0.01 - 6.0 * _dyp, FOOT,
                 y_from=-0.01, rotation=90, ha="right", va="center", gap_pt=2.0,
                 lead_color="#B4B4B4")
    # The count row: one baseline 7 pt over the block, each count on a leader from its own column's
    # top edge, seated by the same solver as the names under the block.
    SG_label_row(ax, [(xc, "{:,}".format(n), INK) for xc, n in silent_n], 1.0 + 7.0 * _dyp, FOOT,
                 y_from=1.0, rotation=0, ha="center", va="bottom", gap_pt=2.5, lead_color="#B4B4B4")
    # No head for the count row: the "GERP silent" row label right under it names what is counted,
    # and a head set beside it met that label's first line.
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.text(-0.008, 1 - float(d.silent.mean()) / 2, "GERP\nsilent", fontsize=FOOT, ha="right",
            va="center", color=INK, linespacing=1.12)
    ax.text(-0.008, (1 - float(d.silent.mean())) / 2, "GERP\nscores it", fontsize=FOOT,
            ha="right", va="center", color=MUTED, linespacing=1.12)

    sil = d[d.silent]
    sco = d[~d.silent]
    v = sil.evo2_1001.to_numpy()
    y = sil.label.to_numpy()
    ok = np.isfinite(v)
    r = rankdata(v[ok]); n1 = int((y[ok] == 1).sum()); n0 = int((y[ok] == 0).sum())
    a_sil = (r[y[ok] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    hs = float(d.silent.mean())
    ax.text(1.008, 1 - hs / 2, "%.1f%%\ncatalogued" % (100 * (sil.label == 1).mean()),
            fontsize=FOOT, ha="left", va="center", color=PATHO_C, linespacing=1.3, clip_on=False)
    ax.text(1.008, (1 - hs) / 2, "%.1f%%\ncatalogued" % (100 * (sco.label == 1).mean()),
            fontsize=FOOT, ha="left", va="center", color=MUTED, linespacing=1.3, clip_on=False)
    ax.text(0.0, YTOP - 0.02, "%s unscored; Evo 2 reads them at %.3f"
            % ("{:,}".format(int(d.silent.sum())), a_sil),
            fontsize=FOOT, ha="left", va="top", color=INK)


# ---------------------------------------------------------------- c. the gnomon
def panel_c(ax):
    """NOT CALLED: this panel is not on the plate. Kept only so the
    numbers it carries can be traced; main() does not reference it.

    One slope triangle used as a measuring instrument: horizontal null leg, vertical rise leg,
    hypotenuse tilt IS the estimand, thickened into a fan by the SPECIES-RESAMPLED 95% interval.
    Deliberately not a series-across-stations (ban 7) and not a dot+CI (ban 4).

    The fan used to be the leave-one-species-out range (+0.101 to +0.139). A jackknife spread is not a
    confidence interval: it is ~4x narrower than the species bootstrap, while a pale fan tapering around
    a heavy line is the universal grammar of a confidence band. A reader took away a precision the nine
    non-independent species do not support. The honest band is what fig5_stats already computed.
    """
    G = J["regime"]
    c = G["contrast"]
    lo, hi = G["contrast_ci"]
    lo_d = max(lo, 0.0)                              # the wedge is drawn in the positive quadrant
    ax.add_patch(plt.Polygon([(0, 0), (1, 0), (1, hi)], closed=True, facecolor=FAIL, alpha=0.16,
                             edgecolor="none", zorder=2))
    ax.add_patch(plt.Polygon([(0, 0), (1, 0), (1, lo_d)], closed=True, facecolor="white",
                             edgecolor="none", zorder=3))
    ax.add_patch(plt.Polygon([(0, 0), (1, 0), (1, c)], closed=True, facecolor="none",
                             edgecolor=FAIL, lw=1.4, zorder=5))
    ax.plot([0, 1], [0, 0], color=BEDROCK, lw=1.0, zorder=6)
    ax.plot([1, 1], [0, c], color=FAIL, lw=1.0, zorder=6)
    ax.text(0.5, -0.006, "no regime dependence", fontsize=FOOT, color=BEDROCK, ha="center",
            va="top", style="italic")
    ax.text(1.012, c / 2, f"{c:+.3f}\nper decade", fontsize=FOOT, color=FAIL, ha="left",
            va="center", fontweight="bold", linespacing=1.45)
    ax.text(0.985, hi * 1.02, f"species-resampled 95%\n{lo:+.3f} to {hi:+.3f}", fontsize=FOOT,
            color=MUTED, ha="right", va="bottom", linespacing=1.45, style="italic")
    ax.set_xlim(-0.03, 1.30); ax.set_ylim(-0.028, hi * 1.22)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["", "One decade of panel size"])
    ax.set_yticks([0, 0.05, 0.10])
    ax.set_ylabel("conservation − Evo 2\nAUROC gained", fontsize=AXIS, linespacing=1.45, labelpad=2.5)
    _spines(ax)


# ---------------------------------------------------------------- d. condition matrix
def panel_d(ax):
    """8 species x 3 conditions, BOTH axes categorical, no marginal track -- the carve-out ban 9 leaves
    open. Replaces the specified three-rail strand bundle, which was ban 7."""
    rows = [r for r in J["supervised"]["rows"]]
    order = [s for s in TREE if s in {r["species"] for r in rows}]
    by = {r["species"]: r for r in rows}
    conds = [("CNN\ntrained here", "auroc_cnn"), ("CNN\nspecies held out", "cnn_loso"),
             ("Evo 2\nzero-shot", "auroc_evo2")]
    M = np.array([[by[s][k] for _, k in conds] for s in order])
    # SEQUENTIAL, not diverging. The old ramp ran red -> cream -> blue over vmin 0.45 to vmax 0.95,
    # which puts its neutral colour at AUROC 0.70 -- a value with no meaning. The natural centre for
    # an AUROC is chance, 0.50, and only ONE of these 24 cells is below it (human, CNN held out, at
    # 0.482). The quantity is one-sided in this panel, so a diverging form invents a midpoint and,
    # worse, is non-monotone in lightness: two different AUROCs an equal distance either side of
    # 0.70 printed at the same grey. Below-chance is still marked, and marked without colour -- a
    # FAIL-coloured outline and a bold numeral, both below.
    #
    # The stop POSITIONS are solved against the 24 values this panel actually holds, not chosen by
    # eye. Black-or-white ink on a mid-tone bottoms out at 4.16:1 (at cell luminance 0.202), so a
    # ramp that lingers there loses the 4.5:1 small-text floor; these positions carry the ramp
    # across that band where no datum sits. Worst numeral contrast is 5.92:1 against the old ramp's
    # 4.54:1, so this is monotone in lightness AND more legible.
    cmap = LinearSegmentedColormap.from_list(
        # DESATURATED 30%, to keep this table's saturated ink in line with the rest of the set.
        # The stops are pulled toward the neutral
        # grey of their OWN luminance, so everything the comment above establishes still holds:
        # measured L* is 0.943/0.590/0.296/0.133 against 0.943/0.602/0.320/0.152 undesaturated,
        # the ramp is still monotone in lightness, and the worst best-ink contrast is 5.74:1,
        # which is clear of the 4.5:1 small-text floor the adaptive
        # ink picker below depends on.
        "auroc", [(0.00, "#FBF8F5"), (0.35, "#9DD1EF"), (0.90, "#519CC7"), (1.00, "#1C6C99")])
    vmin, vmax = 0.45, 0.95
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest", zorder=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            # Ink chosen by MEASURED luminance, not by hand-picked cutoffs. The old rule
            # (white if v < 0.58 or v > 0.87) put white 5 pt numerals on light blue at 3.1-3.8:1 in 5 of
            # 24 cells -- below the 4.5:1 small-text floor, and on the Evo2 column the caption is about.
            rgb = cmap((v - vmin) / (vmax - vmin))[:3]
            lum = 0.2126 * _lin(rgb[0]) + 0.7152 * _lin(rgb[1]) + 0.0722 * _lin(rgb[2])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FOOT,
                    color=("white" if lum < 0.18 else INK),
                    fontweight=("bold" if v < 0.50 else "normal"), zorder=4)
            if v < 0.50:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=FAIL, lw=1.0,
                                           zorder=5))
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c for c, _ in conds], fontsize=FOOT, linespacing=1.45)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[:1].upper() + o[1:] for o in order], fontsize=FOOT)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _lin(c):
    """sRGB -> linear, for the WCAG relative-luminance test in panel d."""
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


# ---------------------------------------------------------------- e. the ledger
def panel_e(ax):
    """NOT CALLED: this panel is not on the plate. Kept only so the
    numbers it carries can be traced; main() does not reference it.

    Evaluability ledger. THREE facts that were previously collapsed into two:
      (1) whether a panel's consequence annotation EXISTS at all -- a species with no consequence file
          must not have its counts defaulted to 0 and drawn as the biological claim "no missense at
          all". A missing input file must never render as a result;
      (2) whether a panel CAN host a protein LM (>=10 missense of both classes);
      (3) whether ESM-2 was actually scored on it AT USABLE SIZE -- cattle has an ESM file with 15 rows
          against the 155 missense the same row advertises, so "scored" from os.path.exists overstated
          coverage tenfold and made the header's "barely overlap" false.
    """
    P = J["protein_lm"]
    cen = {c["species"]: c for c in P["census"]}
    rows = [cen[sp] for sp in TREE]
    struck = []
    ax.set_xlim(0, 1); ax.set_ylim(len(rows) + 0.6, -1.9); ax.axis("off")

    def _status(c):
        if c["evaluable"] is True:
            if c["esm_usable"]:
                return f"scored, n = {c['esm_n']:,}", (INK, "bold", "normal")
            if c["esm_file"]:
                return f"only {c['esm_n']} of {c['missense']:,} scored", (FAIL, "bold", "normal")
            return "EVALUABLE, never scored", (FAIL, "bold", "normal")
        # SHORT FORM: "only N of the rarer class" would repeat the same
        # five words on five rows and "(no census)" would restate the column header. The counts survive.
        if not c["annotated"]:
            return "not annotated", ("#B0ABA4", "normal", "italic")
        return f"rarer class n = {min(c['pos'], c['neg'])}", ("#B0ABA4", "normal", "italic")

    stat = {c["species"]: _status(c) for c in rows}

    # COLUMN GRID IS MEASURED, NOT TYPED. At 5.3 pt the hand-set x of 0.40/0.50 happened to clear;
    # at 6 pt "path" and "benign" set solid and came back from the PDF as the single span
    # "pathbenign" -- a Text-vs-Text overlap audit cannot see two labels that merged into one. The
    # five columns are now laid out from their own measured widths with equal gutters, so the grid
    # follows the type size instead of the type having to fit a frozen grid.
    fig = ax.figure; rend = fig.canvas.get_renderer()
    axw = ax.get_window_extent(rend).width

    def _w(txt, **kw):
        t = ax.text(0.0, 0.0, txt, fontsize=FOOT, **kw)
        fig.canvas.draw(); out = t.get_window_extent(rend).width / axw; t.remove()
        return out

    w_sp = max([_w("species")] + [_w(c["species"], fontweight="bold") for c in rows])
    w_num = []
    for key, head in (("missense", "missense"), ("pos", "pos"), ("neg", "neg")):
        w_num.append(max([_w(head)] + [_w("—" if c[key] is None else f"{c[key]:,}") for c in rows]))
    w_st = max([_w("status")] + [_w(t, fontweight=fw, style=st) for t, (_, fw, st) in stat.values()])
    gut = max(0.012, (1.0 - (w_sp + sum(w_num) + w_st)) / 4.0)
    X_SP = 0.0
    X_MI = w_sp + gut + w_num[0]
    X_PA = X_MI + gut + w_num[1]
    X_BE = X_PA + gut + w_num[2]
    X_ST = X_BE + gut

    for x, lab, ha in ((X_SP, "species", "left"), (X_MI, "missense", "right"),
                       (X_PA, "pos", "right"), (X_BE, "neg", "right"), (X_ST, "status", "left")):
        ax.text(x, -1.55, lab, fontsize=FOOT, color=MUTED, ha=ha, va="bottom")
    ax.plot([0, 1], [-1.25, -1.25], color="#C4BFB8", lw=0.5)
    for i, c in enumerate(rows):
        live = c["evaluable"] is True
        unann = not c["annotated"]
        col = INK if live else "#B0ABA4"
        t_sp = ax.text(X_SP, i, c["species"], fontsize=FOOT, color=col, ha="left", va="center",
                       fontweight=("bold" if live else "normal"))
        if not live:
            struck.append((t_sp, i))
        for x, key in ((X_MI, "missense"), (X_PA, "pos"), (X_BE, "neg")):
            v = c[key]
            ax.text(x, i, ("—" if v is None else f"{v:,}"), fontsize=FOOT,
                    color=("#C6C1BA" if unann else col), ha="right", va="center")
        txt, (tc, fw, st) = stat[c["species"]]
        ax.text(X_ST, i, txt, fontsize=FOOT, color=tc, ha="left", va="center",
                fontweight=fw, style=st)
    # Strike the NAME ONLY, sized to the MEASURED glyph box. A full-width rule used to run through the
    # numerals of every excluded row -- a 0.55 pt line across 5.3 pt digits, so "20 17 3" and "56 48 8"
    # were struck out in the literal sense. It was invisible to the overlap audit because that compared
    # Text to Text and Text to bbox patch only; a Line2D was never in the comparison set at all.
    # Measuring rather than guessing a fixed width keeps "cat" and "pig" from trailing a leader line.
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    for t_sp, i in struck:
        bb = inv.transform(t_sp.get_window_extent(ax.figure.canvas.get_renderer()))
        ax.plot([bb[0][0], bb[1][0]], [i, i], color="#C9C4BC", lw=0.55, zorder=6)

    ax.text(0.0, len(rows) + 0.20,
            f"{P['n_evaluable']} of {P['n_annotated']} annotated panels can host a protein LM;\n"
            f"ESM-2 has a file for {P['n_esm_file']} but is scored at usable size on only "
            f"{P['n_esm_usable']}",
            # Bold was emphasis, not encoding. These four counts are the panel's own summary
            # and are not in the legend (which is at 291 words against a 300-word cap), so the
            # sentence stays as a data annotation -- set in the same neutral ink as every other
            # count on the plate.
            fontsize=FOOT, color=MUTED, ha="left", va="top", linespacing=1.45)


def main():
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    # SPACER ROWS, not a uniform hspace. matplotlib's hspace is a single fraction applied between every
    # row, so sizing it for the largest gap would put that gap everywhere. Row 1 holds no axes; each
    # gap is sized for what actually sits in it.
    # Panels b and c sit side by side in ONE row and panel a takes the remaining height. Its plates
    # are sized from their own box, so that height becomes larger cells and taller short strips --
    # which the positive-class halves on chicken/goat/sheep/pig need to show a hole texture.
    gs = fig.add_gridspec(3, 2, height_ratios=[131.0, 7.0, 33.5], hspace=0.0, wspace=0.42,
                          left=0.092, right=0.972, top=0.95000, bottom=0.11800)

    tot = sum(R[s]["n"] for s in TREE)
    sb = sum(round(R[s]["reach_neg"] * R[s]["n_neg"]) for s in TREE)
    sp_ = sum(round(R[s]["reach_pos"] * R[s]["n_pos"]) for s in TREE)

    panel_a(fig, gs[0, :])
    # ALIGNED. panel a's letter is placed in figure coords at style_gb's shared letter margin, the
    # same x panel b's letter uses, set flush left so a and b share one letter margin.
    fig.text(SG_letter_x(fig), 0.96400, "a", fontsize=PANEL, fontweight="bold", ha="left", va="bottom",
             color=INK)
    _zs = [z for r in J["per_species"] for z in (r.get("runs_z_benign"), r.get("runs_z_pathogenic"))
           if z is not None]
    _cl = [r["species"] for r in J["per_species"]
           if (r.get("runs_z_benign") or 0) < -2 or (r.get("runs_z_pathogenic") or 0) < -2]
    _nsig = sum(1 for r in J["per_species"] if _sig_gap(r))
    # "half height = class size" belongs HERE, not only in the footer: with one cell size per plate the
    # pathogenic half is a short strip, and a reader who does not know why could read the clear area
    # above it as punched-out rather than as absent variants. The clustering count moves to the footer.
    fig.text(0.972, 0.96400,
             f"Punched = no score  ·  half height = class size  ·  bold name and red gap = CI "
             f"excludes 0 ({_nsig} of 9)",
             fontsize=FOOT, color=MUTED, ha="right", va="bottom")
    fig.add_artist(plt.Line2D([0.092, 0.972], [0.98000, 0.98000], color="#C4BFB8", lw=0.6))

    # the headline number rides in the title, not a caption: the only slot left under this panel sits
    # directly above panel d's own title, where a caption reads as d's rather than b's.
    axb = fig.add_subplot(gs[2, 0]); panel_b(axb)
    _pen = max(r["must_call_penalty"] for r in J["per_species"])
    _plab(axb, "b", left=True, text=f"What the silence costs: up to {_pen:.3f} AUROC")
    # The slope triangle (panel_c) is not on the plate. Its numbers -- the
    # regime contrast +0.119 AUROC per decade of panel size, species-resampled 95% CI
    # [+0.019, +0.171], p = 0.0117, against a "no regime dependence" null leg -- are
    # in reports/fig5_reach.json["regime"] and belong in the legend.

    # panel_d() draws the plate's panel c, the condition matrix.
    axc = fig.add_subplot(gs[2, 1]); panel_d(axc); _plab(axc, "c", "Labels do not rescue you")

    # The ESM-2 evaluability ledger (panel_e) is not on the plate. Its
    # per-species missense/pos/neg census and the summary "3 of 9 annotated panels can
    # host a protein LM; ESM-2 has a file for 6 but is scored at usable size on only 1" are
    # in reports/fig5_reach.json["protein_lm"] and belong in the legend.

    # TITLE AND SUBTITLE MOVED TO THE MANUSCRIPT LEGEND (house style, src/ccs/style_gb.py).
    # The canvas carried a bold title and a six-line italic paragraph that argued the figure's
    # claim (the co-scorable tie, the per-species win/loss/tie counts, Nucleotide Transformer's
    # reach and its 0.595 atlas mean). A journal figure states none of that on the plate: the
    # legend does, set in the journal's own face. Every number in the deleted paragraph is in
    # the legend text. The 22 mm it occupied goes to the panels.
    C = J["coding_confound"]
    # The species NAMES that carry the extremes are looked up, not typed. The footer this replaces
    # hardcoded "z = -25.3 (human) to -2.1 (pig)" -- values that existed nowhere in the data and named
    # the wrong species. Typing the label next to a computed number reintroduces exactly that bug.
    _zpairs = [(z, r["species"]) for r in J["per_species"]
               for z in (r.get("runs_z_benign"), r.get("runs_z_pathogenic")) if z is not None]
    (_zmin, _zmin_sp), (_zmax, _zmax_sp) = min(_zpairs), max(_zpairs)
    _hu = next(r for r in C["per_species"] if r["species"] == "human")
    _cn = [r["coding_frac_neg"] for r in C["per_species"] if r["species"] != "human"]
    _cp = [r["coding_frac_pos"] for r in C["per_species"] if r["coding_frac_pos"] is not None]
    # METHODS FOOTER IS IN THE MANUSCRIPT LEGEND, not on the canvas: a methods paragraph is pure
    # prose, not a data label.

    os.makedirs(ROOT / "reports" / "figures", exist_ok=True)
    base = str(ROOT / "reports" / "figures" / "Figure5_reach")
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
    fig.savefig(base + ".pdf", dpi=600, bbox_inches=mpl.transforms.Bbox(
        [[0.0, _y0], [fig.get_figwidth(), fig.get_figheight()]]), metadata={"CreationDate": None})
    fig.savefig(base + ".png", dpi=600)
    # The 4k screen export is produced HERE so it has a producer in the repo. It previously existed in
    # reports/figures with no script anywhere in src/ or scripts/ that could have made it -- an orphan
    # a reviewer could mistake for the figure of record.
    fig.savefig(base + "_4k.png", dpi=4096 / (W_MM * MM))
    return fig


if __name__ == "__main__":
    main()
