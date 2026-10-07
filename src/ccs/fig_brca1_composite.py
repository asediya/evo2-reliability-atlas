"""Composite BRCA1 positive-control figure (Additional file 1's Figure S6): Evo 2 reconstructs the experimental
variant-effect map in SEQUENCE and STRUCTURE, across both structured domains, predicted vs measured.
 a  domain-architecture track + experimental SGE truth track + Evo 2 saturation quilt (8192-bp),
    with the RING zinc cage (7 Cys + His41) marked in an annotation lane
 b  2x2 structure grid: {RING, BRCT} x {Evo 2 predicted, SGE measured} on a shared blue->red ramp
 c  per-residue Evo 2 vs measured deleteriousness (continuous SGE); rho RING +0.56 / BRCT +0.50 /
    pooled +0.52, with the statistics printed in the strip below the panel rather than over it

All annotations verified against reports/brca1_residue_scores.parquet + sourced BRCA1 biology.
Reference amino-acid identities come from the literature/canonical numbering, never from our VEP
`aa` column (which mislabels several BRCT reference residues).

Type, colour and canvas come from src/ccs/style_gb.py. Nothing here sets a font size of its own.

WHAT MOVED TO THE LEGEND (this pass). Three blocks of prose left the canvas; the legend has to
carry them, and none of them is data:
  1. "reference notebook, 1B: 0.73  ->  reproduces, not beats" -- editorial commentary, and the
     one true bbox collision in the submitted file (it ran through "res 1837 . max Evo 2").
  2. the three-line italic note under panel b about independent min/max scaling.
  3. the PDB composition line ("RING 1JM7 . BRCA1 1-103 + BARD1 (gray) + 2 Zn2+ / BRCT 1T29 .
     tandem BRCT + BACH1 phosphopeptide"). The PDB accessions themselves stay on the canvas as
     the column subtitles; only the composition detail moved.
Two numeric values leave the canvas with them -- 0.73 (the 1B notebook AUROC) and 103 (the RING
construct's last residue) -- and both are stated in the proposed legend text.
"""
import sys, os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, rankdata
import figstyle as fs
from figtext import restore_text_layer

# The shared Genome Biology house style. Imported the same way as figtext: relative when this file
# runs as part of the package, flat when src/ccs is on sys.path directly. figstyle updates rcParams
# at import time, so SG.rc() is applied AFTER it and wins -- this figure takes its whole type scale,
# its canvas and its ink from the contract, and a change there reaches it without an edit here.
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG

plt.rcParams.update(SG.rc())

# RING and BRCT are the partition this whole figure is organised by -- the domain track in a, the
# four structure panels in b and the markers in c -- so the pair has to separate by lightness as
# well as by hue: blue against amber is the standard CVD-safe opposition, and a lightness gap keeps
# it in greyscale. Both colours also have to clear WCAG as marks and as type, because the key in a
# sets its swatches as glyphs and the headers in b set the domain names in their colours. So BRCT
# is the text-safe ochre (4.59:1 on white; 3.92:1 against the #EDEDED of unannotated positions)
# and RING the navy #0A2A5A (14.1:1 on white), which puts 3.07:1 between them in greyscale.
# Measured under the Machado dichromacy matrices the two sit 0.70 (deuteranopia), 0.62
# (protanopia) and 0.71 (tritanopia) apart in sRGB, against 0.50 for a fully distinguishable pair.
# Panel c also carries redundant marker SHAPES (DOM_MARK), but the track and the key in a do not
# and cannot, so there the colours do the work on their own. Type goes through text_safe(), which
# returns both colours unchanged because both already clear 4.5:1.
RING_C, BRCT_C, LINK_C = "#0A2A5A", SG.OCHRE_TEXT, "#B4B4B4"
RING_TXT, BRCT_TXT = SG.text_safe(RING_C), SG.text_safe(BRCT_C)
# CVD-safe, luminance-ordered single-hue ORANGE ordinal ramp for the SGE class (FUNC < INT < LOF): no
# red-green, survives greyscale + deuteranopia/protanopia, and warm-hue so light FUNC != grey no-coverage.
SGE_FUNC, SGE_INT, SGE_LOF = "#FDBE85", "#E6550D", "#7F2704"
LOF_C = "#B12A20"          # deleterious/pathogenic highlight (C61G) — same sense as the red del. pole
MISS_C = "#D81B8C"         # 'model miss' highlight (start-loss), decoupled from deleteriousness red
DOM_MARK = {0: "o", 1: "^", 2: "s"}   # RING circle, BRCT triangle, linker square — redundant (non-colour) key
CAP = SG.MUTED             # secondary ink: provenance, keys, units
ZINC_LIG = [24, 27, 39, 41, 44, 47, 61, 64]   # C3HC4 RING zinc cage: 7 Cys + His41

# Multi-line labels need a leading TALLER than the glyph box, or consecutive lines of the same
# label overlap each other by a fraction of a point. At matplotlib's default 1.2 the advance for
# an 8 pt Arial line is 8.71 pt against a 8.94 pt glyph box, and four such pairs were reported as
# collisions in the submitted PDF: the two-line y-label, the two-line colour-bar label and both
# panel-b row labels. Not visible at print size, but a bbox scan cannot tell them from a real
# overlap, and the fix costs a fraction of a millimetre.
LEAD = 1.45


def load_heatmap():
    w = pl.read_parquet("data/interim/brca1_windows.parquet")
    s = pl.read_parquet("data/processed/scores_cloud/brca1_evo2_40b_meanll_8192.parquet")
    d = w.join(s, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    pos = np.array([int(v.split("_")[1]) for v in vid]); alt = np.array([v.split("_")[3] for v in vid])
    score = -d["evo2_meanll_delta"].to_numpy().astype(float)
    fclass = np.array(d["func_class"].to_list())
    # BRCA1 is MINUS-strand (pos vs residue rho = -0.96): descending genomic coord = transcript 5'->3'
    # = protein N->C, so the figure reads RING (N-term) -> BRCT (C-term) left to right.
    upos = np.array(sorted(set(pos), reverse=True)); xr = {p: i for i, p in enumerate(upos)}; N = len(upos)
    ALT = {"A": 0, "C": 1, "G": 2, "T": 3}
    M = np.full((4, N), np.nan)
    for p, a, sc in zip(pos, alt, score):
        M[ALT[a], xr[p]] = sc
    rank = {"FUNC": 0, "INT": 1, "LOF": 2}
    truth = np.zeros(N)
    for p, fc in zip(pos, fclass):
        truth[xr[p]] = max(truth[xr[p]], rank[fc])
    # genomic pos -> residue, then residue -> mean x-index (for residue-anchored annotations)
    vr = pl.read_parquet("data/interim/brca1_variant_residues.parquet")
    posres = {int(v.split("_")[1]): r for v, r in zip(vr["variant_id"].to_list(), vr["residue"].to_list())}
    acc = defaultdict(list)
    for p in upos:
        r = posres.get(p)
        if r is not None:
            acc[r].append(xr[p])
    res2x = {int(r): float(np.mean(v)) for r, v in acc.items()}

    def dom(res):
        if res is None: return 0
        if res <= 109: return 1
        if 1642 <= res <= 1863: return 2
        return 0
    domtrack = np.array([dom(posres.get(p)) for p in upos])
    bm = np.nanmedian(score[fclass == "FUNC"]); lo, hi = np.nanpercentile(score, [3, 97])
    return M, truth, domtrack, TwoSlopeNorm(vmin=lo, vcenter=bm, vmax=hi), N, res2x


# Fractions of the ring_evo2 render, MEASURED on the shipped PDF at 600 dpi. 3Dmol's addResLabels
# (make_brca1_structure.py, label_res = [24, 61]) bakes its labels into the raster at a size the
# renderer picks, and here that came out at roughly 2.2 pt: a third of style_gb.FLOOR, unreadable
# at print size, and invisible to EVERY gate in this repo because raster text is not in the PDF's
# text layer. pdfcheck sees no span, the 6.5 pt floor is never tested.
#
# Re-rendering needs the 3Dmol/browser pipeline, which is not available here, so the baked labels
# are covered on their own white plates and redrawn as real text at ANNOT. They also read "CYS24"
# and "CYS61" while the legend and panel a use the one-letter form (C61G), so the redraw settles
# that too. If the render is ever regenerated these fractions must be re-measured -- they are
# fractions, not pixels, so they survive a change of resolution but not a change of view.
_CYS_BOX = {"C24": (0.6333, 0.7241, 0.5110, 0.5450),
            "C61": (0.6509, 0.7300, 0.7577, 0.7873)}


def _relabel_cys(ax, ncol, nrow):
    """Cover the ~2.2 pt baked residue labels and redraw them legibly."""
    # No separate cover patch: "C24" at 6.8 pt sets to about 14 x 6 pt and the baked box is
    # 7.7 x 3.1 pt, so the replacement's own white bbox, centred on the same point, hides it.
    for name, (fx0, fx1, fy0, fy1) in _CYS_BOX.items():
        ax.text(0.5 * (fx0 + fx1) * ncol, 0.5 * (fy0 + fy1) * nrow, name, fontsize=SG.ANNOT,
                ha="center", va="center", color=SG.INK, zorder=5, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.4))


def autocrop(path):
    img = mpimg.imread(path)
    m = (img[:, :, :3] < 0.965).any(axis=2)
    ys, xs = np.where(m)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else img


def _auroc(y, s):
    """AUROC as P(score_pos > score_neg) via ranks; y binary, s continuous score."""
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def variant_level():
    """Per-variant arrays for the AUROC (all 3,893 substitutions), with genomic site for clustering."""
    w = pl.read_parquet("data/interim/brca1_windows.parquet")
    s = pl.read_parquet("data/processed/scores_cloud/brca1_evo2_40b_meanll_8192.parquet")
    d = w.join(s, on="variant_id", how="inner")
    score = -d["evo2_meanll_delta"].to_numpy().astype(float)          # deleteriousness
    islof = (np.array(d["func_class"].to_list()) == "LOF").astype(int)  # LOF = positive; FUNC+INT = negative
    site = np.array([int(v.split("_")[1]) for v in d["variant_id"].to_list()])
    return score, islof, site


def boot_auroc(score, islof, site, B=1000, seed=3):
    """95% CI for AUROC, resampling genomic SITES (≤3 subs/site are non-independent)."""
    rng = np.random.default_rng(seed)
    us = np.unique(site); by = {int(x): np.where(site == x)[0] for x in us}
    out = []
    for _ in range(B):
        idx = np.concatenate([by[int(x)] for x in rng.choice(us, len(us), replace=True)])
        a = _auroc(islof[idx], score[idx])
        if np.isfinite(a): out.append(a)
    return np.percentile(out, [2.5, 97.5])


def boot_rho(a, b, B=2000, seed=5):
    """95% CI for Spearman rho, resampling residues with replacement."""
    rng = np.random.default_rng(seed); n = len(a); out = []
    for _ in range(B):
        ix = rng.integers(0, n, n); out.append(spearmanr(a[ix], b[ix])[0])
    return np.percentile(out, [2.5, 97.5])


def main():
    M, truth, domtrack, norm, N, res2x = load_heatmap()
    rs = pl.read_parquet("reports/brca1_residue_scores.parquet")
    res = rs["residue"].to_numpy(); ev = rs["evo2_mean"].to_numpy()
    mdel = -rs["sge_score_mean"].to_numpy()            # measured deleteriousness (continuous SGE function score)
    rho_all, p_all = spearmanr(ev, mdel)
    ring_m = res <= 109; brct_m = (res >= 1642) & (res <= 1863)   # SAME bounds as the colouring + legend
    rho_ring = spearmanr(ev[ring_m], mdel[ring_m])[0]
    rho_brct = spearmanr(ev[brct_m], mdel[brct_m])[0]
    n_ring = int(ring_m.sum()); n_brct = int(brct_m.sum())
    rlo, rhi = boot_rho(ev, mdel)                                 # residue-bootstrap 95% CI for pooled rho
    vscore, vislof, vsite = variant_level()
    auroc_pt = _auroc(vislof, vscore); alo, ahi = boot_auroc(vscore, vislof, vsite)
    n_lof = int(vislof.sum()); n_neg = len(vislof) - n_lof

    # The site-clustered interval quoted in the Results and in this figure's
    # legend is computed here; compiled_results.parquet carries only the variant-level
    # [0.862, 0.886], not the site-clustered [0.858, 0.889]. This
    # builder needs data/, so the interval travels as its own small artifact.
    _stats = {
        "_meta": {
            "what": "BRCA1 saturation-editing variant-level discrimination and its site-clustered CI",
            "estimator": "AUROC, LOF against FUNC+INT; 95% percentile bootstrap resampling genomic "
                         "SITES, because up to three substitutions per site are not independent",
            "n_boot": 1000, "seed": 3,
            "source": "src/ccs/fig_brca1_composite.py, the builder of Additional file 1's Figure S6 (file Figure_BRCA1)",
            "note": "compiled_results.parquet carries the VARIANT-LEVEL interval for the same point "
                    "estimate. The two are different resampling units and both are reported.",
        },
        "auroc": float(auroc_pt),
        "ci95_site_clustered": [float(alo), float(ahi)],
        "n_lof": n_lof, "n_func_int": n_neg, "n_variants": int(len(vislof)),
        "n_sites": int(len(np.unique(vsite))),
    }
    with open("reports/brca1_stats.json", "w", encoding="utf-8", newline="\n") as _fh:
        json.dump(_stats, _fh, indent=1)
    print("  wrote reports/brca1_stats.json  AUROC %.4f  site-clustered CI [%.4f, %.4f]"
          % (auroc_pt, alo, ahi))
    rmap = {int(r): (float(md), float(e)) for r, md, e in zip(res, mdel, ev)}   # residue -> (measured, predicted)

    INK_ = SG.INK
    # 208 -> 195 mm (SG.H_MM_MAX). Genome Biology's 225 mm cap is for FIGURE PLUS LEGEND, and this
    # legend is long, so the graphic has to stop short of it. The 13 mm came out of the three prose
    # blocks that moved to the legend and a small trim of panel a, whose three stacked tracks were
    # taller than the type in them needs.
    H_PT = SG.H_MM_MAX * 72.0 / 25.4
    W_PT = SG.W_MM * 72.0 / 25.4
    PT = 1.0 / H_PT                # one point as a figure fraction, vertically
    # The absolute y anchors below (TOP_PT, BOT_PT, 474, 477, 484) were authored as points
    # measured down from the top of a 195 mm plate. The plate is now 173 mm
    # (style_gb.H_MM_MAX: a 300-word legend measures 47 mm, so 195 + legend broke the
    # 225 mm ceiling). Reading them through PT after the change put the six-line panel-c
    # statistics block at 484-541 pt on a 490 pt page: five of its six lines fell off the
    # sheet entirely, and a fully off-page span does not appear in PyMuPDF extraction at
    # all, so the gate reported one 1.2 pt overhang rather than five missing lines.
    # Y() maps an authored point to its PROPORTION of the authored plate, so the whole
    # stack scales 173/195 and every band keeps its share. PT stays a true point and is
    # still what the type-relative offsets below (marker pitch, letter rise) are built from:
    # those are sized against glyph boxes, which did not shrink.
    AUTH_PT = 195.0 * 72.0 / 25.4
    def Y(y_pt):
        return 1.0 - y_pt / AUTH_PT
    fig = plt.figure(figsize=(SG.W_MM / 25.4, SG.H_MM_MAX / 25.4))
    TOP_PT, BOT_PT = 22.0, 454.0   # first ink; foot of the panel b / panel c axes
    # left 0.085 -> 0.093: the "experiment" track label is the left-most ink on the page and at
    # SG.ANNOT it ended 1.3 mm from the trim, inside a printer's tolerance rather than outside it.
    GS_LEFT = 0.093
    gs = GridSpec(7, 2, height_ratios=[0.20, 0.12, 0.12, 0.58, 0.30, 0.66, 0.66], width_ratios=[1.06, 1],
                  hspace=0.0, wspace=0.30, left=GS_LEFT, right=0.895,
                  top=Y(TOP_PT), bottom=Y(BOT_PT))

    # ---- a0: annotation lane (RING zinc cage marks) ----
    axAnn = fig.add_subplot(gs[0, :]); axAnn.set_xlim(-0.5, N - 0.5); axAnn.set_ylim(0, 1); axAnn.axis("off")
    lx = {r: res2x[r] for r in ZINC_LIG if r in res2x}
    xs_lig = sorted(lx.values())
    for r, x in lx.items():
        c = LOF_C if r == 61 else "#444"; lw = 1.0 if r == 61 else 0.8
        axAnn.plot([x, x], [0.05, 0.36], color=c, lw=lw)
    axAnn.plot([xs_lig[0], xs_lig[-1]], [0.36, 0.36], color="#444", lw=0.8)
    # At 170 mm both labels sit in the leftmost quarter of the lane (the zinc ligands span x 92-270 of
    # ~1,300), and the two-line cage caption grew into the C61G callout. Separated VERTICALLY — the cage
    # caption above its own bracket, C61G below it — rather than by nudging x, which cannot work when
    # both must stay anchored near the same residues.
    # The caption's second line is counted, not asserted. Every ligand is loss-of-function in the
    # experiment (worst class LOF, rank 2), but not every one is deleterious in Evo 2: ranked among
    # the 326 scored residues, seven sit above the 90th percentile and C47 at the 56th (Note S43).
    _pct = {int(r_): 100.0 * (k_ - 1) / (len(ev) - 1)
            for r_, k_ in zip(res, rankdata(ev, method="average"))}
    _worst = dict(zip(rs["residue"].to_list(), rs["sge_worst"].to_list()))
    assert all(_worst[r_] == 2 for r_ in ZINC_LIG), "a zinc ligand is not LOF in the experiment"
    n_del = sum(_pct[r_] > 90.0 for r_ in ZINC_LIG)
    axAnn.text((xs_lig[0] + xs_lig[-1]) / 2, 0.58,
               "RING zinc cage · 7 Cys + His41\nLOF in experiment · %d of %d deleterious in Evo 2"
               % (n_del, len(ZINC_LIG)),
               ha="center", va="bottom", fontsize=SG.ANNOT, color=INK_, linespacing=LEAD)
    axAnn.annotate("C61G · pathogenic", xy=(lx[61], 0.20), xytext=(xs_lig[-1] + N * 0.045, 0.10),
                   ha="left", va="center", fontsize=SG.ANNOT, color=LOF_C,
                   arrowprops=dict(arrowstyle="-", color=LOF_C, lw=0.5))
    # House style: a bold lower-case letter and nothing else; the legend names each panel.
    # x is the shared panel-letter margin in FIGURE coords, not a fraction of axAnn's own axes, so
    # this letter stands in the same column as b.
    from matplotlib.transforms import blended_transform_factory as _blend
    # The lane's two-line caption overflows the lane and reaches page y 13.4 pt, so anchoring the
    # letter to the lane would leave the caption above it. The letter takes the plate's own top
    # margin instead -- 3.4 pt, the same as Figure 1's -- so it is the topmost mark on the page.
    fig.text(SG.letter_x(fig), 1.0 - 3.4 / (fig.get_size_inches()[1] * 72.0), "a",
             fontsize=SG.PANEL, fontweight="bold", va="top", ha="left", color=INK_)

    # ---- a: domain track + SGE truth + Evo2 quilt ----
    axD = fig.add_subplot(gs[1, :])
    axD.imshow(domtrack[None, :], aspect="auto", cmap=ListedColormap(["#EDEDED", RING_C, BRCT_C]),
               vmin=0, vmax=2, interpolation="nearest")
    axD.set_yticks([0]); axD.set_yticklabels(["Domain"], fontsize=SG.ANNOT); axD.set_xticks([])
    for s in axD.spines.values(): s.set_visible(False)
    axD.tick_params(length=0)

    axSg = fig.add_subplot(gs[2, :])
    axSg.imshow(truth[None, :], aspect="auto", cmap=ListedColormap([SGE_FUNC, SGE_INT, SGE_LOF]),
                vmin=0, vmax=2, interpolation="nearest")
    axSg.set_yticks([0]); axSg.set_yticklabels(["Experiment"], fontsize=SG.ANNOT); axSg.set_xticks([])
    for s in axSg.spines.values(): s.set_visible(False)
    axSg.tick_params(length=0)

    axH = fig.add_subplot(gs[3, :])
    # SG.tol_del_cmap(), not RdBu_r: RdBu_r's two ends have identical relative luminance, so in
    # greyscale a maximally tolerated substitution and a maximally deleterious one are the same
    # grey and this quilt carries no signal. See style_gb.TOL_DEL_NODES for the measurements.
    # The four structure renders in panel b were re-mapped onto the same ramp by
    # tools/recolour_brca1_structures.py, so panel a, panel b and the shared bar still agree.
    im = axH.imshow(M, aspect="auto", cmap=SG.tol_del_cmap(), norm=norm, interpolation="nearest")
    axH.set_yticks(range(4)); axH.set_yticklabels(list("ACGT"), fontsize=SG.TICK)
    axH.set_ylabel("Evo 2\nalt base", fontsize=SG.AXIS, linespacing=LEAD)
    axH.set_xticks([]); axH.set_xlabel("BRCA1 coding substitutions  ·  protein N→C  ·  introns collapsed",
                                       fontsize=SG.AXIS)
    for s in axH.spines.values(): s.set_edgecolor("#CCC")

    # The right-hand margin is 17 mm at 170 mm wide and every key in it is pinned to the track it
    # explains. Pinned to the AXES, not to hand-tuned figure fractions: the canvas lost 13 mm in
    # this pass, and every one of those fractions would otherwise have had to be re-tuned by hand.
    pD, pS, pH = axD.get_position(), axSg.get_position(), axH.get_position()
    KX, KX2 = 0.905, 0.918          # swatch column, label column
    # 0.44 -> 0.36 of the quilt height. The bar is pinned to the quilt's foot and grows upward,
    # so on the shorter 173 mm canvas its top tick ("0.010") rose into the SGE key's last row
    # ("LOF"), which is pinned to a different axes. Both live in the same 17 mm margin column;
    # the bar is the one with slack, since its job is a ramp and not a scale to read values off.
    cax = fig.add_axes([0.8985, pH.y0, 0.012, 0.36 * pH.height])   # aligned with the foot of the quilt
    cb = fig.colorbar(im, cax=cax)
    # Everything between the bar and the trim is on a 40 pt budget: tick, pad, "0.010", labelpad,
    # and two stacked lines of rotated label. Raising the leading to LEAD widened the label by
    # 1.8 pt and pushed its last line 0.9 pt OFF the page, so the bar, the gridspec's right edge
    # and both pads each give a little back and this label keeps a 1.3 pt tighter leading than the
    # rest — still 0.4 pt clear of its own glyph box, which is all the leading has to buy here.
    cb.set_label("Evo 2  −Δ mean-LL\n(8,192 bp) · red = deleterious", fontsize=SG.ANNOT, labelpad=2.5,
                 linespacing=1.30)
    # Explicit ticks and explicit labels. Shortening the bar to clear the SGE key handed the
    # locator a 34 pt run and it dropped to "0.00"/"0.01" -- two decimals on a scale whose real
    # maximum is 0.01486, which reads as if the ramp topped out at one hundredth. Two ticks, not
    # the three the 195 mm plate carried: this norm is not linear, so 0.005 and 0.010 sit 7.4 pt
    # apart at this length and their labels overlap. A ramp keyed at its ends is the honest form
    # here -- the panel asks the reader to compare cells, not to read values off the bar.
    cb.set_ticks([0.000, 0.001, 0.010])
    cb.ax.set_yticklabels(["0.000", "0.001", "0.010"])
    cb.ax.tick_params(labelsize=SG.ANNOT, length=2, pad=1.2)
    # domain legend (swatches), sitting against the domain track it keys
    for i, (lab, c) in enumerate([("RING", RING_C), ("BRCT", BRCT_C)]):
        y = pD.y1 - (3.0 + i * 10.0) * PT   # pitch > the 8.94 pt glyph box of a ■ at SG.AXIS
        fig.text(KX, y, "■", color=c, fontsize=SG.AXIS, va="center")
        fig.text(KX2, y, lab, fontsize=SG.ANNOT, va="center", color=CAP)
    # experiment-track KEY: real swatches (not colour-name text), CVD-safe ordinal ramp.
    # As one 24 mm line the header would overrun the page or sit on the domain raster, so it is
    # wrapped onto four short lines, the widest of which ("subs per site:") is 38 pt.
    # FLOOR: a four-line key in a fixed reservation beside the colour swatches.
    fig.text(KX, pS.y1 + 1.0 * PT, "SGE class\n(Findlay 2018)\nworst of ≤3\nsubs per site:",
             fontsize=SG.FLOOR, va="top", color=CAP, linespacing=LEAD)
    # The label is set in its swatch's colour; SGE_FUNC alone was 1.63:1 on white. text_safe
    # darkens the TYPE only -- the swatch above it keeps the full-chroma hue.
    for i, (lab, c) in enumerate([("FUNC", SG.text_safe(SGE_FUNC)), ("INT", SG.text_safe(SGE_INT)),
                                  ("LOF", SG.text_safe(SGE_LOF))]):
        y = pS.y1 - (41.0 + i * 9.5) * PT
        fig.text(KX, y, "■", color=c, fontsize=SG.TICK, va="center")
        fig.text(KX2 - 0.001, y, lab, fontsize=SG.ANNOT, va="center", color=CAP)

    # ---- b: 2x2 structure grid ----
    sgs = gs[5:7, 0].subgridspec(2, 2, hspace=0.0, wspace=0.04)
    grid = [["ring_evo2", "brct_evo2"], ["ring_sge", "brct_sge"]]
    rowlab = ["Predicted\n(Evo 2)", "Measured\n(SGE)"]; collab = ["RING", "BRCT"]
    pdbid = ["PDB 1JM7", "PDB 1T29"]; colc = [0.19, 0.40]; grid_top = 0.50
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(sgs[i, j])
            p = ax.get_position()
            if i == 0:
                colc[j] = 0.5 * (p.x0 + p.x1); grid_top = p.y1
            ax.axis("off")
            _im = autocrop(f"reports/figures/brca1_{grid[i][j]}.png")
            ax.imshow(_im)
            if grid[i][j] == "ring_evo2":
                _relabel_cys(ax, _im.shape[1], _im.shape[0])
            ax.set_anchor("S" if i == 0 else "N")     # pack the two rows toward the shared centre -> no mid gap
            if j == 0:
                # Right edge 0.5 mm clear of the structure render, so the label does not sit on the
                # BARD1 ribbon at the render's left edge. The clearance is in millimetres, not a
                # fraction of the render's width.
                from matplotlib.transforms import ScaledTranslation as _ST
                ax.text(0.0, 0.5, rowlab[i], rotation=90, va="center", ha="right",
                        transform=ax.transAxes + _ST(-0.5 / 25.4, 0.0, fig.dpi_scale_trans),
                        fontsize=SG.AXIS, fontweight="bold", color=INK_, linespacing=LEAD)
    for j in range(2):     # aligned column titles above the grid (fixed y so RING/BRCT line up)
        fig.text(colc[j], grid_top + 11 * PT, collab[j], ha="center", va="bottom",
                 fontsize=SG.AXIS, fontweight="bold", color=(RING_TXT if j == 0 else BRCT_TXT))
        fig.text(colc[j], grid_top + 1.5 * PT, pdbid[j], ha="center", va="bottom",
                 fontsize=SG.ANNOT, color=CAP)
    # The structures' composition — BARD1 partner (gray), the two Zn2+, the BACH1 phosphopeptide
    # cleft, the 1–103 construct bounds — is legend material and no longer set under the colour cue.
    # The accessions stay: they are what a reader needs to fetch the same coordinates.
    fig.text(SG.letter_x(fig), grid_top + 10 * PT, "b", fontsize=SG.PANEL, fontweight="bold", color=INK_,
             va="bottom", ha="left")
    # shared blue->red deleteriousness cue under panel b (each map scaled independently; that fact
    # is now stated in the legend rather than in three lines of italic prose on the canvas)
    cueax = fig.add_axes([0.205, Y(474.0), 0.15, 8 * PT])
    cueax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=SG.tol_del_cmap())
    cueax.set_xticks([]); cueax.set_yticks([])
    for s in cueax.spines.values(): s.set_edgecolor("#BBB"); s.set_linewidth(0.5)
    fig.text(0.205, Y(477.0), "Tolerated", fontsize=SG.ANNOT, ha="left", va="top", color=CAP)
    fig.text(0.355, Y(477.0), "Deleterious", fontsize=SG.ANNOT, ha="right", va="top", color=CAP)

    # ---- c: quantified -- predicted vs measured (continuous, per residue), coloured by domain ----
    axV = fig.add_subplot(gs[5:7, 1])
    dom_r = np.where(res <= 109, 0, np.where((res >= 1642) & (res <= 1863), 1, 2))
    # The 11 linker residues are drawn over the two domains. Drawn behind them, ten of the eleven grey
    # squares would sit under the RING and BRCT cloud, and the legend names them.
    dcol = {0: RING_C, 1: BRCT_C, 2: LINK_C}; zt = {2: 4, 0: 3, 1: 3}
    for k in (2, 0, 1):
        m = dom_r == k
        axV.scatter(mdel[m], ev[m], s=18, color=dcol[k], marker=DOM_MARK[k], alpha=0.72,
                    edgecolors="white", linewidths=0.3, zorder=zt[k])   # shape = redundant CVD key

    # concordant extremes: ring + label the two named residues (both high in Evo 2 AND experiment).
    # RING zinc ligands are the blue (RING) points in this upper region (see legend) -- no oversized ellipse.
    def _tag(r, text, xytext, col, arrow="-", rad=0.0, ha="left"):
        x, y = rmap[r]
        axV.scatter([x], [y], s=36, facecolors="none", edgecolors=col, linewidths=1.0, zorder=6)
        # Two things printed over these labels and no text-vs-text scan could see either: cloud
        # markers sitting on the glyphs (which made "start-loss" read as "stad-loss"), and the
        # leader itself running through "max Evo 2" like a strike-through. A white halo masks the
        # markers, a high zorder puts the text above the cloud, and shrinkA holds the leader off
        # the text box instead of letting it start inside it.
        # NO PATH EFFECT. The white halo that used to be here (withStroke, 2.4 pt) made matplotlib
        # render these three strings as vector PATHS and leave only an alpha-0 copy in the PDF text
        # layer -- so pdfcheck could not measure their type size, could not test them for
        # collisions, and reported bboxes that do not correspond to the painted glyphs. The halo
        # was added to mask cloud markers sitting on the glyphs; the labels were subsequently moved
        # into bands the scatter does not enter (nothing above 0.0129 except residue 1837, and the
        # 0.0105-0.0120 band is empty between x 1.0 and 2.4) and shrinkA now holds the leader off
        # the text box, so nothing is left to mask. style_gb's rule is no strokes behind text.
        axV.annotate(text, xy=(x, y), xytext=xytext, fontsize=SG.ANNOT, color=col, ha=ha,
                     va="center", zorder=10,
                     arrowprops=dict(arrowstyle=arrow, color=col, lw=0.7, shrinkA=3,
                                     connectionstyle=f"arc3,rad={rad}"))
    # With the statistics card gone from the top of the panel these two labels move UP into the
    # space it used to cover, beside the residues they name. Both are right-anchored so the leader
    # leaves the text's right edge and travels away from the type instead of across it, and both
    # sit in bands the scatter does not enter (nothing lies above 0.0129 except residue 1837, and
    # the 0.0105–0.0120 band is empty between x 1.0 and 2.4).
    if 1837 in rmap:
        _tag(1837, "Res 1837 · max Evo 2", (1.62, 0.01435), "#33506F", ha="right")
    if 96 in rmap:
        _tag(96, "Res 96 · max SGE", (2.30, 0.01125), "#33506F", ha="right")
    # honest model-miss: start-loss (residue 1) -- SGE LOF, Evo 2 near baseline. Coloured MISS (magenta),
    # decoupled from the deleteriousness-red used elsewhere.
    if 1 in rmap:
        # BELOW and right of the residue, in the band under the cloud's right flank that no marker
        # enters: seated above it at (1.72, 0.0044) the label's first letters sat on BRCT markers.
        _tag(1, "Start-loss (res 1)", (2.02, 0.0009), MISS_C, arrow="->")
    # (residue 1775 / M1775R under-call is discussed in the legend, NOT ringed on-figure: any leader to it
    #  would have to cross the dense cloud and could misread as a spurious negative trend in a no-fit plot.)

    # labelpad 3: the strip under this panel holds two things stacked with no elastic between them
    # -- this label and panel c's statistics -- and at AXIS 9 / ANNOT 7.4 a pad of 6 crosses them
    # by 1.4 pt. There is no room below to move the statistics into (the page is already the full
    # 173 mm and the block ends 4 pt off the foot), so the room comes from here. Measured at
    # labelpad 3: 3.2 pt of real ink between the tick digits and this label, 2.6 pt between this
    # label and the block.
    axV.set_xlabel("Measured deleteriousness   (SGE, −function score)", fontsize=SG.AXIS, labelpad=3)
    # The default locator emits a −0.002 y-tick that lies OUTSIDE ylim; matplotlib still draws its
    # label, pinned at the axes corner, where it collided with the x-axis label. Pin the ticks inside
    # the range instead — and carry the top one to 0.015, because the data really does reach 0.0149.
    # The old 0.012 top tick, with the statistics card floating over everything above 0.0098, is why
    # the top third of this panel read as empty: it was not empty, it was covered.
    axV.set_yticks([0.000, 0.005, 0.010, 0.015])
    axV.set_ylabel("Evo 2 predicted deleteriousness   (per residue)", fontsize=SG.AXIS)
    pe = int(np.floor(np.log10(p_all))) if p_all > 0 else -300
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.default": "regular",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})
    pstr = f"{p_all / 10.0 ** pe:.0f} $\\times$ 10$^{{{pe}}}$"   # 7 x 10^-24, as AF1 Note S60
    axV.legend(handles=[Line2D([0], [0], marker=DOM_MARK[0], color="none", markerfacecolor=RING_C, markersize=6.5, label="RING"),
                        Line2D([0], [0], marker=DOM_MARK[1], color="none", markerfacecolor=BRCT_C, markersize=6.5, label="BRCT"),
                        # "linker", without "(n = 11)": the count ran the row out of the marker-free
                        # corner and onto a BRCT marker; the legend gives it.
                        Line2D([0], [0], marker=DOM_MARK[2], color="none", markerfacecolor=LINK_C, markersize=6.5, label="Linker")],
               # UPPER LEFT AT y = 0.86. At lower right this key would sit exactly where the
               # "start-loss (res 1)" ring and its leader land: the pink circle would enclose the "N"
               # of "RING", so the key would read as though it were the thing being annotated. A ring
               # is a path, not a text span, so a text-vs-text collision count is blind by
               # construction; tools/inkunder.py, which redacts the text and measures the ink
               # underneath, puts the word there on 25% cover at 14.7% edge density, the worst on the
               # page. The marker-free rectangle at x < 0.7, y > 0.0105 is 60 x 68 pt: too small for
               # the nine-line stats card (209 pt wide, which is why it sits under the panel), ample
               # for three legend rows. The 0.86 matters -- flush at 1.0 the key would cross the
               # "res 1837 - max Evo 2" leader label by 3.8 pt, trading an ink collision for a worse
               # text one. Measured at 0.86: ZERO spans on textured ink anywhere in the figure, and
               # the text grazes are 0.08 pt.
               loc="upper left", bbox_to_anchor=(0.0, 0.86),
               frameon=False, fontsize=SG.TICK, handletextpad=0.25, borderpad=0.2)
    for s in ("top", "right"): axV.spines[s].set_visible(False)
    fig.text(0.5, grid_top + 10 * PT, "c", fontsize=SG.PANEL, fontweight="bold", color=INK_,
             va="bottom", ha="left")

    # ---- panel c's statistics, off the data and onto empty canvas ----
    # Genome Biology wants the key in the graphic, so every number stays on the page; what changed
    # is where. The nine-line card this replaces floated at the top-left of the scatter at alpha
    # 0.72 and covered the four most deleterious residues in the study — 1837 among them, the very
    # point the "max Evo 2" leader points at — which is why the panel appeared to top out at 0.0098
    # against a real maximum of 0.01486. It cannot simply be made smaller and kept inside: the
    # largest marker-free rectangle in the scatter is about 60 × 68 pt (x < 0.7, y > 0.0105) and the
    # widest line below is 209 pt. So it moves to the empty strip beneath the panel, left edge
    # aligned with panel c's own letter. The editorial line that used to close it
    # ("reference notebook, 1B: 0.73 → reproduces, not beats") is not a key and is not here; it is
    # an argument, and it belongs to the legend in the journal's voice.
    fig.text(0.5, Y(485.0),
             # "Spearman" is not decoration. This paper uses ρ for TWO quantities: the share of
             # pathogenic-benign pairs a score orders (Figures 2 and 4, defined in Methods as
             # ρ = r₊r₋) and Spearman's rank correlation (here, and in Figure S3). A bare ρ
             # beside an AUROC leaves a reader to guess which. The header names it once, and the
             # three values below inherit it.
             # No p-value on the plate: its exponent, set as a superscript, rendered at 5.2 pt, under the
             # 7.0 pt floor, and the interval beside it already carries the inference; Note S43 prints
             # it (p = 7 x 10^-24).
             f"Per-residue rank agreement, Spearman ρ  (n = {len(ev)})\n"
             f" RING ρ = {rho_ring:+.2f} (n = {n_ring})  ·  BRCT ρ = {rho_brct:+.2f} (n = {n_brct})\n"
             f" Pooled ρ = {rho_all:+.2f}  [{rlo:+.2f}, {rhi:+.2f}]\n"
             f"Variant-level discrimination  (n = {n_lof + n_neg:,})\n"
             f" AUROC = {auroc_pt:.3f}  [{alo:.3f}, {ahi:.3f}]  ·  LOF ({n_lof:,}) vs FUNC+INT ({n_neg:,})\n"
             f"  95% CIs: site-clustered / residue bootstrap",
             fontsize=SG.ANNOT, color=INK_, ha="left", va="top", linespacing=1.40)

    # TITLE AND STANDFIRST LIVE IN THE MANUSCRIPT LEGEND (house style, src/ccs/style_gb.py).
    # The plate carried a two-line bold title and a two-line italic provenance block; the legend
    # opens with the same title and names the panel, the assay, the 3,893 substitutions, the
    # Findlay reference, the readout and the 0.874 AUROC. Panel c prints that AUROC with its
    # interval below the panel as well, so no number leaves the figure for it.
    os.makedirs("reports/figures", exist_ok=True)
    # haloed text is rendered as outlines and leaves the PDF text layer. The
    # transparent twin puts it back without changing a rendered pixel.
    restore_text_layer(fig)
    # dpi= is what sets the resolution of RASTERS embedded in the PDF (structure renders, heatmap).
    # Without it matplotlib embeds them at ~100 dpi regardless of the vector text being scalable.
    fig.savefig("reports/figures/Figure_BRCA1.pdf", dpi=600, metadata={"CreationDate": None})   # vector master
    fig.savefig("reports/figures/Figure_BRCA1.png", dpi=600)    # crisp default (>300 for combination art)
    if os.environ.get("HIRES"):                                                       # heavy exports on demand
        fig.savefig("reports/figures/Figure_BRCA1_4k.png", dpi=384)       # ~4K width
        fig.savefig("reports/figures/Figure_BRCA1_1600dpi.png", dpi=1600)  # print master (large)
        print("wrote 4k + 1600dpi PNGs", flush=True)
    print(f"wrote Figure_BRCA1  (RING rho {rho_ring:+.2f}, BRCT {rho_brct:+.2f}, pooled {rho_all:+.2f}); "
          f"zinc ligands x={[round(res2x[r],1) for r in ZINC_LIG if r in res2x]}")


if __name__ == "__main__":
    main()
