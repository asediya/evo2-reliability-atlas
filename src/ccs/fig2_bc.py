"""Panels B (raincloud separation atlas) and C (ROC-curve fan) — from the verified per-variant join
(reports/fig2_pervariant.parquet, all 9 AUROCs asserted to reconcile). Standalone test render."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import gaussian_kde, rankdata
from sklearn.metrics import roc_auc_score
import fig2_style as S
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG

ORDER_B = ["human", "dog", "cat", "horse", "pig", "cattle", "sheep", "goat", "chicken"]  # ladder, human top


def _roc(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P, N = y.sum(), (1 - y).sum()
    return np.concatenate([[0], fp / N]), np.concatenate([[0], tp / P])


def panel_B(ax, order=None):
    """`order` is the row order, top to bottom; ORDER_B when omitted. The atlas plate passes the
    species tree's order (fig2_style.LEAF_ORDER) so this panel's rows run in the same order as
    panel a's and sit level with them when the plate aligns the two axes row for row.
    Each species keeps the point-strip jitter it had under ORDER_B -- the generator is seeded by the
    species' ORDER_B index, not by its row -- so reordering moves rows and changes no mark within one.
    """
    order = list(ORDER_B if order is None else order)
    assert sorted(order) == sorted(ORDER_B), order
    pv = pl.read_parquet("reports/fig2_pervariant.parquet")
    summ = json.load(open("reports/fig2b_summary.json"))
    ben_all = pv.filter(pl.col("label") == 0)["deleteriousness"].to_numpy()
    pat_all = pv.filter(pl.col("label") == 1)["deleteriousness"].to_numpy()
    xlo = float(np.percentile(ben_all, 1)); xhi = float(np.percentile(pat_all, 92))   # focus where separation lives
    xs = np.linspace(xlo, xhi, 240); span = xhi - xlo
    # The right pad is the seat of the AUROC / d label column, which is set in points and so does
    # not scale with the panel. At 0.32 span the widest row -- human, "d 1.1 (n1500)", four digits
    # where every other row has two or three -- ran 1.3 pt off the right edge of the page and would
    # have been cut off in print. 0.375 span buys the column ~9 pt of page margin; the violins lose
    # 3% of their width, which is invisible. A longer string can re-break it, so the saving plate
    # has to measure the label block against the page.
    # ylim top 1.52. The left column prints a bare "1,500/1,500" per row, and figure keys belong in
    # the graphic, so the canvas says which number is which. Without the extra room the band above
    # the first species holds 8.8 pt and a line of ANNOT needs 8.3, so the room is made rather than
    # borrowed: the rows keep their data positions and the panel gives up 3.7% of its vertical
    # scale, which is invisible on a violin and cheaper than dropping a row or setting the key below
    # the floor.
    # The right pad is 0.415 span, set with the left margin: the axes box is fixed, so widening one side alone
    # moves xhi right and pushes the right-hand stat column off the page (the _save guard
    # catches it). Both margins are set together, which holds that column in place.
    ax.set_xlim(xlo - span * 0.56, xhi + span * 0.415); ax.set_ylim(-8.9, 1.52)
    for a in np.linspace(0, xhi, 4):
        ax.axvline(a, color=S.GRID, lw=0.5, zorder=0)
    for i, sp in enumerate(order):
        # MUTED, Figure 6 only. Species is named on the row, so the fill's chroma is
        # redundant; 35%% toward neutral brings this plate into the set's range. Figure 7 is built
        # by other functions in this module and is not muted.
        yb = -i; col = S.desat(S.species_color(sp), 0.35)
        d = pv.filter(pl.col("species") == sp)
        val = d["deleteriousness"].to_numpy(); lab = d["label"].to_numpy()
        ben, pat = val[lab == 0], val[lab == 1]
        small = len(pat) < 25
        for grp, c, sign in [(ben, S.BENIGN, +1), (pat, S.PATHO, -1)]:
            if len(grp) > 8:
                dens = gaussian_kde(grp)(xs); dens = dens / dens.max() * 0.34
                a = 0.30 if (small and sign < 0) else 0.66
                ax.fill_between(xs, yb, yb + sign * dens, color=c, alpha=a, lw=0.4, edgecolor=c, zorder=3)
        rng = np.random.default_rng(ORDER_B.index(sp))     # the species' own seed, whatever its row
        # The two strips get their own sub-bands. The half-violins above already encode the class
        # by DIRECTION (negatives up, positives down, as the key says), but both point strips used
        # to be jittered into one band and separated by hue alone -- and the two hues measure a
        # greyscale contrast ratio of 1.15, i.e. the same grey. In greyscale each row collapsed to
        # one undifferentiated cloud. They are 0.46 and 0.42 apart under the deuteranopia and
        # protanopia matrices, so this was a greyscale failure and not a colour-vision one; the fix
        # is the same either way, and it costs nothing but 0.1 of a row.
        for grp, c, y0 in [(ben, S.BENIGN, -0.33), (pat, S.PATHO, -0.47)]:
            gc = np.clip(grp, xlo, xhi)
            ax.scatter(gc, yb + y0 - rng.uniform(0, 0.10, len(grp)), s=2.2, color=c, alpha=0.42,
                       edgecolors="none", zorder=2)
        nbeyond = int((pat > xhi).sum())                                   # pathogenic clipped tail (honest flag)
        if nbeyond:
            # Seated just inside the right edge, where the clipped points actually are, instead of
            # in the AUROC/d label column — at 6 pt it overlapped the next row's AUROC there.
            # At yb-0.20 the flag sits in the violin fill -- solid, so flat -- and the ink under it measures
            # 7.2% edge density against a 10% gate. At yb-0.38 it would sit in the middle of the benign point
            # strip, the busiest ink in the row, at 10.3%. Still inside the row it belongs to, still beside
            # the clipped tail it names. Measured with tools/inkunder.py, not judged by eye.
            ax.text(xhi + span * 0.015, yb - 0.20, f"+{nbeyond}→", ha="right", va="center",
                    fontsize=SG.ANNOT, color=S.PATHO, zorder=6)
        # class-mean shift arrow + Cohen's d + AUROC
        mb, mp = ben.mean(), pat.mean()
        ax.annotate("", xy=(mp, yb + 0.07), xytext=(mb, yb + 0.07),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3), zorder=4)
        ax.scatter([mb, mp], [yb + 0.07] * 2, s=11, color=[S.BENIGN, S.PATHO], zorder=5, edgecolors="white", lw=0.4)
        s = summ[sp]
        # The swatch keys the row's species colour, so it has to sit BESIDE the label, not under it.
        # It was at span*0.03 with the text starting at span*0.02 -- 0.9 pt to the RIGHT of the
        # text's own left edge -- so the disc printed on the "A" of "AUROC", overlapping it by
        # 2.6 pt. Nothing caught it: a swatch is a path, not a span, so no text-vs-text test sees
        # it, and tools/inkunder.py passes it because a single flat disc has almost no edge density
        # under the glyph, which is the right answer to the question that tool asks and the wrong
        # answer to this one. The column moves right instead; the widest row lands 2.2 pt inside
        # the page edge.
        ax.scatter([xhi + span * 0.03], [yb + 0.34], s=12, color=col, zorder=5, edgecolors="none")   # colour swatch
        ax.text(xhi + span * 0.06, yb + 0.34, f"AUROC {s['auroc']:.2f}", ha="left", va="center",
                fontsize=SG.ANNOT, color=S.INK)
        # "d 1.1 (n 1,500)": one space between symbol and value, none doubled, which is how panel d
        # already prints the same two quantities ("human  AUROC 0.974, d 1.1"). The n takes the same
        # "symbol space value" form, so the column reads down as one form; it is also 8.4 pt narrower
        # than an "n = " form, which keeps the widest row -- the only species with a four-digit n --
        # inside the right edge of the page at ANNOT 7.4. The row already prints n_pos and n_neg in full
        # at its left ("1,500/1,500").
        ax.text(xhi + span * 0.06, yb - 0.02, f"d {s['cohens_d']:.1f} (n {s['n_pos']:,})", ha="left", va="center",
                fontsize=SG.ANNOT, color=S.INK)
        # species name + n label at left (name above, n below)
        # ONE INK, like panel a's ring: the species NAME is set in INK, not in its species colour,
        # because a hue on the name would restate what the word already says.
        ax.text(xlo - (xhi - xlo) * 0.045, yb + 0.19, sp, ha="right", va="center", fontsize=SG.TICK,
                color=S.INK)
        # ":," -- the stat column prints "n 1,500", so this prints "1,500/1,500": one number, one
        # format, inside one panel.
        ax.text(xlo - (xhi - xlo) * 0.045, yb - 0.25, f"{s['n_pos']:,}/{s['n_neg']:,}",
                ha="right", va="center", fontsize=SG.ANNOT, color=S.MUTED)
    # THE KEY FOR THAT PAIR, seated on the column it describes rather than in the legend.
    # "positives / negatives", not "pathogenic / benign": eight of these nine species are OMIA, and
    # the package's wording convention reserves "pathogenic" for human ClinVar and calls the OMIA
    # entries catalogued positives. The classes are defined in the legend; the canvas names them in
    # the same words the ▲/▼ key below uses.
    ax.text(xlo - (xhi - xlo) * 0.045, 0.80, "positives / negatives",
            ha="right", va="center", fontsize=SG.ANNOT, color=S.CAP)
    # One labelled tick every 0.005, minor graduations every 0.0025. The automatic locator put a
    # label every 0.0025, which at the house tick size leaves 2.3 pt between "−0.0025" and "0.0000"
    # -- close enough that the PDF text extractor merges them into the single run "−0.00250.0000",
    # i.e. they print as one glued number. Note that a bbox collision test over text SPANS cannot
    # see this: once two labels are merged there is only one span left to test.
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_minor_locator(MultipleLocator(0.0025))
    # four decimals, as before, so the axis reads the same as the values it measures
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}".replace("-", "−")))
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.5)
    ax.set_yticks([]); ax.set_xlabel("Evo 2 deleteriousness  (−Δ mean-LL)", fontsize=SG.AXIS, color=S.CAP)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=SG.TICK)
    # The key names the two half-violins by direction and stops there. It used to go on to
    # "(upper strip negative, lower positive)", which made it 82 mm long: right-aligned here it
    # reached 26 mm back across the atlas plate's panel a and sat over that panel's dot plot. The
    # strips are separated by position and by the class colours of the violins above them, and the
    # sentence that says which strip is which is the legend's.
    # In words, not glyphs: "▲ negative ▼ positive" used two triangles that appear nowhere in the
    # panel, standing for the directions the two half-violins are drawn in.
    ax.text(xhi, 1.20, "negatives above, positives below",
            ha="right", fontsize=SG.ANNOT, color=S.CAP)
    # Right-anchored on the AXES edge, not on the data maximum. Seated at xhi this
    # line would run left far enough to cross the panel letter by 1.5 pt at PANEL 12 -- the letter sits
    # on the axes' left edge, and this is the only string on the page that shares its band. The
    # x here is an axes FRACTION: get_yaxis_transform is x-in-axes, y-in-data, which is what this
    # line wants -- it keeps its data row and does not guess where the page edge is. The band it
    # sits in is empty; the label column below it starts a full row down.
    ax.text(1.0, 1.48, "+n→ = positives beyond the axis limit", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=SG.ANNOT, color=S.CAP)


def panel_C(ax_roc):
    """ONE hue-neutral pooled hero ROC + bootstrap CI ribbon + a faint legend-free clade brush of all
    9 species (gestalt = every curve pinned to the corner, none on the diagonal), with the
    early-retrieval operating point marked ON the curves.

    This used to be two lettered panels: the ROC fan, and a separate strip of sensitivity at 90%
    specificity per species. Sensitivity at 90% specificity IS a point on the ROC -- it is the
    true-positive rate where false-positive rate reaches 0.10 -- so the strip re-plotted nine
    coordinates already drawn one panel above it, and no claim in the paper rested on it. The
    operating region is now a rule at FPR 0.10 across the curves that define it, with the range of
    sensitivities it cuts stated once. Nothing measured is lost; a duplicate display is."""
    pv = pl.read_parquet("reports/fig2_pervariant.parquet")
    summ = json.load(open("reports/fig2b_summary.json"))
    allv = pv["deleteriousness"].to_numpy(); alll = pv["label"].to_numpy().astype(float)

    # -------- ax_roc: pooled hero + CI ribbon + faint clade brush --------
    ax_roc.set_aspect("equal"); ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=(0, (4, 3)), zorder=1)      # chance diagonal (FULL range)
    grid = np.linspace(0, 1, 100); boots = []; rng = np.random.default_rng(3)
    for _ in range(300):
        b = rng.integers(0, len(allv), len(allv))
        f, t = _roc(alll[b], allv[b]); boots.append(np.interp(grid, f, t))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ax_roc.fill_between(grid, lo, hi, color=S.EVO2, alpha=0.16, lw=0, zorder=2)
    for sp in S.LEAF_ORDER:                                                          # faint brush, NO labels/legend
        d = pv.filter(pl.col("species") == sp)
        f, t = _roc(d["label"].to_numpy().astype(float), d["deleteriousness"].to_numpy())
        ax_roc.plot(f, t, color=S.species_color(sp), lw=0.5, alpha=0.38, zorder=3, drawstyle="steps-post")
        if summ[sp]["n_pos"] < 60:                                                   # count-the-thresholds dots (small n)
            ax_roc.scatter(f, t, s=3.5, color=S.species_color(sp), alpha=0.5, zorder=3, linewidths=0)
    f, t = _roc(alll, allv)                                                          # hue-neutral hero (reads in greyscale/CVD)
    ax_roc.plot(f, t, color="#222222", lw=2.0, zorder=5, solid_capstyle="round", drawstyle="steps-post")
    # Drawn on the FULL 8,192-bp panel: fig2b_data.py reads the full panel, so the curve is a
    # full-panel estimate that agrees with the manuscript's pooled AUROC.
    _n_sub, _auc_sub = len(alll), roc_auc_score(alll, allv)
    # What the black curve is, and how much data it is drawn from.
    ax_roc.text(0.05, 0.02,
                f"pooled AUROC {_auc_sub:.3f}\nn = {_n_sub:,} (8,192-bp panel)",
                ha="left", va="bottom", fontsize=SG.ANNOT, color=S.INK, linespacing=1.35)
    # The key for the nine faint curves, moved up into the band those curves run through -- it sat
    # in the empty lower triangle, as far from what it decodes as the panel allows. It stays in the
    # graphic: it names a plotted series, which is the one kind of text the house style keeps here.
    ax_roc.text(0.50, 0.72, "faint = 9 species", ha="left", va="center",
                fontsize=SG.ANNOT, color=S.MUTED)
    ax_roc.set_xlabel("false-positive rate", fontsize=SG.AXIS, color=S.CAP)
    ax_roc.set_ylabel("true-positive rate", fontsize=SG.AXIS, color=S.CAP)
    ax_roc.set_xticks([0, 0.5, 1]); ax_roc.set_yticks([0, 0.5, 1])
    for s in ("top", "right"): ax_roc.spines[s].set_visible(False)
    ax_roc.tick_params(labelsize=SG.TICK)

    # -------- the early-retrieval operating point, marked on the curves it is read from --------
    # Sensitivity at 90% specificity is the height of each curve at FPR 0.10. The rule below is
    # that abscissa; the dots are where each species' curve crosses it. Range computed, never typed.
    sens = {}
    for sp in S.LEAF_ORDER:
        d = pv.filter(pl.col("species") == sp)
        f, t = _roc(d["label"].to_numpy().astype(float), d["deleteriousness"].to_numpy())
        sens[sp] = float(np.interp(0.10, f, t))
    ax_roc.axvline(0.10, color=S.PATHO, lw=0.8, ls=(0, (3, 2)), alpha=0.85, zorder=4)
    for sp, v in sens.items():
        ax_roc.scatter([0.10], [v], s=7, color=S.species_color(sp), zorder=6,
                       edgecolors="white", linewidths=0.35)
    s_lo, s_hi = min(sens.values()), max(sens.values())
    # This was a sentence on a curved leader -- "early retrieval at 10% FPR" -- which reads as
    # commentary about the curves rather than as a label on the rule it points at. The rule is at
    # FPR 0.10 and the dots on it are the nine sensitivities; the label now sits against that rule
    # and states what it marks. Both numbers are unchanged and still computed, never typed. The
    # leader went with the sentence: a label 0.03 axis units from its own rule needs no arrow.
    ax_roc.text(0.135, 0.615, f"10% FPR\nsensitivity {s_lo:.2f}–{s_hi:.2f}",
                fontsize=SG.ANNOT, color=S.INK, ha="left", va="center", zorder=7, linespacing=1.45)


def main():
    from matplotlib.gridspec import GridSpec
    figB, axB = plt.subplots(figsize=(6.8, 6.4)); panel_B(axB)
    figB.savefig("reports/figures/_fig2_B.png", bbox_inches="tight", dpi=200)
    figC = plt.figure(figsize=(2.7, 4.1))                       # true ~2.2in-wide panel proportions
    gc = GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.42, left=0.17, right=0.96, top=0.955, bottom=0.11)
    # panel_C takes ONE axes: the sensitivity-at-90%-specificity strip is folded into the ROC
    # ("Sensitivity at 90% specificity IS a point on the ROC").
    panel_C(figC.add_subplot(gc[0]))
    figC.savefig("reports/figures/_fig2_C.png", bbox_inches="tight", dpi=300)
    print("wrote reports/figures/_fig2_B.png and _fig2_C.png")


if __name__ == "__main__":
    main()
