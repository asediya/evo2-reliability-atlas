"""Additional file 1's Figure S7: discrimination against divergence and the GERP plane.

  FigureS7_measurement_supp.pdf (Additional file 1, Figure S7)
                                       a discrimination against divergence
                                       b the GERP plane with its paired strip

Both panels come from fig2_measure.py (panel_c and panel_d). The submission's Figures 9 and 10 are drawn by
fig7_atlas.py and fig8_readout.py, which read the same atlas through fig2_measure and fig2_style; FIGURES.md
maps every figure to its builder.

    python src/ccs/fig2_split.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fig2_measure as MEAS
# The house style module. Imported the same way as figtext: relative when this file is
# run as part of the package, flat when src/ccs is on sys.path directly.
try:
    from . import style_gb as SG
except ImportError:
    import style_gb as SG
from figtext import restore_text_layer

MM = 1 / 25.4
W_MM = 170.0                    # GB: 170 mm wide, 225 mm tall INCLUDING the legend
# Authoring width in inches. The panels were written for a 12.2 in canvas; at that size the composite
# had to be scaled to 0.55 to reach 170 mm, which is what drove the smallest type to 4.3 pt. Each
# half is authored narrower so the scale to 170 mm is gentler and the type survives it.
H2_MM, H3_MM = 173, 173     # graphic height. The 225 mm cap in the guidelines covers
                                # "figure and legend", and a 300-word legend sets to roughly
                                # 25-35 mm, so a full-width graphic has to stop near 195 mm.

R = "reports"


def _fy(fig, pt_from_top):
    """A height given in points from the TOP edge, as a figure fraction (from the bottom)."""
    return 1.0 - pt_from_top / (fig.get_size_inches()[1] * 72.0)


def _fh(fig, pt):
    """A height in points as a figure fraction."""
    return pt / (fig.get_size_inches()[1] * 72.0)


def _letter(fig, ax, letter, x=None, pad_pt=4.5):
    """House style (src/ccs/style_gb.py): a bold lower-case letter over `ax` and nothing else, in INK.

    No panel carries a sentence-length heading beside its letter: such a heading either repeats the
    legend's per-panel sentence or states the finding the legend exists to state, and setting it bold
    in the Evo 2 blue is emphasis, not data. The call sites keep those strings as
    documentation of what each panel is; the canvas prints the letter only.

    x is the SHARED letter margin (SG.letter_x) for a LEFT-column panel and the axes' own left edge
    for a right-hand one -- that is its corner. Without the half-page test two panels that form a
    ROW both go to the margin and merge into one "ab" span. The height is clamped in POINTS
    (SG.letter_top): `pad_pt` is a distance above the axes, and a va="bottom" letter's em box stands
    1.118 em above its anchor, so an unclamped top-row letter leaves the page.
    """
    pos = ax.get_position()
    if x is None:
        x = SG.letter_x(fig) if pos.x0 < 0.5 else pos.x0
    fig.text(x, min(pos.y1 + _fh(fig, pad_pt), SG.letter_top(fig, margin_pt=0.4)), letter,
             fontsize=SG.PANEL, fontweight="bold", va="bottom", ha="left", color=SG.INK)


def build_supplement(N):
    """The supplementary plate (Additional file 1, Figure S7): fig2_measure's
    panels c and d.

      a  discrimination against divergence (fig2_measure.panel_c), with the pooled interval and the
         pooled value printed in a note above it;
      b  the GERP plane with its paired per-species strip (fig2_measure.panel_d).

    Set in the supplementary house style (style_gb), saved through the _save guard below. Panel a's
    row of group names sits under its tick labels and clears them, and its y-label takes the default
    pad, since no panel sits to its left. The seven species that share one divergence are named
    directly under their own intervals, in their text-safe colours. Panel b's key sits above the
    plane rather than on it, and its hexbin carries a count scale in the gutter beside it.
    """
    fig = plt.figure(figsize=(W_MM * MM, H3_MM * MM))
    # Panel a's top edge is set by the two-line note above it, which must stay on the canvas;
    # panel b sits 12 mm under panel a's axis title (its letter, and the strip's "difference" head).
    axA = fig.add_axes([0.115, 0.700, 0.805, 0.235])
    axB1 = fig.add_axes([0.115, 0.305, 0.440, 0.255])
    axB2 = fig.add_axes([0.720, 0.305, 0.200, 0.255])
    _pool, _dist, _k = MEAS.panel_c(axA)
    # Panel a's species, named where they stand. Human and chicken each have a divergence of their
    # own and are named by the tick row; the seven that share 92 My are dodged across their slot, so
    # each is named directly under its own interval, on one baseline just above the axis line, in
    # its text-safe colour (fig2_style.species_text_color, >= 4.5:1), the colour panel b's strip uses
    # for the same names. Each name is placed at the x of its own marker, found by its marker colour.
    import matplotlib.colors as mcolors
    from matplotlib.collections import PathCollection
    _forest = MEAS._load()[0]["forest"]
    _cnt = {}
    for _r in _forest:
        _cnt[_r["divergence_my"]] = _cnt.get(_r["divergence_my"], 0) + 1
    _shared = [_r["species"] for _r in _forest if _cnt[_r["divergence_my"]] > 1]
    _by_colour = {mcolors.to_hex(MEAS.S.species_color(sp)): sp for sp in _shared}
    assert len(_by_colour) == len(_shared), "two shared-slot species share a marker colour"
    _xs = {}
    for _c in axA.collections:
        if isinstance(_c, PathCollection) and len(_c.get_offsets()) == 1:
            _sp = _by_colour.get(mcolors.to_hex(_c.get_facecolor()[0]))
            if _sp is not None:
                _xs[_sp] = float(_c.get_offsets()[0][0])
    assert set(_xs) == set(_shared), (sorted(_xs), sorted(_shared))
    _name_tf = matplotlib.transforms.offset_copy(axA.get_xaxis_transform(), fig=fig, y=1.5, units="points")
    for _sp, _x in _xs.items():
        axA.text(_x, 0, _sp[:1].upper() + _sp[1:], transform=_name_tf, fontsize=SG.ANNOT, ha="center",
                 va="bottom", color=MEAS.S.species_text_color(_sp), clip_on=False, zorder=7)
    MEAS.panel_d(axB1, axB2)
    # Panel b's key, moved off the plane: one row above its top-left corner, on the letter's line, so
    # that no catalogued allele or hexagon sits under it. Same handles, same words, same grey.
    _lg0 = axB1.get_legend()
    _hd, _lb = _lg0.legend_handles, [t.get_text() for t in _lg0.get_texts()]
    _lg0.remove()
    _lg = axB1.legend(_hd, _lb, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                      borderpad=0.0, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.4,
                      fontsize=SG.ANNOT)
    for _t in _lg.get_texts():
        _t.set_color(SG.MUTED)
    # The hexbin, stroked in its own fill: unstroked hexagons leave hairline seams between
    # neighbours in most PDF renderers.
    from matplotlib.collections import PolyCollection
    _hb = [c for c in axB1.collections if isinstance(c, PolyCollection)]
    assert len(_hb) == 1, "panel b draws one hexbin"
    _hb = _hb[0]
    _hb.set_edgecolor("face")
    _hb.set_linewidth(0.3)
    # Its count scale, in the gutter between the plane and the strip, at the foot of the plane.
    _cmax = float(np.nanmax(_hb.get_array()))
    _ticks = [t for t in (1, 10, 100, 1000) if t <= _cmax]
    _pB = axB1.get_position()
    _cax = fig.add_axes([_pB.x1 + 4.0 / W_MM, _pB.y0, 1.3 / W_MM, 0.5 * _pB.height])
    _cb = fig.colorbar(_hb, cax=_cax)
    _cb.set_ticks(_ticks)
    _cb.set_ticklabels(["{:,}".format(t) for t in _ticks])
    _cb.minorticks_off()
    _cb.ax.tick_params(labelsize=SG.ANNOT, length=2, pad=1.2, width=0.4)
    _cb.outline.set_linewidth(0.3)
    # One line, no longer than the bar: the decade ticks (1, 10, ...) evenly spaced say "log".
    _cb.set_label("Variants per bin", fontsize=SG.ANNOT, labelpad=2.0)
    _LX = SG.letter_x(fig)
    for ax, letter in ((axA, "a"), (axB1, "b")):
        pos = ax.get_position()
        fig.text(_LX, min(pos.y1 + 0.016, SG.letter_top(fig, margin_pt=0.4)), letter,
                 fontsize=SG.PANEL, fontweight="bold", va="bottom", ha="left", color=SG.INK)
    # The note for panel a, right-aligned over the panel it measures.
    # The species count is the number of points the panel draws, not a typed 9.
    # The pooled value only. The Spearman test of AUROC against divergence
    # runs over nine points of which seven share one divergence, and the legend itself calls it
    # uninformative; a statistic that cannot inform does not belong on the plate.
    pos = axA.get_position()
    fig.text(pos.x1, pos.y1 + _fh(fig, 5.0),
             "Pooled %.3f [%.3f, %.3f]" % (_pool["auroc"], _pool["lo"], _pool["hi"]),
             fontsize=SG.ANNOT, color=SG.MUTED, ha="right", va="bottom", linespacing=1.4)
    _ = (_dist, _k)
    return fig


def _save(fig, stem):
    """Save at the declared canvas size.

    No bbox_inches='tight': the canvas is already 170 mm wide by construction, and tight-cropping a
    figure whose text is absolute in points does not converge -- rescaling to hit 170 mm moves the
    tight box, which moves the required scale. Fixing the canvas and placing everything in figure
    coordinates is what fig5_final and fig6_final do, and it makes the page size exact.
    """
    import fitz
    os.makedirs("reports/figures", exist_ok=True)
    pdf = f"reports/figures/{stem}.pdf"
    # A guard rather than a fix. Haloed text renders as vector outlines and drops out
    # of the PDF text layer, so every haloed string needs a transparent extractable twin.
    # House style sets no path effect on text in this figure, so this adds nothing
    # -- and it will go on adding nothing unless a halo is introduced.
    restore_text_layer(fig)
    # GB: figures are "closely cropped to minimise the amount of white space surrounding the
    # illustration". Crop the PAGE at the save, to the drawn extent plus 3 mm, and only ever
    # upward from the BOTTOM edge: the width stays the declared 170 mm, the top edge does not
    # move, and no artist is re-placed:
    # nothing authored in millimetres from the top is touched. A plate already tighter than
    # 3 mm keeps its own margin (the clamp at 0).
    _y0 = min(max(0.0, fig.get_tightbbox(fig.canvas.get_renderer()).y0 - 3.0 * MM),
              fig.get_figheight())
    SG.tidy_minus(fig)   # ASCII hyphen -> U+2212 in numeric labels; see style_gb.tidy_minus
    fig.savefig(pdf, dpi=600, bbox_inches=matplotlib.transforms.Bbox(
        [[0.0, _y0], [fig.get_figwidth(), fig.get_figheight()]]), metadata={"CreationDate": None})
    fig.savefig(f"reports/figures/{stem}.png", dpi=300)
    plt.close(fig)
    # Guard, not decoration: read the text back out of the PDF that was just written and refuse a
    # figure whose type is undersized or clipped. A label column set in points and seated against a
    # data coordinate can run off the right edge of the page with nothing else to catch it, and the
    # figure then looks finished and prints with its last character sheared off.
    doc = fitz.open(pdf); pg = doc[0]; r = pg.rect
    sp = [(s["bbox"], s["text"], s["size"]) for b in pg.get_text("dict")["blocks"]
          for ln in b.get("lines", []) for s in ln["spans"] if s["text"].strip()]
    doc.close()
    out = [(t, bb) for bb, t, _ in sp
           if bb[0] < -0.01 or bb[1] < -0.01 or bb[2] > r.width + 0.01 or bb[3] > r.height + 0.01]
    tiny = [(t, z) for _, t, z in sp if z < SG.FLOOR - 1e-6]
    assert not out, f"{stem}: text outside the page rect: {out}"
    assert not tiny, f"{stem}: type below the {SG.FLOOR} pt floor: {tiny}"
    assert r.width * 25.4 / 72 <= SG.W_MM + 0.05 and r.height * 25.4 / 72 <= SG.H_MM_MAX + 0.05
    print(f"wrote {pdf}  {r.width * 25.4 / 72:.1f} x {r.height * 25.4 / 72:.1f} mm  "
          f"({len(sp)} spans, min type {min(z for _, _, z in sp):.2f} pt, "
          f"right margin {(r.width - max(bb[2] for bb, _, _ in sp)) * 25.4 / 72:.1f} mm)")
    return pdf


def main():
    # The shared contract, applied before anything is drawn: Arial first (the plate is set in
    # ArialMT and must stay there), type 42 embedding, the one type scale, the neutral ink.
    plt.rcParams.update(SG.rc())
    _save(build_supplement(None), "FigureS7_measurement_supp")


if __name__ == "__main__":
    main()
