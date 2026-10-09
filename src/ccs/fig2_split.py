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

    Every panel used to carry a sentence-length heading beside its letter, set bold in the Evo 2
    blue; each either repeated the legend's per-panel sentence or stated the finding the legend
    exists to state, and the blue was emphasis, not data. The call sites keep those strings as
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
    """The supplementary plate (Additional file 1, Figure S7): the measurement plate's former
    panels c and d, unchanged in content.

      a  discrimination against divergence (fig2_measure.panel_c), with the pooled interval and the
         Spearman test that panel carried in a note above it;
      b  the GERP plane with its paired per-species strip (fig2_measure.panel_d).

    Same house style and the same _save guard as the two main plates. The panels are larger here
    than they were as a third and a fourth panel; every string they print is the one they printed
    there. Two placements changed, both on panel a (see panel_c): the row of group names under its
    tick labels, which collided with them by 1.47 pt on the measurement plate, now clears them, and
    its y-label is back at the default pad now that no panel sits to its left.
    """
    fig = plt.figure(figsize=(W_MM * MM, H3_MM * MM))
    # Panel a's top edge is set by the two-line note above it, which must stay on the canvas;
    # panel b sits 12 mm under panel a's axis title (its letter, and the strip's "difference" head).
    axA = fig.add_axes([0.115, 0.700, 0.805, 0.235])
    axB1 = fig.add_axes([0.115, 0.305, 0.440, 0.255])
    axB2 = fig.add_axes([0.720, 0.305, 0.200, 0.255])
    _pool, _dist, _k = MEAS.panel_c(axA)
    # A species key for panel a. On the measurement plate its colours were named by the species
    # strip of the panel beside it; on a plate of its own the panel needs its own key. Entries run in
    # the order the points are drawn, left to right: by divergence, and alphabetically within the
    # seven mammals that share 92 My, as panel_c dodges them. Words in ink, colour on the markers only.
    from matplotlib.lines import Line2D
    _rows = sorted(MEAS._load()[0]["forest"], key=lambda r: (r["divergence_my"], r["species"]))
    _h = [Line2D([], [], ls="none", marker="o", markersize=4.2, markerfacecolor=MEAS.S.species_color(r["species"]),
                 markeredgecolor="white", markeredgewidth=.4) for r in _rows]
    axA.legend(_h, [r["species"][:1].upper() + r["species"][1:] for r in _rows], loc="lower right", ncol=3, frameon=False,
               fontsize=SG.ANNOT, handletextpad=.25, columnspacing=1.1, labelspacing=.35,
               borderaxespad=.3)
    MEAS.panel_d(axB1, axB2)
    _LX = SG.letter_x(fig)
    for ax, letter in ((axA, "a"), (axB1, "b")):
        pos = ax.get_position()
        fig.text(_LX, min(pos.y1 + 0.016, SG.letter_top(fig, margin_pt=0.4)), letter,
                 fontsize=SG.PANEL, fontweight="bold", va="bottom", ha="left", color=SG.INK)
    # The note panel a carried on the measurement plate, right-aligned over the panel it measures.
    # The species count is the number of points the panel draws, not a typed 9.
    # The pooled value only. The Spearman test of AUROC against divergence it used to print beside it
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
    # Kept as a guard rather than as a fix. Haloed text renders as vector outlines and drops out
    # of the PDF text layer, so every haloed string used to need a transparent extractable twin.
    # House style has since removed the last path effect from text in both figures, so this now
    # adds nothing -- and it will go on adding nothing unless a halo comes back.
    restore_text_layer(fig)
    # GB: figures are "closely cropped to minimise the amount of white space surrounding the
    # illustration". Crop the PAGE at the save, to the drawn extent plus 3 mm, and only ever
    # upward from the BOTTOM edge: the width stays the declared 170 mm, the top edge does not
    # move, and no artist is re-placed. This is NOT the H_MM change FIGURES.md warns about --
    # nothing authored in millimetres from the top is touched. A plate already tighter than
    # 3 mm keeps its own margin (the clamp at 0).
    _y0 = min(max(0.0, fig.get_tightbbox(fig.canvas.get_renderer()).y0 - 3.0 * MM),
              fig.get_figheight())
    SG.tidy_minus(fig)   # ASCII hyphen -> U+2212 in numeric labels; see style_gb.tidy_minus
    fig.savefig(pdf, dpi=600, bbox_inches=matplotlib.transforms.Bbox(
        [[0.0, _y0], [fig.get_figwidth(), fig.get_figheight()]]))
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
    # The shared contract, applied before anything is drawn: Arial first (both figures went out as
    # ArialMT and must stay there), type 42 embedding, the one type scale, the neutral ink.
    plt.rcParams.update(SG.rc())
    _save(build_supplement(None), "FigureS7_measurement_supp")


if __name__ == "__main__":
    main()
