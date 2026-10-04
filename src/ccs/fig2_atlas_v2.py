"""Figure 2 composite (reworked) — The Cross-Species Reliability Atlas, 8 dense annotated panels.
Descriptive finding-stating titles, a key-numbers strip, a shared legend, tighter grid."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import fig2_hero as HERO
import fig2_bc as BC
import fig2_defgh as DG
import fig2_style as S

TITLES = {   # (letter, title) — narrow middle-band panels use 2 lines to fit their 3-col width
    "A": ("A", "Discrimination rings the amniote tree — no distance grading (underpowered)"),
    "B": ("B", "Positives ride a heavier right tail — even as they thin"),
    "C": ("C", "ROCs hug the corner —\nnever collapse to chance"),
    "D": ("D", "Discrimination melts to the\n0.824 missense floor"),
    "H": ("H", "Context lifts every stratum\nbut the missense bedrock"),
    "G": ("G", "Ranking ≠ separation\n(human: high AUROC, lowest d)"),
    "E": ("E", "No detectable distance grading (n = 9, underpowered — one deep anchor)"),
    "F": ("F", "Matches conservation overall — trails on human, leads on goat & cattle"),
}


def _title(fig, ss, key, pad=0.008, gap=0.016, fs_name=7.0):
    """Place panel letter + descriptive title in FIGURE coords, anchored to the gridspec CELL
    (subplotspec) rather than the axes. The subplotspec position is derived from the GridSpec
    layout (subplotpars + height_ratios/hspace) and is INDEPENDENT of any per-axes aspect
    adjustment, so aspect-equal panels (A, C) — whose axes box shrinks and re-centres inside its
    cell — no longer drag their letter/title off the band baseline. All panels sharing a gridspec
    row therefore sit on ONE common baseline. va='bottom' anchors the bottom of the text block, so
    1- and 2-line titles align at the same baseline (2-line titles simply grow upward)."""
    lab, name = TITLES[key]
    two = "\n" in name
    pos = ss.get_position(fig)                 # cell rect (figure fraction), aspect-independent
    x0, yb = pos.x0, pos.y1 + pad              # cell left edge + common baseline for this band
    fig.text(x0, yb, lab, fontsize=12, fontweight="bold", va="bottom", ha="left")
    fig.text(x0 + gap, yb, name, fontsize=(6.6 if two else fs_name), fontweight="bold",
             va="bottom", ha="left", color=S.EVO2, linespacing=1.1)


def main():
    fig = plt.figure(figsize=(12.2, 15.2))
    gs = GridSpec(3, 12, height_ratios=[46, 31, 25], hspace=0.42, wspace=1.0,
                  left=0.055, right=0.965, top=0.905, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0:7]); HERO.build(axA)
    axB = fig.add_subplot(gs[0, 7:12]); BC.panel_B(axB)
    axC = fig.add_subplot(gs[1, 0:3]); axC.axis("off")             # container; letter/title anchor to THIS cell
    cin = GridSpecFromSubplotSpec(2, 1, subplot_spec=axC.get_subplotspec(), height_ratios=[2.0, 1.15], hspace=0.42)
    BC.panel_C(fig.add_subplot(cin[0]), fig.add_subplot(cin[1]))
    axD = fig.add_subplot(gs[1, 3:6]); DG.panel_D(axD)
    axG = fig.add_subplot(gs[1, 6:9]); DG.panel_G(axG)          # C-D-G-H reads left-to-right (alphabetical)
    axH = fig.add_subplot(gs[1, 9:12]); DG.panel_H(axH)
    axE = fig.add_subplot(gs[2, 0:6]); DG.panel_E(axE)
    axF = fig.add_subplot(gs[2, 6:12]); DG.panel_F(axF)
    for k, ax in [("A", axA), ("B", axB), ("C", axC), ("D", axD), ("H", axH), ("G", axG), ("E", axE), ("F", axF)]:
        _title(fig, ax.get_subplotspec(), k)

    # ---- title + subtitle ----
    fig.suptitle("The Cross-Species Discrimination Atlas — Evo 2's zero-shot pathogenic-vs-benign separation across the ~319 My mammal–bird split",
                 x=0.055, y=0.975, ha="left", fontsize=13.0, fontweight="bold")
    fig.text(0.055, 0.957, "9 species (8 placental mammals + 1 bird) · OMIA disease-gene variants, coding + non-coding · Evo 2-40B, 8192-bp mean-LL, zero-shot · "
             "calibration/trust-layer: see Fig 4", fontsize=8.4, color=S.CAP, style="italic")

    # ---- KEY-NUMBERS STRIP (colour-coded chips = mini table-of-contents) ----
    chips = [("pooled AUROC 0.956 [.945,.965]", S.EVO2), ("n = 3,506", S.INK),
             ("readout lift +0.071 [.057,.084]", "#2E8B57"), ("missense floor 0.824", S.PATHO),
             ("distance ρ = −0.27 (ns)", "#9AA0AA"), ("Cohen's d median ≈ 1.65 (well-powered spp.)", S.MUTED)]
    x = 0.055
    for txt, col in chips:
        t = fig.text(x, 0.94, txt, fontsize=7.0, color="white", fontweight="bold", va="center", ha="left",
                     bbox=dict(boxstyle="round,pad=0.32", fc=col, ec="none"))
        fig.canvas.draw(); bb = t.get_window_extent(); x += bb.width / fig.bbox.width + 0.012

    # ---- SHARED LEGEND (bottom): clades + silhouettes + ember/slate + conventions ----
    ly = 0.045
    fig.text(0.055, ly, "clades:", fontsize=6.6, fontweight="bold", color=S.INK, va="center")
    cx = 0.10
    for sp in S.LEAF_ORDER:
        im = OffsetImage(S.load_silhouette(sp, S.species_color(sp), longest=70), zoom=0.10)
        fig.add_artist(AnnotationBbox(im, (cx, ly), frameon=False, xycoords="figure fraction"))
        cx += 0.037
    fig.text(0.055, ly - 0.018, "pathogenic (ember) / benign (slate)  ·  arc base = 0.5 chance, lighter tip = 1001→8192 readout gain  ·  "
             "95% CIs by variant bootstrap (per-panel n shown)  ·  small pathogenic n → Cohen's d upward-biased",
             fontsize=5.9, color=S.MUTED, va="center")
    fig.text(0.055, ly - 0.033, "D/H 'non-coding' stratum previews the regulatory arm (Fig 3)  ·  atlas pooled 0.956 (n=3,506) is a superset of the type-match parent ALL 0.962 (n=2,881; coding+non-coding)  ·  D's melt is prevalence-independent (rank-based AUROC)",
             fontsize=5.9, color=S.MUTED, va="center")

    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig("reports/figures/Figure2_atlas.png", bbox_inches="tight", dpi=190)     # working preview
    # dpi= sets the resolution of RASTERS embedded in the PDF (the PhyloPic silhouettes).
    # Without it they embed at ~100 dpi; measured widths were as low as 4-11 px.
    fig.savefig("reports/figures/Figure2_atlas.pdf", bbox_inches="tight", dpi=600)    # vector (submission master)
    if os.environ.get("FIG2_HIRES"):                                                  # submission-grade rasters
        fig.savefig("reports/figures/Figure2_atlas_600dpi.png", bbox_inches="tight", dpi=600)
        fig.savefig("reports/figures/Figure2_atlas_4k.png", bbox_inches="tight", dpi=315)   # ~3840 px wide
    print("wrote reports/figures/Figure2_atlas  (reworked 8-panel composite)")


if __name__ == "__main__":
    main()
