"""Full-figure collision audit: text-vs-text, text-over-ANYTHING, and canvas clipping.

The earlier checker only compared text boxes with other text boxes and with line/scatter vertices, so it
reported "clean" while labels sat on rasters, filled areas and patches. This renders the figure twice —
once normally, once with every text artist hidden — and measures how much ink lies underneath each text
box. That catches text over images, patches, spans, fills, lines and markers alike, with no per-artist
special-casing.

Run: python -m src.ccs.check_figure <module>
"""
import sys, os, importlib
import numpy as np
import matplotlib
import matplotlib.text as mtext
matplotlib.use("Agg")

MIN_MM = 0.35          # ignore hairline text-text touches
# A 4.5% threshold never fired for text over THIN marks -- a legend sitting across three 1 px curves, a
# callout on a dashed line, labels over hairline underscores all scored 1-3% and passed while being
# obviously wrong on screen. 1.2% is roughly "one hairline crosses this box", which is what we want to see.
INK_FRAC = 0.012
DPI = 200


def _texts(fig):
    out = []
    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            out.append((f"ax{i}", t))
        if ax.axison:
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                out.append((f"ax{i}:tick", lab))
            for o in (ax.xaxis.label, ax.yaxis.label):
                out.append((f"ax{i}:axlabel", o))
        lg = ax.get_legend()
        if lg is not None:
            for lt in lg.get_texts():
                out.append((f"ax{i}:legend", lt))
    for t in fig.texts:
        out.append(("figure", t))
    return [(w, t) for w, t in out
            if t.get_visible() and (t.get_text() or "").strip()
            and (getattr(t, "axes", None) is None or t.axes.get_visible())]


def audit(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for w, t in _texts(fig):
        try:
            # Annotation.get_window_extent includes the LEADER LINE, so a callout in clear space reads as
            # heavily inked because its box stretches down to the curve it points at. Measure the glyphs
            # only -- the leader is meant to cross the plot.
            bb = (mtext.Text.get_window_extent(t, renderer=r)
                  if isinstance(t, mtext.Annotation) else t.get_window_extent(renderer=r))
            # A text with bbox=dict(...) (the panel-letter chips) paints an opaque patch far larger than
            # its glyphs, and Text.get_window_extent EXCLUDES that patch -- so a 3.1 x 4.1 mm solid chip
            # was modelled as a ~1.5 x 2.5 mm glyph box and its overprinting went unseen.
            bp = t.get_bbox_patch()
            if bp is not None:
                try:
                    bb = bb.union([bb, bp.get_window_extent(r)])
                except Exception:
                    pass
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            items.append((w, t.get_text().strip(), bb))
    dpi = fig.dpi
    W, H = fig.canvas.get_width_height()

    # ---- 1. text vs text ----
    tt = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            w1, s1, b1 = items[i]; w2, s2, b2 = items[j]
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            mx, my = dx / dpi * 25.4, dy / dpi * 25.4
            if mx > MIN_MM and my > MIN_MM:
                tt.append((mx * my, mx, my, w1, s1, w2, s2))
    tt.sort(reverse=True)

    # ---- 2. clipped by the canvas ----
    clip = []
    for w, s, bb in items:
        over = max(0.0, -bb.x0) + max(0.0, bb.x1 - W) + max(0.0, -bb.y0) + max(0.0, bb.y1 - H)
        if over > 1.0:
            clip.append((over / dpi * 25.4, w, s))
    clip.sort(reverse=True)

    # ---- 3. text over ANY ink: render once with all text hidden, then sample under each box ----
    vis = [(t, t.get_visible()) for _, t in _texts(fig)]
    for t, _ in vis:
        t.set_visible(False)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.int16)
    for t, v in vis:
        t.set_visible(v)
    fig.canvas.draw()
    # background = the modal colour (white/card); anything materially different is ink
    base = np.array([255, 255, 255], np.int16)
    diff = np.abs(bg - base).sum(axis=2)
    ink = diff > 26
    over = []
    for w, s, bb in items:
        x0, x1 = int(max(0, bb.x0)), int(min(W, bb.x1))
        y0, y1 = int(max(0, H - bb.y1)), int(min(H, H - bb.y0))     # buffer is top-down
        if x1 <= x0 or y1 <= y0:
            continue
        patch = ink[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        f = patch.mean()
        if f > INK_FRAC:
            over.append((f, w, s, (x1 - x0) / dpi * 25.4, (y1 - y0) / dpi * 25.4))
    over.sort(reverse=True)
    return items, tt, clip, over


def main():
    # Import the figure module WITH its package context. Loading it by bare name leaves
    # __package__ empty, so any `from .style_gb import ...` inside it raises ImportError
    # (fig4_trust, fig5_final, fig6_final and the fig2_* modules all use one).
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    _name = sys.argv[1] if len(sys.argv) > 1 else "fig4_trust"
    mod = importlib.import_module(_name if "." in _name else "ccs." + _name)
    fig = mod.main()
    # AUDIT AT THE RESOLUTION THAT SHIPS. DPI was declared and never used; audit() read fig.dpi, which is
    # 100 for the figure main() returns, while the deposited assets are 600. The same unmodified clip test
    # reports 0 clipped at dpi 100 and 1 clipped (0.88 mm outside) at dpi 600 -- which is how a caption
    # with a glyph sliced in half was reported clean.
    fig.set_dpi(600)
    fig.canvas.draw()
    items, tt, clip, over = audit(fig)
    print(f"visible text objects : {len(items)}")
    print(f"TEXT-vs-TEXT overlaps: {len(tt)}")
    for _, mx, my, w1, s1, w2, s2 in tt[:12]:
        print(f"   [{mx:5.1f} x {my:4.1f} mm] {w1:14s} {s1[:40]!r}")
        print(f"                        vs {w2:14s} {s2[:40]!r}")
    print(f"CLIPPED BY CANVAS    : {len(clip)}")
    for mm, w, s in clip[:12]:
        print(f"   [{mm:5.1f} mm outside] {w:14s} {s[:56]!r}")
    print(f"TEXT OVER OTHER INK  : {len(over)}   (>{INK_FRAC:.0%} of the box is non-background)")
    for f, w, s, mx, my in over[:40]:
        print(f"   [{f:5.1%} ink, {mx:4.1f}x{my:4.1f} mm] {w:14s} {s[:46]!r}")
    return 1 if (tt or clip) else 0


if __name__ == "__main__":
    sys.exit(main())
