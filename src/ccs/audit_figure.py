"""Measured figure audit — the checks check_figure.py is blind to, in one place.

Run: python -m src.ccs.audit_figure <module> [<module> ...]
     e.g. python -m src.ccs.audit_figure fig3_rebuild fig4_trust

WHY THIS EXISTS. `check_figure.py` has eight recorded blind spots. The three
that matter most, and which this module covers:

  4. `Text.get_window_extent` EXCLUDES the bbox patch, so an opaque panel-letter chip painted over a
     neighbouring label is invisible. Chips are measured from their PATCH here.
  7. LINES vs TEXT were never compared at all. A full-width 0.55 pt strikethrough ran through the 5.3 pt
     numerals of six ledger rows in Fig 5 while the checker reported zero overlaps. Line SEGMENTS are
     tested against text boxes here.
  8. The line test needs CLIPPING. `Line2D.get_xydata()` returns raw data, not what was drawn: a CI band
     plotted past `ylim` is clipped by matplotlib but the raw points are not, so an unclipped test
     invents crossings that do not exist on the page. Two of three reported crossings on Fig 6 were this
     artefact. Segments are clipped to `get_clip_box()` before testing.

Also measured: out-of-canvas text, smallest type actually drawn, full-width dead bands, ink coverage,
and the page box (170 x 225 mm INCLUDING the legend).

NOT a pass/fail gate. Intentional marks WILL be reported — a strikethrough is supposed to cross its own
text. Read the list; do not automate on the count.
"""
import importlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.text import Text

# A referee on a cp437 console or with LC_ALL=C otherwise gets a traceback and a nonzero
# exit from a run that succeeded; build_tables.py even wrote its outputs first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DPI = 600                      # audit at the resolution that SHIPS, never at the default 100
PX2MM = 25.4 / DPI
GB_W, GB_H = 170.0, 225.0      # incl. legend


def _clip(a, b, box):
    """Liang-Barsky clip of a segment to an artist's clip box. Returns None if fully outside."""
    if box is None:
        return a, b
    t0, t1 = 0.0, 1.0
    d = b - a
    for pi, qi in ((-d[0], a[0] - box.x0), (d[0], box.x1 - a[0]),
                   (-d[1], a[1] - box.y0), (d[1], box.y1 - a[1])):
        if abs(pi) < 1e-12:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    return a + t0 * d, a + t1 * d


def _crosses(p, q, bb, inset):
    """Does segment p-q cross the text box, shrunk by `inset` px so grazes are ignored?"""
    x0, y0, x1, y1 = bb.x0 + inset, bb.y0 + inset, bb.x1 - inset, bb.y1 - inset
    if x1 <= x0 or y1 <= y0:
        return False
    t0, t1 = 0.0, 1.0
    d = q - p
    for pi, qi in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 <= t1


def audit(fig, name="figure", tol_mm=0.30, inset_mm=0.12, dead_mm=8.0):
    fig.set_dpi(DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * DPI

    texts, chips = [], []
    for t in fig.findobj(Text):
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible():
            continue
        try:
            # An Annotation's extent INCLUDES its leader line, so a callout in clear space measures as
            # a box stretching all the way to the point it annotates — reporting overlaps that are the
            # arrow, not the glyphs. Measure the text only (figure-qa-tooling blind spot 2).
            bb = (Text.get_window_extent(t, renderer=r)
                  if isinstance(t, matplotlib.text.Annotation) else t.get_window_extent(r))
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        # matplotlib leaves degenerate Text artists behind (zero-size, or a non-finite extent from an
        # unresolved transform). They draw nothing; counting them reports phantom out-of-canvas hits.
        if not np.all(np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1])):
            continue
        texts.append((s, bb, round(t.get_fontsize(), 2)))
        if t.get_bbox_patch() is not None:                 # blind spot 4
            chips.append((s, t.get_bbox_patch().get_window_extent(r)))

    segs = []
    for ln in fig.findobj(Line2D):                          # blind spot 7
        if not ln.get_visible() or ln.get_linestyle() == "None":
            continue
        try:
            pts = ln.get_transform().transform(ln.get_xydata())
        except Exception:
            continue
        box = ln.get_clip_box() if ln.get_clip_on() else None      # blind spot 8
        for a, b in zip(pts[:-1], pts[1:]):
            if not np.all(np.isfinite([*a, *b])):
                continue
            c = _clip(np.asarray(a, float), np.asarray(b, float), box)
            if c:
                segs.append(c)

    inset = inset_mm / PX2MM
    tol = tol_mm / PX2MM
    through = [(fz, s) for s, bb, fz in texts if any(_crosses(a, b, bb, inset) for a, b in segs)]

    def ov(a, b):
        dx = min(a.x1, b.x1) - max(a.x0, b.x0)
        dy = min(a.y1, b.y1) - max(a.y0, b.y0)
        return (dx, dy) if dx > 0 and dy > 0 else None

    boxes = [("text", s, bb) for s, bb, _ in texts] + [("chip", s, bb) for s, bb in chips]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            k1, s1, b1 = boxes[i]
            k2, s2, b2 = boxes[j]
            if s1 == s2 and k1 != k2:                      # a chip and its own letter
                continue
            o = ov(b1, b2)
            if o and o[0] > tol and o[1] > tol:
                overlaps.append((s1, s2, o[0] * PX2MM, o[1] * PX2MM))

    oob = [s for s, bb, _ in texts if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1]

    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].mean(axis=2)
    ink = arr < 245
    prof = ink.mean(axis=1)
    bands, i = [], 0
    while i < len(prof):
        if prof[i] < 0.002:
            j = i
            while j < len(prof) and prof[j] < 0.002:
                j += 1
            if (j - i) * PX2MM >= dead_mm:
                bands.append(((len(prof) - j) * PX2MM, (j - i) * PX2MM))
            i = j
        else:
            i += 1

    w_mm, h_mm = W * PX2MM, H * PX2MM
    print(f"\n=== {name}  {w_mm:.1f} x {h_mm:.1f} mm @ {DPI} dpi ===")
    print(f"  text/chip overlaps : {len(overlaps)}")
    for s1, s2, dx, dy in overlaps[:10]:
        print(f"      {s1[:34]!r} X {s2[:34]!r}  ({dx:.2f} x {dy:.2f} mm)")
    print(f"  line-through-text  : {len(through)}   (intentional strikethroughs count here)")
    for fz, s in through[:10]:
        print(f"      {fz} pt  {s[:56]!r}")
    print(f"  out of canvas      : {len(oob)}")
    for s in oob[:6]:
        print(f"      {s[:56]!r}")
    print(f"  smallest type      : {min(f for _, _, f in texts):.1f} pt")
    print(f"  dead bands >={dead_mm:.0f} mm  : {len(bands)}  ({sum(b for _, b in bands):.1f} mm total)")
    for y, b in bands[:6]:
        print(f"      {b:.1f} mm tall, {y:.1f} mm from the bottom")
    print(f"  ink coverage       : {ink.mean():.1%}")
    ok_w = "OK" if w_mm <= GB_W + 0.5 else f"OVER by {w_mm - GB_W:.1f}"
    ok_h = "OK" if h_mm <= GB_H + 0.5 else f"OVER by {h_mm - GB_H:.1f}"
    print(f"  GB {GB_W:.0f} x {GB_H:.0f} mm   : width {ok_w} · height {ok_h} "
          f"({GB_H - h_mm:.0f} mm left for the legend)")
    return {"overlaps": len(overlaps), "through": len(through), "oob": len(oob),
            "min_pt": min(f for _, _, f in texts), "dead_mm": sum(b for _, b in bands),
            "w_mm": w_mm, "h_mm": h_mm}


def main(mods):
    import matplotlib.pyplot as plt
    for m in mods:
        mod = importlib.import_module(f"src.ccs.{m}")
        # not every figure module returns its Figure (fig3_rebuild does not); fall back to the
        # current figure rather than making each module conform
        fig = mod.main() or plt.gcf()
        audit(fig, name=m)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fig6_final"])
