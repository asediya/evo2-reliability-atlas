# -*- coding: utf-8 -*-
"""Report TEXT THAT SITS ON DATA INK -- the one defect pdfcheck.py cannot see.

pdfcheck compares text spans against other text spans. A caption dropped into the middle of a
scatter, a key laid over its own curves, an n-count on top of a hexbin: every one of those is a
clean pass there, because nothing it looks at overlaps.

METHOD. The page is rendered twice. Once as it is. Once with every text character redacted but
images and line art left alone (`fill=None`, so redaction paints nothing). The second render is
therefore the figure's INK WITHOUT ITS WORDS. For each text span we then count, inside its own
box, how many pixels of that ink-only render are not paper-white. That fraction is what a reader
sees behind the words, measured rather than guessed, and it needs no rule about what counts as a
curve or a marker.

    python tools/inkunder.py FIG.pdf [...] [--dpi 300] [--warn 0.04] [--fail 0.10]
"""
import argparse
import os
import sys

import numpy as np
import pymupdf as fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _ink_only(path, dpi):
    """Render the page with its text removed and everything else kept."""
    d = fitz.open(path)
    p = d[0]
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    # fill=None: the redaction removes the glyphs and paints nothing in their place,
                    # so whatever was drawn under them survives. A default redaction fills the box
                    # with white and would report every span as sitting on clean paper.
                    p.add_redact_annot(fitz.Rect(s["bbox"]), fill=None)
    # Keep images and line art; remove only text.
    p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                       graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    pm = p.get_pixmap(dpi=dpi)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    d.close()
    full = fitz.open(path)[0].get_pixmap(dpi=dpi)
    b = np.frombuffer(full.samples, dtype=np.uint8).reshape(full.height, full.width, full.n)
    # RGB is kept for the visibility test and only collapsed for the coverage/texture ones.
    # min-over-channels was wrong for visibility: dark ink (min 21) on a yellow cell (min 30)
    # differs by 9, so two perfectly legible cells in Figure 1 were reported as painted over.
    return a[..., :3], b[..., :3], dpi / 72.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--warn", type=float, default=0.04)
    ap.add_argument("--fail", type=float, default=0.10)
    ap.add_argument("--busy", type=float, default=0.03,
                    help="min edge density under the box; separates a flat cell fill from data")
    ap.add_argument("--white", type=int, default=245, help="pixels at or above this are paper")
    a = ap.parse_args()

    worst = 0.0
    n_hidden = 0            # summed over every file: a hidden span in any PDF fails the run
    for path in a.pdfs:
        ink, full, sc = _ink_only(path, a.dpi)
        H, W = ink.shape[:2]
        rows = []
        hidden = []
        for b in fitz.open(path)[0].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"].strip()
                    if not t:
                        continue
                    x0, y0, x1, y1 = (int(round(v * sc)) for v in s["bbox"])
                    x0, y0 = max(x0, 0), max(y0, 0)
                    x1, y1 = min(x1, W), min(y1, H)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    # HIDDEN TEXT. A span still in the PDF but painted over by something drawn
                    # later -- a neighbouring panel's opaque background, a filled patch -- extracts
                    # perfectly and collides with nothing, so every other gate passes it. If the
                    # render WITH text is identical to the render WITHOUT it over this box, the
                    # span put no pixels on the page: the reader never sees it.
                    sub_i = ink[y0:y1, x0:x1].astype(np.int16)
                    sub_f = full[y0:y1, x0:x1].astype(np.int16)
                    # per-channel, then max: a colour change in ANY channel means the text painted.
                    glyph = (np.abs(sub_f - sub_i).max(axis=2) > 8).any(axis=0)
                    # abs, not "<": white text reversed out of a dark cell LIGHTENS the page.
                    if not glyph.any():
                        hidden.append((t, "wholly"))
                        continue
                    # NOT ATTEMPTED: partial clipping, where only a span's last glyphs are
                    # painted over (Figure 6 shipped "variant-leve"). Every rule tried here fires on
                    # the trailing advance a short span's bbox carries anyway -- a 2.6 pt pad on
                    # "0.97" is indistinguishable from a 2.5 pt clipped "l" on a 271 pt line. An
                    # unreliable gate is worse than none, so this reports only WHOLLY hidden text.
                    box = ink[y0:y1, x0:x1].min(axis=2).astype(np.int16)
                    cover = float((box < a.white).mean())
                    # BUSY, not just covered. A number printed inside its own heatmap cell sits on
                    # 100% ink and is perfectly legible, because that ink is FLAT. What ruins text
                    # is TEXTURED ink -- scattered points, curves, hatch -- so the discriminator is
                    # edge density under the box, not coverage. A solid fill has no internal edges.
                    if box.shape[0] < 2 or box.shape[1] < 2:
                        continue
                    gx = np.abs(np.diff(box, axis=1))[:-1, :]
                    gy = np.abs(np.diff(box, axis=0))[:, :-1]
                    busy = float(((gx > 24) | (gy > 24)).mean())
                    if cover >= a.warn and busy >= a.busy:
                        rows.append((busy, cover, t))
        rows.sort(reverse=True)
        n_fail = sum(1 for bz, _, _ in rows if bz >= a.fail)
        print("%s: %d span(s) on textured ink, %d over %.0f%% busy, %d HIDDEN"
              % (os.path.basename(path), len(rows), n_fail, 100 * a.fail, len(hidden)))
        for t, why in hidden[:8]:
            print("    PAINTED OVER (%s): %r" % (why, t[:66]))
        for bz, cv, t in rows[:12]:
            print("    busy %4.1f%%  cover %5.1f%%   %r" % (100 * bz, 100 * cv, t[:70]))
        worst = max(worst, rows[0][0] if rows else 0.0)
        n_hidden += len(hidden)
    # `n_hidden` counts WHOLLY hidden text in every file -- the worst outcome this tool can find -- so it fails the run
    # on its own: `worst`, from which the exit code is otherwise built, comes only from
    # the busy-span rows.
    if n_hidden:
        print("  %d string(s) are WHOLLY hidden beneath ink; that is a failure whatever the "
              "busy-span score is." % n_hidden)
        return 1
    return 1 if worst >= a.fail else 0


if __name__ == "__main__":
    sys.exit(main())
