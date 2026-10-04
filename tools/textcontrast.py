# -*- coding: utf-8 -*-
"""WCAG contrast for every text span, measured against the ink ACTUALLY BEHIND IT.

Auditing figure text against a white page is wrong wherever the text sits on something: a value
printed in a heatmap cell, a percentage reversed out of a bar, a label on a shaded band. Measured
that way, white-on-dark scores 1.0 (an apparent catastrophe) and dark-on-yellow scores 12 (an
apparent pass) when the real numbers are the other way round.

This renders the page with the text redacted (see tools/inkunder.py) and takes the MEDIAN colour of
that ink-only render inside each span's own box as the background. The result is what a reader's
eye receives.

    python tools/textcontrast.py FIG.pdf [...] [--min 4.5] [--dpi 300]

WCAG 2.1: 4.5:1 for body text, 3.0:1 for text at 18 pt (or 14 pt bold) and above. Nothing in these
figures is that large, so 4.5 is the bar everywhere.
"""
import argparse
import os
import sys

import numpy as np
import pymupdf as fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _lin(c):
    c = c / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lum(rgb):
    r, g, b = (_lin(np.asarray(rgb, dtype=float))).tolist()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(fg, bg):
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _ink_only(path, dpi):
    d = fitz.open(path)
    p = d[0]
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    p.add_redact_annot(fitz.Rect(s["bbox"]), fill=None)
    p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                       graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    pm = p.get_pixmap(dpi=dpi)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    d.close()
    return a[..., :3], dpi / 72.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--min", type=float, default=4.5)
    ap.add_argument("--show", type=int, default=14)
    a = ap.parse_args()
    worst_overall = 99.0
    for path in a.pdfs:
        ink, sc = _ink_only(path, a.dpi)
        H, W = ink.shape[:2]
        bad = []
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
                    patch = ink[y0:y1, x0:x1].reshape(-1, 3)
                    bg = np.median(patch, axis=0)
                    fg = ((s["color"] >> 16) & 255, (s["color"] >> 8) & 255, s["color"] & 255)
                    r = _ratio(fg, bg)
                    if r < a.min:
                        bad.append((r, "#%02X%02X%02X" % fg,
                                    "#%02X%02X%02X" % tuple(int(v) for v in bg), t))
        bad.sort()
        worst = bad[0][0] if bad else 99.0
        worst_overall = min(worst_overall, worst)
        print("%s: %d span(s) below %.1f:1%s"
              % (os.path.basename(path), len(bad), a.min,
                 ("   worst %.2f:1" % worst) if bad else ""))
        for r, fg, bg, t in bad[:a.show]:
            print("    %5.2f:1  ink %s on %s   %r" % (r, fg, bg, t[:52]))
    return 1 if worst_overall < a.min else 0


if __name__ == "__main__":
    sys.exit(main())
