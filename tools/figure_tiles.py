# -*- coding: utf-8 -*-
"""Tile every figure at high resolution and screen each tile for collisions.

WHY. The span-based overlap scan compares text bounding boxes against each other. It is blind to
the case that has actually shipped in this project: a DRAWN element -- a marker, a band, a leader
line, a filled patch -- printed on top of text. Figure 1 shipped with a data marker over the BRCT
stats box and no scan saw it.

This does three things the earlier scan does not:

  1. text vs VECTOR DRAWING and text vs IMAGE, not only text vs text
  2. renders the page as a grid of tiles at 600 dpi so every square millimetre can be looked at
  3. reports which tile each collision falls in, so the flagged tile can be opened directly

    python tools/figure_tiles.py                  # screen only
    python tools/figure_tiles.py --render         # + write every tile
    python tools/figure_tiles.py --render --only Figure2

Tiles are written under reports/figures/_tiles/<figure>/r<row>c<col>.png and are NOT part of the
deposit; the directory is disposable.
"""
import argparse
import os
import shutil
import sys

import fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SUB = "reports/submission"
OUT = "reports/figures/_tiles"
DPI = 600
COLS, ROWS = 4, 6            # 24 tiles per figure; 8 figures -> ~200 tiles

# A drawing that merely *contains* text is a background panel, not a collision. Only flag when a
# drawing overlaps text partially -- that is what printing-over looks like.
CONTAIN_SLACK = 1.0

# Flags that survive classification and were checked BY EYE at 560 dpi, one by one.
# Each is a bounding-box artifact, not a visible collision:
#   Figure6_curator     12  three-line stats blocks with tight leading; the boxes overlap, the glyphs do not
# The gate reports a DEVIATION from these, not their existence -- a new flag is a real finding, and
# a disappeared one means something moved. Re-verify and update the number if you change a figure.
# EXPECTED is keyed on the builder-output names in tools/stage_figures.py, which is authoritative.
EXPECTED = {"Figure_BRCA1": (0, 0), "Figure7_atlas": (0, 0), "Figure8_readout": (0, 0),
            "FigureS7_measurement_supp": (0, 0),
            "Figure5_reach": (0, 0), "Figure3_blindspot": (0, 0), "Figure4_trust": (0, 0),
            "Figure6_curator": (12, 0), "figS1_missing_panels": (0, 0)}


def rect_of(d):
    r = d.get("rect")
    return fitz.Rect(r) if r is not None else None


def spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            d = l.get("dir", (1.0, 0.0))
            for s in l.get("spans", []):
                if s["text"].strip():
                    out.append((fitz.Rect(s["bbox"]), s["text"].strip(), s["size"], d))
    return out


def benign(a, b, sa, sb, da, db):
    """Classify a text-vs-text overlap that is a property of measurement, not of the artwork.

    Three shapes account for essentially every flag on this figure set, and all three were checked
    by eye at 560 dpi:

      rotated   axis-aligned bounding boxes of ROTATED text overlap even when the glyphs are far
                apart -- vertical species labels sitting in clean white gaps can still
                report 0.07.
      stacked   consecutive lines of ONE text object: the ascender box of one line touches the
                descender box of the line above. Unavoidable, and correct typography.
      column    adjacent rows of a table or category axis, same size, same left edge.

    Anything else is returned False and reported, which is what the gate is for.
    """
    if abs(da[1]) > 1e-6 or abs(db[1]) > 1e-6:
        return "rotated"
    same_size = abs(sa - sb) < 0.35
    # stacked lines / rows: strong horizontal agreement, vertical adjacency
    x_ov = min(a.x1, b.x1) - max(a.x0, b.x0)
    narrower = min(a.width, b.width)
    if same_size and narrower > 0 and x_ov / narrower > 0.30:
        gap = max(a.y0, b.y0) - min(a.y1, b.y1)
        # gap must be near zero from EITHER side. Only an upper bound would make this classifier
        # dangerous: Figure 7's axis label printed straight across its tick labels at a similar
        # size, and with no floor that real collision would have been absorbed as "stacked".
        # Genuine line-boundary contact overlaps by a few percent of the line height, not a third.
        if -0.18 * max(a.height, b.height) < gap < 0.45 * max(a.height, b.height):
            return "stacked"
    if same_size and abs(a.x0 - b.x0) < 1.5:
        return "column"
    return None


def screen(path, name, render, page_rect):
    d = fitz.open(path)
    page = d[0]
    sp = spans(page)
    flags = []

    # --- text vs text -------------------------------------------------------------
    classified = {}
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            a, b = sp[i][0], sp[j][0]
            it = a & b
            if it.is_empty or it.get_area() <= 0:
                continue
            frac = it.get_area() / min(a.get_area(), b.get_area())
            if frac <= 0.02:
                continue
            why = benign(a, b, sp[i][2], sp[j][2], sp[i][3], sp[j][3])
            if why:
                classified[why] = classified.get(why, 0) + 1
                continue
            flags.append(("text/text", frac, sp[i][1][:30], sp[j][1][:30], it))

    # --- text vs vector drawing --------------------------------------------------
    draws = []
    try:
        for dr in page.get_drawings():
            r = rect_of(dr)
            if r and r.get_area() > 0:
                fill = dr.get("fill")
                # Luminance decides whether a fill can actually hide a glyph. A white halo
                # (path_effects withStroke) and a pale tint band both overlap text constantly and
                # neither obscures it -- flagging them buries the one case that matters under
                # dozens that do not. Only a dark or saturated fill can occlude dark text.
                lum = None
                if fill is not None:
                    try:
                        lum = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
                    except Exception:
                        lum = 0.0
                    # Alpha matters as much as colour: a CI ribbon drawn in saturated blue at
                    # alpha 0.16 reads as pale on the page but records a dark fill. Composite it
                    # against white before judging, or every ribbon in the paper is a false alarm.
                    op = dr.get("fill_opacity")
                    if op is None:
                        op = 1.0
                    lum = lum * op + 1.0 * (1.0 - op)
                draws.append((r, fill is not None, lum, dr.get("width") or 0))
    except Exception:
        pass
    for r_txt, txt, size, _d in sp:
        if r_txt.get_area() <= 0:
            continue
        for r_dr, filled, lum, wid in draws:
            if not filled:
                continue
            if lum is None or lum > 0.62:          # white halo / pale tint: cannot hide dark text
                continue
            it = r_txt & r_dr
            if it.is_empty or it.get_area() <= 0:
                continue
            # background panel behind the text: fully contains it -> fine
            if (r_dr.x0 <= r_txt.x0 + CONTAIN_SLACK and r_dr.x1 >= r_txt.x1 - CONTAIN_SLACK
                    and r_dr.y0 <= r_txt.y0 + CONTAIN_SLACK and r_dr.y1 >= r_txt.y1 - CONTAIN_SLACK):
                continue
            frac = it.get_area() / r_txt.get_area()
            if frac > 0.18:
                flags.append(("text/graphic", frac, txt[:30], "filled shape", it))

    # --- text vs raster image ----------------------------------------------------
    try:
        for img in page.get_image_rects(full=True) if hasattr(page, "get_image_rects") else []:
            ir = fitz.Rect(img)
            for r_txt, txt, size, _d in sp:
                it = r_txt & ir
                if not it.is_empty and it.get_area() / max(1e-6, r_txt.get_area()) > 0.35:
                    flags.append(("text/image", it.get_area() / r_txt.get_area(),
                                  txt[:30], "raster", it))
    except Exception:
        pass

    # --- tile rendering ----------------------------------------------------------
    R = page.rect
    tw, th = R.width / COLS, R.height / ROWS
    tiles_written = 0
    if render:
        tdir = os.path.join(OUT, name)
        os.makedirs(tdir, exist_ok=True)
        for r in range(ROWS):
            for c in range(COLS):
                clip = fitz.Rect(R.x0 + c * tw, R.y0 + r * th,
                                 R.x0 + (c + 1) * tw, R.y0 + (r + 1) * th)
                page.get_pixmap(dpi=DPI, clip=clip).save(
                    os.path.join(tdir, "r%dc%d.png" % (r, c)))
                tiles_written += 1
    d.close()

    def tile_of(rect):
        c = min(COLS - 1, max(0, int((rect.x0 - R.x0) / tw)))
        r = min(ROWS - 1, max(0, int((rect.y0 - R.y0) / th)))
        return "r%dc%d" % (r, c)

    return flags, tiles_written, tile_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="write every tile as a PNG")
    ap.add_argument("--only", help="substring of the figure filename")
    ap.add_argument("--clean", action="store_true", help="remove the tile directory first")
    a = ap.parse_args()

    if a.clean and os.path.isdir(OUT):
        shutil.rmtree(OUT)

    # os.listdir raises before any guard below if the directory is simply absent, which is the
    # ordinary case from a clean extraction.
    figs = sorted(f for f in (os.listdir(SUB) if os.path.isdir(SUB) else [])
                  if f.endswith(".pdf") and ("Figure" in f or "figure" in f))
    if a.only:
        figs = [f for f in figs if a.only.lower() in f.lower()]

    # Refusing to report a pass over zero figures. reports/submission/ holds BUILT plates and is not
    # part of the code deposit, so from a clean extraction this list is empty. Without this guard the
    # loop below would run zero times, leave `deviations` empty and print "every figure matches its
    # by-eye-verified flag count" with exit 0. tools/check_figure_clearance.py also exits 3 on
    # the same condition.
    if not figs:
        print("NO FIGURE PDFs FOUND in %s (cwd %s)." % (SUB, os.getcwd()))
        print("This tool measures BUILT plates; %s appears only after staging and is not part of" % SUB)
        print("the code deposit. Stage the figures first, e.g. python tools/stage_figures.py,")
        print("then re-run. Refusing to report a pass over zero figures.")
        sys.exit(3)

    grand_t, grand_g, tiles = 0, 0, 0
    deviations, unverified = [], []
    print("tiling %dx%d at %d dpi\n" % (COLS, ROWS, DPI))
    print("  %-32s %7s %9s %8s  %s" % ("figure", "txt/txt", "txt/graph", "tiles", "worst tiles"))
    for f in figs:
        name = f.replace(".pdf", "")
        d = fitz.open(os.path.join(SUB, f))
        pr = d[0].rect
        d.close()
        flags, nt, tile_of = screen(os.path.join(SUB, f), name, a.render, pr)
        tt = [x for x in flags if x[0] == "text/text"]
        tg = [x for x in flags if x[0] != "text/text"]
        grand_t += len(tt)
        grand_g += len(tg)
        tiles += nt
        exp = EXPECTED.get(name)
        note = ""
        if exp is None:
            # A figure the baseline does not cover is not a pass. Its flags were never inspected by
            # eye, so "matches its by-eye-verified flag count" cannot be said of it.
            note = "** NOT IN THE VERIFIED BASELINE **"
            unverified.append((name, (len(tt), len(tg))))
        if exp is not None:
            if (len(tt), len(tg)) == exp:
                note = "as verified"
            else:
                note = "** CHANGED from verified %d/%d **" % exp
                deviations.append((name, (len(tt), len(tg)), exp))
        worst = sorted(flags, key=lambda x: -x[1])[:3]
        print("  %-32s %7d %9d %8d  %-18s %s"
              % (f, len(tt), len(tg), nt, note,
                 ", ".join("%s(%.2f)" % (tile_of(x[4]), x[1]) for x in worst)))
        for kind, frac, s1, s2, rect in sorted(tg, key=lambda x: -x[1])[:6]:
            print("        %-13s %.2f  %-32s over/under %s   tile %s"
                  % (kind, frac, s1, s2, tile_of(rect)))

    print("\n%d text/text, %d text/graphic-or-image, %d tiles written"
          % (grand_t, grand_g, tiles))
    # Fail on CHANGE, not on existence. Every surviving flag was inspected by eye; a raw non-zero
    # count would make this gate permanently red, which is the same as switching it off.
    if unverified:
        print("\n%d figure(s) carry NO by-eye-verified baseline entry:" % len(unverified))
        for nm, got in unverified:
            print("  %-30s %d text/text, %d text/graphic — never inspected" % (nm, got[0], got[1]))
        print("  EXPECTED is keyed on builder-output names (Figure7_atlas, Figure4_trust, ...).")
        print("  Re-key it, or add these figures to it once their flags have been inspected.")
    if deviations:
        print("\n%d figure(s) DEVIATE from the by-eye-verified baseline:" % len(deviations))
        for nm, got, exp in deviations:
            print("  %-30s got %d/%d, verified %d/%d" % (nm, got[0], got[1], exp[0], exp[1]))
    if deviations or unverified:
        sys.exit(1)
    print("every figure matches its by-eye-verified flag count")
    sys.exit(0)


if __name__ == "__main__":
    main()
