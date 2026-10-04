# -*- coding: utf-8 -*-
"""Check that every panel title clears the text above it.

WHY THIS EXISTS. tools/figure_tiles.py measures bounding-box INTERSECTION, so two labels can sit
0.7 pt apart and pass every automated check while reading, to a human, as one run-together block.
That is what happened when Figure 2's early-retrieval strip was given its own panel letter: its
title landed 0.7 pt under the ROC panel's x-axis label. The boxes never touched, so nothing
complained; the reviewer's eye caught it.

The naive generalisation -- flag any two near-touching labels -- is useless here, because these
figures are full of deliberate stacks (a bold species name directly over its sample count, a tick
label under an axis label). Those sit 0.5-2.8 pt apart by design and flagging them trains a reader
to ignore the tool.

So this checks the one structural case that actually goes wrong: a PANEL TITLE, which by
construction opens a new panel, must not crowd whatever sits above it. Panel titles are the bold
runs emitted by _title() in the figure builders -- a lone letter followed by a coloured bold phrase.

KNOWN LIMITS, stated so the output is not over-trusted. Title detection is heuristic (a lone bold
letter with a bold phrase on the same baseline) and misses some layouts -- it found 3 of that plate's
5 titles. And a small gap is not automatically a defect: a bold title under a lighter caption reads
as separate at 1.5 pt, while the two same-weight panel-level labels that prompted this ran together
at 0.7 pt. The number tells you where to look; your eye decides. This never fails a build.

    python tools/check_figure_clearance.py [--min PT]
"""
import argparse
import glob
import os
import re
import sys

import fitz

# Panel titles carry Greek and typographic characters. When this gate's output is captured rather
# than sent to a terminal, Windows selects cp1252 and the first Delta raises UnicodeEncodeError, so
# the gate dies part-way through the figure list. It had already reported four figures and never
# reached the other four, which in a summary reads as a pass for everything it never checked.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# This module used to leave SUB as a bare relative path with no anchoring, so its input set
# depended on the caller's cwd. Run from anywhere but the repo root it globbed nothing, printed
# "every panel title clears 6.0 pt" over an empty file list, and exited 0 -- and run_all_gates.py
# recorded that as PASS. A gate that reports a pass while checking zero figures is worse than no
# gate, so this now anchors itself and refuses to pass over an empty set (see main()).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# reports/submission/ holds the staged, submitted-name PDFs and is NOT part of the code deposit;
# reports/figures/ is what the builders write and does exist once any builder has run. Try both.
SUB_DIRS = ("reports/submission", "reports/figures")
DEFAULT_MIN = 6.0          # points of headroom a panel title needs to read as its own heading


def panel_titles(page):
    """Bold phrases that open a panel: emitted at >=6 pt alongside a lone panel letter."""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    spans.append(s)
    letters = [s for s in spans
               if re.fullmatch(r"[a-h]", s["text"].strip()) and "Bold" in s["font"]]
    out = []
    for L in letters:
        lr = fitz.Rect(L["bbox"])
        # the title phrase sits on the same baseline, just to the right of the letter
        for s in spans:
            r = fitz.Rect(s["bbox"])
            if (s is not L and "Bold" in s["font"] and s["size"] >= 6.0
                    and abs(r.y1 - lr.y1) < 3.0 and 0 < r.x0 - lr.x1 < 25):
                out.append((lr | r, L["text"].strip(), s["text"].strip()))
                break
    return out, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=DEFAULT_MIN)
    ap.add_argument("--dir", default=None,
                    help="directory of figure PDFs (default: reports/submission, then reports/figures)")
    ap.add_argument("--min-font", type=float, default=None,
                    help="fail if any live text is below this many points. Off by default: the "
                         "journal publishes no numeric minimum and this tool does not invent one. "
                         "The measured minimum is always PRINTED so the number is visible.")
    args = ap.parse_args()

    dirs = [args.dir] if args.dir else list(SUB_DIRS)
    files, used = [], None
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "Figure*.pdf"))) + \
            sorted(glob.glob(os.path.join(d, "Additional_file_1_Fig*.pdf")))
        if files:
            used = d
            break
    if not files:
        print("NO FIGURE PDFs FOUND in %s (cwd %s)." % (" or ".join(dirs), os.getcwd()))
        print("This gate checks built PDFs. reports/submission/ is not part of the code deposit,")
        print("and reports/figures/ appears only after a builder runs. Build first, e.g.")
        print("    python -m src.ccs.fig2_split && python -m src.ccs.fig4_trust")
        print("then re-run. Refusing to report a pass over zero figures.")
        return 3
    print("reading %d figure(s) from %s/" % (len(files), used))
    bad, smallest = [], []
    print("%-32s %6s %9s  %s" % ("figure", "panels", "min gap", "tightest panel title"))
    for f in files:
        page = fitz.open(f)[0]
        titles, spans = panel_titles(page)
        boxes = [(fitz.Rect(sp["bbox"]), sp["size"])
                 for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
                 for sp in l["spans"] if sp["text"].strip()]
        live = [sz for _r, sz in boxes]
        ncol = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                inter = boxes[i][0] & boxes[j][0]
                if inter.is_valid and inter.get_area() > 1.0:
                    ncol += 1
        smallest.append((os.path.basename(f), min(live) if live else None,
                         sum(1 for v in live if v < (args.min_font or 6.0)), len(live), ncol))
        # Only text belonging to ANOTHER PANEL counts. A header band above the first panel sits
        # close on purpose and is not a collision; another panel's axis label running into this
        # panel's heading is. "Belongs to another panel" = it sits below some other panel title.
        worst = None
        for tr, letter, name in titles:
            for s in spans:
                r = fitz.Rect(s["bbox"])
                if r.y1 > tr.y0:                       # not above the title
                    continue
                if min(r.x1, tr.x1) - max(r.x0, tr.x0) <= 0:
                    continue                            # not in the same column
                owned = any(r.y0 > o.y1 and min(r.x1, o.x1) - max(r.x0, o.x0) > 0
                            for o, _, _ in titles if o is not tr)
                if not owned:
                    continue                            # figure-level header, not another panel
                gap = tr.y0 - r.y1
                if worst is None or gap < worst[0]:
                    worst = (gap, letter, name, s["text"].strip())
        name = os.path.basename(f)
        if worst is None:
            print("%-32s %6d %9s  %s" % (name, len(titles), "-", "nothing above any title"))
            continue
        flag = "  ** under %.1f pt **" % args.min if worst[0] < args.min else ""
        print("%-32s %6d %9.1f  (%s) %.22s  <- %.22s%s"
              % (name, len(titles), worst[0], worst[1], worst[2], worst[3], flag))
        if worst[0] < args.min:
            bad.append((name, worst[0], worst[1]))

    print()
    if bad:
        print("%d panel title(s) under %.1f pt of headroom -- LOOK AT THESE, do not assume:"
              % (len(bad), args.min))
        for n, g, L in bad:
            print("  %-30s panel %s at %.1f pt" % (n, L, g))
        print("\nAdvisory, not a gate. A small number here is not automatically a defect: a bold\n"
              "title under a lighter caption reads as separate at 1.5 pt, while two same-weight\n"
              "panel-level labels ran together at 0.7 pt. Crop the region and look.")
    else:
        print("every panel title clears %.1f pt" % args.min)

    thr = args.min_font or 6.0
    print()
    print("%-32s %10s %10s %8s  %s"
          % ("figure", "min type", "spans<%.1f" % thr, "collide", "total spans"))
    for name, mn, nsmall, tot, ncol in smallest:
        print("%-32s %9s %10d %8d  %d"
              % (name, ("%.2f pt" % mn) if mn is not None else "no text", nsmall, ncol, tot))
    print("'collide' counts pairs of text boxes whose overlap exceeds 1 pt^2. A few are normal "
          "(a\nhalo behind a label, a legend key over its patch); a jump after a layout change is "
          "the signal.")
    if args.min_font:
        under = [(n, m) for n, m, _nsmall, _tot, _ncol in smallest
                 if m is not None and m < args.min_font]
        if under:
            print("\n%d figure(s) carry live text below %.2f pt:" % (len(under), args.min_font))
            for n, m in under:
                print("  %-30s %.2f pt" % (n, m))
            return 1
        print("\nevery figure clears %.2f pt of live type" % args.min_font)
    else:
        print("\nType sizes are reported, not enforced: Genome Biology states that all information "
              "must be\nlegible at final size but publishes no numeric minimum, so this tool does "
              "not invent one.\nPass --min-font PT to turn the column above into a gate.")
    return 0                                   # clearance stays advisory; the empty set does not


if __name__ == "__main__":
    sys.exit(main())
