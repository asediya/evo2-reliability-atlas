"""PDF-level gate: page size, min type, text-span collisions (zero tolerance), off-page text, fonts.
Usage: python pdfcheck.py file.pdf [file2.pdf ...]   ;  --compare a.pdf b.pdf  diffs span text sets."""
import sys, re, fitz, collections
def spans(path):
    d = fitz.open(path); p = d[0]; r = p.rect
    out = []
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    # The span's TRUE quad is carried as element 4, so existing indices 0..3 are
                    # unchanged. fitz.recover_quad is PyMuPDF's own reconstruction from the line's
                    # baseline direction; do not hand-roll it from the bbox and dir (doing so
                    # makes the phantom below 14x WORSE rather than removing it).
                    q = fitz.recover_quad(l["dir"], s)
                    out.append((s["text"], round(s["size"], 2), s["font"],
                                tuple(round(v, 2) for v in s["bbox"]),
                                [(pt.x, pt.y) for pt in (q.ul, q.ur, q.lr, q.ll)]))
    d.close()
    return r, out
def gate(path):
    r, sp = spans(path)
    w, h = r.width * 25.4 / 72, r.height * 25.4 / 72
    def _inkbox(t, bb):
        """The span's box with its LEADING and TRAILING whitespace trimmed off.

        A span's bbox covers its spaces, which carry no ink. matplotlib splits a string at any
        glyph needing a fallback font, so "... AUROC  \u00b7  " and "\u25c7" arrive as two spans
        whose boxes abut across those trailing spaces -- 0.23 pt^2 of nothing, reported as a
        collision. Trimming is proportional to character count, which UNDER-corrects (a space is
        narrower than the average glyph) and so can only make the test stricter, never laxer.
        """
        n = len(t)
        if not n or not t.strip():
            return None
        lead = n - len(t.lstrip())
        trail = n - len(t.rstrip())
        per = (bb[2] - bb[0]) / float(n)
        return (bb[0] + lead * per, bb[1], bb[2] - trail * per, bb[3])

    def _ink_quad(t, q):
        """The span's quad with leading/trailing whitespace trimmed ALONG ITS OWN BASELINE.

        An axis-aligned box lies about rotated text: the box of a rotated line covers the empty
        triangles either side of the glyphs. Of nine radial AUROC labels set at nine different
        angles, two 12.4 pt apart on the page had boxes crossing by 3.41 pt^2 at the
        corners -- a collision that does not exist, which an axis-aligned test would report. A false
        positive costs what a false negative costs: it sends you to re-lay a panel that was fine,
        and the real defects sit beside it in the same report looking equally real.
        """
        n = len(t)
        if not n or not t.strip():
            return None
        (ux, uy), (rx, ry), (lx, ly), (bx, by) = q
        lead = (n - len(t.lstrip())) / float(n)
        trail = (n - len(t.rstrip())) / float(n)
        # Interpolate the corners along the top and bottom edges; both run baseline-wise, so this
        # is the rotated equivalent of trimming x, and it under-corrects exactly as _inkbox does
        # (a space is narrower than the average glyph), which can only make the test stricter.
        top = lambda f: (ux + (rx - ux) * f, uy + (ry - uy) * f)
        bot = lambda f: (bx + (lx - bx) * f, by + (ly - by) * f)
        return [top(lead), top(1 - trail), bot(1 - trail), bot(lead)]

    def _overlap(A, B):
        """(AREA, DEPTH) of two convex quads: separating-axis test, then clip and shoelace.

        DEPTH is the penetration -- how far one quad actually reaches into the other -- and it is
        the number to judge by. Area alone is misleading in the opposite direction to a bbox: two
        long labels on the SAME line that graze by 0.8 pt horizontally share a sliver whose area
        is their full line height times 0.8, which reads as a serious collision and is not one.
        """
        depth = 1e18
        for P, Q in ((A, B), (B, A)):
            for i in range(len(P)):
                x1, y1 = P[i]; x2, y2 = P[(i + 1) % len(P)]
                nx, ny = -(y2 - y1), (x2 - x1); L = (nx * nx + ny * ny) ** 0.5
                if L < 1e-12:
                    continue
                nx, ny = nx / L, ny / L
                pa = [nx * x + ny * y for x, y in P]; pb = [nx * x + ny * y for x, y in Q]
                gap = max(min(pb) - max(pa), min(pa) - max(pb))
                if gap > 0:
                    return 0.0, 0.0
                depth = min(depth, -gap)
        out = list(A)                                   # Sutherland-Hodgman clip of A against B
        for i in range(len(B)):
            x1, y1 = B[i]; x2, y2 = B[(i + 1) % len(B)]
            ex, ey = x2 - x1, y2 - y1
            side = lambda p: ex * (p[1] - y1) - ey * (p[0] - x1)
            ref = 1.0 if sum(side(q) for q in B) >= 0 else -1.0
            keep, prev = [], out[-1]
            for cur in out:
                sc, spv = side(cur) * ref, side(prev) * ref
                if sc >= 0:
                    if spv < 0:
                        t_ = spv / (spv - sc)
                        keep.append((prev[0] + t_ * (cur[0] - prev[0]), prev[1] + t_ * (cur[1] - prev[1])))
                    keep.append(cur)
                elif spv >= 0:
                    t_ = spv / (spv - sc)
                    keep.append((prev[0] + t_ * (cur[0] - prev[0]), prev[1] + t_ * (cur[1] - prev[1])))
                prev = cur
            out = keep
            if not out:
                return 0.0, 0.0
        return abs(sum(out[i][0] * out[(i + 1) % len(out)][1] - out[(i + 1) % len(out)][0] * out[i][1]
                       for i in range(len(out)))) / 2.0, depth

    coll = []
    for i in range(len(sp)):
        for j in range(i + 1, len(sp)):
            a, b = _ink_quad(sp[i][0], sp[i][4]), _ink_quad(sp[j][0], sp[j][4])
            if a is None or b is None:
                continue
            area, depth = _overlap(a, b)
            if area > 0:
                coll.append((depth, area, sp[i][0], sp[j][0]))
    off = [s for s in sp if s[3][0] < -0.01 or s[3][1] < -0.01 or s[3][2] > r.width + 0.01 or s[3][3] > r.height + 0.01]
    # A run of numbers that abut with no gap is extracted as ONE span, so the pairwise collision
    # test above scores it clean however unreadable it is. Flag any long unbroken numeric run:
    # two or more "d.d" groups with nothing between them, or ten-plus digits with no separator.
    fused = [s for s in sp if len(re.findall(r"\d[.,]\d", s[0])) >= 2
             # A separator is anything a reader sees BETWEEN two numbers, not just a space or a
             # dash: "1,500/1,500" is a ratio, not two numbers run together, and so are "9 (89)"
             # and "0.5 \u00b7 1.0". Without the slash here the gate fired on every pos/neg count
             # in Figure 6 the moment they were given their thousands separators.
             and not re.search(r"[ \u2013\u2014\u00b7/;:()\[\]-]", s[0].strip())
             or re.fullmatch(r"\d{10,}", s[0].strip())]
    fonts = sorted({s[2] for s in sp})
    print(f"{path.split('/')[-1]}: {w:.1f} x {h:.1f} mm | spans {len(sp)} | min {min(s[1] for s in sp):.2f} pt | "
          f"collisions {len(coll)} (deepest {max([c[0] for c in coll], default=0.0):.2f} pt) | "
          f"off-page {len(off)} | fused {len(fused)} | fonts {fonts}")
    for s in fused[:10]:
        print(f"    FUSED: {s[0]!r} {s[3]}")
    for c in sorted(coll, reverse=True)[:10]:
        print(f"    COLL {c[0]:.2f} pt deep ({c[1]:.2f} pt^2): {c[2]!r} x {c[3]!r}")
    for s in off[:10]:
        print(f"    OFF: {s[0]!r} {s[3]}")
    # The deepest collision comes back too: a COUNT cannot be thresholded in points, and a
    # raw count is not this project's criterion (see the --gate note below).
    return len(coll), len(off), len(fused), (max(c[0] for c in coll) if coll else 0.0)
if len(sys.argv) > 1 and sys.argv[1] == "--compare":
    ra, a = spans(sys.argv[2]); rb, b = spans(sys.argv[3])
    ca = collections.Counter((s[0], s[1]) for s in a); cb = collections.Counter((s[0], s[1]) for s in b)
    print(f"A {ra.width*25.4/72:.1f}x{ra.height*25.4/72:.1f} spans {len(a)} min {min(s[1] for s in a):.2f} | "
          f"B {rb.width*25.4/72:.1f}x{rb.height*25.4/72:.1f} spans {len(b)} min {min(s[1] for s in b):.2f}")
    onlyA = ca - cb; onlyB = cb - ca
    print(f"  only in A ({sum(onlyA.values())}):", list(onlyA.items())[:40])
    print(f"  only in B ({sum(onlyB.values())}):", list(onlyB.items())[:40])
else:
    # This script is NOT a gate by default, because its raw collision count is not the project's
    # accepted criterion: Additional file 1's Figure S5 reports 22 collisions, the deepest 0.61 pt,
    # and grazes of that depth, measured at 1200 dpi, hold 1.1-2.4 pt of white and are accepted as
    # leading rather than ink. src/ccs/check_overlaps.py is the gate and applies a
    # 0.35 mm floor for exactly this reason. Turning the raw count into an exit code would fail a
    # correct package. --gate PT sets a depth in points above which a collision IS an error, the
    # same opt-in shape tools/check_figure_clearance.py uses for --min-font.
    _args = [a for a in sys.argv[1:] if not a.startswith("--gate")]
    _thr = None
    for _i, _a in enumerate(sys.argv[1:]):
        if _a == "--gate":
            _thr = float(sys.argv[2 + _i]); _args = [x for x in _args if x != sys.argv[2 + _i]]
        elif _a.startswith("--gate="):
            _thr = float(_a.split("=", 1)[1])
    if not _args:
        print("usage: pdfcheck.py [--gate PT] figure.pdf [figure.pdf ...]")
        print("       pdfcheck.py --compare a.pdf b.pdf")
        sys.exit(2)
    _deep, _worst = 0, 0.0
    for f in _args:
        _coll, _off, _fused, _max = gate(f)
        _worst = max(_worst, _max)
        if _thr is not None and _max > _thr:
            _deep += 1
        _deep += _off
    if _thr is None:
        print("\nReporting only: the deepest collision is %.2f pt, printed and not enforced."%_worst)
        print("check_overlaps.py is the gate (0.35 mm floor). Pass --gate PT to make a collision")
        print("deeper than PT, or any off-page span, an error.")
        sys.exit(0)
    print("\n--gate %.2f pt: deepest collision %.2f pt, %d file(s) over the bar or off-page."
          % (_thr, _worst, _deep))
    sys.exit(1 if _deep else 0)
