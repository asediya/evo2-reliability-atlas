"""Detect overlapping VISIBLE text in a rendered figure, from real geometry rather than by eye.

Eyeballing a 180 x 244 mm figure at preview size misses collisions that are obvious in print. This
builds the figure from source, writes it, and measures the TEXT AS IT PRINTS.

MEASURES THE PAGE, NOT THE LAYOUT BOX. Comparing matplotlib's
text artists by `get_window_extent()` measures the layout box, and it is not the ink:

  * a MULTI-LINE Text is one artist, so its box is the union of all its lines. Two annotation
    blocks whose lines interleave -- one's last line above the other's first -- read as
    overlapping when no line touches.
  * a LEGEND box includes its handle column and padding, so two legends near each other collide
    with no text involved.
  * every box carries leading, which is not ink.

A verification tool that manufactures evidence against its own figure is worse than
no tool: it fails loudly and wrongly, in the reader's hands. So the geometry comes from the
written PDF -- one box per printed LINE, the true quad for rotated text, separating-axis test --
which is the same measurement tools/pdfcheck.py makes, and agrees with it. What this keeps over
pdfcheck is that it builds from source and attributes every string to the axes it belongs to.

Run:  python -m src.ccs.check_overlaps [module | figure.pdf]      (default: fig4_trust)

Five of the figure modules -- fig2_split, fig3_rebuild, fig_brca1_composite, fig7_atlas and
fig8_readout -- write their PDFs and return None rather than a Figure. Pass such a PDF instead and it checks
the page directly; the only thing lost is the ax0/ax1 attribution, which needs a live Figure.
"""
import sys, os, importlib, itertools
import matplotlib
matplotlib.use("Agg")

MIN_MM = 0.35          # ignore hairline touches below this: a box carries leading, ink does not


def visible(a):
    try:
        if not a.get_visible():
            return False
    except Exception:
        return False
    # figtext.restore_text_layer adds a fully transparent twin of every haloed string, so the
    # string survives in the PDF text layer that path effects strip. Each twin sits exactly on its
    # original. Counted as visible text they produced one collision per twin, every one of them a
    # string against its own copy, with nothing wrong on the page.
    try:
        if a.get_alpha() == 0.0:
            return False
    except Exception:
        pass
    ax = getattr(a, "axes", None)
    if ax is not None and not ax.get_visible():
        return False
    return True


def _axes_boxes(fig):
    """Every axes' rectangle in PDF points, top-left origin, for attributing spans to panels."""
    h = fig.get_size_inches()[1] * 72.0
    out = []
    for i, ax in enumerate(fig.axes):
        p = ax.get_position()
        w = fig.get_size_inches()[0] * 72.0
        out.append((f"ax{i}", p.x0 * w, h - p.y1 * h, p.x1 * w, h - p.y0 * h))
    return out


def _quad(ln, sp):
    """The span's true quad. An axis-aligned box lies about rotated text (see tools/pdfcheck.py)."""
    import pymupdf
    dx, dy = ln.get("dir", (1.0, 0.0))
    if abs(dy) < 1e-6 and dx > 0:
        x0, y0, x1, y1 = sp["bbox"]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    q = pymupdf.recover_quad(ln["dir"], sp)
    return [(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll)]


def _overlap_mm(A, B):
    """(width, height) of the overlap in mm, 0 if the quads are apart (separating-axis test)."""
    for P, Q in ((A, B), (B, A)):
        for i in range(len(P)):
            x1, y1 = P[i]; x2, y2 = P[(i + 1) % len(P)]
            nx, ny = -(y2 - y1), (x2 - x1); L = (nx * nx + ny * ny) ** 0.5
            if L < 1e-12:
                continue
            nx, ny = nx / L, ny / L
            pa = [nx * x + ny * y for x, y in P]; pb = [nx * x + ny * y for x, y in Q]
            if max(min(pb) - max(pa), min(pa) - max(pb)) > 0:
                return 0.0, 0.0
    ax0, ax1 = min(p[0] for p in A), max(p[0] for p in A)
    ay0, ay1 = min(p[1] for p in A), max(p[1] for p in A)
    bx0, bx1 = min(p[0] for p in B), max(p[0] for p in B)
    by0, by1 = min(p[1] for p in B), max(p[1] for p in B)
    return ((min(ax1, bx1) - max(ax0, bx0)) * 25.4 / 72.0,
            (min(ay1, by1) - max(ay0, by0)) * 25.4 / 72.0)


def report(fig, pdf=None):
    """Report every pair of printed lines overlapping by more than MIN_MM, from the page itself.

    `fig` may be None when `pdf` is given: the geometry comes from the file either way, and the
    Figure is used only to name which axes each string belongs to.
    """
    import tempfile, pymupdf
    tmp = pdf
    if fig is not None:
        fig.canvas.draw()
        if tmp is None:
            tmp = os.path.join(tempfile.mkdtemp(), "check_overlaps.pdf")
            fig.savefig(tmp)
    boxes = _axes_boxes(fig) if fig is not None else []
    pg = pymupdf.open(tmp)[0]
    items = []
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                t = sp["text"].strip()
                if not t:
                    continue
                cx = (sp["bbox"][0] + sp["bbox"][2]) / 2.0
                cy = (sp["bbox"][1] + sp["bbox"][3]) / 2.0
                where = next((n for n, x0, y0, x1, y1 in boxes if x0 <= cx <= x1 and y0 <= cy <= y1),
                             "figure")
                items.append((where, t, _quad(ln, sp)))
    hits = []
    for (w1, s1, q1), (w2, s2, q2) in itertools.combinations(items, 2):
        mmx, mmy = _overlap_mm(q1, q2)
        if mmx > MIN_MM and mmy > MIN_MM:
            hits.append((mmx * mmy, mmx, mmy, w1, s1, w2, s2))
    hits.sort(reverse=True)
    print(f"printed text lines  : {len(items)}")
    print(f"OVERLAPPING PAIRS   : {len(hits)}\n")
    for _, mx, my, w1, s1, w2, s2 in hits:
        print(f"  [{mx:5.1f} x {my:4.1f} mm]  {w1:15s} {s1[:46]!r}")
        print(f"                          vs {w2:15s} {s2[:46]!r}")
    return hits


def _reports_snapshot(root="reports"):
    """Every file under reports/ at this moment."""
    out = set()
    for d, _, fs in os.walk(root):
        for f in fs:
            out.add(os.path.join(d, f))
    return out


def _clean_new(before, keep, root="reports"):
    """Leave the archive as it was found.

    Building a figure to measure it is this tool's method, not its output. fig4_trust.main() writes
    reports/figures/Figure4_trust.pdf, .png and _preview.png, and none of the three is in
    MANIFEST.sha256 -- so leaving them would make check_manifest.py report 3 UNDECLARED
    files on an otherwise intact archive, which reads as a packaging fault and is not one.
    Files that were already present are never touched, so an author who has built their figures
    keeps them.
    """
    made = sorted(_reports_snapshot(root) - before)
    if not made:
        return
    if keep:
        print("\nkept %d build output(s) at --keep:" % len(made))
        for q in made:
            print("    %s" % q)
        print("  These are NOT in MANIFEST.sha256; check_manifest.py will call them undeclared.")
        return
    left = []
    for q in made:
        try:
            os.remove(q)
        except OSError as e:
            left.append("%s (%s)" % (q, e))
    print("\nremoved %d build output(s) written while measuring, leaving the archive as found;"
          % (len(made) - len(left)))
    print("  pass --keep to retain them:")
    for q in made:
        print("    %s" % q)
    for q in left:
        print("  COULD NOT REMOVE %s" % q)



if __name__ == "__main__":
    # Import the figure module WITH its package context. Loading it by bare name leaves
    # __package__ empty, so any `from .style_gb import ...` inside it raises ImportError
    # (fig4_trust, fig5_final, fig6_final and the fig2_* modules all use one).
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    _args = [a for a in sys.argv[1:] if a != "--keep"]
    _keep = "--keep" in sys.argv[1:]
    _name = _args[0] if _args else "fig4_trust"
    if _name.lower().endswith(".pdf"):
        # Measuring an existing PDF writes nothing, so there is nothing to clean up.
        sys.exit(1 if report(None, pdf=_name) else 0)
    # Snapshot BEFORE the import: a figure module can write at import time as well as in main().
    _before = _reports_snapshot()
    try:
        mod = importlib.import_module(_name if "." in _name else "ccs." + _name)
        _fig = mod.main()
        if _fig is None:
            # Not an error in the module: it writes its own files. Say which, rather than dying on
            # `None.canvas` -- a traceback from a verification tool reads as a failing figure.
            print(f"{_name}.main() returned no Figure; it writes its own PDFs. Re-run against one:\n"
                  f"    python -m src.ccs.check_overlaps reports/figures/<name>.pdf")
            sys.exit(0)
        sys.exit(1 if report(_fig) else 0)
    finally:
        # Runs on the sys.exit paths above and on an exception, so a crash mid-build does not strand
        # files either.
        _clean_new(_before, _keep)
