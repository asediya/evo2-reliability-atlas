# -*- coding: utf-8 -*-
"""Rebuild every submitted figure and stage it into reports/submission/.

WHY THIS FILE EXISTS. finalize_submission.sh regenerates the manuscript, the supplement and the
code deposit, but not the figures: builders write to reports/figures/ and the deliverables are read
from reports/submission/. Without this file a rebuilt figure reaches the
submission only if someone remembers to copy it, and the export step snapshots whatever is in
reports/submission/ -- so a stale figure could be shipped as final without any gate objecting.

The builder-to-deliverable mapping is the one documented in FIGURES.md. Names differ on purpose:
`fig3_rebuild` emits "Figure3_blindspot.pdf", which ships as Additional file 1's Figure S2.

    python tools/stage_figures.py            # rebuild, then stage
    python tools/stage_figures.py --no-build # stage what is already built
    python tools/stage_figures.py --check    # report staleness, change nothing
"""
import argparse
import io
import os
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILT = os.path.join(ROOT, "reports", "figures")
# v2 stages into its own directory. The v1 set is locked and must not be written to; honouring the
# same CCS_SUBMISSION_OUT the docx builder uses keeps every stage of the pipeline pointed at one
# version, which is what went wrong when the figures and the manuscript came from different ones.
SUB = os.path.join(ROOT, os.environ.get("CCS_SUBMISSION_OUT", os.path.join("reports", "submission")))

# (builder module, built filename, shipped filename). One builder may emit two figures.
#
# The built names follow the PROJECT numbering and the shipped names the MANUSCRIPT numbering;
# the two differ, and several project plates ship in Additional file 1. That is the reason
# this mapping has to be written down rather than inferred from filenames.
#
# The mapping is established three independent ways, because a filename proves nothing here:
#   * by digest, against the upload;
#   * by the recompute layer each builder reads (eqtl_* -> S2, selection_evo2_40b -> S3,
#     fig4_reconciliation -> S4, splice.json -> Figure 8), per FIGURES.md;
#   * against Additional file 1 Note S58, whose set of figures that need the undeposited data/
#     tree -- Figures S3, S5 and S6 -- matches FIGURES.md.
#
# Left column: builder module. Middle: what it writes into reports/figures/. Right: the name the
# reader receives in the upload. The Additional_file_1_* plates are embedded
# in the supplement and are NOT shipped beside it as separate PDFs; that name is the staging name.
FIGURES = [
    ("fig5_final",           "Figure5_reach.pdf",        "Additional_file_1_FigureS5.pdf"),
    ("fig_brca1_composite",  "Figure_BRCA1.pdf",         "Additional_file_1_FigureS6.pdf"),
    ("fig7_atlas",           "Figure7_atlas.pdf",        "fig9.pdf"),
    ("fig8_readout",         "Figure8_readout.pdf",      "fig10.pdf"),
    ("fig2_split",           "FigureS7_measurement_supp.pdf", "Additional_file_1_FigureS7.pdf"),
    ("fig3_rebuild",         "Figure3_blindspot.pdf",    "Additional_file_1_FigureS2.pdf"),
    ("fig6_final",           "Figure6_curator.pdf",      "Additional_file_1_FigureS3.pdf"),
    ("fig4_trust",           "Figure4_trust.pdf",        "Additional_file_1_FigureS4.pdf"),
    ("figS1_missing_panels", "figS1_missing_panels.pdf", "Additional_file_1_FigureS1.pdf"),
]
# Submitted Figures 1 to 8 and Additional file 1's Figures S8 and S9 are NOT staged here: they are built by
# Additional file 3's own figures/build_*.py, which write beside themselves: build_concept.py builds
# submitted Figure 1, build_fig2.py Figure 2, build_fig_bounds.py Figure 3, build_fig_decide.py
# Figure 4, build_fig_frontier.py Figure 5, build_fig1.py Figure 6, build_fig_evidence.py Figure 7,
# build_fig4.py Figure 8, build_fig_transfer.py Figure S8 and build_fig_plate.py Figure S9 (three keep internal
# numbers). See FIGURES.md.
#
# H_MAX is the GRAPHIC height, not the journal's 225 mm. That 225 covers the figure AND the legend
# set beneath it, and the legends here set to 23.7-47.4 mm, so a plate has to stop at 173 mm.
# A gate written against 225.5 passes every figure this archive has ever emitted and so tests
# nothing that matters.
W_MAX, H_MAX = 170.5, 173.5


def interpreter():
    env = os.environ.get("CCS_PYTHON")
    if env and os.path.exists(env):
        return env
    for c in (os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
              os.path.join(ROOT, ".venv", "bin", "python")):
        if os.path.exists(c):
            return c
    return sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true", help="stage without re-running builders")
    ap.add_argument("--check", action="store_true", help="report only; change nothing")
    a = ap.parse_args()
    py = interpreter()

    if not (a.no_build or a.check):
        for mod in sorted({m for m, _, _ in FIGURES}):
            script = os.path.join(ROOT, "src", "ccs", mod + ".py")
            if not os.path.exists(script):
                print("  SKIP %s (no such builder)" % mod)
                continue
            p = subprocess.run([py, "-m", "src.ccs." + mod], cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            print("  build %-22s %s" % (mod, "OK" if p.returncode == 0 else "FAILED"))
            if p.returncode != 0:
                sys.stdout.write(p.stdout.decode("utf-8", "replace")[-1200:])
                sys.exit("builder %s failed; refusing to stage a partial figure set" % mod)

    import fitz
    os.makedirs(SUB, exist_ok=True)
    rows, stale, bad = [], 0, 0
    for _mod, built, shipped in FIGURES:
        src, dst = os.path.join(BUILT, built), os.path.join(SUB, shipped)
        if not os.path.exists(src):
            print("  MISSING build output: %s" % built)
            bad += 1
            continue
        older = (not os.path.exists(dst)) or os.path.getmtime(dst) < os.path.getmtime(src) - 1
        if older:
            stale += 1
        if not a.check and older:
            shutil.copy2(src, dst)
        d = fitz.open(dst if os.path.exists(dst) else src)
        r = d[0].rect
        d.close()
        w, h = r.width * 25.4 / 72, r.height * 25.4 / 72
        fits = w <= W_MAX and h <= H_MAX
        if not fits:
            bad += 1
        rows.append((shipped, w, h, fits, older))
        print("  %-32s %6.1f x %6.1f mm  %s%s"
              % (shipped, w, h, "within envelope" if fits else "** EXCEEDS 170 x 225 **",
                 "   (was stale)" if older else ""))

    # Files in reports/submission/ carrying a BUILDER name rather than a shipped name are leftovers
    # from before this mapping existed. They
    # matter because they COLLIDE BY NUMBER with live figures (Figure3_blindspot is Additional file 1's
    # Figure S2, not Figure 3) and because the export step snapshots whatever is in this directory,
    # so one could be shipped with no gate objecting. Reported, never deleted here: a script should
    # not remove a PDF on the strength of a naming convention.
    shipped_names = {dst for _, _, dst in FIGURES}
    builder_names = {src for _, src, _ in FIGURES}
    orphans = sorted(n for n in os.listdir(SUB)
                     if n in builder_names and n not in shipped_names)
    if orphans:
        print("\n  %d ORPHAN(S) in reports/submission/ carry a builder name, not a shipped name:"
              % len(orphans))
        for n in orphans:
            print("    %-28s collides by number with a live figure -- delete it" % n)

    if a.check:
        print("\n%d of %d shipped figures are behind their builder" % (stale, len(FIGURES)))
        sys.exit(1 if (stale or bad or orphans) else 0)
    # Print the directory actually written. The literal path here said reports/submission/ while the
    # env var sent the files elsewhere, which is exactly the kind of line that makes a reader certify
    # the wrong version.
    print("\nstaged %d figure(s) into %s" % (stale, os.path.relpath(SUB, ROOT).replace("\\", "/")))
    if bad:
        sys.exit("%d figure(s) missing or outside the Genome Biology envelope" % bad)


if __name__ == "__main__":
    main()
