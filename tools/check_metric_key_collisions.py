# -*- coding: utf-8 -*-
"""Every reused metric name in the recompute layer must say which panel it belongs to.

The deposit reuses bare metric names across different panels, readouts and
subsets with no discriminating field. `auroc_gerp` is human GERP AUROC in two files and means the
881-variant ESM-2-scorable subset in one and the 2,880-variant GERP-covered subset in the other;
both are correct and neither key says which. That naming is the mechanism behind the largest defect class -- a correct number attached to the wrong panel -- because it makes the
mistake easy to make and hard to catch.

This groups every (species, metric-name) pair across the artefacts and flags a collision when the
same name carries materially different values in two files. A collision is allowed only if the
blocks that carry it declare themselves: a sibling `panel`, `n`, `readout` or `subset` key, or an
explicit entry in ALLOWED below saying why the pair is fine.

    python tools/check_metric_key_collisions.py

Exit 0 when every collision is declared, 1 otherwise.
"""
import glob
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SPECIES = ("goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human")
# Fields that make a block self-describing. Any one of them is enough.
DISCRIMINATORS = ("panel", "n", "readout", "subset", "n_variants", "n_scoreable",
                  "n_gerp_scoreable", "arm", "checkpoint", "model",
                  # blocks that already name their own panel by another spelling
                  "n_pos", "n_matched", "n_pos_matched", "readout_note", "assembly")
# Rounding, not disagreement. Two values of the same quantity that agree to this are one value.
TOL = 1e-3
# Measurement names, the ones whose panel matters. A bare count or an interval endpoint is not what
# this check is about; a discrimination or calibration figure attached to the wrong panel is.
MEASUREMENT = ("auroc", "auprc", "ece", "lift", "reach", "rho", "delta", "penalty", "coverage",
               "err_detect", "capture", "brier", "d_fm", "dfm", "gap", "margin")

# Pairs that genuinely name the same quantity on the same panel and differ only by construction
# already explained inside the files. Each needs a reason; writing the reason is the check.
ALLOWED = {
    # (metric, species): why
}


def walk(node, path, out, parent):
    """Collect (metric-name, species, value, path, sibling-keys) for every numeric leaf."""
    if isinstance(node, dict):
        keys = set(node)
        for k, v in node.items():
            walk(v, path + [k], out, keys)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        sp = next((s for s in SPECIES if s in path), None)
        if sp is None:
            return
        metric = path[-1]
        if metric in DISCRIMINATORS or metric.startswith("_"):
            return
        # A leaf named for a species is a species-keyed value, not a metric whose name is ambiguous;
        # its parent path already says what it is. And a count or a bound is not the quantity this
        # check is about. Restricting to measurement names keeps the signal: a noisy gate is a gate
        # that gets switched off, which this repository has already learned once.
        if metric in SPECIES:
            return
        if not any(metric.startswith(pre) for pre in MEASUREMENT):
            return
        out.append((metric, sp, float(node), "/".join(path), parent))


def main():
    files = sorted(set(glob.glob("reports/*.json") + glob.glob("analyses/results/*.json")))
    seen = {}
    unparseable = []
    for f in files:
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception as e:
            # A file dropped here is counted and reported, never left to vanish from the scan while the
            # summary counts the GLOB and prints CLEAN at exit 0.
            # A total that hides what it never saw is the one
            # output this gate must not produce.
            unparseable.append((f, e))
            continue
        rows = []
        walk(d, [], rows, set())
        for metric, sp, val, path, sib in rows:
            seen.setdefault((metric, sp), []).append((f, val, path, sib))

    undeclared, declared = [], 0
    for (metric, sp), hits in sorted(seen.items()):
        by_file = {}
        for f, val, path, sib in hits:
            by_file.setdefault(f, []).append((val, path, sib))
        if len(by_file) < 2:
            continue
        vals = [v for h in hits for v in (h[1],)]
        if max(vals) - min(vals) <= TOL:
            continue                       # rounding, not a collision
        if (metric, sp) in ALLOWED:
            declared += 1
            continue
        # declared if EVERY file carrying it gives that block a discriminating sibling
        ok = all(any(k in sib for k in DISCRIMINATORS) for _, _, _, sib in hits)
        if ok:
            declared += 1
        else:
            bare = sorted({f for f, _, _, sib in hits
                           if not any(k in sib for k in DISCRIMINATORS)})
            undeclared.append((metric, sp, round(min(vals), 4), round(max(vals), 4), bare))

    print("=" * 90)
    print("  METRIC-NAME COLLISIONS ACROSS THE RECOMPUTE LAYER")
    print("=" * 90)
    print("  artefacts scanned      : %d of %d globbed (%d unparseable)"
          % (len(files) - len(unparseable), len(files), len(unparseable)))
    for _f, _e in unparseable:
        print("    UNPARSEABLE %s: %s" % (os.path.basename(_f), _e))
    print("  (metric, species) pairs: %d" % len(seen))
    print("  collisions declared    : %d" % declared)
    print("  collisions UNDECLARED  : %d" % len(undeclared))
    for metric, sp, lo, hi, bare in undeclared[:40]:
        print("    %-28s %-8s %.4f .. %.4f   bare in: %s"
              % (metric, sp, lo, hi, ", ".join(os.path.basename(b) for b in bare)))
    print()
    if unparseable:
        print("  RESULT: INCOMPLETE -- %d artefact(s) could not be parsed and were scanned for"
              % len(unparseable))
        print("  nothing. That is not a clean result; fix or remove them and re-run.")
        return 1
    if undeclared:
        print("  RESULT: FAIL -- a metric name means two things and no field says which.")
        print("  Give each block a panel/n/readout/subset field, or add it to ALLOWED with a reason.")
        return 1
    print("  RESULT: CLEAN -- every reused metric name carries a discriminating field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
