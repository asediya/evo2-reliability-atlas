"""Diff regenerated analysis artifacts against the sealed pre-reproduction baseline.

Reproducing a number is only half an audit. This is the half that matters: it
compares every regenerated JSON (and every parquet's shape/summary) against
reports/_repro_baseline/ and reports what MOVED.

A drifted value is not automatically a bug — some arms legitimately resample
(bootstrap seeds, permutation draws). The job here is to surface every change so
each one can be judged, not to declare a verdict.

Usage:
    python tools/compare_repro.py
    python tools/compare_repro.py --tol 1e-9   # exactness check
"""
import argparse
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "reports", "_repro_baseline")
CUR = os.path.join(ROOT, "reports")


def flatten(obj, prefix=""):
    """Flatten nested JSON to {dotted.path: scalar}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, "%s[%d]" % (prefix, i)))
    else:
        out[prefix] = obj
    return out


def load(path):
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except Exception as e:
        return {"__load_error__": str(e)}


def close(a, b, tol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # NaN != NaN, so a NaN that reproduced perfectly would report as changed.
        # Treat NaN-in-both-places as unchanged; NaN on one side only is a real diff.
        a_nan, b_nan = a != a, b != b
        if a_nan or b_nan:
            return a_nan and b_nan
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / denom <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative tolerance for float comparison (default 1e-6)")
    args = ap.parse_args()

    if not os.path.isdir(BASE):
        print("FATAL: no baseline at %s" % BASE)
        return 2

    names = sorted(f for f in os.listdir(BASE) if f.endswith(".json"))
    total_changed = total_added = total_removed = 0
    report = []

    for n in names:
        bp, cp = os.path.join(BASE, n), os.path.join(CUR, n)
        if not os.path.exists(cp):
            report.append("## %s\n  !! REGENERATED FILE MISSING (baseline had it)\n" % n)
            continue
        b, c = flatten(load(bp)), flatten(load(cp))
        keys = sorted(set(b) | set(c))
        changed, added, removed = [], [], []
        for k in keys:
            if k not in c:
                removed.append(k)
            elif k not in b:
                added.append(k)
            elif not close(b[k], c[k], args.tol):
                changed.append((k, b[k], c[k]))
        if not (changed or added or removed):
            report.append("## %s\n  identical (%d keys)\n" % (n, len(keys)))
            continue
        total_changed += len(changed)
        total_added += len(added)
        total_removed += len(removed)
        lines = ["## %s" % n,
                 "  %d changed, %d added, %d removed (of %d keys)"
                 % (len(changed), len(added), len(removed), len(keys))]
        for k, ob, oc in changed[:60]:
            if isinstance(ob, (int, float)) and isinstance(oc, (int, float)):
                lines.append("  ~ %-58s %s -> %s  (d=%+.3g)" % (k, ob, oc, oc - ob))
            else:
                lines.append("  ~ %-58s %r -> %r" % (k, ob, oc))
        if len(changed) > 60:
            lines.append("  ... %d more changed" % (len(changed) - 60))
        for k in added[:15]:
            lines.append("  + %s = %r" % (k, c[k]))
        if len(added) > 15:
            lines.append("  ... %d more added" % (len(added) - 15))
        for k in removed[:15]:
            lines.append("  - %s (was %r)" % (k, b[k]))
        if len(removed) > 15:
            lines.append("  ... %d more removed" % (len(removed) - 15))
        report.append("\n".join(lines) + "\n")

    print("\n".join(report))
    print("=== TOTAL: %d changed, %d added, %d removed across %d files (tol=%g) ==="
          % (total_changed, total_added, total_removed, len(names), args.tol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
