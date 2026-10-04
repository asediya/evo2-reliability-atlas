# -*- coding: utf-8 -*-
"""Run every verification gate in the archive and report one table.

The gates were written one at a time, each in response to a defect that reached a deliverable.
Nothing else runs them together, so a gate can stay broken without anyone noticing. This runs all
of them and prints one line each.

Gates fall into three groups, and the difference is declared here rather than left for the reader to
work out from a traceback. Four run from the archive alone (the count main() prints is authoritative). Nine read the raw score and label tree
under data/ (or analyses/data/), which is not deposited; from a clone they
report the path they wanted. Those are marked NODATA, not FAIL, and they do not set the exit code.
reports/DATA_MANIFEST.md gives the public source of every path they ask for.

The third group is a gate that recomputes correctly and then cannot write its result, because the
archive ships read-only so that a stale artefact fails loudly instead of being silently overwritten
(docs/REPRODUCING.md). Those are marked RDONLY, also not FAIL, and the remedy is printed: a bare
PermissionError reads as a failing verification gate, so the tool says what happened instead of
leaving a traceback to be interpreted.

Exit 0 when every gate that can run passes.

Two checks are deliberately NOT in this list and have to be run separately. The 166-test
package suite is `pytest glmtrust`. The reproduction benchmark is
`python glmtrust/benchmarks/reproduce_paper_trust_layer.py` after `pip install './glmtrust[io]'`,
and it is the only artefact in the deposit that catches a leave-one-group-out calibration leak,
so a reader checking that property should run it rather than rely on this runner.

    python tools/run_all_gates.py
"""
import concurrent.futures as cf
import os
import subprocess
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
TIMEOUT = 1800

# Runs from the archive as shipped.
SELF_CONTAINED = [
    "tools/check_traceability.py",
    "tools/verify_delong_scale.py",
    "src/ccs/check_overlaps.py",
    # It is self-contained, passes on a clean extraction ("every reused metric name carries a
    # discriminating field") and has a real non-zero exit path, so it belongs here.
    "tools/check_metric_key_collisions.py",
]

# Reads the undeposited raw tree. Expected to stop at a missing path from a clone.
NEEDS_DATA = [
    # Reads BUILT figure PDFs, which appear only after a builder has run, so a clean extraction
    # has none. It used to sit in SELF_CONTAINED, where it globbed an absent directory, printed
    # "every panel title clears 6.0 pt" over zero files and was recorded as PASS. It now exits 3
    # on an empty set, which belongs here as NODATA rather than as a phantom pass.
    "tools/check_figure_clearance.py",
    # Examined zero species from a clean extraction and still exited 0, which the runner recorded
    # as a PASS over no data. It now exits 3 when it finds no consequence table.
    "tools/verify_species_gene_structure.py",
    "tools/verify_from_data.py",
    "tools/verify_bh_seed_stability.py",
    "tools/verify_seed_stability.py",
    "tools/verify_reach_against_paper.py",
    "src/ccs/check_atlas_ref_ok.py",
    "analyses/scripts/verify_new_results.py",
    # Reads data/external/, which is not deposited, so it belongs in
    # NEEDS_DATA rather than SELF_CONTAINED.
    "tools/check_var_off.py",
]

# These four report numbers but declare no pass criterion: an AST scan finds no sys.exit, no
# raise and no non-zero return anywhere in any of them, so each exits 0 whatever it finds.
# Counted as GATES they would claim a verdict none of them can give. They are diagnostics, and
# the runner prints them under their own heading and excludes them from the gate tally. Giving
# them an invented pass criterion would be worse: the seed-dependence they measure is a PUBLISHED
# result (Table S20), not a defect.
DIAGNOSTICS = [
    "src/ccs/check_human_prevalence.py",
    "tools/verify_gene_clustering.py",
    "tools/verify_naive_delta_ci.py",
    "tools/verify_transcript_aggregation.py",
]

GATES = SELF_CONTAINED + NEEDS_DATA

# These size their own worker pool to the machine. Run alongside the rest they oversubscribe the
# cores and time out, which the summary then reports as failing gates. Run one at a time instead.
SERIAL = {
    "tools/verify_seed_stability.py",
    "tools/verify_bh_seed_stability.py",
    "tools/verify_gene_clustering.py",
    "tools/verify_naive_delta_ci.py",
}


# Every phrasing by which a child says a number it recomputed DISAGREES with a published one.
# Matching only "N failed" is too narrow: analyses/scripts/verify_new_results.py reports
# "N artefact(s) did not reproduce", which its own docstring calls a defect, and that phrasing
# would otherwise fall through to the NODATA bucket and out of the exit code. Keep these anchored
# to an explicit count or an explicit verdict marker, so that a gate printing "0 failed" or
# describing the check in prose is not mistaken for a failure.
FAILURE_SIGNALS = (
    re.compile(r"\b[1-9]\d* failed\b"),
    re.compile(r"\b[1-9]\d* artefact\(s\) did not reproduce\b"),
    re.compile(r"\*\*\* FAIL \*\*\*"),
)


# A gate that must be told what to enforce is not a gate until it is told. check_figure_clearance.py
# PRINTS the smallest live type it measures and reaches its `return 1` only under --min-font, which
# the runner never passed -- so it was recorded as PASS whatever it measured, and said so itself:
# "Pass --min-font PT to turn the column above into a gate." 7.0 pt is this paper's shipped floor.
GATE_ARGS = {
    "tools/check_figure_clearance.py": ["--min-font", "7.0"],
}


def run(path):
    if not os.path.exists(path):
        # SIX fields, like every other return here. main() unpacks six AFTER the whole suite
        # has run, so a 5-tuple would crash it and discard every result whenever one gate file
        # is absent.
        return path, None, "not present", 0.0, False, False
    t0 = time.time()
    try:
        p = subprocess.run([PY, path] + GATE_ARGS.get(path, []),
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return path, 124, "TIMEOUT after %ds" % TIMEOUT, time.time() - t0, False, False
    out = (p.stdout or "").strip().splitlines()
    err = (p.stderr or "").strip().splitlines()
    msg = out[-1][:120] if out else (err[-1][:120] if err else "")
    blob_ = (p.stderr or "") + "\n" + (p.stdout or "")
    # MUST be decided BEFORE `readonly` and BEFORE `stopped_at_path`, because both of those are
    # excuses and a reported failure overrides every excuse.
    reported_failure = any(rx.search(blob_) for rx in FAILURE_SIGNALS)
    # A gate that computed its answer and then could not write it has not failed -- UNLESS it
    # reported a failure first. `readonly` consults reported_failure because main() tests it
    # before FAIL: otherwise a gate that printed its own "*** FAIL ***" and then hit PermissionError
    # writing its JSON would be filed RDONLY, have its own verdict line overwritten by the message
    # below, and drop out of the exit code.
    # The archive ships read-only on purpose, so that is the ordinary path and not a corner case.
    # Read the exception type rather than the message, which is localised.
    readonly = (not reported_failure
                and p.returncode != 0 and "PermissionError" in (p.stderr or ""))
    if readonly and out:
        msg = "recomputed, but could not write its output (archive is read-only)"
    # A NEEDS_DATA gate stops at a missing path (exit 3 by convention, or a FileNotFoundError on
    # stderr). Anything else is the gate recomputing a value and DISAGREEING with it, which must
    # surface as FAIL rather than be buried in the NODATA bucket.
    blob = (p.stderr or "") + "\n" + (p.stdout or "")
    # A SUBSTRING ANYWHERE IN THE OUTPUT MUST NOT ERASE A NUMERICAL FAILURE. verify_from_data.py
    # catches per section and continues, so one section can hit an undeposited path while another
    # reports a real mismatch. Classifying the whole child as NODATA on the mere presence of
    # "FileNotFoundError" would bury the mismatch and print "every gate that can run ... passes".
    # A child that reports failed checks is a FAIL whatever else its output mentions.
    stopped_at_path = (not reported_failure
                       and (p.returncode == 3
                            or "FileNotFoundError" in blob
                            or "No such file or directory" in blob
                            or "MISSING INPUT" in blob))
    return path, p.returncode, msg, time.time() - t0, readonly, stopped_at_path


def main():
    parallel = [g for g in GATES if g not in SERIAL]
    serial = [g for g in GATES if g in SERIAL]
    workers = min(len(parallel), max(2, (os.cpu_count() or 4) - 2))
    print("  %d gates: %d self-contained, %d needing the undeposited data tree"
          % (len(GATES), len(SELF_CONTAINED), len(NEEDS_DATA)))
    print("  %d diagnostics, which report numbers but declare no pass criterion and are not "
          "counted as gates" % len(DIAGNOSTICS))
    print("  %d in parallel on %d workers, then %d serially\n"
          % (len(parallel), workers, len(serial)))

    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        results.extend(ex.map(run, parallel))
    for g in serial:
        results.append(run(g))

    fails, nodata, missing, passed, rdonly = [], [], [], [], []
    order = {p: i for i, p in enumerate(GATES)}
    for path, rc, msg, secs, ro, stopped in sorted(results, key=lambda r: order.get(r[0], 99)):
        if rc is None:
            state, bucket = "SKIP", missing
        elif rc == 0:
            state, bucket = "PASS", passed
        elif ro:
            state, bucket = "RDONLY", rdonly
        elif path in NEEDS_DATA and stopped:
            state, bucket = "NODATA", nodata
        else:
            state, bucket = "FAIL", fails
        bucket.append(path)
        print("  %-6s %-46s %6.1fs  %s" % (state, path, secs, msg))

    print()
    print("  %d passed, %d failed, %d stopped at an undeposited path, %d blocked by the "
          "read-only archive, %d not present"
          % (len(passed), len(fails), len(nodata), len(rdonly), len(missing)))
    if rdonly:
        print()
        print("  %d gate(s) recomputed their answer and then could not write it. The archive ships"
              % len(rdonly))
        print("  read-only on purpose, so that a stale artefact fails loudly rather than being")
        print("  silently overwritten; this is not a packaging fault (docs/REPRODUCING.md).")
        print("  Take a writable copy and re-run:")
        print("      cp -r . /tmp/ccs && chmod -R u+w /tmp/ccs && cd /tmp/ccs")
        print("      python tools/run_all_gates.py")
        print("  blocked: %s" % ", ".join(rdonly))
    if DIAGNOSTICS:
        print()
        print("  DIAGNOSTICS (no pass criterion; their output is for reading, not for a verdict)")
        for d in DIAGNOSTICS:
            path, rc, dmsg, secs, ro, stopped = run(d)
            print("  %-6s %-46s %6.1fs  %s"
                  % ("-" if rc is None else "ran", d, secs, "not present" if rc is None else dmsg))

    if fails:
        print("  failing gates: %s" % ", ".join(fails))
    elif not rdonly:
        print("  every gate that can run from this archive passes.")
    else:
        print()
        print("  no gate failed: %d passed and the rest could not run in place." % len(passed))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
