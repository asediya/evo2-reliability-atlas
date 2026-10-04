# -*- coding: utf-8 -*-
"""Re-run every CPU analysis registered in CASES and check its artefact is byte-reproducible.

This project has twice shipped a number that came from an artefact older than its own input, and
both times every other gate passed because they read the stale file rather than rebuilding it. The
defence is to rebuild and compare, not to inspect.

Each analysis is re-run into a temporary copy of its output, then compared with what is on disk. A
difference means either the artefact is stale or the analysis is not deterministic; both are
defects and both are reported rather than tolerated. GPU-dependent steps are skipped, since they
depend on scores that cannot be regenerated without the model, but the analyses that consume those
scores are checked here.

    python analyses/scripts/verify_new_results.py
"""
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PY = sys.executable
NEW = "analyses/results"

# script -> the artefact it must reproduce
CASES = [
    ("analyses/scripts/mfass_reach.py", "mfass_reach.json"),
    ("analyses/scripts/mfass_random500k_coverage.py", "mfass_random500k_coverage.json"),
    ("analyses/scripts/mfass_evo2_vs_specialists.py", "mfass_evo2_vs_specialists.json"),
    ("analyses/scripts/clinvar_evo2_participant.py", "clinvar_evo2_participant.json"),
    ("analyses/scripts/strand_analyse_40b.py", "strand_40b.json"),
    ("analyses/scripts/strand_ladder.py", "strand_ladder.json"),
]


# A regeneration script that cannot START, because an undeposited upstream input is absent, has not
# shown that its artefact fails to reproduce -- it has shown nothing at all, so it is reported as
# NEEDS, not as "did not reproduce". Each of the six scripts prints its own
# reason and exits 1; these are the phrasings they use.
MISSING_INPUT = (
    "FileNotFoundError",
    "No such file or directory",
    "not present yet",
    "scores yet",
    "is absent",
    "MISSING INPUT",
)


def canonical(p):
    """Compare parsed JSON, not raw bytes: key order and float formatting are not the claim."""
    with io.open(p, encoding="utf-8") as f:
        return json.dumps(json.load(f), sort_keys=True, indent=1)


def main():
    fails, skips, needs = [], [], []
    for script, art in CASES:
        path = os.path.join(NEW, art)
        if not os.path.exists(path):
            print("  SKIP  %-46s (no artefact; upstream scores absent)" % art)
            skips.append(art)
            continue
        with tempfile.TemporaryDirectory() as td:
            keep = os.path.join(td, art)
            shutil.copy2(path, keep)
            # The deposit ships its results read-only, so that a stale artefact fails loudly instead of
            # being overwritten by accident. A re-run has to write it, so it is made writable for the run;
            # afterwards the reviewed file, bytes and mode, is always put back, so this check never alters
            # the deposit whatever the outcome.
            mode = stat.S_IMODE(os.stat(path).st_mode)
            os.chmod(path, mode | stat.S_IWUSR)
            try:
                r = subprocess.run([PY, script], capture_output=True, text=True)
                if r.returncode != 0:
                    blob = (r.stdout or "") + "\n" + (r.stderr or "")
                    tail = [ln.strip() for ln in blob.splitlines() if ln.strip()]
                    # The old form was `... % r.stderr.strip().splitlines()[-1:] or ""`, which printed a
                    # LIST (and "[]" whenever stderr was empty), hiding the reason on four of six cases.
                    why = tail[-1][:100] if tail else "no output"
                    if any(m in blob for m in MISSING_INPUT):
                        print("  NEEDS %-46s cannot re-run: %s" % (art, why))
                        needs.append(art)
                        continue
                    print("  FAIL  %-46s script exited %d" % (art, r.returncode))
                    print("        %s" % why)
                    fails.append(art)
                    continue
                same = canonical(keep) == canonical(path)
                print("  %s  %-46s %s"
                      % ("OK  " if same else "DIFF", art,
                         "reproduced" if same else "artefact changed on re-run"))
                if not same:
                    fails.append(art)
            finally:
                os.chmod(path, mode | stat.S_IWUSR)
                shutil.copy2(keep, path)          # the reviewed version, not the re-run
                os.chmod(path, mode)

    print()
    if fails:
        print("  %d artefact(s) did not reproduce: %s" % (len(fails), ", ".join(fails)))
        return 1
    extra = []
    if skips:
        extra.append("%d skipped for a missing artefact" % len(skips))
    if needs:
        extra.append("%d could not re-run for want of an undeposited input" % len(needs))
    # A TOTAL HIDES WHAT IT NEVER SAW. This registry covers 6 scripts; analyses/results/ holds far
    # more artefacts, and an artefact with no CASES entry is not checked at all. Print the coverage
    # beside the total so "all N reproduced" cannot be read as "everything reproduced".
    try:
        _present = len([f for f in os.listdir(NEW) if f.endswith(".json")])
        print("  coverage: %d of the %d artefacts in %s are registered in CASES; the rest are not"
              % (len(CASES), _present, NEW))
        print("            checked by this script at all.")
    except OSError:
        pass
    _ran = len(CASES) - len(skips) - len(needs)
    print("  all %d re-runnable artefacts reproduced%s"
          % (_ran, "; " + ", ".join(extra) if extra else ""))
    if _ran == 0:
        # A VACUOUS PASS IS NOT A PASS. From the archive alone every CASES entry skips or needs
        # an undeposited input, so returning 0 would claim a pass having re-run NOTHING, and
        # tools/run_all_gates.py would count it among the passes.
        # Exit 3 is this archive's "stopped at an undeposited path"
        # convention, which the runner files as NODATA; check_var_off.py and
        # verify_species_gene_structure.py already use it.
        print()
        print("  RESULT: NO DATA -- 0 of %d registered artefacts could be re-run, and that is"
              % len(CASES))
        print("  NOT a pass. Every case above wanted an input this archive does not carry.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
