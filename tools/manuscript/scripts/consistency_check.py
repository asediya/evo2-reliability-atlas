"""Internal cross-consistency checker: does the manuscript agree with ITSELF?

Motivation. The existing `verify-numbers` check compares manuscript numbers against
the recompute layer (COMPILED_RESULTS + the deposited JSONs) — i.e. manuscript-vs-source. It does not
compare manuscript numbers against EACH OTHER, and it barely covers figure legends. Internal
numerical contradictions, including those in figure legends, are mechanically
detectable. This script detects them.

Method. For each registered QUANTITY we hold a list of regexes. Every match anywhere in the
manuscript (including figure legends), the supplementary prose and the supplementary tables is
collected with its line number. If a quantity resolves to more than one distinct value, that is a
contradiction unless the quantity is explicitly declared multi-valued (READOUT_SPLIT), in which case
we assert the exact expected value set instead.

The manuscript is not deposited in Additional file 2 (two copies would eventually diverge), so it
must be named on the command line. The supplementary prose and tables ship under the names this
file expects and are picked up automatically:

    python tools/manuscript/scripts/consistency_check.py --manuscript path/to/Manuscript.md

Exit 0 clean, 1 on a contradiction, 2 on a missing input.
"""
import io
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FILES = [
    "reports/manuscript.md",
    "reports/supplementary_prose.md",
    "reports/supplementary_tables.md",
]

# name -> (list of regexes, expected distinct values or None)
# A quantity legitimately taking >1 value must declare the full set, so a NEW stray value still fails.
QUANTITIES = {
    # The second pattern used to be `eQTL[^.]{0,80}?AUROC[^.]{0,20}?(0\.\d{3})`, which also caught
    # sentences reporting a CHANGE in AUROC rather than a level: "on the eQTL subpanel a 2x window
    # at a fixed readout moves AUROC by 0.002" registered 0.002 as a second value for this quantity
    # and failed the gate on a sentence that is not about it. Requiring reads/of/at/is in front of
    # the number keeps it to statements of a level.
    "eQTL AUROC (20,000-variant panel, 1,001 bp)": (
        [r"0\.488",
         r"eQTL[^.]{0,80}?AUROC[^.]{0,20}?(?:reads?|of|at|is)\s+\*{0,2}(0\.\d{3})"], {"0.488"}),
    "eQTL AUROC, consequence-matched 2,000 subpanel": ([r"0\.498"], {"0.498"}),
    "atlas macro AUROC (8,192 bp)": ([r"macro AUROC of (0\.\d{3})", r"macro (0\.943)"], {"0.943"}),
    # Required "is", which neither form in the manuscript uses ("a pooled value of **0.973**",
    # "Pooled AUROC 0.973 [").
    "atlas pooled AUROC (8,192 bp)": (
        [r"[Pp]ooled AUROC \*{0,2}(0\.\d{3})", r"pooled value of \*{0,2}(0\.\d{3})"], {"0.973"}),
    # 8,192-bp panel (11,109 variants): cattle 0.9745 is the maximum
    # and pig 0.8633 the minimum, both computed on the SAME variant set Figures 9a and 10a use — NOT
    # the GERP-restricted head-to-head set.
    # The manuscript says "running from **0.974** in cattle down to **0.863** in pig", not
    # "0.974 (cattle)". A registered quantity that silently fails to match is worse than an
    # unregistered one: it reads as scanned. See the not-found guard in main().
    "atlas max per-species AUROC (cattle)": (
        [r"from \*{0,2}(0\.\d{3})\*{0,2} in cattle"], {"0.974"}),
    "atlas min per-species AUROC (pig)": (
        [r"to \*{0,2}(0\.\d{3})\*{0,2} in pig"], {"0.863"}),
    "GERP macro in the 8,192 head-to-head": (
        [r"macro 0\.939 against (0\.\d{3})", r"GERP is unchanged at (0\.\d{3})"], {"0.831"}),
    # Two DECLARED values: +0.065 is the species mean
    # (this paper's primary aggregation) and +0.092 is pooled. Both must appear with the
    # aggregation named; a third value, or either one unlabelled, is a contradiction.
    "readout effect (atlas, matched variants)": (
        [r"worth \+(0\.\d{3}) AUROC", r"the \+(0\.\d{3}) AUROC difference between them"], {"0.065"}),
    "readout effect, pooled": ([r"\+\*?\*?(0\.092)\*?\*? pooled"], {"0.092"}),
    # The coding-vs-eQTL fold ratio is not registered. Registering it would make
    # this checker DEFEND a number the Results explicitly refuse to quote ("we do not quote a
    # ratio, because the denominator sits close to zero"). The ratio is absent from every deliverable.
    # The calibration-transfer ECE bound is not registered: it is stated in the supplement as a
    # bound rather than as a single
    # value (Note S48: "bounded above by approximately 0.014 ECE"), and a checker that demands one
    # exact value for a quantity the text deliberately reports as an upper bound counts a correct
    # manuscript as a contradiction. Note S48 carries the four estimator
    # intervals it would otherwise summarise.
    "BRCA1 AUROC (8,192-bp mean-LL)": ([r"BRCA1[^.]{0,60}?\((0\.874)\)", r"reached AUROC (0\.874)"], {"0.874"}),
    "orthogonal contribution (pooled dFM)": ([r"pooled ΔFM of \*?\*?\+(0\.\d{4})"], {"0.0202"}),
    # Two DIFFERENT estimands, not to be conflated:
    # +0.0485 is the species-mean vs the best conservation baseline on the 9,532-variant co-scoreable
    # panel at 1,001 bp; +0.047 is the within-set margin vs GERP on the 9,529 variants carrying BOTH an
    # 8,192-bp score and a finite GERP value. The two sets differ by three variants and are not the same
    # analysis; do not merge them.
    # Subtracting Table 1's two columns (Evo 2 full panel
    # minus GERP scoreable subset) gives the reach-mismatched comparison Table 1's caption forbids.
    # The co-scoreable species-mean is +0.047 and is tracked below.
    "1,001-bp margin vs GERP (co-scoreable set)": (
        [r"raises its margin over GERP from \+(0\.\d{3})"], {"0.047"}),
    # Two legitimately different sets: the full panel, and the subset carrying an 8,192-bp score.
    "atlas full panel N": ([r"(11,130) variants"], {"11,130"}),
    "atlas 8,192-bp panel N": ([r"n = (11,109)"], {"11,109"}),
    "window size, 2 kb rung": ([r"(2,04[89]) ?bp"], {"2,049"}),
}


FLAGS = {"--manuscript": FILES[0], "--prose": FILES[1], "--tables": FILES[2]}


def resolve(argv):
    """Map each required input to a path.

    Two forms are accepted. `--manuscript PATH` names the slot explicitly and is the form to use for
    the manuscript, which is not deposited here and so never carries the basename this file expects.
    A bare positional argument substitutes by basename, which works for the two supplementary files
    because they ship under their own names.
    """
    paths = {f: f for f in FILES}
    extra = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in FLAGS:
            if i + 1 >= len(argv):
                raise SystemExit("%s needs a path" % a)
            paths[FLAGS[a]] = argv[i + 1]
            i += 2
            continue
        if "=" in a and a.split("=", 1)[0] in FLAGS:
            k, v = a.split("=", 1)
            paths[FLAGS[k]] = v
            i += 1
            continue
        if a.startswith("-"):
            i += 1
            continue
        base = os.path.basename(a)
        for f in FILES:
            if os.path.basename(f) == base:
                paths[f] = a
                break
        else:
            extra.append(a)
        i += 1
    return paths, extra


def load(paths, extra=()):
    docs = {}
    for p in list(paths.values()) + list(extra):
        if os.path.exists(p) and p not in docs:
            docs[p] = io.open(p, encoding="utf-8").read().splitlines()
    return docs


def main():
    argv = sys.argv[1:]
    paths, extra = resolve(argv)
    # A path given on the command line must exist and is actually read, and the refusal below
    # applies whatever arguments are passed. Without both, a run could scan whatever subset
    # happens to be present and print
    # "0 contradiction(s)" with exit 0 -- exactly the vacuous pass the README promises
    # cannot happen.
    ghosts = [a for a in argv if not a.startswith("-") and not os.path.exists(a)]
    missing = [r for r, p in paths.items() if not os.path.exists(p)]
    if ghosts or missing:
        if ghosts:
            print("MISSING INPUT(S) named on the command line: %s" % ", ".join(ghosts))
        if missing:
            print("MISSING INPUT(S): %s" % ", ".join(missing))
        print("This check requires all of %s. Refusing to report a partial scan as clean: run from "
              "the study repository root with every input present, or pass an explicit path for "
              "each one." % ", ".join(FILES))
        return 2
    docs = load(paths, extra)
    if not docs:
        print("no input files found"); return 1
    print("Scanning %d file(s): %s\n" % (len(docs), ", ".join(os.path.basename(f) for f in docs)))

    bad = 0
    for name, (pats, expected) in QUANTITIES.items():
        found = {}          # value -> [(file, lineno)]
        for f, lines in docs.items():
            for i, line in enumerate(lines, 1):
                for p in pats:
                    for m in re.finditer(p, line):
                        v = m.group(1) if m.groups() else m.group(0)
                        found.setdefault(v, []).append((os.path.basename(f), i))
        if not found:
            # A registered quantity that matches nothing is not a pass. Silence is not agreement:
            # either the quantity is stated and the pattern is stale, or it is no longer stated
            # and the entry should be removed. Both need a human, so both fail the gate.
            print("  --  %-52s (NOT FOUND: pattern stale, or the quantity is gone)" % name)
            bad += 1
            continue
        vals = set(found)
        ok = (vals == expected) if expected is not None else (len(vals) == 1)
        if ok:
            print("  OK  %-52s %s" % (name, ", ".join(sorted(vals))))
        else:
            bad += 1
            exp = ("expected {%s}" % ", ".join(sorted(expected))) if expected else "expected a single value"
            print("  ** CONTRADICTION  %s\n        %s, found {%s}" % (name, exp, ", ".join(sorted(vals))))
            for v in sorted(vals):
                loc = "; ".join("%s:%d" % x for x in found[v][:6])
                print("          %-10s <- %s" % (v, loc))
    print("\n%d contradiction(s)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
