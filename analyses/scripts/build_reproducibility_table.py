# -*- coding: utf-8 -*-
"""Build the reviewer-facing reproducibility table: every headline number, its artefact, its check.

A reviewer receives a PDF. They cannot run anything, and a sentence saying that code and data are
available buys nothing, because every paper says it. What they can do in ninety seconds is take one
number from the abstract, find it in a table, and see the file it came from and the check that binds
it there. That converts a claim about rigour into a standing offer to be caught.

The table is generated, never typed. Each row reads its value out of the deposited artefact at build
time, so a row cannot survive the number it describes changing.

The strongest column is the last one. `tools/verify_from_data.py` re-derives these quantities from
raw data and imports nothing from the analysis code, so its checks are an independent
reimplementation rather than a restatement. That distinction is worth a sentence in the legend,
because most reproducibility statements cannot make it.

    python analyses/scripts/build_reproducibility_table.py
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT_MD = "analyses/drafts/reproducibility_table.md"
OUT_JSON = "analyses/results/reproducibility_table.json"


def jget(path, *keys, default=None):
    """Read one value out of an artefact, or return a marker naming what is missing."""
    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except Exception:
        return None, "artefact not readable: %s" % path
    for k in keys:
        try:
            d = d[k]
        except (KeyError, IndexError, TypeError):
            return None, "key %s absent in %s" % ("/".join(map(str, keys)), path)
    return d, None


# claim, artefact, key path, format, the gate that binds it
SPEC = [
    ("Macro AUROC across nine species, 8,192-bp readout",
     "reports/readout_effect_fullpanel.json", ("macro", "auroc_8192"), "%.4f",
     "verify_from_data.py, atlas"),
    ("Macro AUROC, 1,001-bp readout",
     "reports/readout_effect_fullpanel.json", ("macro", "auroc_1001"), "%.4f",
     "verify_from_data.py, atlas"),
    ("Readout effect on identical variants",
     "reports/readout_effect_fullpanel.json", ("macro", "delta"), "%+.4f",
     "verify_from_data.py, atlas"),
    ("Variants carrying both readouts",
     "reports/readout_effect_fullpanel.json", ("n_variants_both_readouts",), "%d",
     "verify_from_data.py, atlas"),
    ("Pooled errors at the 8,192-bp trust layer",
     "reports/trust_layer_8192.json", ("pooled", "errors"), "%d",
     "verify_from_data.py, trust"),
    ("Error capture at 15% refusal, 8,192-bp arm",
     "reports/trust_layer_8192.json", ("pooled", "capture"), "%.4f",
     "verify_from_data.py, trust"),
    ("Macro expected calibration error, 8,192-bp arm",
     "reports/trust_layer_8192.json", ("pooled", "macro_ece"), "%.4f",
     "glmtrust benchmark"),
    ("Pig cis-eQTL, Evo 2, panel as published",
     "analyses/results/eqtl_tss_matched.json",
     ("unmatched_as_published", "auroc_evo2"), "%.4f", "eqtl_tss_matched.py"),
    ("Pig cis-eQTL, positional baseline after matching",
     "analyses/results/eqtl_tss_matched.json",
     ("arm2_within_chromosome_with_caliper", "auroc_tss_proximity"), "%.4f",
     "eqtl_tss_matched.py"),
    ("Splicing panel, Evo 2 (1B)",
     "analyses/results/mfass_evo2_vs_specialists.json",
     ("models", "1b", "auroc", "Evo 2 (1b)"), "%.4f", "verify_new_results.py"),
    ("Splicing panel, SpliceAI",
     "analyses/results/mfass_evo2_vs_specialists.json",
     ("models", "1b", "auroc", "SpliceAI delta max"), "%.4f", "verify_new_results.py"),
    ("Splicing panel, Pangolin",
     "analyses/results/mfass_evo2_vs_specialists.json",
     ("models", "1b", "auroc", "Pangolin delta max (abs)"), "%.4f", "verify_new_results.py"),
    # The 1b path is REQUIRED, not decorative. The top level of clinvar_evo2_participant.json
    # is whichever checkpoint SOURCES found first (7b as deposited), but the paper reports these rungs
    # at 1B (Table S19, "ClinVar consequence rungs at Evo 2-1B"). Reading the top level
    # gives 0.7361/0.7241 against the published 0.7023/0.6497 and would make two correct values
    # look irreproducible.
    ("ClinVar 3' UTR rung, Evo 2",
     "analyses/results/clinvar_evo2_participant.json",
     ("models", "1b", "consequence", "3_prime_UTR_variant", "auroc"), "%.4f", "verify_new_results.py"),
    ("ClinVar 5' UTR rung, Evo 2",
     "analyses/results/clinvar_evo2_participant.json",
     ("models", "1b", "consequence", "5_prime_UTR_variant", "auroc"), "%.4f", "verify_new_results.py"),
    ("Evo 2's own reach penalty on ClinVar",
     "analyses/results/clinvar_evo2_participant.json",
     ("models", "1b", "reach", "penalty"), "%+.4f", "verify_new_results.py"),
    ("S-Cap coverage, six-gene benchmark",
     "analyses/results/mfass_reach.json",
     ("predictors", "S-Cap sens minimum (rev)", "coverage"), "%.4f", "verify_new_results.py"),
    ("S-Cap coverage, 500,000 random variants",
     "analyses/results/mfass_random500k_coverage.json",
     ("predictors", "S-Cap sens minimum (rev)", "coverage_random"), "%.4f",
     "verify_new_results.py"),
    ("Variants answered by all eight splice predictors",
     "analyses/results/mfass_random500k_coverage.json",
     ("joint", "answered_by_all"), "%.4f", "verify_new_results.py"),
    ("Strand-choice AUROC shift, 40B, 8,192 bp",
     "analyses/results/strand_ladder.json",
     ("rungs", "40B", "strand_choice_shift"), "%.4f", "verify_new_results.py"),
    ("Strand-choice shift as a percentage of the 8,192-bp margin",
     "analyses/results/strand_ladder.json",
     ("rungs", "40B", "shift_as_pct_of_8192_margin"), "%.1f%%", "verify_new_results.py"),
]


def main():
    rows, broken = [], []
    for claim, path, keys, fmt, gate in SPEC:
        val, err = jget(path, *keys)
        if err:
            broken.append((claim, err))
            continue
        rows.append((claim, fmt % val, path, gate))

    print("  %-56s %10s  %s" % ("claim", "value", "artefact"))
    for claim, val, path, gate in rows:
        print("  %-56s %10s  %s" % (claim[:56], val, path))
    if broken:
        print("\n  %d row(s) could not be built:" % len(broken))
        for claim, err in broken:
            print("    %-56s %s" % (claim[:56], err))

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with io.open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Reproducibility table\n\n")
        f.write("Generated by `analyses/scripts/build_reproducibility_table.py`. Every value is "
                "read from the artefact named beside it at build time, so a row cannot outlive the "
                "number it describes.\n\n")
        f.write("`tools/verify_from_data.py` re-derives these quantities from raw data and imports "
                "nothing from the analysis code, so its checks are an independent "
                "reimplementation and not a restatement of the pipeline that produced them.\n\n")
        f.write("| claim | value | artefact | bound by |\n|---|---|---|---|\n")
        for claim, val, path, gate in rows:
            f.write("| %s | %s | `%s` | `%s` |\n" % (claim, val, path, gate))
        if broken:
            f.write("\n**Rows that could not be built** (each is a defect, not an omission):\n\n")
            for claim, err in broken:
                f.write("- %s: %s\n" % (claim, err))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"_generated_by": "analyses/scripts/build_reproducibility_table.py",
                   "rows": [{"claim": a, "value": b, "artefact": c, "bound_by": d}
                            for a, b, c, d in rows],
                   "unbuildable": [{"claim": a, "reason": b} for a, b in broken]}, f, indent=2)
    print("\n  %d row(s) built, %d unbuildable; wrote %s" % (len(rows), len(broken), OUT_MD))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
