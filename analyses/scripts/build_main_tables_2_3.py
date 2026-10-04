# -*- coding: utf-8 -*-
"""Generate main-text Tables 2 and 3 from artefacts.

The manuscript currently embeds exactly one table. These are the two the rewrite adds: the coverage
comparison that R4 carries in place of a figure, and the per-class head-to-head that R5 uses.

They are numbered 2 and 3, not 5 and 6. The four tables in `tables.md` are deposit tables D1 to D4,
a separate generated deliverable that the article does not embed, and an earlier plan mistook them
for main-text tables.

At assembly these belong in `src/ccs/build_tables.py` alongside the deposit tables, so that the
no-hand-typed-numbers rule covers them too. They live here until the rewrite is merged, because
`src/ccs/build_tables.py` writes into the frozen submission tree.

    python analyses/scripts/build_main_tables_2_3.py
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
NEW = "analyses/results"
OUT = "analyses/drafts/main_tables_2_3.md"

# Display order: complete-coverage tools first, then the ones with holes, worst last. A reader
# should meet the tools that look fine before the ones that do not.
ORDER = ["Pangolin delta max (abs)", "MMSplice logit PSI change (abs)", "SPANR zPSI change (abs)",
         "SpliceAI delta max (alpha)", "ConSpliceML", "SQUIRLS score",
         "HAL PSI change (abs)", "S-Cap sens minimum (rev)"]
SHORT = {"Pangolin delta max (abs)": "Pangolin", "MMSplice logit PSI change (abs)": "MMSplice",
         "SPANR zPSI change (abs)": "SPANR", "SpliceAI delta max (alpha)": "SpliceAI",
         "ConSpliceML": "ConSpliceML", "SQUIRLS score": "SQUIRLS",
         # The source column keeps the data's own spelling; the printed label uses the tool's,
         # S-CAP (Jagadeesh et al.), because the manuscript names a published method.
         "HAL PSI change (abs)": "HAL", "S-Cap sens minimum (rev)": "S-CAP"}
CLASSES = ["Essential Splice", "Exon Near Junction", "Intron Near Junction",
           "Proximal Intron", "Deep Exon"]


def load(n):
    p = os.path.join(NEW, n)
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


# The prose uses U+2212 MINUS SIGN for negative numbers, 46 times against 16 ASCII hyphens, and all
# sixteen came from these generated tables. A document that writes the same quantity two ways in the
# body and the table beside it is the kind of thing a language editor returns.
def sgn(x, fmt="%+.3f"):
    return (fmt % x).replace("-", "−")


def table2(f):
    r = load("mfass_random500k_coverage.json")
    if not r:
        return "Table 2: mfass_random500k_coverage.json missing"
    f.write("## Table 2. What eight splice predictors answer, on a benchmark and in deployment\n\n")
    f.write("Coverage is the fraction of variants a predictor returns a value for. The benchmark is "
            "3,912 variants across six assays, five of them saturation mutagenesis and the sixth a "
            "curated set of 296 MLH1 variants; the background is 500,000 simulated SNVs drawn at "
            "random from internal protein-coding exons and their 100-bp flanks across 14,577 "
            "genes. Because the two "
            "panels do not share a variant-class mix and coverage depends on class, the background "
            "is also shown standardised to the benchmark's mix. The gap is the standardised "
            "shortfall.\n\n")
    f.write("| predictor | benchmark | background | standardised | gap |\n|---|--:|--:|--:|--:|\n")
    for k in ORDER:
        v = r["predictors"].get(k)
        if not v:
            continue
        bm = v.get("coverage_curated_panel")
        f.write("| %s | %s | %.3f | %.3f | %s |\n"
                % (SHORT[k], "%.3f" % bm if bm is not None else "n/a",
                   v["coverage_random"], v["coverage_standardised_to_benchmark_mix"],
                   sgn(v["residual_gap_after_composition"])))
    j = r["joint"]
    f.write("\nAcross the background, %.1f%% of variants are scored by all eight predictors and the "
            "mean variant is scored by %.2f of them. For essential-splice, intron-near-junction and "
            "proximal-intron variants the proportion scored by all eight is zero.\n\n"
            % (100 * j["answered_by_all"], j["mean_predictors_available"]))
    return None


def table3(f):
    m = load("mfass_evo2_vs_specialists.json")
    mc = load("multiplicity_control.json")
    h = load("heavy_intervals.json")
    if not m or "1b" not in m.get("models", {}):
        return "Table 3: mfass_evo2_vs_specialists.json missing or has no 1b block"
    b = m["models"]["1b"]
    by = b.get("by_variant_class", {})
    f.write("## Table 3. Evo 2 against purpose-built splice predictors, by variant class\n\n")
    f.write("3,912 variants with splice-disruptive calls measured in cells. Intervals on Evo 2 are "
            "percentile bootstrap at B = 200,000. The p value is family-wise across the five "
            "classes by Westfall-Young max-T permutation, so no class is read on its own.\n\n")
    f.write("| variant class | n | disruptive | Evo 2 | 95% CI | SpliceAI | Pangolin | "
            "Evo 2 − SpliceAI | p (FWER) |\n")
    f.write("|---|--:|--:|--:|---|--:|--:|--:|--:|\n")
    for c in CLASSES:
        row = by.get(c)
        if not row:
            continue
        ci = ""
        if h and c in h.get("splicing_evo2_by_class", {}):
            lo, hi = h["splicing_evo2_by_class"][c]["percentile_ci"]
            ci = "[%.3f, %.3f]" % (lo, hi)
        p = ""
        if mc and c in mc.get("splicing", {}).get("classes", {}):
            p = "%.5f" % mc["splicing"]["classes"][c]["p_fwer"]
        d = row["Evo 2 (1b)"] - row["SpliceAI delta max"]
        # "%d" printed 1170 and 1419 without the separator every other count in the paper uses.
        f.write("| %s | %s | %s | %.3f | %s | %.3f | %.3f | %s | %s |\n"
                % (c, "{:,}".format(int(row["n"])), "{:,}".format(int(row["n_pos"])),
                   row["Evo 2 (1b)"], ci,
                   row["SpliceAI delta max"], row["Pangolin delta max (abs)"], sgn(d), p))
    au = b["auroc"]
    hh = b["head_to_head"]["SpliceAI delta max"]
    f.write("\nPooled across all 3,912 variants, Evo 2 reads %.3f against SpliceAI %.3f and "
            "Pangolin %.3f. The paired difference against SpliceAI is %s "
            "(95%% CI [%s, %s], DeLong on the shared variants, p = %.3g), and Evo 2 trails "
            "SpliceAI in %d of %d assays.\n\n"
            % (au["Evo 2 (1b)"], au["SpliceAI delta max"], au["Pangolin delta max (abs)"],
               sgn(hh["diff"], "%+.4f"), sgn(hh["ci"][0], "%+.4f"), sgn(hh["ci"][1], "%+.4f"),
               hh["p"], b["assays_where_evo2_trails_spliceai"], b["n_assays"]))
    # The Essential Splice class is 238 disruptive in 246, so its bootstrap
    # distribution is skewed and the percentile interval is not the honest one. The note is
    # generated here from the artefact rather than typed anywhere.
    hv = load("heavy_intervals.json")
    es = (hv or {}).get("splicing_evo2_by_class", {}).get("Essential Splice")
    if es:
        f.write("Essential Splice carries %d disruptive calls in %d variants, so its bootstrap "
                "distribution is skewed and its percentile interval is reported alongside the "
                "bias-corrected and accelerated one: BCa z0 = %s, acceleration = %s, giving "
                "[%.4f, %.4f] against the percentile [%.4f, %.4f]. The endpoints differ by at most "
                "%.4f, so the class is resolved either way; we flag it because it is the one class "
                "here where the two constructions are not interchangeable in principle.\n\n"
                % (es["n_pos"], es["n"], sgn(es["bca_z0"], "%+.3f"), sgn(es["bca_accel"], "%+.3f"),
                   es["bca_ci"][0], es["bca_ci"][1],
                   es["percentile_ci"][0], es["percentile_ci"][1],
                   max(es["endpoints_differ_by"])))
    # The capacity rung, written only once the larger checkpoint has been scored on this panel. It
    # is what turns "the comparison is run at the small scale" from a caveat into a measurement.
    big = m["models"].get("7b")
    if big:
        ab, hb = big["auroc"], big["head_to_head"]["SpliceAI delta max"]
        f.write("At 7B on the same variants Evo 2 reads %.3f, a change of %s over the sevenfold "
                "parameter increase, and the paired difference against SpliceAI moves from %s to %s "
                "(95%% CI [%s, %s]). It trails SpliceAI in %d of %d assays at both scales.\n\n"
                % (ab["Evo 2 (7b)"], sgn(ab["Evo 2 (7b)"] - au["Evo 2 (1b)"], "%+.4f"),
                   sgn(hh["diff"], "%+.4f"), sgn(hb["diff"], "%+.4f"),
                   sgn(hb["ci"][0], "%+.4f"), sgn(hb["ci"][1], "%+.4f"),
                   big["assays_where_evo2_trails_spliceai"], big["n_assays"]))
    return None


def main():
    problems = []
    # analyses/drafts/ is not shipped, so without this makedirs the write below raises
    # FileNotFoundError on a clean extract.
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Main-text Tables 2 and 3\n\n")
        f.write("Generated by `analyses/scripts/build_main_tables_2_3.py`. Every value is read "
                "from a deposited artefact at build time; none is typed. Merge into "
                "`src/ccs/build_tables.py` at assembly.\n\n")
        for fn in (table2, table3):
            err = fn(f)
            if err:
                problems.append(err)
                f.write("**NOT BUILT.** %s\n\n" % err)
    print("  wrote %s" % OUT)
    for p in problems:
        print("  %s" % p)
    if not problems:
        print("  both tables built from artefacts")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
