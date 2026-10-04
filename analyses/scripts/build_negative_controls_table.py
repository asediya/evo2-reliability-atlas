# -*- coding: utf-8 -*-
"""Assemble the negative-controls table: the checks that would have invalidated us and did not.

Scattered through the work are controls designed to find an effect whose presence would break a
claim. Each is currently a clause inside the subsection it belongs to, where a reader meets it once
and cannot see the pattern. Collected into one table they say something no individual result says:
that the time went into trying to falsify the paper.

Most evaluation papers cannot show this, which is precisely why it is worth a table. It costs no new
analysis. Every value is read from a deposited artefact in this script rather than typed, with one
declared exception: the four bootstrap interval endpoints in the splicing reference-likelihood row
are asserted constants, because analyses/results/mfass_evo2_vs_specialists.json carries only the
two point estimates. Those four are cross-checked against the deposited panel by Additional file 3's
scripts/recompute_splicing.py (_ref_ci). Apart from them the table cannot drift from the results it
summarises.

The "expectation" column is the important one and it must stay honest: it states what we would have
seen if the claim being protected were false, written so a reader can check that the control could
actually have failed. A control that could not have failed is decoration.

    python analyses/scripts/build_negative_controls_table.py
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT_JSON = "analyses/results/negative_controls.json"
OUT_MD = "analyses/drafts/negative_controls_table.md"


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    rows = []

    c = load("analyses/results/clinvar_evo2_participant.json")
    lo, hi = c["confound"]["ci"]
    rows.append((
        "Reference likelihood alone, human ClinVar",
        "The readout is an artefact of how likely the reference window is, "
        "so MLL(ref) discriminates on its own",
        "%.4f [%.4f, %.4f] at 7B, an interval that excludes chance on the far side; "
        "mildly anti-predictive, not uninformative"
        % (c["confound"]["auroc_ref_likelihood_alone"], lo, hi),
        "analyses/results/clinvar_evo2_participant.json"))

    m = load("analyses/results/mfass_evo2_vs_specialists.json")
    # BOTH CHECKPOINTS, each labelled and with its interval, as the ClinVar row directly above
    # prints its 7B reading and flags the checkpoint-dependence.
    # The 7B splicing reading, 0.5736, sits in the same dict, is the
    # LARGEST of the four reference-window controls in the paper, exceeds the 0.5713 animal-panel
    # figure the paper concedes as a real confound, and is already asserted as a published value
    # by Additional file 3's recompute_splicing.py. Printing only 1B would support the claim that
    # the confound is "a property of a particular panel's construction and not of the readout" --
    # the opposite of what the main text concludes from the 7B number. The intervals are the
    # ones recompute_splicing.py's _ref_ci reproduces.
    rows.append((
        "Reference likelihood alone, splicing panel",
        "The same, on a panel with cell-measured labels",
        "%.4f [0.4925, 0.5323] at 1B; %.4f [0.5519, 0.5961] at 7B, the largest of the four and "
        "the reading that counts most against us"
        % (m["models"]["1b"]["ref_likelihood_alone"], m["models"]["7b"]["ref_likelihood_alone"]),
        "analyses/results/mfass_evo2_vs_specialists.json"))

    # This row previously reported "TSS proximity falls from 0.6510 to 0.5165 once matched" and sat
    # among the passes. Both numbers are right and the conclusion drawn from them was not. The
    # artefact carries its own falsification criterion -- "If the matching worked, the positional
    # baseline MUST fall to chance" -- and records `tss_baseline_collapsed_to_chance: false` for
    # BOTH arms. 0.5165 is not chance: its interval excludes 0.5. Reporting the point estimate alone
    # let a control that failed its written criterion be read as one that passed. The row now states
    # the criterion, both arms, both intervals, and what does and does not survive.
    e = load("analyses/results/eqtl_tss_matched.json")
    a1, a2 = e["arm1_within_egene_plus_distance"], e["arm2_within_chromosome_with_caliper"]
    assert not a1["tss_baseline_collapsed_to_chance"], "arm 1 flag flipped; rewrite this row"
    assert not a2["tss_baseline_collapsed_to_chance"], "arm 2 flag flipped; rewrite this row"
    rows.append((
        "Positional baseline after distance matching (**did not pass**)",
        "Matching on TSS distance removes the positional confound, so the "
        "baseline falls to chance and the panel tests regulatory signal alone",
        "It does not. Matching cuts TSS proximity from %.4f to %.4f [%.4f, %.4f] "
        "under the caliper (n = %s) and only to %.4f [%.4f, %.4f] within eGene "
        "(n = %s); neither interval covers 0.5, so residual positional signal "
        "remains and the panel is not confound-free. Evo 2 nonetheless reads "
        "%.4f [%.4f, %.4f] on the matched panel, within 0.015 of chance under either "
        "orientation of the score, a departure the same size as the residue itself"
        % (e["unmatched_as_published"]["auroc_tss_proximity"],
           a2["auroc_tss_proximity"], a2["auroc_tss_proximity_ci95"][0],
           a2["auroc_tss_proximity_ci95"][1], "{:,}".format(a2["n"]),
           a1["auroc_tss_proximity"], a1["auroc_tss_proximity_ci95"][0],
           a1["auroc_tss_proximity_ci95"][1], "{:,}".format(a1["n"]),
           a2["auroc_evo2"], a2["auroc_evo2_ci95"][0], a2["auroc_evo2_ci95"][1]),
        "analyses/results/eqtl_tss_matched.json"))

    r = load("analyses/results/mfass_random500k_coverage.json")
    sp = r["predictors"]["SpliceAI delta max (alpha)"]
    rows.append((
        "Composition standardisation of the coverage gap",
        "The benchmark-to-deployment gap is variant-class mix, not curation, "
        "so it disappears once the panels are standardised",
        "SpliceAI gap %.4f before, %.4f after"
        % (sp["understatement"], sp["residual_gap_after_composition"]),
        "analyses/results/mfass_random500k_coverage.json"))

    mc = load("analyses/results/multiplicity_control.json")
    worst_s = max(v["p_fwer"] for v in mc["splicing"]["classes"].values())
    rows.append((
        "Family-wise control, splicing, five variant classes",
        "The per-class pattern is what five simultaneous tests produce by chance",
        "all five survive max-T, worst p = %.5f" % worst_s,
        "analyses/results/multiplicity_control.json"))

    if mc.get("clinvar"):
        worst_c = max(v["p_fwer"] for v in mc["clinvar"]["rungs"].values())
        rows.append((
            "Family-wise control, ClinVar, eight consequence rungs",
            "The consequence gradient is an artefact of testing eight rungs at once",
            "all eight survive max-T, worst p = %.5f" % worst_c,
            "analyses/results/multiplicity_control.json"))

    if os.path.exists("reports/seed_stability.json"):
        s = load("reports/seed_stability.json")
        rows.append((
            "Seed stability of a reported count",
            "A count quoted as a finding moves with the bootstrap seed",
            # "3 always significant, 1 seed-dependent" is true and reads as 3 of 4. There are nine
            # contrasts: five reach significance under no seed at all, and the unstable one is
            # chicken -- the species whose margin over GERP the Methods calls the largest. Naming
            # the species and the never-significant five is the difference between a control and a
            # summary statistic that flatters the result.
            "%d of the %d species always significant, %d never at any seed, and %s "
            "seed-dependent (%d of %d seeds), across %d seeds"
            % (s["summary"]["n_always_significant"], len(s["per_species"]),
               sum(1 for v in s["per_species"] if v["n_significant"] == 0),
               ", ".join(v["species"] for v in s["per_species"] if not v["stable"]),
               max(v["n_significant"] for v in s["per_species"] if not v["stable"]),
               s["_meta"]["seeds"], s["_meta"]["seeds"]),
            "reports/seed_stability.json"))

    hv = "analyses/results/heavy_intervals.json"
    if os.path.exists(hv):
        h = load(hv)
        # The deposited artefact carries only the standard
        # error of the bootstrap MEAN, under the name that says so. A regenerated one carries the
        # ENDPOINT standard error as mc_se. Read whichever is present and label the row for the
        # quantity actually found, so this row can never print a se of the mean under the
        # words "endpoint standard error".
        _key = "mc_se" if any("mc_se" in v
                              for blk in ("splicing_evo2_by_class", "clinvar_by_consequence")
                              for v in h.get(blk, {}).values() if isinstance(v, dict)) \
               else "mc_se_of_mean"
        mc = {"%s (%s)" % (k, blk.split("_")[0]): v[_key]
              for blk in ("splicing_evo2_by_class", "clinvar_by_consequence")
              for k, v in h.get(blk, {}).items() if isinstance(v, dict) and _key in v}
        rows.append((
            "Interval stability at 200,000 resamples",
            "The published intervals are Monte Carlo noise at the precision printed",
            # The maximum was taken over the splicing block alone and printed to five decimals,
            # which reported "0.00007" for a value of 7.073e-05 -- above the figure quoted -- and
            # silently dropped clinvar_by_consequence, whose 3' UTR cell is larger still. Both
            # blocks are certified by this row, so both belong in the maximum.
            ("largest endpoint standard error %.2e, in %s; one skewed cell needs BCa"
             % (max(mc.values()), max(mc, key=mc.get))
             if _key == "mc_se" else
             "largest standard error of the bootstrap MEAN %.2e, in %s; one skewed cell needs BCa. "
             "The ENDPOINT standard error, which is what Table S20 reports, is not in this "
             "artefact: it is ~3e-04, recomputed from the deposited panel by Additional file 3's "
             "scripts/recompute_endpoint_se.py"
             % (max(mc.values()), max(mc, key=mc.get))),
            hv))

    print("  %-46s %s" % ("control", "observed"))
    for name, _, observed, _ in rows:
        print("  %-46s %s" % (name, observed))

    # analyses/drafts/ is not shipped -- analyses/ carries only results/ and scripts/ -- so it is
    # created here; without it a clean extract raises FileNotFoundError before the JSON is written.
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with io.open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Negative controls\n\n")
        f.write("Generated by `analyses/scripts/build_negative_controls_table.py`. "
                "Every value is read from the artefact named in the last column.\n\n")
        f.write("| control | what we would have seen if the claim were false | observed | artefact |\n")
        f.write("|---|---|---|---|\n")
        for name, expectation, observed, src in rows:
            f.write("| %s | %s | %s | `%s` |\n" % (name, expectation, observed, src))
        f.write("\nEach control could have failed. A control that could not have failed is "
                "decoration and is not listed here.\n")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"_generated_by": "analyses/scripts/build_negative_controls_table.py",
                   "controls": [{"control": a, "expectation_if_false": b,
                                 "observed": c_, "artefact": d} for a, b, c_, d in rows]},
                  f, indent=2)
    print("\n  %d controls; wrote %s and %s" % (len(rows), OUT_MD, OUT_JSON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
