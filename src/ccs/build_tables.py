"""Paper tables — built from the figure recompute layers, never typed by hand.

Run: python -m src.ccs.build_tables
Emits: reports/tables.json   (machine-readable, one entry per table)
       reports/tables.md     (submission-ready markdown, GB style)

WHY A RECOMPUTE LAYER AND NOT A HAND-WRITTEN TABLE
A table transcribed by hand can drift from the data unnoticed; every table here is derived instead.

Every number below is read from a JSON produced by a figure's stats module, and the headline values are
ASSERTED against literals held in _assert_canonical so that a silent drift fails the build
instead of reaching a table. That function opens no file; see its docstring.

THE TWO PANELS ARE NOT THE SAME PANEL. Section 1 (8192 bp mean-LL) is n = 11,109 over 9 species; the
GERP-comparable set (1001 bp) is n = 11,130. Table 1 carries both, in labelled columns, because the
readout is worth +0.092 AUROC pooled (section 2) and mixing them is the single easiest way to
publish a wrong number here.
"""
import json
import os
from pathlib import Path
import sys

# A referee on a cp437 console or with LC_ALL=C otherwise gets a traceback and a nonzero
# exit from a run that succeeded; build_tables.py even wrote its outputs first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# amniote tree topology, matching the figures -- NOT sorted by n or by any plotted value
TREE = ["chicken", "human", "dog", "cat", "horse", "pig", "cattle", "sheep", "goat"]
CLADE = {"human": "primate", "dog": "carnivore", "cat": "carnivore", "horse": "perissodactyl",
         "pig": "suid", "cattle": "ruminant", "sheep": "ruminant", "goat": "ruminant",
         "chicken": "bird"}


def _load(name):
    return json.load(open(ROOT / "reports" / name, encoding="utf-8"))


def _f(v, nd=3, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _ci(lo, hi, nd=3):
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def table1_atlas(F2, F6):
    """Cross-species reliability atlas. BOTH panels and BOTH readouts, explicitly labelled."""
    forest = {r["species"]: r for r in F2["forest"]}
    gerp = F2["gerp"]
    dfm = {r["sp"]: r for r in F6["decomposition"]["per_species"]}
    rows = []
    for sp in TREE:
        f, g = forest.get(sp, {}), gerp.get(sp, {})
        d = dfm.get(sp, {})
        rows.append({
            "species": sp, "clade": CLADE[sp],
            "n_8192": f.get("n"), "auroc_8192": f.get("auroc"),
            "ci_8192": [f.get("lo"), f.get("hi")],
            "auroc_1001": f.get("auroc_1001"), "readout_delta": f.get("readout_delta"),
            "n_gerp_panel": g.get("n"), "evo2_gerp_panel": g.get("evo2"), "gerp": g.get("gerp"),
            "delta_vs_gerp": g.get("delta"), "delta_ci": [g.get("lo"), g.get("hi")],
            "verdict": g.get("verdict"),
            "d_fm": d.get("d_fm"), "n_decomp": d.get("n"),
        })
    return {
        "id": "T1", "title": "Cross-species reliability atlas for Evo 2-40B",
        "rows": rows,
        "pooled_8192": F2["pooled"], "readout_pooled": F2["readout_pooled"],
        "d_fm_pooled": F6["decomposition"]["pooled"],
        "_note": ("The 8192 bp columns and the GERP-comparable columns are DIFFERENT PANELS with "
                  "different n. The readout is worth +0.092 AUROC pooled, so the two must never be "
                  "compared across columns. dFM is the estimator-consistent value (all three AUROC "
                  "terms from one 5-fold CV logistic), not the deposited decomposition table's."),
    }


def table2_trust(F4):
    """Trust layer: calibration, the null, abstention, conformal. All MACRO unless labelled."""
    grid = F4["ece_grid"]
    ESTS = ("width10", "width15", "mass10", "mass15")

    # `platt_LOSO` and `trivial_sigmoid_LOSO` are BIT-IDENTICAL in this file — macro, pooled and every
    # per-species value, under all four binning estimators. That is not a coincidence and not a bug:
    # transferring a Platt map across species IS fitting a two-parameter global sigmoid, so they are one
    # procedure under two names. Listing them as separate rows (as COMPILED_RESULTS 6a still does)
    # invites a reader to think two different methods happen to tie, which is the same misreading
    # section 6c documents for the deposited table. Collapsed to one labelled row.
    ident = all(grid[e].get("platt_LOSO", {}).get("macro") ==
                grid[e].get("trivial_sigmoid_LOSO", {}).get("macro") for e in ESTS)
    methods = ([("isotonic_LOSO", "isotonic transfer (LOSO)"),
                ("platt_LOSO", "Platt transfer = trivial global sigmoid (LOSO)"),
                ("oracle_isotonic", "in-species oracle (unattainable bound)")] if ident else
               [("isotonic_LOSO", "isotonic transfer (LOSO)"), ("platt_LOSO", "Platt transfer (LOSO)"),
                ("trivial_sigmoid_LOSO", "trivial global sigmoid (LOSO)"),
                ("oracle_isotonic", "in-species oracle (unattainable bound)")])
    cal = []
    for key, label in methods:
        row = {"method": label, "key": key}
        for est in ESTS:
            blk = grid.get(est, {}).get(key)
            row[est] = blk.get("macro") if isinstance(blk, dict) else None
        cal.append(row)

    # "a" and "b" are the two sides of isotonic_LOSO_vs_trivial, so a bare "a_better" reached the
    # rendered table as an untranslated enum while its three neighbours read "indistinguishable" --
    # a reader had no way to tell which side "a" was, on the one row where the comparison is not
    # null. ECE is an error, so a negative delta means the first-named method is the lower-error one;
    # the direction is asserted against the sign below rather than trusted.
    VERDICT = {"a_better": "isotonic transfer better",
               "b_better": "trivial sigmoid better",
               "ns": "indistinguishable",
               "indistinguishable": "indistinguishable"}
    null = []
    for est in ("width10", "width15", "mass10", "mass15"):
        v = F4["paired_verdicts"].get(f"isotonic_LOSO_vs_trivial__{est}__macro")
        if v:
            raw = v["verdict"]
            if raw == "a_better":
                assert v["delta"] < 0, ("verdict says isotonic transfer wins but its Delta ECE is "
                                        "%+.4f; ECE is an error, so a win must be negative." % v["delta"])
            if raw == "b_better":
                assert v["delta"] > 0, ("verdict says the trivial sigmoid wins but Delta ECE is "
                                        "%+.4f." % v["delta"])
            null.append({"estimator": est, "delta": v["delta"], "ci": [v["lo"], v["hi"]],
                         "verdict": VERDICT.get(raw, raw)})

    ab = F4["abstention"]
    honest, rand = ab["isotonic_LOSO"], ab["isotonic_LOSO"]
    absten = [{"coverage": c, "macro_error": e, "random_control": r}
              for c, e, r in zip(honest["coverage"], honest["macro_error"], rand["macro_random"])]

    C = F4["conformal"]
    conf = [{"species": s, "n": n, "cov_path_marginal": cm, "abstain_marginal": am,
             "cov_path_mondrian": cd, "abstain_mondrian": ad}
            for s, n, cm, am, cd, ad in zip(C["sp"], C["n"], C["cov_path_marg"], C["abstain_marg"],
                                            C["cov_path"], C["abstain"])]
    return {
        "id": "T2", "title": "What the trust layer delivers, and what it costs",
        "calibration_macro_ece": cal, "paired_null_macro": null,
        "abstention": absten, "conformal": conf,
        "_note": ("Aggregation is MACRO throughout: pooling lets human (n=3,000 at 50% prevalence) "
                  "dominate a cohort otherwise near 9%. The abstention arm is the honest LOSO one — "
                  "NOT the in-species oracle, which is an unattainable bound. Any conformal claim is "
                  "the MONDRIAN arm; the marginal arm under-covers the positive class and is a "
                  "documented artifact. Platt transfer and the trivial global sigmoid are ONE "
                  "estimator, bit-identical here under every binning rule and in every species — "
                  "transferring a Platt map across species IS fitting two global parameters."),
    }


def table3_reach(F5):
    """Reach: what each method can be RUN on."""
    rows = []
    for sp in TREE:
        r = next((x for x in F5["per_species"] if x["species"] == sp), None)
        if not r:
            continue
        ci = r.get("class_gap_ci") or [None, None]
        zb, zp = r.get("runs_z_benign"), r.get("runs_z_pathogenic")
        rows.append({
            "species": sp, "n": r["n"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
            "reach_neg": r["reach_neg"], "reach_pos": r["reach_pos"],
            "class_gap": r["class_gap"], "class_gap_ci": ci,
            "gap_significant": bool(ci[0] is not None and ci[0] > 0),
            "must_call_cost": r["must_call_penalty"],
            "holes_clustered": bool((zb or 0) < -2 or (zp or 0) < -2),
        })
    C = F5["coding_confound"]
    # The matched verdict is read from the per-species intervals, not typed: a typed "in ANY species"
    # goes false the first time an interval excludes zero in either direction.
    _mci = [(x["species"], x["gap_matched_ci"]) for x in C["per_species"] if x.get("gap_matched_ci")]
    _pos = [s_ for s_, ci_ in _mci if ci_[0] > 0]
    _neg = [s_ for s_, ci_ in _mci if ci_[1] < 0]
    _conf = ("matched on consequence category the gap stays positive and separable from zero in "
             + ", ".join(_pos)) if _pos else ("matched on consequence category no species keeps a "
                                             "positive gap separable from zero")
    _conf += (", and it reverses in " + ", ".join(_neg)) if _neg else ""
    return {
        "id": "T3", "title": "Reach — what conservation can be run on, and what the silence costs",
        "rows": rows,
        "n_gap_significant": sum(1 for r in rows if r["gap_significant"]),
        "confound": {"n_matchable": C["n_matchable"], "n_sig_matched": C["n_sig_matched"],
                     "unmatchable": C["unmatchable_species"], "unannotated": C["unannotated_species"]},
        "_note": ("Missingness in the GERP tracks is NaN, not null — notna() reports 100% reach and "
                  "deletes this entire result. 'Must-call cost' is the AUROC lost when every variant "
                  "must be answered, from the closed-form pairwise abstention null (a pair touching a "
                  "no-call contributes 0.5); it is NOT median imputation. THE CONFOUND IS NOT "
                  "RESOLVED: " + _conf + "."),
    }


def table4_baselines(F5):
    """Baseline suite, read through Table 3: every conservation AUROC is on a class-selected subset."""
    rows = []
    for sp in TREE:
        r = next((x for x in F5["per_species"] if x["species"] == sp), None)
        if not r:
            continue
        rows.append({"species": sp, "n": r["n"],
                     "evo2_covered": r["auroc_evo2_covered"], "gerp_covered": r["auroc_gerp_covered"],
                     "gerp_abstain": r["auroc_gerp_full_abstain"],
                     "evo2_full": r["auroc_evo2_full"]})
    sup = {r["species"]: r for r in F5["supervised"]["rows"]}
    cnn = [{"species": s, "cnn_within": sup[s]["auroc_cnn"], "cnn_loso": sup[s]["cnn_loso"],
            "kmer_within": sup[s].get("auroc_kmer"), "evo2": sup[s]["auroc_evo2"]}
           for s in TREE if s in sup]
    P = F5["protein_lm"]
    return {
        "id": "T4", "title": "Baseline suite — conservation, a peer DNA-LM, a protein LM, and supervised models",
        "conservation": rows, "supervised": cnn,
        "pooled_shared": F5["pooled_shared"],
        "nt_reach": F5["nt_reach"],
        "protein_lm": {"human": P["human"], "n_evaluable": P["n_evaluable"],
                       "n_annotated": P["n_annotated"], "n_esm_usable": P["n_esm_usable"]},
        "_note": ("Every GERP AUROC here is measured on the subset GERP can score — see Table D3. The "
                  "'abstain' column is the same AUROC once unscorable variants must be answered. "
                  "Supervised rows are point estimates only: the baseline scripts retain no "
                  "per-variant predictions, so paired intervals are not computable from the deposit. "
                  "Goat has no CNN run."),
    }


def _assert_canonical(T1, T2, T3, T4):
    """Assert the eight headline values against literals held here. A drift fails the build.

    The expected values are written into this function, not read from a file. Earlier messages
    said "COMPILED_RESULTS says ...", which named a document this module never opens and
    implied a link that does not exist. What the gate catches is drift in the recompute-layer
    JSONs against the numbers the manuscript prints; reconciling the two is a manual step.
    """
    checks = [
        ("pooled 8192 AUROC", T1["pooled_8192"]["auroc"], 0.973, 5e-4),
        ("readout delta", T1["readout_pooled"]["delta"], 0.092, 5e-4),
        ("pooled dFM", T1["d_fm_pooled"]["d_fm"], 0.0202, 5e-3),
        ("pooled shared GERP", T4["pooled_shared"]["auroc_gerp"], 0.8780, 5e-4),
        ("pooled shared Evo2", T4["pooled_shared"]["auroc_evo2"], 0.8805, 5e-4),
        ("gap significant in", T3["n_gap_significant"], 4, 0.1),
        ("matched-confound significant in", T3["confound"]["n_sig_matched"], 0, 0.1),
    ]
    iso = next(r for r in T2["calibration_macro_ece"] if r["key"] == "isotonic_LOSO")
    checks.append(("isotonic macro ECE (width10)", iso["width10"], 0.0533, 5e-4))
    bad = [(n, got, want) for n, got, want, tol in checks
           if got is None or abs(got - want) > tol]
    for n, got, want in bad:
        print(f"  !! DRIFT: {n} = {got}, expected {want}")
    assert not bad, "table values drifted from the published figures — reconcile before publishing"
    print(f"  {len(checks)} canonical values asserted, all match")


def _md(T1, T2, T3, T4):
    L = []
    L.append("# Deposit tables — generated, do not edit by hand\n")
    L.append("Built by `src/ccs/build_tables.py` from the figure recompute layers; headline values are "
             "asserted against literals in build_tables.py at build time. Regenerate rather than "
             "patch.\n")
    # These are NOT the tables printed in the manuscript. The D prefix makes the deposit tables
    # self-identifying without renumbering anything in the manuscript.
    # The manuscript's Table 1 is the eight-splice-predictor reach table,
    # five columns, no species and no readout; the six-column cross-species table is Table 2, and
    # it reports BOTH readouts. The two tables also differ in panel: Table 2's per-species n are
    # the locus-clustered panels (goat 99, horse 781, cat 1,365, cattle 2,068, dog 2,497) against
    # D1's (98, 766, 1,362, 2,067, 2,496), so the point estimates coincide at 3 dp while the
    # intervals do not. Do not describe them as the same column.
    L.append("**These are the deposit's own tables, D1-D4, and they are not the tables printed in the "
             "manuscript.** They present the same recompute layer more fully -- Table D1 carries the "
             "8,192-bp panel with fourteen columns; the manuscript's eight-column Table 2 reports "
             "the same nine species at both readouts against GERP, on slightly different "
             "per-species panels, so its point estimates agree with D1 at three decimals while "
             "its intervals do not.\n")

    L.append(f"\n## Table D1. {T1['title']}\n")
    p = T1["pooled_8192"]; rp = T1["readout_pooled"]; dp = T1["d_fm_pooled"]
    L.append(f"Pooled at 8192 bp mean-LL: **{p['auroc']:.3f}** [{p['lo']:.3f}, {p['hi']:.3f}], "
             f"n = {p['n']:,}. Readout lift 1001 bp → 8192 bp: **{rp['delta']:+.3f}** "
             f"[{rp['ci'][0]:+.3f}, {rp['ci'][1]:+.3f}]. Pooled ΔFM **{dp['d_fm']:+.4f}** "
             f"[{dp['d_fm_ci'][0]:+.4f}, {dp['d_fm_ci'][1]:+.4f}].\n")
    # The two "95% CI" columns belong to different quantities measured on different panels: the
    # first to the 8,192-bp AUROC, the second to the Evo 2 minus GERP difference on the smaller
    # GERP-scoreable panel. Repeating the bare label made them look interchangeable and left a
    # reader no way to tell which interval attached to which estimate.
    L.append("| species | clade | 8192 bp panel n | AUROC @8192 | 95% CI (AUROC @8192) | "
             "AUROC @1001 | readout Δ | GERP panel n | Evo 2 | GERP | Δ vs GERP | "
             "95% CI (Δ vs GERP) | verdict | ΔFM |")
    L.append("|---|---|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|--:|")
    for r in T1["rows"]:
        L.append(f"| {r['species']} | {r['clade']} | {r['n_8192']:,} | {_f(r['auroc_8192'])} | "
                 f"{_ci(*r['ci_8192'])} | {_f(r['auroc_1001'])} | {_f(r['readout_delta'], 3, True)} | "
                 f"{r['n_gerp_panel']:,} | {_f(r['evo2_gerp_panel'])} | {_f(r['gerp'])} | "
                 f"{_f(r['delta_vs_gerp'], 3, True)} | {_ci(*r['delta_ci'])} | {r['verdict']} | "
                 f"{_f(r['d_fm'], 4, True)} |")
    L.append(f"\n*{T1['_note']}*\n")

    L.append(f"\n## Table D2. {T2['title']}\n")
    L.append("**2a — calibration, macro ECE by binning estimator**\n")
    L.append("| method | equal-width 10 | equal-width 15 | equal-mass 10 | equal-mass 15 |")
    L.append("|---|--:|--:|--:|--:|")
    for r in T2["calibration_macro_ece"]:
        L.append(f"| {r['method']} | {_f(r['width10'], 4)} | {_f(r['width15'], 4)} | "
                 f"{_f(r['mass10'], 4)} | {_f(r['mass15'], 4)} |")
    L.append("\n**2b — the headline is a NULL: isotonic transfer vs a trivial 2-parameter sigmoid**\n")
    L.append("| estimator | Δ ECE | 95% CI | verdict |")
    L.append("|---|--:|:--|:--|")
    for r in T2["paired_null_macro"]:
        L.append(f"| {r['estimator']} | {_f(r['delta'], 4, True)} | {_ci(*r['ci'], 4)} | {r['verdict']} |")
    L.append("\n**2c — abstention (honest LOSO, macro error)**\n")
    L.append("| coverage | " + " | ".join(f"{a['coverage']:.2f}" for a in T2["abstention"]) + " |")
    L.append("|---|" + "--:|" * len(T2["abstention"]))
    L.append("| macro error | " + " | ".join(_f(a["macro_error"], 4) for a in T2["abstention"]) + " |")
    L.append("| random control | " + " | ".join(_f(a["random_control"], 4) for a in T2["abstention"]) + " |")
    L.append("\n**2d — conformal: report the Mondrian arm**\n")
    # Both "abstains" columns carried the same label while reporting different constructions, so the
    # marginal and Mondrian abstention rates -- which differ substantially, and whose difference is
    # the point of this table -- were indistinguishable by header alone.
    L.append("| species | n | positive-class coverage (marginal) | abstains (marginal) | "
             "positive-class coverage (Mondrian) | abstains (Mondrian) |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for r in T2["conformal"]:
        L.append(f"| {r['species']} | {r['n']:,} | {_f(r['cov_path_marginal'])} | "
                 f"{_f(r['abstain_marginal'], 2)} | {_f(r['cov_path_mondrian'])} | "
                 f"{_f(r['abstain_mondrian'], 2)} |")
    L.append(f"\n*{T2['_note']}*\n")

    L.append(f"\n## Table D3. {T3['title']}\n")
    L.append("| species | n | negatives scorable | positives scorable | class gap | 95% CI | "
             "CI excludes 0 | must-call cost | holes clustered |")
    L.append("|---|--:|--:|--:|--:|:--|:-:|--:|:-:|")
    for r in T3["rows"]:
        L.append(f"| {r['species']} | {r['n']:,} | {r['reach_neg']:.3f} | {r['reach_pos']:.3f} | "
                 f"{_f(r['class_gap'], 3, True)} | {_ci(*r['class_gap_ci'])} | "
                 f"{'yes' if r['gap_significant'] else 'no'} | −{r['must_call_cost']:.3f} | "
                 f"{'yes' if r['holes_clustered'] else 'no'} |")
    c = T3["confound"]
    _un = (" and %s is unannotated" % ", ".join(c["unannotated"])) if c["unannotated"] else ""
    L.append(f"\nCI excludes zero in **{T3['n_gap_significant']} of 9**. Consequence-matched control: "
             f"significant in **{c['n_sig_matched']} of {c['n_matchable']}** matchable species; "
             f"{', '.join(c['unmatchable'])} have no matchable stratum{_un}.\n")
    L.append(f"*{T3['_note']}*\n")

    L.append(f"\n## Table D4. {T4['title']}\n")
    ps = T4["pooled_shared"]; nt = T4["nt_reach"]; P = T4["protein_lm"]
    L.append(f"Over the **{ps['n']:,}** variants both Evo 2 and conservation can score it is a tie: "
             f"Evo 2 **{ps['auroc_evo2']:.4f}** vs GERP **{ps['auroc_gerp']:.4f}**. Nucleotide "
             f"Transformer scores **{nt['nt_scored_total']:,}/{nt['gerp_blind_total']:,}** of the "
             f"GERP-blind variants — full reach is a DNA-LM class property, not an Evo 2 one.\n")
    L.append("| species | n | Evo 2 (covered) | GERP (covered) | GERP (must answer) | Evo 2 (full panel) |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for r in T4["conservation"]:
        L.append(f"| {r['species']} | {r['n']:,} | {_f(r['evo2_covered'], 4)} | "
                 f"{_f(r['gerp_covered'], 4)} | {_f(r['gerp_abstain'], 4)} | {_f(r['evo2_full'], 4)} |")
    L.append("\n**Supervised baselines (point estimates only)**\n")
    L.append("| species | k-mer GBM (within) | CNN (within) | CNN (species held out) | Evo 2 zero-shot |")
    L.append("|---|--:|--:|--:|--:|")
    for r in T4["supervised"]:
        L.append(f"| {r['species']} | {_f(r['kmer_within'])} | {_f(r['cnn_within'])} | "
                 f"{_f(r['cnn_loso'])} | {_f(r['evo2'])} |")
    h = P["human"]
    L.append(f"\n**Protein LM**: only **{P['n_evaluable']} of {P['n_annotated']}** annotated panels can "
             f"host one, and ESM-2 is scored at usable size on **{P['n_esm_usable']}**. On human "
             f"(n = {h['n']:,}): ESM-2 {h['auroc_esm']:.3f} vs Evo 2 {h['auroc_evo2']:.3f}, "
             f"Δ **{h['delta_esm_minus_evo2']:+.3f}** "
             f"[{h['delta_ci'][0]:+.3f}, {h['delta_ci'][1]:+.3f}].\n")
    L.append(f"*{T4['_note']}*\n")
    return "\n".join(L)


def main():
    F2, F4, F5, F6 = (_load("fig2_data.json"), _load("fig4_reconciliation.json"),
                      _load("fig5_reach.json"), _load("fig6_free.json"))
    T1 = table1_atlas(F2, F6)
    T2 = table2_trust(F4)
    T3 = table3_reach(F5)
    T4 = table4_baselines(F5)
    _assert_canonical(T1, T2, T3, T4)

    os.makedirs(ROOT / "reports", exist_ok=True)
    json.dump({"T1": T1, "T2": T2, "T3": T3, "T4": T4},
              open(ROOT / "reports/tables.json", "w", encoding="utf-8"), indent=1)
    open(ROOT / "reports/tables.md", "w", encoding="utf-8").write(_md(T1, T2, T3, T4))
    print(f"  T1 {len(T1['rows'])} species · T2 {len(T2['conformal'])} conformal rows · "
          f"T3 {len(T3['rows'])} species · T4 {len(T4['conservation'])} species, "
          f"{len(T4['supervised'])} supervised")
    print("wrote reports/tables.json and reports/tables.md")


if __name__ == "__main__":
    main()
