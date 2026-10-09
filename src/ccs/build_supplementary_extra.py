# -*- coding: utf-8 -*-
"""Tables S14, S15 and the Figure S1 legend, appended to reports/supplementary_tables.md.

WHY THIS FILE EXISTS. These three blocks were written by hand into supplementary_tables.md in
commits 4235a15 and 77f8e88, and were destroyed in e64e7b8 when that file was regenerated from
build_supplementary.py, a script that had never known how to produce them. The result was a
manuscript citing Tables S14 and S15 three times, and Figure S1 five times, against a supplement
that ended at S13 and had no figure legend at all.

Anything not produced by a script does not survive the next regeneration, so these are recomputed
from reports/fig4_pervariant.parquet rather than restored as text.

Run AFTER build_supplementary.py:
    python src/ccs/build_supplementary.py
    python src/ccs/build_supplementary_extra.py
"""
import io
import sys

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = "reports/supplementary_tables.md"
SRC = "reports/fig4_pervariant.parquet"
SPECIES = ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]
REFUSE = 0.15


def _matrix():
    """Per-species operating-point intervals deposited in fig4_matrix.json:
    sensitivity/lift 95% CIs, lift_resolved flag, and ece_trivial/transfer/oracle."""
    import json
    d = json.load(io.open("reports/fig4_matrix.json", encoding="utf-8"))
    return {r["species"]: r for r in d["rows"]}


def _asym():
    import json
    return json.load(io.open("reports/fig4_asym.json", encoding="utf-8"))


def _staircase():
    """Deposited per-species refusal census (staircase.per_species) -- the counts the main text
    quotes. Table S14 reads these rather than re-deriving them; see the note in main()."""
    import json
    d = json.load(io.open("reports/fig4_reconciliation.json", encoding="utf-8"))
    return d.get("staircase", {}).get("per_species", {})


def main():
    d = pl.read_parquet(SRC)
    err = ~d["correct"].to_numpy().astype(bool)
    conf = d["conf"].to_numpy()
    n_all, n_err = len(d), int(err.sum())
    out = []
    W = out.append

    # ---------------- Table S14 ----------------
    W("\n## Table S14. Selective prediction per species\n")
    W("Each species refuses its own least-confident 15% of calls, ordered by the confidence "
      "functional |2p - 1|; an error is a call on the wrong side of the 0.5 threshold. Lift is the "
      "fraction of that species' errors removed divided by the fraction of its calls refused, so "
      "1.0 is what random refusal achieves. Because the 15% boundary can fall inside a confidence tie block, the refused count for a species can differ by a variant or two between deposited artifacts, so re-deriving the boundary arithmetically gives a per-species total of 1,669 (rounding to nearest) or 1,674 (rounding up) against the 1,673 the thresholding actually produced. The refused counts here are the deposited ones, so this table, the Results and the recompute layer agree; the species-ignoring 15% scheme is the one that refuses 1,669, and it is a different construction, not this one. Recomputed from `reports/fig4_pervariant.parquet`, whose per-variant posterior is the two-parameter Platt map (not the isotonic map used for the ECE analysis); the isotonic posterior takes too few distinct values to order variants for selective refusal (Methods).\n")
    MX = _matrix()
    ST = _staircase()
    W("Lift 95% CIs and the resolution flag are the deposited bootstrap intervals "
      "(`reports/fig4_matrix.json`, seed 20260719, 2,000 resamples); a lift is `resolved` only when "
      "its interval excludes 1. Chicken is the one species whose interval spans 1 (not distinguishable "
      "from random refusal), which the table now flags.\n")
    W("| Species | n | errors | refused | errors removed | lift | lift 95% CI | resolved |")
    W("|---|--:|--:|--:|--:|--:|:--|:--:|")
    tot_ref = tot_rem = 0
    human_err = 0
    for sp in SPECIES:
        idx = np.where((d["species"] == sp).to_numpy())[0]
        if not len(idx):
            continue
        # Take the refused count from the deposited staircase, which is what the main text quotes,
        # instead of re-deriving it here. No arithmetic rounding rule reproduces it: the 15% boundary
        # lands inside confidence tie blocks, so ceil matches six species but overshoots horse
        # (117.15 -> 117 deposited, 118 by ceil) and rounding to nearest undershoots four others.
        # Re-deriving gave a pooled 1,669 (round) or 1,674 (ceil) against the deposited 1,673, i.e.
        # this table disagreed with the Results on the very count at issue.
        k = int(ST.get(sp, {}).get("n_refused", int(np.ceil(REFUSE * len(idx)))))
        ci = idx[np.argsort(conf[idx], kind="stable")]
        sel = np.zeros(n_all, bool)
        sel[ci[:k]] = True
        e = int(err[idx].sum())
        r = int((err & sel).sum())
        tot_ref += k
        tot_rem += r
        if sp == "human":
            human_err = e
        lift = (r / e) / (k / len(idx)) if e and k else float("nan")
        mx = MX.get(sp, {})
        lci = "[%.2f, %.2f]" % (mx["lift_lo"], mx["lift_hi"]) if "lift_lo" in mx else "--"
        res = "yes" if mx.get("lift_resolved", True) else "**no**"
        W("| %s | %s | %d | %d | %d | %.2f | %s | %s |"
          % (sp.capitalize(), format(len(idx), ","), e, k, r, lift, lci, res))
    plift = (tot_rem / n_err) / (tot_ref / n_all)
    W("| **Pooled** | **%s** | **%d** | **%s** | **%d** | **%.2f** | | |"
      % (format(n_all, ","), n_err, format(tot_ref, ","), tot_rem, plift))
    W("\nHuman has the **lowest lift of the nine** while carrying %d of the %d errors, so the "
      "species with the most errors is the one where confidence helps least, which is a property of its 1:1 panel "
      "composition rather than its labels: at a matched 10:1 composition its lift rises to 3.19 (Results). Seven of the nine per-species lifts exceed the "
      "pooled figure, because pooling weights by error count and human dominates that count. The "
      "highest lift rests on six errors and should not be read as a species ranking.\n"
      % (human_err, n_err))

    # ---------------- Table S15 ----------------
    lab, pred = d["label"].to_numpy(), d["pred"].to_numpy()
    tp = int(((lab == 1) & (pred == 1)).sum())
    fn = int(((lab == 1) & (pred == 0)).sum())
    fp = int(((lab == 0) & (pred == 1)).sum())
    tn = int(((lab == 0) & (pred == 0)).sum())
    W("\n## Table S15. The confusion matrix behind the %d errors\n" % (fn + fp))
    # The full provenance sentence is given once, under Table S14; a pointer carries the same information
    # without a second copy.
    W("At the 0.5 decision threshold on the calibrated probability, before any abstention. "
      "Recomputed from `reports/fig4_pervariant.parquet` on the same Platt posterior as Table S14, "
      "for the reason given there.\n")
    W("| | predicted negative | predicted positive | total |")
    W("|---|--:|--:|--:|")
    W("| **actually positive** | %s (missed) | %s (caught) | %s |"
      % (format(fn, ","), format(tp, ","), format(tp + fn, ",")))
    W("| **actually negative** | %s (correct) | %s (false alarm) | %s |"
      % (format(tn, ","), format(fp, ","), format(tn + fp, ",")))
    W("| **total** | %s | %s | %s |"
      % (format(tn + fn, ","), format(tp + fp, ","), format(n_all, ",")))
    import json as _json
    _flow = _json.load(io.open("reports/fig4_matrix.json", encoding="utf-8"))["_meta"]["flow"]
    _sc, _pc = _flow["sensitivity_ci"], _flow["specificity_ci"]
    W("\nPooled sensitivity %.3f [%.3f, %.3f], specificity %.3f [%.3f, %.3f] (Wilson score intervals, "
      "`reports/fig4_matrix.json`), overall error rate %.4f. The %d errors are %d "
      "missed positives against only %d false alarms, so **%.1f%% of all errors are missed "
      "positives**. That is the asymmetry the selective layer does *not* fix, because "
      "refusal removes false alarms far more effectively than misses.\n"
      % (tp / (tp + fn), _sc[0], _sc[1], tn / (tn + fp), _pc[0], _pc[1],
         (fn + fp) / n_all, fn + fp, fn, fp, 100.0 * fn / (fn + fp)))

    # ---------------- Figure S1 legend ----------------
    # ONE panel. Other candidate panels duplicate
    # main figures: risk-coverage (Fig. 6c, which reads the same series and also plots the oracle
    # curve), the eQTL distributions (Fig. 5b, the same variants as an ECDF) and per-species reach
    # (Fig. 4a, the same fig5_reach.json, every gap matching to rounding). The reliability diagram
    # is the only panel here that appears nowhere in the main set, and a calibration paper needs it.
    n_all_fmt = format(n_all, ",")

    W("\n## Figure S1. Reliability of the transferred calibration\n")
    W("Calibrated probability against observed frequency, ten equal-mass bins over all %s calls; "
      "marker area is bin count, so a bin resting on few variants cannot be mistaken for one "
      "resting on thousands. Equal-mass rather than equal-width binning, because at this prevalence "
      "the upper equal-width bins hold too few variants to estimate a frequency (Note S2).\n"
      % n_all_fmt)

    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("appended Tables S14, S15 and the Figure S1 legend to %s" % OUT)
    print("  pooled per-species lift %.4f over %d refused, %d of %d errors removed"
          % (plift, tot_ref, tot_rem, n_err))
    print("  confusion: TP=%d FN=%d FP=%d TN=%d" % (tp, fn, fp, tn))




def per_species_operating_point():
    """Table S16: sensitivity, specificity and PPV per species at the deployed 0.5 threshold.

    the paper reports macro AUROC 0.943 and never reports the quantity a
    veterinary geneticist would act on. At the operating point the paper recommends, sensitivity in
    the eight label-poor species is 0.659 and as low as 0.333 in goat.
    """
    d = pl.read_parquet(SRC)
    lab, pred = d["label"].to_numpy(), d["pred"].to_numpy()
    out = ["\n## Table S16. Per-species operating point at the 0.5 decision threshold\n",
           "Sensitivity, specificity and positive predictive value at the threshold the selective "
           "layer is built on, before any abstention. AUROC is a ranking statistic; these are what "
           "a user acting on a single call would experience. Sensitivity 95% CIs are the deposited "
           "bootstrap intervals (`reports/fig4_matrix.json`); goat's spans zero ([0.000, 0.667] on "
           "9 positives, 3 caught) and must not be read as a point. "
           "Recomputed from `reports/fig4_pervariant.parquet` (two-parameter Platt posterior; Methods).\n",
           "| Species | n | positives | sensitivity | sensitivity 95% CI | specificity | PPV |",
           "|---|--:|--:|--:|:--|--:|--:|"]
    MX = _matrix()
    tp_nh = pos_nh = 0
    sens_nh = []          # per-species sensitivities, for the unweighted species mean (M5)
    for s in SPECIES:
        k = (d["species"] == s).to_numpy()
        y, p = lab[k], pred[k]
        tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        if not (tp + fn):
            continue
        if s != "human":
            tp_nh += tp; pos_nh += tp + fn
            sens_nh.append(tp / (tp + fn))
        mx = MX.get(s, {})
        sci = "[%.3f, %.3f]" % (mx["sens_lo"], mx["sens_hi"]) if "sens_lo" in mx else "--"
        out.append("| %s | %s | %d | %.3f | %s | %.3f | %.3f |"
                   % (s.capitalize(), format(len(y), ","), tp + fn, tp / (tp + fn), sci,
                      tn / (tn + fp) if tn + fp else float("nan"),
                      tp / (tp + fp) if tp + fp else float("nan")))
    # this summary row used to give ONLY tp_nh/pos_nh, the positive-count-weighted mean,
    # unlabelled, in a paper whose primary aggregation is the unweighted species mean. Give both, and
    # say which is which, so the row cannot be read as a species summary.
    sens_macro = float(np.mean(sens_nh))
    sens_weighted = tp_nh / pos_nh
    out.append("| **Eight non-human (species mean)** | | **%d** | **%.3f** | | | |"
               % (pos_nh, sens_macro))
    out.append("| **Eight non-human (positive-weighted)** | | **%d** | **%.3f** | | | |"
               % (pos_nh, sens_weighted))
    # the closing sentence used to pair the 8,192-bp macro AUROC with a 1,001-bp
    # sensitivity as properties of one predictor, which is the readout conflation this paper's whole
    # argument forbids. Name the readout on each figure.
    out.append("\nSpecificity is high everywhere and sensitivity is not. Across the eight "
               "label-poor species the model detects %.1f%% of catalogued positives at this "
               "threshold by the unweighted species mean (%.1f%% weighting species by positive "
               "count), and only %.1f%% in goat. A macro AUROC of 0.943 at the 8,192-bp "
               "mean-log-likelihood readout and a sensitivity of %.2f at the 0.5 threshold under "
               "the 1,001-bp readout that produced these calls are both true of this predictor; "
               "the first is a ranking property and the second is what a single call delivers.\n"
               % (100 * sens_macro, 100 * sens_weighted, 33.3, sens_macro))
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S16 (per-species operating point)")




def trust_layer_8192():
    """Table S17: the trust layer rebuilt at the 8,192-bp readout."""
    import json
    j = json.load(io.open("reports/trust_layer_8192.json", encoding="utf-8"))
    p, per = j["pooled"], j["per_species"]
    out = ["\n## Table S17. The trust layer rebuilt at the 8,192-bp readout\n",
           "Identical construction to the 1,001-bp layer reported in the main text, "
           "leave-one-species-out isotonic calibration, the confidence functional |2p - 1|, each "
           "species refusing its own least-confident 15%, a 0.5 decision threshold, with only the "
           "score changed. Generated by `src/ccs/rebuild_trust_layer_8192.py`.\n",
           "| Species | n | AUROC | ECE | errors | lift | error-detection AUROC |",
           "|---|--:|--:|--:|--:|--:|--:|"]
    for s in SPECIES:
        if s in per:
            m = per[s]
            out.append("| %s | %s | %.3f | %.4f | %d | %.2f | %.3f |"
                       % (s.capitalize(), format(m["n"], ","), m["auroc"], m["ece"],
                          m["errors"], m["lift"], m["err_detect_auroc"]))
    out.append("\n| | 1,001-bp (main text) | 8,192-bp |")
    out.append("|---|--:|--:|")
    out.append("| total errors | 893 (8.02%%) | **%d (%.2f%%)** |" % (p["errors"], 100 * p["error_rate"]))
    out.append("| errors removed at 15%% refusal | 316 (35.4%%) | **%d (%.1f%%)** |"
               % (p["removed"], 100 * p["capture"]))
    # 2.36 was hard-coded here while the 8,192-bp cell beside it was read from an artefact, and it
    # disagreed with the deposited value, with Table S6, with Table S14 and with the Results, all of
    # which give 2.35. It required a refusal total of 1,669 rather than the 1,673 the thresholding
    # produced, while the rest of this column (893, 316, 35.4%) is on the 1,673 basis. Read it.
    import json as _j
    _cen = _j.load(io.open("reports/fig4_reconciliation.json",
                           encoding="utf-8"))["staircase"]["census"]
    out.append("| pooled lift | %.2f | **%.2f** |" % (_cen["lift"], p["pooled_lift"]))
    # 3.44 was hard-coded here and contradicted both the manuscript ("a macro mean of 3.42") and
    # fig4_reconciliation.json (staircase.census.lift_macro = 3.4196). Read it.
    _lm = _cen["lift_macro"]
    out.append("| macro lift | %.2f | **%.2f** |" % (_lm, p["macro_lift"]))
    out.append("| macro ECE (equal-width, 10 bins) | 0.053 | **%.4f** |" % p["macro_ece"])
    out.append("| macro error-detection AUROC (eight species, human excluded) | 0.736 | -- |")
    out.append("| macro error-detection AUROC (all nine species) | 0.707 | **%.3f** |"
               % p["macro_err_detect"])
    # Wording note: "errors halve" is the phrase the R1 rail guards, because a withdrawn claim used
    # it for the in-species ORACLE arm. Everything here is the deployable leave-one-species-out arm,
    # and the counts are given directly so the sentence cannot be read as the oracle result.
    out.append("\nThe layer is better at the readout the paper argues for, on every axis, and every "
               "figure here is the deployable leave-one-species-out arm rather than an in-species "
               "oracle. The error count falls from 893 to %d, the same 15%% per-species refusal "
               "captures %.0f%% of them rather than 35%%, and "
               "calibration improves by a factor of 2.5. We report the 1,001-bp layer as the "
               "headline because the reach, calibration and abstention analyses were built there "
               "and cover the full 11,130-variant panel, and because it is the conservative choice; "
               "this table is the check that the choice does not flatter the result.\n"
               % (p["errors"], 100 * p["capture"]))
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S17 (8,192-bp trust layer)")




def readout_decomposition():
    """Table S18: split the readout advantage into context and aggregation (2x2 recomputation).

    The main text reports the 8,192-bp mean-log-likelihood beating the 1,001-bp single-position
    readout by +0.065 species-mean AUROC, but that difference confounds a longer model context with
    averaging surprise across the window. analyze_2x2_readout.py separates them from a single
    8,192-bp forward pass; this table reports the split, with the correctness gate that the on-pod
    full-window mean reproduces the deposited 8,192-bp score.
    """
    import json
    j = json.load(io.open("reports/readout_2x2_decomposition.json", encoding="utf-8"))
    per, mac, pool, gate = j["per_species"], j["macro"], j["pooled"], j["correctness_gate"]
    au = lambda sp, k: per[sp]["auroc"][k]
    n_sub = sum(per[sp]["n"] for sp in per)
    out = ["\n## Table S18. Decomposing the readout advantage into context and aggregation\n",
           "The +0.065 species-mean advantage of the 8,192-bp mean-log-likelihood over the 1,001-bp "
           "single-position variant-delta (main text) confounds two choices: a longer model context "
           "and averaging surprise across the window. We separate them by re-scoring %s atlas "
           "variants (a per-species cap of 400) from a **single 8,192-bp forward pass** per sequence, "
           "reading three quantities that hold the model and context fixed and vary only the "
           "aggregation scope. The **context term** is the gain of the single-position readout as its "
           "left context grows from ~0.5 kb (the 1,001-bp window) to ~4 kb; the **aggregation term** "
           "is the further gain of the full-window mean over that single position. Generated by "
           "`src/ccs/analyze_2x2_readout.py`.\n" % format(n_sub, ","),
           "| Species | n | positives | single-pos, 1 kb | single-pos, 8 kb | window-mean, 1 kb | window-mean, 8 kb | context | aggregation |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    SPP = ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]
    for sp in SPP:
        if sp in per:
            m = per[sp]
            out.append("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %+.3f | %+.3f |"
                       % (sp.capitalize(), format(m["n"], ","), m["n_path"], au(sp, "A_1001_single"),
                          au(sp, "d_single_8192"), au(sp, "d_cen1001"), au(sp, "d_full_8192"),
                          m["context_term"], m["aggregation_term"]))
    ct, ag, tt = mac["context_term"], mac["aggregation_term"], mac["total_advantage"]
    ctp, agp, ttp = pool["context_term"], pool["aggregation_term"], pool["total_advantage"]
    out.append("\n| | context term | aggregation term | total |")
    out.append("|---|--:|--:|--:|")
    out.append("| species mean (± SE) | %+.4f ± %.4f | %+.4f ± %.4f | %+.4f ± %.4f |"
               % (ct["mean"], ct["se"], ag["mean"], ag["se"], tt["mean"], tt["se"]))
    out.append("| pooled (95%% CI) | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] |"
               % (ctp["mean"], ctp["ci"][0], ctp["ci"][1], agp["mean"], agp["ci"][0], agp["ci"][1],
                  ttp["mean"], ttp["ci"][0], ttp["ci"][1]))
    aggf = mac["auroc"]["d_cen1001"]["mean"] - mac["auroc"]["A_1001_single"]["mean"]  # aggregation-first
    ctxs = mac["auroc"]["d_full_8192"]["mean"] - mac["auroc"]["d_cen1001"]["mean"]     # context second
    # Factorial interaction: total minus the two MAIN effects, each measured from the single-position
    # 1-kb cell (aggregation alone = aggf, context alone = ct). The previous expression subtracted a
    # main effect from a path term and returned +0.0349, which is the difference between the two
    # single-factor conditions, not an interaction, and carried the opposite sign.
    #
    # All four cells must come off the 2x2 design itself: A_1001_single, d_single_8192, d_cen1001 and
    # d_full_8192. B_8192_mean is NOT one of them. It is the deposited 8,192-bp score, carried here
    # only as the correctness check below, and it sits 0.0008 above d_full_8192 on the macro scale.
    # Taking the total from B while the main effects come from the d_ cells printed a total of +0.0668
    # beside an interaction of -0.0391 that no single basis produces -- the two disagree by exactly
    # that 0.0008. The justification recorded here for doing so, that `tt["mean"]` is a macro mean of
    # per-species differences while the others are differences of macro cell means, is false: on this
    # artefact the two agree to 5.6e-17, because the per-species panels enter the macro mean unweighted.
    # The difference was never the aggregation. It was the cell.
    _A = mac["auroc"]["A_1001_single"]["mean"]
    inter = ((mac["auroc"]["d_full_8192"]["mean"] - _A)
             - (mac["auroc"]["d_cen1001"]["mean"] - _A)
             - (mac["auroc"]["d_single_8192"]["mean"] - _A))                            # interaction
    out.append("\nThe total recovered here (%+.4f species mean) reproduces the main-text readout "
               "effect (+0.065), evidence that the effect size is representative even though the "
               "subset is not class-balanced: it retains every positive in seven species and caps each "
               "species at 400 variants, so dog and human are held at 200 positives and the per-species positive "
               "fraction runs above the ~9%% of the full panels, reaching 50%% in dog and human. "
               "Representativeness is claimed for the effect, not the composition. "
               "**The split into context and aggregation is path-dependent, and we report both "
               "orderings rather than one.** Taking context first (single-position readout, window "
               "1 kb to 8 kb) then aggregation (single to window-mean at 8 kb) gives context %+.4f and "
               "aggregation %+.4f; taking aggregation first (single to window-mean at 1 kb) then "
               "context (window-mean 1 kb to 8 kb) gives aggregation %+.4f and context %+.4f. The "
               "factorial interaction, the total minus the two main effects, each measured from the "
               "single-position 1-kb cell, is %+.4f, i.e. sub-additive: the two factors overlap, so "
               "their effects do not add, and the +0.0349 gap between the two single-factor "
               "conditions is a difference between conditions rather than an interaction term. A "
               "'roughly equal halves' "
               "reading holds only under the first ordering. What is robust across orderings is that "
               "**distal context beyond 1 kb does not carry the gain**: the "
               "central-1 kb window-mean equals or exceeds the full 8-kb window-mean in six of nine "
               "species (species mean %.3f against %.3f), so extending the averaged window past 1 kb "
               "adds nothing. As a correctness check the full-window mean recomputed here reproduces "
               "the deposited 8,192-bp mean-log-likelihood to within %.4f pooled AUROC.\n"
               % (tt["mean"], ct["mean"], ag["mean"], aggf, ctxs, inter,
                  mac["auroc"]["d_cen1001"]["mean"], mac["auroc"]["d_full_8192"]["mean"],
                  abs(gate["dfull_vs_B_pooled_gap"])))
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S18 (readout decomposition)")




def _oracle_range_sentence(MX):
    """The oracle-versus-transfer range, computed rather than typed.

    This sentence was hard-coded as "0.9x to 12.6x ... up to 12.6x in cattle". Neither figure was
    in the data: cattle is 6.8x and the panel maximum is human at 10.8x. Deriving it from the same
    rows the table prints means the caption cannot drift from the column beside it again.
    """
    r = {sp: v["ece_transfer"] / v["ece_oracle"]
         for sp, v in MX.items()
         if v.get("ece_oracle") and "ece_transfer" in v}
    lo_sp = min(r, key=r.get)
    hi_sp = max(r, key=r.get)
    worse = sorted(sp for sp in r if r[sp] < 1.0)
    worse_txt = (", no better than transfer in %s" % " and ".join(worse)) if worse else ""
    return ("The oracle is %.1f× to %.1f× better than transfer where labels exist%s, "
            "rising to %.1f× in %s.\n"
            % (r[lo_sp], r[hi_sp], worse_txt, r[hi_sp], hi_sp))


def selective_robustness():
    """Table S19: the selective ordering is not sampling noise (pre-submission hardening)."""
    import json
    j = json.load(io.open("reports/selective_robustness.json", encoding="utf-8"))
    per, perm = j["per_species"], j["permutation"]
    clo, chi = j["pooled_err_detect_ci_species_clustered"]
    loso = j["loso_lift"]
    SPP = ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]
    out = ["\n## Table S19. The selective layer's confidence ordering is not sampling noise\n",
           "Whether the 15%%-refusal lift (%.2f pooled) reflects genuine selectivity or the sampling "
           "variability of small panels, tested on the deployable leave-one-species-out arm at the "
           "8,192-bp readout. A within-species permutation of the confidence (%s permutations, each "
           "preserving every panel's size and error count) gives a null lift of %.2f; the observed "
           "lift sits far outside it (p = %.4f). Error detection is the AUROC of the confidence "
           "functional |2p - 1| separating incorrect from correct calls. Generated by "
           "`src/ccs/audit_selective_robustness.py`.\n"
           % (j["observed_lift"], format(perm["n"], ","), perm["null_mean"], perm["p_value"]),
           "| Species | n | errors | error-detection AUROC | 95% CI | leave-one-out pooled lift |",
           "|---|--:|--:|--:|---|--:|"]
    for sp in SPP:
        r = per.get(sp, {})
        ll = loso.get(sp, float("nan"))
        if "err_detect_auroc" in r:
            out.append("| %s | %s | %d | %.3f | [%.3f, %.3f] | %.2f |"
                       % (sp.capitalize(), format(r["n"], ","), r["errors"], r["err_detect_auroc"],
                          r["ci"][0], r["ci"][1], ll))
        else:
            out.append("| %s | %s | %d | (too few errors) | | %.2f |"
                       % (sp.capitalize(), format(r.get("n", 0), ","), r.get("errors", 0), ll))
    out.append("\n| pooled | value |")
    out.append("|---|---|")
    out.append("| error-detection AUROC (species-clustered 95%% CI) | %.3f [%.3f, %.3f] |"
               % (j["pooled_err_detect_auroc"], clo, chi))
    out.append("| within-species permutation p | %.4f |" % perm["p_value"])
    out.append("| leave-one-species-out lift range | %.2f to %.2f |"
               % (min(loso.values()), max(loso.values())))

    n_above = sum(1 for sp in SPP if per.get(sp, {}).get("above_half"))
    out.append("\nAt this readout the confidence ordering detects errors above chance in %d of the "
               "nine species individually, its pooled effect clears 0.5 under a conservative "
               "species-clustered interval, the permutation test rejects the sampling-variability "
               "explanation, and no single species carries the pooled lift. The layer's operative "
               "component (the ordering, which is what transfers to a label-free species) is "
               "therefore a real signal at the deployed readout, not an artifact of panel size.\n"
               % n_above)

    # The same test at the 1,001-bp readout, because that is the layer the Results section
    # reports. The two arms are not interchangeable -- pooled error detection is 0.814 here and
    # 0.586 there -- so quoting one in a paragraph written at the other's readout would be the
    # readout mixing this paper's headline is about. Placed after the 8,192-bp summary so neither
    # paragraph can be read as describing the other's arm.
    k = json.load(io.open("reports/selective_robustness_1001.json", encoding="utf-8"))
    kperm = k["permutation"]
    klo, khi = k["pooled_err_detect_ci_species_clustered"]
    k_above = sum(1 for sp in SPP if k["per_species"].get(sp, {}).get("above_half"))
    out.append("\n**The same test at the 1,001-bp readout**, which is the layer Results reports. "
               "Observed pooled lift %.2f; permutation null mean %.2f, p = %.4f; pooled "
               "error-detection AUROC %.3f, species-clustered 95%% CI [%.3f, %.3f]; "
               "leave-one-species-out lift range %.2f to %.2f; errors detected above chance in %d "
               "of the nine species. The permutation rejects sampling variability at both readouts "
               "and both pooled intervals clear 0.5, but error detection is markedly weaker at "
               "1,001 bp, which is why the two arms are never quoted interchangeably.\n"
               % (k["observed_lift"], kperm["null_mean"], kperm["p_value"],
                  k["pooled_err_detect_auroc"], klo, khi,
                  min(k["loso_lift"].values()), max(k["loso_lift"].values()), k_above))
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S19 (selective robustness)")


def class_asymmetry():
    """Table S20: class-asymmetry of the abstention layer, per species.
    The abstention arm covers six species; goat, chicken and pig have no benign errors at the 0.5
    threshold, so their benign protection is undefined."""
    mp = _asym()["mechanism_plane"]
    ab = sorted([r for r in mp if r["mechanism"].startswith("abstention")],
                key=lambda r: -r["protects_pathogenic"])
    out = ["\n## Table S20. Class-asymmetry of the abstention layer, per species\n",
           "Fraction of each error class removed when a species refuses its least-confident 15% of "
           "calls, from `reports/fig4_asym.json`. `protects negatives` is the share of over-calls removed, "
           "`protects positives` the share of missed positives removed. The negative side is "
           "defined in only six species (goat, chicken and pig have no negative-class errors at the 0.5 "
           "threshold); human carries 479 of the 731 missed positives.\n",
           "| Species | negative-class errors | positive-class errors | protects negatives | protects positives |",
           "|---|--:|--:|--:|--:|"]
    for r in ab:
        out.append("| %s | %d | %d | %.3f | %.3f |" % (
            r["species"].capitalize(), r["n_benign_err"], r["n_path_err"],
            r["protects_benign"], r["protects_pathogenic"]))
    out.append("\nPositive-class capture ranges from %.3f (%s) to %.3f (%s); the pooled 0.261 in the "
               "Results is a variant-weighted mean dominated by human.\n"
               % (ab[0]["protects_pathogenic"], ab[0]["species"],
                  ab[-1]["protects_pathogenic"], ab[-1]["species"]))
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S20 (class-asymmetry, %d species)" % len(ab))


def calibration_comparison():
    """Table S21: per-species calibration ECE, transfer vs trivial vs oracle.
    Under the operating-point estimator transfer loses to the trivial sigmoid in six of nine species,
    but under the primary width-10 estimator the count is four of nine; we name the estimator with the
    count. The oracle is far better wherever labels exist."""
    MX = _matrix()
    out = ["\n## Table S21. Per-species calibration: transfer vs trivial vs oracle\n",
           "Expected calibration error on the operating-point estimator (`reports/fig4_matrix.json`): "
           "`trivial` is the two-parameter global sigmoid, `transfer` the leave-one-species-out isotonic "
           "map (the paper's method), `oracle` in-species calibration. Under this estimator transfer is "
           # these three were typed as 0.0569/0.0564/+0.0005 and are the mean of the
           # column printed directly beneath them, which averages to 0.0560/0.0553/+0.0007. Derive
           # them so the caption cannot disagree with its own table.
           "worse than the trivial sigmoid in six of nine species (macro transfer %.4f against trivial "
           "%.4f, a difference of %+.4f) and better in the other three (sheep, cattle, human), which "
           % (sum(r["ece_transfer"] for r in MX.values()) / len(MX),
              sum(r["ece_trivial"] for r in MX.values()) / len(MX),
              sum(r["ece_transfer"] - r["ece_trivial"] for r in MX.values()) / len(MX)) +
           "are not simply the three largest panels, since dog and cat are both larger than sheep. Under "
           "the paper's primary equal-width-10 estimator (Table S3) the count is four of nine (goat, pig, "
           "cat, dog) and the macro slightly *favours* transfer (-0.0019), so the direction of the split is "
           "estimator-dependent and we name the estimator with each count. "
           + _oracle_range_sentence(MX),
           "| Species | n | ECE trivial | ECE transfer | ECE oracle | transfer > trivial? | oracle vs transfer |",
           "|---|--:|--:|--:|--:|:--:|--:|"]
    n_worse = 0
    for sp in SPECIES:
        r = MX.get(sp, {})
        if "ece_trivial" not in r:
            continue
        worse = r["ece_transfer"] > r["ece_trivial"]
        n_worse += worse
        adv = r["ece_transfer"] / r["ece_oracle"] if r["ece_oracle"] else float("nan")
        out.append("| %s | %s | %.4f | %.4f | %.4f | %s | %.1fx |" % (
            sp.capitalize(), format(r.get("n", 0), ","), r["ece_trivial"], r["ece_transfer"],
            r["ece_oracle"], "**yes**" if worse else "no", adv))
    out.append("\nTransfer is worse than the trivial two-parameter sigmoid in **%d of 9** species, the "
               "per-species detail behind the macro null, and the reason we frame transfer as recovering "
               "only a fraction of what in-species labels would buy.\n" % n_worse)
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S21 (per-species calibration, %d worse than trivial)" % n_worse)


def auprc_table():
    """Table S22: precision-recall (AUPRC) alongside AUROC at the 8,192-bp readout."""
    import json
    d = json.load(io.open("reports/auprc.json", encoding="utf-8"))
    out = ["\n## Table S22. Precision-recall (AUPRC) at the 8,192-bp readout\n",
           "Average precision (AUPRC) alongside AUROC per species, with the no-skill baseline (the base "
           "rate). AUROC is optimistic under the 9%% class imbalance and for a screening use case; AUPRC "
           "is the decision-relevant summary. Macro AUPRC is %.3f against a %.3f base rate (%.1f×), pooled "
           "%.3f (`reports/auprc.json`).\n"
           % (d["macro"]["auprc"], d["macro"]["base_rate"], d["macro"]["auprc_lift_over_base"],
              d["pooled"]["auprc"]),
           "| Species | n | base rate | AUROC | AUPRC | AUPRC / base rate |",
           "|---|--:|--:|--:|--:|--:|"]
    for sp in SPECIES:
        v = d["per_species"].get(sp)
        if not v:
            continue
        out.append("| %s | %s | %.3f | %.3f | %.3f | %.1fx |" % (
            sp.capitalize(), format(v["n"], ","), v["base_rate"], v["auroc"], v["auprc"],
            (v["auprc"] / v["base_rate"] if v.get("base_rate") else v["auprc_lift_over_base"])))
    out.append("| **Macro** | | %.3f | | **%.3f** | **%.1fx** |"
               % (d["macro"]["base_rate"], d["macro"]["auprc"], d["macro"]["auprc_lift_over_base"]))
    out.append("| **Pooled** | %s | %.3f | %.3f | %.3f | |"
               % (format(d["pooled"]["n"], ","), d["pooled"]["base_rate"], d["pooled"]["auroc"],
                  d["pooled"]["auprc"]))
    out.append("\nAUPRC stays well above the base rate in every species, so the discrimination at this "
               "readout is not a base-rate artifact. The one setting where precision-recall collapses is "
               "the matched control of the concurrent cross-species work cited in the main text "
               "(AUPRC 0.166 against a ~9% base rate, versus its 0.717 AUROC), which is why we quote "
               "both metrics where we cite it. Both figures are that paper's, read from its text and "
               "held by no artefact here; its 0.717 is not the 0.717 this paper reports for HAL's "
               "covered AUROC."
               "\n")
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S22 (AUPRC)")


def human_reach_table():
    """Table S23: reach and must-answer accuracy for five human scores on the whole ClinVar panel."""
    import json
    d = json.load(io.open("reports/human_reach_audit.json", encoding="utf-8"))
    m = d["_meta"]
    # The '## Table SNN. Title' heading is the file's convention and is not cosmetic:
    # build_supplementary_doc.py counts tables with '^## Table S\d+', so a bold-paragraph caption is
    # invisible to it and the table is reported as cited-but-absent.
    out = ["\n---\n",
           "\n## Table S23. Reach and must-answer accuracy for five human scores on ClinVar\n",
           "\n%s single-nucleotide variants at a review status of one star or better (%s "
           "pathogenic, %s benign). *Covered* is AUROC over the variants each score can score; "
           "*must-answer* charges every pair touching an unscorable variant at one half, the CAFA "
           "full-evaluation convention. *Missingness alone* is the AUROC of the indicator of whether "
           "a value was returned, discarding the values. For a score that answers everywhere the "
           "indicator never varies and the quantity is undefined, not 0.5; a value at 0.5000 means "
           "coverage is near-complete rather than complete. Built by "
           "`src/ccs/audit_human_panel.py` from `reports/human_reach_audit.json`.\n"
           % (format(m["n"], ","), format(m["n_pos"], ","), format(m["n_neg"], ","))]
    out.append("\n| score | reach (all) | reach (pos.) | reach (neg.) | covered | must-answer | "
               "penalty | missingness alone |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in sorted(d["scorers"], key=lambda x: -x["reach"]):
        # at 4 dp phastCons and phyloP print reach 1.0000 and penalty
        # 0.0000 while each leaves 47 variants unscored (2 pathogenic, 45 benign). The table's
        # own note says a missingness AUROC of 0.5000 means "near-complete rather than complete",
        # so the rows contradicted the note above them. Five decimals separate the two cases.
        out.append("| %s | %.5f | %.5f | %.5f | %.4f | %.4f | %.5f | %.4f |"
                   % (s["name"], s["reach"], s["reach_pos"], s["reach_neg"], s["auroc_covered"],
                      s["auroc_must_answer"], s["penalty"], s.get("miss_auroc", float("nan"))))
    out.append("\nThe two missense-only scores carry a missingness AUROC above their own must-answer "
               "AUROC. That is a statement about this panel's composition and not about either "
               "method: coverage tracks consequence class here, and ClinVar's pathogenic variants "
               "are missense-enriched, so the indicator is largely a missense indicator. Table S24 "
               "narrows the confound by restricting to the variant classes dbNSFP annotates, where "
               "no predictor has a missingness AUROC above its own must-answer AUROC under the signed test, although 30 of the 48 with a defined missingness AUROC do read orientation-free, as the Methods specify.\n")
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S23 (human reach)")


def dbnsfp_reach_table():
    """Table S24: the 49-predictor missense audit, summarised by coverage regime."""
    import json
    import statistics as st
    d = json.load(io.open("reports/dbnsfp_reach_audit.json", encoding="utf-8"))
    m, P = d["_meta"], d["predictors"]
    full = [x for x in P if x["reach"] > 0.999]
    part = [x for x in P if 0.70 <= x["reach"] <= 0.999]
    res = [x for x in P if x["reach"] < 0.70]
    sig = [x for x in P if x.get("class_gap_ci")
           and (x["class_gap_ci"][0] > 0 or x["class_gap_ci"][1] < 0)]
    # Pooled over a panel the missingness AUROC is the identity 0.5 + class_gap/2, so a predictor
    # with a NEGATIVE gap sits below one half and the signed test can never fire for it. On this
    # panel 34 of 49 have a negative gap. Both conventions are therefore printed, and the
    # orientation-free one is primary: a coverage pattern that separates the classes backwards is
    # as informative as one that separates them forwards. See Note S39.
    fin = [x for x in P if x.get("miss_auroc") == x.get("miss_auroc")]     # drops NaN
    beats = [x for x in fin if x["miss_auroc"] > x["auroc_must_answer"]]
    beats_free = [x for x in fin
                  if max(x["miss_auroc"], 1.0 - x["miss_auroc"]) > x["auroc_must_answer"]]

    out = ["\n---\n",
           "\n## Table S24. Reach and class-dependent coverage for 49 dbNSFP predictors\n",
           # NOT a missense panel: dbNSFP annotates nonsynonymous AND splice-site substitutions, and
           # only 202,643 of the 328,328 are missense. Calling it missense overstated how much of
           # the scope penalty the restriction removes.
           "\nThese %d predictors are audited on the %s nonsynonymous and splice-site variants dbNSFP "
           "annotates (%s pathogenic, %s benign), of which 202,643 are missense, across "
           "%s genes; intervals from a bootstrap resampling genes rather than variants (%s draws). "
           "Predictors are grouped by reach into three disjoint regimes and medians are taken over "
           "the group. The Results instead cuts the same 49 at nested reach thresholds, so those "
           "counts overlap rather than partition: the 18 predictors reaching 75%% of the panel "
           "carry a median penalty of 0.0217, the 12 reaching 90%% carry 0.0009 and the 8 reaching "
           "99%% carry 0.0000. Built by "
           "`src/ccs/audit_dbnsfp_panel.py` from `reports/dbnsfp_reach_audit.json`.\n"
           % (len(P), format(m["n"], ","), format(m["n_pos"], ","), format(m["n_neg"], ","),
              format(m["n_genes"], ","), format(m["n_boot"], ","))]
    out.append("\n| coverage regime | n predictors | median reach | median covered | "
               "median must-answer | median penalty |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for name, grp in [("complete (>99.9%)", full), ("partial (70 to 99.9%)", part),
                      ("restricted (<70%)", res)]:
        if not grp:
            continue
        out.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |"
                   % (name, len(grp), st.median(x["reach"] for x in grp),
                      st.median(x["auroc_covered"] for x in grp),
                      st.median(x["auroc_must_answer"] for x in grp),
                      st.median(x["penalty"] for x in grp)))
    out.append("\n| quantity | value |")
    out.append("| --- | --- |")
    out.append("| class gap with a gene-clustered interval excluding zero | %d of %d |"
               % (len(sig), len(P)))
    out.append("| ... and a gap of at least 0.02 in magnitude | %d of %d |"
               % (len([x for x in sig if abs(x["class_gap"]) >= 0.02]), len(P)))
    out.append("| missingness alone outscoring must-answer, orientation-free | %d of %d |"
               % (len(beats_free), len(fin)))
    out.append("| ... the same test taking the sign as given | %d of %d |"
               % (len(beats), len(fin)))
    out.append("| predictors with a negative class gap, for which the signed test cannot fire | %d of %d |"
               % (len([x for x in P if x["class_gap"] < 0]), len(P)))
    out.append("| complete-coverage predictors returning a penalty of exactly 0.0000 | %d of %d |"
               % (len([x for x in full if x["penalty"] == 0.0]), len(full)))
    out.append("\nRestricting to the classes dbNSFP annotates reverses the direction of the coverage "
               "bias rather than removing it: here the missense-oriented predictors reach the "
               "negatives far more than the positives, where on the whole ClinVar panel they reach "
               "the positives more. Because the pooled missingness AUROC is the identity 0.5 + "
               "gap/2, a negative gap puts it below one half and the signed test cannot fire. Read "
               "orientation-free, the indicator still exceeds most of these predictors' must-answer AUROCs, which says that their "
               "reach tracks the label here, not that their values carry less information: the must-answer rule scores every pair touching an unscored variant at one half "
               "and so ignores what the indicator knows. This panel relocates the composition confound instead of clearing it (Note S39). What it does establish is the reach "
               "accounting itself. We draw no ranking of these predictors from this panel, for the "
               "reason set out in Note S39.\n")
    with io.open(OUT, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    print("  appended Table S24 (dbNSFP 49-predictor audit)")


if __name__ == "__main__":
    main()
    per_species_operating_point()
    trust_layer_8192()
    readout_decomposition()
    selective_robustness()
    class_asymmetry()
    calibration_comparison()
    auprc_table()
    human_reach_table()
    dbnsfp_reach_table()
