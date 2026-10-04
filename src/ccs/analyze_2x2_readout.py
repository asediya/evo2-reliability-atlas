# -*- coding: utf-8 -*-
"""Decompose the 8192-mean-LL vs 1001-single-position readout advantage.

The manuscript reports that scoring Evo 2 as an 8,192-bp mean log-likelihood beats the 1,001-bp
single-position variant-delta readout by a macro AUROC of roughly +0.065 (pooled +0.092). Two
mechanisms are confounded in that comparison:

  CONTEXT     the mean-LL readout sees 8,192 bp; the single-position readout saw only 1,001 bp;
  AGGREGATION the mean-LL readout averages the per-position surprise over thousands of positions,
              the single-position readout reads one position (the variant token).

This job re-scores the whole atlas from a SINGLE 8,192-bp forward pass per sequence, emitting three
readouts that hold the model and context fixed and vary only the aggregation scope:

  d_single    log-likelihood delta at the variant position only        (8,192-bp context, 1 position)
  d_cen1001   mean delta over the central 1,001 bp                      (8,192-bp context, 1,001 pos)
  d_full      mean delta over the full 8,192 bp                         (8,192-bp context, all pos)

With A = the original 1,001-bp single-position readout (evo2_40b_neg) and B = the validated 8,192-bp
mean-LL (evo2_meanll_delta), the headline gap decomposes additively (in the 8,192-context frame):

  AUROC(d_full) - AUROC(A)  =  [AUROC(d_single) - AUROC(A)]  +  [AUROC(d_full) - AUROC(d_single)]
       total advantage             context term (1001->8192)        aggregation term (single->mean)

CORRECTNESS GATE: d_full is an independent, on-pod recomputation of the 8,192-bp mean-LL, so it must
reproduce B. If pooled |AUROC(d_full) - AUROC(B)| exceeds 0.015 the on-pod readout is not trustworthy
and NOTHING here should be folded into the manuscript.

    python src/ccs/analyze_2x2_readout.py [readout2x2_dir]
    -> reports/readout_2x2_decomposition.json
"""
import io
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig5_stats import SP  # noqa: E402  species -> (gerp_stem, local_stem)

DIR = sys.argv[1] if len(sys.argv) > 1 else "data/interim/readout2x2"
A_FILE = "data/processed/scores/%s_evo2_40b_local_scores.parquet"   # 1001-bp single-position (col evo2_40b_neg)
B_FILE = "data/processed/scores_cloud/atlas8192_%s_meanll_8192.parquet"  # 8192-bp mean-LL (col evo2_meanll_delta)
OUT = "reports/readout_2x2_decomposition.json"
SEED = 20260723
NBOOT = 4000
GATE = 0.015


def lab(vids):
    return np.array([0 if str(v).startswith("neg_") else 1 for v in vids], int)


def oriented(y, s):
    """AUROC oriented so higher score = pathogenic; also return the sign used."""
    a = roc_auc_score(y, s)
    return (a, +1) if a >= 0.5 else (1 - a, -1)


def main():
    rng = np.random.default_rng(SEED)
    # readouts, in aggregation order: original-1001-single, 8192-single, 8192-cen1001, 8192-full, validated-B
    READOUTS = ["A_1001_single", "d_single_8192", "d_cen1001", "d_full_8192", "B_8192_mean"]
    per = {}
    pool = {k: [] for k in READOUTS}
    pool_y = []
    for sp in SP:
        f2 = os.path.join(DIR, "%s_2x2.parquet" % sp)
        if not os.path.exists(f2):
            print("  %-8s (2x2 not pulled yet, skipping)" % sp)
            continue
        d = pl.read_parquet(f2).select(["variant_id", "d_full", "d_cen1001", "d_single"])
        a = pl.read_parquet(A_FILE % sp).select(["variant_id", pl.col("evo2_40b_neg").alias("A_1001_single")])
        b = pl.read_parquet(B_FILE % sp).select(["variant_id", pl.col("evo2_meanll_delta").alias("B_8192_mean")])
        j = d.join(a, on="variant_id", how="inner").join(b, on="variant_id", how="inner")
        y = lab(j["variant_id"].to_list())
        if len(set(y.tolist())) < 2:
            print("  %-8s (single-class after join, skipping)" % sp)
            continue
        cols = {"A_1001_single": j["A_1001_single"].to_numpy().astype(float),
                "d_single_8192": j["d_single"].to_numpy().astype(float),
                "d_cen1001": j["d_cen1001"].to_numpy().astype(float),
                "d_full_8192": j["d_full"].to_numpy().astype(float),
                "B_8192_mean": j["B_8192_mean"].to_numpy().astype(float)}
        aur, sign = {}, {}
        for k in READOUTS:
            s = cols[k]
            m = np.isfinite(s)
            aur[k], sign[k] = oriented(y[m], s[m])
        per[sp] = {"n": int(len(y)), "n_path": int(y.sum()),
                   "auroc": {k: float(aur[k]) for k in READOUTS},
                   "context_term": float(aur["d_single_8192"] - aur["A_1001_single"]),
                   "aggregation_term": float(aur["d_full_8192"] - aur["d_single_8192"]),
                   "total_advantage": float(aur["d_full_8192"] - aur["A_1001_single"]),
                   "dfull_vs_B_gap": float(aur["d_full_8192"] - aur["B_8192_mean"])}
        # accumulate for pooled, storing each score with its per-species orientation applied
        for k in READOUTS:
            pool[k].append(sign[k] * cols[k])
        pool_y.append(y)
        print("  %-8s n=%4d | A %.3f  d_single %.3f  d_cen1001 %.3f  d_full %.3f  (B %.3f)  "
              "| ctx %+.3f  agg %+.3f  tot %+.3f" %
              (sp, len(y), aur["A_1001_single"], aur["d_single_8192"], aur["d_cen1001"],
               aur["d_full_8192"], aur["B_8192_mean"],
               per[sp]["context_term"], per[sp]["aggregation_term"], per[sp]["total_advantage"]))

    if not per:
        print("\nNo species pulled yet — run after the job completes and files are pulled.")
        return

    # ---- macro (mean across species) with across-species SE
    def macro(key, sub):
        v = np.array([per[sp][key][sub] if sub else per[sp][key] for sp in per])
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    macro_aur = {k: macro("auroc", k) for k in READOUTS}
    macro_ctx = macro("context_term", None)
    macro_agg = macro("aggregation_term", None)
    macro_tot = macro("total_advantage", None)

    # ---- pooled AUROC + bootstrap CIs on the two decomposition terms
    Y = np.concatenate(pool_y)
    P = {k: np.concatenate(pool[k]) for k in READOUTS}
    def pooled_auc(k, idx=None):
        s = P[k] if idx is None else P[k][idx]
        yy = Y if idx is None else Y[idx]
        m = np.isfinite(s)
        return roc_auc_score(yy[m], s[m])
    pooled_aur = {k: float(pooled_auc(k)) for k in READOUTS}
    ctx_b, agg_b, tot_b = [], [], []
    n = len(Y)
    for _ in range(NBOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(Y[i])) < 2:
            continue
        a_A = pooled_auc("A_1001_single", i); a_s = pooled_auc("d_single_8192", i); a_f = pooled_auc("d_full_8192", i)
        ctx_b.append(a_s - a_A); agg_b.append(a_f - a_s); tot_b.append(a_f - a_A)
    def ci(b):
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    dfull_B_pooled = pooled_aur["d_full_8192"] - pooled_aur["B_8192_mean"]
    gate_ok = abs(dfull_B_pooled) <= GATE

    print("\nMACRO (mean across %d species):" % len(per))
    for k in READOUTS:
        print("  %-16s %.4f +/- %.4f" % (k, macro_aur[k][0], macro_aur[k][1]))
    print("  context term    %+.4f +/- %.4f" % macro_ctx)
    print("  aggregation term%+.4f +/- %.4f" % macro_agg)
    print("  total advantage %+.4f +/- %.4f" % macro_tot)
    print("\nPOOLED (all variants):")
    for k in READOUTS:
        print("  %-16s %.4f" % (k, pooled_aur[k]))
    print("  context term    %+.4f  95%% CI [%+.4f, %+.4f]" % (np.mean(ctx_b), *ci(ctx_b)))
    print("  aggregation term%+.4f  95%% CI [%+.4f, %+.4f]" % (np.mean(agg_b), *ci(agg_b)))
    print("  total advantage %+.4f  95%% CI [%+.4f, %+.4f]" % (np.mean(tot_b), *ci(tot_b)))

    print("\nCORRECTNESS GATE (d_full must reproduce validated B):")
    print("  pooled AUROC(d_full) %.4f  vs  AUROC(B) %.4f   gap %+.4f   [tol +/-%.3f]"
          % (pooled_aur["d_full_8192"], pooled_aur["B_8192_mean"], dfull_B_pooled, GATE))
    print("  GATE %s" % ("PASSED — on-pod readout reproduces the validated 8192 mean-LL; "
                          "decomposition is trustworthy" if gate_ok else
                          "FAILED — on-pod readout does NOT match B; DO NOT fold into the manuscript"))

    # verdict on the mechanism
    if gate_ok:
        agg = np.mean(agg_b); ctx = np.mean(ctx_b)
        if agg > 0 and agg > 2 * abs(ctx):
            mech = "AGGREGATION-DOMINATED: averaging surprise over the window, not the extra context, drives the advantage."
        elif ctx > 0 and ctx > 2 * abs(agg):
            mech = "CONTEXT-DOMINATED: the longer context drives the advantage; aggregation adds little."
        else:
            mech = "MIXED: context and aggregation both contribute materially."
        print("\nMECHANISM: " + mech)
    else:
        mech = "gate failed"

    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps({
        "species": list(per.keys()),
        "per_species": per,
        "macro": {"auroc": {k: {"mean": macro_aur[k][0], "se": macro_aur[k][1]} for k in READOUTS},
                  "context_term": {"mean": macro_ctx[0], "se": macro_ctx[1]},
                  "aggregation_term": {"mean": macro_agg[0], "se": macro_agg[1]},
                  "total_advantage": {"mean": macro_tot[0], "se": macro_tot[1]}},
        "pooled": {"auroc": pooled_aur,
                   "context_term": {"mean": float(np.mean(ctx_b)), "ci": ci(ctx_b)},
                   "aggregation_term": {"mean": float(np.mean(agg_b)), "ci": ci(agg_b)},
                   "total_advantage": {"mean": float(np.mean(tot_b)), "ci": ci(tot_b)}},
        "correctness_gate": {"dfull_vs_B_pooled_gap": float(dfull_B_pooled),
                             "tolerance": GATE, "passed": bool(gate_ok)},
        "mechanism": mech, "n_bootstrap": NBOOT,
    }, indent=2) + "\n")
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
