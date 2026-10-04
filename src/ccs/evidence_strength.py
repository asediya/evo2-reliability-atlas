"""C10(a): likelihood ratios for the human arm at thresholds taken from the benign distribution.

No ACMG/AMP evidence-strength assignment is made here, and no likelihood ratio in this sweep is
mapped onto an evidence tier. The sweep is in-sample: thresholds are set at quantiles of the benign
score distribution and the false-positive rate is then read off that same benign set, so no
held-out data enter. The benign set is also not consequence-matched to the positives, which are
coding-dominated, so a threshold taken from it does not carry over to a missense variant. The
output is an internal ranking diagnostic and is not intended for use in variant curation.

This is deliberately human-only. It needs a clinically adjudicated benign set, and human is the one
species in this paper that has one. That is the point rather than a limitation of the script: the
deliverable a clinician wants is exactly the deliverable the eight label-poor species cannot have,
which is why the rest of the paper reports a decision procedure instead.

Positives: the atlas human arm, ClinVar Pathogenic/Likely_pathogenic.
Negatives: ClinVar Benign/Likely_benign, the same set built in clinvar_benign_arm.py.
Readout:   1,001-bp single-position, identical to that arm.

The likelihood ratio is a ratio of within-class densities, so it does not depend on the class
ratio in this panel, which is 1,500 against 17,376 and is an artefact of what has been scored.
"""
import json
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ATLAS_SCORES = "data/processed/scores/human_evo2_40b_local_scores.parquet"
ATLAS_LABELS = "data/interim/atlas8192/human_windows_8192.parquet"
CV_SCORES = "data/processed/scores/clinvar_evo2_raw.parquet"
CV_LABELS = "data/interim/clinvar_evo2_windows.parquet"



def key(col):
    return (col.str.replace_all("^neg_", "").str.replace_all(":", "_").str.replace_all(r"^chr", ""))


pos = (pl.read_parquet(ATLAS_LABELS).select(["variant_id", "label"]).filter(pl.col("label") == 1)
       .join(pl.read_parquet(ATLAS_SCORES), on="variant_id", how="inner")
       .with_columns(key(pl.col("variant_id")).alias("k")))
neg = (pl.read_parquet(CV_LABELS).select(["variant_id", "label"]).filter(pl.col("label") == 0)
       .join(pl.read_parquet(CV_SCORES), on="variant_id", how="inner")
       .with_columns(key(pl.col("variant_id")).alias("k")))
neg = neg.filter(~pl.col("k").is_in(pos["k"].to_list()))

sp = pos["evo2_40b_neg"].to_numpy().astype(float)
sn = neg["evo2_40b_neg"].to_numpy().astype(float)
sp = sp[np.isfinite(sp)]
sn = sn[np.isfinite(sn)]
print("positives %d (ClinVar P/LP)  negatives %d (ClinVar B/LB)" % (len(sp), len(sn)))


def lr_at(thr):
    """LR+ for the rule 'score >= thr', with a Haldane correction so an empty benign cell is
    reported as a bound rather than as infinity."""
    tp = float((sp >= thr).sum())
    fp = float((sn >= thr).sum())
    sens = tp / len(sp)
    fpr = fp / len(sn)
    if fp == 0:
        fpr_b = 0.5 / len(sn)          # one-sided bound, not a point estimate
        return sens / fpr_b, sens, fpr, True
    return sens / fpr, sens, fpr, False


def boot_lr(thr, B=2000, seed=17):
    """Percentile CI for LR+ at a fixed threshold, resampling both classes independently.

    The resampling unit is the variant, not the gene: these intervals are not gene-clustered."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        p = sp[rng.integers(0, len(sp), len(sp))]
        n = sn[rng.integers(0, len(sn), len(sn))]
        fp = float((n >= thr).sum())
        fpr = (0.5 / len(n)) if fp == 0 else fp / len(n)
        out.append((p >= thr).mean() / fpr)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# Sweep thresholds over the observed score range at benign-set quantiles, which is where the
# discrimination that matters happens.
qs = [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
rows = []
print("\n%-8s %10s %8s %8s %9s  %s" % ("benign q", "threshold", "sens", "FPR", "LR+", "95% CI"))
for q in qs:
    thr = float(np.percentile(sn, q))
    lr, sens, fpr, bounded = lr_at(thr)
    lo, hi = boot_lr(thr)
    rows.append({"benign_quantile": q, "threshold": round(thr, 6), "sensitivity": round(sens, 4),
                 "fpr": round(fpr, 6), "lr_plus": round(lr, 2),
                 "lr_plus_ci95": [round(lo, 2), round(hi, 2)], "lr_is_lower_bound": bounded})
    print("%-8s %10.5f %8.4f %8.5f %9.2f  [%.1f, %.1f]%s"
          % (q, thr, sens, fpr, lr, lo, hi, "  (bound)" if bounded else ""))

best = max(rows, key=lambda r: r["lr_plus"])
print("\nhighest LR+ observed: %.2f at benign q%.1f (sensitivity %.3f)"
      % (best["lr_plus"], best["benign_quantile"], best["sensitivity"]))
print("no ACMG/AMP evidence tier is assigned: the sweep is in-sample and the benign set is not "
      "consequence-matched to the positives")

out = {"_meta": {"evidence_strength": ("No ACMG/AMP evidence-strength assignment is made and no "
                                       "tier is reported. The sweep is in-sample: thresholds are "
                                       "quantiles of the benign score distribution and the FPR is "
                                       "read off that same set. The benign set is 12,999 noncoding "
                                       "against 4,377 coding and is not consequence-matched to the "
                                       "coding-dominated positives."),
                 "positives": "atlas human arm, ClinVar P/LP",
                 "negatives": "ClinVar B/LB",
                 "readout": "1,001-bp single-position (evo2_40b_neg)",
                 "scope": ("Human only, and necessarily so: it requires a clinically adjudicated "
                           "benign set, which the eight label-poor species do not have. Not "
                           "transferable to them, and no evidence-strength claim is made for them."),
                 "note": ("LR+ is a ratio of within-class densities and does not depend on the "
                          "1,500:17,376 class ratio in this panel.")},
       "sweep": rows,
       "best": {"lr_plus": best["lr_plus"], "benign_quantile": best["benign_quantile"],
                "sensitivity": best["sensitivity"]}}
json.dump(out, open("reports/evidence_strength.json", "w"), indent=2)
print("\nwrote reports/evidence_strength.json")
