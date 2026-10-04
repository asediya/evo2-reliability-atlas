# -*- coding: utf-8 -*-
"""Regression: does glmtrust.audit still re-derive every published reach number exactly?

This mattered when the auditor was written and matters more now. Since then the module has gained
missingness AUROC, strata, materiality gating, a per-class stratum gate, and a switch from
bootstrap to closed-form intervals -- any of which could have perturbed quantities the manuscript
already reports.

Compares against reports/fig5_reach.json, the tracked recompute-layer artifact. NOT against
COMPILED_RESULTS.md, which is not canonical.

A VACUOUS PASS IS A FAILURE. The first version of this script assumed the wrong JSON shape, matched
no keys, and printed "CLEAN -- every published reach number still reproduces" having checked
exactly zero quantities. Every check below therefore has to be seen to fire: the script exits
non-zero unless it verified EXACTLY EXPECT_CHECKS quantities across all nine species -- an exact
count, not a floor, so that a quantity vanishing from the artifact fails the gate rather than
slipping under it.
"""
import json
import os
import math
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
# run from the repository root wherever it is checked out, not a path baked in at authoring time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "glmtrust/src")
sys.path.insert(0, "src/ccs")
from glmtrust.audit import Scorer, audit                              # noqa: E402
from fig5_stats import SP                                             # noqa: E402

# EXACT, not a floor. The true count is 108 -- nine species times twelve quantities -- and setting
# the gate to 100 left room for eight published values to vanish from the artifact without the check
# noticing, while the docstring above promised that a vacuous pass is a failure. A floor eight short
# of the ceiling is a floor that tolerates exactly the silent-shrinkage failure it was written to
# catch. If the artifact legitimately gains or loses a quantity, this number is edited deliberately.
EXPECT_CHECKS = 108
TOL = 5e-4

ref = json.load(open("reports/fig5_reach.json", encoding="utf-8"))["per_species"]
by_species = {r["species"]: r for r in ref}


def panel(sp):
    """Rebuild the panel exactly as fig5_stats.load does, so any difference is the auditor's."""
    cons, ev = SP[sp]
    g = pl.read_parquet(f"data/processed/conservation/{cons}_gerp.parquet").select(["variant_id", "gerp"])
    s = pl.read_parquet(f"data/processed/scores/{ev}_evo2_40b_local_scores.parquet") \
          .select(["variant_id", pl.col("evo2_40b_neg").alias("evo2")])
    d = g.join(s, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in vid], dtype=int)
    return y, d["gerp"].to_numpy().astype(float), d["evo2"].to_numpy().astype(float)


checked = mism = 0
rows = []
for sp in SP:
    if sp not in by_species:
        print("  %-9s NOT IN ARTIFACT" % sp)
        continue
    r = by_species[sp]
    y, gerp, evo2 = panel(sp)
    rep = audit(y, [Scorer("gerp", gerp, readout="r"), Scorer("evo2", evo2, readout="r")],
                n_boot=200, seed=0)
    a = [s for s in rep.scorers if s.name == "gerp"][0]
    e = [s for s in rep.scorers if s.name == "evo2"][0]
    pair = rep.pairs[0] if rep.pairs else None      # a=gerp, b=evo2, on the shared subset

    pairs = [("n", r.get("n"), a.n), ("n_pos", r.get("n_pos"), a.n_pos),
             ("n_neg", r.get("n_neg"), a.n_neg),
             ("reach_pos", r.get("reach_pos"), a.reach_pos),
             ("reach_neg", r.get("reach_neg"), a.reach_neg),
             ("reach_all", r.get("reach_all"), a.reach),
             ("class_gap", r.get("class_gap"), a.class_gap),
             ("auroc_gerp_covered", r.get("auroc_gerp_covered"), a.auroc_covered),
             ("auroc_gerp_full_abstain", r.get("auroc_gerp_full_abstain"), a.auroc_must_answer),
             ("must_call_penalty", r.get("must_call_penalty"), a.penalty),
             # In the artifact "evo2_covered" means Evo 2 restricted to the subset GERP can score,
             # and "evo2_full" means Evo 2 over the whole panel. Evo 2 reaches everything, so its
             # OWN covered set is the full panel -- mapping evo2_covered onto that reported seven
             # false regressions before the two keys were read side by side.
             ("auroc_evo2_full", r.get("auroc_evo2_full"), e.auroc_covered),
             ("auroc_evo2_on_gerp_subset", r.get("auroc_evo2_covered"),
              pair.auroc_b_matched if pair else None)]
    bad = []
    for name, pub, got in pairs:
        if pub is None:
            continue
        # A COMPARISON THAT COULD NOT BE MADE IS NOT A COMPARISON THAT PASSED. `abs(nan-nan) >= TOL`
        # is False, so counting `checked` first would let 108 NaN quantities report "108 quantities
        # checked, 0 mismatches, CLEAN". Reject non-finite before the count.
        if got is None or not (math.isfinite(float(pub)) and math.isfinite(float(got))):
            mism += 1
            bad.append("%s pub=%s got=%s -- NON-FINITE, cannot be verified" % (name, pub, got))
            continue
        checked += 1
        if abs(float(pub) - float(got)) >= TOL:
            mism += 1
            bad.append("%s pub=%.6f got=%.6f" % (name, float(pub), float(got)))
    rows.append((sp, len([p for p in pairs if p[1] is not None]), bad))

print("  %-9s %8s  %s" % ("species", "checked", "result"))
for sp, n, bad in rows:
    print("  %-9s %8d  %s" % (sp, n, "ok" if not bad else "MISMATCH: " + "; ".join(bad)))

print()
print("  quantities checked : %d" % checked)
print("  mismatches         : %d" % mism)
if checked != EXPECT_CHECKS:
    print("  RESULT: INCONCLUSIVE -- %d quantities compared, expected exactly %d. A pass here "
          "would mean nothing." % (checked, EXPECT_CHECKS))
    sys.exit(2)
if mism:
    print("  RESULT: REGRESSION")
    sys.exit(1)
print("  RESULT: CLEAN -- all %d published quantities still reproduce to %g" % (checked, TOL))
