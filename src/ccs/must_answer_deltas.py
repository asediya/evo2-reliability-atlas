# -*- coding: utf-8 -*-
"""Evo 2 minus GERP under must-answer accounting, with a confidence interval, per species.

WHAT IS MISSING FROM THE PUBLISHED ARTIFACT. reports/fig5_reach.json carries GERP's own must-answer
penalty with an interval (`must_call_penalty`, `penalty_ci`) and both methods' point accuracies
(`auroc_gerp_full_abstain`, `auroc_evo2_full`), but nothing on the DIFFERENCE between them. So the
comparison the reach argument actually rests on -- when both methods must answer for every variant
in the panel, is Evo 2 ahead, and by how much -- is stated as two numbers side by side rather than
as a tested contrast.

It could not easily have been otherwise before: the must-answer AUROC was a closed form in the
reach with no inference attached, and a bootstrap over nine panels was affordable but was never the
quantity anyone had asked for. glmtrust.delong.must_answer_delta_ci now supplies it exactly, using
the placement-value representation (an unreachable variant wins half of every pair it touches), so
this is a closed-form contrast rather than a resampled one.

Reported alongside the covered-subset contrast, because the gap between the two IS the argument.

    python src/ccs/must_answer_deltas.py --out reports/must_answer_deltas.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "glmtrust/src")
sys.path.insert(0, "src/ccs")
from glmtrust.audit import Scorer, audit                              # noqa: E402
from fig5_stats import SP                                             # noqa: E402

READOUT = "1001bp single-position"


def panel(sp):
    """Rebuild the panel exactly as fig5_stats.load does."""
    cons, ev = SP[sp]
    g = pl.read_parquet(f"data/processed/conservation/{cons}_gerp.parquet").select(["variant_id", "gerp"])
    s = pl.read_parquet(f"data/processed/scores/{ev}_evo2_40b_local_scores.parquet") \
          .select(["variant_id", pl.col("evo2_40b_neg").alias("evo2")])
    d = g.join(s, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in vid], dtype=int)
    return y, d["gerp"].to_numpy().astype(float), d["evo2"].to_numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/must_answer_deltas.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    rows = []
    print("  Evo 2 minus GERP, per species. Positive favours Evo 2.")
    print()
    print("  %-9s %7s %8s %22s %24s" %
          ("species", "n", "reach", "covered-subset delta", "must-answer delta"))
    for sp in SP:
        y, gerp, evo2 = panel(sp)
        rep = audit(y, [Scorer("evo2", evo2, readout=READOUT),
                        Scorer("gerp", gerp, readout=READOUT)],
                    n_boot=a.n_boot, seed=0)
        p = rep.pairs[0]
        g = [s for s in rep.scorers if s.name == "gerp"][0]
        sig_m = "*" if p.matched_delta_excludes_zero else " "
        sig_a = "*" if p.must_answer_delta_excludes_zero else " "
        print("  %-9s %7s %7.1f%% %+8.4f [%+.3f,%+.3f]%s %+8.4f [%+.3f,%+.3f]%s"
              % (sp, format(len(y), ","), 100 * g.reach,
                 p.delta_matched, p.delta_matched_ci[0], p.delta_matched_ci[1], sig_m,
                 p.delta_must_answer, p.delta_must_answer_ci[0], p.delta_must_answer_ci[1], sig_a))
        rows.append({
            "species": sp, "n": int(len(y)), "gerp_reach": float(g.reach),
            "delta_matched": float(p.delta_matched),
            "delta_matched_ci": [float(v) for v in p.delta_matched_ci],
            "delta_matched_excludes_zero": bool(p.matched_delta_excludes_zero),
            "delta_must_answer": float(p.delta_must_answer),
            "delta_must_answer_ci": [float(v) for v in p.delta_must_answer_ci],
            "delta_must_answer_excludes_zero": bool(p.must_answer_delta_excludes_zero),
        })

    n_m = sum(r["delta_matched_excludes_zero"] and r["delta_matched"] > 0 for r in rows)
    n_a = sum(r["delta_must_answer_excludes_zero"] and r["delta_must_answer"] > 0 for r in rows)
    print()
    print("  species where Evo 2 is ahead with an interval excluding zero:")
    print("    on the variants both can score : %d of %d" % (n_m, len(rows)))
    print("    when both must answer          : %d of %d" % (n_a, len(rows)))
    print()
    print("  * = interval excludes zero")

    payload = {"_meta": {"readout": READOUT, "n_boot": a.n_boot,
                         "must_answer_interval": "closed form (DeLong on must-answer placements)",
                         "matched_interval": "paired bootstrap (panels are below delong_above)",
                         "sign": "positive favours Evo 2"},
               "per_species": rows,
               "summary": {"evo2_ahead_matched": n_m, "evo2_ahead_must_answer": n_a,
                           "n_species": len(rows)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
