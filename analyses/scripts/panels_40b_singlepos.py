# -*- coding: utf-8 -*-
"""Evo 2-40B on the splicing and ClinVar panels, at the single-position readout.

WHAT THIS IS NOT: a capacity rung for Table 3. The deposited 1B and 7B scores for these two panels
come from `analyses/scripts/score_mfass_evo2.py`, which sums the log-likelihood over the whole
1,001-bp window. These 40B scores come from `src/ccs/score_evo2_40b_local.py`, which is a
single-position next-token readout. Reading 40B from here against 1B from there moves the readout
and the capacity at the same time, and this paper has already had to retract one ladder built that
way (Note S48, the strand ladder, where panel and window length moved with capacity).

A 40B window-sum rung is not reachable with this pipeline: the window-sum path calls BioNeMo's
`predict_evo2`, which loads the whole model onto the GPU, and its runner accepts 1b and 7b only.
In this pipeline the 40B single-position readout runs through the block-streaming scorer; 40B's only
window readout is the 8,192-bp mean log-likelihood of src/ccs/score_evo2_meanll.py, not a window sum.
Closing that gap needs a window-sum path that accepts 40B.

So this artefact records what was measured and refuses to compute a delta against the other readout.

    python analyses/scripts/panels_40b_singlepos.py

Writes analyses/results/panels_40b_singlepos.json.
"""
import io
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

OUT = "analyses/results/panels_40b_singlepos.json"
PANELS = [
    ("splicing", "analyses/data/mfass/mfass_w1001.parquet",
     "analyses/data/mfass/mfass_evo2_40b_scores.parquet", "variant_class"),
    ("clinvar", "analyses/data/clinvar/clinvar_w1001.parquet",
     "analyses/data/clinvar/clinvar_evo2_40b_scores.parquet", "consequence"),
]
B = 2000
SEED = 20260807


def boot_ci(y, s, rng):
    idx = np.arange(len(y))
    out = []
    for _ in range(B):
        take = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        out.append(roc_auc_score(y[take], s[take]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for name, wp, sp, grp in PANELS:
        if not (os.path.exists(wp) and os.path.exists(sp)):
            print("  skip %s (missing input)" % name)
            continue
        d = pl.read_parquet(wp).join(pl.read_parquet(sp), on="variant_id")
        d = d.drop_nulls(["label", "evo2_40b_neg"])
        y = d["label"].to_numpy().astype(int)
        s = d["evo2_40b_neg"].to_numpy().astype(float)
        block = {"n": int(d.height), "n_pos": int(y.sum()),
                 "auroc": float(roc_auc_score(y, s)),
                 "auroc_ci95": boot_ci(y, s, rng),
                 "n_exactly_zero": int((s == 0).sum()),
                 "by_class": {}}
        for g in sorted(d[grp].unique().to_list()):
            m = d.filter(pl.col(grp) == g)
            yy = m["label"].to_numpy().astype(int)
            ss = m["evo2_40b_neg"].to_numpy().astype(float)
            cell = {"n": int(m.height), "n_pos": int(yy.sum())}
            if len(np.unique(yy)) < 2:
                cell["auroc"] = None
                cell["_why"] = "single class in this stratum"
            else:
                cell["auroc"] = float(roc_auc_score(yy, ss))
                cell["auroc_ci95"] = boot_ci(yy, ss, rng)
                # An AUROC resting on a handful of one class is not a reading. Essential Splice is
                # 238 disruptive in 246: eight negatives decide it.
                small = min(int(yy.sum()), int((1 - yy).sum()))
                if small < 30:
                    cell["_caution"] = ("only %d in the smaller class; the interval is the honest "
                                        "summary and the point estimate should not be quoted alone"
                                        % small)
            block["by_class"][g] = cell
        res[name] = block

    out = {
        "_generated_by": "analyses/scripts/panels_40b_singlepos.py",
        "_readout": "1,001-bp SINGLE-POSITION next-token delta, negated (src/ccs/score_evo2_40b_local.py)",
        "_not_comparable_to": (
            "the 1B and 7B scores deposited for these same panels, which are a WINDOW-SUM "
            "log-likelihood readout (analyses/scripts/score_mfass_evo2.py). Capacity and readout "
            "would move together, which is the confound Note S48 retracted for the strand ladder."),
        "_why_no_40b_window_sum": (
            "BioNeMo's predict_evo2 loads the whole model to the GPU and its runner accepts 1b and "
            "7b only; in this pipeline the 40B single-position readout runs through block "
            "streaming, and 40B has no window-sum readout."),
        "bootstrap": {"B": B, "seed": SEED, "unit": "variant"},
        "panels": res,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    for name, b in res.items():
        lo, hi = (b["auroc_ci95"] or [float("nan")] * 2)
        print("  %-9s n=%5d pos=%5d  AUROC %.4f [%.4f, %.4f]"
              % (name, b["n"], b["n_pos"], b["auroc"], lo, hi))
        for g, c in b["by_class"].items():
            note = "  <- %s" % c["_caution"][:46] if "_caution" in c else ""
            print("      %-32s n=%5d  %s%s"
                  % (g, c["n"], "     n/a" if c["auroc"] is None else "%.4f" % c["auroc"], note))
    print("\n  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
