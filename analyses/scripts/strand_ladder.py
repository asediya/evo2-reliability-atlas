# -*- coding: utf-8 -*-
"""The strand-consistency capacity ladder, with capacity as the only thing that varies.

The published 1B and 7B strand numbers were measured on 389 variants in 1,002-bp windows and the
40B run on 327 variants in 8,192-bp windows. Placed in a column those three points appear to show
the inconsistency reversing at the deployed scale, but panel and window length move with capacity,
so nothing is identified. This reads 1B and 7B scored on the identical 8,192-bp panel the 40B run
used, so the ladder means what it appears to mean.

One readout caveat, stated because it changes what is comparable. The window-sum scorer returns the
summed log-likelihood over the window and the 40B job returned the mean, so sums are divided by the
window length here. Pearson correlations and AUROCs are unaffected by that scaling in any case;
only the absolute disagreement would have been, and it is put on the same footing.

    python analyses/scripts/strand_ladder.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analyses/results/strand_ladder.json"
W = 8192
# Two margins, because a shift measured at one readout has to be compared with the margin at the
# same readout. An earlier version divided the 8,192-bp strand shift by MARGIN_1001 and so compared
# an effect measured at one readout on 327 variants against a margin measured at another readout on
# 9,532. It read as 3.3x when the like-for-like figure is 8% of the margin.
MARGIN_1001 = 0.0024   # Evo 2 minus GERP, 1,001-bp single-position readout, 9,532 co-scorable variants
MARGIN_8192 = 0.0960   # Evo 2 minus GERP, 8,192-bp readout, pooled random effects (meta_8192.json)
MARGIN = MARGIN_8192   # this panel is 8,192 bp, so this is the commensurable one


def auroc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def from_local(p):
    """One row per (variant, orientation) from the window-sum scorer; pivot to one row per variant."""
    d = pd.read_parquet(p)
    d["vid"] = d["variant_id"].str.rsplit("|", n=1).str[0]
    w = d.pivot(index="vid", columns="orientation", values="evo2_delta")
    lab = d.drop_duplicates("vid").set_index("vid")["label"]
    return (w["fwd"].to_numpy() / W, w["rc"].to_numpy() / W,
            lab.loc[w.index].to_numpy().astype(int))


def from_40b(p):
    d = pd.read_parquet(p)
    return (d["alt_full"].to_numpy() - d["ref_full"].to_numpy(),
            d["rcalt_full"].to_numpy() - d["rcref_full"].to_numpy(),
            d["label"].to_numpy().astype(int))


def stats(fwd, rev, y):
    ok = np.isfinite(fwd) & np.isfinite(rev)
    fwd, rev, y = fwd[ok], rev[ok], y[ok]
    a_f, a_r = auroc(y, -fwd), auroc(y, -rev)
    avg = auroc(y, -(fwd + rev) / 2.0)
    return {"n": int(len(y)), "pearson": float(np.corrcoef(fwd, rev)[0, 1]),
            "spearman": float(pd.Series(fwd).corr(pd.Series(rev), method="spearman")),
            "mean_abs_disagreement": float(np.abs(fwd - rev).mean()),
            "as_fraction_of_sd": float(np.abs(fwd - rev).mean() / np.std(fwd)),
            "auroc_forward": a_f, "auroc_revcomp": a_r, "auroc_strand_averaged": avg,
            "strand_choice_shift": float(abs(a_f - a_r)),
            "shift_as_pct_of_8192_margin": float(100 * abs(a_f - a_r) / MARGIN_8192),
                      "shift_over_margin_1001_NOT_COMMENSURABLE": float(abs(a_f - a_r) / MARGIN_1001),
            "gain_from_averaging": float(avg - a_f)}


def main():
    rungs = {}
    for size in ("1b", "7b"):
        p = "analyses/data/strand/strand_ladder_evo2_%s_scores.parquet" % size
        if os.path.exists(p):
            rungs[size.upper()] = stats(*from_local(p))
    p40 = "analyses/data/strand/from40b/strand40b_scores.parquet"
    if os.path.exists(p40):
        rungs["40B"] = stats(*from_40b(p40))
    if not rungs:
        print("  no strand scores yet")
        return 1

    order = [k for k in ("1B", "7B", "40B") if k in rungs]
    print("  same panel, same 8,192-bp window, capacity varying\n")
    keys = ["n", "pearson", "spearman", "mean_abs_disagreement", "as_fraction_of_sd",
            "auroc_forward", "auroc_revcomp", "auroc_strand_averaged",
            "strand_choice_shift", "shift_as_pct_of_8192_margin", "gain_from_averaging"]
    print("  %-26s" % "" + "".join("%12s" % k for k in order))
    for k in keys:
        print("  %-26s" % k + "".join(
            ("%12d" % rungs[o][k]) if k == "n" else ("%12.4f" % rungs[o][k]) for o in order))

    res = {"_generated_by": "analyses/scripts/strand_ladder.py",
           "_panel": "the 327-variant 8,192-bp panel, identical across rungs",
           "_margin_8192": MARGIN_8192, "_margin_1001": MARGIN_1001, "rungs": rungs}
    if len(order) >= 2:
        p_series = [rungs[o]["pearson"] for o in order]
        res["monotone_worsening"] = bool(all(b < a for a, b in zip(p_series, p_series[1:])))
        res["direction"] = ("worsens with capacity" if res["monotone_worsening"]
                            else "not monotone in capacity")
        print("\n  strand agreement %s" % res["direction"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
