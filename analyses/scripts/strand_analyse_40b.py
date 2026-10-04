# -*- coding: utf-8 -*-
"""Extend the strand-consistency ladder to the deployed 40B model at 8,192 bp.

At 1B and 7B, agreement between a window and its reverse complement gets *worse* as capacity grows:
Pearson 0.577 -> 0.442, and the AUROC an arbitrary strand choice moves grows 0.018 -> 0.028, which
is 7.6x then 11.8x the 0.0024 margin the paper's headline comparison rests on. The obvious reply is
that the trend reverses at the deployed scale. This measures whether it does.

The same run also carries the reference-likelihood confound at 8,192 bp: MLL(ref) scored on its own,
which the deposited scorer computes and discards.

    python analyses/scripts/strand_analyse_40b.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SRC = "analyses/data/strand/from40b/strand40b_scores.parquet"
OUT = "analyses/results/strand_40b.json"
# A shift measured at one readout must be divided by the margin at the SAME readout. This script
# declares window_bp = 8192 (below), so it divides by MARGIN_8192. Dividing by 0.0024, the 1,001-bp
# single-position margin measured on 9,532 co-scorable variants, would compare an 8,192-bp effect
# on 327 variants against a 1,001-bp margin on 9,532 and read several-fold too large.
# shift_over_margin is not a published value (it appears in no traced row and
# in neither the manuscript nor the supplement).
MARGIN_1001 = 0.0024   # Evo 2 minus GERP, 1,001-bp single-position readout, 9,532 co-scorable variants
MARGIN_8192 = 0.0960   # Evo 2 minus GERP, 8,192-bp readout, pooled random effects (meta_8192.json)
MARGIN = MARGIN_8192   # this panel is 8,192 bp, so this is the commensurable one
# Typed in from the earlier strand run (strand_analyse.py), which used a different and larger
# panel than the 327 variants scored here. They are kept so this script can report whether the
# 40B rung continues the 1B -> 7B trend, and they are NOT the published ladder. The published
# ladder is strand_ladder.json, where 1B and 40B are measured on one identical 327-variant
# panel and 1B reads pearson 0.7732, shift 0.0081. The two are not one series.
LADDER = {"1B": {"pearson": 0.577, "shift": 0.018},
          "7B": {"pearson": 0.442, "shift": 0.028}}
RNG = np.random.default_rng(20260805)


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


def main():
    if not os.path.exists(SRC):
        print("  %s not present yet - run the 40B strand job first" % SRC)
        return 1
    d = pl.read_parquet(SRC)
    print("  scored variants: %d" % len(d))

    res = {"n": int(len(d)), "window_bp": 8192, "margin": MARGIN,
           "margin_readout": "8192bp", "margin_1001_for_reference": MARGIN_1001,
           "readouts": {}}
    # The two window readouts only. The single-token readout ("at") is not analysed: its reverse-complement
    # columns index one position off the variant (P(ref) + P(alt) exceeds 1 in 208 of the 327 variants), so
    # the deposited table, Additional file 3's tables/strand40b_scores.parquet, carries no rcref_at or rcalt_at.
    for readout in ("full", "cen"):
        fwd = d["alt_" + readout].to_numpy() - d["ref_" + readout].to_numpy()
        rev = d["rcalt_" + readout].to_numpy() - d["rcref_" + readout].to_numpy()
        ok = np.isfinite(fwd) & np.isfinite(rev)
        if ok.sum() < 10:
            continue
        r = float(np.corrcoef(fwd[ok], rev[ok])[0, 1])
        block = {"pearson": r,
                 "mean_abs_disagreement": float(np.abs(fwd[ok] - rev[ok]).mean())}

        if "label" in d.columns and d["label"].null_count() < len(d):
            y = d["label"].fill_null(-1).to_numpy().astype(int)
            m = ok & (y >= 0)
            a_f = auroc(y[m], -fwd[m])
            a_r = auroc(y[m], -rev[m])
            a_avg = auroc(y[m], -(fwd[m] + rev[m]) / 2.0)
            shift = abs(a_f - a_r)
            block.update({"auroc_forward": a_f, "auroc_revcomp": a_r,
                          "auroc_strand_averaged": a_avg,
                          "strand_choice_shift": shift,
                          "shift_over_margin": float(shift / MARGIN)})
        res["readouts"][readout] = block
        print("\n  readout %s" % readout)
        for k, v in block.items():
            print("    %-24s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))

    # the ladder: does the deployed model reverse the 1B -> 7B trend, or continue it?
    dep = res["readouts"].get("full", {})
    res["ladder"] = dict(LADDER, **{"40B": {"pearson": dep.get("pearson"),
                                            "shift": dep.get("strand_choice_shift")}})
    if dep.get("pearson") is not None:
        worsens = dep["pearson"] < LADDER["7B"]["pearson"]
        res["trend_continues"] = bool(worsens)
        print("\n  ladder (Pearson forward vs reverse complement)")
        for k in ("1B", "7B", "40B"):
            v = res["ladder"][k]["pearson"]
            print("    %-4s %s" % (k, "%.4f" % v if v is not None else "n/a"))
        print("    trend %s at the deployed scale"
              % ("continues to worsen" if worsens else "reverses"))

    # reference-likelihood confound at 8,192 bp
    if "label" in d.columns and d["label"].null_count() < len(d):
        y = d["label"].fill_null(-1).to_numpy().astype(int)
        m = y >= 0
        a_ref = auroc(y[m], d["ref_full"].to_numpy()[m])
        res["ref_likelihood_alone"] = a_ref
        print("\n  reference-likelihood alone at 8,192 bp: AUROC %.4f" % a_ref)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n  wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
