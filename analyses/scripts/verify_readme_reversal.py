# -*- coding: utf-8 -*-
"""Check the claim on glmtrust's own front page, on the panel it names.

glmtrust/README.md opens with: on a 1,434,335-variant ClinVar panel AlphaMissense leads REVEL by
+0.0056 AUROC as usually quoted, and REVEL leads by 0.0096 on the variants both can score, so the
verdict reverses.

That is the tool's headline. A reader will test it before trusting anything else the package says,
and it should therefore be reproducible from the deposited score files rather than resting on a
number in a paper. This recomputes both figures from data/processed/human_scorers/ and prints them
beside the claim.

    python analyses/scripts/verify_readme_reversal.py
"""
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

CLAIM_QUOTED = +0.0056     # AlphaMissense minus REVEL, each on its own covered subset
CLAIM_MATCHED = -0.0096    # AlphaMissense minus REVEL, on the variants both reach
SRC = "data/processed/human_scorers"


def auroc(y, s):
    """Mann-Whitney AUROC with midranks."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    need = [os.path.join(SRC, f + ".parquet") for f in ("alphamissense", "revel")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("  cannot check: %s absent (it lives in the undeposited data tree)" % missing[0])
        return 0

    # The scorer files carry variant_id and one score column; the labels live with the panel.
    panel = "data/processed/clinvar_panel.parquet"
    if not os.path.exists(panel):
        print("  cannot check: %s absent (it lives in the undeposited data tree)" % panel)
        return 0
    # The README's panel is ClinVar at one review star or better, which is the filter the
    # study uses throughout. Recomputing without it gives a different panel size and
    # different deltas, and the claim then looks wrong when it is only unfiltered.
    lab = (pl.read_parquet(panel, columns=["variant_id", "label", "stars"])
             .filter(pl.col("stars") >= 1)
             .select(["variant_id", "label"]))
    am = pl.read_parquet(need[0])
    rv = pl.read_parquet(need[1])

    d = (lab.join(am, on="variant_id", how="left")
            .join(rv, on="variant_id", how="left"))
    y = d["label"].to_numpy().astype(int)
    a = d["alphamissense"].to_numpy().astype(float)
    r = d["revel"].to_numpy().astype(float)

    print("  panel: %s variants, %s pathogenic" % (format(len(y), ","), format(int(y.sum()), ",")))
    print("  reach: AlphaMissense %.2f%%   REVEL %.2f%%"
          % (100 * np.isfinite(a).mean(), 100 * np.isfinite(r).mean()))

    quoted = auroc(y, a) - auroc(y, r)
    both = np.isfinite(a) & np.isfinite(r)
    matched = auroc(y[both], a[both]) - auroc(y[both], r[both])

    print()
    print("  %-42s %9s %9s" % ("", "recomputed", "README"))
    print("  %-42s %+9.4f %+9.4f" % ("as usually quoted (own covered subsets)", quoted, CLAIM_QUOTED))
    print("  %-42s %+9.4f %+9.4f" % ("on the %s variants both reach" % format(int(both.sum()), ","),
                                     matched, CLAIM_MATCHED))
    print()
    reverses = (quoted > 0) != (matched > 0)
    print("  sign reverses between the two readings: %s" % ("YES" if reverses else "NO"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
