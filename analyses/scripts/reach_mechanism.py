# -*- coding: utf-8 -*-
"""B7 (free version) — is a conservation track's silence a property of the VARIANT or the ALIGNMENT?

The manuscript's largest stated unknown: "Why the missingness is class-dependent we could not
establish." The expensive answer needs the Compara alignment blocks. This is the free one, and it
may be the better evidence.

GERP and phyloP are built from DIFFERENT multiple-sequence alignments. Query both at the SAME
variants and two outcomes are possible:

  they fail together   -> something about those variants defeats alignment in general, and
                          missingness is a property of the variant
  they fail separately -> silence is a property of the particular alignment the track was built
                          from, which is the paper's thesis demonstrated rather than asserted

Run on the human atlas panel, the only one where both tracks are genuine third-party releases
(Ensembl 91-mammal GERP; UCSC phyloP100way). The cattle phyloP in the deposit is project-computed
and flagged provisional, so it is deliberately not used here.

Two traps, both live in this file. Missingness is NaN, not null: is_null() reports 100% reach
and deletes the result. And pybigtools fills uncovered bases with 0.0 unless missing=nan is passed,
which would turn every unalignable position into a real score of zero.

    python analyses/scripts/reach_mechanism.py
"""
import io
import json
import math
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analyses/results"
os.makedirs(OUT, exist_ok=True)


def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def newcombe(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return None, [None, None]
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k1 / n1 - k2 / n2
    return d, [d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2),
               d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)]


def main():
    g = pl.read_parquet("reports/gerp_pervariant.parquet").filter(pl.col("species") == "human")
    p = pl.read_parquet(os.path.join(OUT, "human_phylop_pervariant.parquet"))
    j = g.join(p, on="variant_id", how="inner")

    gv = j["gerp"].to_numpy().astype(float)
    pv = j["phylop"].to_numpy().astype(float)
    lab = j["label"].to_numpy().astype(int)
    fg, fp = np.isfinite(gv), np.isfinite(pv)
    n = len(j)

    out = {"_generated_by": "analyses/scripts/reach_mechanism.py",
           "_panel": "human atlas, %d variants, the only panel where both conservation tracks are "
                     "genuine third-party releases" % n,
           "_tracks": {"gerp": "Ensembl 91-mammal GERP, as deposited",
                       "phylop": "UCSC phyloP100way (hg38), queried here"},
           "_question": "Do two different alignments go silent on the same variants?",
           "n": n, "n_pos": int((lab == 1).sum()), "n_neg": int((lab == 0).sum())}

    both, neither = int((fg & fp).sum()), int((~fg & ~fp).sum())
    g_only, p_only = int((fg & ~fp).sum()), int((~fg & fp).sum())
    out["reach"] = {"gerp": float(fg.mean()), "phylop": float(fp.mean()),
                    "both": both / n, "neither": neither / n,
                    "gerp_only": g_only / n, "phylop_only": p_only / n,
                    "n_gerp_cannot_score": int((~fg).sum()),
                    "n_phylop_cannot_score": int((~fp).sum()),
                    "n_gerp_holes_that_phylop_fills": p_only,
                    "fraction_of_gerp_holes_phylop_fills":
                        (p_only / int((~fg).sum())) if (~fg).sum() else None}

    obs = (both + neither) / n
    exp = fg.mean() * fp.mean() + (1 - fg.mean()) * (1 - fp.mean())
    out["agreement"] = {
        "observed": float(obs), "expected_by_chance": float(exp),
        "cohens_kappa": float((obs - exp) / (1 - exp)) if exp < 1 else None,
        "_degenerate": bool(fp.all() or (~fp).all()),
        "_reading": "If one track reaches every variant while the other does not, the two "
                    "alignments cannot be failing on the same sequence. Silence is then a "
                    "property of the alignment, not of the variant."}

    cls = {}
    for name, f in (("gerp", fg), ("phylop", fp)):
        kp, np_ = int(f[lab == 1].sum()), int((lab == 1).sum())
        kn, nn = int(f[lab == 0].sum()), int((lab == 0).sum())
        d, ci = newcombe(kn, nn, kp, np_)
        cls[name] = {"reach_pathogenic": kp / np_ if np_ else None,
                     "reach_benign": kn / nn if nn else None,
                     "class_gap_benign_minus_pathogenic": d, "gap_ci95_newcombe": ci,
                     "gap_excludes_zero": bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))}
    out["class_dependence"] = cls

    # Does the choice of track change the measured discrimination on the variants both can score?
    def auroc(score, y):
        s = np.asarray(score, float)
        m = np.isfinite(s)
        s, y = s[m], np.asarray(y)[m]
        if len(set(y.tolist())) < 2:
            return None
        # MIDRANKS. Plain argsort ranks break ties by array order, and these are conservation
        # tracks: 75.3% of GERP's finite values on this panel sit in a tie block, so ordinary ranks
        # make the AUROC a function of row order.
        order = np.argsort(s, kind="mergesort")
        r = np.empty(len(s), float)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    m = fg & fp
    out["discrimination_on_the_shared_set"] = {
        "n": int(m.sum()),
        "auroc_gerp": auroc(gv[m], lab[m]), "auroc_phylop": auroc(pv[m], lab[m]),
        # NOT the full panel: auroc() drops non-finite scores, so each of these is the AUROC on the
        # variants that track can score -- the covered AUROC. "full_panel" would misname the very
        # reach mismatch this paper is about.
        "auroc_gerp_covered": auroc(gv, lab), "auroc_phylop_covered": auroc(pv, lab),
        "_note": "Two conservation tracks on one panel. A difference here is a track difference, "
                 "not a biology difference, and it bounds how much of any Evo 2 margin over "
                 "'conservation' is really a margin over one particular track."}

    q = os.path.join(OUT, "reach_mechanism.json")
    io.open(q, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    r = out["reach"]
    print("  human atlas, %s variants (%s pathogenic, %s benign)"
          % ("{:,}".format(n), "{:,}".format(out["n_pos"]), "{:,}".format(out["n_neg"])))
    print("    GERP reach    %.4f   cannot score %s" % (r["gerp"], "{:,}".format(r["n_gerp_cannot_score"])))
    print("    phyloP reach  %.4f   cannot score %s" % (r["phylop"], "{:,}".format(r["n_phylop_cannot_score"])))
    print("    both %.4f   neither %.4f   GERP-only %.4f   phyloP-only %.4f"
          % (r["both"], r["neither"], r["gerp_only"], r["phylop_only"]))
    if r["fraction_of_gerp_holes_phylop_fills"] is not None:
        print("    -> phyloP fills %.1f%% of the variants GERP cannot score"
              % (100 * r["fraction_of_gerp_holes_phylop_fills"]))
    print()
    print("  class gap (benign reach minus pathogenic reach), same variants")
    for k, v in cls.items():
        if v["class_gap_benign_minus_pathogenic"] is None:
            continue
        print("    %-7s path %.4f  benign %.4f  gap %+.4f [%+.4f, %+.4f] %s"
              % (k, v["reach_pathogenic"], v["reach_benign"],
                 v["class_gap_benign_minus_pathogenic"], v["gap_ci95_newcombe"][0],
                 v["gap_ci95_newcombe"][1], "excludes zero" if v["gap_excludes_zero"] else "ns"))
    d = out["discrimination_on_the_shared_set"]
    print()
    print("  discrimination, shared set n=%s:  GERP %.4f   phyloP %.4f"
          % ("{:,}".format(d["n"]), d["auroc_gerp"], d["auroc_phylop"]))
    print("  full panel:                       GERP %.4f   phyloP %.4f"
          % (d["auroc_gerp_covered"], d["auroc_phylop_covered"]))
    print()
    print("  wrote %s" % q)


if __name__ == "__main__":
    main()
