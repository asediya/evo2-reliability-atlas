"""STEP 2 hardening: is 'Evo2 beats conservation' REAL across species, after multiple-testing correction?

Reviewer-proofing. The atlas reports a per-species bootstrap statistic `p_40b_gt_cons` = P(delta > 0),
the bootstrap fraction of resamples where AUROC(Evo2-40B) > AUROC(conservation). That is NOT a p-value
and it was never corrected for testing 9 species at once. Here we turn it into a proper one-sided
p-value  p = 1 - P(delta > 0)  for ALL 9 species and apply Benjamini-Hochberg FDR at q = 0.05, then
report how many species survive.

Conservation baseline = GERP (the atlas's sole conservation baseline; the on-disk *_gerp_phylop tracks
were byte-identical copies of GERP and are no longer used). We read the atlas directly so this stays
consistent with build_atlas.py.

  python src/ccs/build_beats_conservation_ci.py   -> logs/beats_conservation_ci.md
"""
import json
import os
import sys

try:  # keep the Δ/≤ glyphs printable on a cp1252 Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ATLAS = "data/processed/atlas_40b.json"
ALPHA = 0.05


def bh_reject(pvals, alpha):
    """Benjamini-Hochberg step-up. Returns (reject_mask, qvals) aligned to input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    # largest rank i (1-based) with p_(i) <= (i/m)*alpha -> reject all up to it
    max_i = 0
    for rank, idx in enumerate(order, 1):
        if pvals[idx] <= rank / m * alpha:
            max_i = rank
    reject = [False] * m
    for rank, idx in enumerate(order, 1):
        if rank <= max_i:
            reject[idx] = True
    # BH-adjusted q-values (monotone from the largest rank down)
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = min(prev, pvals[idx] * m / rank)
        q[idx] = val
        prev = val
    return reject, q


def main():
    atlas = json.load(open(ATLAS))
    rows = []
    for r in atlas:
        pg = r.get("p_40b_gt_cons")
        if pg is None:
            continue
        rows.append(dict(sp=r["species"], n=r["n"], pos=r["n_pos"],
                         auroc_evo2=r.get("auroc_evo2_40b"), auroc_gerp=r.get("auroc_best_cons"),
                         delta=r.get("delta_40b_vs_cons"), p_gt0=pg, p=round(1.0 - pg, 4)))

    reject, q = bh_reject([x["p"] for x in rows], ALPHA)
    for x, rj, qq in zip(rows, reject, q):
        x["q"] = round(qq, 4)
        x["verdict"] = "SIGNIFICANT (BH)" if rj else "n.s. after FDR"
    n_sig = sum(reject)
    m = len(rows)
    survivors = [x["sp"] for x, rj in zip(rows, reject) if rj]

    # order table by raw p ascending for readability
    rows.sort(key=lambda x: x["p"])
    lines = ["# STEP 2 - does Evo2-40B beat conservation (GERP)? One-sided bootstrap p + BH-FDR across 9 species", "",
             "Baseline = GERP (sole conservation baseline; the fabricated byte-identical phyloP copies are no "
             "longer used). The atlas statistic `p_40b_gt_cons` is the bootstrap fraction P(delta>0); we convert "
             "it to a one-sided p-value  p = 1 - P(delta>0)  and apply Benjamini-Hochberg at q<=0.05.", "",
             "| species | n | pos | AUROC Evo2 | AUROC GERP | Δ | P(Δ>0) | one-sided p | BH q | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['sp']} | {x['n']} | {x['pos']} | {x['auroc_evo2']} | {x['auroc_gerp']} | "
                     f"{x['delta']:+.3f} | {x['p_gt0']:.3f} | {x['p']:.3f} | {x['q']:.3f} | {x['verdict']} |")
    lines += ["",
              f"**FDR-corrected result: {n_sig}/{m} species beat GERP after Benjamini-Hochberg (q<=0.05): "
              f"{', '.join(survivors)}.** Both survivors are floor-limited (P(Δ>0)=1.000 -> one-sided p=0.000). ",
              "",
              "**Honest reading:** the bootstrap P(Δ>0) is NOT a p-value and was uncorrected for 9 simultaneous "
              "tests. Once converted to a proper one-sided p-value and BH-corrected, only the two species whose "
              "bootstrap never once favored conservation survive; every borderline win (chicken +0.137 at raw "
              "p=0.028, and all smaller margins) is not significant after FDR. The atlas's headline contribution "
              "is the CALIBRATION / trust-layer transfer, not a raw-accuracy win over conservation."]

    os.makedirs("logs", exist_ok=True)
    open("logs/beats_conservation_ci.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
