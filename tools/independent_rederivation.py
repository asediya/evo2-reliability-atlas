# -*- coding: utf-8 -*-
"""Re-derive every headline number WITHOUT using the code that produced it.

The point of this script is that it shares as little as possible with the analysis it checks. If my
auditor has a bug, or if I have simply invented a number, this should disagree.

Deliberate independence:
  * AUROC comes from sklearn.metrics.roc_auc_score, never from glmtrust.metrics.auroc.
  * Reach comes from direct boolean counting on the raw arrays, never from ScorerAudit.
  * The must-answer AUROC is checked by BRUTE-FORCE enumeration of every (positive, negative) pair
    on a subsample, never from the closed form.
  * The ClinVar label counts are recounted straight from the .vcf.gz by string matching, never from
    the parquet my parser wrote.
  * The matched delta is recomputed by masking and calling sklearn twice.

Anything that disagrees is printed as MISMATCH and the script exits non-zero.
"""
import gzip
import os
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
# this hard-coded the author's drive, so the script could not run
# from a clean extraction of the deposit. Resolve the repository root from this file instead.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAIL = []
TOL = 1e-4


def check(label, claimed, got, tol=TOL):
    ok = abs(float(claimed) - float(got)) < tol
    if not ok:
        FAIL.append("%s: claimed %s, independently got %s" % (label, claimed, got))
    print("  %-52s claimed %-12s got %-12s %s"
          % (label, ("%.4f" % claimed) if isinstance(claimed, float) else claimed,
             ("%.4f" % got) if isinstance(got, float) else got, "ok" if ok else "MISMATCH"))
    return ok


print("=" * 96)
print("  1. ClinVar panel counts -- recounted from the raw VCF, not from my parquet")
print("=" * 96)
PATH = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BEN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
npos = nneg = 0
with gzip.open("data/external/human_panel/clinvar.vcf.gz", "rt", errors="replace") as fh:
    for line in fh:
        if line[0] == "#":
            continue
        f = line.split("\t", 8)
        if len(f) < 8:
            continue
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            continue
        info = f[7]
        i = info.find("CLNSIG=")
        if i < 0:
            continue
        sig = info[i + 7:].split(";", 1)[0]
        if sig in PATH:
            npos += 1
        elif sig in BEN:
            nneg += 1
check("ClinVar pathogenic SNVs", 189293, npos, tol=0.5)
check("ClinVar benign SNVs", 1295543, nneg, tol=0.5)
check("ClinVar panel total", 1484836, npos + nneg, tol=0.5)

print()
print("=" * 96)
print("  2. Human panel reach and accuracy -- sklearn and direct counting only")
print("=" * 96)
panel = pl.read_parquet("data/processed/clinvar_panel.parquet")
for f in ("alphamissense", "revel"):
    panel = panel.join(pl.read_parquet(f"data/processed/human_scorers/{f}.parquet"),
                       on="variant_id", how="left")
p1 = panel.filter(pl.col("stars") >= 1)
y = p1["label"].to_numpy().astype(int)
am = p1["alphamissense"].to_numpy().astype(float)
rv = p1["revel"].to_numpy().astype(float)
check("panel n at >=1 star", 1434335, len(y), tol=0.5)
check("panel positives at >=1 star", 170680, int(y.sum()), tol=0.5)

for nm, s, r_all, r_pos, r_neg, cov in (("alphamissense", am, .135, .337, .108, .9631),
                                        ("revel", rv, .176, .373, .150, .9575)):
    fin = np.isfinite(s)
    check("%s reach (all)" % nm, r_all, fin.mean(), tol=5e-4)
    check("%s reach (positives)" % nm, r_pos, fin[y == 1].mean(), tol=5e-4)
    check("%s reach (negatives)" % nm, r_neg, fin[y == 0].mean(), tol=5e-4)
    check("%s covered AUROC (sklearn)" % nm, cov, roc_auc_score(y[fin], s[fin]))

print()
print("=" * 96)
print("  3. The reversal -- recomputed by masking, sklearn twice")
print("=" * 96)
both = np.isfinite(am) & np.isfinite(rv)
check("n matched", 192168, int(both.sum()), tol=0.5)
naive = roc_auc_score(y[np.isfinite(am)], am[np.isfinite(am)]) - \
        roc_auc_score(y[np.isfinite(rv)], rv[np.isfinite(rv)])
matched = roc_auc_score(y[both], am[both]) - roc_auc_score(y[both], rv[both])
check("delta as usually reported", 0.0056, naive)
check("delta matched", -0.0096, matched)
print("  -> sign flips: %s" % ((naive > 0) != (matched > 0)))
if (naive > 0) == (matched > 0):
    FAIL.append("the reversal does not reproduce independently")

print()
print("=" * 96)
print("  4. must-answer AUROC -- BRUTE FORCE pair enumeration, no closed form")
print("=" * 96)
rng = np.random.default_rng(0)
idx = rng.choice(len(y), 4000, replace=False)      # brute force is O(pos*neg); subsample
ys, ams = y[idx], am[idx]
pos_i = np.flatnonzero(ys == 1)
neg_i = np.flatnonzero(ys == 0)
fin_s = np.isfinite(ams)
tot = 0.0
for i in pos_i:
    if not fin_s[i]:
        tot += neg_i.size * 0.5
        continue
    si = ams[i]
    for j in neg_i:
        if not fin_s[j]:
            tot += 0.5
        elif si > ams[j]:
            tot += 1.0
        elif si == ams[j]:
            tot += 0.5
brute = tot / (pos_i.size * neg_i.size)
kp, kn = int(fin_s[ys == 1].sum()), int(fin_s[ys == 0].sum())
cov_sub = roc_auc_score(ys[fin_s], ams[fin_s])
closed = (cov_sub * kp * kn + 0.5 * (pos_i.size * neg_i.size - kp * kn)) / (pos_i.size * neg_i.size)
print("  subsample n=%d  (%d pos, %d neg)" % (len(idx), pos_i.size, neg_i.size))
check("must-answer by brute force vs closed form", brute, closed, tol=1e-9)

print()
print("=" * 96)
print("  5. Missense-only claim -- recomputed from scratch")
print("=" * 96)
pm = panel.filter((pl.col("stars") >= 1) & (pl.col("consequence") == "missense_variant"))
ym = pm["label"].to_numpy().astype(int)
amm = pm["alphamissense"].to_numpy().astype(float)
rvm = pm["revel"].to_numpy().astype(float)
check("missense panel n", 204191, len(ym), tol=0.5)
fa, fb = np.isfinite(amm), np.isfinite(rvm)
bm = fa & fb
mdelta = roc_auc_score(ym[bm], amm[bm]) - roc_auc_score(ym[bm], rvm[bm])
check("missense matched delta", -0.0096, mdelta)


def ma(yy, ss):
    f = np.isfinite(ss)
    kp_, kn_ = int(f[yy == 1].sum()), int(f[yy == 0].sum())
    np_, nn_ = int((yy == 1).sum()), int((yy == 0).sum())
    return (roc_auc_score(yy[f], ss[f]) * kp_ * kn_ + 0.5 * (np_ * nn_ - kp_ * kn_)) / (np_ * nn_)


check("missense must-answer delta", -0.0357, ma(ym, amm) - ma(ym, rvm))
check("AlphaMissense must-answer (missense)", 0.9234, ma(ym, amm))
check("REVEL must-answer (missense)", 0.9591, ma(ym, rvm))

print()
print("=" * 96)
print("  6. Cross-species 2-of-9 vs 8-of-9 -- recomputed, sklearn only")
print("=" * 96)
sys.path.insert(0, "src/ccs")
from fig5_stats import SP                                             # noqa: E402

n_cov = n_ma = 0
for sp in SP:
    cons, ev = SP[sp]
    g = pl.read_parquet(f"data/processed/conservation/{cons}_gerp.parquet").select(["variant_id", "gerp"])
    s = pl.read_parquet(f"data/processed/scores/{ev}_evo2_40b_local_scores.parquet") \
          .select(["variant_id", pl.col("evo2_40b_neg").alias("evo2")])
    d = g.join(s, on="variant_id", how="inner")
    vid = d["variant_id"].to_list()
    yy = np.array([0 if str(v).startswith("neg_") else 1 for v in vid])
    gg = d["gerp"].to_numpy().astype(float)
    ee = d["evo2"].to_numpy().astype(float)
    bb = np.isfinite(gg) & np.isfinite(ee)
    # covered-subset contrast, bootstrap CI, sklearn only
    r = np.random.default_rng(0)
    ip, ineg = np.flatnonzero(yy[bb] == 1), np.flatnonzero(yy[bb] == 0)
    yb, gb, eb = yy[bb], gg[bb], ee[bb]
    dr = np.empty(600)
    for k in range(dr.size):
        ii = np.concatenate([r.choice(ip, ip.size, True), r.choice(ineg, ineg.size, True)])
        dr[k] = roc_auc_score(yb[ii], eb[ii]) - roc_auc_score(yb[ii], gb[ii])
    lo, hi = np.percentile(dr, [2.5, 97.5])
    cov_sig = lo > 0
    # must-answer contrast, bootstrap over the FULL panel
    dr2 = np.empty(600)
    ip2, in2 = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    for k in range(dr2.size):
        ii = np.concatenate([r.choice(ip2, ip2.size, True), r.choice(in2, in2.size, True)])
        dr2[k] = ma(yy[ii], ee[ii]) - ma(yy[ii], gg[ii])
    lo2, hi2 = np.percentile(dr2, [2.5, 97.5])
    ma_sig = lo2 > 0
    n_cov += cov_sig
    n_ma += ma_sig
    print("    %-9s covered %+.4f [%+.3f,%+.3f]%s   must-answer %+.4f [%+.3f,%+.3f]%s"
          % (sp, np.median(dr), lo, hi, "*" if cov_sig else " ",
             np.median(dr2), lo2, hi2, "*" if ma_sig else " "))
check("species where Evo2 ahead, covered", 2, n_cov, tol=0.5)
check("species where Evo2 ahead, must-answer", 8, n_ma, tol=0.5)

print()
print("=" * 96)
if FAIL:
    print("  %d MISMATCH(ES):" % len(FAIL))
    for f in FAIL:
        print("    -", f)
    sys.exit(1)
print("  ALL INDEPENDENT RE-DERIVATIONS AGREE")
sys.exit(0)
