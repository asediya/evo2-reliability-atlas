"""THE TYPE-MATCHED ATLAS -- the paper's honest headline.

Problem: atlas negatives were drawn genome-wide from Ensembl variation while positives
are OMIA pathogenic (heavily coding). In cattle: negatives 84% intergenic/intronic, positives 85%
coding-impactful. So the unmatched AUROC largely measures "is this variant in a gene", not "is this
coding variant pathogenic" (cattle 0.900 -> 0.744 missense-only). This replicates competitor rs-7579108
(0.934 -> 0.717 with type-matched controls).

Fix: snpEff consequence for EVERY variant (build-matched DBs we built ourselves), then compare
pathogenic vs benign WITHIN a consequence class. Pooling across species gives the power a single
species lacks. No new Evo2 compute -- re-stratifying variants already scored.

  python src/ccs/type_matched_atlas.py --jobs 50 --nboot 10000
"""
import argparse, os, sys, collections
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SP = ["goat", "chicken", "pig", "sheep", "horse", "cattle", "dog", "human", "cat"]
# snpEff joins compound annotations with '&', severity-first, so the FIRST term carries the
# classification. This set is deliberately identical to fig2_measure.CODING_SO; the two are the
# only places the coding/non-coding split is defined and they must not drift apart. The guard in
# Additional file 3 (recompute_fig1cde.py) asserts they agree on every term in every deposited
# panel, and fails loudly if either is edited alone.
#
# The set holds BARE splice_donor_variant / splice_acceptor_variant, not the compound
# `...&intron_variant` forms it used to hold. Both now classify identically, because the test is
# on the first term; the earlier whole-string membership silently filed a bare
# splice_donor_variant as non-coding, which is how the ClinVar consequence panel's 500
# splice-donor and 500 splice-acceptor variants would have been misread had this rule ever been
# applied to it (it was not -- that panel is stratified term by term and computes no aggregate).
CODING = {"missense_variant", "synonymous_variant", "stop_gained", "stop_lost", "start_lost",
          "frameshift_variant", "initiator_codon_variant",
          "splice_donor_variant", "splice_acceptor_variant"}


# A safety net under the first-term test. Testing the WHOLE string against CODING
# would file 96 protein-altering variants as NON-coding -- 65 missense_variant&splice_region_variant,
# 28 stop_gained&splice_region_variant, 2 start_lost& and 1 stop_lost& -- so the "non-coding
# only" stratum would contain stop-gained variants; on the nine-species atlas panel those 96 alone
# would raise its 8,192-bp AUROC from 0.852 to 0.915.
# The first-term test already prevents that; this catches any future annotation whose
# first term is not coding but a later term alters the CDS.
PROTEIN_ALTERING = {"frameshift_variant", "stop_gained", "stop_lost", "start_lost",
                    "missense_variant"}


def alters_cds(term):
    """True if a (possibly compound) snpEff annotation changes the coding sequence."""
    parts = str(term).split("&")
    return parts[0] in CODING or any(p in PROTEIN_ALTERING for p in parts)


def is_missense(term):
    """True if a (possibly compound) snpEff annotation is missense by its FIRST term, the Methods' rule."""
    return str(term).split("&")[0] == "missense_variant"


def load(sp):
    cf = f"data/processed/consequence_{sp}.parquet"
    sf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not (os.path.exists(cf) and os.path.exists(sf)):
        return None
    d = pl.read_parquet(sf).join(pl.read_parquet(cf), on="variant_id", how="inner")
    # 8192 field-standard scores (capped cloud panels) where available
    c8 = f"data/processed/scores_cloud/atlas8192_{sp}_meanll_8192.parquet"
    if os.path.exists(c8):
        d = d.join(pl.read_parquet(c8), on="variant_id", how="left")
    vid = d["variant_id"].to_list()
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in vid])
    s1 = d["evo2_40b_neg"].to_numpy().astype(float)
    s8 = -d["evo2_meanll_delta"].to_numpy().astype(float) if "evo2_meanll_delta" in d.columns else np.full(len(y), np.nan)
    e = np.array(d["consequence"].to_list())
    return y, s1, s8, e


def boot(y, s, nboot, jobs):
    def one(k):
        rng = np.random.default_rng(k); i = rng.integers(0, len(y), len(y))
        return roc_auc_score(y[i], s[i]) if len(set(y[i])) > 1 else np.nan
    v = np.array(Parallel(n_jobs=jobs)(delayed(one)(k) for k in range(nboot)))
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def rep(y, s, name, nboot, jobs, minn=25):
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < minn or len(set(y)) < 2:
        print(f"  {name:34s} n={len(y):5d}  -- too few / one class"); return None
    a = roc_auc_score(y, s); lo, hi = boot(y, s, nboot, jobs)
    print(f"  {name:34s} n={len(y):5d} (pos {int(y.sum()):4d})  AUROC {a:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    return {"stratum": name, "n": len(y), "pos": int(y.sum()), "auroc": a, "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args()
    data = {sp: d for sp in SP if (d := load(sp))}
    print(f"loaded {len(data)} species\n")

    # ---- the ascertainment picture, pooled ----
    print("=" * 78 + "\nASCERTAINMENT: consequence x label, pooled across species\n" + "=" * 78)
    cp, cn = collections.Counter(), collections.Counter()
    for y, s1, s8, e in data.values():
        for lab, eff in zip(y, e):
            (cp if lab == 1 else cn)[eff] += 1
    tot_p, tot_n = sum(cp.values()), sum(cn.values())
    print(f"  {'consequence':38s} {'POS':>6s} {'NEG':>7s}")
    for eff in sorted(set(cp) | set(cn), key=lambda x: -(cp.get(x, 0) + cn.get(x, 0)))[:9]:
        print(f"  {eff:38s} {cp.get(eff,0):6d} {cn.get(eff,0):7d}")
    ncod_n = sum(v for k, v in cn.items() if k not in CODING)
    print(f"  {'TOTAL':38s} {tot_p:6d} {tot_n:7d}")
    print(f"  -> {100*ncod_n/max(tot_n,1):.0f}% of NEGATIVES are non-coding; "
          f"{100*sum(v for k,v in cp.items() if k in CODING)/max(tot_p,1):.0f}% of POSITIVES are coding")

    rows = []
    for readout, idx in [("1001bp single-pos", 1), ("8192bp mean-LL", 2)]:
        print("\n" + "=" * 78 + f"\nPOOLED ATLAS @ {readout}: unmatched vs TYPE-MATCHED\n" + "=" * 78)
        if not data:
            sys.exit("type_matched_atlas: no input could be loaded, so there is nothing to compute.\n"
                     "This build reads per-species files under data/, which are NOT part of "
                     "the code deposit.\nSee reports/DATA_MANIFEST.md for every data/ path and "
                     "its public source.")
        Y = np.concatenate([d[0] for d in data.values()])
        S = np.concatenate([d[idx] for d in data.values()])
        E = np.concatenate([d[3] for d in data.values()])
        r = rep(Y, S, "ALL (unmatched, as published)", a.nboot, a.jobs)
        if r: rows.append({**r, "readout": readout})
        cds = np.array([alters_cds(x) for x in E])
        # MISSENSE is decided by the first term, as the Methods define the class: snpEff writes the
        # most severe term first, so missense_variant&splice_region_variant is a missense variant that
        # also lies in a splice region, and it counts as missense. Positives and negatives share that
        # first term, which is what makes the stratum type-matched. The coding/non-coding split uses
        # alters_cds(), so a protein-altering later term is filed as coding, never as non-coding.
        mis = np.array([is_missense(x) for x in E])
        for nm, mask in [("MISSENSE-only (type-matched)", mis),
                         ("CODING-only", cds),
                         ("non-coding only", ~cds)]:
            Sx = np.where(mask, S, np.nan)
            r = rep(Y, Sx, nm, a.nboot, a.jobs)
            if r: rows.append({**r, "readout": readout})

    # ---- per-species missense-only @ 8192 ----
    print("\n" + "=" * 78 + "\nPER-SPECIES missense-only (type-matched) @ 8192bp\n" + "=" * 78)
    for sp, (y, s1, s8, e) in data.items():
        s = np.where(np.array([is_missense(x) for x in e]), s8, np.nan)
        r = rep(y, s, sp, max(a.nboot // 4, 2000), a.jobs, minn=20)
        if r: rows.append({**r, "readout": "8192 per-species"})

    os.makedirs("reports", exist_ok=True)
    pl.DataFrame(rows).write_parquet("reports/type_matched_atlas.parquet")
    print(f"\n[type-matched] -> reports/type_matched_atlas.parquet")


if __name__ == "__main__":
    main()
