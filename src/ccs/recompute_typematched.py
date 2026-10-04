"""Recompute the three numbers that still sat on the ascertainment-inflated panel, now TYPE-MATCHED
(missense-only, snpEff consequence). Uses the 1001bp scores (full atlas, best-powered; GERP is
readout-independent) joined to per-variant GERP + snpEff consequence.

  1. Evo2 vs GERP head-to-head, missense-only, per species (Holm-corrected) + pooled
  2. Incremental value: GERP alone vs GERP+Evo2, missense-only pooled (does Evo2 add orthogonal signal?)
  3. Mechanistic: Evo2 AUROC on missense variants where GERP<=0 (constraint absent) -- the grammar-gap test

  python src/ccs/recompute_typematched.py --jobs 50 --nboot 10000
"""
import argparse, os, sys
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SP = ["goat", "chicken", "pig", "sheep", "horse", "cattle", "dog", "human", "cat"]
GF = {sp: f"data/processed/conservation/{sp}_gerp.parquet" for sp in SP}
GF["dog"] = "data/processed/conservation/dog_cf3_gerp.parquet"
GF["cattle"] = "data/processed/conservation/cattle_ensvar_gerp.parquet"


def load(sp, missense=True):
    sf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    cf = f"data/processed/consequence_{sp}.parquet"
    if not (os.path.exists(sf) and os.path.exists(cf) and os.path.exists(GF[sp])):
        return None
    d = (pl.read_parquet(sf)
         .join(pl.read_parquet(GF[sp]), on="variant_id", how="inner")
         .join(pl.read_parquet(cf), on="variant_id", how="inner"))
    if missense:
        d = d.filter(pl.col("consequence") == "missense_variant")
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in d["variant_id"].to_list()])
    e = d["evo2_40b_neg"].to_numpy().astype(float)
    g = d["gerp"].to_numpy().astype(float)
    m = np.isfinite(e) & np.isfinite(g)
    y, e, g = y[m], e[m], g[m]
    return (y, e, g) if (len(y) > 0 and 0 < y.sum() < len(y)) else None


def holm(pvals, names):
    order = np.argsort(pvals); n = len(pvals); adj = np.empty(n); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (n - rank) * pvals[i])); adj[i] = prev
    return dict(zip(names, adj))


def cv_auc(X, y):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        oof[te] = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=50)
    ap.add_argument("--nboot", type=int, default=10000)
    a = ap.parse_args()
    data = {sp: d for sp in SP if (d := load(sp))}
    print(f"loaded {len(data)} species (missense-only)\n")

    # ===== 1. Evo2 vs GERP head-to-head, TYPE-MATCHED =====
    print("=" * 74 + "\n1. Evo2 vs GERP, MISSENSE-ONLY, Holm-corrected\n" + "=" * 74)
    names, praw, deltas = [], [], []
    for sp, (y, e, g) in data.items():
        if len(set(y)) < 2 or len(y) < 20:
            print(f"  {sp:8s} n={len(y):4d}  -- too few"); continue
        de = roc_auc_score(y, e) - roc_auc_score(y, g)

        def one(k):
            rng = np.random.default_rng(k); i = rng.integers(0, len(y), len(y))
            return (roc_auc_score(y[i], e[i]) - roc_auc_score(y[i], g[i])) if len(set(y[i])) > 1 else np.nan
        b = np.array(Parallel(n_jobs=a.jobs)(delayed(one)(k) for k in range(a.nboot)))
        b = b[np.isfinite(b)]
        p = max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1.0 / len(b))
        names.append(sp); praw.append(p); deltas.append(de)
    adj = holm(np.array(praw), names) if names else {}
    win = 0
    for sp, p, de in zip(names, praw, deltas):
        a_ = adj[sp]
        if a_ < 0.05 and de > 0:
            win += 1
        print(f"  {sp:8s} d(Evo2-GERP)={de:+.3f}  p_raw={p:.4f}  p_holm={a_:.4f}{' *' if a_<0.05 else ''}")
    # pooled
    Y = np.concatenate([d[0] for d in data.values()]); E = np.concatenate([d[1] for d in data.values()]); G = np.concatenate([d[2] for d in data.values()])
    ae, ag = roc_auc_score(Y, E), roc_auc_score(Y, G)

    def one(k):
        rng = np.random.default_rng(k); i = rng.integers(0, len(Y), len(Y))
        return (roc_auc_score(Y[i], E[i]) - roc_auc_score(Y[i], G[i])) if len(set(Y[i])) > 1 else np.nan
    b = np.array(Parallel(n_jobs=a.jobs)(delayed(one)(k) for k in range(a.nboot))); b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"\n  POOLED n={len(Y)}  Evo2 {ae:.3f} | GERP {ag:.3f}  delta {ae-ag:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"  Evo2 significantly beats GERP (Holm) in {win}/{len(names)} species; pooled delta {'sig' if lo>0 else 'ns'}")

    # ===== 2. Incremental value, TYPE-MATCHED =====
    print("\n" + "=" * 74 + "\n2. GERP alone vs GERP+Evo2 (missense-only, 5-fold CV logistic)\n" + "=" * 74)
    a_g = cv_auc(G.reshape(-1, 1), Y); a_ge = cv_auc(np.column_stack([G, E]), Y)

    def one2(k):
        rng = np.random.default_rng(k); i = rng.integers(0, len(Y), len(Y))
        if len(set(Y[i])) < 2:
            return np.nan
        return cv_auc(G[i].reshape(-1, 1), Y[i]) - cv_auc(np.column_stack([G[i], E[i]]), Y[i]) * -1 + cv_auc(np.column_stack([G[i], E[i]]), Y[i]) - cv_auc(np.column_stack([G[i], E[i]]), Y[i])
    # simpler: increment per bootstrap
    def inc(k):
        rng = np.random.default_rng(k); i = rng.integers(0, len(Y), len(Y))
        if len(set(Y[i])) < 2:
            return np.nan
        return cv_auc(np.column_stack([G[i], E[i]]), Y[i]) - cv_auc(G[i].reshape(-1, 1), Y[i])
    b = np.array(Parallel(n_jobs=a.jobs)(delayed(inc)(k) for k in range(400))); b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"  POOLED n={len(Y)}  GERP {a_g:.3f} -> GERP+Evo2 {a_ge:.3f}  increment {a_ge-a_g:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"  => Evo2 adds signal orthogonal to conservation (type-matched): {'YES' if lo>0 else 'NO (CI spans 0)'}")

    # ===== 3. Mechanistic: missense where GERP<=0 =====
    print("\n" + "=" * 74 + "\n3. Grammar-gap test: Evo2 on MISSENSE where GERP<=0 (constraint absent)\n" + "=" * 74)
    for lab, mask in [("GERP <= 0 (unconstrained)", G <= 0), ("GERP > 0 (constrained)", G > 0),
                      ("GERP bottom quartile", G <= np.percentile(G, 25))]:
        yy, ee, gg = Y[mask], E[mask], G[mask]
        if len(set(yy)) < 2 or len(yy) < 25:
            print(f"  {lab:28s} n={len(yy):4d} -- too few"); continue
        ae2 = roc_auc_score(yy, ee); ag2 = roc_auc_score(yy, gg)
        def one3(k):
            rng = np.random.default_rng(k); i = rng.integers(0, len(yy), len(yy))
            return roc_auc_score(yy[i], ee[i]) if len(set(yy[i])) > 1 else np.nan
        b = np.array(Parallel(n_jobs=a.jobs)(delayed(one3)(k) for k in range(a.nboot))); b = b[np.isfinite(b)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {lab:28s} n={len(yy):4d} (pos {int(yy.sum())})  Evo2 {ae2:.3f} [{lo:.3f},{hi:.3f}] | GERP {ag2:.3f}")
    print(f"\n  reference: causal common eQTLs Evo2 = 0.498 (properly LD-matched)")


if __name__ == "__main__":
    main()
