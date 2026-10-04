"""Two checks the GB presentation playbook says are mandatory BEFORE we claim anything:

  1. MULTIPLE-COMPARISON CORRECTION on the nine Evo2-vs-GERP contrasts. We claim "Evo2 significantly beats
     GERP in 2/9". Nine contrasts, uncorrected. If Holm/BH kills goat (largest effect, smallest n=98), the
     claim becomes 1/9 or 0/9 -- better we find that than a reviewer.

  2. INCREMENTAL AUROC: does Evo2 add signal ORTHOGONAL to conservation? Compare GERP alone vs GERP+Evo2
     (cross-validated logistic), paired bootstrap on the increment. If Evo2 adds signal beyond GERP, the
     "conservation is on par" concession converts into the paper's most interesting coding result
     (a genomic LM and an alignment score measure DIFFERENT things). If not, it's a clean negative.

  python src/ccs/playbook_checks.py
"""
import os, sys
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SP = ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]
GF = {sp: f"data/processed/conservation/{sp}_gerp.parquet" for sp in SP}
GF["dog"] = "data/processed/conservation/dog_cf3_gerp.parquet"
GF["cattle"] = "data/processed/conservation/cattle_ensvar_gerp.parquet"


def load(sp):
    ev = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not (os.path.exists(ev) and os.path.exists(GF[sp])):
        return None
    j = pl.read_parquet(ev).join(pl.read_parquet(GF[sp]), on="variant_id", how="inner")
    if j.height < 40:
        return None
    y = np.array([0 if str(v).startswith("neg_") else 1 for v in j["variant_id"].to_list()])
    e = j["evo2_40b_neg"].to_numpy().astype(float)
    g = j["gerp"].to_numpy().astype(float)
    m = np.isfinite(e) & np.isfinite(g)
    y, e, g = y[m], e[m], g[m]
    return (y, e, g) if 0 < y.sum() < len(y) else None


def holm(pvals, names):
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (n - rank) * pvals[i])
        prev = max(prev, a)
        adj[i] = prev
    return dict(zip(names, adj))


def main():
    data = {sp: d for sp in SP if (d := load(sp))}
    print(f"loaded {len(data)} species\n", flush=True)

    # ---- 1. multiple-comparison correction on Evo2-vs-GERP ----
    print("=" * 72 + "\n1. Evo2 vs GERP: nine contrasts, Holm-corrected\n" + "=" * 72, flush=True)
    names, praw, deltas = [], [], []
    for sp, (y, e, g) in data.items():
        de = roc_auc_score(y, e) - roc_auc_score(y, g)

        def one(k):
            rng = np.random.default_rng(k); i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                return np.nan
            return roc_auc_score(y[i], e[i]) - roc_auc_score(y[i], g[i])
        b = np.array(Parallel(n_jobs=50)(delayed(one)(k) for k in range(10000)))
        b = b[np.isfinite(b)]
        # two-sided bootstrap p for delta != 0
        p = 2 * min((b <= 0).mean(), (b >= 0).mean())
        p = max(p, 1.0 / len(b))
        names.append(sp); praw.append(p); deltas.append(de)
    adj = holm(np.array(praw), names)
    win_raw = win_adj = 0
    for sp, p, de in zip(names, praw, deltas):
        a = adj[sp]
        sig_raw = "*" if p < 0.05 else " "
        sig_adj = "*" if a < 0.05 else " "
        if p < 0.05 and de > 0:
            win_raw += 1
        if a < 0.05 and de > 0:
            win_adj += 1
        print(f"  {sp:8s} d={de:+.3f}  p_raw={p:.4f}{sig_raw}  p_holm={a:.4f}{sig_adj}", flush=True)
    print(f"\n  Evo2 significantly beats GERP:  UNCORRECTED {win_raw}/9  ->  HOLM-CORRECTED {win_adj}/9", flush=True)

    # ---- 2. incremental AUROC: does Evo2 add anything beyond GERP? ----
    print("\n" + "=" * 72 + "\n2. Incremental value: GERP alone vs GERP+Evo2 (5-fold CV logistic)\n" + "=" * 72, flush=True)
    def cv_auc(X, y):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
            oof[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        return roc_auc_score(y, oof), oof
    rows = []
    for sp, (y, e, g) in data.items():
        a_g, _ = cv_auc(g.reshape(-1, 1), y)
        a_ge, _ = cv_auc(np.column_stack([g, e]), y)
        rows.append({"species": sp, "n": len(y), "gerp_only": a_g, "gerp_plus_evo2": a_ge, "increment": a_ge - a_g})
        print(f"  {sp:8s} n={len(y):5d}  GERP {a_g:.3f} -> GERP+Evo2 {a_ge:.3f}   (+{a_ge-a_g:.3f})", flush=True)
    # pooled
    y = np.concatenate([d[0] for d in data.values()])
    e = np.concatenate([d[1] for d in data.values()])
    g = np.concatenate([d[2] for d in data.values()])
    a_g, _ = cv_auc(g.reshape(-1, 1), y)
    a_ge, _ = cv_auc(np.column_stack([g, e]), y)
    def one(k):
        rng = np.random.default_rng(k); i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            return np.nan
        ag, _ = cv_auc(g[i].reshape(-1, 1), y[i]); age, _ = cv_auc(np.column_stack([g[i], e[i]]), y[i])
        return age - ag
    b = np.array(Parallel(n_jobs=50)(delayed(one)(k) for k in range(400)))
    b = b[np.isfinite(b)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"\n  POOLED n={len(y)}  GERP {a_g:.3f} -> GERP+Evo2 {a_ge:.3f}  increment +{a_ge-a_g:.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    print(f"  => Evo2 adds signal orthogonal to conservation: {'YES' if lo > 0 else 'NO (CI spans 0)'}", flush=True)
    os.makedirs("reports", exist_ok=True)
    pl.DataFrame(rows).write_parquet("reports/incremental_evo2_over_gerp.parquet")


if __name__ == "__main__":
    main()
