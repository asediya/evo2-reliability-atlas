"""Supervised probe on Evo2 embeddings — the strong mode for variant effect.
Reads an embedding .npz (ref_emb, alt_emb) + a panel parquet (labels), builds features from the
ref/alt hidden states, and evaluates a cross-validated logistic probe (honest within-species AUROC).
Also supports cross-species transfer: --train a.npz,b.npz --test c.npz (train probe on some species,
test on a held-out species).

Within-species CV:
  python probe_evo2_embeddings.py --emb human_emb.npz --panel human_scoring_windows.parquet --feat delta
"""
import argparse, sys
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def feats(npz, how):
    r, a = npz["ref_emb"], npz["alt_emb"]
    if how == "delta":   return a - r
    if how == "alt":     return a
    if how == "concat":  return np.concatenate([r, a, a - r], axis=1)
    raise ValueError(how)


def load(emb_path, panel_path, how):
    z = np.load(emb_path, allow_pickle=True)
    X = feats(z, how)
    vid = [str(v) for v in z["variant_id"]]
    lab = dict(zip(pl.read_parquet(panel_path)["variant_id"].to_list(),
                   pl.read_parquet(panel_path)["label"].to_list()))
    y = np.array([lab[v] for v in vid])
    return X, y


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb"); ap.add_argument("--panel")
    ap.add_argument("--feat", default="delta", choices=["delta", "alt", "concat"])
    ap.add_argument("--train", help="comma emb.npz:panel.parquet pairs")
    ap.add_argument("--test", help="emb.npz:panel.parquet for held-out species")
    a = ap.parse_args()
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

    if a.train and a.test:  # cross-species transfer
        Xtr, ytr = [], []
        for pair in a.train.split(","):
            e, p = pair.split(":"); X, y = load(e, p, a.feat); Xtr.append(X); ytr.append(y)
        Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
        e, p = a.test.split(":"); Xte, yte = load(e, p, a.feat)
        m = clf().fit(Xtr, ytr); s = m.predict_proba(Xte)[:, 1]
        lo, hi = boot_ci(yte, s)
        print(f"TRANSFER: train n={len(ytr)} -> test n={len(yte)}  AUROC={roc_auc_score(yte,s):.3f} (95%CI {lo:.2f}-{hi:.2f})")
    else:                    # within-species CV
        X, y = load(a.emb, a.panel, a.feat)
        p = cross_val_predict(clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        lo, hi = boot_ci(y, p)
        print(f"embedding-probe ({a.feat}, 5-fold CV): AUROC={roc_auc_score(y,p):.3f} "
              f"(95%CI {lo:.2f}-{hi:.2f})  n={len(y)}, {int(y.sum())} pos, dim={X.shape[1]}")


if __name__ == "__main__":
    main()
