"""The trust-layer (Checkpoint-1 novel core): calibrate a variant-effect score, then provide
RISK-CONTROLLED SELECTIVE PREDICTION — guarantee a target precision on the variants the model
calls 'deleterious', and report how many true disease variants that recovers (recall).

For imbalanced deleterious detection (prevalence ~10%), raw accuracy is meaningless (call-all-benign
=90%). The meaningful guarantee for a vet is: "of the variants I flag, >=X% are truly deleterious."

Outputs per scorer (Evo2, phyloP, ensemble):
  - discrimination: AUROC, AUPRC (bootstrap CIs)
  - calibration: Brier, ECE (isotonic cross-conformal)
  - risk-controlled operating points: at target precision {0.90,0.95}, achieved test precision + recall,
    selected via a binomial lower-confidence-bound (Clopper-Pearson) on a calibration split (RCPS-style),
    averaged over repeated splits.
  - a precision-recall curve saved for figures.

  python trust_layer.py --panel omia_panel.parquet --evo2 omia_evo2_scores.parquet --phylop omia_phylop.parquet --out trust_results
"""
import argparse, os, sys, json
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import beta as beta_dist
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def cc_isotonic(score, y, n_splits=5, seed=0):
    p = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(score.reshape(-1, 1), y):
        ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(score[tr], y[tr])
        p[te] = ir.predict(score[te])
    return p


def prec_lcb(k_correct, n, conf=0.9):
    """Clopper-Pearson lower bound on precision given k_correct of n calls."""
    if n == 0:
        return 0.0
    if k_correct == n:
        return (1 - conf) ** (1 / n)
    return float(beta_dist.ppf(1 - conf, k_correct, n - k_correct + 1))


def rcps_precision_op(score, y, target_prec, delta=0.1, n_rep=50, test_frac=0.5, seed=0):
    """RCPS-style: on calibration, pick the LOWEST score threshold whose precision-LCB>=target
    (maximizing calls); evaluate precision+recall on held-out test. Repeat, average."""
    rng = np.random.default_rng(seed)
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_frac, random_state=seed)
    precs, recs, ncalls = [], [], []
    P_total = int(y.sum())
    for cal, te in sss.split(score.reshape(-1, 1), y):
        sc, yc = score[cal], y[cal]
        order = np.argsort(-sc, kind="stable")
        yc_o = yc[order]
        cum_correct = np.cumsum(yc_o)
        ks = np.arange(1, len(yc_o) + 1)
        # precision-LCB at each threshold (call top-k on calib)
        lcbs = np.array([prec_lcb(int(cum_correct[i]), int(ks[i]), 1 - delta) for i in range(len(ks))])
        ok = np.where(lcbs >= target_prec)[0]
        if len(ok) == 0:
            precs.append(np.nan); recs.append(0.0); ncalls.append(0); continue
        k_star = ok.max()                      # most calls that still guarantee
        thr = sc[order][k_star]                # score threshold
        # evaluate on test
        st, yt = score[te], y[te]
        called = st >= thr
        if called.sum() == 0:
            precs.append(np.nan); recs.append(0.0); ncalls.append(0); continue
        precs.append(float(yt[called].mean()))
        recs.append(float(yt[called].sum() / max(1, yt.sum())))
        ncalls.append(int(called.sum()))
    return (np.nanmean(precs), np.nanmean(recs), np.nanmean(ncalls),
            float(np.mean(np.array(recs) > 0)))


def boot_auc(y, s, nb=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); a = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--evo2", required=True)
    ap.add_argument("--phylop", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = (pl.read_parquet(a.panel)
          .join(pl.read_parquet(a.evo2), on="variant_id", how="inner")
          .join(pl.read_parquet(a.phylop), on="variant_id", how="inner"))
    df = df.filter(pl.col("phylop").is_not_nan() & pl.col("evo2_neg").is_not_nan())
    y = df["label"].to_numpy().astype(int)
    scores = {"evo2": df["evo2_neg"].to_numpy().astype(float),
              "phylop": df["phylop"].to_numpy().astype(float)}
    # calibrated ensemble: logistic(evo2_neg, phylop), cross-conformal probability = its own score
    X = np.column_stack([scores["evo2"], scores["phylop"]])
    ens = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]); ens[te] = lr.predict_proba(X[te])[:, 1]
    scores["evo2+phylop"] = ens

    print(f"n={len(y)}  positives(OMIA deleterious)={int(y.sum())}  prevalence={y.mean():.3f}\n")
    print(f"{'scorer':14s}{'AUROC':>18s}{'AUPRC':>8s}{'Brier':>8s}{'ECE':>7s} | "
          f"{'prec@target':>12s}{'recall':>8s}{'#calls':>7s}")
    results = {}
    for name, s in scores.items():
        p = cc_isotonic(s, y) if name != "evo2+phylop" else s  # ensemble already a prob
        auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s)
        aup = average_precision_score(y, s); br = brier_score_loss(y, p); e = ece(y, p)
        row = {"auroc": auc, "auroc_ci": [lo, hi], "auprc": aup, "brier": br, "ece": e, "ops": {}}
        line = f"{name:14s}{auc:6.3f}[{lo:.3f},{hi:.3f}]{aup:8.3f}{br:8.3f}{e:7.3f} | "
        for tp in (0.90, 0.95):
            pr, rc, nc, feas = rcps_precision_op(s, y, tp)
            row["ops"][f"prec{tp}"] = {"achieved_prec": pr, "recall": rc, "n_calls": nc, "feasible_frac": feas}
            if tp == 0.90:
                line += f"{pr:6.3f}@{tp:.2f}{rc:8.3f}{nc:7.0f}"
        print(line)
        results[name] = row
        # PR curve for figures
        order = np.argsort(-s, kind="stable"); yo = y[order]; cc = np.cumsum(yo); ks = np.arange(1, len(yo) + 1)
        pl.DataFrame({"n_calls": ks, "precision": cc / ks, "recall": cc / max(1, y.sum())}).write_parquet(
            os.path.join(a.out, f"pr_{name.replace('+','_')}.parquet"))

    json.dump(results, open(os.path.join(a.out, "trust_results.json"), "w"), indent=2)
    print(f"\nwrote {a.out}/trust_results.json + PR curves")
    print("\nREAD: at a GUARANTEED precision target, 'recall' = fraction of true cattle disease variants "
          "the trust-layer confidently recovers; the rest it abstains on. This is the vet-facing 'trust-meter'.")


if __name__ == "__main__":
    main()
