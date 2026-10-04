"""Evaluate a benchmark panel's scores against its labels (AUROC / AUPRC).

For BRCA1 this reproduces the Evo2 paper's headline metric: the model's mean-LL delta is negative
for loss-of-function variants, so the deleteriousness score is -delta and

    AUROC = roc_auc_score(y, -delta)          # y = 1 for LOF

The paper reports Evo2-1B AUROC = 0.73 over all 3893 SNVs; this script prints the achieved AUROC
next to that target so we can confirm our harness matches.

The scores parquet must have a variant_id column plus one score column (auto-detected: the first
column that isn't variant_id, or pass --score-col). The panel parquet supplies the labels.

  python src/ccs/eval_benchmarks.py --panel brca1 --scores data/processed/scores/brca1_evo2_1b_meanll.parquet
  python src/ccs/eval_benchmarks.py --panel brca1 --scores <path> --score-col evo2_meanll_delta
"""
import argparse, os, sys
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PANELS = {"brca1": ("data/interim/brca1_windows.parquet", 0.73)}


def evaluate(y, delta):
    """Return (auroc, auprc) using -delta as the deleteriousness score (low/negative delta = LOF).
    NaN scores are dropped."""
    y = np.asarray(y).astype(int)
    delta = np.asarray(delta, dtype=float)
    m = ~np.isnan(delta)
    y, delta = y[m], delta[m]
    s = -delta                                  # negate: LOF has the most-negative delta
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, len(y), int(y.sum())
    return roc_auc_score(y, s), average_precision_score(y, s), len(y), int(y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="brca1", choices=list(PANELS))
    ap.add_argument("--scores", required=True, help="parquet with [variant_id, <score>]")
    ap.add_argument("--score-col", default=None,
                    help="score column (default: first non-variant_id column)")
    a = ap.parse_args()

    panel_path, target = PANELS[a.panel]
    panel = pl.read_parquet(panel_path).select(["variant_id", "label"])
    scores = pl.read_parquet(a.scores)

    score_col = a.score_col
    if score_col is None:
        cands = [c for c in scores.columns if c != "variant_id"]
        if not cands:
            ap.error(f"no score column found in {a.scores} (columns: {scores.columns})")
        score_col = cands[0]

    df = panel.join(scores.select(["variant_id", score_col]), on="variant_id", how="inner")
    n_panel, n_join = panel.height, df.height
    df = df.drop_nulls(subset=[score_col])

    y = df["label"].to_numpy()
    delta = df[score_col].to_numpy()
    auroc, auprc, n, n_pos = evaluate(y, delta)

    print(f"panel={a.panel}  scores={os.path.basename(a.scores)}  score_col={score_col}")
    print(f"  matched {n_join}/{n_panel} panel variants; evaluated n={n} "
          f"(pos={n_pos}, prevalence={n_pos / n:.3f})" if n else "  no scored variants")
    print(f"  orientation: AUROC = roc_auc_score(y, -{score_col})   (LOF => negative delta)")
    print(f"  AUROC = {auroc:.4f}   (paper target for 1B = {target:.2f})")
    print(f"  AUPRC = {auprc:.4f}")
    if not np.isnan(auroc):
        # a sanity guard: if the un-negated orientation scores much better, the sign is flipped
        auroc_pos = roc_auc_score(y.astype(int), delta)
        if auroc_pos > auroc:
            print(f"  NOTE: AUROC with +delta = {auroc_pos:.4f} > {auroc:.4f}; "
                  f"if this panel's positives have HIGHER scores, drop the negation.")


if __name__ == "__main__":
    main()
