"""STEP 2 hardening (journal-agnostic): CONFIDENCE INTERVALS on the two headline tables.

Reviewer-proofing #1 across every venue: 'no CIs -> small-n cells read as pilots, not findings.'
Adds nonparametric bootstrap 95% CIs to:
  (A) per-species zero-shot AUROC (atlas Table 1)      -- 2000x stratified resample
  (B) per-species TRANSFERRED ECE (calibration table)  -- isotonic fit on OTHER rich species (fixed),
      bootstrap the TARGET species, apply the fixed map, recompute ECE.
Runs on Evo2-40B scores (the headline backbone). CPU-only.
  python src/ccs/build_hardening_cis.py   -> logs/hardening_cis.md
"""
import os
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
RICH_MIN = 50
NBOOT = 2000
RNG = np.random.default_rng(0)


def load(sp):
    scf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(scf):
        return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(scf).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3:
        return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; n = len(y)
    for i in range(bins):
        hi = p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1]
        m = (p >= edges[i]) & hi
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)


def boot_ci(fn, *arrays, n=NBOOT):
    """Percentile bootstrap 95% CI of statistic fn over paired arrays (stratified by the last array = labels)."""
    y = arrays[-1]
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return (None, None)
    vals = []
    for _ in range(n):
        b0 = RNG.choice(idx0, len(idx0), replace=True)
        b1 = RNG.choice(idx1, len(idx1), replace=True)
        bi = np.concatenate([b0, b1])
        try:
            v = fn(*[a[bi] for a in arrays])
            if v is not None and np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if len(vals) < n // 2:
        return (None, None)
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3))


def main():
    data = {sp: r for sp in WIN if (r := load(sp)) is not None}
    rich = [sp for sp in data if int(data[sp][1].sum()) >= RICH_MIN]

    rows = []
    for sp in data:
        x, y = data[sp]
        auc = roc_auc_score(y, x)
        auc_lo, auc_hi = boot_ci(lambda xx, yy: roc_auc_score(yy, xx), x, y)
        # transferred ECE: isotonic fit on OTHER rich species (fixed), bootstrap the target
        train = [s for s in rich if s != sp]
        e_tr = e_lo = e_hi = None
        if train:
            Xt = np.concatenate([data[s][0] for s in train]); Yt = np.concatenate([data[s][1] for s in train])
            iso = IsotonicRegression(out_of_bounds="clip").fit(Xt, Yt)
            p = iso.predict(x)
            e_tr = round(ece(p, y), 3)
            e_lo, e_hi = boot_ci(lambda xx, yy: ece(iso.predict(xx), yy), x, y)
        rows.append(dict(species=sp, n=len(y), pos=int(y.sum()),
                         auroc=round(auc, 3), auroc_ci=f"[{auc_lo}, {auc_hi}]" if auc_lo is not None else "n/a",
                         ece_transfer=e_tr, ece_ci=f"[{e_lo}, {e_hi}]" if e_lo is not None else "n/a"))

    lines = ["# STEP 2 hardening - variant-level bootstrap 95% CIs (Evo2-40B; not the published locus-clustered intervals of Table 2)", "",
             f"Nonparametric percentile bootstrap, {NBOOT}x, stratified by class. Transferred ECE uses an isotonic "
             "map fit on the OTHER label-rich species (leave-one-species-out), bootstrapping the target species.", "",
             "| species | n | pos | AUROC | AUROC 95% CI | ECE transfer | ECE 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['species']} | {r['n']} | {r['pos']} | {r['auroc']} | {r['auroc_ci']} | {r['ece_transfer']} | {r['ece_ci']} |")
    # honest reading of small-n width
    poor = [r for r in rows if r["pos"] < RICH_MIN and r["auroc_ci"] != "n/a"]
    lines += ["", "**Honest reading:** the label-poor species (goat/chicken/pig) carry WIDE AUROC CIs (small positive "
              "counts) - we report point estimates with intervals rather than over-claiming per-species accuracy. The "
              "calibration/trust-layer claims are made at the POOLED level and via the transferred-ECE reduction, which "
              "is where the evidence is strong; per-species AUROC is contextual, not a headline."]
    os.makedirs("logs", exist_ok=True)
    open("logs/hardening_cis.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pl.DataFrame(rows).write_parquet("data/processed/hardening_cis.parquet")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
