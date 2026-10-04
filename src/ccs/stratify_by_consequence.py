"""'Where it breaks' by variant class: per-consequence-class AUROC of Evo2 / phyloP / ensemble
on the composition-matched cattle OMIA panel. Class = OMIA 'Variant Effect'. Each class's
positives are scored against ALL matched background negatives."""
import os
import sys
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = f"{CCS_ROOT}"

df = (pl.read_parquet(f"{B}/data/interim/omia_matched_panel.parquet")
      .join(pl.read_parquet(f"{B}/data/processed/scores/omia_matched_evo2_scores.parquet"), on="variant_id")
      .join(pl.read_parquet(f"{B}/data/processed/conservation/omia_matched_phylop.parquet"), on="variant_id")
      .filter(pl.col("phylop").is_not_nan() & pl.col("evo2_neg").is_not_nan()))
y = df["label"].to_numpy().astype(int)
X = np.column_stack([df["evo2_neg"].to_numpy(), df["phylop"].to_numpy()])
ens = np.full(len(y), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    ens[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
df = df.with_columns(pl.Series("ensemble", ens))

snv = pl.read_csv(f"{B}/data/raw/truth/omia_cattle_snvs.csv", infer_schema_length=0).select(["variant_id", "omia_variant_id"])
allv = (pl.read_csv(f"{B}/data/raw/truth/omia_all_variants.csv", infer_schema_length=0)
        .filter(pl.col("Species Name") == "taurine cattle")
        .select([pl.col("OMIA Variant ID").cast(pl.Utf8).alias("omia_variant_id"),
                 pl.col("Variant Effect").alias("effect")]))
cls = snv.with_columns(pl.col("omia_variant_id").cast(pl.Utf8)).join(allv, on="omia_variant_id", how="left")
df = df.join(cls.select(["variant_id", "effect"]).unique(subset=["variant_id"]), on="variant_id", how="left")

lab = df["label"].to_numpy(); eff = df["effect"].to_numpy().astype(object)
neg = lab == 0


def auc_for(mask_pos):
    sel = mask_pos | neg
    yy = lab[sel].astype(int)
    r = {}
    for s in ["evo2_neg", "phylop", "ensemble"]:
        v = df[s].to_numpy()[sel]
        r[s] = roc_auc_score(yy, v) if len(np.unique(yy)) > 1 else float("nan")
    return int(mask_pos.sum()), r


print(f"{'class':24s}{'Npos':>5s}{'Evo2':>8s}{'phyloP':>8s}{'ensemble':>10s}")
n, r = auc_for(lab == 1)
print(f"{'ALL':24s}{n:5d}{r['evo2_neg']:8.3f}{r['phylop']:8.3f}{r['ensemble']:10.3f}")
for c in ["nonsense (stop-gain)", "splicing", "missense", "regulatory"]:
    m = (lab == 1) & (eff == c)
    n, r = auc_for(m)
    tag = "" if n >= 15 else "  (underpowered)"
    if n >= 4:
        print(f"{c:24s}{n:5d}{r['evo2_neg']:8.3f}{r['phylop']:8.3f}{r['ensemble']:10.3f}{tag}")
print(f"\n(class annotation coverage: {int((eff!=None).sum() - neg.sum())} of {int((lab==1).sum())} positives labeled)")
