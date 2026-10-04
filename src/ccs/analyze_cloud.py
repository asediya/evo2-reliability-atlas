"""Analyze the CLOUD max-bp mean-LL ablation (run on a big GPU with the model resident).

Two things at the field-standard / max window that need the model resident on the GPU:
  - BRCA1 8192-bp mean-LL  = coding positive control (expect AUROC ~0.73; proves the harness works).
  - eQTL 8192 / 16384-bp mean-LL = the regulatory test under the "correct" full-window readout.

  python src/ccs/analyze_cloud.py  -> logs/eqtl_ablation_cloud.md
"""
import os, sys
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def boot(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def scorecol(df):
    return [c for c in df.columns if c != "variant_id"][0]


L = ["# Cloud max-bp mean-LL ablation results", ""]

# --- BRCA1 coding positive control -------------------------------------------------
bf = "data/processed/scores/brca1_evo2_40b_meanll_8192.parquet"
if os.path.exists(bf):
    lab = pl.read_parquet("data/interim/brca1_labels.parquet").select(["variant_id", "label"])
    sc = pl.read_parquet(bf)
    d = lab.join(sc.select(["variant_id", pl.col(scorecol(sc)).alias("s")]), on="variant_id", how="inner").drop_nulls()
    y = d["label"].to_numpy().astype(int); s = -d["s"].to_numpy().astype(float)  # deleterious = -delta
    au = roc_auc_score(y, s); lo, hi = boot(y, s)
    L += [f"**BRCA1 8192-bp mean-LL positive control: AUROC = {au:.3f} (95% CI {lo:.3f}-{hi:.3f}), n={d.height}**",
          "(expect ~0.73; confirms the full-window mean-LL harness works on a coding gold-standard)", ""]
else:
    L += ["BRCA1 positive control: PENDING", ""]

# --- Atlas 9-species (P4 backbone) at 8192-bp mean-LL ------------------------------
ATLAS_DIR = "data/processed/scores/atlas8192"
atlas_rows = []
for sp in ["goat", "chicken", "pig", "sheep", "horse", "cat", "cattle", "dog", "human"]:
    sf = f"{ATLAS_DIR}/{sp}_meanll_8192.parquet"
    wf = f"data/interim/atlas8192/{sp}_windows_8192.parquet"
    if not (os.path.exists(sf) and os.path.exists(wf)):
        atlas_rows.append((sp, None, None)); continue
    w = pl.read_parquet(wf).select(["variant_id", "label"])
    sc = pl.read_parquet(sf)
    d = w.join(sc.select(["variant_id", pl.col(scorecol(sc)).alias("s")]), on="variant_id", how="inner").drop_nulls()
    if d.height < 10 or d["label"].n_unique() < 2:
        atlas_rows.append((sp, d.height, None)); continue
    y = d["label"].to_numpy().astype(int); s = -d["s"].to_numpy().astype(float)  # deleterious = -delta
    au = roc_auc_score(y, s); au = max(au, 1 - au)
    atlas_rows.append((sp, d.height, au))
have = [a for _, _, a in atlas_rows if a is not None]
L += ["## Atlas - per-species AUROC at 8192-bp mean-LL (P4 backbone; first look)",
      "| species | n | AUROC (8192 mean-LL) |", "|---|---|---|"]
for sp, n, a in atlas_rows:
    L.append(f"| {sp} | {n if n else '-'} | {('%.3f' % a) if a is not None else 'PENDING'} |")
if have:
    L.append(f"\n**mean AUROC = {sum(have)/len(have):.3f}** (vs the 1002-bp single-position atlas mean ~0.878). "
             "Full conservation-matched / calibration-transfer / beats-GERP analysis runs on the host after copy-back.")
L.append("")

# --- eQTL regulatory at max bp -----------------------------------------------------
samp = pl.read_parquet("data/interim/ablation/eqtl_abl_sample.parquet").select(["variant_id", "label"])
L += ["| eQTL harness | n | AUROC(|delta|) | 95% CI | AUROC(signed) |", "|---|---|---|---|---|"]
for W in [8192, 16384]:
    f = f"data/processed/scores/eqtl_abl_{W}_ll.parquet"
    if not os.path.exists(f):
        L.append(f"| mean-LL @{W} | - | PENDING | - | - |"); continue
    sc = pl.read_parquet(f)
    d = samp.join(sc.select(["variant_id", pl.col(scorecol(sc)).alias("s")]), on="variant_id", how="inner").drop_nulls()
    y = d["label"].to_numpy().astype(int); s = d["s"].to_numpy().astype(float)
    amag = roc_auc_score(y, np.abs(s)); asgn = roc_auc_score(y, s); asgn = max(asgn, 1 - asgn)
    lo, hi = boot(y, np.abs(s))
    L.append(f"| mean-LL @{W} | {d.height} | {amag:.3f} | {lo:.3f}-{hi:.3f} | {asgn:.3f} |")

L += ["", "## Read",
      "- BRCA1 ~0.73 AND eQTL ~0.5 at 8192/16384  => the regulatory blind spot SURVIVES the field-standard",
      "  full-window mean-LL harness => it is a MODEL property, not an artifact of the block-streaming harness => critique #2 fully closed.",
      "- eQTL rises materially above 0.5 under the better harness => re-scope the regulatory claim honestly.",
      "", "Baseline for comparison (block-streaming, 1002-bp single-position): eQTL |LLR| AUROC 0.496 (full 20k) / 0.521 (this 2k sample)."]
os.makedirs("logs", exist_ok=True)
open("logs/eqtl_ablation_cloud.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
