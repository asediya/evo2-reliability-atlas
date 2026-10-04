"""Paper-1 figures from the composition-matched cattle OMIA panel:
(a) reliability-by-consequence-class AUROC, (b) calibration/reliability diagram,
(c) the trust-meter: precision-recall with the guaranteed-precision operating point.
Colorblind-safe (Wong) palette. Saves PNG + PDF to reports/figures/."""
import os
import sys, os
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = f"{CCS_ROOT}"
OUT = f"{B}/reports/figures"; os.makedirs(OUT, exist_ok=True)
C_EVO, C_PHY, C_ENS = "#0072B2", "#E69F00", "#009E73"  # Wong colorblind-safe
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

df = (pl.read_parquet(f"{B}/data/interim/omia_matched_panel.parquet")
      .join(pl.read_parquet(f"{B}/data/processed/scores/omia_matched_evo2_scores.parquet"), on="variant_id")
      .join(pl.read_parquet(f"{B}/data/processed/conservation/omia_matched_phylop.parquet"), on="variant_id")
      .filter(pl.col("phylop").is_not_nan() & pl.col("evo2_neg").is_not_nan()))
y = df["label"].to_numpy().astype(int)
evo = df["evo2_neg"].to_numpy(); phy = df["phylop"].to_numpy()
X = np.column_stack([evo, phy]); ens = np.full(len(y), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    ens[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

# consequence class
snv = pl.read_csv(f"{B}/data/raw/truth/omia_cattle_snvs.csv", infer_schema_length=0).select(["variant_id", "omia_variant_id"])
allv = (pl.read_csv(f"{B}/data/raw/truth/omia_all_variants.csv", infer_schema_length=0)
        .filter(pl.col("Species Name") == "taurine cattle")
        .select([pl.col("OMIA Variant ID").cast(pl.Utf8).alias("omia_variant_id"), pl.col("Variant Effect").alias("effect")]))
cls = snv.with_columns(pl.col("omia_variant_id").cast(pl.Utf8)).join(allv, on="omia_variant_id", how="left")
eff = df.join(cls.select(["variant_id", "effect"]).unique(subset=["variant_id"]), on="variant_id", how="left")["effect"].to_numpy().astype(object)
neg = y == 0

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) reliability by class
classes = [("stop-gain", "nonsense (stop-gain)"), ("splicing", "splicing"), ("missense", "missense"), ("regulatory", "regulatory")]
labels, Ns, aucs = ["ALL"], [int(y.sum())], {"evo2": [], "phylop": [], "ens": []}
def a3(mask):
    sel = mask | neg; yy = y[sel]
    return (roc_auc_score(yy, evo[sel]), roc_auc_score(yy, phy[sel]), roc_auc_score(yy, ens[sel]))
for nm, m in [("ALL", y == 1)] + [(n, (y == 1) & (eff == c)) for n, c in classes]:
    if nm != "ALL": labels.append(nm.replace(" ", "\n")); Ns.append(int(m.sum()))
    v = a3(m); aucs["evo2"].append(v[0]); aucs["phylop"].append(v[1]); aucs["ens"].append(v[2])
xp = np.arange(len(labels)); w = 0.26
ax[0].bar(xp - w, aucs["evo2"], w, label="Evo2", color=C_EVO)
ax[0].bar(xp, aucs["phylop"], w, label="conservation", color=C_PHY)
ax[0].bar(xp + w, aucs["ens"], w, label="ensemble", color=C_ENS)
ax[0].axhline(0.5, ls=":", c="gray", lw=1)
for i, n in enumerate(Ns): ax[0].text(xp[i], 1.02, f"N={n}", ha="center", fontsize=7, color="#555")
ax[0].set_xticks(xp); ax[0].set_xticklabels([labels[0]] + [c[0].replace(" ", "\n") for c in classes], fontsize=8)
ax[0].set_ylim(0.45, 1.08); ax[0].set_ylabel("AUROC"); ax[0].set_title("(a) Reliability by variant class", loc="left", fontweight="bold")
ax[0].legend(frameon=False, fontsize=8, loc="lower left")

# (b) calibration (reliability diagram) for ensemble
# (b) two complementary axes: Evo2 vs conservation (visualizes the orthogonal-signal headline)
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
neg_i = np.where(y == 0)[0]; neg_s = rng.choice(neg_i, size=min(600, len(neg_i)), replace=False)
pos_i = np.where(y == 1)[0]
ax[1].scatter(phy[neg_s], evo[neg_s], s=8, c="#BBBBBB", alpha=0.5, edgecolor="none", label="benign (background)")
ax[1].scatter(phy[pos_i], evo[pos_i], s=16, c="#D55E00", alpha=0.85, edgecolor="none", label="deleterious (OMIA)")
r = float(spearmanr(phy, evo).statistic)
hi = float(np.nanpercentile(evo[pos_i], 97))  # clip lone outlier so the bulk is visible
ax[1].set_ylim(-8, hi * 1.05)
ax[1].set_xlabel("conservation (phyloP)"); ax[1].set_ylabel("Evo2 deleteriousness (−ΔLL)")
ax[1].set_title("(b) Two complementary axes", loc="left", fontweight="bold")
ax[1].legend(frameon=False, fontsize=8, loc="upper right", markerscale=1.6)
ax[1].text(0.04, 0.93, f"Spearman r = {r:.2f}\n→ not redundant", transform=ax[1].transAxes, fontsize=8)
# calibration ECE (reported in caption/text, not plotted — probs are bimodal)
p_cal = np.full(len(y), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(ens.reshape(-1, 1), y):
    ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(ens[tr], y[tr]); p_cal[te] = ir.predict(ens[te])
ece = float(sum(len(g) / len(y) * abs(y[g].mean() - p_cal[g].mean()) for g in np.array_split(np.argsort(p_cal), 10)))

# (c) trust-meter: precision-recall + guaranteed operating point
for s, c, nm in [(evo, C_EVO, "Evo2"), (phy, C_PHY, "conservation"), (ens, C_ENS, "ensemble")]:
    pr, rc, _ = precision_recall_curve(y, s); ax[2].plot(rc, pr, color=c, label=nm, lw=1.8)
ax[2].axhline(0.90, ls="--", c="#333", lw=1); ax[2].text(0.02, 0.905, "90% precision guarantee", fontsize=7)
# ensemble operating point at 90% guaranteed precision -> recall 0.74 (from trust_layer RCPS)
ax[2].scatter([0.74], [0.947], color=C_ENS, zorder=5, s=45, edgecolor="k")
ax[2].annotate("ensemble @ guaranteed 90%\nprec → 74% recall", (0.74, 0.947), (0.30, 0.55), fontsize=7,
               arrowprops=dict(arrowstyle="->", color="#333"))
ax[2].set_xlabel("recall (disease variants recovered)"); ax[2].set_ylabel("precision")
ax[2].set_ylim(0, 1.02); ax[2].set_title("(c) Trust-meter: precision–recall", loc="left", fontweight="bold")
ax[2].legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle("Cattle OMIA (composition-matched): Evo2 + conservation + calibrated trust-layer", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/fig1_cattle_trustlayer.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_cattle_trustlayer.pdf", bbox_inches="tight")
print(f"wrote {OUT}/fig1_cattle_trustlayer.png (+.pdf)")
print(f"per-class AUROC ensemble: " + ", ".join(f"{l.replace(chr(10),' ')}={a:.3f}" for l, a in zip(['ALL']+[c[0] for c in classes], aucs['ens'])))
