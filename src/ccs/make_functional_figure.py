"""Figure 2 — the large-N, conservation-controlled well-powering:
(a) AUROC by panel (OMIA disease / population LOF / population missense) x scorer, with bootstrap CIs
    -> the difficulty gradient AND Evo2 beating conservation once conservation is controlled (LOF).
(b) Calibration across the difficulty spectrum: isotonic-calibrated ensemble reliability for all three
    panels; the graded missense panel (N=3762) gives the informative curve spanning [0,1].
(c) Trust-meter: precision-recall for the LOF ensemble with the 90%-precision guarantee.
"""
import os
import sys, os, json
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Repository root, derived so this runs outside the machine it was written on.
CCS_ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = f"{CCS_ROOT}"; OUT = f"{B}/reports/figures"
C_EVO, C_PHY, C_ENS = "#0072B2", "#E69F00", "#009E73"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

PANELS = [
    ("OMIA disease", f"{B}/data/interim/omia_matched_panel.parquet",
     f"{B}/data/processed/scores/omia_matched_evo2_scores.parquet", f"{B}/data/processed/conservation/omia_matched_phylop.parquet",
     f"{B}/results/trust_cattle_matched/trust_results.json", "#333333"),
    ("population LOF", f"{B}/data/interim/func_lof_panel.parquet",
     f"{B}/data/processed/scores/func_evo2_scores.parquet", f"{B}/data/processed/conservation/func_phylop.parquet",
     f"{B}/results/trust_func_lof_panel/trust_results.json", "#0072B2"),
    ("population missense", f"{B}/data/interim/func_mis_panel.parquet",
     f"{B}/data/processed/scores/func_evo2_scores.parquet", f"{B}/data/processed/conservation/func_phylop.parquet",
     f"{B}/results/trust_func_mis_panel/trust_results.json", "#CC79A7"),
]


def load(panel, evo, phy):
    d = (pl.read_parquet(panel).join(pl.read_parquet(evo), on="variant_id", how="inner")
         .join(pl.read_parquet(phy), on="variant_id", how="inner")
         .filter(pl.col("phylop").is_not_nan() & pl.col("evo2_neg").is_not_nan()))
    y = d["label"].to_numpy().astype(int); e = d["evo2_neg"].to_numpy(); p = d["phylop"].to_numpy()
    X = np.column_stack([e, p]); ens = np.full(len(y), np.nan); pcal = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]); ens[te] = lr.predict_proba(X[te])[:, 1]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(ens.reshape(-1, 1), y):
        ir = IsotonicRegression(out_of_bounds="clip").fit(ens[tr], y[tr]); pcal[te] = ir.predict(ens[te])
    return y, e, p, ens, pcal


fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

# (a) AUROC by panel x scorer with CIs (from trust_results.json)
labels = [nm for nm, *_ in PANELS]
xg = np.arange(len(PANELS)); w = 0.26
for j, (key, col) in enumerate([("evo2", C_EVO), ("phylop", C_PHY), ("evo2+phylop", C_ENS)]):
    vals, los, his = [], [], []
    for nm, pan, evo, phy, js, _ in PANELS:
        r = json.load(open(js))[key]; vals.append(r["auroc"]); los.append(r["auroc"] - r["auroc_ci"][0]); his.append(r["auroc_ci"][1] - r["auroc"])
    ax[0].bar(xg + (j - 1) * w, vals, w, yerr=[los, his], capsize=3, color=col,
              label={"evo2": "Evo2", "phylop": "conservation", "evo2+phylop": "ensemble"}[key], error_kw=dict(lw=1))
ax[0].axhline(0.5, ls=":", c="gray", lw=1)
ax[0].set_xticks(xg); ax[0].set_xticklabels(["OMIA\ndisease", "population\nLOF", "population\nmissense"], fontsize=8.5)
ax[0].set_ylim(0.45, 1.02); ax[0].set_ylabel("AUROC")
ax[0].set_title("(a) Signal by panel (conservation controlled →)", loc="left", fontweight="bold", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
ax[0].annotate("Evo2 > conservation\nwhen both classes\nare conserved coding", (1, 0.655), (1.35, 0.52),
               fontsize=7.2, ha="left", arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

# (b) calibration across difficulty (isotonic-calibrated ensemble, rank-equal-count bins)
ax[1].plot([0, 1], [0, 1], ls=":", c="gray", lw=1, label="perfect")
for nm, pan, evo, phy, js, col in PANELS:
    y, e, p, ens, pcal = load(pan, evo, phy)
    g = np.array_split(np.argsort(pcal), 10)
    xs = [float(pcal[k].mean()) for k in g]; ys = [float(y[k].mean()) for k in g]
    ece = float(sum(len(k) / len(y) * abs(y[k].mean() - pcal[k].mean()) for k in g))
    ax[1].plot(xs, ys, "-o", color=col, ms=4, lw=1.3, label=f"{nm} (ECE {ece:.3f})")
ax[1].set_xlabel("predicted P(deleterious)"); ax[1].set_ylabel("observed fraction")
ax[1].set_title("(b) Calibrated across the difficulty spectrum", loc="left", fontweight="bold", fontsize=10.5)
ax[1].legend(frameon=False, fontsize=7.6, loc="upper left")

# (c) trust-meter: LOF ensemble precision-recall + 90% guarantee
yL, eL, pL, ensL, _ = load(PANELS[1][1], PANELS[1][2], PANELS[1][3])
for s, c, nm in [(eL, C_EVO, "Evo2"), (pL, C_PHY, "conservation"), (ensL, C_ENS, "ensemble")]:
    pr, rc, _ = precision_recall_curve(yL, s); ax[2].plot(rc, pr, color=c, lw=1.8, label=nm)
ax[2].axhline(0.90, ls="--", c="#333", lw=1); ax[2].text(0.02, 0.915, "90% precision guarantee", fontsize=7)
opL = json.load(open(PANELS[1][4]))["evo2+phylop"]["ops"]["prec0.9"]
ax[2].scatter([opL["recall"]], [opL["achieved_prec"]], color=C_ENS, s=45, edgecolor="k", zorder=5)
ax[2].annotate(f"ensemble @ guaranteed 90%\nprec → {opL['recall']*100:.0f}% recall", (opL["recall"], opL["achieved_prec"]),
               (0.20, 0.55), fontsize=7, arrowprops=dict(arrowstyle="->", color="#333"))
ax[2].set_xlabel("recall (LOF variants recovered)"); ax[2].set_ylabel("precision"); ax[2].set_ylim(0.3, 1.02)
ax[2].set_title("(c) Trust-meter: population LOF", loc="left", fontweight="bold", fontsize=10.5)
ax[2].legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle("Large-N, conservation-controlled replication: the FM's isolated functional signal + calibration across difficulty", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/fig2_functional_wellpowered.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_functional_wellpowered.pdf", bbox_inches="tight")
print(f"wrote {OUT}/fig2_functional_wellpowered.png (+.pdf)")
