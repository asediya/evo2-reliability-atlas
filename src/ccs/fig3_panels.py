"""Figure 3 mined-finding panels B–I. Each is a bespoke 'chance-bedrock' idiom, self-contained
(loads its parquet, recomputes, draws). Orientation: evo2_40b_neg & nt_neg as-is; mean-LL delta negated."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl, json
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import fig3_style as S


def _20k():
    cand = pl.read_parquet("data/interim/eqtl_candidates.parquet")
    ev = pl.read_parquet("data/processed/scores/eqtl_evo2_40b.parquet")
    return cand.join(ev, on="variant_id", how="inner")


def _2k():
    lab = pl.read_parquet("data/interim/ablation/eqtl_abl_sample.parquet")
    ev = pl.read_parquet("data/processed/scores_cloud/eqtl_abl_8192_ll.parquet")
    nt = pl.read_parquet("data/processed/scores/nt/eqtl_nt.parquet")
    return lab.join(ev, on="variant_id", how="inner").join(nt, on="variant_id", how="inner")


# ---------------------------------------------------------------- B : conservation bathymetry
def panel_B(ax):
    g = pl.read_parquet("data/interim/ablation/eqtl_gerp_pilot.parquet").drop_nulls("gerp_exact")
    ge = g["gerp_exact"].to_numpy(); lab = g["label"].to_numpy()
    cau, con = ge[lab == 1], ge[lab == 0]
    xs = np.linspace(-7, 4.5, 300); kc = gaussian_kde(cau)(xs); ko = gaussian_kde(con)(xs)
    ax.set_xlim(-7, 4.5); ax.set_ylim(-1.15, 1.15)
    ax.axvspan(-7, 0, color="#A9C4D8", alpha=0.16, zorder=0)                   # accelerated basin (submerged)
    ax.axvspan(2, 4.5, color="#E7EEE9", alpha=0.8, zorder=0)                   # conserved highland
    ax.axvline(0, color=S.SEA, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.axvline(2, color="#6E8B7B", lw=1.2, ls=(0, (2, 2)), zorder=2)
    ax.text(3.2, 0.0, "conserved\nhighland\nEMPTY <1%", ha="center", va="center", fontsize=5.4, color="#5E7A6B", style="italic")
    ax.fill_between(xs, 0, kc / kc.max(), color=S.FAIL, alpha=0.35, zorder=3)
    ax.fill_between(xs, 0, -ko / ko.max(), color=S._shade(S.EQTL, -0.05), alpha=0.35, zorder=3)
    ax.plot(xs, kc / kc.max(), color=S.FAIL, lw=1.2, zorder=4)
    ax.plot(xs, -ko / ko.max(), color=S._shade(S.EQTL, -0.2), lw=1.2, zorder=4)
    ax.text(-6.7, 0.86, "causal eQTLs", fontsize=6.0, color=S.FAIL, fontweight="bold")
    ax.text(-6.7, -0.9, "LD-matched controls", fontsize=6.0, color=S._shade(S.EQTL, -0.2), fontweight="bold", va="top")
    ax.text(-6.7, 0.30, f"GERP AUROC {S.auroc(lab, ge):.2f} — the two ridges\ncoincide; 71% accelerated, <1% conserved.\nNo conservation axis to separate on.",
            fontsize=5.2, color=S.MUTED, va="top", linespacing=1.2)
    ax.set_yticks([]); ax.set_xlabel("GERP conservation   (accelerated ← sea 0 → conserved)", fontsize=6.6, color=S.CAP)
    ax.tick_params(labelsize=6.0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- C : effect-size ghost ramp
def panel_C(ax):
    d = _20k().filter(pl.col("label") == 1)
    z = d["absz"].to_numpy(); sc = d["evo2_40b_neg"].to_numpy()
    scz = (sc - sc.mean()) / sc.std()
    qs = np.quantile(z, np.linspace(0, 1, 8)); cen, mean_s = [], []
    for i in range(7):
        m = (z >= qs[i]) & (z <= qs[i + 1])
        cen.append(z[m].mean()); mean_s.append(scz[m].mean())
    cen = np.array(cen); mean_s = np.array(mean_s)
    ax.set_xlim(qs[0] - 1, qs[-1] + 1); ax.set_ylim(-1.2, 1.6)
    ax.axhspan(-0.35, 0.35, color="#EDEDED", alpha=0.8, zorder=0)              # null band
    ax.fill_between([qs[0], qs[-1]], [0, 0], [0, 1.5], color=S.CODING, alpha=0.09, zorder=0)  # phantom ramp
    ax.plot([qs[0], qs[-1]], [0, 1.5], color=S.CODING, lw=1.0, ls=(0, (3, 2)), alpha=0.5, zorder=1)
    ax.text(qs[-1] * 0.8, 1.35, "ramp a truth-tracking\nmodel would climb", fontsize=5.2, color=S._shade(S.CODING, -0.1), style="italic", ha="right")
    for x, y in zip(cen, mean_s):
        ax.plot([x, x], [0, y], color="#CFCFCF", lw=0.7, zorder=2)             # missed elevation
    ax.plot(cen, mean_s, color=S._shade(S.EQTL, -0.2), lw=1.6, zorder=3, solid_capstyle="round")
    ax.scatter(cen, mean_s, s=24, color=S._shade(S.EQTL, -0.2), edgecolors="white", lw=0.7, zorder=4)
    ax.axhline(0, color=S.BEDROCK, lw=1.4, zorder=3)
    ax.text(qs[0], -0.9, "Evo 2 response is DEAD FLAT across a 4× rise in true effect size\n(top-10% |z| AUROC 0.489, ρ = −0.03 — within noise)",
            fontsize=5.2, color=S.MUTED, va="center", linespacing=1.2)
    ax.set_xlabel("eQTL effect size  |z|  (fine-mapping)", fontsize=6.6, color=S.CAP)
    ax.set_ylabel("Evo 2 score (z)", fontsize=7, color=S.CAP); ax.tick_params(labelsize=6.0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- D : inverted confidence gauge
def panel_D(ax):
    d = _20k().filter(pl.col("label") == 1)
    pip = d["pip"].to_numpy(); sc = d["evo2_40b_neg"].to_numpy(); scz = (sc - sc.mean()) / sc.std()
    order = np.argsort(pip); k = 10; idx = np.array_split(order, k)
    pc = np.array([pip[i].mean() for i in idx]); ms = np.array([scz[i].mean() for i in idx])
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.15, 1.2); ax.set_aspect("equal")
    th = np.linspace(np.pi, 0, k)                                              # low-PIP (left) -> PIP=1 (right)
    R = 1.0
    ax.plot(R * np.cos(np.linspace(np.pi, 0, 60)), R * np.sin(np.linspace(np.pi, 0, 60)), color="#CCC", lw=1.2, zorder=1)
    # ghost 'expected' ribbon rising toward certainty
    rexp = 0.45 + 0.42 * np.linspace(0, 1, k)
    ax.plot(rexp * np.cos(th), rexp * np.sin(th), color=S.CODING, lw=1.0, ls=(0, (3, 2)), alpha=0.5, zorder=2)
    ax.text(0.0, 1.06, "expected: score rises with certainty", fontsize=5.0, color=S._shade(S.CODING, -0.1), ha="center", style="italic")
    # observed ribbon — SAGS toward PIP=1
    robs = 0.62 + 0.30 * (ms - ms.min()) / (ms.max() - ms.min() + 1e-9)
    ax.plot(robs * np.cos(th), robs * np.sin(th), color=S._shade(S.EQTL, -0.2), lw=2.4, zorder=4, solid_capstyle="round")
    ax.scatter(robs * np.cos(th), robs * np.sin(th), s=14, color=S._shade(S.EQTL, -0.2), zorder=5)
    ax.annotate("PIP → 1.0\n(surest causal)", xy=(R, 0.02), xytext=(0.75, 0.35), fontsize=5.2, color=S.FAIL, ha="center",
                arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=0.7))
    ax.text(-1.02, 0.30, "low PIP", fontsize=5.4, color=S.MUTED)
    ax.text(0.0, 0.12, "as certainty ↑,\nmodel score ↓", ha="center", fontsize=6.0, color=S.FAIL, fontweight="bold")
    ax.text(0.0, -0.10, "PIP = 1.0 tail AUROC 0.475 (below chance) · decile ρ = −0.81", ha="center", fontsize=5.0, color=S.MUTED)
    ax.axis("off")


# ---------------------------------------------------------------- E : phantom rank-1 riverbed
def panel_E(ax):
    d = _20k()
    gid = d["gene_id"].to_numpy(); y = d["label"].to_numpy(); s = d["evo2_40b_neg"].to_numpy()
    pct = []
    for g in np.unique(gid):
        m = gid == g; yy = y[m]; ss = s[m]
        if yy.sum() < 1 or (yy == 0).sum() < 1 or m.sum() < 11:
            continue
        for ci in np.where(yy == 1)[0]:
            pct.append((ss < ss[ci]).mean())
    pct = np.array(pct)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.15)
    h, edges = np.histogram(pct, bins=12, range=(0, 1), density=True)
    xc = (edges[:-1] + edges[1:]) / 2
    ax.axhline(1.0, color=S.MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2)           # uniform null waterline
    ax.text(0.5, 1.05, "uniform null (no signal)", ha="center", fontsize=5.2, color=S.MUTED, style="italic")
    ax.fill_between(xc, 0, h, color=S._shade(S.EQTL, -0.05), alpha=0.5, step="mid", zorder=3)
    ax.plot(xc, h, color=S._shade(S.EQTL, -0.2), lw=1.2, drawstyle="steps-mid", zorder=4)
    ax.add_patch(Polygon([(0.90, 0), (1.02, 0), (1.02, 1.02), (0.90, 1.02)], closed=True, facecolor=S.CODING, alpha=0.10, zorder=1))
    ax.annotate("a working model\npiles mass HERE\n(rank-1) — it never comes", xy=(0.96, 0.5), xytext=(0.62, 0.72),
                fontsize=5.2, color=S._shade(S.CODING, -0.1), ha="center", style="italic",
                arrowprops=dict(arrowstyle="->", color=S.CODING, lw=0.7))
    ax.text(0.02, 0.16, "more causal variants LOSE to all their LD partners (17.9%)\nthan beat them (15.1%);  mean rank-percentile 0.474",
            fontsize=5.2, color=S.MUTED, va="center", linespacing=1.2)
    ax.set_xlabel("causal variant's rank-percentile among its LD partners  (1 = beats all)", fontsize=6.4, color=S.CAP)
    ax.set_yticks([]); ax.tick_params(labelsize=6.0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- F : no-safe-harbor funnel
def panel_F(ax):
    d = _20k(); y = d["label"].to_numpy(); s = d["evo2_40b_neg"].to_numpy()
    subs = []
    for col, arr in [("chrom", d["chrom"].to_numpy()), ("gene", d["gene_id"].to_numpy())]:
        for v in np.unique(arr):
            m = arr == v
            if y[m].sum() >= 10 and (y[m] == 0).sum() >= 10 and m.sum() >= 40:
                subs.append((S.auroc(y[m], s[m]), int(m.sum())))
    zt = np.quantile(d["absz"].to_numpy(), [0, .33, .66, 1]); az = d["absz"].to_numpy()
    for i in range(3):
        m = (az >= zt[i]) & (az <= zt[i + 1]); subs.append((S.auroc(y[m], s[m]), int(m.sum())))
    dec = np.quantile(s, np.linspace(0, 1, 11))
    aur = np.array([a for a, n in subs]); prec = np.array([np.sqrt(n) for a, n in subs])
    ax.set_xlim(0.40, 0.60); ax.set_ylim(prec.min() * 0.8, prec.max() * 1.08)
    ax.axvspan(0.5, 0.60, color="#E7EEE9", alpha=0.6, zorder=0)                # 'safe harbor' — stays empty
    ax.text(0.545, prec.max() * 0.9, "safe harbor\n(beats chance)\n— EMPTY", ha="center", fontsize=5.4, color="#5E7A6B", style="italic")
    ax.axvline(0.5, color=S.BEDROCK, lw=1.6, zorder=3)
    yy = np.linspace(prec.min() * 0.8, prec.max() * 1.08, 40)                  # null sampling cone
    for sd in (1.0, 2.0):
        ax.plot(0.5 + sd * 0.5 / (yy + 1), yy, color="#CCC", lw=0.6, zorder=1)
        ax.plot(0.5 - sd * 0.5 / (yy + 1), yy, color="#CCC", lw=0.6, zorder=1)
    ax.scatter(aur, prec, s=16, color=S._shade(S.EQTL, -0.15), alpha=0.75, edgecolors="none", zorder=4)
    ax.scatter([0.4875], [prec.max() * 1.02], s=40, marker="D", color=S.FAIL, zorder=5)
    ax.annotate("all ~49 strata pool LEFT of chance\noverall 0.487, 95% CI [.478,.497]\nnot one subgroup escapes",
                xy=(0.4875, prec.max() * 1.02), xytext=(0.415, prec.max() * 0.55), fontsize=5.2, color=S.FAIL, va="center",
                arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=0.7), linespacing=1.2)
    ax.set_xlabel("subgroup AUROC", fontsize=6.6, color=S.CAP)
    ax.set_ylabel("precision √n", fontsize=7, color=S.CAP); ax.tick_params(labelsize=6.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- G : holding hands (Evo2 vs NT)
def panel_G(ax):
    lab = pl.read_parquet("data/interim/ablation/eqtl_abl_sample.parquet")
    nt = pl.read_parquet("data/processed/scores/nt/eqtl_nt.parquet")
    e40 = pl.read_parquet("data/processed/scores/eqtl_evo2_40b.parquet")
    d = lab.join(nt, on="variant_id", how="inner").join(e40, on="variant_id", how="inner")
    ev = d["evo2_40b_neg"].to_numpy(); nt = d["nt_neg"].to_numpy(); lab = d["label"].to_numpy()
    ev = (ev - ev.mean()) / ev.std(); nt = (nt - nt.mean()) / nt.std()
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.hexbin(ev, nt, gridsize=26, cmap="Greys", mincnt=1, zorder=1)
    r = np.corrcoef(ev, nt)[0, 1]
    xx = np.array([-3, 3]); ax.plot(xx, r * xx, color=S.FAIL, lw=1.4, ls=(0, (4, 2)), zorder=3)
    ax.text(2.9, 2.4, f"they agree with\neach other\nr = {r:.2f}", ha="right", fontsize=5.6, color=S.FAIL, fontweight="bold", va="top")
    ax.text(-3.3, -2.6, "…but neither with truth:\nAUROC 0.485 (Evo 2) / 0.486 (NT)\nshared axis orthogonal to causality (ρ ≈ −0.02)",
            fontsize=5.2, color=S.MUTED, va="top", linespacing=1.2)
    ax.set_xlabel("Evo 2-40B score (z)", fontsize=6.6, color=S.CAP)
    ax.set_ylabel("Nucleotide Transformer score (z)", fontsize=6.6, color=S.CAP); ax.tick_params(labelsize=6.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- H : divergence caliper
def panel_H(ax):
    D = json.load(open("reports/fig3_data.json"))["A_scale_ladder"]
    xs = np.log10([1, 7, 40]); cod = [D["coding"][m] for m in ["1B", "7B", "40B"]]
    eq = {e["model"]: e for e in D["eqtl"]}; eqa = [eq[m]["auroc"] for m in ["1B", "7B", "40B"]]
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.3); ax.set_ylim(0.44, 0.99)
    ax.axhspan(0.44, 0.5, color="#A9C4D8", alpha=0.35, zorder=0)
    ax.axhline(0.5, color=S.SEA, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ax.plot(xs, cod, color=S.CODING, lw=2.0, zorder=3); ax.scatter(xs, cod, s=28, facecolors="white", edgecolors=S.CODING, lw=1.5, zorder=4)
    ax.plot(xs, eqa, color=S._shade(S.EQTL, -0.15), lw=1.8, zorder=3); ax.scatter(xs, eqa, s=22, color=S._shade(S.EQTL, -0.15), edgecolors="white", lw=0.7, zorder=4)
    for x, c, e in zip(xs, cod, eqa):                                          # calipers — identical jaw at every scale
        ax.plot([x, x], [e, c], color="#8C7E63", lw=1.0, zorder=3)
        ax.plot([x - 0.03, x + 0.03], [c, c], color="#8C7E63", lw=1.0, zorder=3)
        ax.plot([x - 0.03, x + 0.03], [e, e], color="#8C7E63", lw=1.0, zorder=3)
        ax.text(x + 0.04, (c + e) / 2, f"{c - e:.2f}", fontsize=5.2, color="#8C7E63", va="center", fontweight="bold")
    ax.text(xs[-1], cod[-1] + 0.01, "coding (Fig 2 ref)", fontsize=5.4, color=S.CODING, ha="right", va="bottom", fontweight="bold")
    ax.text(xs[-1], eqa[-1] - 0.03, "eQTL (chance)", fontsize=5.4, color=S._shade(S.EQTL, -0.3), ha="right", fontweight="bold")
    ax.text(np.mean(xs), 0.60, "same gap at 1B and 40B —\nscaling 40× closes nothing", ha="center", fontsize=5.4, color=S.MUTED, style="italic")
    ax.set_xticks(xs); ax.set_xticklabels(["1B", "7B", "40B"], fontsize=6.4)
    ax.set_xlabel("model scale", fontsize=6.6, color=S.CAP)
    ax.set_ylabel("AUROC", fontsize=7.4, color=S.CAP); ax.tick_params(labelsize=6.0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- I : camouflage swarm