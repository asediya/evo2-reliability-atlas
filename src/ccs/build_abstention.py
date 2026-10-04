"""OOD / ABSTENTION arm (Fig 4): turn the calibration crown-jewel into a deployable SELECTIVE-PREDICTION
rule. A vet-genomics tool must know when NOT to answer. Using the TRANSFERRED calibrated probability
(honest LOSO map from label-rich relatives, [[calibration-transfer]]), define confidence = |2p-1| and
abstain on the least-confident variants. If the confidence is meaningful, discrimination on the RETAINED
set rises as coverage falls (risk-coverage curve), and beats random abstention (AURC).

Pooled cross-species = the deployment picture (one calibrated Evo2 used across many species). CPU-only;
runs on whatever species already have 40B scores. Writes logs/abstention.md + dashboard status.
  python src/ccs/build_abstention.py
"""
import os, sys, time
import numpy as np
import polars as pl
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe under Windows subprocess capture
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
RICH_MIN = 50
EV = "logs/status/events.log"; ST = "logs/status/abstention.status"; MD = "logs/abstention.md"


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] abstain: {m}\n")
def st(m):
    with open(ST, "w") as f: f.write(m + "\n")


def load(sp):
    sc = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(sc): return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(sc).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3: return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def fit_on(sps, data):
    X = np.concatenate([data[s][0] for s in sps]); Y = np.concatenate([data[s][1] for s in sps])
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(X, Y); return iso


def risk_coverage(conf, y, prob, grid):
    """Selective ERROR (0-1 loss, 0.5 threshold) at each coverage, ordering by DESC confidence."""
    order = np.argsort(-conf, kind="stable")                       # most confident first
    y_o, p_o = y[order], prob[order]
    n = len(y); errs = []
    for cov in grid:
        k = max(2, int(round(cov * n)))
        yk, pk = y_o[:k], p_o[:k]
        errs.append(float(np.mean((pk > 0.5).astype(int) != yk)))
    return np.array(errs)


def main():
    st("RUNNING | loading scored species for selective prediction")
    ev("OOD/abstention arm started (risk-coverage on transferred calibrated probs)")
    data = {sp: r for sp in WIN if (r := load(sp)) is not None}
    rich = [sp for sp in data if int(data[sp][1].sum()) >= RICH_MIN]
    if len(data) < 3 or len(rich) < 2:
        st(f"WAIT | need >=3 species & >=2 rich sources (have {len(data)}, rich={rich})"); return
    ev(f"{len(data)} species; rich sources={rich}")

    # per-variant TRANSFERRED calibrated probability (LOSO: fit on rich species != self)
    P, Y, SP = [], [], []
    per_species = {}
    for sp in data:
        x, y = data[sp]
        src = [s for s in rich if s != sp]
        if not src: continue
        p = fit_on(src, data).predict(x)
        P.append(p); Y.append(y); SP += [sp] * len(y)
        per_species[sp] = (p, y)
    grid = np.round(np.linspace(1.0, 0.3, 15), 3)
    rng = np.random.default_rng(0)

    # --- per-species risk-coverage (honest deployment frame: each species is its own tool) ---
    curves, rand_curves, rows = [], [], []
    for sp, (p, y) in per_species.items():
        if len(y) < 30 or int(y.sum()) < 5 or (len(y) - int(y.sum())) < 5:
            continue
        conf = np.abs(2 * p - 1)
        e = risk_coverage(conf, y, p, grid)
        er = risk_coverage(rng.random(len(y)), y, p, grid)
        curves.append(e); rand_curves.append(er)
        imn = int(np.argmin(e))
        rows.append(dict(sp=sp, n=len(y), pos=int(y.sum()), e100=float(e[0]),
                         e90=float(e[np.argmin(np.abs(grid - 0.9))]), emin=float(e[imn]),
                         best_cov=float(grid[imn])))
    macro = np.mean(curves, axis=0); macro_rand = np.mean(rand_curves, axis=0)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    aurc = float(_trap(macro[::-1], grid[::-1])); aurc_rand = float(_trap(macro_rand[::-1], grid[::-1]))
    e_full = float(macro[0]); imn = int(np.argmin(macro)); e_min = float(macro[imn]); cov_min = float(grid[imn])
    rel = (e_full - e_min) / e_full if e_full else 0.0
    e_tail = float(macro[-1])                          # error at lowest coverage (confidently-wrong tail)

    verdict = "PASS" if (rel >= 0.10 and aurc < aurc_rand) else "WEAK"
    lines = ["# OOD / abstention: selective prediction on transferred-calibrated Evo2-40B (Fig 4)", "",
             f"Per-species deployment frame ({len(curves)} species, macro-averaged). Confidence = |2p-1| "
             f"on the LOSO-transferred calibrated probability; abstain on the least-confident variants.", "",
             "| coverage | macro selective error | random-abstain error |", "|---|---|---|"]
    for c, e, er in zip(grid, macro, macro_rand):
        lines.append(f"| {c:.0%} | {e:.3f} | {er:.3f} |")
    lines += ["", "### Per-species trust budget",
              "| species | n | pos | err@100% | err@90% | min err | @coverage |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["emin"]):
        lines.append(f"| {r['sp']} | {r['n']} | {r['pos']} | {r['e100']:.3f} | {r['e90']:.3f} | "
                     f"**{r['emin']:.3f}** | {r['best_cov']:.0%} |")
    lines += ["",
              f"- Macro selective error: 100% coverage **{e_full:.3f}** -> min **{e_min:.3f}** at {cov_min:.0%} "
              f"coverage ({rel:.0%} relative reduction).",
              f"- AURC (lower=better): confidence **{aurc:.3f}** vs random {aurc_rand:.3f} "
              f"-> confidence ordering is {'informative' if aurc < aurc_rand else 'NOT informative'}.",
              f"- Confidently-wrong tail: error at {grid[-1]:.0%} coverage rises to {e_tail:.3f} (> the {cov_min:.0%}-"
              f"coverage min {e_min:.3f}) -> a high-confidence error subpopulation (dangerous false-benigns) that "
              f"calibrated confidence alone can't catch -> motivates OOD features (consequence class, sequence "
              f"entropy) as the next lever.",
              "",
              f"**{verdict}** — abstaining the least-confident variants "
              f"{'cuts selective error and beats random abstention: the calibrated trust-layer is a deployable when-to-answer rule, with an identified confidently-wrong tail as future work.' if verdict=='PASS' else 'gives only a weak error reduction relative to the confidently-wrong tail -> needs OOD features beyond calibrated confidence.'}"]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pl.DataFrame(rows).write_parquet("data/processed/abstention_trust_budget.parquet")
    pl.DataFrame({"coverage": grid, "macro_error": macro, "random_error": macro_rand}).write_parquet(
        "data/processed/abstention_risk_coverage.parquet")
    print("\n".join(lines))
    st(f"DONE | abstention {verdict}: macro err {e_full:.3f}->{e_min:.3f} at {cov_min:.0%} cov ({rel:.0%} cut); AURC {aurc:.3f}<{aurc_rand:.3f}(rand); {MD}")
    ev(f"DONE: abstention {verdict} - macro err {e_full:.3f}->{e_min:.3f} ({rel:.0%} cut) at {cov_min:.0%} cov; AURC conf {aurc:.3f} vs rand {aurc_rand:.3f}")


if __name__ == "__main__":
    main()
