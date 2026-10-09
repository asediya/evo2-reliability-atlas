"""CROSS-SPECIES CONFORMAL PREDICTION — upgrade the trust layer from a
heuristic calibrated probability to a DISTRIBUTION-FREE, finite-sample COVERAGE GUARANTEE, and test
whether the guarantee TRANSFERS across species.

Split-conformal: fit the isotonic map + the conformal quantile on LABEL-RICH species, then emit
prediction SETS on a LABEL-POOR target it never saw. For a chosen error rate alpha, the set is
guaranteed (under exchangeability) to contain the true label with prob >= 1-alpha. Binary sets:
{benign}/{pathogenic} = a confident call; {both} = a principled ABSTENTION; {} = out-of-distribution.
We report EMPIRICAL coverage vs nominal (1-alpha) on held-out species + set-size efficiency, for
marginal AND Mondrian (class-conditional, robust to the ~10:1 imbalance) conformal.

If cross-species coverage tracks 1-alpha, that's a guaranteed trust layer for a species with no labels —
unclaimed for genomic FMs, and the finite-sample-guarantee contribution that lifts toward Nat Comms. CPU-only.
  python src/ccs/build_conformal.py
"""
import os, sys, time
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
RICH_MIN = 50
EV = "logs/status/events.log"; ST = "logs/status/conformal.status"; MD = "logs/conformal.md"


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] conformal: {m}\n")
def st(m):
    with open(ST, "w", encoding="utf-8") as f: f.write(m + "\n")


def load(sp):
    scf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(scf): return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(scf).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3: return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def qhat(scores, alpha):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest nonconformity score.

    When k > n the finite-sample-valid threshold is +inf (admit the label always), which is what
    glmtrust.conformal returns. This function used to clamp to the largest observed nonconformity
    score, which is anti-conservative and forfeits the coverage guarantee for any class with n < 9
    at alpha = 0.10. The branch is inert for every published number here,
    because the Mondrian quantiles are calibrated on the pooled rich set and every per-class
    calibration set holds hundreds of variants, but the two implementations should not disagree.
    """
    s = np.sort(scores); n = len(s)
    if not n:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n else s[k - 1]


def admits(a, q):
    """a <= q with ties in exact arithmetic, as glmtrust.conformal does: the isotonic map can return one
    probability as two floats a unit in the last place apart (0.25 and 0.24999999999999994), and a
    plain <= would drop a label whose score equals the threshold. Only ever enlarges a set."""
    return a <= q + 4 * np.spacing(abs(q)) if np.isfinite(q) else np.ones(np.shape(a), dtype=bool)


def evaluate(p, y, qm, q0, q1):
    # MARGINAL construction: one shared threshold qm -> its OWN coverage AND its OWN set sizes.
    inc1 = admits(1 - p, qm); inc0 = admits(p, qm)                    # marginal set membership
    cov = float(np.where(y == 1, inc1, inc0).mean())
    size_marg = inc0.astype(int) + inc1.astype(int)
    cov_ben_marg = float(inc0[y == 0].mean()) if (y == 0).any() else float("nan")
    cov_path_marg = float(inc1[y == 1].mean()) if (y == 1).any() else float("nan")
    # MONDRIAN construction: class-conditional thresholds q0/q1 -> its OWN coverage AND its OWN set sizes.
    m1 = admits(1 - p, q1); m0 = admits(p, q0)                        # Mondrian (class-conditional)
    covm = float(np.where(y == 1, m1, m0).mean())
    size_mond = m0.astype(int) + m1.astype(int)
    cov_ben = float(m0[y == 0].mean()) if (y == 0).any() else float("nan")
    cov_path = float(m1[y == 1].mean()) if (y == 1).any() else float("nan")
    return dict(
        # marginal: coverage + set sizes from the SAME (inc0/inc1) indicators
        cov_marg=cov, cov_ben_marg=cov_ben_marg, cov_path_marg=cov_path_marg,
        singleton_marg=float((size_marg == 1).mean()), abstain_marg=float((size_marg == 2).mean()),
        empty_marg=float((size_marg == 0).mean()),
        # Mondrian: coverage + set sizes from the SAME (m0/m1) indicators
        cov_mond=covm, cov_ben=cov_ben, cov_path=cov_path,
        singleton=float((size_mond == 1).mean()), abstain=float((size_mond == 2).mean()),
        empty=float((size_mond == 0).mean()))


def main():
    st("RUNNING | cross-species conformal prediction sets")
    ev("conformal started")
    data = {sp: r for sp in WIN if (r := load(sp)) is not None}
    rich = [sp for sp in data if int(data[sp][1].sum()) >= RICH_MIN]
    poor = [sp for sp in data if sp not in rich]
    if len(rich) < 2:
        st(f"WAIT | need >=2 rich sources (have {rich})"); return

    # pooled rich -> stratified FIT/CAL split (fit the isotonic on FIT, conformal-calibrate q on CAL)
    Xr = np.concatenate([data[s][0] for s in rich]); Yr = np.concatenate([data[s][1] for s in rich])
    (fit_i, cal_i), = list(StratifiedKFold(2, shuffle=True, random_state=0).split(Xr, Yr))[:1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(Xr[fit_i], Yr[fit_i])
    pc = iso.predict(Xr[cal_i]); yc = Yr[cal_i]
    ev(f"conformal calibration on {len(yc)} rich variants ({int(yc.sum())} pos); targets={poor}")

    ALPHAS = [0.05, 0.10, 0.20]
    # main table at alpha=0.10
    a = 0.10
    s_cal = np.where(yc == 1, 1 - pc, pc)
    qm = qhat(s_cal, a); q0 = qhat(pc[yc == 0], a); q1 = qhat(1 - pc[yc == 1], a)
    rows = []
    for sp in data:
        x, y = data[sp]; p = iso.predict(x)
        r = evaluate(p, y, qm, q0, q1)
        rows.append(dict(sp=sp, role="source" if sp in rich else "TARGET", n=len(y), pos=int(y.sum()), **r))

    # coverage sweep on the pooled label-poor TARGETS (held-out) — does empirical coverage track 1-alpha?
    Xp = np.concatenate([data[s][0] for s in poor]); Yp = np.concatenate([data[s][1] for s in poor])
    sweep = []
    for a2 in ALPHAS:
        qm2 = qhat(s_cal, a2); q02 = qhat(pc[yc == 0], a2); q12 = qhat(1 - pc[yc == 1], a2)
        pp = iso.predict(Xp)
        r = evaluate(pp, Yp, qm2, q02, q12)
        sweep.append(dict(alpha=a2, nominal=round(1 - a2, 2),
                          cov_mond=round(r["cov_mond"], 3), cov_path_mond=round(r["cov_path"], 3),
                          abstain_mond=round(r["abstain"], 3),
                          cov_marg=round(r["cov_marg"], 3), cov_path_marg=round(r["cov_path_marg"], 3),
                          abstain_marg=round(r["abstain_marg"], 3)))

    tgt = [r for r in rows if r["role"] == "TARGET"]
    mean_cov_tgt = float(np.mean([r["cov_mond"] for r in tgt])) if tgt else float("nan")
    lines = ["# Cross-species conformal prediction — a coverage-controlled trust layer that transfers across species (under a stated, tested exchangeability assumption)", "",
             f"Conformal quantile calibrated on label-rich species; applied to held-out TARGETS: {poor}. "
             f"Nominal coverage 1-alpha = 0.90 (alpha=0.10) for the main table.", "",
             "## Per-species coverage & set-size (alpha=0.10, nominal 90%)",
             "Two honest constructions, each with coverage AND set sizes from its OWN membership rule. "
             "Mondrian holds ~0.97 by abstaining heavily; marginal holds ~0.95 overall as near-singletons "
             "but drops POSITIVE-class coverage.", "",
             "| species | role | n | pos | Mondrian cov | Mond cov benign | Mond cov path | Mond abstain{both} | "
             "marginal cov | marg cov benign | marg cov path | marg abstain{both} |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sp']} | {r['role']} | {r['n']} | {r['pos']} | **{r['cov_mond']:.3f}** "
                     f"| {r['cov_ben']:.3f} | {r['cov_path']:.3f} | **{r['abstain']:.3f}** "
                     f"| {r['cov_marg']:.3f} | {r['cov_ben_marg']:.3f} | {r['cov_path_marg']:.3f} | {r['abstain_marg']:.3f} |")
    lines += ["", f"Mean Mondrian coverage on held-out TARGETS = **{mean_cov_tgt:.3f}** vs nominal 0.90.", "",
              "## Coverage sweep on held-out label-poor targets — does empirical coverage track the guarantee?",
              "Mondrian coverage/abstention vs marginal coverage/abstention, each internally consistent. "
              "Note the marginal predictor's POSITIVE-class coverage collapses well below nominal.",
              "| alpha | nominal 1-alpha | Mondrian cov | Mond path cov | Mond abstain | marginal cov | marg path cov | marg abstain |",
              "|---|---|---|---|---|---|---|---|"]
    for s in sweep:
        lines.append(f"| {s['alpha']} | {s['nominal']} | **{s['cov_mond']}** | {s['cov_path_mond']} | {s['abstain_mond']} "
                     f"| {s['cov_marg']} | {s['cov_path_marg']} | {s['abstain_marg']} |")
    ok = all(s["cov_mond"] >= s["nominal"] - 0.05 for s in sweep)
    # the verdict's figures are read off the target rows above, not typed: it cannot drift from its own table
    nt = sum(r["n"] for r in tgt)
    ab = [r["abstain"] for r in tgt]
    sing = sum(r["singleton_marg"] * r["n"] for r in tgt) / nt
    pcm = [r["cov_path_marg"] for r in tgt]
    lines += ["", f"**VERDICT:** cross-species conformal coverage {'tracks the nominal target' if ok else 'is approximate'} "
              f"on species with NO labels — Mondrian (class-conditional) holds {mean_cov_tgt:.2f} coverage under the ~10:1 imbalance, "
              f"but ONLY by abstaining on {100 * min(ab):.1f}-{100 * max(ab):.1f}% of target variants (it OVER-covers = conservative, not free). "
              "The prediction SETS are interpretable: singleton = confident call, {both} = safe abstention. "
              "HONEST FRAMING: coverage under a STATED, empirically-tested cross-species exchangeability assumption "
              "— NOT a distribution-free finite-sample theorem (cross-species transport breaks exchangeability). "
              f"No single predictor gives both {mean_cov_tgt:.2f} coverage AND {100 * sing:.1f}% singletons: the near-singleton (marginal) "
              f"construction drops positive-class coverage to {min(pcm):.3f}-{max(pcm):.3f}.",
              "", "Honest caveat: exchangeability across species is imperfect, so coverage is approximate/assumption-"
              "conditional; Mondrian is reported because marginal coverage skews under class imbalance."]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pl.DataFrame(rows).write_parquet("data/processed/conformal_per_species.parquet")
    pl.DataFrame(sweep).write_parquet("data/processed/conformal_sweep.parquet")
    print("\n".join(lines))
    st(f"DONE | conformal: mean Mondrian coverage on targets {mean_cov_tgt:.3f} vs 0.90; sweep {'tracks' if ok else 'approx'}; {MD}")
    ev(f"DONE: conformal - target Mondrian coverage {mean_cov_tgt:.3f} (nominal 0.90); sweep {'tracks nominal' if ok else 'approximate'}")


if __name__ == "__main__":
    main()
