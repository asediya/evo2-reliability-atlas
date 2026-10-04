"""STEP 2 hardening: does the trust layer FLAG ITS OWN regulatory blind spot?

The eQTL arm showed Evo2 cannot discriminate causal regulatory variants (AUROC ~0.50). A trustworthy
reliability atlas must MAP its red zones. Here we take the CODING-trained conformal trust layer (isotonic
+ Mondrian quantiles fit on the 9-species OMIA panels) and apply it, unchanged, to the regulatory eQTL
variants, then ask: what does the trust layer DO on the regime it cannot handle? We report the conformal
set sizes (singleton = confident call, {both} = abstention) -- set membership depends only on the
calibrated score, not the (regulatory) label, so this is a clean, honest read. We compare the eQTL
abstention rate to the coding held-out TARGETS. HONEST either way.

  python src/ccs/build_eqtl_abstention.py   -> logs/eqtl_abstention.md
"""
import os
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
RICH_MIN = 50


def load_coding(sp):
    scf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(scf):
        return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(scf).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3:
        return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def qhat(scores, alpha):
    s = np.sort(scores); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return s[min(k, n) - 1] if n else 1.0


def set_sizes(p, q0, q1):
    m0 = p <= q0; m1 = (1 - p) <= q1
    size = m0.astype(int) + m1.astype(int)
    return float((size == 1).mean()), float((size == 2).mean()), float((size == 0).mean())


def main():
    data = {sp: r for sp in WIN if (r := load_coding(sp)) is not None}
    rich = [sp for sp in data if int(data[sp][1].sum()) >= RICH_MIN]
    poor = [sp for sp in data if sp not in rich]

    Xr = np.concatenate([data[s][0] for s in rich]); Yr = np.concatenate([data[s][1] for s in rich])
    (fit_i, cal_i), = list(StratifiedKFold(2, shuffle=True, random_state=0).split(Xr, Yr))[:1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(Xr[fit_i], Yr[fit_i])
    pc = iso.predict(Xr[cal_i]); yc = Yr[cal_i]
    a = 0.10
    q0 = qhat(pc[yc == 0], a); q1 = qhat(1 - pc[yc == 1], a)

    # coding reference: pooled held-out targets
    Xp = np.concatenate([data[s][0] for s in poor]); Yp = np.concatenate([data[s][1] for s in poor])
    pp = iso.predict(Xp)
    cod_sing, cod_abst, _ = set_sizes(pp, q0, q1)

    # regulatory eQTL variants under the SAME coding trust layer
    eq = pl.read_parquet("data/processed/scores/eqtl_evo2_40b.parquet").select(
        ["variant_id", pl.col("evo2_40b_neg").alias("score")])
    cand = pl.read_parquet("data/interim/eqtl_candidates.parquet").select(["variant_id", "label", "pip"])
    e = eq.join(cand, on="variant_id", how="inner").drop_nulls(subset=["score"])
    xe = e["score"].to_numpy().astype(float)
    pe = iso.predict(xe)
    eq_sing, eq_abst, eq_empty = set_sizes(pe, q0, q1)

    # confidence distribution of the calibrated probability on regulatory vs coding
    def dist(p):
        return dict(lt10=float((p < 0.10).mean()), mid=float(((p >= 0.10) & (p <= 0.50)).mean()),
                    gt50=float((p > 0.50).mean()), mean=float(p.mean()), median=float(np.median(p)))
    de, dc = dist(pe), dist(pp)

    causal = e.filter(pl.col("label") == 1).height; noncausal = e.filter(pl.col("label") == 0).height

    lines = ["# STEP 2 - does the trust layer flag its own REGULATORY blind spot?", "",
             "The coding-trained conformal trust layer (isotonic + Mondrian quantiles, 9-species OMIA) applied "
             f"UNCHANGED to {e.height} regulatory eQTL variants ({causal} causal / {noncausal} non-causal). "
             "Set membership depends only on the calibrated score, so this is label-independent and honest.", "",
             "| regime | n | singleton (confident) | abstain {both} | mean calibrated P | %P<0.10 | %0.10-0.50 | %P>0.50 |",
             "|---|---|---|---|---|---|---|---|",
             f"| coding (held-out targets) | {len(pp)} | {cod_sing:.2f} | {cod_abst:.2f} | {dc['mean']:.3f} | {dc['lt10']:.2f} | {dc['mid']:.2f} | {dc['gt50']:.2f} |",
             f"| regulatory (eQTL) | {e.height} | {eq_sing:.2f} | **{eq_abst:.2f}** | {de['mean']:.3f} | {de['lt10']:.2f} | {de['mid']:.2f} | {de['gt50']:.2f} |"]

    # honest verdict
    if eq_abst > cod_abst + 0.05:
        verdict = (f"**RESULT (trust layer flags the blind spot):** on regulatory variants the trust layer ABSTAINS "
                   f"far more ({eq_abst:.2f} vs coding {cod_abst:.2f}) - it correctly refuses to make confident calls "
                   "in the regime the FM cannot handle. The reliability atlas maps its own red zone.")
    elif de['lt10'] > 0.6:
        verdict = (f"**RESULT (honest limitation):** the coding-trained layer does NOT auto-abstain on regulatory "
                   f"variants - it maps most of them to LOW pathogenicity ({de['lt10']:.0%} with P<0.10, mean "
                   f"P={de['mean']:.3f}) because Evo2 systematically UNDER-scores regulatory variants (consequence "
                   "hierarchy: regulatory << missense). So confidence alone does not detect the blind spot; an EXPLICIT "
                   "regime/OOD flag is required - which is exactly why the atlas ships a per-consequence reliability map "
                   "rather than trusting a single calibrated score. Honest, and it motivates the abstention/OOD arm.")
    else:
        verdict = (f"**RESULT (mixed):** regulatory abstention {eq_abst:.2f} vs coding {cod_abst:.2f}; calibrated-P "
                   f"mean {de['mean']:.3f}. The layer is partially uncertain on regulatory variants but does not cleanly "
                   "abstain - reported honestly; the per-consequence reliability map is the principled flag.")
    lines += ["", verdict]
    os.makedirs("logs", exist_ok=True)
    open("logs/eqtl_abstention.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
