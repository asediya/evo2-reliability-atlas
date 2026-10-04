"""MECHANISM: is Evo2 a learned evolutionary-constraint detector -- and is that WHY it is blind to causal
regulatory variants? Four tests, all on data in hand (no Evo2 compute, no embeddings):

  A. Can CONSERVATION itself find causal eQTLs? GERP AUROC on causal(PIP>=.9) vs LD-matched control(PIP<=.001).
     If GERP is also at chance -> no constraint-based method can solve this task.
  B. Are causal eQTLs conserved AT ALL? mean GERP causal vs control + Mann-Whitney.
  C. Is Evo2 a conservation proxy? Spearman(evo2, GERP) on coding (atlas) vs regulatory (eQTL).
  D. Dose-response: does Evo2 work on the LARGEST-effect / highest-PIP eQTLs (partial competence)?

If A+B are null and C is high on coding -> mechanism: "Evo2 learned constraint; causal common eQTLs carry no
constraint signal; the blind spot is the PREDICTED consequence of genome-only training, not a bug."

  python src/ccs/mechanism_constraint.py
"""
import os, sys
import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BW = "data/raw/conservation/pig/pig_gerp.Sscrofa11.1.bw"


def gerp_lookup(chroms, poss):
    import pybigtools
    bw = pybigtools.open(BW)
    have = set(bw.chroms().keys())
    out = np.full(len(poss), np.nan)
    for i, (c, p) in enumerate(zip(chroms, poss)):
        c = str(c)
        if c not in have:
            continue
        try:
            v = bw.values(c, int(p) - 1, int(p))
            if v is not None and len(v):
                out[i] = v[0]
        except Exception:
            pass
    return out


def main():
    s = pl.read_parquet("data/interim/ablation/eqtl_abl_sample.parquet")
    y = s["label"].to_numpy().astype(int)
    print(f"eQTL panel: n={len(y)} (causal {y.sum()}, control {(1-y).sum()})", flush=True)

    print("\n--- extracting GERP for eQTL variants ---", flush=True)
    g = gerp_lookup(s["chrom"].to_list(), s["pos"].to_list())
    ok = np.isfinite(g)
    print(f"  GERP available for {ok.sum()}/{len(g)} variants", flush=True)
    yg, gg = y[ok], g[ok]

    print("\n=== A. Can CONSERVATION find causal eQTLs? ===", flush=True)
    au = roc_auc_score(yg, gg) if len(set(yg)) > 1 else float("nan")
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(yg), len(yg))
        if len(set(yg[i])) > 1:
            boots.append(roc_auc_score(yg[i], gg[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  GERP AUROC on causal-vs-control eQTL: {au:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    print(f"  (Evo2-40B on the same task: ~0.498; NT: 0.486)", flush=True)

    print("\n=== B. Are causal eQTLs conserved at all? ===", flush=True)
    a, b = gg[yg == 1], gg[yg == 0]
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    print(f"  mean GERP  causal {a.mean():+.3f} (median {np.median(a):+.3f})", flush=True)
    print(f"  mean GERP control {b.mean():+.3f} (median {np.median(b):+.3f})", flush=True)
    print(f"  Mann-Whitney p = {p:.3g}  -> {'NO conservation difference' if p > 0.05 else 'difference exists'}", flush=True)

    print("\n=== C. Is Evo2 a conservation proxy? Spearman(evo2, GERP) ===", flush=True)
    ef = "data/processed/scores_cloud/eqtl_abl_8192_ll.parquet"
    if os.path.exists(ef):
        j = pl.read_parquet(ef).join(s.select(["variant_id", "chrom", "pos"]), on="variant_id", how="inner")
        ev = -j["evo2_meanll_delta"].to_numpy()
        gj = gerp_lookup(j["chrom"].to_list(), j["pos"].to_list())
        m = np.isfinite(ev) & np.isfinite(gj)
        r, pv = spearmanr(ev[m], gj[m])
        print(f"  REGULATORY (eQTL): rho = {r:+.3f} (p={pv:.2g}, n={m.sum()})", flush=True)
    # coding: atlas species with GERP
    rs = []
    for sp in ["cattle", "dog", "human", "cat", "horse", "sheep", "chicken", "goat", "pig"]:
        f8 = f"data/processed/scores_cloud/atlas8192_{sp}_meanll_8192.parquet"
        gf = f"data/processed/conservation/{'dog_cf3' if sp == 'dog' else 'cattle_ensvar' if sp == 'cattle' else sp}_gerp.parquet"
        if not (os.path.exists(f8) and os.path.exists(gf)):
            continue
        j = pl.read_parquet(f8).join(pl.read_parquet(gf), on="variant_id", how="inner")
        if j.height < 30:
            continue
        ev = -j["evo2_meanll_delta"].to_numpy(); gv = j["gerp"].to_numpy()
        m = np.isfinite(ev) & np.isfinite(gv)
        if m.sum() < 30:
            continue
        r, _ = spearmanr(ev[m], gv[m])
        rs.append((sp, r, int(m.sum())))
        print(f"  CODING {sp:8s}: rho = {r:+.3f} (n={m.sum()})", flush=True)
    if rs:
        print(f"  CODING mean rho = {np.mean([r for _, r, _ in rs]):+.3f}", flush=True)

    print("\n=== D. Dose-response: does Evo2 work on the strongest eQTLs? ===", flush=True)
    if os.path.exists(ef):
        j = pl.read_parquet(ef).join(s, on="variant_id", how="inner")
        ev = -j["evo2_meanll_delta"].to_numpy(); yy = j["label"].to_numpy().astype(int)
        az = j["absz"].to_numpy() if "absz" in j.columns else None
        m = np.isfinite(ev)
        if az is not None:
            qs = np.nanpercentile(az[m], [50, 75, 90])
            for lab, thr in [("|z| top 50%", qs[0]), ("|z| top 25%", qs[1]), ("|z| top 10%", qs[2])]:
                sel = m & (az >= thr)
                if sel.sum() > 40 and len(set(yy[sel])) > 1:
                    print(f"  {lab:12s} n={sel.sum():4d}  Evo2 AUROC {roc_auc_score(yy[sel], ev[sel]):.3f}", flush=True)
        print(f"  ALL          n={m.sum():4d}  Evo2 AUROC {roc_auc_score(yy[m], ev[m]):.3f}", flush=True)


if __name__ == "__main__":
    main()
