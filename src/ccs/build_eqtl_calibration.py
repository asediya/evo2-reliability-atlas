"""PLAN #2 — eQTL as CALIBRATION (regulatory reliability gap): does the calibration/reliability of Evo2
transfer from CODING disease variants (OMIA) to REGULATORY variants (fine-mapped causal cis-eQTLs), or is
there a regulatory reliability gap the trust-layer must flag? This is a NEW axis for the crown jewel —
calibration transfer across VARIANT TYPE (coding->regulatory), distinct from the cross-SPECIES axis and
distinct from TraitGym/LOL-EVE's discrimination benchmark.

We fit the isotonic calibration map on pooled OMIA CODING variants, transfer it to the eQTL panel (causal
vs non-causal cis-eQTLs), and measure ECE. Compared to an eQTL-oracle (in-panel CV) and no-calibration.
Also report discrimination (AUROC) for context. Either outcome is publishable + on-thesis:
  - transfers (low ECE)  -> reliability generalises across variant types = a robust trust layer.
  - gap (high ECE / AUROC~0.5) -> a REGULATORY reliability blind spot the calibrated trust-layer flags.

Waits for the eQTL scores (bench Stage D). CPU-only.
  python src/ccs/build_eqtl_calibration.py
"""
import os, sys, time
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
SC = "data/processed/scores/eqtl_evo2_40b.parquet"
CAND = "data/interim/eqtl_candidates.parquet"
EV = "logs/status/events.log"; ST = "logs/status/eqtl_calib.status"; MD = "logs/eqtl_calibration.md"


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] eqtl-calib: {m}\n")
def st(m):
    with open(ST, "w", encoding="utf-8") as f: f.write(m + "\n")


def load_omia(sp):
    scf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(scf): return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(scf).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3: return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def ece(p, y, bins=10, adaptive=False):
    n = len(y)
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1))); edges[0] = 0.0; edges[-1] = 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if m.sum(): e += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)


def oracle(x, y):
    if int(y.sum()) < 8: return None
    p = np.zeros(len(x))
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(x, y):
        p[te] = IsotonicRegression(out_of_bounds="clip").fit(x[tr], y[tr]).predict(x[te])
    return p


def main():
    if not os.path.exists(SC):
        st("WAIT | eQTL scores not written yet (bench Stage D)"); return
    cand = pl.read_parquet(CAND).select(["variant_id", "label"])
    sdf = pl.read_parquet(SC).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = cand.join(sdf, on="variant_id", how="inner").drop_nulls()
    if d.height < 19000:        # wait for the full 20k panel (bench Stage D) before the final analysis
        st(f"WAIT | eQTL {d.height}/20000 scored"); return
    xq = d["score"].to_numpy().astype(float); yq = d["label"].to_numpy().astype(int)

    # discrimination (context; the TraitGym-claimed part) — signed and magnitude
    auroc_signed = roc_auc_score(yq, xq)
    auroc_mag = roc_auc_score(yq, np.abs(xq))
    # use whichever orients causal-high as the calibration score
    xq_cal = xq if auroc_signed >= 0.5 else -xq

    # OMIA coding calibration -> transfer to eQTL regulatory
    omia = {sp: r for sp in WIN if (r := load_omia(sp)) is not None}
    Xo = np.concatenate([omia[s][0] for s in omia]); Yo = np.concatenate([omia[s][1] for s in omia])
    iso_coding = IsotonicRegression(out_of_bounds="clip").fit(Xo, Yo)
    p_transfer = iso_coding.predict(xq_cal if auroc_signed >= 0.5 else xq)  # coding map on eQTL scores
    p_none = (xq_cal - xq_cal.min()) / (xq_cal.max() - xq_cal.min() + 1e-9)
    p_oracle = oracle(xq_cal, yq)

    e_tr = ece(p_transfer, yq, adaptive=True); e_no = ece(p_none, yq, adaptive=True)
    e_or = ece(p_oracle, yq, adaptive=True) if p_oracle is not None else float("nan")

    npos = int(yq.sum()); nneg = int(len(yq) - yq.sum())
    disc = "discriminates" if max(auroc_signed, auroc_mag) >= 0.55 else "does NOT discriminate (~chance)"
    if max(auroc_signed, auroc_mag) < 0.55:
        verdict = "REGULATORY BLIND SPOT"
        interp = ("Evo2 does not distinguish causal cis-eQTLs from LD-matched non-causal (~chance), so it is "
                  "uninformative for regulatory causality — a blind spot the calibrated trust-layer must flag "
                  "(abstain / maximum uncertainty on regulatory variants). Honest negative, on-thesis.")
    elif e_tr <= e_or * 1.5 or e_tr <= 0.06:
        verdict = "CALIBRATION TRANSFERS coding->regulatory"
        interp = ("The coding-learned calibration is well-calibrated on regulatory variants -> Evo2's reliability "
                  "generalises across variant TYPES, not just species. A robust, portable trust layer.")
    else:
        verdict = "REGULATORY RECALIBRATION NEEDED"
        interp = ("Evo2 discriminates causal eQTLs but the coding calibration is MIScalibrated on them "
                  "(transfer ECE >> oracle) -> regulatory variants need their own calibration; the coding map "
                  "over/under-trusts them. The trust-layer must recalibrate per variant type.")

    lines = ["# PLAN #2 — eQTL as CALIBRATION: does reliability transfer coding->regulatory? (regulatory reliability gap)", "",
             f"Panel: {npos} causal (PIP>=0.9) + {nneg} non-causal LD-partner cis-eQTLs (pig, Sscrofa11.1).", "",
             "## Discrimination (context — the TraitGym/LOL-EVE-claimed axis)",
             f"- AUROC (signed LLR): **{auroc_signed:.3f}** | AUROC (|LLR|): **{auroc_mag:.3f}** -> Evo2 {disc}.", "",
             "## Calibration transfer CODING -> REGULATORY (the novel axis)",
             "Coding calibration = isotonic fit on pooled OMIA disease variants, applied to the eQTL panel. Adaptive-ECE:",
             f"- no-calibration: **{e_no:.3f}**",
             f"- coding->regulatory TRANSFER: **{e_tr:.3f}**",
             f"- eQTL oracle (in-panel CV): **{e_or:.3f}**", "",
             f"**VERDICT: {verdict}** — {interp}"]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pl.DataFrame([dict(auroc_signed=round(auroc_signed, 3), auroc_mag=round(auroc_mag, 3),
                       ece_none=round(e_no, 4), ece_transfer=round(e_tr, 4), ece_oracle=round(e_or, 4),
                       npos=npos, nneg=nneg, verdict=verdict)]).write_parquet("data/processed/eqtl_calibration.parquet")
    print("\n".join(lines))
    st(f"DONE | eQTL-calib {verdict}: AUROC {max(auroc_signed,auroc_mag):.3f}; coding->reg ECE {e_tr:.3f} (oracle {e_or:.3f}, none {e_no:.3f}); {MD}")
    ev(f"DONE: eQTL calibration {verdict} - AUROC {max(auroc_signed,auroc_mag):.3f}, transfer ECE {e_tr:.3f} vs oracle {e_or:.3f}")


if __name__ == "__main__":
    main()
