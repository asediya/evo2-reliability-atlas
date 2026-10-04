"""eQTL/regulatory arm result: does Evo2-40B's variant-effect signal recognise CAUSAL cis-regulatory
variants (high-PIP fine-mapped pig eQTLs) over non-causal LD partners in the same eGene windows?

Evo2 score = |log P(ref) - log P(alt)| at the variant position (evo2_40b_neg magnitude) = how strongly
the model "cares" about the base. Hypothesis: causal regulatory bases sit at more functional/constrained
positions, so |LLR| is larger. We report:
  (1) global AUROC (causal vs non-causal), signed + magnitude
  (2) within-eGene test: is the causal variant ranked above its own non-causal cis-neighbours?
      (precision@1 per gene + mean causal rank-percentile) -- the LD-controlled, fair contrast
Either outcome is publishable: signal => Evo2 reads cis-regulation; ~0.5 => a regulatory blind spot
that the calibration layer must flag (both feed the paper). Runs after eQTL scoring.
  python src/ccs/build_eqtl_analysis.py
"""
import os, time
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SC = "data/processed/scores/eqtl_evo2_40b.parquet"
CAND = "data/interim/eqtl_candidates.parquet"
EV = "logs/status/events.log"; ST = "logs/status/eqtl.status"
MD = "logs/eqtl_regulatory.md"


def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] eqtl: {m}\n")
def st(m):
    with open(ST, "w") as f: f.write(m + "\n")


def main():
    if not os.path.exists(SC):
        st("WAIT | eQTL scores not written yet"); return
    cand = pl.read_parquet(CAND).select(["variant_id", "gene_id", "pip", "absz", "label"])
    s = pl.read_parquet(SC).select(["variant_id", "evo2_40b_neg"])
    d = cand.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 200:
        st(f"WAIT | only {d.height} eQTL variants scored"); return

    d = d.with_columns(pl.col("evo2_40b_neg").abs().alias("mag"))
    y = d["label"].to_numpy()
    auroc_mag = roc_auc_score(y, d["mag"].to_numpy())
    auroc_sgn = roc_auc_score(y, d["evo2_40b_neg"].to_numpy())
    # association-strength baseline (|z|) — sanity ceiling; Evo2 must add signal beyond raw eQTL strength
    auroc_absz = roc_auc_score(y, d["absz"].to_numpy())

    # within-eGene: for genes with >=1 causal and >=1 non-causal, does causal top the local ranking?
    #
    # Scored on the SIGNED, causal-oriented column, matching fig3_data.py, which is the
    # source of every eQTL number in the paper. This script previously ranked on |LLR| and
    # took the MAX percentile among a gene's causal variants while printing "0.5 = chance".
    # Both were wrong: the null of a max over k causal variants is k/(k+1) (~0.78 here, at
    # ~3.5 causal per eGene), not 0.5, so the old 0.649 read as above-chance when it was not.
    # We now average over every causal variant. The null is NOT 0.5: the statistic is
    # mean(sgn < sgn[causal]), whose denominator counts every cis-variant including the causal
    # one, so a variant cannot score below itself and E[.] = (k-1)/2k < 0.5. For this panel
    # (mean k = 12.44 over the 1,430 eGenes that survive the single-class filter below, holding
    # 17,783 of the 20,000 variants -- NOT 20000/1430 = 13.99, which mismatches the full panel's
    # numerator with the kept-gene denominator) the analytic null is 0.4381, and a permutation returns
    # 0.4379 (SD 0.0048). Printing "0.5 = chance" here understated the null by 0.062 and would
    # have made a below-chance result look merely mediocre. Higher is BETTER.
    #
    # precision@1 likewise needs its own null: with ~3.5 causal of ~14 variants per eGene,
    # chance is the per-eGene causal base rate (~0.241), not 0.
    p1_hits = p1_tot = 0; rankpcts = []; baserates = []
    for _, g in d.group_by("gene_id"):
        lab = g["label"].to_numpy()
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        sgn = g["evo2_40b_neg"].to_numpy()            # signed, higher = more deleterious
        top_is_causal = lab[np.argmax(sgn)] == 1
        p1_tot += 1; p1_hits += int(top_is_causal)
        baserates.append(float(lab.mean()))
        for ci in np.where(lab == 1)[0]:
            rankpcts.append(float((sgn < sgn[ci]).mean()))
    prec1 = p1_hits / p1_tot if p1_tot else float("nan")
    base1 = float(np.mean(baserates)) if baserates else float("nan")
    mean_rankpct = float(np.mean(rankpcts)) if rankpcts else float("nan")

    npos = int(y.sum()); nneg = int(len(y) - y.sum())
    verdict = ("SIGNAL" if auroc_mag >= 0.55 else "WEAK" if auroc_mag >= 0.52 else "BLIND-SPOT")
    lines = [
        "# eQTL / regulatory arm: Evo2-40B vs fine-mapped causal pig cis-eQTLs (Sscrofa11.1)", "",
        f"- Panel: {npos} causal (PIP>=0.9) + {nneg} non-causal cis variants from the SAME eGenes (LD-controlled).",
        f"- Global AUROC (|LLR| magnitude): **{auroc_mag:.3f}**   (signed LLR: {auroc_sgn:.3f})",
        f"- Baseline AUROC (|z| eQTL-association strength): {auroc_absz:.3f}  (fine-mapping input; not causal per se)",
        f"- Within-eGene precision@1 (causal is top-scoring of its cis set): **{prec1:.3f}** "
        f"over {p1_tot} eGenes  (chance = per-eGene causal base rate {base1:.3f})",
        f"- Within-eGene mean causal rank-percentile: {mean_rankpct:.3f}  (chance = 0.438 for this "
        f"eGene-size distribution, NOT 0.5; higher is better; "
        f"averaged over every causal variant, not the best one per gene)",
        "",
        f"**{verdict}** — " + {
            "SIGNAL": "Evo2 ranks causal cis-regulatory variants above LD-matched non-causal neighbours: it reads cis-regulation, not just coding constraint.",
            "WEAK": "Evo2 shows a faint but above-chance regulatory signal; report as a partial capability the calibration layer should down-weight.",
            "BLIND-SPOT": "Evo2 does NOT distinguish causal from non-causal cis-regulatory variants (~chance). Honest negative: a regulatory blind spot the trust-layer must flag.",
        }[verdict],
    ]
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    st(f"DONE | eQTL {verdict}: AUROC(|LLR|)={auroc_mag:.3f}, within-gene P@1={prec1:.3f}; {MD}")
    ev(f"DONE: eQTL/regulatory {verdict} - AUROC(|LLR|)={auroc_mag:.3f} signed={auroc_sgn:.3f} P@1={prec1:.3f} over {p1_tot} eGenes")


if __name__ == "__main__":
    main()
