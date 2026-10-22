"""A TRANSPORTED positive-class false-negative-rate certificate (conformal risk control).

Changes the guarantee OBJECT from marginal coverage to a clinically meaningful one: control
E[FNR on positives] <= alpha (bound the missed-positive rate), calibrate the decision
threshold on LABEL-RICH species via Conformal Risk Control, then TRANSPORT it to a species that
contributed ZERO labels and check (a) the guarantee HOLDS on the held-out species and (b) it is
NON-TRIVIAL (still calls a useful fraction benign; not 'abstain on everything').

Decision rule: call PATHOGENIC if calibrated p >= lambda, else BENIGN. FNR = P(p<lambda | y=1) is
monotone increasing in lambda -> CRC picks the LARGEST lambda whose calibration FNR (+ finite-sample
correction) <= alpha, maximising the negative-call region subject to the missed-positive guarantee.

The certificate transports if FNR_target ~ alpha (holds) AND the benign-call fraction stays
substantial across zero-label targets. CPU-only.
  python src/ccs/build_crc_certificate.py   -> logs/crc_certificate.md
"""
import os
import sys
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

WIN = {"goat": "goat_scoring_windows", "chicken": "chicken_scoring_windows",
       "pig": "pig_scoring_windows_real", "sheep": "sheep_scoring_windows",
       "horse": "horse_scoring_windows", "cat": "cat_scoring_windows",
       "cattle": "cattle_ensvar_scoring_windows", "dog": "dog_cf3_scoring_windows",
       "human": "human_scoring_windows"}
RICH_MIN = 50
ALPHAS = [0.05, 0.10, 0.20]


def load(sp):
    scf = f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet"
    if not os.path.exists(scf): return None
    w = pl.read_parquet(f"data/interim/{WIN[sp]}.parquet").select(["variant_id", "label"])
    s = pl.read_parquet(scf).select(["variant_id", pl.col("evo2_40b_neg").alias("score")])
    d = w.join(s, on="variant_id", how="inner").drop_nulls()
    if d.height < 20 or int(d["label"].sum()) < 3: return None
    return d["score"].to_numpy().astype(float), d["label"].to_numpy().astype(int)


def crc_lambda(p_pos, alpha):
    """CRC: largest lambda s.t. calibration FNR + finite-sample correction <= alpha.
    FNR(lambda) = mean(p_pos < lambda). Conservative correction (n+1)/n * alpha - 1/n (RCPS-style)."""
    n = len(p_pos)
    target = max(0.0, ((n + 1) * alpha - 1) / n)          # finite-sample-corrected budget
    s = np.sort(p_pos)
    # With k = floor(target * n), lambda = s[k] is the largest threshold whose calibration FNR, mean(p_pos < lambda),
    # is at most k/n: positives tied with s[k] stay on the pathogenic side. (k < n always, since target < 1.)
    k = int(np.floor(target * n))
    return s[k]                                          # the (k+1)-th smallest: mean(p_pos < s[k]) <= k/n even with ties


def main():
    data = {sp: r for sp in WIN if (r := load(sp)) is not None}
    rich = [sp for sp in data if int(data[sp][1].sum()) >= RICH_MIN]
    poor = [sp for sp in data if sp not in rich]

    # LOSO isotonic calibrated probs; pooled rich for CRC calibration of lambda
    def cal_probs(target):
        train = [s for s in rich if s != target]
        Xt = np.concatenate([data[s][0] for s in train]); Yt = np.concatenate([data[s][1] for s in train])
        iso = IsotonicRegression(out_of_bounds="clip").fit(Xt, Yt)
        # calibration pathogenic probs = pooled train positives passed through the map
        pcal_pos = iso.predict(Xt[Yt == 1])
        x, y = data[target]
        return iso.predict(x), y, pcal_pos

    lines = ["# Transported positive-class FNR certificate (conformal risk control)", "",
             "Control E[missed-positive rate] <= alpha; calibrate lambda on label-rich species via Conformal Risk "
             "Control; transport to each held-out species. VALID if achieved FNR <= alpha + 0.02 (a 0.02 tolerance "
             "on the nominal level; the strict count is in the verdict); NON-TRIVIAL if it still calls a real "
             "fraction benign (benign-rate > 0.15).", ""]
    verdicts = []
    for a in ALPHAS:
        lines += [f"## alpha = {a} (guarantee: <= {a:.0%} of true positives missed)",
                  "| held-out species | role | n | pos | achieved FNR | <=alpha+0.02? | benign-call rate | informative? |",
                  "|---|---|---|---|---|---|---|---|"]
        for sp in data:
            p, y, pcal_pos = cal_probs(sp)
            lam = crc_lambda(pcal_pos, a)
            called_path = p >= lam
            fnr = float((~called_path[y == 1]).mean()) if (y == 1).any() else float("nan")
            benign_rate = float((~called_path).mean())
            ok = fnr <= a + 0.02
            info = benign_rate > 0.15
            role = "source" if sp in rich else "TARGET"
            lines.append(f"| {sp} | {role} | {len(y)} | {int(y.sum())} | {fnr:.3f} | {'YES' if ok else 'no'} "
                         f"| {benign_rate:.2f} | {'yes' if info else 'no'} |")
            if role == "TARGET":
                verdicts.append((a, sp, ok, info, fnr, benign_rate, fnr <= a))
        lines.append("")

    # gate verdict on the ZERO-LABEL targets
    tgt = [v for v in verdicts]
    holds = sum(1 for _, _, ok, _, _, _, _ in tgt if ok)
    strict = [(a, sp) for a, sp, _, _, _, _, s in tgt if s]
    nontrivial = sum(1 for _, _, ok, info, _, _, _ in tgt if ok and info)
    if not tgt:
        # Both comparisons reduce to 0 >= 0.0 on an empty target set, so this would print "PASS"
        # and exit 0 having evaluated nothing -- the opposite of the deposited note's verdict.
        # Exit 3 is this archive's "stopped at an undeposited path" convention.
        print("  RESULT: NO DATA -- 0 (alpha x zero-label-target) cases were evaluated, because")
        print("  the window panels this certificate reads are not part of the code deposit (see")
        print("  reports/DATA_MANIFEST.md). Nothing was examined, so this is NOT a pass.")
        miss = [f for sp in WIN for f in (f"data/processed/scores/{sp}_evo2_40b_local_scores.parquet",
                                          f"data/interim/{WIN[sp]}.parquet") if not os.path.exists(f)]
        if miss:
            print("  Missing inputs (%d):" % len(miss))
            print("".join("    %s\n" % f for f in miss), end="")
        return 3
    passed = holds >= 0.7 * len(tgt) and nontrivial >= 0.5 * len(tgt)
    lines += ["## VERDICT",
              f"Across {len(tgt)} (alpha x zero-label-target) cases: guarantee HELD in {holds}/{len(tgt)} within the "
              f"tolerance ({len(strict)}/{len(tgt)} strictly"
              + "".join(", %s at alpha = %s" % (sp, a) for a, sp in strict) + "), "
              f"and HELD while NON-TRIVIAL (benign-rate>15%) in {nontrivial}/{len(tgt)}.",
              "",
              ("**PASS** - the transported positive-class-FNR certificate holds and is non-trivial on zero-label "
               "species."
               if passed else
               "**FAIL/WEAK** - the certificate is vacuous or does not transport non-trivially on OMIA (small n / "
               "weak separation).")]
    os.makedirs("logs", exist_ok=True)
    open("logs/crc_certificate.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
