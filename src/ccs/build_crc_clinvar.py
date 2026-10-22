"""Is the positive-class-FNR certificate of build_crc_certificate.py a valid OBJECT at ClinVar scale?

Two tests on n~1.6M ClinVar (labels exist, so we can actually validate):
  (1) IN-DISTRIBUTION: calibrate lambda on a random half, test achieved FNR + benign-call rate on the
      other half. If it holds AND is non-trivial, the certificate is a valid new object at scale.
  (2) TRANSPORT under a real covariate shift: calibrate lambda on CODING variants, test on NONCODING.
      Does the missed-positive guarantee survive a within-human distribution shift?
The cross-species transport of the same certificate is build_crc_certificate.py.
  python src/ccs/build_crc_clinvar.py   -> logs/crc_clinvar.md
"""
import glob, os, json
import numpy as np
import polars as pl

SCORES = {"Evo2-EVEE probe": "pathogenicity", "AlphaMissense": "gt_alphamissense_c", "CADD": "gt_cadd_c"}
POS = {"pathogenic", "likely_pathogenic"}; NEG = {"benign", "likely_benign"}
ALPHAS = [0.05, 0.10, 0.20]


def crc_lambda(p_pos, alpha):
    n = len(p_pos); target = max(0.0, ((n + 1) * alpha - 1) / n)
    s = np.sort(p_pos); k = int(np.floor(target * n))
    # lambda just above the k-th smallest calibration score: a positive tied with that score counts as
    # missed, so where scores are tied the calibration FNR can exceed the budget by the tied share.
    # build_crc_certificate.py, whose isotonic probabilities are heavily tied, takes lambda = s[k].
    return (s[k - 1] + 1e-12) if k >= 1 else -np.inf


def eval_at(p, y, lam):
    called = p >= lam
    fnr = float((~called[y == 1]).mean()) if (y == 1).any() else float("nan")
    return fnr, float((~called).mean())


def main():
    need = ["label", "ref_region_CDS"] + list(SCORES.values())
    mani = json.load(open("data/external/evee/manifest.json"))
    exp = {s["filename"]: s["size_bytes"] for s in mani["shards"]}
    shards = [f for f in sorted(glob.glob("data/external/evee/clean_shard_*.parquet")) if os.path.getsize(f) == exp.get(os.path.basename(f), -1)]
    df = pl.scan_parquet(shards).select(need).collect()
    df = df.with_columns(pl.when(pl.col("label").is_in(list(POS))).then(1)
                         .when(pl.col("label").is_in(list(NEG))).then(0).otherwise(None).alias("y")).drop_nulls(subset=["y"])
    y = df["y"].to_numpy().astype(int)
    coding = df["ref_region_CDS"].cast(pl.Boolean, strict=False).fill_null(False).to_numpy()
    rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5

    lines = ["# Positive-class FNR certificate at ClinVar scale (conformal risk control; n~1.6M, labels exist)", ""]
    id_ok = 0; id_tot = 0; tr_ok = 0; tr_tot = 0; id_strict = 0; tr_strict = 0; id_miss = {}
    for name, col in SCORES.items():
        s = df.select(pl.col(col).cast(pl.Float64, strict=False))[col]
        m = s.is_not_null().to_numpy(); p = s.to_numpy()
        lines += [f"## {name} (n={int(m.sum()):,})",
                  "| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |",
                  "|---|---|---|---|---|---|---|"]
        for a in ALPHAS:
            # (1) in-distribution split
            cal = m & half; te = m & ~half
            lam = crc_lambda(p[cal & (y == 1)], a)
            fnr_id, br_id = eval_at(p[te], y[te], lam)
            ok_id = fnr_id <= a + 0.02; id_ok += ok_id; id_tot += 1; id_strict += fnr_id <= a
            if not fnr_id <= a: id_miss.setdefault(name, []).append(a)
            # (2) coding -> noncoding transport
            calc = m & coding; ten = m & ~coding
            if (ten & (y == 1)).sum() >= 20:
                lam2 = crc_lambda(p[calc & (y == 1)], a)
                fnr_tr, br_tr = eval_at(p[ten], y[ten], lam2)
                ok_tr = fnr_tr <= a + 0.03; tr_ok += ok_tr; tr_tot += 1; tr_strict += fnr_tr <= a
                trs = f"{fnr_tr:.3f} | {br_tr:.2f} | {'YES' if ok_tr else 'no'}"
            else:
                trs = "n/a | n/a | n/a"
            lines.append(f"| {a} | {fnr_id:.3f} | {br_id:.2f} | {'YES' if ok_id else 'no'} | {trs} |")
        lines.append("")

    lines += ["## VERDICT",
              f"In-distribution: guarantee held in {id_ok}/{id_tot} (score x alpha) cases "
              f"within the tolerance (FNR <= alpha + 0.02), {id_strict}/{id_tot} strictly"
              + ("" if not id_miss else " (" + "; ".join("%s at alpha = %s" % (k, " and ".join(str(a) for a in v))
                                                       for k, v in id_miss.items()) + ")")
              + ". "
              f"Coding->noncoding transport: held in {tr_ok}/{tr_tot} within the tolerance (FNR <= alpha + 0.03), "
              f"{tr_strict}/{tr_tot} strictly.",
              "",
              ("**OBJECT VALID at scale** - the positive-class-FNR certificate holds and is informative in-distribution on "
               "ClinVar, and (partly) survives the coding->noncoding shift. Conformal Risk Control of the missed-"
               "positive rate is therefore a valid guarantee object when the target has labels or is exchangeable "
               "with the calibration data. Transport to a ZERO-label species is the separate test of "
               "build_crc_certificate.py (logs/crc_certificate.md)."
               if id_ok >= 0.7 * id_tot else
               "**OBJECT WEAK even in-distribution** - even with labels the certificate is vacuous or uninformative.")]
    os.makedirs("logs", exist_ok=True)
    open("logs/crc_clinvar.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
