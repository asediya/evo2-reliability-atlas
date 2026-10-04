# -*- coding: utf-8 -*-
"""FP8 batch-size determinism control: does batch composition move a score, class-dependently?

WHY THIS FILE EXISTS. reports/ablation_fp8.json is cited in the Methods as the control underwriting
the reproducibility of the whole GPU scoring layer, and a check found
that NO script in the deposit produced it: grep across src/, tools/ and glmtrust/ returned zero
matches for the filename and zero for the check's own logic. Every other published number traces to
a builder. This is that builder, so the computation is inspectable and re-runnable rather than
taken on trust.

WHAT IT RUNS ON. The two runs the deposited artefact compared -- the same 600 variants at batch 4
and at batch 32 -- are deposited per variant in reports/ablation_fp8.parquet, with their labels in
reports/_recon_pervariant_trust.parquet, so this script regenerates the deposited JSON from
Additional file 2 alone. Two score files may also be passed explicitly, for a re-run at other batch
sizes.

    python src/ccs/build_ablation_fp8.py --a scores_batch4.parquet --b scores_batch32.parquet
"""
import argparse
import io
import json
import os
import sys

OUT = "reports/ablation_fp8.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="scores at the first batch size (parquet: variant_id, label, score)")
    ap.add_argument("--b", help="scores at the second batch size, same variants")
    ap.add_argument("--batch-sizes", nargs=2, type=int, default=[4, 32])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import numpy as np
    import polars as pl
    from sklearn.metrics import roc_auc_score

    # This script used to refuse to run, on the grounds that "the two runs
    # were not retained, so this control is the one published number the archive cannot
    # regenerate", and the Methods repeated that claim. Both were wrong: the per-variant scores at
    # BOTH batch sizes are deposited in reports/ablation_fp8.parquet (variant_id, delta_b4,
    # delta_b32) and the labels join from reports/_recon_pervariant_trust.parquet across all 600
    # variants. The paper was understating its own reproducibility. This control now regenerates
    # from Additional file 2 alone, with the two explicit score files still accepted.
    DEPOSIT, LABELS = "reports/ablation_fp8.parquet", "reports/_recon_pervariant_trust.parquet"

    if args.a and args.b and os.path.exists(args.a) and os.path.exists(args.b):
        a = pl.read_parquet(args.a).select(["variant_id", "label", "score"])
        b = pl.read_parquet(args.b).select(["variant_id", "score"])
        j = a.join(b, on="variant_id", how="inner", suffix="_b")
        if not j.height:
            sys.exit("build_ablation_fp8: the two files share no variant_id")
        sa = j["score"].to_numpy().astype(float)
        sb = j["score_b"].to_numpy().astype(float)
        y = j["label"].to_numpy().astype(int)
    elif os.path.exists(DEPOSIT) and os.path.exists(LABELS):
        dep = pl.read_parquet(DEPOSIT).select(["variant_id", "delta_b4", "delta_b32"])
        lab = pl.read_parquet(LABELS).select(["variant_id", "label"]).unique(subset=["variant_id"])
        j = dep.join(lab, on="variant_id", how="inner")
        if j.height != dep.height:
            sys.exit("build_ablation_fp8: %d of %d deposited variants carry no label"
                     % (dep.height - j.height, dep.height))
        sa = j["delta_b4"].to_numpy().astype(float)
        sb = j["delta_b32"].to_numpy().astype(float)
        y = j["label"].to_numpy().astype(int)
        print("regenerated from the deposit: %s + labels from %s" % (DEPOSIT, LABELS))
    else:
        sys.stderr.write(
            "build_ablation_fp8: no input. Expected %s and %s in the deposit, or two score\n"
            "files passed explicitly:\n"
            "    python src/ccs/build_ablation_fp8.py --a batch4.parquet --b batch32.parquet\n"
            % (DEPOSIT, LABELS))
        return 3

    d = sa - sb

    # The question is not whether scores move, but whether they move DIFFERENTLY by class: a
    # class-correlated offset would inflate AUROC directly, whereas a common offset cannot.
    class_effect = (float(d[y == 1].mean() - d[y == 0].mean())
                    if (y == 1).any() and (y == 0).any() else 0.0)
    n_bit = int((sa == sb).sum())
    payload = {
        "n": int(j.height),
        "n_pos": int((y == 1).sum()),
        "batch_sizes": list(args.batch_sizes),
        "max_abs_diff": float(np.abs(d).max()),
        "bit_identical": n_bit,
        "frac_bit_identical": n_bit / float(j.height),
        "class_diff_in_batch_effect": class_effect,
        "auroc_diff": (float(roc_auc_score(y, sa) - roc_auc_score(y, sb))
                       if 0 < y.sum() < len(y) else 0.0),
        "verdict": ("no class-correlated FP8 batch effect" if abs(class_effect) < 1e-9
                    else "class-correlated batch effect present"),
    }
    io.open(args.out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, indent=1) + "\n")
    print("wrote %s: %d/%d bit-identical, max |diff| %.3g, class effect %.3g"
          % (args.out, n_bit, j.height, payload["max_abs_diff"], class_effect))
    return 0


if __name__ == "__main__":
    sys.exit(main())
