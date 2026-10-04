"""Week-1 go/no-go: join NT-LLR + phyloP onto the labeled panel, compute AUROC/AUPRC
(overall + per-tissue) for the foundation model vs the conservation baseline, and apply
the pre-registered decision rule.

Score orientation (reported transparently, since orientation must not be chosen post-hoc):
  - nt_llr   : signed logP(alt)-logP(ref). A disruptive variant lowers P(alt) => negative.
  - nt_neg   : -nt_llr  (deleteriousness; GPN-style, higher = more disruptive)
  - nt_abs   : |nt_llr| (effect magnitude, direction-agnostic) -- PRIMARY FM score
  - phylop   : signed conservation (higher = more conserved) -- PRIMARY baseline
  - abs_phylop
"""
import argparse, os, sys
import numpy as np
import polars as pl
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate, paired_delta, auroc_auprc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--nt", required=True)
    ap.add_argument("--phylop", required=True)
    ap.add_argument("--species", default="cattle")
    a = ap.parse_args()

    panel = pl.read_parquet(a.panel)
    nt = pl.read_parquet(a.nt)
    ph = pl.read_parquet(a.phylop)
    df = panel.join(nt, on="variant_id", how="inner").join(ph, on="variant_id", how="inner")
    df = df.with_columns([
        (-pl.col("nt_llr")).alias("nt_neg"),
        pl.col("nt_llr").abs().alias("nt_abs"),
        pl.col("phylop").abs().alias("abs_phylop"),
    ])
    print(f"=== {a.species.upper()} Week-1 panel ===")
    print(f"joined rows: {df.height}  (pos={df.filter(pl.col('label')==1).height} "
          f"neg={df.filter(pl.col('label')==0).height})")

    print("\n--- OVERALL (pooled across tissues) ---")
    evaluate(df, "label", ["nt_neg", "nt_abs", "nt_llr", "phylop", "abs_phylop"])

    y = df["label"].to_numpy().astype(int)
    print("\n--- PAIRED FM vs conservation (bootstrap) ---")
    for fm in ["nt_abs", "nt_neg"]:
        for base in ["phylop", "abs_phylop"]:
            d = paired_delta(y, df[fm].to_numpy().astype(float), df[base].to_numpy().astype(float))
            print(f"  ΔAUROC({fm} - {base}) = {d['delta_auroc']:+.3f}  "
                  f"95%CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  P(FM>base)={d['p_gt0']:.3f}")

    # per-tissue AUROC for the primary scores (consistency check)
    print("\n--- per-tissue AUROC (nt_abs | phylop | n_pos) ---")
    for t, sub in sorted(df.group_by("tissue"), key=lambda x: -x[1].height):
        tt = t[0] if isinstance(t, tuple) else t
        npos = int((sub["label"] == 1).sum())
        if npos < 30:
            continue
        yy = sub["label"].to_numpy().astype(int)
        a1, _ = auroc_auprc(yy, sub["nt_abs"].to_numpy().astype(float))
        a2, _ = auroc_auprc(yy, sub["phylop"].to_numpy().astype(float))
        print(f"  {tt:18s} nt_abs={a1:.3f}  phylop={a2:.3f}  npos={npos}")

    # pre-registered verdict on the primary comparison
    d = paired_delta(y, df["nt_abs"].to_numpy().astype(float), df["phylop"].to_numpy().astype(float))
    print("\n=== VERDICT (primary: nt_abs vs phylop) ===")
    if d["delta_auroc"] >= 0.05 and d["ci"][0] > 0:
        print(f"GO / FM-swing: ΔAUROC={d['delta_auroc']:+.3f} (CI excludes 0). FM beats conservation.")
    elif d["delta_auroc"] <= -0.05 and d["ci"][1] < 0:
        print(f"CONSERVATION WINS: ΔAUROC={d['delta_auroc']:+.3f}. Lead with calibration-as-product; "
              f"conservation is the hard baseline.")
    else:
        print(f"PIVOT / calibration-as-product: ΔAUROC={d['delta_auroc']:+.3f} "
              f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] — FM ~matches conservation. "
              f"The abstention/atlas layer is the contribution; it wraps whichever scorer wins.")


if __name__ == "__main__":
    main()
