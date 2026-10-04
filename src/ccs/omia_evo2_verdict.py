import sys, os
import polars as pl
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate, paired_delta

panel = pl.read_parquet(sys.argv[1])
ev = pl.read_parquet(sys.argv[2])
ph = pl.read_parquet(sys.argv[3])
df = panel.join(ev, on="variant_id", how="inner").join(ph, on="variant_id", how="inner")
df = df.with_columns(pl.col("phylop").abs().alias("abs_phylop"))
print(f"=== CATTLE OMIA: Evo2 vs conservation ===")
print(f"joined {df.height}  (pos={df.filter(pl.col('label')==1).height} neg={df.filter(pl.col('label')==0).height})\n")
evaluate(df, "label", ["evo2_neg", "evo2_delta", "phylop", "abs_phylop"])
y = df["label"].to_numpy().astype(int)
print()
for fm in ["evo2_neg"]:
    for base in ["phylop"]:
        d = paired_delta(y, df[fm].to_numpy().astype(float), df[base].to_numpy().astype(float))
        print(f"  dAUROC({fm} - {base}) = {d['delta_auroc']:+.3f}  "
              f"95%CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  P(Evo2>cons)={d['p_gt0']:.3f}")
