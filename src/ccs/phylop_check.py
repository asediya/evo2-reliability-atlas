"""Cheap diagnostic: does conservation (phyloP) separate positives from the NEW background
negatives? If phyloP AUROC jumps to ~0.6+, the control-set fix worked and NT re-scoring is
warranted. If it stays ~0.5, there is a deeper issue (coordinates / positive definition).
No GPU needed."""
import sys, os
import polars as pl
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate

panel = pl.read_parquet(sys.argv[1])
ph = pl.read_parquet(sys.argv[2])
df = panel.join(ph, on="variant_id", how="inner").with_columns(pl.col("phylop").abs().alias("abs_phylop"))
print(f"joined {df.height} (pos={df.filter(pl.col('label')==1).height} neg={df.filter(pl.col('label')==0).height})")
print("mean phyloP  pos vs neg:",
      round(df.filter(pl.col('label')==1)['phylop'].mean(), 3),
      round(df.filter(pl.col('label')==0)['phylop'].mean(), 3))
evaluate(df, "label", ["phylop", "abs_phylop"])
