"""Summarize OMIA coverage per species: total variants, those with a genomic coordinate,
and single-base substitutions (SNVs, Evo2-scorable). Answers 'max species / max variants'."""
import sys, re
import polars as pl
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

df = pl.read_csv(sys.argv[1], infer_schema_length=0)  # all strings
g = "g. or m."
snv_pat = re.compile(r"[gm]\.\d+[ACGTNacgtn]>[ACGTacgt]")

df = df.with_columns([
    pl.col(g).fill_null("").alias("_g"),
])
df = df.with_columns([
    (pl.col("_g").str.len_chars() > 0).alias("has_coord"),
    pl.col("_g").map_elements(lambda s: bool(snv_pat.search(s)), return_dtype=pl.Boolean).alias("is_snv"),
])

by = df.group_by("Species Name").agg([
    pl.len().alias("variants"),
    pl.col("has_coord").sum().alias("with_coord"),
    pl.col("is_snv").sum().alias("snvs"),
]).sort("snvs", descending=True)

print(f"OMIA export: {df.height} total variants across {df['Species Name'].n_unique()} species\n")
print(f"{'Species':32s} {'variants':>8s} {'w/coord':>8s} {'SNVs':>6s}")
tot_v = tot_c = tot_s = 0
shown = 0
for r in by.iter_rows(named=True):
    tot_v += r["variants"]; tot_c += r["with_coord"]; tot_s += r["snvs"]
    if r["snvs"] >= 3 or shown < 25:
        print(f"{(r['Species Name'] or '?')[:32]:32s} {r['variants']:>8d} {r['with_coord']:>8d} {r['snvs']:>6d}")
        shown += 1
print(f"\n{'TOTAL (all species)':32s} {tot_v:>8d} {tot_c:>8d} {tot_s:>6d}")
# species with >= 10 SNVs (the meaningful ones)
big = by.filter(pl.col("snvs") >= 10)
print(f"\nSpecies with >=10 SNVs: {big.height}  (their SNV total: {int(big['snvs'].sum())})")
print(f"Species with >=20 SNVs: {by.filter(pl.col('snvs') >= 20).height}")
