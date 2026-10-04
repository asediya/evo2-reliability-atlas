"""Host-side: turn a windows parquet into a FASTA for BioNeMo predict_evo2.
Each variant -> two records: v{i}__ref and v{i}__alt. Writes an idmap v{i} -> variant_id.
"""
import argparse, sys, json
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--idmap", required=True)
    a = ap.parse_args()

    df = pl.read_parquet(a.inp)
    vids = df["variant_id"].to_list()
    ref = df["ref_seq"].to_list()
    alt = df["alt_seq"].to_list()
    idmap = {}
    with open(a.fasta, "w") as f:
        for i, (v, rs, als) in enumerate(zip(vids, ref, alt)):
            f.write(f">v{i}__ref\n{rs}\n>v{i}__alt\n{als}\n")
            idmap[f"v{i}"] = v
    json.dump(idmap, open(a.idmap, "w"))
    print(f"wrote {a.fasta}: {len(vids)} variants x2 = {2*len(vids)} seqs")


if __name__ == "__main__":
    main()
