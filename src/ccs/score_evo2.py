"""Evo2 zero-shot variant scoring — reconstruction of the CP1 harness (which ran inside the
BioNeMo container). Runs anywhere `evo2` is importable: the BioNeMo
2.6.2 container OR a CUDA environment with the Arc `evo2` pip package.

Method (matches Arc Institute's BRCA1 zero-shot VEP + the cached CP1 schema):
  evo2_delta = score_sequences(alt_window) - score_sequences(ref_window)   # logL(alt) - logL(ref)
  evo2_neg   = -evo2_delta                                                 # deleteriousness (higher = worse)

Input parquet: variant_id, ref_seq, alt_seq  (equal-length DNA windows; alt differs at one base)
Output parquet: variant_id, evo2_delta, evo2_neg

  python score_evo2.py --in data/interim/pig_scoring_windows.parquet \
      --out data/processed/scores/pig_evo2_scores.parquet --model evo2_1b_base --batch 8
"""
import argparse, sys, time
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_model(name):
    # Arc evo2 package (also the import path inside BioNeMo 2.6.2)
    from evo2 import Evo2
    return Evo2(name)


def score_seqs(model, seqs, batch):
    out = []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        sc = model.score_sequences(chunk)          # list[float], mean log-likelihood per seq
        out.extend([float(x) for x in sc])
    return np.asarray(out, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="evo2_1b_base")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    df = pl.read_parquet(a.inp)
    lens = df["ref_seq"].str.len_chars().unique().to_list()
    assert len(lens) == 1, f"windows must be equal length, got {lens[:5]}"
    ref = df["ref_seq"].to_list(); alt = df["alt_seq"].to_list()
    vids = df["variant_id"].to_list()
    print(f"scoring {len(vids)} variants x2 windows (len {lens[0]}) with {a.model} ...", flush=True)

    model = load_model(a.model)
    t0 = time.time()
    ref_s = score_seqs(model, ref, a.batch)
    print(f"  ref done {time.time()-t0:.0f}s", flush=True)
    alt_s = score_seqs(model, alt, a.batch)
    delta = alt_s - ref_s

    out = pl.DataFrame({"variant_id": vids,
                        "evo2_delta": delta,
                        "evo2_neg": -delta})
    out.write_parquet(a.out)
    print(f"wrote {a.out}  rows={out.height}  in {time.time()-t0:.0f}s "
          f"(delta mean {float(delta.mean()):.3f})")


if __name__ == "__main__":
    main()
