# -*- coding: utf-8 -*-
"""Build the 8,192-bp payload for the one 40B run that answers A2, A3 and the readout split.

Four forward passes per variant — reference and alternate, each forward and reverse-complemented —
at 8,192 bp yield everything still outstanding:

  A2  strand consistency at the headline model, at both readouts
  A3  MLL(ref), which the deposited scorer computes and discards, so the readout advantage can be
      stratified by how likely the model finds the background window
  the readout decomposition on identical variants, since a single pass over an 8,192-bp window
  gives the full-window mean, the central-1,001 mean and the single-position log-likelihood at once

The variants are the same 389 already scored at 1B and 7B, so the capacity trend extends to 40B
rather than starting again.

    python analyses/scripts/build_40b_payload.py
"""
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "analyses/data/strand/upload40b"


def main():
    sub = pl.read_parquet("analyses/data/strand/strand_subset_windows.parquet")
    want = set(sub["variant_id"].to_list())
    print("  subset: %d variants already scored at 1B and 7B" % len(want))

    frames, missing = [], set(want)
    for f in sorted(glob.glob("data/interim/atlas8192/*_windows_8192.parquet")):
        sp = os.path.basename(f).replace("_windows_8192.parquet", "")
        d = pl.read_parquet(f)
        keep = d.filter(pl.col("variant_id").is_in(list(want)))
        if not len(keep):
            continue
        cols = ["variant_id", "ref_seq", "alt_seq"]
        if "var_off" in keep.columns:
            cols.append("var_off")
        keep = keep.select(cols).with_columns(pl.lit(sp).alias("species"))
        frames.append(keep)
        missing -= set(keep["variant_id"].to_list())
        print("  %-14s %4d of the subset carry an 8,192-bp window" % (sp, len(keep)))

    if not frames:
        raise SystemExit("no 8,192-bp windows matched the subset")
    p = pl.concat(frames, how="diagonal").unique(subset=["variant_id"])
    lab = sub.select(["variant_id", "label"])
    p = p.join(lab, on="variant_id", how="left")

    os.makedirs(OUT, exist_ok=True)
    q = os.path.join(OUT, "strand40b_w8192.parquet")
    p.write_parquet(q)
    n_seq = len(p["ref_seq"][0])
    print()
    print("  matched %d of %d; %d have no 8,192-bp window" % (len(p), len(want), len(missing)))
    print("  wrote %s  (%d variants, %d bp windows, %.1f MB)"
          % (q, len(p), n_seq, os.path.getsize(q) / 1e6))
    print("  passes needed: %d variants x 4 (ref/alt x fwd/revcomp) = %d sequences"
          % (len(p), 4 * len(p)))


if __name__ == "__main__":
    main()
