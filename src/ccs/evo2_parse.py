"""Host-side: parse BioNeMo predict_evo2 --output-log-prob-seqs output into per-variant
Evo2 scores. delta = logP(alt) - logP(ref); evo2_neg = -delta (deleteriousness, matches CP1).

Inputs: predictions__rank_0.pt {log_probs_seqs[N], seq_idx[N]}, seq_idx_map.json {name:index},
        idmap.json {v{i}: variant_id}.
"""
import argparse, sys, json
import torch
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--seqmap", required=True)
    ap.add_argument("--idmap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    o = torch.load(a.pred, map_location="cpu")
    lp = o["log_probs_seqs"].float().numpy()
    sidx = o["seq_idx"].long().numpy()
    seqmap = json.load(open(a.seqmap))   # name -> index value
    idmap = json.load(open(a.idmap))     # v{i} -> variant_id

    pos_of_idx = {int(ix): p for p, ix in enumerate(sidx)}   # index value -> row in lp

    def lp_of(name):
        return float(lp[pos_of_idx[int(seqmap[name])]])

    rows = []
    missing = 0
    for vk, vid in idmap.items():
        rn, an = f"{vk}__ref", f"{vk}__alt"
        if rn in seqmap and an in seqmap:
            d = lp_of(an) - lp_of(rn)
            rows.append((vid, d, -d))
        else:
            missing += 1
    out = pl.DataFrame(rows, schema=["variant_id", "evo2_delta", "evo2_neg"], orient="row")
    out.write_parquet(a.out)
    print(f"wrote {a.out}: {out.height} variants (missing {missing}) "
          f"delta mean {float(np.mean([r[1] for r in rows])):.3f}")


if __name__ == "__main__":
    main()
