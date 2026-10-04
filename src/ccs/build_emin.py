# -*- coding: utf-8 -*-
"""Deposit the per-species selective-error floor (emin): reports/fig4_emin.json

The Results quote a per-species error floor ("0.000-0.025 in eight species
to 0.155 in human") whose only source was a value computed inside `build_abstention.py` and never
written to a deposited artifact, so a reader could not check it.

emin is the lowest selective error a species reaches at any coverage on the deployable
leave-one-species-out arm: variants are ordered by confidence |2p - 1| on the transferred isotonic
posterior, the most-confident fraction is retained at each coverage on the paper's grid, and the
error rate on the retained set is evaluated at the 0.5 decision threshold. The minimum over the grid
is that species' floor.

    python src/ccs/build_emin.py
"""
import io
import json
import os
import sys

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.environ.get("CCS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PV = os.path.join(ROOT, "reports", "_recon_pervariant_trust.parquet")
GRID = os.path.join(ROOT, "reports", "fig4_reconciliation.json")
OUT = os.path.join(ROOT, "reports", "fig4_emin.json")


def main():
    pv = pl.read_parquet(PV)
    grid = json.load(io.open(GRID, encoding="utf-8"))["abstention"]["isotonic_LOSO"]["coverage"]
    out = {
        "_definition": ("lowest selective error reached at any coverage on the grid, per species, on "
                        "the deployable leave-one-species-out isotonic arm; confidence is |2p-1| and "
                        "calls are made at the 0.5 threshold"),
        "coverage_grid": grid,
        "arm": "isotonic_LOSO (no in-species oracle)",
        "per_species": {},
    }
    for sp in sorted(set(pv["species"].to_list())):
        s = pv.filter(pl.col("species") == sp)
        p = s["isotonic_LOSO"].to_numpy()
        y = s["label"].to_numpy().astype(int)
        keep = ~np.isnan(p)
        p, y = p[keep], y[keep]
        conf = np.abs(2 * p - 1)
        order = np.argsort(-conf, kind="stable")
        p, y = p[order], y[order]
        wrong = ((p >= 0.5).astype(int) != y).astype(float)
        curve = {}
        for cov in grid:
            k = max(1, int(round(cov * len(y))))
            curve[f"{cov:.2f}"] = float(wrong[:k].mean())
        emin = min(curve.values())
        at = min(curve, key=lambda c: curve[c])
        out["per_species"][sp] = {
            "n": int(len(y)),
            "error_full_coverage": round(curve[f"{max(grid):.2f}"], 4),
            "emin": round(emin, 4),
            "emin_at_coverage": float(at),
            "curve": {k: round(v, 4) for k, v in curve.items()},
        }

    vals = {k: v["emin"] for k, v in out["per_species"].items()}
    nonhuman = {k: v for k, v in vals.items() if k != "human"}
    out["summary"] = {
        "eight_species_range": [round(min(nonhuman.values()), 4), round(max(nonhuman.values()), 4)],
        "human": vals.get("human"),
    }
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1) + "\n")

    print("wrote reports/fig4_emin.json")
    for k, v in sorted(vals.items(), key=lambda kv: kv[1]):
        print(f"  {k:8s} emin={v:.4f}  at coverage {out['per_species'][k]['emin_at_coverage']:.2f}")
    print(f"\neight non-human species: {out['summary']['eight_species_range']}   human: {vals.get('human')}")
    # this printed a comparison against a claim that appears in
    # neither the manuscript nor the supplement, and the computed maximum is 0.0275.
    print("no manuscript claim is asserted here; compare against reports/fig4_emin.json")


if __name__ == "__main__":
    main()
