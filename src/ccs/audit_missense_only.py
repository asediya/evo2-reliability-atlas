# -*- coding: utf-8 -*-
"""AlphaMissense vs REVEL on missense variants only -- the panel neither scorer can object to.

The whole-ClinVar reversal invites a fair objection: nobody benchmarks a missense-only predictor on
introns, so a reversal there is a statement about the panel rather than the scorers. Restricting to
missense removes that objection entirely, and the question becomes whether reach still changes the
answer on the panel both tools explicitly claim.
"""
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
# this hard-coded the author's drive, so the script could not run
# from a clean extraction of the deposit. Resolve the repository root from this file instead.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "glmtrust/src")
from glmtrust.audit import Scorer, audit                              # noqa: E402
from glmtrust.card import write_card                                  # noqa: E402

panel = pl.read_parquet("data/processed/clinvar_panel.parquet")
for f in ("alphamissense", "revel"):
    panel = panel.join(pl.read_parquet(f"data/processed/human_scorers/{f}.parquet"),
                       on="variant_id", how="left")

for stars in (1, 2):
    p = panel.filter((pl.col("stars") >= stars) & (pl.col("consequence") == "missense_variant"))
    y = p["label"].to_numpy().astype(int)
    print("=" * 84)
    print("  MISSENSE ONLY, >=%d star   n=%s   %s pathogenic / %s benign"
          % (stars, format(len(y), ","), format(int(y.sum()), ","),
             format(int((y == 0).sum()), ",")))
    print("=" * 84)
    rep = audit(y, [Scorer("alphamissense", p["alphamissense"].to_numpy().astype(float),
                           readout="per-substitution point score"),
                    Scorer("revel", p["revel"].to_numpy().astype(float),
                           readout="per-substitution point score")],
                n_boot=200, seed=0)
    print(rep)
    print()
    if stars == 1:
        write_card(rep, "reports/missense_only_audit.html",
                   title="AlphaMissense vs REVEL on ClinVar missense variants",
                   subtitle="%s variants at 1+ review star; the panel both scorers claim"
                            % format(len(y), ","))
