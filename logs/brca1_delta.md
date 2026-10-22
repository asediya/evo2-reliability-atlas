# BRCA1 positive control — variant-delta (evo2_40b_neg)

Scored 3893 BRCA1 SNVs (823 loss-of-function / 3070 functional or intermediate) with the SAME variant-delta
scorer + 1001bp windows as the cross-species atlas (score_evo2_40b_local.py). This validates the
single-position harness; the 8,192-bp mean-LL readout is reported separately.

- **AUROC (evo2_40b_neg vs loss-of-function) = 0.632**
- mean evo2_40b_neg: loss-of-function 4.366 vs functional or intermediate 2.825 (delta +1.541; higher=more damaging)

**PARTIAL** — signal present but AUROC below the 0.73 the Arc Institute notebook prints for Evo 2-1B (variant-delta != mean-LL protocol; short 1001bp context).
