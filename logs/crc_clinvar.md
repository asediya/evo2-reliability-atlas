# GATE follow-up - pathogenic-FNR certificate at ClinVar scale (n~1.6M, labels exist)

## Evo2-EVEE probe (n=1,599,200)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.049 | 0.83 | YES | 0.005 | 0.10 | YES |
| 0.1 | 0.098 | 0.84 | YES | 0.021 | 0.12 | YES |
| 0.2 | 0.197 | 0.86 | YES | 0.070 | 0.17 | YES |

## AlphaMissense (n=204,631)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.56 | YES | 0.012 | 0.07 | YES |
| 0.1 | 0.098 | 0.64 | YES | 0.028 | 0.11 | YES |
| 0.2 | 0.195 | 0.71 | YES | 0.077 | 0.20 | YES |

## CADD (n=334,153)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.052 | 0.41 | YES | 0.014 | 0.03 | YES |
| 0.1 | 0.100 | 0.47 | YES | 0.036 | 0.06 | YES |
| 0.2 | 0.201 | 0.55 | YES | 0.132 | 0.17 | YES |

## VERDICT
In-distribution: guarantee held in 9/9 (score x alpha) cases with non-trivial benign-rates. Coding->noncoding transport: held in 9/9.

**OBJECT VALID at scale** - the pathogenic-FNR certificate holds and is informative in-distribution on ClinVar, and (partly) survives the coding->noncoding shift. So Conformal Risk Control for missed-pathogenics IS a legitimate new guarantee object when the target has SOME labels or is exchangeable. The genuinely HARD/open part is transport to a ZERO-label species (naive cross-species gate failed) - that needs the weighted correction and may be fundamentally limited. HONEST FRAMING for the paper: lead the FNR certificate on ClinVar/label-rich settings; present zero-label transport as coverage (which works) + FNR-under-stated-assumptions, not as a universal free lunch.
