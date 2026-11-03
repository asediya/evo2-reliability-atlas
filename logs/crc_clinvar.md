# Positive-class FNR certificate at ClinVar scale (conformal risk control; n~1.6M, labels exist)

## Evo2-EVEE probe (n=1,599,200)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.049 | 0.83 | yes | 0.005 | 0.10 | yes |
| 0.1 | 0.098 | 0.84 | yes | 0.021 | 0.12 | yes |
| 0.2 | 0.197 | 0.86 | yes | 0.070 | 0.17 | yes |

## AlphaMissense (n=204,631)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.050 | 0.56 | yes | 0.012 | 0.07 | yes |
| 0.1 | 0.098 | 0.64 | yes | 0.028 | 0.11 | yes |
| 0.2 | 0.195 | 0.71 | yes | 0.077 | 0.20 | yes |

## CADD (n=334,153)
| alpha | test (in-dist) FNR | benign-rate | holds? | coding->noncoding FNR | benign-rate | holds? |
|---|---|---|---|---|---|---|
| 0.05 | 0.052 | 0.41 | yes | 0.014 | 0.03 | yes |
| 0.1 | 0.100 | 0.47 | yes | 0.036 | 0.06 | yes |
| 0.2 | 0.201 | 0.55 | yes | 0.132 | 0.17 | yes |

## Summary
In-distribution: guarantee held in 9/9 (score x alpha) cases within the tolerance (FNR <= alpha + 0.02), 7/9 strictly (CADD at alpha = 0.05 and 0.2). Coding->noncoding transport: held in 9/9 within the tolerance (FNR <= alpha + 0.03), 9/9 strictly.

The positive-class-FNR certificate holds within the tolerance and is informative in-distribution on ClinVar, and (partly) survives the coding->noncoding shift. Conformal Risk Control of the missed-positive rate is therefore a valid guarantee when the target has labels or is exchangeable with the calibration data. Transport to a zero-label species is the separate test of build_crc_certificate.py (logs/crc_certificate.md).
