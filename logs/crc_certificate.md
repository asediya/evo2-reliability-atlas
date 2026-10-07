# GATE TEST (Idea 2) - transported pathogenic-FNR certificate

Control E[missed-pathogenic rate] <= alpha; calibrate lambda on label-rich species via Conformal Risk Control; transport to each held-out species. VALID if achieved FNR <= alpha + 0.02 (a 0.02 tolerance on the nominal level; the strict count is in the verdict); NON-TRIVIAL if it still calls a real fraction benign (benign-rate > 0.15).

## alpha = 0.05 (guarantee: <= 5% of true pathogenics missed)
| held-out species | role | n | pos | achieved FNR | <=alpha+0.02? | benign-call rate | informative? |
|---|---|---|---|---|---|---|---|
| goat | TARGET | 99 | 9 | 0.111 | no | 0.72 | yes |
| chicken | TARGET | 308 | 28 | 0.321 | no | 0.87 | yes |
| pig | TARGET | 396 | 36 | 0.194 | no | 0.73 | yes |
| sheep | source | 616 | 56 | 0.107 | no | 0.70 | yes |
| horse | source | 781 | 71 | 0.225 | no | 0.82 | yes |
| cat | source | 1365 | 125 | 0.216 | no | 0.72 | yes |
| cattle | source | 2068 | 188 | 0.186 | no | 0.77 | yes |
| dog | source | 2497 | 227 | 0.216 | no | 0.85 | yes |
| human | source | 3000 | 1500 | 0.075 | no | 0.12 | no |

## alpha = 0.1 (guarantee: <= 10% of true pathogenics missed)
| held-out species | role | n | pos | achieved FNR | <=alpha+0.02? | benign-call rate | informative? |
|---|---|---|---|---|---|---|---|
| goat | TARGET | 99 | 9 | 0.111 | YES | 0.72 | yes |
| chicken | TARGET | 308 | 28 | 0.321 | no | 0.87 | yes |
| pig | TARGET | 396 | 36 | 0.194 | no | 0.73 | yes |
| sheep | source | 616 | 56 | 0.107 | YES | 0.70 | yes |
| horse | source | 781 | 71 | 0.225 | no | 0.82 | yes |
| cat | source | 1365 | 125 | 0.216 | no | 0.72 | yes |
| cattle | source | 2068 | 188 | 0.186 | no | 0.77 | yes |
| dog | source | 2497 | 227 | 0.216 | no | 0.85 | yes |
| human | source | 3000 | 1500 | 0.151 | no | 0.26 | yes |

## alpha = 0.2 (guarantee: <= 20% of true pathogenics missed)
| held-out species | role | n | pos | achieved FNR | <=alpha+0.02? | benign-call rate | informative? |
|---|---|---|---|---|---|---|---|
| goat | TARGET | 99 | 9 | 0.111 | YES | 0.82 | yes |
| chicken | TARGET | 308 | 28 | 0.429 | no | 0.92 | yes |
| pig | TARGET | 396 | 36 | 0.222 | no | 0.84 | yes |
| sheep | source | 616 | 56 | 0.107 | YES | 0.80 | yes |
| horse | source | 781 | 71 | 0.225 | no | 0.82 | yes |
| cat | source | 1365 | 125 | 0.296 | no | 0.91 | yes |
| cattle | source | 2068 | 188 | 0.186 | YES | 0.77 | yes |
| dog | source | 2497 | 227 | 0.242 | no | 0.90 | yes |
| human | source | 3000 | 1500 | 0.199 | YES | 0.38 | yes |

## GATE VERDICT
Across 9 (alpha x zero-label-target) cases: guarantee HELD in 2/9 within the tolerance (1/9 strictly, goat at alpha = 0.2), and was NON-TRIVIAL (benign-rate>15%) in 2/9.

**GATE FAIL/WEAK** - the certificate is vacuous or does not transport non-trivially on OMIA (small n / weak separation). Re-run the gate on the large-N ClinVar/ProteinGym data before concluding; if it fails there too, the theorem does not carry the paper and the calibration result is what should be reported.
