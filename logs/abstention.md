# OOD / abstention: selective prediction on transferred-calibrated Evo2-40B (Additional file 1, Table S26 and Figure S4)

The random-abstain column is one seeded draw (seed 0); the published random control is in Table S26.

Per-species deployment frame (9 species, macro-averaged). Confidence = |2p-1| on the LOSO-transferred calibrated probability; abstain on the least-confident variants.

| coverage | macro selective error | random-abstain error |
|---|---|---|
| 100% | 0.058 | 0.058 |
| 95% | 0.045 | 0.056 |
| 90% | 0.040 | 0.056 |
| 85% | 0.038 | 0.056 |
| 80% | 0.038 | 0.057 |
| 75% | 0.038 | 0.057 |
| 70% | 0.039 | 0.059 |
| 65% | 0.039 | 0.060 |
| 60% | 0.041 | 0.060 |
| 55% | 0.043 | 0.060 |
| 50% | 0.045 | 0.060 |
| 45% | 0.046 | 0.062 |
| 40% | 0.044 | 0.060 |
| 35% | 0.043 | 0.064 |
| 30% | 0.041 | 0.064 |

### Per-species trust budget
| species | n | pos | err@100% | err@90% | min err | @coverage |
|---|---|---|---|---|---|---|
| goat | 99 | 9 | 0.061 | 0.022 | **0.000** | 40% |
| chicken | 308 | 28 | 0.039 | 0.032 | **0.009** | 35% |
| sheep | 616 | 56 | 0.029 | 0.013 | **0.013** | 90% |
| dog | 2497 | 227 | 0.034 | 0.023 | **0.017** | 30% |
| pig | 396 | 36 | 0.051 | 0.025 | **0.019** | 65% |
| horse | 781 | 71 | 0.038 | 0.026 | **0.021** | 30% |
| cattle | 2068 | 188 | 0.044 | 0.030 | **0.025** | 50% |
| cat | 1365 | 125 | 0.043 | 0.028 | **0.027** | 80% |
| human | 3000 | 1500 | 0.179 | 0.160 | **0.155** | 85% |

- Macro selective error: 100% coverage **0.058** -> min **0.038** at 85% coverage (35% relative reduction).
- AURC (lower=better): confidence **0.029** vs random 0.041 -> confidence ordering is informative.
- Confidently-wrong tail: error at 30% coverage rises to 0.041 (> the 85%-coverage min 0.038) -> a high-confidence error subpopulation (dangerous false-benigns) that calibrated confidence alone can't catch -> motivates OOD features (consequence class, sequence entropy) as the next lever.

**PASS** — abstaining the least-confident variants cuts selective error and beats random abstention: the calibrated trust-layer is a deployable when-to-answer rule, with an identified confidently-wrong tail as future work.
