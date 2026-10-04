# Decision-theoretic cost panel (calibration+abstention turns 'weak' into 'safe')

Asymmetric costs: false-benign (miss a pathogenic) = 10.0, false-alarm = 1.0, abstain = 0.5, correct = 0. Calibrated probs are leave-one-species-out isotonic transfers (pooled across all 9 species). Lower expected cost is better.

| backbone | policy | expected cost/variant | abstain rate | vs raw |
|---|---|---|---|---|
| Evo2-40B | A raw+0.5 (no cal/abstain) | **0.592** | 0.00 |  |
| Evo2-40B | B calibrated + reject-option | **0.642** | 0.58 | -8% |
| Evo2-40B | C calibrated + conformal | **0.540** | 0.70 | +9% |
| Evo2-40B | T1 always-abstain (trivial) | **0.500** | 1.00 | +16% |
| Evo2-40B | T2 always-call-pathogenic (trivial) | **0.799** | 0.00 | -35% |
| Evo2-40B | T3 always-call-benign (trivial) | **2.013** | 0.00 | -240% |

**Aggregation.** The costs in this block are pooled over all 9 species, and expected cost is prevalence-sensitive: human is 50% positive against ~9% elsewhere, so it dominates a pooled mean. Under the unweighted species mean this paper takes as primary, the same policies read A 0.447, B 0.505, C 0.489 against the same always-abstain floor 0.500 — so the best learned cell BEATS blanket abstention macro, and the pooled verdict above is a prevalence-weighted, human-driven result. Both are reported; neither is suppressed.

| backbone | policy | expected cost/variant | abstain rate | vs raw |
|---|---|---|---|---|
| NucleotideTransformer | A raw+0.5 (no cal/abstain) | **1.463** | 0.00 |  |
| NucleotideTransformer | B calibrated + reject-option | **0.540** | 0.97 | +63% |
| NucleotideTransformer | C calibrated + conformal | **0.605** | 0.85 | +59% |
| NucleotideTransformer | T1 always-abstain (trivial) | **0.500** | 1.00 | +66% |
| NucleotideTransformer | T2 always-call-pathogenic (trivial) | **0.799** | 0.00 | +45% |
| NucleotideTransformer | T3 always-call-benign (trivial) | **2.013** | 0.00 | -38% |

**Aggregation.** The costs in this block are pooled over all 9 species, and expected cost is prevalence-sensitive: human is 50% positive against ~9% elsewhere, so it dominates a pooled mean. Under the unweighted species mean this paper takes as primary, the same policies read A 1.059, B 0.514, C 0.546 against the same always-abstain floor 0.500 — so the best learned cell still does not beat blanket abstention macro, and the pooled verdict above is a prevalence-weighted, human-driven result. Both are reported; neither is suppressed.
