# Decision-theoretic cost panel (calibration and abstention against trivial policies)

Asymmetric costs: false-benign (miss a pathogenic) = 10.0, false-alarm = 1.0, abstain = 0.5, correct = 0. Calibrated probs are leave-one-species-out isotonic transfers (pooled across all 9 species). Lower expected cost is better.


| backbone | policy | expected cost/variant | abstain rate | vs raw |
|---|---|---|---|---|
| Evo2-40B | A raw+0.5 (no cal/abstain) | **0.592** | 0.00 |  |
| Evo2-40B | B calibrated + reject-option | **0.642** | 0.58 | -8% |
| Evo2-40B | C calibrated + conformal | **0.536** | 0.71 | +9% |
| Evo2-40B | T1 always-abstain (trivial) | **0.500** | 1.00 | +16% |
| Evo2-40B | T2 always-call-pathogenic (trivial) | **0.799** | 0.00 | -35% |
| Evo2-40B | T3 always-call-benign (trivial) | **2.013** | 0.00 | -240% |

**Aggregation.** The costs in this block are pooled over all 9 species, and expected cost is prevalence-sensitive: human is 50% positive against ~9% elsewhere, so it dominates a pooled mean. Under the unweighted species mean this paper takes as primary, the same policies read A 0.447, B 0.505, C 0.487 against the same always-abstain floor 0.500 — so the best learned cell BEATS blanket abstention macro, and the pooled verdict above is a prevalence-weighted, human-driven result. Both are reported; neither is suppressed.


| backbone | policy | expected cost/variant | abstain rate | vs raw |
|---|---|---|---|---|
| NucleotideTransformer | A raw+0.5 (no cal/abstain) | **1.463** | 0.00 |  |
| NucleotideTransformer | B calibrated + reject-option | **0.540** | 0.97 | +63% |
| NucleotideTransformer | C calibrated + conformal | **0.603** | 0.86 | +59% |
| NucleotideTransformer | T1 always-abstain (trivial) | **0.500** | 1.00 | +66% |
| NucleotideTransformer | T2 always-call-pathogenic (trivial) | **0.799** | 0.00 | +45% |
| NucleotideTransformer | T3 always-call-benign (trivial) | **2.013** | 0.00 | -38% |

**Aggregation.** The costs in this block are pooled over all 9 species, and expected cost is prevalence-sensitive: human is 50% positive against ~9% elsewhere, so it dominates a pooled mean. Under the unweighted species mean this paper takes as primary, the same policies read A 1.059, B 0.514, C 0.544 against the same always-abstain floor 0.500 — so the best learned cell still does not beat blanket abstention macro, and the pooled verdict above is a prevalence-weighted, human-driven result. Both are reported; neither is suppressed.


**Trivial-baseline reality check.** At the chosen cost matrix (C_FN=10.0, C_FP=1.0, C_AB=0.5) the always-abstain floor is exactly C_AB = 0.500. NO learned policy — for either backbone — beats it: the best calibrated cell (NT reject-option ~0.540, Evo2 conformal ~0.536) sits ABOVE 0.500. NT's 'low' cost is an artifact of abstaining on ~97% of variants, i.e. it approaches the zero-information floor from above without reaching it. Under this cost matrix the conclusion is that calibration+abstention does not add decision value — blanket abstention dominates. Answering only wins once deferral is made expensive relative to errors (two-action Bayes break-even C_AB ~ 0.56 for Evo2, ~1.13 for NT) or the miss penalty is softened (e.g. C_FN=2, C_FP=1, C_AB=0.5: Evo2 two-action cost ~0.14, NT ~0.42 — both beat 0.500). Raising C_FN further does NOT rescue answering; it entrenches abstention.

**Pooled versus macro, stated plainly.** The 'no learned policy beats always-abstain' verdict above is pooled. It does not survive the paper's primary aggregation: see the macro line under each backbone. The pooled result is dominated by human, which is the only 1:1 panel and therefore carries roughly 5.5x the miss-cost exposure of a 10:1 panel. Both aggregations are deposited in reports/decision_panel.json.
