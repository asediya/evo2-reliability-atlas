# Cross-species conformal prediction — a coverage-controlled trust layer that transfers across species (under a stated, tested exchangeability assumption)

Conformal quantile calibrated on label-rich species; applied to held-out TARGETS: ['goat', 'chicken', 'pig']. Nominal coverage 1-alpha = 0.90 (alpha=0.10) for the main table.

## Per-species coverage & set-size (alpha=0.10, nominal 90%)
Two honest constructions, each with coverage AND set sizes from its OWN membership rule. Mondrian holds ~0.97 by abstaining heavily; marginal holds ~0.95 overall as near-singletons but drops PATHOGENIC-class coverage.

| species | role | n | pos | Mondrian cov | Mond cov benign | Mond cov path | Mond abstain{both} | marginal cov | marg cov benign | marg cov path | marg abstain{both} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goat | TARGET | 99 | 9 | **0.970** | 0.967 | 1.000 | **0.505** | 0.929 | 1.000 | 0.222 | 0.000 |
| chicken | TARGET | 308 | 28 | **0.984** | 0.986 | 0.964 | **0.627** | 0.961 | 1.000 | 0.571 | 0.000 |
| pig | TARGET | 396 | 36 | **0.957** | 0.964 | 0.889 | **0.657** | 0.949 | 1.000 | 0.444 | 0.000 |
| sheep | source | 616 | 56 | **0.935** | 0.938 | 0.911 | **0.594** | 0.963 | 0.986 | 0.732 | 0.000 |
| horse | source | 781 | 71 | **0.945** | 0.942 | 0.972 | **0.680** | 0.959 | 0.992 | 0.634 | 0.000 |
| cat | source | 1365 | 125 | **0.966** | 0.969 | 0.936 | **0.704** | 0.954 | 0.995 | 0.544 | 0.000 |
| cattle | source | 2068 | 188 | **0.914** | 0.907 | 0.984 | **0.635** | 0.950 | 0.979 | 0.660 | 0.000 |
| dog | source | 2497 | 227 | **0.961** | 0.963 | 0.943 | **0.625** | 0.963 | 0.994 | 0.648 | 0.000 |
| human | source | 3000 | 1500 | **0.867** | 0.757 | 0.978 | **0.434** | 0.807 | 0.934 | 0.679 | 0.000 |

Mean Mondrian coverage on held-out TARGETS = **0.970** vs nominal 0.90.

## Coverage sweep on held-out label-poor targets — does empirical coverage track the guarantee?
Mondrian coverage/abstention vs marginal coverage/abstention, each internally consistent. Note the marginal predictor's PATHOGENIC-class coverage collapses well below nominal.
| alpha | nominal 1-alpha | Mondrian cov | Mond path cov | Mond abstain | marginal cov | marg path cov | marg abstain |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.95 | **0.993** | 0.932 | 0.658 | 0.978 | 0.753 | 0.178 |
| 0.1 | 0.9 | **0.969** | 0.932 | 0.626 | 0.951 | 0.466 | 0.0 |
| 0.2 | 0.8 | **0.953** | 0.753 | 0.127 | 0.917 | 0.356 | 0.0 |

**VERDICT:** cross-species conformal coverage tracks the nominal target on species with NO labels — Mondrian (class-conditional) holds ~0.97 coverage under the ~10:1 imbalance, but ONLY by abstaining on ~50-66% of target variants (it OVER-covers = conservative, not free). The prediction SETS are interpretable: singleton = confident call, {both} = safe abstention. HONEST FRAMING: coverage under a STATED, empirically-tested cross-species exchangeability assumption — NOT a distribution-free finite-sample theorem (cross-species transport breaks exchangeability). No single predictor gives both ~0.97 coverage AND ~99% singletons: the near-singleton (marginal) construction drops pathogenic-class coverage to ~0.22-0.57.

Honest caveat: exchangeability across species is imperfect, so coverage is approximate/assumption-conditional; Mondrian is reported because marginal coverage skews under class imbalance.
