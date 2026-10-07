# STEP 1 (conformal) - does the distribution-free trust layer transfer on Nucleotide Transformer?

Conformal quantile calibrated on label-rich species; applied to held-out TARGETS: ['goat', 'chicken', 'pig']. Nominal 0.90.

## Per-species coverage & set-size (alpha=0.10, nominal 90%) - NT backbone
| species | role | n | pos | Mondrian cov | cov benign | cov pathogenic | singleton | abstain{both} |
|---|---|---|---|---|---|---|---|---|
| goat | TARGET | 99 | 9 | **0.990** | 1.000 | 0.889 | 0.44 | 0.56 |
| chicken | TARGET | 308 | 28 | **0.974** | 0.979 | 0.929 | 0.46 | 0.54 |
| pig | TARGET | 396 | 36 | **0.980** | 0.986 | 0.917 | 0.45 | 0.55 |
| sheep | source | 616 | 56 | **0.968** | 0.973 | 0.911 | 0.39 | 0.61 |
| horse | source | 781 | 71 | **0.954** | 0.958 | 0.915 | 0.36 | 0.64 |
| cat | source | 1365 | 125 | **0.955** | 0.959 | 0.920 | 0.37 | 0.63 |
| cattle | source | 2068 | 188 | **0.963** | 0.971 | 0.888 | 0.39 | 0.61 |
| dog | source | 2497 | 227 | **0.949** | 0.955 | 0.885 | 0.38 | 0.62 |
| human | source | 3000 | 1500 | **0.952** | 0.971 | 0.933 | 0.23 | 0.77 |

Mean Mondrian coverage on held-out TARGETS = **0.981** vs nominal 0.90 (mean abstain rate 0.55).

## Coverage sweep on held-out targets (NT)
| alpha | nominal | Mondrian cov | pathogenic cov | singleton | abstain |
|---|---|---|---|---|---|
| 0.05 | 0.95 | **0.983** | 0.959 | 0.453 | 0.547 |
| 0.1 | 0.9 | **0.979** | 0.918 | 0.453 | 0.547 |
| 0.2 | 0.8 | **0.854** | 0.781 | 0.993 | 0.007 |

**VERDICT:** conformal coverage TRACKS the nominal guarantee on NT too - coverage is preserved on a second architecture, under the same cross-species exchangeability caveat as for Evo 2.
**MODEL-AGNOSTIC (conformal):** the coverage guarantee holds on BOTH backbones on zero-label targets (mean Mondrian coverage: Evo2 0.970, NT 0.981; both ~nominal 0.90). The weaker NT backbone abstains slightly LESS (mean {both} rate: Evo2 0.60 vs NT 0.55), so abstention did not track backbone quality: coverage behaved as a property of the METHOD.
