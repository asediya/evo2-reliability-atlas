# STEP 2 hardening - conformal coverage under PREVALENCE SHIFT (de-risking the guarantee claim)

Split-conformal fit on label-rich species (nominal 90%), applied to the pooled held-out TARGETS ['goat', 'chicken', 'pig'] (natural pathogenic prevalence 0.09). We RESAMPLE the targets to a range of pathogenic prevalences and report MARGINAL vs MONDRIAN (class-conditional) coverage + set sizes.

| target prevalence | marginal cov | Mondrian cov | singleton rate | abstain{both} rate |
|---|---|---|---|---|
| 0.05 | 0.973 | **0.970** | 0.36 | 0.64 |
| 0.10 | 0.947 | **0.969** | 0.38 | 0.62 |
| 0.20 | 0.893 | **0.964** | 0.41 | 0.59 |
| 0.35 | 0.812 | **0.958** | 0.47 | 0.53 |
| 0.50 | 0.733 | **0.952** | 0.53 | 0.47 |

**Result:** Mondrian coverage stays ~nominal across prevalences from 5% to 50% (min 0.952 vs 0.90), while MARGINAL coverage drifts (0.733-0.973) as prevalence changes. The class-conditional (Mondrian) guarantee is ROBUST to the prevalence/imbalance shift that a reviewer would attack -- which is why it, not marginal coverage, is the headline.

**Honest reframing (replaces the over-claim):** we do NOT claim a distribution-free finite-sample guarantee on an unseen species; cross-species transport violates exact exchangeability. We claim, and here TEST, coverage under a stated cross-species exchangeability assumption -- empirically it holds (Mondrian ~nominal) and degrades gracefully, and the observed OVER-coverage is conservative finite-sample slack (a small calibration set), i.e. an efficiency cost, not evidence the guarantee fails. Set sizes/abstention are reported so coverage is never read in isolation.
