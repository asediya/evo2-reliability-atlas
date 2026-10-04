# STEP 2 hardening - bootstrap 95% CIs on the headline tables (Evo2-40B)

Nonparametric percentile bootstrap, 2000x, stratified by class. Transferred ECE uses an isotonic map fit on the OTHER label-rich species (leave-one-species-out), bootstrapping the target species.

| species | n | pos | AUROC | AUROC 95% CI | ECE transfer | ECE 95% CI |
|---|---|---|---|---|---|---|
| goat | 99 | 9 | 0.951 | [0.87, 1.0] | 0.056 | [0.02, 0.093] |
| chicken | 308 | 28 | 0.861 | [0.768, 0.936] | 0.02 | [0.005, 0.038] |
| pig | 396 | 36 | 0.848 | [0.741, 0.939] | 0.038 | [0.02, 0.057] |
| sheep | 616 | 56 | 0.903 | [0.827, 0.965] | 0.053 | [0.045, 0.067] |
| horse | 781 | 71 | 0.88 | [0.821, 0.933] | 0.043 | [0.036, 0.056] |
| cat | 1365 | 125 | 0.846 | [0.795, 0.892] | 0.041 | [0.035, 0.053] |
| cattle | 2068 | 188 | 0.9 | [0.87, 0.928] | 0.061 | [0.055, 0.068] |
| dog | 2497 | 227 | 0.888 | [0.857, 0.915] | 0.05 | [0.044, 0.057] |
| human | 3000 | 1500 | 0.825 | [0.808, 0.841] | 0.117 | [0.107, 0.13] |

**Honest reading:** the label-poor species (goat/chicken/pig) carry WIDE AUROC CIs (small positive counts) - we report point estimates with intervals rather than over-claiming per-species accuracy. The calibration/trust-layer claims are made at the POOLED level and via the transferred-ECE reduction, which is where the evidence is strong; per-species AUROC is contextual, not a headline.
