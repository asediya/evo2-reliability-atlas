# Crown-jewel rigor: calibration-transfer ECE with bootstrap CIs

2000× bootstrap per species. ECE lower=better; Δ = ECE_none − ECE_transfer (>0 = transfer helps). 'sig' = 95% bootstrap CI of Δ excludes 0.

| species | n | pos | role | ECE none [95% CI] | ECE transfer [95% CI] | Δ [95% CI] | p | sig |
|---|---|---|---|---|---|---|---|---|
| pig | 396 | 36 | target | 0.394 [0.374, 0.413] | 0.039 [0.018, 0.057] | **+0.356** [+0.331, +0.378] | 0.000 | YES |
| goat | 99 | 9 | target | 0.330 [0.287, 0.367] | 0.060 [0.025, 0.097] | **+0.269** [+0.222, +0.309] | 0.000 | YES |
| chicken | 308 | 28 | target | 0.224 [0.200, 0.246] | 0.021 [0.006, 0.041] | **+0.204** [+0.190, +0.212] | 0.000 | YES |
| cat | 1365 | 125 | source | 0.439 [0.427, 0.450] | 0.044 [0.034, 0.054] | **+0.395** [+0.384, +0.406] | 0.000 | YES |
| horse | 781 | 71 | source | 0.354 [0.339, 0.368] | 0.046 [0.033, 0.057] | **+0.308** [+0.295, +0.319] | 0.000 | YES |
| sheep | 616 | 56 | source | 0.336 [0.321, 0.351] | 0.056 [0.044, 0.066] | **+0.281** [+0.266, +0.294] | 0.000 | YES |
| dog | 2497 | 227 | source | 0.322 [0.315, 0.330] | 0.051 [0.044, 0.057] | **+0.272** [+0.265, +0.277] | 0.000 | YES |
| cattle | 2068 | 188 | source | 0.318 [0.309, 0.327] | 0.062 [0.054, 0.070] | **+0.257** [+0.250, +0.264] | 0.000 | YES |
| human | 3000 | 1500 | source | 0.155 [0.142, 0.169] | 0.119 [0.106, 0.132] | **+0.036** [+0.013, +0.059] | 0.001 | YES |

**Label-poor targets where transfer SIGNIFICANTLY beats no-calibration: 3/3** (95% CI of Δ excludes 0).
This is the resampling backbone of the calibration-transfer arm: transferred calibration is not just a lower point estimate — the improvement survives resampling at small N.
