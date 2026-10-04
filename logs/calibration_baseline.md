# Cross-species calibration transfer vs REAL baselines

Sibling report to `calibration_transfer.md` (which reports the LOSO isotonic transfer + oracle).
This adds the two baselines the transfer claim must actually beat: a pooled **global logistic
sigmoid** (Platt, 2-param) fit on the same source species, and a **base-rate constant** predictor
(pooled source prevalence). Everything is leave-one-species-out. ECE reported both equal-width and
adaptive/equal-mass, plus Brier (a proper score).

Label-rich sources (LOSO training pool): ['sheep', 'horse', 'cat', 'cattle', 'dog', 'human']

## Equal-width ECE (10 bins)

| species | pos | prev | ecw_none | ecw_base_rate | ecw_global_sigmoid | ecw_iso_transfer | ecw_oracle |
|---|---|---|---|---|---|---|---|
| goat | 9 | 0.091 | 0.323 | 0.119 | 0.048 | 0.056 | 0.063 |
| chicken | 28 | 0.091 | 0.224 | 0.119 | 0.021 | 0.02 | 0.014 |
| pig | 36 | 0.091 | 0.393 | 0.119 | 0.037 | 0.038 | 0.014 |
| sheep | 56 | 0.091 | 0.335 | 0.126 | 0.055 | 0.053 | 0.014 |
| horse | 71 | 0.091 | 0.351 | 0.129 | 0.044 | 0.043 | 0.013 |
| cat | 125 | 0.092 | 0.438 | 0.136 | 0.037 | 0.041 | 0.006 |
| cattle | 188 | 0.091 | 0.316 | 0.149 | 0.071 | 0.061 | 0.008 |
| dog | 227 | 0.091 | 0.322 | 0.157 | 0.049 | 0.05 | 0.003 |
| human | 1500 | 0.5 | 0.155 | 0.409 | 0.134 | 0.117 | 0.007 |

## Adaptive / equal-mass ECE (10 quantile bins)

| species | pos | ecm_none | ecm_base_rate | ecm_global_sigmoid | ecm_iso_transfer | ecm_oracle |
|---|---|---|---|---|---|---|
| goat | 9 | 0.316 | 0.121 | 0.061 | 0.068 | 0.067 |
| chicken | 28 | 0.21 | 0.119 | 0.022 | 0.024 | 0.028 |
| pig | 36 | 0.366 | 0.119 | 0.048 | 0.057 | 0.015 |
| sheep | 56 | 0.323 | 0.126 | 0.06 | 0.053 | 0.018 |
| horse | 71 | 0.34 | 0.129 | 0.038 | 0.039 | 0.014 |
| cat | 125 | 0.428 | 0.136 | 0.038 | 0.046 | 0.017 |
| cattle | 188 | 0.314 | 0.149 | 0.068 | 0.061 | 0.005 |
| dog | 227 | 0.306 | 0.157 | 0.039 | 0.048 | 0.011 |
| human | 1500 | 0.151 | 0.409 | 0.135 | 0.118 | 0.009 |

## Brier score (proper scoring rule)

| species | pos | br_none | br_base_rate | br_global_sigmoid | br_iso_transfer | br_oracle |
|---|---|---|---|---|---|---|
| goat | 9 | 0.152 | 0.097 | 0.041 | 0.046 | 0.056 |
| chicken | 28 | 0.092 | 0.097 | 0.04 | 0.038 | 0.039 |
| pig | 36 | 0.196 | 0.097 | 0.042 | 0.043 | 0.035 |
| sheep | 56 | 0.151 | 0.099 | 0.032 | 0.028 | 0.028 |
| horse | 71 | 0.168 | 0.099 | 0.04 | 0.037 | 0.038 |
| cat | 125 | 0.242 | 0.102 | 0.041 | 0.04 | 0.039 |
| cattle | 188 | 0.152 | 0.105 | 0.049 | 0.042 | 0.038 |
| dog | 227 | 0.143 | 0.107 | 0.036 | 0.033 | 0.031 |
| human | 1500 | 0.17 | 0.417 | 0.16 | 0.158 | 0.136 |

## Prevalence grid: isotonic-transfer equal-width ECE re-weighted to 9/25/50% positives

| species | prev | prev09 | prev25 | prev50 |
|---|---|---|---|---|
| goat | 0.091 | 0.056 | 0.099 | 0.259 |
| chicken | 0.091 | 0.021 | 0.068 | 0.191 |
| pig | 0.091 | 0.038 | 0.084 | 0.229 |
| sheep | 0.091 | 0.053 | 0.037 | 0.098 |
| horse | 0.091 | 0.043 | 0.033 | 0.142 |
| cat | 0.092 | 0.042 | 0.05 | 0.177 |
| cattle | 0.091 | 0.061 | 0.013 | 0.108 |
| dog | 0.091 | 0.05 | 0.028 | 0.137 |
| human | 0.5 | 0.062 | 0.071 | 0.117 |

## Verdict (honest)

- Global sigmoid is **on par with** isotonic transfer: sigmoid wins 4/9 species on equal-width ECE (median ECE sigmoid=0.048 vs isotonic=0.050).
- Brier is effectively tied (median sigmoid=0.041 vs isotonic=0.040); a 2-param sigmoid buys the same reliability as the nonparametric transfer.
- The base-rate constant is a genuine floor and both calibrators clear it.
- Transfer ECE is **prevalence-SENSITIVE only toward 50%** (non-monotone: FINE near the ~9% deployment prevalence, can even improve at 25%, rises only toward a balanced 50%). e.g. dog 0.05@9% -> 0.137@50%.
- Takeaway: the calibration-transfer win is real but NOT better than a trivial global sigmoid, and it is only trustworthy near the prevalence it was measured at.
