# Cross-species calibration transfer (LOSO + oracle + nearest-relative)

Label-rich sources (isotonic fit): ['sheep', 'horse', 'cat', 'cattle', 'dog', 'human']

| species | clade | pos | auroc | ece_none | ece_transfer | ece_oracle | ece_nearest | nearest_from |
|---|---|---|---|---|---|---|---|---|
| goat | ruminant | 9 | 0.951 | 0.323 | 0.056 | 0.063 | 0.04 | sheep,cattle |
| chicken | bird | 28 | 0.861 | 0.224 | 0.02 | 0.014 | None | - |
| pig | suid | 36 | 0.848 | 0.393 | 0.038 | 0.014 | None | - |
| sheep | ruminant | 56 | 0.903 | 0.335 | 0.053 | 0.014 | 0.009 | cattle |
| horse | perissodactyl | 71 | 0.88 | 0.351 | 0.043 | 0.013 | None | - |
| cat | carnivore | 125 | 0.846 | 0.438 | 0.041 | 0.006 | 0.01 | dog |
| cattle | ruminant | 188 | 0.9 | 0.316 | 0.061 | 0.008 | 0.025 | sheep |
| dog | carnivore | 227 | 0.888 | 0.322 | 0.05 | 0.003 | 0.012 | cat |
| human | primate | 1500 | 0.825 | 0.155 | 0.117 | 0.007 | None | - |

**Verdict:** transfer beat no-calibration on 3/3 label-poor species; median ECE transfer=0.050 vs oracle=0.013 (closer = transfer works).

> See `logs/calibration_baseline.md` for the honest head-to-head against a pooled global sigmoid (Platt) + base-rate baselines, equal-mass ECE, Brier, and the prevalence-fragility grid.
