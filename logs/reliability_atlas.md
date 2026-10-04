> **Development log.** Despite the title, this is not the submitted Table 1. It
> reports the 1,001-bp single-position series, whose species mean is 0.878; the submitted Table 2
> leads with the 8,192-bp reference-implementation readout at a species mean of 0.943, and the two
> readouts are the subject of the paper's first accounting result. Read this as the 1,001-bp readout,
> not as the paper's headline.

# Calibrated cross-species reliability atlas — master table (paper Table 1)

One row per species. **AUROC 40B** = Evo2-40B zero-shot; **Δvs cons/1B** = paired gain over best conservation / Evo2-1B; **ECE** = calibration (no-cal → transferred); **ΔFM** = signal the FM adds beyond conservation; **@low-cons** = Evo2 AUROC at below-median-conservation sites; **trust** = abstention selective error (100% coverage → min).

| species | clade | N | pos | AUROC 40B | Δvs cons | Δvs 1B | ECE none→transfer | ΔFM | Evo2 @low-cons | trust err 100%→min@cov |
|---|---|---|---|---|---|---|---|---|---|---|
| human | primate | 3000 | 1500 | 0.825 | -0.048 | +0.146 | 0.155→0.117 | +0.035 | 0.850 | 0.179→0.155@85% |
| cattle | ruminant | 2068 | 188 | 0.900 | +0.075 | +0.231 | 0.316→0.061 | +0.086 | 0.902 | 0.044→0.025@50% |
| dog | carnivore | 2497 | 227 | 0.888 | +0.009 | +0.217 | 0.322→0.050 | +0.050 | 0.807 | 0.034→0.017@30% |
| cat | carnivore | 1365 | 125 | 0.846 | +0.013 | +0.221 | 0.438→0.041 | +0.031 | 0.585 | 0.043→0.027@80% |
| horse | perissodactyl | 781 | 71 | 0.880 | -0.033 | +0.216 | 0.351→0.043 | +0.039 | 0.947 | 0.038→0.021@30% |
| sheep | ruminant | 616 | 56 | 0.903 | +0.047 | +0.227 | 0.335→0.053 | +0.047 | 0.500 | 0.029→0.013@90% |
| pig | suid | 396 | 36 | 0.848 | +0.063 | +0.200 | 0.393→0.038 | +0.015 | 0.737 | 0.051→0.019@65% |
| goat | ruminant | 99 | 9 | 0.951 | +0.173 | +0.263 | 0.323→0.056 | — | — | 0.061→0.000@40% |
| chicken | bird | 308 | 28 | 0.861 | +0.137 | +0.201 | 0.224→0.020 | +0.117 | 0.866 | 0.039→0.009@35% |

Species with 40B scored: 9/9. Mean AUROC-40B = 0.878.
