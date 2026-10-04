# STEP 2 - does Evo2-40B beat conservation (GERP)? One-sided bootstrap p + BH-FDR across 9 species

Baseline = GERP (sole conservation baseline; the fabricated byte-identical phyloP copies are no longer used). The atlas statistic `p_40b_gt_cons` is the bootstrap fraction P(delta>0); we convert it to a one-sided p-value  p = 1 - P(delta>0)  and apply Benjamini-Hochberg at q<=0.05.

| species | n | pos | AUROC Evo2 | AUROC GERP | Δ | P(Δ>0) | one-sided p | BH q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| goat | 99 | 9 | 0.951 | 0.78 | +0.173 | 1.000 | 0.000 | 0.000 | SIGNIFICANT (BH) |
| cattle | 2068 | 188 | 0.9 | 0.825 | +0.075 | 1.000 | 0.000 | 0.000 | SIGNIFICANT (BH) |
| chicken | 308 | 28 | 0.861 | 0.725 | +0.137 | 0.972 | 0.028 | 0.084 | n.s. after FDR |
| sheep | 616 | 56 | 0.903 | 0.865 | +0.047 | 0.844 | 0.156 | 0.351 | n.s. after FDR |
| pig | 396 | 36 | 0.848 | 0.751 | +0.063 | 0.776 | 0.224 | 0.403 | n.s. after FDR |
| dog | 2497 | 227 | 0.888 | 0.886 | +0.009 | 0.695 | 0.305 | 0.408 | n.s. after FDR |
| cat | 1365 | 125 | 0.846 | 0.846 | +0.013 | 0.683 | 0.317 | 0.408 | n.s. after FDR |
| horse | 781 | 71 | 0.88 | 0.915 | -0.033 | 0.199 | 0.801 | 0.901 | n.s. after FDR |
| human | 3000 | 1500 | 0.825 | 0.874 | -0.048 | 0.000 | 1.000 | 1.000 | n.s. after FDR |

**FDR-corrected result: 2/9 species beat GERP after Benjamini-Hochberg (q<=0.05): goat, cattle.** Both survivors are floor-limited (P(Δ>0)=1.000 -> one-sided p=0.000).

**Honest reading:** the bootstrap P(Δ>0) is NOT a p-value and was uncorrected for 9 simultaneous tests. Once converted to a proper one-sided p-value and BH-corrected, only the two species whose bootstrap never once favored conservation survive; every borderline win (chicken +0.137 at raw p=0.028, and all smaller margins) is not significant after FDR. The atlas's headline contribution is the CALIBRATION / trust-layer transfer, not a raw-accuracy win over conservation.
