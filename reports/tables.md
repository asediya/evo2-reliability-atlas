# Deposit tables — generated, do not edit by hand

Built by `src/ccs/build_tables.py` from the figure recompute layers; headline values are asserted against literals in build_tables.py at build time. Regenerate rather than patch.

**These are the deposit's own tables, D1-D4, and they are not the tables printed in the manuscript.** They present the same recompute layer more fully -- Table D1 carries the 8,192-bp panel with fourteen columns; the manuscript's eight-column Table 2 reports the same nine species at both readouts against GERP, on slightly different per-species panels, so its point estimates agree with D1 at three decimals while its intervals do not.


## Table D1. Cross-species reliability atlas for Evo 2-40B

Pooled at 8192 bp mean-LL: **0.973** [0.968, 0.978], n = 11,109. Readout lift 1001 bp → 8192 bp: **+0.092** [+0.083, +0.101]. Pooled ΔFM **+0.0202** [+0.0132, +0.0279].

| species | clade | 8192 bp panel n | AUROC @8192 | 95% CI (AUROC @8192) | AUROC @1001 | readout Δ | GERP panel n | Evo 2 | GERP | Δ vs GERP | 95% CI (Δ vs GERP) | verdict | ΔFM |
|---|---|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|:--|--:|
| chicken | bird | 308 | 0.954 | [0.913, 0.986] | 0.861 | +0.093 | 274 | 0.862 | 0.725 | +0.137 | [-0.007, 0.293] | ns | +0.1398 |
| human | primate | 3,000 | 0.974 | [0.968, 0.979] | 0.825 | +0.149 | 2,880 | 0.825 | 0.874 | -0.048 | [-0.069, -0.028] | *** | +0.0351 |
| dog | carnivore | 2,496 | 0.970 | [0.952, 0.984] | 0.888 | +0.082 | 2,278 | 0.895 | 0.886 | +0.009 | [-0.029, 0.048] | ns | +0.0514 |
| cat | carnivore | 1,362 | 0.890 | [0.838, 0.938] | 0.848 | +0.043 | 1,037 | 0.859 | 0.846 | +0.013 | [-0.042, 0.069] | ns | +0.0330 |
| horse | perissodactyl | 766 | 0.941 | [0.895, 0.978] | 0.882 | +0.059 | 379 | 0.882 | 0.915 | -0.033 | [-0.110, 0.042] | ns | +0.0415 |
| pig | suid | 396 | 0.863 | [0.753, 0.957] | 0.848 | +0.015 | 168 | 0.814 | 0.751 | +0.063 | [-0.116, 0.233] | ns | +0.0239 |
| cattle | ruminant | 2,067 | 0.974 | [0.958, 0.987] | 0.900 | +0.074 | 1,985 | 0.900 | 0.825 | +0.075 | [0.030, 0.121] | *** | +0.0863 |
| sheep | ruminant | 616 | 0.962 | [0.917, 0.994] | 0.903 | +0.060 | 451 | 0.912 | 0.865 | +0.047 | [-0.045, 0.134] | ns | +0.0482 |
| goat | ruminant | 98 | 0.959 | [0.907, 0.994] | 0.950 | +0.009 | 80 | 0.953 | 0.780 | +0.173 | [0.079, 0.274] | *** | +0.1797 |

*The 8192 bp columns and the GERP-comparable columns are DIFFERENT PANELS with different n. The readout is worth +0.092 AUROC pooled, so the two must never be compared across columns. dFM is the estimator-consistent value (all three AUROC terms from one 5-fold CV logistic), not the deposited decomposition table's.*


## Table D2. What the trust layer delivers, and what it costs

**2a — calibration, macro ECE by binning estimator**

| method | equal-width 10 | equal-width 15 | equal-mass 10 | equal-mass 15 |
|---|--:|--:|--:|--:|
| isotonic transfer (LOSO) | 0.0533 | 0.0557 | 0.0560 | 0.0554 |
| Platt transfer = trivial global sigmoid (LOSO) | 0.0551 | 0.0562 | 0.0553 | 0.0621 |
| in-species oracle (unattainable bound) | 0.0157 | 0.0176 | 0.0201 | 0.0226 |

**2b — the headline is a NULL: isotonic transfer vs a trivial 2-parameter sigmoid**

| estimator | Δ ECE | 95% CI | verdict |
|---|--:|:--|:--|
| width10 | -0.0019 | [-0.0086, 0.0003] | indistinguishable |
| width15 | -0.0004 | [-0.0076, 0.0021] | indistinguishable |
| mass10 | +0.0007 | [-0.0073, 0.0042] | indistinguishable |
| mass15 | -0.0067 | [-0.0140, -0.0019] | isotonic transfer better |

**2c — abstention (honest LOSO, macro error)**

| coverage | 1.00 | 0.95 | 0.90 | 0.85 | 0.80 | 0.75 | 0.70 | 0.65 | 0.60 | 0.55 | 0.50 | 0.45 | 0.40 | 0.35 | 0.30 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| macro error | 0.0577 | 0.0450 | 0.0399 | 0.0376 | 0.0377 | 0.0378 | 0.0388 | 0.0394 | 0.0410 | 0.0426 | 0.0449 | 0.0458 | 0.0436 | 0.0428 | 0.0412 |
| random control | 0.0577 | 0.0588 | 0.0577 | 0.0557 | 0.0562 | 0.0568 | 0.0566 | 0.0561 | 0.0550 | 0.0541 | 0.0540 | 0.0531 | 0.0547 | 0.0569 | 0.0508 |

**2d — conformal: report the Mondrian arm**

| species | n | positive-class coverage (marginal) | abstains (marginal) | positive-class coverage (Mondrian) | abstains (Mondrian) |
|---|--:|--:|--:|--:|--:|
| goat | 99 | 0.222 | 0.00 | 1.000 | 0.51 |
| chicken | 308 | 0.571 | 0.00 | 0.964 | 0.63 |
| pig | 396 | 0.444 | 0.00 | 0.889 | 0.66 |
| sheep | 616 | 0.732 | 0.00 | 0.911 | 0.59 |
| horse | 781 | 0.634 | 0.00 | 0.972 | 0.68 |
| cat | 1,365 | 0.544 | 0.00 | 0.936 | 0.70 |
| cattle | 2,068 | 0.660 | 0.00 | 0.984 | 0.64 |
| dog | 2,497 | 0.648 | 0.00 | 0.943 | 0.62 |
| human | 3,000 | 0.679 | 0.00 | 0.978 | 0.43 |

*Aggregation is MACRO throughout: pooling lets human (n=3,000 at 50% prevalence) dominate a cohort otherwise near 9%. The abstention arm is the honest LOSO one — NOT the in-species oracle, which is an unattainable bound. Any conformal claim is the MONDRIAN arm; the marginal arm under-covers the positive class and is a documented artifact. Platt transfer and the trivial global sigmoid are ONE estimator, bit-identical here under every binning rule and in every species — transferring a Platt map across species IS fitting two global parameters.*


## Table D3. Reach — what conservation can be run on, and what the silence costs

| species | n | negatives scorable | positives scorable | class gap | 95% CI | CI excludes 0 | must-call cost | holes clustered |
|---|--:|--:|--:|--:|:--|:-:|--:|:-:|
| chicken | 308 | 0.882 | 0.964 | +0.082 | [-0.063, 0.134] | no | −0.034 | no |
| human | 3,000 | 0.957 | 0.963 | +0.007 | [-0.007, 0.021] | no | −0.029 | yes |
| dog | 2,497 | 0.911 | 0.921 | +0.009 | [-0.035, 0.040] | no | −0.062 | yes |
| cat | 1,365 | 0.742 | 0.936 | +0.194 | [0.132, 0.234] | yes | −0.106 | no |
| horse | 781 | 0.441 | 0.930 | +0.489 | [0.397, 0.543] | yes | −0.245 | no |
| pig | 396 | 0.392 | 0.750 | +0.358 | [0.190, 0.481] | yes | −0.177 | no |
| cattle | 2,068 | 0.956 | 1.000 | +0.044 | [0.022, 0.054] | yes | −0.014 | yes |
| sheep | 616 | 0.732 | 0.732 | +0.000 | [-0.133, 0.105] | no | −0.169 | no |
| goat | 99 | 0.800 | 0.889 | +0.089 | [-0.242, 0.220] | no | −0.081 | no |

CI excludes zero in **4 of 9**. Consequence-matched control: significant in **0 of 5** matchable species; goat, chicken, pig, sheep have no matchable stratum.

*Missingness in the GERP tracks is NaN, not null — notna() reports 100% reach and deletes this entire result. 'Must-call cost' is the AUROC lost when every variant must be answered, from the closed-form pairwise abstention null (a pair touching a no-call contributes 0.5); it is NOT median imputation. THE CONFOUND IS NOT RESOLVED: matched on consequence category no species keeps a positive gap separable from zero, and it reverses in dog.*


## Table D4. Baseline suite — conservation, a peer DNA-LM, a protein LM, and supervised models

Over the **9,532** variants both Evo 2 and conservation can score it is a tie: Evo 2 **0.8805** vs GERP **0.8780**. Nucleotide Transformer scores **1,598/1,598** of the GERP-blind variants — full reach is a DNA-LM class property, not an Evo 2 one.

| species | n | Evo 2 (covered) | GERP (covered) | GERP (must answer) | Evo 2 (full panel) |
|---|--:|--:|--:|--:|--:|
| chicken | 308 | 0.8623 | 0.7251 | 0.6915 | 0.8608 |
| human | 3,000 | 0.8253 | 0.8737 | 0.8444 | 0.8249 |
| dog | 2,497 | 0.8953 | 0.8858 | 0.8238 | 0.8878 |
| cat | 1,365 | 0.8594 | 0.8460 | 0.7403 | 0.8465 |
| horse | 781 | 0.8817 | 0.9148 | 0.6700 | 0.8805 |
| pig | 396 | 0.8138 | 0.7509 | 0.5737 | 0.8481 |
| cattle | 2,068 | 0.8999 | 0.8250 | 0.8107 | 0.9000 |
| sheep | 616 | 0.9117 | 0.8648 | 0.6955 | 0.9028 |
| goat | 99 | 0.9531 | 0.7804 | 0.6994 | 0.9506 |

**Supervised baselines (point estimates only)**

| species | k-mer GBM (within) | CNN (within) | CNN (species held out) | Evo 2 zero-shot |
|---|--:|--:|--:|--:|
| chicken | 0.677 | 0.745 | 0.651 | 0.861 |
| human | 0.635 | 0.545 | 0.482 | 0.825 |
| dog | 0.716 | 0.711 | 0.748 | 0.888 |
| cat | 0.713 | 0.721 | 0.755 | 0.848 |
| horse | 0.789 | 0.792 | 0.663 | 0.882 |
| pig | 0.683 | 0.763 | 0.821 | 0.848 |
| cattle | 0.807 | 0.787 | 0.761 | 0.900 |
| sheep | 0.868 | 0.840 | 0.855 | 0.903 |

**Protein LM**: only **3 of 9** annotated panels can host one, and ESM-2 is scored at usable size on **1**. On human (n = 912): ESM-2 0.887 vs Evo 2 0.838, Δ **+0.049** [+0.019, +0.079].

*Every GERP AUROC here is measured on the subset GERP can score — see Table D3. The 'abstain' column is the same AUROC once unscorable variants must be answered. Supervised rows are point estimates only: the baseline scripts retain no per-variant predictions, so paired intervals are not computable from the deposit. Goat has no CNN run.*
