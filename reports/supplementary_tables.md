# Supplementary tables, exported from the submitted Additional file 1

## Table S34. Evo 2-1B, summed over a 1,001-bp window, against purpose-built splice predictors, by variant class

| Variant class | n | disruptive | Evo 2-1B | 95% CI | SpliceAI | Pangolin | Evo 2 − SpliceAI | p (FWER) |
|---|---|---|---|---|---|---|---|---|
| Essential Splice | 246 | 238 | 0.853 | [0.788, 0.911] | 0.968 | 0.928 | −0.116 | 0.01006 |
| Exon Near Junction | 434 | 148 | 0.554 | [0.496, 0.612] | 0.794 | 0.800 | −0.239 | < 0.00002 |
| Intron Near Junction | 643 | 188 | 0.721 | [0.674, 0.768] | 0.904 | 0.917 | −0.182 | < 0.00002 |
| Proximal Intron | 1,170 | 114 | 0.535 | [0.474, 0.596] | 0.656 | 0.656 | −0.121 | 0.00618 |
| Deep Exon | 1,419 | 372 | 0.471 | [0.438, 0.504] | 0.652 | 0.665 | −0.181 | < 0.00002 |

Table S34 notes. 3,912 variants with splice-disruptive calls, measured in cells except for 82 essential-splice calls admitted by rule (Note S35). The Evo 2 column is Evo 2-1B with its log-likelihood summed over a 1,001-bp window, the only readout at which all six assays were scored; on the BRCA1 assay Evo 2-40B at the 8,192-bp readout reads 0.974 (Note S65). SpliceAI and Pangolin use roughly 10,000 nt of flanking sequence, so the columns are not matched on input context (Methods). Intervals on Evo 2 are within-class stratified percentile bootstraps at B = 200,000 (positives and negatives resampled separately at their observed class sizes, so no replicate can lose a class; Essential Splice carries eight negatives). The p-value is family-wise across the five classes by Westfall–Young max-T permutation over 50,000 draws, so no class is read on its own; the smallest p the test can resolve is 1/50,001 = 0.00002, and the three cells shown as "< 0.00002" sit on that floor and are bounds rather than estimates. Rows are ordered by class size, smallest first; Fig. 8a draws the same five classes in a different, positional order. Class negatives are n minus the disruptive count: 8, 286, 455, 1,056 and 1,047 in the row order above. Differences are computed on the unrounded AUROCs and can disagree by 0.001 with the subtraction of the two rounded columns beside them. Pooled over all 3,912 variants Evo 2 reads 0.621 against SpliceAI 0.799 and Pangolin 0.803, a paired difference of −0.1771 (95% CI [−0.2008, −0.1534], DeLong on the shared variants); that pooled comparison is not covered by the family-wise correction.

## Table S1. The readout effect on identical variants

| Species | n | AUROC 8,192-bp mean-LL | 95% CI | AUROC 1,001-bp single-token (same variants) |
|---|---|---|---|---|
| Human | 3,000 | 0.974 | [0.968, 0.979] | 0.825 |
| Cattle | 2,067 | 0.974 | [0.958, 0.987] | 0.900 |
| Dog | 2,496 | 0.970 | [0.952, 0.984] | 0.888 |
| Sheep | 616 | 0.962 | [0.917, 0.994] | 0.903 |
| Goat | 98 | 0.959 | [0.907, 0.994] | 0.950 |
| Pig | 396 | 0.863 | [0.753, 0.957] | 0.848 |
| Horse | 766 | 0.941 | [0.895, 0.978] | 0.883 |
| Cat | 1,362 | 0.890 | [0.838, 0.938] | 0.848 |
| Chicken | 308 | 0.954 | [0.913, 0.986] | 0.861 |

Both columns are computed on the same variants, every variant carrying an 8,192-bp mean-log-likelihood score, 11,109 of the 11,130 in the panel, so the two readouts are matched variant-for-variant and isolate the readout effect: +0.065 AUROC by the species mean, +0.092 pooled [+0.083, +0.101]. Table 2 is computed on the full 11,130-variant panel and this table on the 11,109 carrying an 8,192-bp score, so n differs in the five species that lose variants at 8,192 bp (horse 15, cat 3, and one each in goat, cattle and dog; Note S27), and the 1,001-bp AUROC shifts with it at three decimals in goat, horse and cat. That panel difference is the whole of the residual, and the readout effect above is computed on the matched subset. The intervals in this table are variant-level and are not cluster-robust. Goat's [0.907, 0.994] is therefore the unclustered interval, and it is printed here for completeness only: goat's nine positives occupy two loci, eight of them within a 570-bp span, giving 1.2 effective independent positions, which is why Table 2 declines to print an interval for goat at all. Read goat's AUROC as a descriptive point estimate, as Table 2 does.

## Table S2. Reach: what the conservation baseline can be run on

| Species | n | positive scorable | negative scorable | class gap | deployment AUROC penalty |
|---|---|---|---|---|---|
| Goat | 99 | 0.889 | 0.800 | +0.089 | −0.081 |
| Chicken | 308 | 0.964 | 0.882 | +0.082 | −0.034 |
| Pig | 396 | 0.750 | 0.392 | +0.358 | −0.177 |
| Sheep | 616 | 0.732 | 0.732 | +0.000 | −0.169 |
| Horse | 781 | 0.930 | 0.441 | +0.489 | −0.245 |
| Cat | 1,365 | 0.936 | 0.742 | +0.194 | −0.106 |
| Cattle | 2,068 | 1.000 | 0.956 | +0.044 | −0.014 |
| Dog | 2,497 | 0.921 | 0.911 | +0.009 | −0.062 |
| Human | 3,000 | 0.963 | 0.957 | +0.007 | −0.029 |

Conservation scores are undefined where the alignment or track is unavailable, and missingness is marginally associated with class in these panels; the consequence-matched analysis does not establish its cause. Every per-species conservation AUROC computed this way is therefore computed on a class-skewed subset of the panel it is drawn from, whose full-panel value is unidentified without an assumption about the unscored variants; we establish that within our own nine panels and do not survey the published literature. The missingness is also spatially structured rather than random: a Wald–Wolfowitz runs test on the no-call pattern rejects randomly scattered no-calls in the three largest panels (human z = −13.87 negative and −13.94 positive, dog −10.86, cattle −7.31). That cuts both ways: it is independent evidence that the holes are structural, and it means the per-variant reach intervals of the Figure S5 legend assume an independence of missingness that the runs test rejects, so they understate uncertainty in exactly those three panels. The test only detects clustering on the scale of the spacing between sampled variants, which is 0.41 Mb in human, 0.58 Mb in dog and 0.97 Mb in cattle, and 2.42 Mb in pig and 2.43 Mb in horse. It has no power against kilobase-scale alignment gaps, so non-rejection in a panel is not evidence that the no-calls are unclustered. The penalty column above charges every ranking pair that touches a no-call at one half, following the full evaluation mode of the Critical Assessment of Functional Annotation (CAFA) in charging for silence, though CAFA itself assigns zero. That is a scoring convention and not a forecast of what conservation would emit if it were forced to answer, so we give the alternatives here. Under the paper's rule the penalty is −0.1019 by the unweighted species mean and −0.2448 in horse. Giving each no-call the conservation score's own median value and then taking AUROC over the whole panel gives −0.0235 and −0.0100. Drawing each no-call from the conservation score's own score distribution gives −0.0398 and −0.0650. The two alternatives are milder because the scored negatives sit at low conservation and the scored positives at high, so a filled-in value inherits information from that arrangement. We keep the first rule throughout this paper, and a reader who prefers a different one should take these figures in place of the tabulated ones.

## Table S3. Baseline suite

| Species | n | 1D-CNN | k-mer | Evo 2-40B (zero-shot, 1,001-bp single token) | CNN leave-one-species-out |
|---|---|---|---|---|---|
| Chicken | 308 | 0.745 | 0.677 | 0.861 | 0.651 |
| Pig | 396 | 0.763 | 0.683 | 0.848 | 0.821 |
| Sheep | 616 | 0.840 | 0.868 | 0.903 | 0.855 |
| Horse | 766 | 0.792 | 0.789 | 0.883 | 0.663 |
| Cat | 1,362 | 0.721 | 0.713 | 0.848 | 0.755 |
| Cattle | 2,067 | 0.787 | 0.807 | 0.900 | 0.761 |
| Dog | 2,496 | 0.711 | 0.716 | 0.888 | 0.748 |
| Human | 3,000 | 0.545 | 0.635 | 0.825 | 0.482 |

Supervised models trained on the labels themselves. The n column is the panel each baseline was trained and evaluated on, which for four species is marginally smaller than the full panel in Table 2 (horse 766 against 781, cat 1,362 against 1,365, cattle 2,067 against 2,068, dog 2,496 against 2,497), because a baseline needs a window every model can score. The Evo 2 column is computed on this same reduced set, not carried over from Table 2, so the comparison is like-for-like. Point estimates only, without paired intervals. Goat's 99 variants are too few to fit the supervised baselines (Note S33).
**Table S3 (continued)**

| Species | n | n pos | ESM-2 | Evo 2-40B | GERP (n scorable) | GERP covered | GERP must-answer |
|---|---|---|---|---|---|---|---|
| Human | 912 | 652 | 0.887 | 0.838 | 881 | 0.860 | 0.835 |

Protein language model, evaluable only where both classes carry at least ten missense variants, three of the nine panels, and scored at usable size only in human (Note S5).
The n column applies to ESM-2 and Evo 2-40B. GERP is measured on the subset it can score, given separately; its must-answer column charges it for the rest at chance, the same accounting used in Figure S5.

## Table S4. Whole-atlas and eQTL discrimination across model scale

| Model | whole-atlas macro AUROC (n = 11,109) | eQTL AUROC | 95% CI | n (eQTL) |
|---|---|---|---|---|
| Evo 2-1B | 0.868 | 0.492 | [0.467, 0.518] | 2,000 |
| Evo 2-7B | 0.928 | 0.488 | [0.464, 0.514] | 2,000 |
| Evo 2-40B | 0.943 | 0.498 | [0.473, 0.523] | 2,000 |

The whole-atlas column is the nine-species mean AUROC at the 8,192-bp readout on all 11,109 variants, not a coding subset; its steps have paired species-mean t intervals that include zero, +0.061 [−0.016, +0.137] from 1B to 7B and +0.015 [−0.003, +0.032] from 7B to 40B. Without human, whose 1B scores are anomalous and whose 1B-to-7B step is +0.324, the eight OMIA species read 0.896, 0.924 and 0.939 and step by +0.028 [+0.014, +0.042] from 1B to 7B and by +0.015 [−0.005, +0.035] from 7B to 40B (Note S15). The eQTL column is the 2,000-variant consequence-matched subpanel, 1,000 of each class, on which no rung moves from chance. Every cell of both columns is read at the 8,192-bp mean-log-likelihood readout, scored at each checkpoint by the scorer that loads it whole (Note S30).
**Table S4 (continued)**

| Scorer | AUROC | 95% CI | n |
|---|---|---|---|
| Evo 2-40B | 0.498 | [0.473, 0.523] | 2,000 |
| NT-500M | 0.486 | [0.460, 0.513] | 2,000 |
| GERP (exact) | 0.494 | [0.455, 0.532] | 884 |
| GERP (mean25) | 0.473 | [0.437, 0.510] | 1,015 |
| GERP (absmax25) | 0.532 | [0.497, 0.569] | 1,015 |

Every sequence-based scorer collapses together on the same panel. GERP is shown as one row per flavour because its reach on this panel is flavour-dependent (n varies with how many variants carry a finite score); each row is a genuine estimator with its own bootstrap interval, in place of an uninterpretable average across flavours.

## Table S5. Equivalence testing, Evo 2 versus GERP on the co-scorable panel

| Bootstrap | 90% CI | verdict at SESOI 0.02 |
|---|---|---|
| species-clustered (reported) | [−0.0237, +0.0555] | equivalence NOT established |
| variant-level (contrast only) | [−0.0095, +0.0145] | equivalent |

Observed pooled difference +0.0024 at the 1,001-bp single-token readout, on n = 9,532 over 9 species. The unit of analysis decides the answer, and we report the species-clustered interval because species are this study's unit of generalisation.
**Table S5 (continued)**

| SESOI | p (TOST) | equivalent? |
|---|---|---|
| 0.01 | 0.3928 | no |
| 0.02 | 0.2845 | no |
| 0.05 | 0.0750 | no |

## Table S6. Label-free validation against the site-frequency spectrum

| MAF bin | n | mean MAF | mean deleteriousness |
|---|---|---|---|
| ultra-rare | 5,607 | 0.00062 | 0.772 |
| rare | 5,624 | 0.00505 | 0.511 |
| low | 5,581 | 0.02436 | 0.319 |
| common | 5,596 | 0.12139 | 0.201 |
| major | 5,598 | 0.38986 | 0.178 |

Cattle (ARS-UCD1.2), 28,006 variants. Mean Evo 2-40B deleteriousness at the 1,001-bp single-token readout, by minor-allele-frequency bin; Spearman rs = −0.099 [−0.111, −0.088], p = 2.9 × 10−62. The claim is the monotone gradient at this n, not the magnitude of rs.
Fold change, major-allele bin to ultra-rare bin: 4.33.
**Table S6 (continued)**

| phyloP quintile | phyloP range | n | Spearman rs | 95% CI | p |
|---|---|---|---|---|---|
| 1 | −15.95 to −1.28 | 5,447 | −0.018 | [−0.044, +0.011] | 0.194 |
| 2 | −1.28 to −0.38 | 5,442 | −0.054 | [−0.081, −0.028] | 7.79 × 10−5 |
| 3 | −0.38 to 0.03 | 5,381 | −0.092 | [−0.120, −0.065] | 1.13 × 10−11 |
| 4 | 0.03 to 0.39 | 5,505 | −0.109 | [−0.138, −0.083] | 4.06 × 10−16 |
| 5 | 0.39 to 9.85 | 5,459 | −0.118 | [−0.143, −0.093] | 2.25 × 10−18 |

Conditioned on conservation (phyloP quintiles). The frequency–deleteriousness relationship survives at fixed conservation in 4 of 5 quintiles, and its interval crosses zero exactly where conservation itself has no signal, with no disease label anywhere in the analysis.
Strata with negative rs: 5 of 5; strata whose CI excludes zero: 4.

## Table S7. Evo 2 versus GERP is readout-dependent (co-scorable variants)

| Species | n (co-scorable) | Evo 2 @1,001-bp | Evo 2 @8,192-bp | GERP | Δ (8,192 − GERP) |
|---|---|---|---|---|---|
| Goat | 80 | 0.953 | 0.958 | 0.780 | +0.178 |
| Chicken | 274 | 0.862 | 0.956 | 0.725 | +0.231 |
| Pig | 168 | 0.814 | 0.826 | 0.751 | +0.075 |
| Sheep | 451 | 0.912 | 0.950 | 0.865 | +0.086 |
| Horse | 379 | 0.882 | 0.938 | 0.915 | +0.023 |
| Cat | 1,034 | 0.860 | 0.902 | 0.856 | +0.045 |
| Cattle | 1,985 | 0.900 | 0.974 | 0.825 | +0.149 |
| Dog | 2,278 | 0.895 | 0.974 | 0.886 | +0.089 |
| Human | 2,880 | 0.825 | 0.974 | 0.874 | +0.100 |
| **Macro** |  | 0.878 | 0.939 | 0.831 | +0.108 |

On the variants both methods can score, Evo 2 scored at the 1,001-bp single-token readout and at the 8,192-bp mean-log-likelihood readout, against the same fixed GERP baseline, per species. Moving Evo 2 from the single-token to the mean-log-likelihood readout raises its macro margin over GERP from +0.047 (leading 7 of 9 species) to +0.108 (9 of 9), while GERP is unchanged, so the near-parity at the single-token readout is an artefact of the readout, not the panel. The GERP column here is restricted to variants that also carry an 8,192-bp score, so it differs slightly from the GERP column of Table 2, which uses every variant GERP can score; for cat the two sets are 1,034 and 1,037 variants and GERP reads 0.856 against 0.846. On the 8,192-bp difference: macro +0.108, 95% CI [+0.058, +0.159] by the t interval on eight degrees of freedom and [+0.060, +0.162] by the paired species-clustered bootstrap. Per-species DeLong intervals for the 8,192-bp difference are given in Note S11. Six of the nine exclude zero, so the 9 of 9 in this table is a sign count over point estimates.

## Table S8. phyloP versus GERP where a phyloP track is available

| Species | n (matched) | GERP | phyloP | phyloP − GERP | Evo 2 @8,192 | Evo 2 @8,192 − phyloP | Evo 2 @1,001 | Evo 2 @1,001 − phyloP |
|---|---|---|---|---|---|---|---|---|
| Cattle | 1,921 | 0.828 | 0.931 | +0.103 | 0.978 | +0.047 | 0.909 | −0.022 |
| Chicken | 274 | 0.725 | 0.793 | +0.068 | 0.956 | +0.163 | 0.862 | +0.069 |
| Human | 2,880 | 0.874 | 0.925 | +0.051 | 0.974 | +0.049 | 0.825 | −0.100 |

GERP is the sole cross-species conservation baseline because it is the only per-base score available for all nine assemblies (Methods). For chicken (galGal6/GRCg6a UCSC phyloP77way) and human (hg38/GRCh38 UCSC phyloP100way), which carry a genuine UCSC phyloP track, and cattle (bosTau9), whose phyloP comes from Zenodo record 13332541 (Roslin) over a Cactus 241-way alignment rather than from UCSC, so that row rests on a different alignment and species set from the two beside it, we recomputed conservation discrimination from phyloP queried at each variant position. All four scores above are evaluated on exactly the same variants (the intersection of the 8,192-bp-scored, GERP-scorable and phyloP-scorable sets; cattle n = 1,921), so every comparison is like-for-like. Two things follow. First, phyloP is a stronger baseline than GERP in all three species, so every Evo 2-versus-conservation margin reported against GERP is an upper bound rather than a conservative one; GERP is kept as the primary comparator for uniformity across all nine assemblies, not for strength. Second, and more important, Evo 2 at the reference-implementation 8,192-bp readout beats even phyloP in all three, so the readout head-to-head (Table S7) is robust to using the stronger baseline; at the 1,001-bp readout, by contrast, phyloP beats Evo 2 in two of three, reinforcing that the single-token readout understates the model.
phyloP was queried with pybigtools from genuine bigWig tracks; chromosome coordinates were mapped through the UCSC chromAlias, and no positive variant was dropped for want of a coordinate in any species. Per-variant phyloP is deposited for human only (analyses/results/human_phylop_pervariant.parquet, Additional file 2), from which the human row recomputes; for cattle and chicken the cross-check kept only its summaries (reports/phylop_crosscheck.json, Additional file 2), so those two rows cannot be recomputed from the deposit.

## Table S9. Decomposing the readout advantage into left context with scorer, and aggregation

| Species | n | positives | single-token, 1 kb | single-token, 8 kb | window-mean, central 1 kb | window-mean, 8 kb | left context and scorer | aggregation |
|---|---|---|---|---|---|---|---|---|
| Goat | 98 | 9 | 0.950 | 0.951 | 0.948 | 0.955 | +0.001 | +0.004 |
| Chicken | 228 | 28 | 0.863 | 0.916 | 0.952 | 0.947 | +0.054 | +0.031 |
| Pig | 236 | 36 | 0.843 | 0.866 | 0.878 | 0.861 | +0.023 | −0.005 |
| Sheep | 256 | 56 | 0.903 | 0.957 | 0.965 | 0.961 | +0.054 | +0.004 |
| Horse | 271 | 71 | 0.884 | 0.899 | 0.957 | 0.945 | +0.015 | +0.046 |
| Cat | 322 | 122 | 0.840 | 0.877 | 0.895 | 0.887 | +0.037 | +0.010 |
| Cattle | 388 | 188 | 0.893 | 0.931 | 0.969 | 0.973 | +0.038 | +0.042 |
| Dog | 400 | 200 | 0.890 | 0.929 | 0.979 | 0.979 | +0.039 | +0.050 |
| Human | 400 | 200 | 0.814 | 0.875 | 0.974 | 0.972 | +0.061 | +0.097 |

Both readouts are accounting choices, not results. The +0.065 species-mean advantage of the 8,192-bp mean-log-likelihood over the 1,001-bp single-token variant-delta (Fig. 10a) confounds two choices: a longer model context and averaging surprise across the window. We separate them on 2,599 atlas variants (at most 200 positives and 200 negatives per species), reading three quantities from a single 8,192-bp forward pass per sequence, which hold the model and context fixed and vary only the aggregation scope, and a fourth from the separate 1,001-bp pass. The left-context-and-scorer term is the gain of the single-token readout as its left context grows from ~0.5 kb (the 1,001-bp window) to ~4 kb, a step that also moves from the streamed scorer to the resident one; the aggregation term is the further gain of the full-window mean over that single position.
**Table S9 (continued)**

|  | left context and scorer term | aggregation term | total |
|---|---|---|---|
| species mean (± SE) | +0.0359 ± 0.0066 | +0.0309 ± 0.0107 | +0.0668 ± 0.0149 |
| pooled (95% CI) | +0.0377 [+0.0286, +0.0469] | +0.0374 [+0.0267, +0.0483] | +0.0751 [+0.0611, +0.0898] |

The total recovered here (+0.0668 species mean) reproduces the readout effect of Fig. 10a (+0.065), evidence that the effect size is representative even though the subset is not class-balanced: it retains every positive in seven species and keeps at most 200 positives and 200 negatives in each, so dog and human are held at 200 positives and the per-species positive fraction runs above the ~9% of the full panels, reaching 50% in dog (human's full panel is already 1:1). The split into context and aggregation is path-dependent. Taking context first (single-token readout, window 1 kb to 8 kb) then aggregation (single to window-mean at 8 kb) gives context +0.0359 and aggregation +0.0309; taking aggregation first (single to window-mean at 1 kb) then context (window-mean 1 kb to 8 kb) gives aggregation +0.0708 and context −0.0040. The total minus the two first steps, each measured from the single-token 1-kb cell, is −0.0399. That residual is not a factorial interaction and does not show that the two factors overlap: the fourth cell of a complete two-by-two, a window mean at 1,001-bp conditioning context, is not part of the design, so the aggregation-first path substitutes the central-1-kb mean of the 8,192-bp pass, which moves conditioning context as well as aggregation. The +0.0349 gap between the two single-factor conditions is likewise a difference between conditions rather than an interaction term. A 'roughly equal halves' reading holds only under the first ordering. What is robust across orderings is that the averaged window can stop at 1 kb: the central 1-kb window-mean exceeds the full 8-kb window-mean in six of nine species (dog's two values tie at the printed precision) (species mean 0.946 against 0.942), so extending the averaged window past 1 kb adds nothing. As a correctness check the full-window mean recomputed here reproduces the deposited 8,192-bp mean-log-likelihood to within 0.0003 pooled AUROC.

## Table S10. Precision-recall (AUPRC) at the 8,192-bp readout

| Species | n | base rate | AUROC | AUPRC | AUPRC / base rate |
|---|---|---|---|---|---|
| Goat | 98 | 0.092 | 0.959 | 0.735 | 8.0× |
| Chicken | 308 | 0.091 | 0.954 | 0.844 | 9.3× |
| Pig | 396 | 0.091 | 0.863 | 0.809 | 8.9× |
| Sheep | 616 | 0.091 | 0.962 | 0.886 | 9.7× |
| Horse | 766 | 0.093 | 0.941 | 0.822 | 8.9× |
| Cat | 1,362 | 0.090 | 0.890 | 0.849 | 9.5× |
| Cattle | 2,067 | 0.091 | 0.974 | 0.862 | 9.5× |
| Dog | 2,496 | 0.091 | 0.970 | 0.911 | 10.0× |
| Human | 3,000 | 0.500 | 0.974 | 0.971 | 1.9× |
| **Macro** |  | 0.137 |  | 0.854 | 6.3× |
| **Pooled** | 11,109 | 0.201 | 0.973 | 0.940 |  |

Average precision (AUPRC) alongside AUROC per species, with the no-skill baseline (the base rate). AUROC is optimistic under the 9% class imbalance and for a screening use case; AUPRC is the decision-relevant summary. Macro AUPRC is 0.854 against a 0.137 base rate (6.3×), pooled 0.940.
AUPRC stays well above the base rate in every species, so the discrimination at this readout is not a base-rate artefact. The one setting where precision-recall collapses is the matched control of the concurrent cross-species work cited in the main text (AUPRC 0.166 against a ~9% base rate, versus its 0.717 AUROC), which is why we quote both metrics in Note S6. Both figures are that paper's, read from its text and held by no artefact here; its 0.717 is not the 0.717 this paper reports for HAL's covered AUROC.

## Table S11. Reach and must-answer accuracy for five human scores on ClinVar

| score | reach (all) | reach (pos.) | reach (neg.) | covered | must-answer | penalty | missingness alone |
|---|---|---|---|---|---|---|---|
| phastCons | 0.99997 | 0.99999 | 0.99996 | 0.8443 | 0.8443 | 0.00002 | 0.5000 |
| phyloP | 0.99997 | 0.99999 | 0.99996 | 0.9217 | 0.9217 | 0.00002 | 0.5000 |
| CADD | 0.99879 | 0.99935 | 0.99872 | 0.9914 | 0.9904 | 0.00095 | 0.5003 |
| REVEL | 0.17642 | 0.37313 | 0.14985 | 0.9575 | 0.5256 | 0.43192 | 0.6116 |
| AlphaMissense | 0.13493 | 0.33665 | 0.10768 | 0.9631 | 0.5168 | 0.44635 | 0.6145 |

1,434,335 single-nucleotide variants at a review status of one star or better (170,680 pathogenic and 1,263,655 benign, the positive and negative classes of this panel). Covered is AUROC over the variants each score can score; must-answer charges every pair touching an unscorable variant at one half, a CAFA-style full-evaluation convention (CAFA itself assigns zero). Missingness alone is the AUROC of the indicator of whether a value was returned, discarding the values. For a score that answers everywhere the indicator never varies and its midrank AUROC is exactly 0.5, as roc_auc_score returns; a value of 0.5000 can also mean reach that is near-complete and equal in both classes.
The two missense-only scores carry a missingness AUROC above their own must-answer AUROC. That is a statement about this panel's composition and not about either method: reach tracks consequence class here, and ClinVar's pathogenic variants are missense-enriched, so the indicator is largely a missense indicator. Table S12 narrows the confound by restricting to the variant classes dbNSFP annotates, where no predictor has a missingness AUROC above its own must-answer AUROC under the signed test, although 30 of the 48 with a defined missingness AUROC do read orientation-free. That comparison is descriptive; the test of the values, what ranking by the indicator and then by value adds to the indicator, excludes zero for all 49 (Note S61).

## Table S12. Reach and its class dependence for 49 dbNSFP predictors

| reach band | n predictors | median reach | median covered | median must-answer | median penalty |
|---|---|---|---|---|---|
| reach >99.9% | 5 | 1.0000 | 0.7866 | 0.7866 | 0.0000 |
| reach 70 to 99.9% | 13 | 0.9211 | 0.9218 | 0.8312 | 0.0486 |
| reach <70% | 31 | 0.5885 | 0.9375 | 0.6226 | 0.3149 |
| missense: reach >99.9% | 6 | 1.0000 | 0.8507 | 0.8507 | 0.0000 |
| missense: reach 70 to 99.9% | 43 | 0.9362 | 0.9410 | 0.8706 | 0.0479 |
| missense: reach <70% | 0 | — | — | — | — |

These 49 predictors are audited on the 328,328 nonsynonymous and splice-site variants dbNSFP annotates (166,149 pathogenic, 162,179 benign), of which 202,643 are missense, across 15,606 genes; the class-gap counts below use a bootstrap resampling genes rather than variants (2,000 draws), and per-predictor intervals are in the deposited artefact. Predictors are grouped by reach into three disjoint bands and medians are taken over the group. Cut instead at nested reach thresholds, the same 49 give counts that overlap rather than partition: the 18 predictors reaching 75% of the panel carry a median penalty of 0.0217, the 12 reaching 90% carry 0.0009 and the 8 reaching 99% carry 0.0000. Rows marked missense repeat the accounting on the 202,643 missense substitutions (15,389 genes), with the same 2,000-draw gene bootstrap. The bands are cut on reach, not on ρ as the figures' pair-coverage regimes are (Figure 1), so on the missense subset GERP_92_mammals sits in the 70 to 99.9% band at ρ = 0.447 (Figure 4).
**Table S12 (continued)**

| quantity | value |
|---|---|
| class gap with a gene-clustered interval excluding zero | 44 of 49 |
| ... and a gap of at least 0.02 in magnitude | 39 of 49 |
| missingness alone outscoring must-answer, orientation-free | 30 of 48 |
| ... the same test taking the sign as given | 0 of 48 |
| predictors with a negative class gap, for which the signed test cannot fire | 34 of 49 |
| complete-reach predictors returning a penalty of exactly 0.0000 | 1 of 5 |
| missense: class gap with a gene-clustered interval excluding zero | 39 of 49 |
| missense: ... and a gap of at least 0.02 in magnitude | 25 of 49 |
| missense: ... and a gap of at least 0.05 in magnitude | 14 of 49 |
| missense: pairwise comparisons identified | 429 of 1,176 |
| missense: ... of which neither side is a conservation track | 180 of 429 |
| pairs feasible with both covered AUROCs at one half or above | 506 of 1,176 (1,174 missense) |
| comparisons identified if unscored variants are no easier than scored | 131 of 1,176 (488 missense) |
| comparisons identified at two review stars or more | 100 of 1,176 (566 missense) |
| values adding to reach: lexicographic gain, gene-clustered interval excluding zero | 49 of 49 (49 of 49 missense) |
| gene × consequence prior, out of fold, against the best must-answer AUROC | 0.974 against 0.968 |
| covered AUROC below the out-of-fold gene prior on its own covered set | 18 of 49 (27 against gene × consequence) |

Restricting to the classes dbNSFP annotates reverses the direction of the reach bias rather than removing it: here the missense-oriented predictors reach the negatives far more than the positives, where on the whole ClinVar panel they reach the positives more. Because the pooled missingness AUROC is the identity 0.5 + gap/2, a negative gap puts it below one half and the signed test cannot fire. Read orientation-free, the indicator still exceeds most of these predictors' must-answer AUROCs, which says that their reach tracks the label here, not that their values carry less information: the must-answer rule scores every pair touching an unscored variant at one half and so ignores what the indicator knows. The test of the values keeps that information: ranking by the indicator and then by value beats the indicator by ρ(A_cov − 0.5), and the gene-resampled interval on that gain excludes zero for all 49 predictors, on the whole panel and on the missense subset. This panel relocates the composition confound instead of clearing it (Note S28). What it does establish is the reach accounting itself. We draw no ranking of these predictors from this panel, for the reason set out in Note S28.

## Table S13. OMIA trait alleles: which positives are not disease variants, and what removing them does

| Species | n | positives | trait positives | AUROC (full panel) | AUROC (disease only) | Δ |
|---|---|---|---|---|---|---|
| Goat | 98 | 9 | 8 | 0.959 | — | — |
| Chicken | 308 | 28 | 20 | 0.954 | 1.000 | +0.0464 |
| Pig | 396 | 36 | 13 | 0.863 | 0.952 | +0.0888 |
| Sheep | 616 | 56 | 27 | 0.962 | 0.990 | +0.0276 |
| Horse | 766 | 71 | 44 | 0.941 | 0.949 | +0.0079 |
| Cat | 1,362 | 122 | 38 | 0.890 | 0.971 | +0.0805 |
| Cattle | 2,067 | 188 | 28 | 0.974 | 0.982 | +0.0077 |
| Dog | 2,496 | 227 | 25 | 0.970 | 0.979 | +0.0098 |
| Human | 3,000 | 1,500 | 0 | 0.974 | 0.974 | +0.0000 |
| Mean, eight species |  |  |  | 0.941 | 0.975 | +0.0336 |
| Pooled | 11,109 | 2,237 | 203 | 0.973 | 0.985 | +0.0118 |

OMIA catalogues single-gene traits and disorders together, so 203 of the 737 scored non-human positives are trait alleles: coat and feather colour and type, morphology such as polydactyly and tail length, production traits such as double muscling and fecundity, blood groups, gait, and resistance to infection, including eight of goat's nine positives, which are PRNP alleles associated with resistance to scrapie. The class of each positive is curated from its OMIA phenotype string and deposited with it (atlas_positive_annotations.parquet in Additional file 3), where the rules and every correction to a phenotype-string pattern are stated; a phenotype that joins a trait to a disorder, such as white coat with deafness, counts as a disease, and the two cat CMAH records whose phenotype names only a 2019 blood-typing panel count, like the other CMAH alleles, as blood-group alleles. Removing the trait alleles and rescoring the identical panel at the identical 8,192-bp readout raises discrimination in every OMIA species that keeps at least five disease alleles, so their presence works against the reported values.
Trait positives removed: 203 of the 737 scored OMIA positives (27.5%), with 44 of 71 in horse and 20 of 28 in chicken. The mean over the eight species that keep at least five disease alleles moves from 0.941 to 0.975, and the pooled AUROC from 0.973 to 0.985. Over the same species and variants the impact ordering moves from 0.954 to 0.974 and the coding flag from 0.906 to 0.928, so the trait alleles hold back the annotation baselines as well. Goat keeps one disease allele, so its disease-only AUROC is not estimated.

## Table S14. The consequence gradient is not the model's alone: GERP through the identical analysis

| Consequence class | n positives (GERP-scorable) | Evo 2 (1,001-bp single token) | GERP | Evo 2 − GERP |
|---|---|---|---|---|
| nonsense (stop-gain) | 110 | 0.954 | 0.834 | +0.120 |
| splicing | 67 | 0.772 | 0.900 | −0.128 |
| missense | 310 | 0.896 | 0.879 | +0.017 |
| regulatory | 21 | 0.579 | 0.720 | −0.141 |

The same positive-count-weighted estimator as the Evo 2 gradient, with GERP in place of Evo 2, but not the same variants, and we state that rather than let the table imply it. GERP is defined only where its track returns a finite value, so this arm runs on the GERP-reachable subset: different positive counts in all four classes (nonsense 110 against Evo 2's 112, splicing 67 against 70, missense 310 against 324, regulatory 21 against 18), a different species set (cattle enters every GERP class and carries none in the Evo 2 arm; goat leaves the nonsense class), and a strictly smaller negative pool. The "Evo 2 − GERP" column therefore differences two AUROCs computed on non-nested sets, the same reach mismatch Table S2 quantifies, arising here inside our own control. Read the two columns as two gradients that share a floor on the regulatory class rather than as a difference: conservation also reaches its floor there, so that floor reflects a constraint structure both scores inherit rather than a decay peculiar to the model. The regulatory cell rests on 21 GERP-reachable positives against 18 unmatched ones and is the least well powered of the four.

## Table S15. How much the human figure depends on which benign variants are drawn

| Negative set | n negatives | AUROC | 95% CI |
|---|---|---|---|
| ClinVar B/LB, all | 17,376 | 0.851 | [0.836, 0.866] |
| ClinVar B/LB, coding only | 4,377 | 0.811 | [0.795, 0.826] |
| ClinVar B/LB, noncoding only | 12,999 | 0.865 | [0.850, 0.879] |

The human arm's 1,500 negatives are a matched sample from ClinVar's benign half; every one of them is a Benign/Likely_benign record, so human is the one species whose negatives are clinically adjudicated rather than catalogued population variation. This rescores the same 1,500 positives against a larger and differently composed sample of the same catalogue, at the identical 1,001-bp readout on GRCh38. Both draws are ClinVar B/LB, so what this bounds is sampling and composition sensitivity, not the population-versus-adjudicated question, which the human arm cannot pose.
Against the atlas panel's own 1,500 benign negatives the same positives give 0.825, so the human figure moves by −0.014 (coding-only) to +0.040 (noncoding-only) with the benign draw; the range is asymmetric, the noncoding draw is the larger move, and it is not a ±0.03 band, within a single catalogue.

## Table S16. Splicing benchmark by assay: Evo 2-1B, summed over a 1,001-bp window, against SpliceAI

| assay | n | disruptive | Evo 2 (1B) | SpliceAI | difference |
|---|---|---|---|---|---|
| BRCA1 | 1,386 | 223 | 0.808 | 0.986 | −0.178 |
| FAS exon 6 | 189 | 115 | 0.462 | 0.729 | −0.267 |
| MLH1 | 296 | 160 | 0.692 | 0.966 | −0.274 |
| POU1F1 exon 2 | 941 | 96 | 0.605 | 0.961 | −0.356 |
| RON exon 11 | 598 | 409 | 0.557 | 0.630 | −0.073 |
| WT1 exon 9 | 502 | 57 | 0.372 | 0.912 | −0.540 |

Six assays scored independently, five of them saturation mutagenesis and the sixth the 296 curated MLH1 variants (Note S35). A pooled deficit means little if one exon drives it. All values are the Evo 2-1B checkpoint with its log-likelihood summed over a 1,001-bp window; on the BRCA1 assay Evo 2-40B at the 8,192-bp readout reads 0.974 (Note S65). The n-weighted mean of the six per-assay deficits against SpliceAI is −0.2626, larger than the pooled deficit of −0.177 given with Table S34 because the assays differ in both size and difficulty; dropping MLH1 moves it to −0.2617, and restricting to the four massively parallel splicing assays widens it to −0.3139. Because the panel is six assays in six genes, the assay and not the variant is the cluster unit for any inferential statement here. Treating the six per-assay deficits as the sample gives a mean of −0.281 with a 95% interval of [−0.448, −0.114], a half-width 7.0 times that of the pooled DeLong interval (Table S34) — the same direction of disagreement between closed-form and cluster-aware standard errors that we report on the dbNSFP panel — so we do not read the pooled interval as evidence that the difference is real. The distribution-free statement is the sign count: six of six, exact two-sided sign test p = 0.031. Against Pangolin the six per-assay deficits are likewise all negative, with an n-weighted mean of −0.259.

## Table S17. Reach by variant class, benchmark and background

| predictor | panel | Essential Splice | Exon Near Junction | Intron Near Junction | Proximal Intron | Deep Exon | overall |
|---|---|---|---|---|---|---|---|
| HAL PSI change (abs) | background | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.435 |
| HAL PSI change (abs) | benchmark | 0.024 | 0.933 | 0.000 | 0.000 | 0.887 | 0.427 |
| MMSplice logit PSI change (abs) | background | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MMSplice logit PSI change (abs) | benchmark | 1.000 | 1.000 | 1.000 | 0.862 | 1.000 | 0.959 |
| SPANR zPSI change (abs) | background | 0.989 | 0.991 | 0.991 | 0.989 | 0.976 | 0.984 |
| SPANR zPSI change (abs) | benchmark | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| SQUIRLS score | background | 0.907 | 0.908 | 0.909 | 0.908 | 0.903 | 0.906 |
| SQUIRLS score | benchmark | 0.679 | 0.975 | 0.975 | 0.999 | 1.000 | 0.973 |
| S-Cap sens minimum (rev) | background | 0.982 | 0.203 | 0.983 | 0.464 | 0.224 | 0.389 |
| S-Cap sens minimum (rev) | benchmark | 0.976 | 0.348 | 0.966 | 0.795 | 0.397 | 0.641 |
| SpliceAI delta max (alpha) | background | 0.961 | 0.961 | 0.960 | 0.959 | 0.959 | 0.959 |
| SpliceAI delta max (alpha) | benchmark | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pangolin delta max (abs) | background | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pangolin delta max (abs) | benchmark | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ConSpliceML | background | 0.950 | 0.952 | 0.953 | 0.953 | 0.949 | 0.951 |
| ConSpliceML | benchmark | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Reach is the fraction of variants a predictor returns a value for, and each predictor is named by the benchmark's score column (PSI, percent spliced in). The background is 500,000 simulated SNVs drawn at random from internal protein-coding exons and their 100-bp flanks across 14,577 genes.

## Table S18. Genes a predictor cannot score, and whether predictors share them

| predictor pair | shared | set sizes | Jaccard |
|---|---|---|---|
| SpliceAI delta max (alpha) vs ConSpliceML | 21 | 100 / 78 | 0.134 |
| SPANR zPSI change (abs) vs S-Cap sens minimum (rev) | 1 | 5 / 7 | 0.091 |
| SQUIRLS score vs SpliceAI delta max (alpha) | 16 | 296 / 100 | 0.042 |
| SQUIRLS score vs ConSpliceML | 12 | 296 / 78 | 0.033 |
| SQUIRLS score vs S-Cap sens minimum (rev) | 7 | 296 / 7 | 0.024 |
| SPANR zPSI change (abs) vs SpliceAI delta max (alpha) | 1 | 5 / 100 | 0.010 |
| SPANR zPSI change (abs) vs SQUIRLS score | 1 | 5 / 296 | 0.003 |
| SPANR zPSI change (abs) vs ConSpliceML | 0 | 5 / 78 | 0.000 |
| S-Cap sens minimum (rev) vs SpliceAI delta max (alpha) | 0 | 7 / 100 | 0.000 |
| S-Cap sens minimum (rev) vs ConSpliceML | 0 | 7 / 78 | 0.000 |

Restricted to genes carrying at least 50 background variants.
In this constructed background, 432 genes are unreachable by at least one predictor and none is unreachable by every predictor with holes. A union of answered sets would therefore reduce gene-level zero reach substantially, but it would require a defined score-combination rule and a new evaluation of class-dependent selection.

## Table S19. ClinVar consequence rungs at Evo 2-1B: discrimination, power and multiplicity

| consequence | positive | negative | AUROC | 95% CI | MDE | p (FWER) |
|---|---|---|---|---|---|---|
| 3 prime UTR variant | 54 | 250 | 0.7023 | [0.6143, 0.7851] | 0.122 | < 0.00002 |
| 5 prime UTR variant | 160 | 250 | 0.6497 | [0.5938, 0.7043] | 0.082 | 0.00042 |
| missense variant | 250 | 250 | 0.8496 | [0.8140, 0.8831] | 0.072 | < 0.00002 |
| non-coding transcript variant | 250 | 250 | 0.7167 | [0.6715, 0.7607] | 0.072 | < 0.00002 |
| nonsense | 250 | 250 | 0.7462 | [0.7020, 0.7885] | 0.072 | < 0.00002 |
| splice acceptor variant | 250 | 250 | 0.7852 | [0.7438, 0.8249] | 0.072 | < 0.00002 |
| splice donor variant | 250 | 250 | 0.7952 | [0.7546, 0.8339] | 0.072 | < 0.00002 |
| synonymous variant | 250 | 250 | 0.7902 | [0.7491, 0.8296] | 0.072 | < 0.00002 |

All values are the Evo 2-1B checkpoint on the 1,001-bp window readout. Intervals are percentile bootstrap at B = 200,000 where available. MDE is the Hanley–McNeil minimum detectable difference from chance at 80% power. The p value is adjusted for the family-wise error rate (FWER) across the eight rungs by Westfall–Young max-T over 50,000 permutations — a separate resampling from the B = 200,000 bootstrap above. The smallest p that permutation count can resolve is 1/50,001 = 0.00002, so the seven cells shown as "< 0.00002" are bounds, not estimates.
Every rung is resolved against its own minimum detectable difference and survives family-wise control across the eight.

## Table S20. Negative controls

| control | what we would have seen if the claim were false | observed | artefact |
|---|---|---|---|
| Reference likelihood alone, human ClinVar | The readout is an artefact of how likely the reference window is, so the reference window's mean log-likelihood discriminates on its own | 0.4610 [0.4426, 0.4802] at 7B, an interval that excludes chance on the far side; mildly anti-predictive, not uninformative | analyses/results/clinvar_evo2_participant.json |
| Reference likelihood alone, splicing panel | The same, on a panel with cell-measured labels | 0.5122 [0.4921, 0.5326] at 1B; 0.5736 [0.5509, 0.5966] at 7B, the largest of the four human-panel readings of Note S37, below the animal panel's central kilobase, 0.6908 | analyses/results/mfass_evo2_vs_specialists.json |
| Positional baseline after distance matching (**did not pass**) | Matching on TSS distance removes the positional confound, so the baseline falls to chance and the panel tests regulatory signal alone | It does not. Matching cuts TSS proximity from 0.6510 to 0.5165 [0.5056, 0.5273] under the caliper (n = 14,402) and only to 0.6282 [0.6144, 0.6429] within eGene (n = 8,372); neither interval covers 0.5, so residual positional signal remains and the panel is not confound-free. Evo 2 reads 0.4852 [0.4747, 0.4957] on the matched panel, within 0.015 of chance under either orientation of the score, and the residue is 0.0165 above chance, a departure of the same size | analyses/results/eqtl_tss_matched.json |
| Composition standardisation of the reach gap | The benchmark-to-deployment gap is variant-class mix, not curation, so it disappears once the panels are standardised | SpliceAI gap 0.0409 before, 0.0407 after | analyses/results/mfass_random500k_coverage.json |
| Family-wise control, splicing, five variant classes | The per-class pattern is what five simultaneous tests produce by chance | all five survive max-T, worst p = 0.01006 | analyses/results/multiplicity_control.json |
| Family-wise control, ClinVar, eight consequence rungs | The consequence gradient is an artefact of testing eight rungs at once | all eight survive max-T, worst p = 0.00042 | analyses/results/multiplicity_control.json |
| Seed stability of a reported count | A count quoted as a finding moves with the bootstrap seed | At the 1,001-bp readout, 3 of the 9 species always significant (goat and cattle for Evo 2, human against it), 5 never at any seed, and chicken seed-dependent (8 of 40 seeds), across 40 seeds | reports/seed_stability.json |
| Interval stability at 200,000 resamples | The published intervals are Monte Carlo noise at the precision printed | largest endpoint standard error ~3e-04, in 3_prime_UTR_variant (clinvar), given to one significant figure because two valid bootstrap realisations of that cell give 2.80e-04 and 2.92e-04; an endpoint near a rounding boundary can therefore move in the third decimal between realisations, as the 3′ UTR upper endpoint does, and the fourth decimal is resampling noise throughout; one skewed cell, the Essential Splice class (Note S35), needs BCa | Additional file 3: scripts/recompute_endpoint_se.py (the endpoint se); Additional file 2: analyses/results/heavy_intervals.json (carries the intervals and the se of the mean) |

Every value is read from the artefact named in the last column.
Each control could have failed; the one that did not pass is set in bold.

## Table S21. Reproducibility table

| claim | value | artefact | bound by |
|---|---|---|---|
| Macro AUROC across nine species, 8,192-bp readout | 0.9431 | reports/readout_effect_fullpanel.json | independent re-derivation, atlas |
| Macro AUROC, 1,001-bp readout | 0.8783 | reports/readout_effect_fullpanel.json | independent re-derivation, atlas |
| Readout effect on identical variants | +0.0647 | reports/readout_effect_fullpanel.json | independent re-derivation, atlas |
| Variants carrying both readouts | 11,109 | reports/readout_effect_fullpanel.json | independent re-derivation, atlas |
| Pooled errors at the 8,192-bp trust layer | 415 | reports/trust_layer_8192.json | independent re-derivation, trust layer |
| Error capture at 15% refusal, 8,192-bp arm | 0.6096 | reports/trust_layer_8192.json | independent re-derivation, trust layer |
| Macro expected calibration error, 8,192-bp arm | 0.0209 | reports/trust_layer_8192.json | glmtrust benchmark |
| Pig cis-eQTL, Evo 2, the 19,341 published-panel variants with a TSS distance | 0.4879 | analyses/results/eqtl_tss_matched.json | matched-baseline recomputation |
| Pig cis-eQTL, positional baseline after matching | 0.5165 | analyses/results/eqtl_tss_matched.json | matched-baseline recomputation |
| Splicing panel, Evo 2 (1B) | 0.6214 | analyses/results/mfass_evo2_vs_specialists.json | re-run and compared |
| Splicing panel, SpliceAI | 0.7986 | analyses/results/mfass_evo2_vs_specialists.json | re-run and compared |
| Splicing panel, Pangolin | 0.8029 | analyses/results/mfass_evo2_vs_specialists.json | re-run and compared |
| ClinVar 3′ UTR rung, Evo 2-1B | 0.7023 | analyses/results/clinvar_evo2_participant.json | re-run and compared |
| ClinVar 5′ UTR rung, Evo 2-1B | 0.6497 | analyses/results/clinvar_evo2_participant.json | re-run and compared |
| Evo 2's own must-answer penalty on ClinVar | +0.0000 | analyses/results/clinvar_evo2_participant.json | re-run and compared |
| S-CAP reach, six-gene benchmark | 0.6406 | analyses/results/mfass_reach.json | re-run and compared |
| S-CAP reach, 500,000 random variants | 0.3895 | analyses/results/mfass_random500k_coverage.json | re-run and compared |
| Variants answered by all eight splice predictors | 0.0792 | analyses/results/mfass_random500k_coverage.json | re-run and compared |
| Strand-choice AUROC shift, 40B, 8,192 bp | 0.0079 | analyses/results/strand_ladder.json | re-run and compared |
| Strand-choice shift as a percentage of the 8,192-bp margin | 8.2% | analyses/results/strand_ladder.json | re-run and compared |

Every value is read from the artefact named beside it at build time, so a row cannot outlive the number it describes.
Rows bound by independent re-derivation are re-derived from the raw score and label files by tools/verify_from_data.py (Additional file 2), a script that imports nothing from the analysis code, so those checks are an independent reimplementation. The glmtrust-benchmark row is bound by glmtrust's packaged benchmark (Methods), and the remaining rows by re-running the analysis that produced them and comparing the artefact byte for byte, which shows the value is current and deterministic but does not reproduce it independently.

## Table S22. Expected calibration error is estimator-dependent

| Binning rule | isotonic (transfer) | trivial sigmoid | in-species oracle | isotonic − sigmoid |
|---|---|---|---|---|
| width5 | 0.0510 | 0.0540 | 0.0129 | −0.0031 |
| mass5 | 0.0497 | 0.0477 | 0.0193 | +0.0020 |
| width10 | 0.0533 | 0.0551 | 0.0157 | −0.0019 |
| mass10 | 0.0560 | 0.0553 | 0.0201 | +0.0007 |
| width15 | 0.0557 | 0.0562 | 0.0176 | −0.0004 |
| mass15 | 0.0554 | 0.0621 | 0.0226 | −0.0067 |
| width20 | 0.0556 | 0.0572 | 0.0206 | −0.0016 |
| mass20 | 0.0579 | 0.0652 | 0.0236 | −0.0074 |
| width30 | 0.0581 | 0.0629 | 0.0229 | −0.0048 |
| mass30 | 0.0581 | 0.0678 | 0.0256 | −0.0097 |
| width50 | 0.0603 | 0.0663 | 0.0248 | −0.0060 |
| mass50 | 0.0605 | 0.0721 | 0.0272 | −0.0116 |

Macro-averaged ECE for the transferred isotonic map, the trivial two-parameter global sigmoid, and an in-species oracle, across twelve binning rules. The formal paired test was run on four of these (ece_grid): 3 of 4 are indistinguishable, with mass15 dissenting in transfer's favour.
**Table S22 (continued)**

| Species | mass10 → effective | mass15 → effective |
|---|---|---|
| Goat | 6 | 9 |
| Chicken | 5 | 6 |
| Pig | 6 | 7 |
| Sheep | 5 | 7 |
| Horse | 4 | 6 |
| Cat | 6 | 8 |
| Cattle | 5 | 7 |
| Dog | 5 | 7 |
| Human | 8 | 11 |

Why equal-mass rules disagree with equal-width rules. The transferred isotonic posterior takes only a few dozen distinct values per species, so an equal-mass rule cannot deliver the bin count it is asked for. Requested versus achieved bin counts are tabulated above.
This degeneracy is the source of the sign disagreement above. It is one-sided: it collapses the realised bin count for the isotonic arm alone, so the sigmoid's upward bias keeps growing with the requested count while the isotonic arm's saturates. Note S44 sets out that mechanism and the bias-subtracted comparison, which leaves no isotonic advantage at any rule.
**Table S22 (continued)**

| Estimator | aggregation | isotonic (transfer) | trivial sigmoid | isotonic − sigmoid |
|---|---|---|---|---|
| width10 | macro | 0.0533 | 0.0551 | −0.0019 |
| width10 | pooled | 0.0046 | 0.0440 | −0.0394 |
| width15 | macro | 0.0557 | 0.0562 | −0.0004 |
| width15 | pooled | 0.0565 | 0.0522 | +0.0043 |
| mass10 | macro | 0.0560 | 0.0553 | +0.0007 |
| mass10 | pooled | 0.0525 | 0.0488 | +0.0037 |
| mass15 | macro | 0.0554 | 0.0621 | −0.0067 |
| mass15 | pooled | 0.0532 | 0.0582 | −0.0050 |

Both aggregations of the four formally tested estimators (eight cells). The paired test was run on equal-width and equal-mass binning at 10 and 15 bins, under macro and pooled aggregation. The two pooled cells disagree in sign (width10 favours transfer, width15 favours the sigmoid), which is the direct evidence that pooled ECE is unstable on this posterior and why the macro aggregation is used throughout.

## Table S23. Calibration transfer per species, leave-one-species-out

| Species | transferred isotonic | transferred Platt | trivial global sigmoid | in-species oracle |
|---|---|---|---|---|
| Goat | 0.0564 | 0.0481 | 0.0481 | 0.0629 |
| Chicken | 0.0203 | 0.0211 | 0.0211 | 0.0139 |
| Pig | 0.0375 | 0.0369 | 0.0369 | 0.0137 |
| Sheep | 0.0531 | 0.0550 | 0.0550 | 0.0136 |
| Horse | 0.0429 | 0.0441 | 0.0441 | 0.0133 |
| Cat | 0.0412 | 0.0370 | 0.0370 | 0.0058 |
| Cattle | 0.0609 | 0.0707 | 0.0707 | 0.0084 |
| Dog | 0.0495 | 0.0491 | 0.0491 | 0.0026 |
| Human | 0.1174 | 0.1339 | 0.1339 | 0.0073 |
| **macro** | **0.0533** | **0.0551** | **0.0551** | **0.0157** |
| pooled | 0.0046 | 0.0440 | 0.0440 | 0.0047 |

Equal-width 10-bin ECE. The transferred Platt map and the trivial global sigmoid are arithmetically the same operation and are bit-identical here, which is why the substantive comparison in the main text is isotonic against that sigmoid.
**Table S23 (continued)**

| calibration set size (variants) | mean transferred adaptive ECE | sd |
|---|---|---|
| 25 | 0.0578 | 0.0396 |
| 50 | 0.0322 | 0.0159 |
| 100 | 0.0303 | 0.0133 |
| 200 | 0.0395 | 0.0215 |
| 500 | 0.0353 | 0.0185 |
| 1,000 | 0.0410 | 0.0198 |
| 10,327 | 0.0470 | 0.0146 |

Calibration-set-size learning curve. The unit is variants, not donor species. It answers whether the smallest panels (goat, n = 99) are overfitting: they are not, and equally, more calibration data does not close the gap to the in-species oracle.
The learning curve is flat within noise.

## Table S24. Prevalence correction to the panel construction ratio

| Species | n | positives | prevalence (source) | prevalence (target) | ECE uncorrected | ECE corrected |
|---|---|---|---|---|---|---|
| Goat | 99 | 9 | 0.210 | 0.091 | 0.056 | 0.040 |
| Chicken | 308 | 28 | 0.210 | 0.091 | 0.020 | 0.025 |
| Pig | 396 | 36 | 0.210 | 0.091 | 0.038 | 0.032 |
| Sheep | 616 | 56 | 0.217 | 0.091 | 0.053 | 0.021 |
| Horse | 781 | 71 | 0.220 | 0.091 | 0.043 | 0.011 |
| Cat | 1,365 | 125 | 0.228 | 0.092 | 0.041 | 0.022 |
| Cattle | 2,068 | 188 | 0.240 | 0.091 | 0.061 | 0.007 |
| Dog | 2,497 | 227 | 0.248 | 0.091 | 0.050 | 0.017 |
| Human | 3,000 | 1,500 | 0.091 | 0.500 | 0.117 | 0.118 |

Calibrated probabilities were mapped from the donor-pool prevalence to the held-out panel's own construction ratio by Elkan–Saerens prior correction. The target is that build ratio, not a deployment prevalence; see Note S46.

## Table S25. Mondrian versus marginal conformal prediction

| Species | n | pos | Mondrian cov. | Mondrian cov. (pos.) | Mondrian abstain | marginal cov. | marginal cov. (pos.) | marginal abstain |
|---|---|---|---|---|---|---|---|---|
| Goat | 99 | 9 | 0.970 | 1.000 | 0.505 | 0.929 | 0.222 | 0.000 |
| Chicken | 308 | 28 | 0.984 | 0.964 | 0.627 | 0.961 | 0.571 | 0.000 |
| Pig | 396 | 36 | 0.957 | 0.889 | 0.657 | 0.949 | 0.444 | 0.000 |
| Sheep | 616 | 56 | 0.935 | 0.911 | 0.594 | 0.963 | 0.732 | 0.000 |
| Horse | 781 | 71 | 0.945 | 0.972 | 0.680 | 0.959 | 0.634 | 0.000 |
| Cat | 1,365 | 125 | 0.966 | 0.936 | 0.704 | 0.954 | 0.544 | 0.000 |
| Cattle | 2,068 | 188 | 0.914 | 0.984 | 0.635 | 0.950 | 0.660 | 0.000 |
| Dog | 2,497 | 227 | 0.961 | 0.943 | 0.625 | 0.963 | 0.648 | 0.000 |
| Human | 3,000 | 1,500 | 0.867 | 0.978 | 0.434 | 0.807 | 0.679 | 0.000 |

The marginal construction abstains far less but lets positive-class coverage collapse, on exactly the variants a clinician most needs to catch. This is why the conformal layer reported in Notes S63 and S66 is the Mondrian (class-conditional) arm. Only goat, chicken and pig are held out. The isotonic map and the conformal quantiles were fitted on the pooled variants of the other six species, so those rows are in-sample. Positive-class coverage in the three held-out species rests on 73 variants: goat is 9 of 9, chicken 27 of 28 and pig 32 of 36. Those coverages have a floor, because an abstention set contains both labels and so covers a positive. The table reports abstention pooled over the two classes. Split by class in the Mondrian arm, it is 0.222 in goat, 0.393 in chicken and 0.222 in pig on positives, against 0.533, 0.650 and 0.700 on negatives. Abstention concentrates in the negative class, and the share of positives receiving a set containing the positive label alone is 0.778, 0.571 and 0.667. No cell carries a confidence interval; pig's exact 95% interval runs from 0.739 to 0.969 and contains the nominal 0.90.

## Table S26. Selective prediction: the risk–coverage curve

| Coverage | macro selective error | random-refusal control | pooled selective error |
|---|---|---|---|
| 100% | 0.0577 | 0.0577 | 0.0774 |
| 95% | 0.0450 | 0.0588 | 0.0649 |
| 90% | 0.0399 | 0.0577 | 0.0589 |
| 85% | 0.0376 | 0.0557 | 0.0586 |
| 80% | 0.0377 | 0.0562 | 0.0611 |
| 75% | 0.0378 | 0.0568 | 0.0638 |
| 70% | 0.0388 | 0.0566 | 0.0648 |
| 65% | 0.0394 | 0.0561 | 0.0672 |
| 60% | 0.0410 | 0.0550 | 0.0711 |
| 55% | 0.0426 | 0.0541 | 0.0755 |
| 50% | 0.0449 | 0.0540 | 0.0810 |
| 45% | 0.0458 | 0.0531 | 0.0873 |
| 40% | 0.0436 | 0.0547 | 0.0946 |
| 35% | 0.0428 | 0.0569 | 0.1014 |
| 30% | 0.0412 | 0.0508 | 0.1030 |

Refusing each species' own least-confident 15% of calls defers 1,673 of 11,130 (a realised 15.03%, the threshold being applied within each species) and captures 35.4% of all errors (316 of 893, on the Platt posterior of Table S27), a lift of 2.35 over refusing at random. Refusing the worst 15% globally instead, ignoring species, defers 1,669 calls and captures 38.4% (343), lift 2.56.
The minimum sits at 85% coverage; refusing beyond it buys nothing and eventually costs, as the retained set becomes small and unrepresentative.
**Table S26 (continued)**

| Coverage | n kept | pooled error | vs do-nothing | sensitivity | specificity |
|---|---|---|---|---|---|
| 100% | 11,130 | 0.0802 | baseline | 0.674 | 0.982 |
| 95% | 10,574 | 0.0651 | better | 0.689 | 0.992 |
| 90% | 10,017 | 0.0601 | better | 0.684 | 0.995 |
| 85% | 9,460 | 0.0581 | better | 0.671 | 0.996 |
| 80% | 8,904 | 0.0576 | better | 0.646 | 0.997 |
| 75% | 8,348 | 0.0580 | better | 0.617 | 0.998 |
| 70% | 7,791 | 0.0585 | better | 0.583 | 0.998 |
| 65% | 7,234 | 0.0592 | better | 0.547 | 0.999 |
| 60% | 6,678 | 0.0615 | better | 0.511 | 0.999 |
| 55% | 6,122 | 0.0642 | better | 0.475 | 0.999 |
| 50% | 5,565 | 0.0668 | better | 0.428 | 1.000 |
| 45% | 5,008 | 0.0707 | better | 0.382 | 1.000 |
| 40% | 4,452 | 0.0761 | better | 0.325 | 1.000 |
| 35% | 3,895 | 0.0809 | worse | 0.298 | 1.000 |
| 30% | 3,339 | 0.0889 | worse | 0.239 | 1.000 |
| 25% | 2,782 | 0.0996 | worse | 0.171 | 1.000 |
| 20% | 2,226 | 0.1141 | worse | 0.070 | 0.999 |

The deployed (Platt) arm, full trajectory to 20% coverage. The isotonic table uses the isotonic arm and stops at 30%; the per-variant deliverable is the Platt arm (Note S63), whose deposited trajectory also reports the retained-set sensitivity. Its error rises monotonically below the 80–85% minimum and crosses the do-nothing baseline of 0.0802, so below that crossing the layer is worse than answering every variant; sensitivity collapses in step while specificity climbs toward 1.0; pushed hard, the layer answers 'benign' to almost everything.

## Table S27. Selective prediction per species

| Species | n | errors | refused | errors removed | lift | lift 95% CI | resolved |
|---|---|---|---|---|---|---|---|
| Goat | 99 | 6 | 15 | 5 | 5.50 | [3.09, 9.90] | yes |
| Chicken | 308 | 12 | 47 | 4 | 2.18 | [0.53, 4.18] | no |
| Pig | 396 | 20 | 60 | 12 | 3.96 | [2.54, 5.66] | yes |
| Sheep | 616 | 21 | 93 | 15 | 4.73 | [3.37, 6.34] | yes |
| Horse | 781 | 31 | 117 | 14 | 3.01 | [1.90, 4.24] | yes |
| Cat | 1,365 | 61 | 205 | 30 | 3.27 | [2.52, 4.14] | yes |
| Cattle | 2,068 | 111 | 311 | 59 | 3.53 | [2.93, 4.19] | yes |
| Dog | 2,497 | 86 | 375 | 37 | 2.86 | [2.19, 3.57] | yes |
| Human | 3,000 | 545 | 450 | 140 | 1.71 | [1.52, 1.93] | yes |
| **Pooled** | **11,130** | **893** | **1,673** | **316** | **2.35** | **–** | **–** |

Each species refuses its own least-confident 15% of calls, ordered by the confidence functional |2p − 1|; an error is a call on the wrong side of the 0.5 threshold. Lift is the fraction of that species' errors removed divided by the fraction of its calls refused, so 1.0 is what random refusal achieves. Because the 15% boundary can fall inside a confidence tie block, the refused count for a species can differ by a variant or two between deposited artefacts, so re-deriving the boundary arithmetically gives a per-species total of 1,669 (rounding to nearest) or 1,674 (rounding up) against the 1,673 the thresholding actually produced. The refused counts here are the deposited ones, so this table, Table S26 and the recompute layer agree. Recomputed from the deposited per-variant file, whose per-variant posterior is the two-parameter Platt map (not the isotonic map used for the ECE analysis); the isotonic posterior takes too few distinct values to order variants for selective refusal (Note S63).
Lift 95% CIs and the resolution flag are the deposited bootstrap intervals (seed 20260719, 2,000 resamples); a lift is resolved only when its interval excludes 1. Chicken is the one species whose interval spans 1 (not distinguishable from random refusal).
Human has the lowest lift of the nine while carrying 545 of the 893 errors, so the species with the most errors is the one where confidence helps least, which is a property of its 1:1 panel composition rather than its labels: at a matched 10:1 composition its lift rises to 3.19. Seven of the nine per-species lifts exceed the pooled figure, because pooling weights by error count and human dominates that count. The highest lift rests on six errors and should not be read as a species ranking.

## Table S28. The confusion matrix behind the 893 errors

|  | predicted negative | predicted positive | total |
|---|---|---|---|
| **actually positive** | 731 (missed) | 1,509 (caught) | 2,240 |
| **actually negative** | 8,728 (correct) | 162 (false alarm) | 8,890 |
| **total** | 9,459 | 1,671 | 11,130 |

At the 0.5 decision threshold on the calibrated probability, before any abstention. Recomputed from the deposited per-variant file on the same Platt posterior as Table S27, for the reason given there.
Pooled sensitivity 0.674 [0.654, 0.693], specificity 0.982 [0.979, 0.984] (deposited bootstrap), overall error rate 0.0802. The 893 errors are 731 missed positives against only 162 false alarms, so 81.9% of all errors are missed positives. That is the asymmetry the selective layer does not fix, because refusal removes false alarms far more effectively than misses.

## Table S29. Per-species operating point at the 0.5 decision threshold

| Species | n | positives | sensitivity | sensitivity 95% CI | specificity | PPV |
|---|---|---|---|---|---|---|
| Goat | 99 | 9 | 0.333 | [0.000, 0.667] | 1.000 | 1.000 |
| Chicken | 308 | 28 | 0.571 | [0.381, 0.760] | 1.000 | 1.000 |
| Pig | 396 | 36 | 0.444 | [0.275, 0.605] | 1.000 | 1.000 |
| Sheep | 616 | 56 | 0.786 | [0.673, 0.889] | 0.984 | 0.830 |
| Horse | 781 | 71 | 0.648 | [0.537, 0.754] | 0.992 | 0.885 |
| Cat | 1,365 | 125 | 0.584 | [0.496, 0.670] | 0.993 | 0.890 |
| Cattle | 2,068 | 188 | 0.718 | [0.654, 0.782] | 0.969 | 0.699 |
| Dog | 2,497 | 227 | 0.683 | [0.623, 0.744] | 0.994 | 0.917 |
| Human | 3,000 | 1,500 | 0.681 | [0.655, 0.705] | 0.956 | 0.939 |
| **Eight non-human (species mean)** | **–** | **740** | **0.596** | **–** | **–** | **–** |
| **Eight non-human (positive-weighted)** | **–** | **740** | **0.659** | **–** | **–** | **–** |

Sensitivity, specificity and positive predictive value at the threshold the selective layer is built on, before any abstention. Sensitivity 95% CIs are the deposited bootstrap intervals; goat's spans zero ([0.000, 0.667] on 9 positives, 3 caught) and must not be read as a point. Recomputed from the deposited per-variant file (two-parameter Platt posterior; Note S63).
Specificity is high everywhere and sensitivity is not. Across the eight non-human species the model detects 59.6% of catalogued positives at this threshold by the unweighted species mean (65.9% weighting species by positive count), and only 33.3% in goat. A macro AUROC of 0.943 at the 8,192-bp mean-log-likelihood readout and a sensitivity of 0.60 at the 0.5 threshold under the 1,001-bp readout that produced these calls are both true of this predictor; the first is a ranking property and the second is what a single call delivers.

## Table S30. The trust layer rebuilt at the 8,192-bp readout

| Species | n | AUROC | ECE | errors | lift | error-detection AUROC |
|---|---|---|---|---|---|---|
| Goat | 98 | 0.959 | 0.0514 | 7 | 5.60 | 0.929 |
| Chicken | 308 | 0.954 | 0.0272 | 9 | 2.23 | 0.743 |
| Pig | 396 | 0.863 | 0.0219 | 11 | 3.05 | 0.597 |
| Sheep | 616 | 0.962 | 0.0095 | 17 | 5.12 | 0.858 |
| Horse | 766 | 0.941 | 0.0115 | 19 | 2.80 | 0.676 |
| Cat | 1,362 | 0.890 | 0.0132 | 34 | 2.95 | 0.631 |
| Cattle | 2,067 | 0.974 | 0.0232 | 57 | 5.03 | 0.880 |
| Dog | 2,496 | 0.970 | 0.0072 | 54 | 4.70 | 0.856 |
| Human | 3,000 | 0.974 | 0.0226 | 207 | 3.93 | 0.828 |

The construction follows the 1,001-bp layer of Notes S63 and S66: leave-one-species-out calibration, the confidence functional |2p − 1|, each species refusing its own least-confident 15%, a 0.5 decision threshold. Besides the score, the refusals here rest on an isotonic rather than a Platt posterior, and the donor pool admits every other species (Notes S52 and S54). With 218 distinct posteriors, a species' 15% boundary can fall inside a tie block; src/ccs/rebuild_trust_layer_8192.py (Additional file 2) breaks such ties by a stable sort, in the order of reports/fig4_pervariant_8192.parquet, so at the boundary the refused set, and with it the lifts here, follow that order rather than confidence.
**Table S30 (continued)**

|  | 1,001-bp (Note S66) | 8,192-bp |
|---|---|---|
| total errors | 893 (8.02%) | 415 (3.74%) |
| errors removed at 15% refusal | 316 (35.4%) | 253 (61.0%) |
| pooled lift | 2.35 | 4.07 |
| macro lift | 3.42 | 3.93 |
| macro ECE (equal-width, 10 bins) | 0.053 | 0.021 |
| macro error-detection AUROC (eight species, human excluded) | 0.736 | 0.771 |
| macro error-detection AUROC (all nine species) | 0.707 | 0.777 |

The layer is better at the readout the paper argues for, on every axis, and every figure here is the deployable leave-one-species-out arm rather than an in-species oracle. The error count falls from 893 to 415, the same 15% per-species refusal captures 61% of them rather than 35%, and calibration improves by a factor of 2.6. We report the 1,001-bp layer as the headline because the reach, calibration and abstention analyses were built there and cover the full 11,130-variant panel; this table is the check that the choice does not flatter the result.

## Table S31. The selective layer's confidence ordering is not sampling noise

| Species | n | errors | error-detection AUROC | 95% CI | leave-one-out pooled lift |
|---|---|---|---|---|---|
| Goat | 98 | 7 | 0.929 | [0.849, 0.985] | 4.04 |
| Chicken | 308 | 9 | 0.743 | [0.557, 0.896] | 4.11 |
| Pig | 396 | 11 | 0.597 | [0.384, 0.801] | 4.09 |
| Sheep | 616 | 17 | 0.858 | [0.733, 0.953] | 4.02 |
| Horse | 766 | 19 | 0.676 | [0.509, 0.827] | 4.13 |
| Cat | 1,362 | 34 | 0.631 | [0.507, 0.748] | 4.17 |
| Cattle | 2,067 | 57 | 0.880 | [0.829, 0.924] | 3.91 |
| Dog | 2,496 | 54 | 0.856 | [0.798, 0.908] | 3.97 |
| Human | 3,000 | 207 | 0.828 | [0.796, 0.858] | 4.20 |

Whether the 15%-refusal lift (4.07 pooled) reflects genuine selectivity or the sampling variability of small panels, tested on the deployable leave-one-species-out arm at the 8,192-bp readout. A within-species permutation of the confidence (10,000 permutations, each preserving every panel's size and error count) gives a null lift of 0.97; the observed lift sits far outside it (p = 0.0001). Error detection is the AUROC of the confidence functional |2p − 1| separating incorrect from correct calls.
**Table S31 (continued)**

| pooled | value |
|---|---|
| error-detection AUROC (species-clustered 95% CI) | 0.814 [0.703, 0.849] |
| within-species permutation p | 0.0001 |
| leave-one-species-out lift range | 3.91 to 4.20 |

At this readout, error-detection ordering was above chance in eight of nine species, taking above chance to mean a bootstrap interval excluding 0.5, and the pooled permutation result was inconsistent with random ordering. The effect was nevertheless heterogeneous: pig's interval spans chance, human is near chance at the 1,001-bp readout and two of the nine species are worse than random there when the whole risk–coverage curve is summarised, and the 15% operating point was selected on these panels. We therefore describe a pooled selective-prediction signal at this operating point, not a validated transfer guarantee for a label-free species.
The same test at the 1,001-bp readout, which is the layer Note S66 reports. Observed pooled lift 2.36; permutation null mean 1.00, p = 0.0001; pooled error-detection AUROC 0.586, species-clustered 95% CI [0.510, 0.768]; leave-one-species-out lift range 2.19 to 3.37; errors detected above chance in seven of the nine species. The permutation rejects sampling variability at both readouts and both pooled intervals clear 0.5, but error detection is markedly weaker at 1,001 bp, which is why the two arms are never quoted interchangeably.

## Table S32. Class-asymmetry of the abstention layer, per species

| Species | negative-class errors | positive-class errors | protects negatives | protects positives |
|---|---|---|---|---|
| Sheep | 9 | 12 | 1.000 | 0.500 |
| Cat | 9 | 52 | 1.000 | 0.404 |
| Horse | 6 | 25 | 0.833 | 0.360 |
| Dog | 14 | 72 | 1.000 | 0.319 |
| Cattle | 58 | 53 | 0.776 | 0.264 |
| Human | 66 | 479 | 0.652 | 0.203 |

Fraction of each error class removed when a species refuses its least-confident 15% of calls. 'protects negatives' is the share of over-calls removed, 'protects positives' the share of missed positives removed. The negative side is defined in only six species (goat, chicken and pig have no negative-class errors at the 0.5 threshold); human carries 479 of the 731 missed positives.
Positive-class capture ranges from 0.500 (sheep) to 0.203 (human) in the six species tabulated, and is 0.833 in goat, 0.600 in pig and 0.333 in chicken, whose errors are all missed positives; the pooled figure of 0.261, 191 of 731, is a variant-weighted mean dominated by human.

## Table S33. Per-species calibration: transfer vs trivial vs oracle

| Species | n | ECE trivial | ECE transfer | ECE oracle | ECE transfer > ECE trivial? | oracle vs transfer |
|---|---|---|---|---|---|---|
| Goat | 99 | 0.0545 | 0.0564 | 0.0657 | yes | 0.9× |
| Chicken | 308 | 0.0215 | 0.0264 | 0.0221 | yes | 1.2× |
| Pig | 396 | 0.0471 | 0.0582 | 0.0130 | yes | 4.5× |
| Sheep | 616 | 0.0588 | 0.0545 | 0.0182 | no | 3.0× |
| Horse | 781 | 0.0377 | 0.0387 | 0.0169 | yes | 2.3× |
| Cat | 1,365 | 0.0377 | 0.0466 | 0.0150 | yes | 3.1× |
| Cattle | 2,068 | 0.0676 | 0.0608 | 0.0089 | no | 6.8× |
| Dog | 2,497 | 0.0383 | 0.0472 | 0.0101 | yes | 4.7× |
| Human | 3,000 | 0.1347 | 0.1153 | 0.0107 | no | 10.8× |

Expected calibration error on the operating-point estimator: trivial is the two-parameter global sigmoid, transfer the leave-one-species-out isotonic map (the paper's method), oracle in-species calibration. Under this estimator, transfer is worse than the trivial sigmoid in six of nine species (macro transfer 0.0560 against trivial 0.0553, a difference of +0.0007) and better in the other three (sheep, cattle, human), which are not simply the three largest panels, since dog and cat are both larger than sheep. Under the paper's primary equal-width-10 estimator (Table S23) the count is four of nine (goat, pig, cat, dog) and the macro slightly favours transfer (−0.0019), so the direction of the split is estimator-dependent and we name the estimator with each count. Where labels exist the oracle ranges from 0.9× (slightly worse than transfer, in goat) to 10.8× better (in human).
Transfer is worse than the trivial two-parameter sigmoid in six of nine species, the per-species detail behind the macro null, and the reason we frame transfer as recovering only a fraction of what in-species labels would buy.
Table S34 is printed with Note S65.

## Table S35. Within-class discrimination at both readouts against GERP, on co-scorable variants

| Species | Class | Positives / negatives | Evo 2, 1,001 bp | Evo 2, 8,192 bp | GERP | Readout gain | Margin at 8,192 bp |
|---|---|---|---|---|---|---|---|
| eight OMIA species, pooled | all classes | 6,649 variants | 0.678 [0.599, 0.759] | 0.773 [0.692, 0.853] | 0.652 [0.554, 0.746] | +0.095 [+0.029, +0.167] | +0.121 [+0.021, +0.239] |
| human | all classes | 2,880 variants | 0.809 [0.775, 0.841] | 0.850 [0.815, 0.882] | 0.837 [0.799, 0.873] | +0.041 [+0.004, +0.078] | +0.012 [−0.030, +0.056] |
| eight OMIA species, pooled | missense | 424 / 72 | 0.783 [0.718, 0.842] | 0.746 [0.671, 0.816] | 0.660 [0.573, 0.744] | −0.037 [−0.115, +0.039] | +0.086 [−0.022, +0.186] |
| eight OMIA species, pooled | synonymous | 5 / 51 | 0.500 [0.273, 0.734] | 0.578 [0.292, 0.895] | 0.344 [0.000, 0.594] | +0.078 [−0.118, +0.385] | +0.234 [+0.000, +0.567] |
| eight OMIA species, pooled | splice region | 12 / 16 | 0.578 [0.278, 0.861] | 0.906 [0.724, 1.000] | 0.875 [0.621, 1.000] | +0.328 [−0.012, +0.656] | +0.031 [−0.181, +0.250] |
| eight OMIA species, pooled | intronic | 27 / 2140 | 0.534 [0.395, 0.686] | 0.715 [0.560, 0.883] | 0.657 [0.478, 0.830] | +0.181 [+0.046, +0.347] | +0.058 [−0.122, +0.277] |
| eight OMIA species, pooled | intergenic or flanking | 15 / 2989 | 0.773 [0.649, 0.903] | 0.854 [0.744, 0.957] | 0.619 [0.420, 0.820] | +0.081 [+0.010, +0.170] | +0.236 [+0.047, +0.442] |
| human | missense | 566 / 163 | 0.829 [0.794, 0.862] | 0.865 [0.831, 0.898] | 0.843 [0.801, 0.882] | +0.036 [+0.001, +0.072] | +0.023 [−0.020, +0.070] |
| human | stop gained | 515 / 5 | 0.773 [0.658, 0.878] | 0.449 [0.144, 0.723] | 0.765 [0.684, 0.845] | −0.324 [−0.730, +0.011] | −0.317 [−0.570, −0.037] |
| human | splice region | 20 / 104 | 0.637 [0.472, 0.806] | 0.934 [0.830, 1.000] | 0.819 [0.686, 0.938] | +0.296 [+0.102, +0.487] | +0.114 [−0.015, +0.263] |
| human | intronic | 5 / 314 | 0.399 [0.125, 0.732] | 0.638 [0.122, 1.000] | 0.597 [0.311, 0.946] | +0.239 [−0.255, +0.712] | +0.040 [−0.291, +0.431] |
| human | intergenic or flanking | 8 / 105 | 0.485 [0.250, 0.744] | 0.770 [0.529, 0.983] | 0.737 [0.475, 0.952] | +0.286 [+0.045, +0.546] | +0.033 [−0.129, +0.192] |

Pairs are formed within species and within consequence class (first snpEff term, splice donor and acceptor pooled as splice site, 3′ and 5′ UTR pooled, intergenic, upstream and downstream pooled as intergenic or flanking), on the 9,529 variants that Evo 2 at both readouts and GERP all score, and pooled over species by pair count. Intervals resample positive-bearing 100-kb loci whole and negatives individually within species (B = 2,000, seed 61). A class row is shown only where the pooled class holds at least five positives and five negatives; the per-species cells meeting the same rule, with each species' share of the pairs and a precision-weighted pool across species (Note S23), are printed by recompute_macro_ci.py. Readout gain is Evo 2 at 8,192 bp minus Evo 2 at 1,001 bp; margin is Evo 2 at 8,192 bp minus GERP; both are computed before rounding and can differ by 0.001 from the subtraction of the printed columns. A class row counts the variants that enter a within-species pair, so a species lacking either label in that class contributes none; the residual class of other first terms is not shown.

## Table S36. Calibrated PP3 and BP4 evidence on ClinVar missense variants, by class

| Tool | ClinVar class | Missense variants | PP3 | BP4 | Indeterminate | Unscored |
|---|---|---|---|---|---|---|
| REVEL | pathogenic | 58,394 | 82.1% | 3.4% | 14.1% | 0.4% |
| REVEL | benign | 145,797 | 3.2% | 79.7% | 14.7% | 2.4% |
| AlphaMissense | pathogenic | 58,394 | 71.4% | 3.7% | 23.1% | 1.7% |
| AlphaMissense | benign | 145,797 | 2.7% | 73.5% | 16.8% | 7.0% |
| CADD | pathogenic | 58,394 | 75.7% | 6.8% | 17.5% | 0.0% |
| CADD | benign | 145,797 | 8.3% | 80.1% | 11.6% | 0.0% |
| phyloP | pathogenic | 58,394 | 58.4% | 7.7% | 33.9% | 0.0% |
| phyloP | benign | 145,797 | 7.0% | 64.3% | 28.7% | 0.0% |

The 204,191 variants of the 1,434,335-variant ClinVar panel that ClinVar annotates as missense. PP3 and BP4 count evidence of any strength, supporting or stronger, from the calibrated intervals of Pejaver et al. [31] (REVEL: BP4 at or below 0.290, PP3 at or above 0.644; CADD: 22.7 and 25.3; phyloP: 1.879 and 7.367) and Bergquist et al. [46] (AlphaMissense: 0.169 and 0.792). Indeterminate scores fall between the two bounds; unscored variants receive no evidence.

## Table S37. Missense AUROC on ClinGen expert-panel labels, with and without the labels that depend on PP3/BP4

| Tool | Pathogenic / benign scored | AUROC, all labels | AUROC, labels standing without PP3/BP4 | Difference [95% CI] |
|---|---|---|---|---|
| REVEL | 3,023 / 1,137 | 0.955 | 0.949 | +0.007 [+0.004, +0.010] |
| AlphaMissense | 3,017 / 1,095 | 0.936 | 0.931 | +0.005 [+0.002, +0.009] |
| CADD | 3,024 / 1,144 | 0.898 | 0.894 | +0.004 [+0.001, +0.008] |
| phyloP | 3,024 / 1,144 | 0.809 | 0.801 | +0.008 [+0.002, +0.015] |

ClinGen Evidence Repository classifications of single-nucleotide variants that are also in the ClinVar panel and annotated missense there (4,168 variants in 129 genes). A label depends on PP3/BP4 when it applied them and reaches its class with those points but not without them (Note S60). The difference is the AUROC on all labels minus the AUROC on the labels that stand without PP3/BP4, computed before rounding; intervals resample genes (B = 2,000, seed 61).

## Table S38. Resampling units, replicate counts and seeds, by reported estimand

| Estimand | Resampling unit | Replicates and seed | Script |
|---|---|---|---|
| Per-species atlas AUROC at both readouts (Table 2) | positive-bearing 100-kb loci whole, negatives individually | 20,000; seed 20260719 | table2_cluster_aware_ci.py |
| Atlas identification verdicts against GERP (Note S55) | as for Table 2 | 20,000; seed 20260923 | recompute_endpoint_fragility.py |
| Margins over GERP and identification verdicts without the positives on chromosomes carrying no negatives (Note S62) | positive-bearing 100-kb loci within species, negatives individually | 2,000, seed 61; 20,000, seed 20260923 | table2_cluster_aware_ci.py |
| Species means: macro AUROC, margins over GERP, paired readout change, increments | species (t interval on k − 1 df) | none | recompute_macro_ci.py, recompute_atlas_strata.py, recompute_fig6pop.py |
| Species-clustered percentile intervals beside the t on the macro rows (Table 2 notes) | species | 200,000; seed 20260906 | recompute_macro_ci.py |
| Species means as a fixed set: margins over GERP and paired readout change (Analyses; Note S11) | positive-bearing 100-kb loci within each species, negatives individually; the nine species held fixed | 2,000; seed 61 | recompute_atlas_strata.py |
| Paired species-clustered head-to-head at 8,192 bp (Note S11) | species, then variants within each drawn species | 4,000; seed 20260722 | build_readout_headtohead.py (Additional file 2) |
| Equivalence test of the pooled 1,001-bp difference (Note S2, Table S5) | species, then variants within each drawn species | 4,000; seed 20260721 | build_tost_equivalence.py (Additional file 2) |
| BCa interval on the nine species-level margins over GERP at 8,192 bp (Note S42) | the nine species-level values | 20,000; seed 20260723 | eqtl_cluster_and_bca.py (Additional file 2) |
| Panel-size slope contrast (Note S43) | species; paired differences permuted over panel size | 10,000; 20,000 permutations; seed 20260720 | recompute_fig1cde.py |
| Within-class AUROCs, gains and margins; the strata of Figure 10b, missense among them; reach gaps; meta-analysis standard errors | positive-bearing 100-kb loci within species, negatives individually | 2,000; seed 61 | recompute_atlas_strata.py; fig8_readout.py (Additional file 2) |
| Capacity-ladder steps by species (Note S15) | positive-bearing 100-kb loci within species, negatives individually | 2,000; seed 61 | recompute_atlas_strata.py |
| Operating points at a fixed false-positive rate, and the consequence filter beside them (Note S29) | positive-bearing 100-kb loci within species, negatives individually | 2,000; seeds 0 and 61 | recompute_macro_ci.py; recompute_atlas_strata.py |
| Class gap in GERP reach | Newcombe hybrid score, closed form | none | fig5_stats.py (Additional file 2) |
| Consequence-matched class gap | stratified Miettinen–Nurminen score interval, closed form | none | fig5_stats.py (Additional file 2) |
| dbNSFP per-predictor intervals and identification confidence sets | genes | 2,000; seed 0 | recompute_dbnsfp.py |
| ClinVar scope-matched differences | genes | 2,000; seed 61 | recompute_identification.py |
| Expert-panel AUROC differences (Table S37) | genes | 2,000; seed 61 | recompute_pp3bp4.py |
| Expert-panel AUROC gains without the benign labels resting on population frequency (Note S67) | genes | 2,000; seed 61 | recompute_pp3bp4.py |
| Scope-matched confidence sets (Note S69) | genes, closed-form clustered variance | none | recompute_dbnsfp.py |
| One-sided 95% lower bounds of the breakdown points (Note S67) | genes | 2,000; seed 0 | recompute_dbnsfp.py |
| Side-by-side against head-to-head differences (Note S69) | genes | 2,000; seed 0 | recompute_dbnsfp.py |
| Per-gene leaders against sampling error (Note S70) | variants within the gene, each class separately | 2,000; seed 14 | recompute_dbnsfp.py |
| Likelihood ratios of the calibrated levels (Table S42) and label dependence by specification (Note S60) | genes | 2,000; seed 61 | recompute_pp3bp4.py |
| Functional-label and BRCA1 AlphaMissense differences (Note S72) | residues | 2,000; seeds 61 and 5 | recompute_pp3bp4.py, recompute_brca1.py |
| GPN-Star against Evo 2 and GERP (Note S73) | positive-bearing 100-kb loci within species, negatives individually | 2,000; seed 61 | recompute_atlas_strata.py |
| Readout-dependent model order on TraitGym (Note S75) | whole chromosomes; causal loci within chromosome | 2,000; seed 20250211 | recompute_strand.py |
| ClinVar reach-panel audit | variants | 200; seed 0 | recompute_clinvar_reach.py; audit_human_panel.py (Additional file 2) |
| ClinVar and splicing consequence rungs | variants within class; Westfall–Young max-T permutations | 200,000; 50,000; seeds from 20260805, one per chunk | recompute_consequence.py, recompute_splicing.py |
| BRCA1 AUROC | genomic sites | 1,000; seed 3 | recompute_brca1.py |
| Reference-only single-token AUROC | variants | 2,000; seed 61 | recompute_strand.py |
| Strand-choice AUROC shift (Note S36) | variants | 2,000; seed 61 | recompute_strand.py |
| eQTL permutation null | labels permuted within eGene | 20,000; seed 20260723 | eqtl_cluster_and_bca.py (Additional file 2) |
| Refusal lifts of the selective layer | variants within species | 2,000; seed 20260719 | fig4_reconcile.py (Additional file 2) |

Seeds are recorded in each script. Intervals that treat variants as independent are given only where the estimand is itself variant-level, and each is labelled as such where it is quoted.

## Table S39. Per-species provenance of the atlas

| Species | Assembly | GERP track | Gene models / snpEff database | Negatives | Positives | Genes | OMIA phenotypes | Trait alleles |
|---|---|---|---|---|---|---|---|---|
| goat | ARS1 (GCA_001704415.1) | Ensembl 110, 91 mammals | Ensembl 112 / ARS1 | Ensembl variation 110 | 9 | 2 | 3 | 8 |
| chicken | GRCg6a (GCA_000002315.5) | Ensembl 106, 27 sauropsids | Ensembl 105 / GRCg6a | Ensembl variation 104 | 28 | 21 | 25 | 20 |
| pig | Sscrofa11.1 (GCA_000003025.6) | Ensembl 110, 91 mammals | Ensembl 112 / Sscrofa11.1 | PigGTEx v0 genotypes | 36 | 29 | 28 | 13 |
| sheep | Oar_rambouillet_v1.0 (GCA_002742125.1) | Ensembl 110, 91 mammals | Ensembl 106 / Oar_rambouillet_v1.0 | Ensembl variation 109 | 56 | 37 | 50 | 27 |
| horse | EquCab3.0 (GCA_002863925.1) | Ensembl 110, 91 mammals | Ensembl 112 / EquCab3.0 | Ensembl variation 110 | 71 | 33 | 43 | 44 |
| cat | F.catus_Fca126_mat1.0 (GCA_018350175.1) | Ensembl 114, 91 mammals | Ensembl 114 / F.catus_Fca126_mat1.0 | Ensembl variation 114 | 125 | 76 | 86 | 39 |
| cattle | ARS-UCD1.2 (GCA_002263795.2) | Ensembl 110, 91 mammals | Ensembl 99 / ARS-UCD1.2.99 (pre-built) | Ensembl variation 110 | 188 | 144 | 149 | 28 |
| dog | CanFam3.1 (GCA_000002285.2) | Ensembl 104, 90 mammals | Ensembl 104 / CanFam3.1 | Ensembl variation 104 | 227 | 178 | 189 | 25 |
| human | GRCh38 (GCA_000001405.15) | Ensembl 110, 91 mammals | Ensembl 112 / GRCh38 | ClinVar benign, 1:1 | 1,500 | 843 | – | – |

Assemblies are those each panel was built and verified on. GERP is read from the per-species Ensembl compara bigWig track on that assembly; the release and alignment given are those of the track that reproduces the atlas's per-base values, value for value for all but two cat positives (build_gerp_windows.py, Additional file 3). Genes and OMIA phenotypes count distinct gene symbols and distinct phenotype strings among the positives; trait alleles are the positives classed as non-disease in atlas_positive_annotations.parquet (Additional file 3). Negatives are drawn from each species' population-variant pool rather than from the positives' genes, so the atlas supports no within-gene comparison; in sheep, chicken and pig the pool does not reach every chromosome (Note S62). The Negatives column names each species' pool and its release: Ensembl variation, at the release given, in seven species, the PigGTEx v0 genotypes in pig, and ClinVar's benign or likely-benign records, drawn 1:1, in human; in the eight non-human species the negatives are matched to the positives on trinucleotide context and alternate allele and not on GC (reports/atlas_controls.json, Additional file 2). Of the 27 positives that OMIA curates as coding but snpEff's first term calls non-coding (Note S62), 16 are in dog, 4 in cattle, 3 each in cat and horse and 1 in sheep; the gene-model releases run from Ensembl 99 (cattle) to 114 (cat).

## Table S40. Per-predictor provenance and accounting on the 49-predictor panel

| Predictor | Scope | Training labels | Allele frequency input | Reach P / B | Covered AUROC | Within-gene AUROC | Penalty |
|---|---|---|---|---|---|---|---|
| SIFT | missense | unsupervised | no | 0.345 / 0.839 | 0.899 | 0.872 | 0.284 |
| SIFT4G | missense | unsupervised | no | 0.348 / 0.834 | 0.897 | 0.897 | 0.282 |
| Polyphen2_HDIV | missense | clinical assertions | no | 0.347 / 0.804 | 0.900 | 0.883 | 0.288 |
| Polyphen2_HVAR | missense | clinical assertions | no | 0.347 / 0.804 | 0.919 | 0.905 | 0.302 |
| MutationTaster | gene-internal variants | clinical assertions | unverified | 0.922 / 0.855 | 0.991 | 0.990 | 0.104 |
| MutationAssessor | missense | unsupervised | no | 0.332 / 0.756 | 0.922 | 0.907 | 0.316 |
| PROVEAN | missense | unsupervised | no | 0.348 / 0.845 | 0.917 | 0.898 | 0.294 |
| VEST4 | missense | clinical assertions | unverified | 0.742 / 0.880 | 0.946 | 0.932 | 0.155 |
| MetaSVM | missense | clinical assertions | yes | 0.362 / 0.894 | 0.936 | 0.886 | 0.295 |
| MetaLR | missense | clinical assertions | yes | 0.362 / 0.894 | 0.937 | 0.903 | 0.296 |
| MetaRNN | rare nsSNVs, all scored | clinical assertions | yes | 0.363 / 0.907 | 0.987 | 0.968 | 0.327 |
| M-CAP | rare missense | clinical assertions | indirect | 0.361 / 0.725 | 0.937 | 0.903 | 0.323 |
| REVEL | missense | clinical assertions | no | 0.350 / 0.852 | 0.971 | 0.944 | 0.330 |
| MutPred2 | missense | clinical assertions | no | 0.359 / 0.857 | 0.961 | 0.944 | 0.319 |
| MVP | missense | clinical assertions | no | 0.357 / 0.837 | 0.949 | 0.912 | 0.315 |
| gMVP | missense | clinical assertions | no | 0.329 / 0.810 | 0.967 | 0.944 | 0.342 |
| MisFit_D | missense | population data | no | 0.346 / 0.834 | 0.945 | 0.933 | 0.317 |
| MisFit_S | missense | population data | yes | 0.346 / 0.834 | 0.880 | 0.928 | 0.271 |
| MPC | missense | clinical assertions | no | 0.311 / 0.728 | 0.854 | 0.861 | 0.274 |
| PrimateAI | missense | population data | no | 0.347 / 0.843 | 0.886 | 0.894 | 0.273 |
| DEOGEN2 | missense | clinical assertions | no | 0.353 / 0.830 | 0.944 | 0.910 | 0.314 |
| BayesDel_addAF | SNVs and indels | clinical assertions | yes | 0.997 / 0.917 | 0.990 | 0.984 | 0.042 |
| BayesDel_noAF | SNVs and indels | clinical assertions | no | 0.997 / 0.917 | 0.983 | 0.976 | 0.041 |
| ClinPred | missense | clinical assertions | yes | 0.382 / 0.892 | 0.983 | 0.969 | 0.319 |
| LIST-S2 | missense | clinical assertions | no | 0.358 / 0.895 | 0.891 | 0.862 | 0.266 |
| VARITY_R | rare missense | clinical assertions | no | 0.332 / 0.770 | 0.968 | 0.950 | 0.349 |
| VARITY_ER | extremely rare missense | clinical assertions | no | 0.332 / 0.770 | 0.964 | 0.944 | 0.345 |
| VARITY_R_LOO | rare missense | clinical assertions | no | 0.332 / 0.770 | 0.967 | 0.947 | 0.347 |
| VARITY_ER_LOO | extremely rare missense | clinical assertions | no | 0.332 / 0.770 | 0.963 | 0.943 | 0.345 |
| ESM1b | missense | unsupervised | no | 0.769 / 0.955 | 0.957 | 0.963 | 0.121 |
| AlphaMissense | missense | population data | no | 0.354 / 0.903 | 0.960 | 0.954 | 0.313 |
| PHACTboost | missense | clinical assertions | no | 0.343 / 0.793 | 0.981 | 0.965 | 0.350 |
| MutFormer | missense | clinical assertions | indirect | 0.359 / 0.875 | 0.935 | 0.883 | 0.298 |
| MutScore | missense | clinical assertions | no | 0.350 / 0.872 | 0.979 | 0.959 | 0.333 |
| popEVE | missense | population data | no | 0.308 / 0.659 | 0.924 | 0.939 | 0.338 |
| CADD_raw | all SNVs | evolutionary proxy | no | 1.000 / 1.000 | 0.968 | 0.976 | 0.000 |
| DANN | all SNVs | evolutionary proxy | no | 0.998 / 0.995 | 0.833 | 0.773 | 0.002 |
| fathmm-XF_coding | coding SNVs | clinical assertions | no | 0.699 / 0.896 | 0.752 | 0.522 | 0.094 |
| Eigen-raw_coding | coding SNVs | unsupervised | no | 0.917 / 0.858 | 0.947 | 0.923 | 0.095 |
| Eigen-PC-raw_coding | coding SNVs | unsupervised | no | 0.917 / 0.858 | 0.922 | 0.882 | 0.090 |
| GERP++_RS | conservation track | none (conservation) | no | 0.998 / 0.994 | 0.801 | 0.693 | 0.002 |
| GERP_92_mammals | conservation track | none (conservation) | no | 0.570 / 0.777 | 0.763 | 0.673 | 0.146 |
| phyloP100way_vertebrate | conservation track | none (conservation) | no | 1.000 / 1.000 | 0.838 | 0.743 | 0.000 |
| phyloP470way_mammalian | conservation track | none (conservation) | no | 0.929 / 0.913 | 0.820 | 0.752 | 0.049 |
| phyloP17way_primate | conservation track | none (conservation) | no | 1.000 / 1.000 | 0.722 | 0.585 | 0.000 |
| phastCons100way_vertebrate | conservation track | none (conservation) | no | 1.000 / 1.000 | 0.787 | 0.685 | 0.000 |
| phastCons470way_mammalian | conservation track | none (conservation) | no | 1.000 / 0.998 | 0.770 | 0.662 | 0.001 |
| phastCons17way_primate | conservation track | none (conservation) | no | 1.000 / 1.000 | 0.691 | 0.580 | 0.000 |
| bStatistic | conservation track | none (conservation) | no | 0.990 / 0.985 | 0.549 | 0.477 | 0.001 |

Scope, training labels and allele-frequency input are taken from each tool's own publication or documentation and from the dbNSFP 5.3.1a description; a cell that could not be confirmed there is marked unverified. Reach P / B is the share of pathogenic and benign variants a predictor scores on the 328,328-variant panel; covered AUROC is on the pairs it scores; within-gene AUROC pools pathogenic–benign pairs formed within a gene; penalty is covered minus must-answer AUROC. The full table, with the missense-subset columns and every bound, is figures/dbnsfp_49_provenance.csv in Additional file 3.

## Table S41. Covered AUROC of the 49 predictors on labels ClinVar already carried and on newer labels

| Predictor | ClinVar in training | Covered AUROC, older labels | Covered AUROC, newer labels | Shift |
|---|---|---|---|---|
| SIFT | no | 0.8789 | 0.9053 | +0.0265 |
| SIFT4G | no | 0.8678 | 0.9057 | +0.0379 |
| Polyphen2_HDIV | no | 0.8626 | 0.9117 | +0.0491 |
| Polyphen2_HVAR | no | 0.8897 | 0.9291 | +0.0394 |
| MutationTaster | yes | 0.9934 | 0.9897 | −0.0037 |
| MutationAssessor | no | 0.8881 | 0.9332 | +0.0451 |
| PROVEAN | no | 0.8915 | 0.9246 | +0.0330 |
| VEST4 | no | 0.9466 | 0.9436 | −0.0030 |
| MetaSVM | no | 0.9332 | 0.9357 | +0.0025 |
| MetaLR | no | 0.9341 | 0.9375 | +0.0035 |
| MetaRNN | yes | 0.9951 | 0.9840 | −0.0111 |
| M-CAP | no | 0.9115 | 0.9408 | +0.0293 |
| REVEL | test | 0.9664 | 0.9714 | +0.0050 |
| MutPred2 | no | 0.9487 | 0.9655 | +0.0168 |
| MVP | yes | 0.9320 | 0.9530 | +0.0210 |
| gMVP | yes | 0.9610 | 0.9682 | +0.0072 |
| MisFit_D | no | 0.9334 | 0.9490 | +0.0156 |
| MisFit_S | no | 0.8893 | 0.8761 | −0.0133 |
| MPC | yes | 0.8494 | 0.8553 | +0.0059 |
| PrimateAI | no | 0.8846 | 0.8859 | +0.0012 |
| DEOGEN2 | no | 0.9328 | 0.9471 | +0.0143 |
| BayesDel_addAF | yes | 0.9936 | 0.9894 | −0.0042 |
| BayesDel_noAF | yes | 0.9795 | 0.9841 | +0.0047 |
| ClinPred | yes | 0.9939 | 0.9804 | −0.0135 |
| LIST-S2 | yes | 0.8729 | 0.8961 | +0.0233 |
| VARITY_R | yes | 0.9606 | 0.9707 | +0.0101 |
| VARITY_ER | yes | 0.9516 | 0.9675 | +0.0159 |
| VARITY_R_LOO | yes | 0.9552 | 0.9704 | +0.0152 |
| VARITY_ER_LOO | yes | 0.9499 | 0.9675 | +0.0176 |
| ESM1b | no | 0.9451 | 0.9605 | +0.0154 |
| AlphaMissense | calibration | 0.9551 | 0.9619 | +0.0067 |
| PHACTboost | yes | 0.9858 | 0.9784 | −0.0074 |
| MutFormer | no | 0.9333 | 0.9334 | +0.0001 |
| MutScore | yes | 0.9748 | 0.9801 | +0.0054 |
| popEVE | evaluation | 0.9072 | 0.9288 | +0.0217 |
| CADD_raw | no | 0.9546 | 0.9719 | +0.0172 |
| DANN | no | 0.7702 | 0.8526 | +0.0825 |
| fathmm-XF_coding | no | 0.6973 | 0.7675 | +0.0702 |
| Eigen-raw_coding | no | 0.9244 | 0.9544 | +0.0300 |
| Eigen-PC-raw_coding | no | 0.8937 | 0.9310 | +0.0373 |
| GERP++_RS | no | 0.7454 | 0.8181 | +0.0727 |
| GERP_92_mammals | no | 0.7109 | 0.7784 | +0.0676 |
| phyloP100way_vertebrate | no | 0.8001 | 0.8487 | +0.0486 |
| phyloP470way_mammalian | no | 0.7842 | 0.8307 | +0.0465 |
| phyloP17way_primate | no | 0.6485 | 0.7445 | +0.0960 |
| phastCons100way_vertebrate | no | 0.7435 | 0.7998 | +0.0563 |
| phastCons470way_mammalian | no | 0.7220 | 0.7845 | +0.0624 |
| phastCons17way_primate | no | 0.6708 | 0.6957 | +0.0249 |
| bStatistic | no | 0.5635 | 0.5423 | −0.0213 |

Covered AUROC on the dbNSFP panel's labels that ClinVar's release of fileDate 2021-01-02 already carried with the same class (older) and on the rest (newer), and the shift from older to newer. ClinVar in training follows the provenance table, figures/dbnsfp_49_provenance.csv in Additional file 3: yes where ClinVar labels trained or tuned the predictor, no where they did not, and test, calibration or evaluation where ClinVar was used only for that. Older and newer are disjoint label sets; the newer are also easier to separate by gene alone (Note S61), so a shift here is not a measure of training overlap on its own. Computed and printed by section F of recompute_identification.py in Additional file 3.

## Table S42. Calibrated PP3 and BP4 levels as likelihood ratios on ClinVar missense variants

| Tool and level | Likelihood ratio, all labels (95% CI) | Pathogenic / benign variants | Stratum closest to target | Target odds |
|---|---|---|---|---|
| REVEL, BP4 very strong | 0, no pathogenic variant at the level | 0 / 736 | 0, all labels | ≤ 1/350 |
| REVEL, BP4 strong | 0.0032 (0.0015 to 0.0050) | 13 / 10,238 | 0.0053, added 2021 to 2023 | ≤ 1/18.7 |
| REVEL, BP4 moderate | 0.024 (0.021 to 0.028) | 830 / 85,152 | 0.034, added 2021 to 2023 | ≤ 1/4.33 |
| REVEL, BP4 supporting | 0.14 (0.12 to 0.16) | 1,131 / 20,115 | 0.19, one star | ≤ 1/2.08 |
| REVEL, indeterminate | 0.96 (0.85 to 1.10) | 8,255 / 21,398 | – | – |
| REVEL, unscored | 0.18 (0.11 to 0.27) | 252 / 3,515 | – | – |
| REVEL, PP3 supporting | 6.81 (6.00 to 7.77) | 7,327 / 2,685 | 4.46, added 2021 to 2023 | ≥ 2.08 |
| REVEL, PP3 moderate | 30.9 (26.9 to 35.1) | 21,486 / 1,738 | 18.9, added 2021 to 2023 | ≥ 4.33 |
| REVEL, PP3 strong | 217 (178 to 263) | 19,100 / 220 | 149, added 2021 to 2023 | ≥ 18.7 |
| AlphaMissense, BP4 moderate | 0.021 (0.018 to 0.023) | 673 / 81,624 | 0.025, added 2021 to 2023 | ≤ 1/4.33 |
| AlphaMissense, BP4 supporting | 0.15 (0.13 to 0.16) | 1,498 / 25,602 | 0.17, added after 2023 | ≤ 1/2.08 |
| AlphaMissense, indeterminate | 1.38 (1.25 to 1.51) | 13,503 / 24,496 | – | – |
| AlphaMissense, unscored | 0.25 (0.15 to 0.38) | 1,019 / 10,177 | – | – |
| AlphaMissense, PP3 supporting | 9.37 (8.38 to 10.5) | 6,447 / 1,718 | 6.95, added 2021 to 2023 | ≥ 2.08 |
| AlphaMissense, PP3 moderate | 23.3 (20.9 to 25.9) | 16,374 / 1,757 | 15.2, added 2021 to 2023 | ≥ 4.33 |
| AlphaMissense, PP3 strong | 111 (93.9 to 132) | 18,880 / 423 | 73.7, added 2021 to 2023 | ≥ 18.7 |
| CADD, BP4 strong | 0.0039 (0.0023 to 0.0058) | 30 / 19,040 | 0.0078, added 2021 to 2023 | ≤ 1/18.7 |
| CADD, BP4 moderate | 0.029 (0.026 to 0.033) | 828 / 70,858 | 0.033, added 2021 to 2023 | ≤ 1/4.33 |
| CADD, BP4 supporting | 0.29 (0.26 to 0.32) | 3,123 / 26,844 | 0.38, added after 2023 | ≤ 1/2.08 |
| CADD, indeterminate | 1.50 (1.36 to 1.65) | 10,207 / 16,985 | – | – |
| CADD, unscored | 0.31 (0 to 0.49) | 1 / 8 | – | – |
| CADD, PP3 supporting | 5.76 (5.37 to 6.21) | 17,821 / 7,719 | 4.18, added 2021 to 2023 | ≥ 2.08 |
| CADD, PP3 moderate | 15.2 (13.9 to 16.5) | 26,384 / 4,343 | 9.11, added 2021 to 2023 | ≥ 4.33 |
| phyloP, BP4 moderate | 0.042 (0.037 to 0.049) | 770 / 45,609 | 0.062, carried by the 2021 release | ≤ 1/4.33 |
| phyloP, BP4 supporting | 0.19 (0.18 to 0.21) | 3,699 / 48,144 | 0.22, added 2021 to 2023 | ≤ 1/2.08 |
| phyloP, indeterminate | 1.18 (1.09 to 1.28) | 19,810 / 41,841 | – | – |
| phyloP, unscored | 0, no pathogenic variant at the level | 0 / 5 | – | – |
| phyloP, PP3 supporting | 7.98 (7.32 to 8.75) | 28,841 / 9,026 | 5.83, added 2021 to 2023 | ≥ 2.08 |
| phyloP, PP3 moderate | 11.2 (9.39 to 13.6) | 5,274 / 1,172 | 7.88, added 2021 to 2023 | ≥ 4.33 |

Each calibrated level on the 204,191 ClinVar missense variants at one review star or better: the ratio of the share of pathogenic variants at the level to the share of benign ones, with genes resampled (B = 2,000; Note S71). Unscored is a declined variant, taken as its own level; indeterminate lies between the innermost PP3 and BP4 bounds. The stratum closest to target is the one of six (all labels; carried by the 2021 release; added 2021 to 2023; added after 2023; one star; two stars or more) in which the ratio lies nearest the odds its strength targets, given as its ratio and name. Targets are the point-system odds of the ACMG/AMP Bayesian framework [49]: 2.08, 4.33, 18.7 and 350 for supporting to very strong PP3, and their reciprocals for BP4; the calibrations [31,46] set their thresholds within the same framework. Levels with very few variants carry ratios that describe those variants only: CADD's unscored ratio rests on one pathogenic variant, so its resampled interval understates its uncertainty, and phyloP's unscored level holds no pathogenic variant. Computed and printed by recompute_pp3bp4.py, section 6; each level's ratio at its inner threshold is in Note S71. Ratios are given to three significant figures from 1 upwards and to two below 1.
