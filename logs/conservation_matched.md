# Conservation-matched discrimination: is Evo2-40B's AUROC real pathogenicity or a region confound?

Negatives are resampled (quantile-binned on the POSITIVE GERP distribution, up to 5:1, 40 reps, seed 0)
so they carry the SAME conservation profile as the positives. If AUROC survives, the signal is
pathogenicity, not conserved-region-vs-intergenic. goat excluded (<10 positives).

| species | n | pos | AUROC (all-neg) | AUROC (cons-matched) | drop | conservation-alone |
|---|---|---|---|---|---|---|
| chicken | 274 | 27 | 0.862 | **0.818** +/- 0.008 | +0.044 | 0.725 |
| pig | 168 | 27 | 0.814 | **0.804** +/- 0.003 | +0.010 | 0.751 |
| sheep | 451 | 41 | 0.912 | **0.911** +/- 0.002 | +0.001 | 0.865 |
| horse | 379 | 66 | 0.882 | **0.846** +/- 0.003 | +0.035 | 0.915 |
| cat | 1037 | 117 | 0.859 | **0.837** +/- 0.003 | +0.022 | 0.846 |
| cattle | 1985 | 188 | 0.900 | **0.851** +/- 0.006 | +0.049 | 0.825 |
| dog | 2278 | 209 | 0.895 | **0.846** +/- 0.002 | +0.049 | 0.886 |
| human | 2880 | 1445 | 0.825 | **0.761** +/- 0.003 | +0.064 | 0.874 |

**Mean across 8 species: AUROC(all-neg) 0.869 -> conservation-matched **0.834** (drop +0.034).**

**Verdict:** Evo2-40B's discrimination SURVIVES conservation-matching (mean ~0.83, range 0.76-0.91) — it is
REAL pathogenicity signal, NOT merely conserved-region-vs-intergenic separation. This addresses the region
confound flagged in the audit. Honest scope: the signal is real but MODEST and does not cleanly beat
conservation (see beats-conservation FDR = 2/9); the FM adds orthogonal signal (decomposition dFM +0.020),
it does not dominate the conservation baseline. Matching is on GERP; a consequence/genic match is a further
control a reviewer may request.
