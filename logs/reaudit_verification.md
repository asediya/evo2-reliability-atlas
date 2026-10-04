# Live re-audit trail (run one-by-one with the user, no black-box agents)

Every finding below was re-verified by RUNNING the actual code live and reading the raw output.
Verdicts: CONFIRMED / OVERSTATED / REFUTED.

## #1 — Calibration transfer ties a trivial global sigmoid — CONFIRMED
Ran `python src/ccs/calibration_transfer.py` (the sigmoid baseline is in the committed script, `baseline_report`, lines 122-205). Raw output, equal-width ECE, global_sigmoid vs iso_transfer:
goat 0.048/0.056, chicken 0.021/0.020, pig 0.037/0.038, sheep 0.055/0.053, horse 0.044/0.043, cat 0.037/0.041, cattle 0.071/0.061, dog 0.049/0.050, human 0.134/0.117.
**Sigmoid wins 4/9; median ECE sigmoid 0.048 vs isotonic 0.050; Brier 0.041 vs 0.040 (tied).** The cross-species-transfer *novelty* is neutralized — a 2-param sigmoid does as well.
Prevalence grid (tightened): dog 0.05@9% -> 0.028@25% -> 0.137@50% = NON-MONOTONE (fine near the 9% operating point; degrades only toward 50%). (The script's own verdict text still says "inflates sharply" — to be softened.)

## #2 — phyloP fabricated (byte-identical copy of GERP) for 4 species — CONFIRMED
Compared the phyloP track (`data/interim/<sp>_gerp_phylop.parquet`, col `phylop`) against the GERP file (`data/processed/conservation/<sp>_gerp.parquet`, col `gerp`), masking to positions where BOTH are finite:
- pig   n=168  corr=1.000000  exact-equal=1.0000  allclose=True
- sheep n=451  corr=1.000000  exact-equal=1.0000  allclose=True
- horse n=379  corr=1.000000  exact-equal=1.0000  allclose=True
- dog   n=2278 corr=1.000000  exact-equal=1.0000  allclose=True
- cattle (control) corr=0.57 on 10 overlapping pts = GENUINELY DISTINCT (allclose=False)
**CONFIRMED.** `build_atlas.py` already sets `phylop=None` for all 9 (self-heal), GERP is the sole conservation baseline (numbers unchanged). NOTE: first check gave nan — a NaN-masking bug in the CHECK, fixed by masking to finite values; the audit finding stands.

## #6 — "beats conservation 6/9" was wrong; real is 7/9-sign / 3/9-nominal / 2/9-FDR — CONFIRMED
Loaded `data/processed/atlas_40b.json`, computed one-sided p = 1 - p_gt_cons, applied Benjamini-Hochberg:
positive-delta by sign = 7/9; nominal one-sided p<0.05 = 3/9 (goat, chicken, cattle); BH-FDR q<=0.05 = **2/9 (goat, cattle)**. The "6/9" figure does NOT reproduce (sign=7, nominal=3). CONFIRMED.

## #7 — clean "LoF > missense" no longer holds — CONFIRMED
Re-ran `python src/ccs/annotate_consequence.py`: nonsense 6.863 (n=112) > missense 5.961 (n=324) > splicing 3.774 (n=70) > regulatory 0.282 (n=18); LoF-group (incl splice) 5.675 < missense 5.961. Script prints "Hierarchy LoF > missense > regulatory holds: **False**". Robust order = coding/splice >> regulatory. CONFIRMED.

## #3 — conformal: 0.97 coverage costs 50-66% abstention (no best-of-both-worlds) — CONFIRMED (+md prose fixed)
Re-ran `python src/ccs/build_conformal.py`. Mondrian TARGET coverage goat/chicken/pig = 0.970/0.984/0.957 at abstain {both} = **0.505/0.627/0.657**; marginal near-singleton keeps ~0.95 overall but pathogenic-class coverage collapses to **0.222/0.571/0.444**. No single predictor gives both. CONFIRMED.
FIXED the md over-claim: `build_conformal.py` title + VERDICT still said "distribution-free finite-sample GUARANTEE" (self-heal fixed the table, not the prose). Softened to "coverage under a stated, tested exchangeability assumption; over-covers by abstaining 50-66%; NOT a theorem". Re-ran -> `logs/conformal.md` now honest.

## #4 — decision panel: always-abstain (0.500) beats every learned cell — CONFIRMED
Re-ran `python src/ccs/build_decision_panel.py`. Trivial always-abstain T1 = **0.500**; best learned Evo2 conformal 0.540, NT reject 0.540 — both ABOVE 0.500. Script's own text: "the honest conclusion is the opposite of the original punchline". CONFIRMED (md already honest from self-heal).

## #8 — low-conservation result is 78.7% human — CONFIRMED
Imported the script's own `merged()` loader; low-cons = `cons < median(cons)` per species (build_decomposition.py:98). Per-species low-cons positives: chicken 6, pig 8, sheep 2, horse 3, cat 15, cattle 23, dog 13, human 258. **Total 328; human 258 = 78.7%.** Non-human pooled: n=3195, pos=70, AUROC=**0.7852**. 6 of 7 non-human species have <20 positives (only cattle 23) = noise-dominated. CONFIRMED (matches tightened "6/7").

## #9 — CP1 "6-sigma" / "0.53 ceiling" have no producing script — CONFIRMED
grep across src/ for "sigma|0.53|ceiling" finds NO script computing a 6-sigma residual or 0.53 composition ceiling (only unrelated uses of "ceiling"). The REAL control `verify/check_composition.py` prints: composition-only OOF AUROC **0.2545**, trinuc chi2 p=**1.000**, GC MWU p=0.620 -> "matched (composition uninformative)". The retracted 0.53/6-sigma numbers were unreproducible text; retraction CONFIRMED.

## #5 — ClinVar "ECE -> 0.001" is a noise tautology — CONFIRMED
Re-ran `python src/ccs/build_clinvar_calibration.py` (150k sample, base rate 0.1712). RANDOM noise (AUROC 0.503) -> calibrated ECE **0.0001**; base-rate constant -> calibrated ECE **0.0000** = as low as the real scores. So the ECE-collapse is isotonic regression, NOT skill. Honest metric = Brier skill (noise 0.000 vs real 0.68-0.92). EVEE probe circular (label pearson 0.957, AUROC 0.997) -> dropped; external raw-ECE median 0.074 (CADD 0.23/REVEL 0.074/AlphaMissense 0.037). CONFIRMED. md regenerated honest.

---
## FINAL SCORECARD (live one-by-one): 9/9 CONFIRMED, 0 REFUTED.
Every prior finding reproduced by running the actual code with the user watching. Nothing hallucinated. Prose fixes applied along the way: conformal.md (dropped "distribution-free GUARANTEE"), calibration_baseline.md (prevalence "inflates sharply" -> non-monotone). The five earlier framing overstatements were already tightened in results-ledger.md.
