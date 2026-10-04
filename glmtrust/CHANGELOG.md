# Changelog

All notable changes to `glmtrust` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [0.1.1] - 2026-08-29

The deposited package: the `audit`, `reach` and `baseline` commands, the `--card` report, `delong` and
`--auto`, with the package, citation, documentation and runtime version metadata of the submitted deposit.

The audit gains the inference and the speed it needed to be used on real panels, and stops crying
wolf on large ones.

### Added
- `audit(..., strata=)` — hold variant composition fixed and ask whether class-dependent reach
  survives it, separating two defects that are otherwise confounded. Strata are screened on **both**
  classes (`min_class`), not on total size: a 184-variant stratum carrying one positive otherwise
  produces a confident-looking reach gap of +0.585 estimated from that single variant.
- `missingness_auroc` and `ScorerAudit.miss_auroc` — the AUROC of the missingness pattern with the
  scores discarded entirely. Pooled this equals `0.5 + class_gap/2` exactly, and the docstring says
  so; it is reported because it is directly comparable with must-answer on the same panel.
- `PairAudit.delta_must_answer` with a closed-form interval. The decision-relevant comparison —
  which scorer is better over the whole panel, each carrying its own no-calls — previously had no
  inference attached at all.
- `glmtrust.delong` — DeLong's method in the O(n log n) form of Sun & Xu (2014), plus
  `must_answer_placements`, which expresses must-answer AUROC as a Mann-Whitney statistic in which an
  unreachable variant has placement value exactly ½.
- `glmtrust.card` — an audit report as one self-contained HTML page: no external assets, no scripts,
  light and dark, printable. It leads with whether the audit *changed* the verdict, and states
  plainly when nothing is wrong.
- Join-integrity check. Two scorers overlapping far below what their reaches imply is the signature
  of a merge on mismatched keys, which otherwise yields a plausible AUROC over a handful of rows.
- `glmtrust audit --auto` — detect score columns instead of naming them, printing both the selection
  and what was skipped, so an annotation table needs no column bookkeeping.
- `--card`, `--strata-col`, `--min-stratum` and `--min-class` on the CLI; `min_gap` and `min_penalty`
  are arguments of the Python `audit()`.
- `lexicographic_gain` and `ScorerAudit.lex_gain` / `lex_gain_ci` / `values_add_to_reach` — what a
  scorer's values add beyond whether it produced one: ranking by the missingness pattern and then by
  value beats the pattern alone by exactly rho * (A_cov - 1/2). A scorer at chance on its covered set
  gains nothing, so this test can fail on the values.
- `audit(..., cluster=)` and `--cluster-col` — every interval becomes a percentile bootstrap over whole
  groups (gene, locus): class gap, the gain above, and both head-to-head deltas. The report states the
  interval level either way.
- `--lower-is-worse` on the CLI, matching `Scorer(higher_is_worse=False)`, and a warning whenever a
  covered AUROC falls below one half, the usual sign of a score stored the other way round.
- `reach_audit`, `ReachReport`, `ReachScorer` and `glmtrust reach` — reach accounting from reach
  indicators alone, for panels whose values cannot be shared: reach, class gaps, rho, both feasibility
  bars and, given covered AUROCs, identified pairs under the sharp bounds and under monotone coverage.
- `benchmarks/reproduce_paper_reach.py` reproduces the dbNSFP pair counts from the deposited reach
  panel and per-predictor covered AUROCs; `benchmarks/fixtures/trust_layer_8192.json` ships the
  reference the trust-layer benchmark checks against, so it runs outside the code deposit.
- `contamination_bounds` and `breakdown_point` — the breakdown frontier of a comparison. Under
  lambda-contamination (Horowitz & Manski) a share lambda of the unscored pairs resolves arbitrarily
  and the rest like the scored ones, which puts the whole-panel AUROC in
  [A - (1 - rho) lambda A, A + (1 - rho) lambda (1 - A)]: the covered AUROC at lambda = 0 and the
  sharp bound at lambda = 1, to the last bit. A pair stays ordered for every lambda below its
  breakdown point lambda* (Masten & Poirier).
- `reach_audit(..., lambdas=)` and `glmtrust reach --lambdas` — with covered AUROCs the report
  carries `breakdown` (lambda* for every pair), `frontier` (pairs ordered at each lambda, by the
  strict test behind `identified`, so the count at lambda = 1 is `identified` and the audit checks
  it) and `median_breakdown_undecided`, prints them as a FRONTIER block after PAIRS, and writes all
  three into the `--out` JSON. On the deposited dbNSFP panel 1,107 of 1,176 pairs stay ordered at
  lambda = 0.01, 83 at lambda = 1, and the 1,093 undecided pairs have a median lambda* of 0.1118.
- `glmtrust.baseline`: `sequence_blind` and `glmtrust baseline` — sequence-blind baselines, the
  out-of-fold positive rate, shrunk with a pseudo-count, of a variant's group (gene), class (consequence) or
  (group, class) cell shrunk towards its class rate, each read as a midrank AUROC averaged over fold
  seeds. The fold assignment is scikit-learn's `StratifiedKFold(shuffle=True)` byte for byte,
  reimplemented so the values do not depend on the installed scikit-learn; `stratified_kfold_folds`,
  `oof_rate`, `oof_group_within_class` and `midrank_auroc` are public.
- `benchmarks/reproduce_paper_baseline.py` reproduces the paper's sequence-blind baselines from
  Additional file 4 alone: gene identity 0.880, consequence class 0.837, gene and consequence 0.974.
- `tests/test_breakdown.py` and `tests/test_baseline.py`: the lambda = 0 and lambda = 1 limits,
  symmetry, a hand-computed pair, a non-increasing frontier, the fold assignment against the documented
  algorithm and against scikit-learn, and the priors against a row-by-row counting form.

### Changed
- The warning about reach and values uses `values_add_to_reach`. Comparing the oriented missingness
  AUROC with the must-answer AUROC is kept as the descriptive `missingness_beats_score`, since the
  one-half rule decides that comparison for the pattern whenever |r_pos - r_neg| > r_pos * r_neg,
  however good the values. The HTML card flags the same rule.
- Warnings now require practical **and** statistical significance (`min_gap`, `min_penalty`). On a
  1.4-million-variant ClinVar panel CADD's reach gap of +0.0006 excludes zero; flagging that beside
  AlphaMissense's +0.229 teaches the reader to ignore the flag. Every number is still printed — only
  the warnings are gated, and the bare statistical test remains available as
  `reach_gap_is_significant`.
- Head-to-head intervals use DeLong above `delong_above` (default 20,000 matched variants) and the
  paired bootstrap below it. A full 1,434,335-variant ClinVar audit takes 42 seconds rather than more
  than half an hour. Just above the switch, at 22,000 matched variants, the two intervals differ by at
  most 0.0002 at either end, the Monte Carlo error of a 2,000-draw bootstrap.
- The missingness interval is the Newcombe interval on the class gap pushed through the identity
  instead of a percentile bootstrap: exact, and better behaved at the boundary.
- Package description and keywords now name the audit, which was invisible on the distribution
  metadata.

### Fixed
Every public entry point now reads input by one contract (`glmtrust._checks`): labels are 0/1 with
every one present, a score is a real number per variant with NaN for a no-call, one value per
variant, and anything else is refused by name with the recoding to apply.
- Labels coded 0.7, NaN, 1/2, -1/+1 or as text were cast silently by the calibrators, the conformal
  classes, `leave_one_group_out`, `missingness_auroc` and DeLong; they are now refused, with the
  recoding to apply.
- `PlattCalibrator` fits on the score centred on its median and scaled by its standard deviation,
  which leaves the fitted map a sigmoid of the raw score. On the raw scale a score of small magnitude
  (likelihood deltas below about 3e-4) stopped the solver at its first iteration and every
  probability collapsed to the prevalence; the fit is now independent of the score's units and agrees
  with the unpenalised maximum-likelihood logistic to 1e-6 in the tests, including on a skewed,
  SpliceAI-like score. A fit left nearly flat although the score separates the classes, which a few
  values near the float limit can cause, raises a RuntimeWarning.
- An infinite score is refused instead of being counted as a no-call; a masked entry is a no-call; a
  2-D input other than a single row or column is refused.
- `audit` and `reach_audit` apply `alpha` to every interval (reach, class gap, missingness); with
  `cluster=` the reach and missingness intervals come from the same whole-cluster draws as the class
  gap. An undefined interval no longer counts as significant. Duplicate scorer names, a missing
  readout, `higher_is_worse` given as text, `n_boot` below 10 and `min_class` below 1 are refused.
  Strata are indexed once, so one stratum per variant no longer takes quadratic time. A scorer whose
  values run opposite to the declared direction is named as such, and one that reaches every variant
  is no longer said to be carried by its reach. An undefined interval on the values' gain (a class
  with a single scored variant) is reported as undefined, not as the values adding nothing. Long
  scorer and stratum names are shortened with '...', and printed in full if shortening would make two
  alike. With `cluster=` the report says that the within-stratum intervals stay variant-level.
  `AuditReport.alpha`, `ScorerAudit.values_run_backwards`, `values_add_nothing` and
  `values_gain_undefined` are new, and the docstrings say which intervals depend on row order.
- `reach_audit` matches scorer names as strings in `reach` and `covered_auroc`, reads pandas missing
  markers, reports how it read each column (`ReachReport.read_as`), refuses a 0/1 or boolean column
  that also has missing entries (it could be an indicator or a score, and the two readings give
  different reach), and warns on a covered AUROC below one half. The report says that both
  feasibility counts use reach alone.
- `metrics.selective_lift` divides by the fraction actually refused, so random refusal scores 1; kept
  and refused counts always partition the panel; capture with no errors is NaN everywhere; lengths,
  masks, `n_bins`, `n_boot`, `alpha` and `seed` are checked; `auroc_ci` returns (nan, nan) instead of
  failing when no resample carries both classes. DeLong treats a masked entry as missing and names a
  scorer of the wrong length.
- `conformal_quantile` computes its rank in exact rational arithmetic and refuses `alpha` outside
  (0, 1); conformal and selective layers refuse NaN or out-of-range calibration probabilities.
- `precision_operating_point` thresholds only between distinct score values, so the certified set is
  the flagged set when scores tie; NaN scores and `delta` outside (0, 1) are refused.
- `TrustLayer` checks its settings when built and when used; `fit` is all-or-nothing; `evaluate` keeps
  no state between calls, accepts `seed=None`, reports the raw score's AUROC with its DeLong interval
  (a score is flagged as running the other way only when that interval lies wholly below one half),
  refuses more folds than the smaller class has variants, applies each group's own refusal exactly
  in the selective figures, and uses DeLong's closed form for the AUROC interval from 20,000 variants;
  `predict` gives a declined (NaN) variant probability NaN and ABSTAIN; `summary(report=...)` renders a
  report without evaluating again and prints its warnings before the numbers.
- `leave_one_group_out` predicts a held-out group with a single class of its own, which is the
  label-free target this function exists for; a missing group label, a single group, and a panel on
  which no group can be predicted are refused. `group_selective_report` warns when variants without a
  probability are left out, counts them in `pooled`, and refuses input with no probability at all.
- `midrank` refuses NaN instead of looping forever; `sequence_blind`, `oof_rate` and
  `oof_group_within_class` refuse more folds than either class has variants, and
  `stratified_kfold_folds` refuses what scikit-learn refuses and warns where it warns;
  `midrank_auroc` reads labels and scores by the same contract as everything else. `render_card`
  refuses a reach report, labels its intervals at the audit's level, calls a verdict reversed only
  when the matched interval excludes zero (an opposite-signed delta whose interval spans zero is
  called inconclusive), and cards a scorer that reaches every variant as not better than chance
  rather than as carried by its reach.
- The command line reads semicolon-separated files with decimal commas, UTF-8 with a byte-order mark,
  UTF-16, Windows-1252, gzip and parquet files whatever their name, and refuses spreadsheets,
  folders, duplicate column names and over-long rows by name. Labels may be text named with
  `--positive-label`/`--negative-label`; an unlabelled variant stops the run unless `--drop-unlabelled`
  is given. `--lower-is-worse` applies to every command. `calibrate` and `transfer` write one row per
  input row, numbered from 0, with `--id-col` carried through. A probable no-call sentinel (a
  conventional code such as -999, or a value set apart from the rest, holding 10% or more of a column
  at its extreme) is noted, and so are exact duplicate rows when some column identifies the rows; a
  number written with a comma in a comma-separated file is refused with the reason; an output folder
  that does not exist is refused before the computation; `--auto` skips coordinate-like columns such
  as `pos(1-based)`; `transfer` exits with an error when no group can be predicted and otherwise
  says how many variants got no probability; `reach` names a `--reach-prefix` that matches nothing
  and the `--covered` columns it looked for; the `audit --out` JSON records `interval_level` and
  `alpha`; any input the package refuses ends in one line rather than a traceback.
- `notebooks/quickstart.ipynb` opens in Google Colab, and its first cell fetches the repository and
  installs the package when they are missing.

### Notes
- `TrustLayer.evaluate` fits each fold's calibration map on half of the fold's training variants and its
  conformal layer on the other half, so that the conformal quantile is fitted on probabilities the map
  has not seen. On the shipped fixture its selective figures read a capture of 0.6124 and a lift of 4.09
  (`notebooks/quickstart.ipynb`). `benchmarks/reproduce_paper_trust_layer.py`, which fits each map on the
  whole training fold through `leave_one_group_out` and `group_selective_report`, reproduces the deposited
  numbers exactly.
- 246 tests, including tests that check the documentation matches the code and one for each input a
  stranger is likely to bring; `pytest --collect-only` reports 246, which is the number that ships.

## [0.1.0] - 2026-07-24

First release. Extracts the trust-layer methodology from the accompanying cross-species study into a
tested, installable package.

### Added
- `TrustLayer` — the whole layer as one object (calibrate → conformal → selective), with a deployment
  path (`fit`/`predict`) and an honest cross-validated `evaluate`.
- `calibration` — `IsotonicCalibrator`, `PlattCalibrator`, and out-of-fold `cross_conformal_calibrate`.
- `conformal` — `SplitConformal` (marginal) and `MondrianConformal` (class-conditional) with
  finite-sample-valid quantiles (valid under exchangeability; see the README scope note).
- `selective` — `SelectivePredictor` (coverage operating point), `precision_operating_point`
  (RCPS-style certified precision) and `group_selective_report` (per-group operating points), with
  class-asymmetric capture accounting.
- `transfer` — `leave_one_group_out` and `transfer_calibration` for the label-free-target setting.
- `benchmarks/reproduce_paper_trust_layer.py` — reproduces the study's deposited 8,192-bp trust-layer
  numbers (pooled capture 0.6096, lift 4.07, macro ECE 0.0209, 415 errors) exactly, through the public
  API, as a check that the package captures the method rather than resembling it.
- `metrics` — ECE (uniform/quantile), Brier, AUROC with bootstrap CI, risk-coverage curve, capture,
  lift.
- A command-line interface, `glmtrust evaluate | calibrate | transfer`; 0.1.1 adds `audit`, `reach` and
  `baseline`, and the `--card` report of `audit`.
- Test suite covering coverage guarantees, calibration improvement, and selective lift; the
  deposited suite collects 246.
