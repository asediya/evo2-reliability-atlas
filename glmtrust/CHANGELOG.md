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
  1.4-million-variant panel every reach gap excludes zero, including gaps of 0.001 worth 0.0007
  AUROC; flagging those beside a genuine +0.229 teaches the reader to ignore the flag. Every number
  is still printed — only the warnings are gated, and the bare statistical test remains available as
  `reach_gap_is_significant`.
- Head-to-head intervals use DeLong above `delong_above` (default 20,000 matched variants) and the
  paired bootstrap below it. A full 1,434,335-variant ClinVar audit takes 42 seconds rather than more
  than half an hour, and the two methods agree to the last reported digit.
- The missingness interval is the Newcombe interval on the class gap pushed through the identity
  instead of a percentile bootstrap: exact, and better behaved at the boundary.
- Package description and keywords now name the audit, which was invisible on the distribution
  metadata.

### Notes
- `TrustLayer.evaluate` fits each fold's calibration map on half of the fold's training variants and its
  conformal layer on the other half, so that the conformal quantile is fitted on probabilities the map
  has not seen. On the shipped fixture its selective figures read a capture of 0.6124 and a lift of 4.08
  (`notebooks/quickstart.ipynb`). `benchmarks/reproduce_paper_trust_layer.py`, which fits each map on the
  whole training fold through `leave_one_group_out` and `group_selective_report`, reproduces the deposited
  numbers exactly.
- 166 tests, including tests that check the documentation matches the code;
  `pytest --collect-only` reports 166, which is the number that ships.

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
  deposited suite collects 166.
