# API reference

Import everything from the top level: `from glmtrust import audit, Scorer, TrustLayer, ...`.

## `glmtrust.audit`

Audit a comparison between variant-effect scorers before believing its headline.

```python
Scorer(name, score, readout, higher_is_worse=True)
```

`score` is one value per panel variant, with **NaN** where the scorer produced nothing — not `None`,
not a sentinel, and not an imputed value. `readout` is **required**: how a model's output becomes one
number per variant can invert a verdict, so scorers declaring different readouts are refused a delta
rather than quietly differenced.

```python
audit(labels, scorers, strata=None, n_boot=2000, seed=0, alpha=0.05,
      min_stratum=40, min_class=10, min_gap=0.02, min_penalty=0.005,
      delong_above=20_000, cluster=None) -> AuditReport
```

| Argument | Meaning |
|---|---|
| `labels` | 0/1 over the **whole** panel, including variants some scorer cannot reach |
| `strata` | one label per variant (usually consequence class); separates class-dependent reach from composition |
| `min_stratum` / `min_class` | a stratum is reported only with this many variants, and this many of **each** label. Screening on total size alone lets a 184-variant stratum holding one positive produce a confident-looking gap |
| `min_gap` / `min_penalty` | practical-significance floors on the *warnings*. At a million variants every gap excludes zero; numbers are always printed, only warnings gated |
| `delong_above` | matched-panel size at which head-to-head intervals switch from paired bootstrap to DeLong's closed form. `None` to always bootstrap |
| `cluster` | one group label per variant (gene, locus). Every interval then becomes a percentile bootstrap over whole groups, and `AuditReport.interval_level` says so; without it intervals are variant-level |

### `ScorerAudit`

| Field / property | Meaning |
|---|---|
| `reach`, `reach_pos`, `reach_neg` | fraction scorable overall and per class, with `reach_pos_ci` / `reach_neg_ci` |
| `class_gap`, `class_gap_ci` | positive minus negative reach (Newcombe interval) |
| `auroc_covered` | accuracy on the variants it can score — the number benchmarks usually quote |
| `auroc_must_answer` | accuracy over the whole panel it is reported for, unscorable variants contributing no information |
| `penalty` | the difference between those two |
| `miss_auroc`, `miss_auroc_ci` | AUROC of the missingness pattern with the scores discarded. Pooled this equals `0.5 + class_gap/2` exactly |
| `miss_auroc_stratified` | the same held within strata — **not** an identity, and the quantity a composition-matched benchmark is still exposed to |
| `reach_gap_is_significant` | interval on the gap excludes zero (statistical only) |
| `reach_is_class_dependent` | significant **and** at least `min_gap` (what the warnings use) |
| `missingness_beats_score` | the oriented `miss_auroc` exceeds `auroc_must_answer`. Descriptive only: the one-half rule discards what the pattern knows, so this can hold for perfect values |
| `lex_gain`, `lex_gain_ci` | what ranking by the missingness pattern and then by value gains over the pattern alone, ρ(A_cov − ½), with its interval |
| `values_add_to_reach` | the interval on `lex_gain` excludes zero: the values carry information beyond where the scorer answers. The warnings use this |
| `strata` | list of `StratumAudit`, `n_strata_dropped` counts those too small to test |

### `PairAudit`

`delta_matched` (+ `delta_matched_ci`) on the variants both reach, `delta_as_usually_reported` with
each on its own covered subset, and `inflation` between them. `comparable_readout` is False when the
readouts differ, in which case no delta is computed.

### Free functions

- `must_answer_auroc(auroc_covered, k_pos, n_pos, k_neg, n_neg)` — the closed form. A pair touching an
  unscorable variant contributes ½, so no imputation is involved.
- `missingness_auroc(labels, observed)` — scores the missingness indicator alone.
- `wilson_interval(k, n, z=1.96)`.
- `lexicographic_gain(auroc_covered, k_pos, n_pos, k_neg, n_neg)` — ρ(A_cov − ½), exactly what ranking by
  the missingness pattern and then by value gains over the pattern alone.
- `contamination_bounds(auroc_covered, rho, lam)` → `(lo, hi)`: the whole-panel AUROC when a share λ
  of the unscored pairs may resolve arbitrarily and the rest resolve like the scored ones
  (Horowitz & Manski, 1995): [A − (1 − ρ)λA, A + (1 − ρ)λ(1 − A)]. The point A at λ = 0; at λ = 1
  the same floats as the sharp identification bounds [ρA, ρA + 1 − ρ].
- `breakdown_point(auroc_i, rho_i, auroc_j, rho_j)` → λ* = (A_i − A_j) / ((1 − ρ_i)A_i + (1 − ρ_j)(1 − A_j))
  for A_i > A_j (Masten & Poirier, 2020): the pair is ordered at every λ < λ*. Symmetric in the pair;
  above 1 the sharp bounds are already disjoint; 0 when A_i = A_j; `inf` when the denominator is 0
  (both scorers reach every pair).

### `reach_audit`

```python
reach_audit(labels, reach, covered_auroc=None, cluster=None, n_boot=2000, seed=0, alpha=0.05,
            lambdas=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0)) -> ReachReport
```

Reach accounting from reach indicators alone, for panels whose score values cannot be shared. `reach`
maps each scorer to a boolean or 0/1 indicator (or a raw score, NaN where declined); `covered_auroc`
maps scorers to their covered AUROCs and turns the pair counts into identification verdicts.
`ReachReport` carries one `ReachScorer` per scorer (reach, class gap, ρ, and with a covered AUROC the
must-answer value and the bounds), and `feasible` (ρ_i + ρ_j > 1), `feasible_sharp`
(ρ_max + ρ_min/2 > 1, both covered AUROCs at least ½), `identified` (disjoint sharp bounds) and
`identified_monotone` (disjoint [ρA, A], unscored variants no easier than scored). The CLI form is
`glmtrust reach TABLE --label-col L --reach-prefix reach__ --covered COVERED.csv`.

With covered AUROCs it also carries the breakdown frontier:

| Field | Meaning |
|---|---|
| `breakdown` | `(name_i, name_j, λ*)` for every pair, `name_i` the higher covered AUROC |
| `frontier` | `{λ: pairs ordered at λ}` for each λ in `lambdas`, by the strict disjointness test that decides `identified`; the count at λ = 1 equals `identified`, and `reach_audit` checks that it does |
| `median_breakdown_undecided` | the median λ* over the pairs `identified` leaves undecided (NaN when there are none) |

On the CLI, `--lambdas 0.01,0.05,0.1,0.2,0.5,1` sets the shares, and the JSON written by `--out` gains
`breakdown`, `frontier` and `median_breakdown_undecided`.

## `glmtrust.baseline`

Sequence-blind baselines: the AUROC a panel's own labels give without reading any sequence or score.

```python
sequence_blind(labels, groups=None, classes=None, folds=5, alpha=1.0, seeds=range(8)) -> SequenceBlindReport
```

Each baseline scores a variant by the positive rate of a category, shrunk with pseudo-count `alpha` and
fitted on the
other folds of a label-stratified split, and is summarised by its midrank AUROC averaged over the fold
seeds. `groups` (usually gene) gives `"group"`, `classes` (usually consequence) gives `"class"`, and
the two together also give `"group within class"`, each (group, class) cell shrunk towards its class
rate. `SequenceBlindReport.per_seed[name]` lists one AUROC per seed and `.mean[name]` is their mean,
with `n`, `n_pos`, `n_neg`, `n_groups`, `n_classes`, `folds`, `alpha` and `seeds`. The CLI form is
`glmtrust baseline TABLE --label-col L --group-col G --class-col C [--folds 5] [--alpha 1.0] [--seeds 8] [--out JSON]`.

The building blocks are public too:

- `stratified_kfold_folds(y, n_splits=5, random_state=0)` — fold index per row, byte for byte the test
  folds of scikit-learn's `StratifiedKFold(n_splits, shuffle=True, random_state)` for an integer seed.
- `oof_rate(key, labels, folds=5, alpha=1.0, seed=0, fold_ids=None)` — out-of-fold
  (pos + α·p) / (n + α) of each row's category; `fold_ids` supplies the folds explicitly.
- `oof_group_within_class(groups, classes, labels, folds=5, alpha=1.0, seed=0, fold_ids=None)` — the
  hierarchical gene-within-consequence prior.
- `midrank_auroc(labels, scores)` — the Mann-Whitney AUROC with ties counted ½.

`benchmarks/reproduce_paper_baseline.py --panel <Additional file 4>` prints the paper's gene-identity,
consequence-class and gene-and-consequence baselines, 0.880, 0.837 and 0.974.

## `glmtrust.card`

- `render_card(report, title=..., subtitle=...)` → one self-contained HTML string: no external
  assets, no scripts, light/dark aware, printable.
- `write_card(report, path, ...)` → writes it and returns the path.

The card leads with the verdict and specifically with whether the audit *changed* it, so a reversal
appears above the tables. When nothing is wrong it says so.

## `glmtrust.delong`

- `delong_delta_ci(labels, score_a, score_b, alpha=0.05)` → `(delta, (lo, hi))` in closed form.
  Both scorers must be complete over the same variants — the method is paired.
- `delong_auroc_variance(labels, scores)` → `(aucs, covariance)` for k scorers.
- `midrank(x)` — ranks with ties averaged.

Asymptotic and symmetric, so `audit` uses it only on large panels and falls back to the bootstrap
below `delong_above`. The two are validated against each other in the tests.

## `TrustLayer`

The whole layer as one object.

```python
TrustLayer(calibration="platt", conformal="mondrian",
           alpha=0.10, coverage=0.85, n_splits=5, seed=0)
```

| Method | Returns |
|---|---|
| `fit(scores, labels, groups=None)` | self; fits the deployable layer (leave-one-group-out calibration if `groups`) |
| `predict(scores)` | dict: `probability`, `conformal_set` (n×2 bool), `conformal_decision`, `selective_decision` |
| `evaluate(scores, labels, groups=None)` | nested dict: `discrimination`, `calibration`, `conformal`, `selective` — all cross-validated |
| `summary(scores, labels, groups=None)` | a human-readable one-block string of `evaluate` |

## `glmtrust.calibration`

- `IsotonicCalibrator()` / `PlattCalibrator()` — `.fit(scores, labels)`, `.predict_proba(scores)`.
- `make_calibrator(method)` — factory for `"isotonic"` | `"platt"`.
- `cross_conformal_calibrate(scores, labels, method="isotonic", n_splits=5, seed=0)` → out-of-fold
  calibrated probabilities in the input order.

## `glmtrust.conformal`

- `SplitConformal(alpha=0.1)` — marginal split conformal.
- `MondrianConformal(alpha=0.1)` — class-conditional split conformal.
  - `.fit(cal_probs, cal_labels)`, `.predict_set(probs, alpha=None)` → (n×2) bool,
    `.predict(probs)` → `0`/`1`/`ABSTAIN`, `.evaluate(probs, labels)` → coverage per class + abstention.
- `conformal_quantile(scores, alpha)` — finite-sample-valid threshold under exchangeability (`+inf` if the set is too small).
- `ABSTAIN` — the sentinel decision value (`-1`).

## `glmtrust.selective`

- `SelectivePredictor(coverage=0.85, decision_threshold=0.5)` — `.fit(probs)`, `.keep_mask(probs)`,
  `.predict(probs)` → `0`/`1`/`ABSTAIN`, `.evaluate(probs, labels)` → coverage, selective error,
  capture, lift and class-asymmetric capture.
- `precision_operating_point(scores, labels, target_precision, delta=0.1, n_rep=50, ...)` → dict with
  achieved precision, recall, mean call count, feasible fraction (RCPS-style guarantee).
- `precision_lower_bound(k_correct, n, confidence=0.9)` — Clopper-Pearson lower bound.
- `margin_confidence(probs)` — the `|2p − 1|` confidence functional.

## `glmtrust.transfer`

- `transfer_calibration(source_scores, source_labels, target_scores, method="isotonic")` → target
  probabilities.
- `leave_one_group_out(scores, labels, groups, method="isotonic")` → `(probs, report)`; `report`
  includes a `"__macro__"` entry. Each group's `auroc` is signed, so a group whose score runs the
  wrong way reads below 0.5 and carries `below_chance: True`.

## `glmtrust.metrics`

`ece(y, p, n_bins=10, strategy="uniform"|"quantile")`, `brier`, `auroc(oriented=False)`, `auprc`,
`auroc_ci`, `risk_coverage_curve`, `capture_at_coverage`, `selective_lift`, `selective_error`.
