# glmtrust

[![tests](https://img.shields.io/badge/tests-246%20passing-brightgreen.svg)](tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**Audit a variant-scorer comparison before believing it — then wrap the scorer so it can abstain.**

Part of the Evo 2 reliability atlas (`github.com/asediya/evo2-reliability-atlas`), and usable on
its own with any scorer.

Two tools, useful independently:

| | |
|---|---|
| **[`glmtrust audit`](#auditing-a-comparison)** | Is this comparison measuring skill, or bookkeeping? Different scorers reach different variants, so two accuracies quoted side by side are routinely computed on different sets, and the difference between them is read as accuracy. |
| **[`glmtrust.TrustLayer`](#the-trust-layer)** | Turn a score into a calibrated probability, a conformal prediction set, and a selective call that abstains when the model is not confident. |

---

## Auditing a comparison

On a 1,434,335-variant ClinVar panel, quoted the way benchmarks usually quote them, AlphaMissense
leads REVEL by +0.0056 AUROC. On the 192,168 variants **both can score**, REVEL leads by 0.0096.
The verdict reverses, and it reverses at every ClinVar review-status threshold.

Both figures are recomputed from the raw score files by
`analyses/scripts/verify_readme_reversal.py`, which prints them beside the claim when the scorer tables are present. From a clone it prints only `cannot check: ... absent (it lives in the undeposited data tree)`, because those tables are not deposited.

How large the effect is depends on the panel, and it is not always this large. On the dbNSFP
panel the audit reports -- **328,328 variants at one ClinVar review star or better**,
not the 352,907 of the unfiltered table, which is a different panel -- coverage differs sharply
by class: REVEL reaches **35% of positives against 85% of negatives**, while the head-to-head
inflation between SIFT, PolyPhen-2 and REVEL stays under 0.007 and reverses nothing. The coverage warning is the reliable part; a
reversed ranking is what you check for, not what you should expect.

```python
from glmtrust import Scorer, audit, write_card

report = audit(labels,
               [Scorer("alphamissense", am, readout="per-substitution point score"),
                Scorer("revel",         rv, readout="per-substitution point score")],
               strata=consequence)          # optional: hold composition fixed
print(report)
write_card(report, "audit.html")            # one self-contained page for a co-author or referee
```

or without writing any Python:

```bash
glmtrust audit variants.parquet --auto --label-col label \
    --strata-col consequence --readout "per-substitution point score" --card audit.html
```

`--auto` finds the score columns itself and prints what it chose, so a dbNSFP- or VEP-annotated
table needs no column bookkeeping.

### What it reports

| | |
|---|---|
| **reach** | what fraction of each class a scorer can be run on at all, with an interval |
| **must-answer AUROC** | what the accuracy becomes over the whole panel it is reported for, with unscorable variants contributing no information. Closed-form in the reach — no imputation |
| **reach alone** | the AUROC of the *missingness pattern* with the scores discarded entirely. On the ClinVar panel this reaches 0.6145 for AlphaMissense. It is descriptive: set beside the must-answer AUROC it can "win" even against perfect values, because the one-half rule discards what the pattern knows |
| **values add** | what ranking by that pattern and then by value gains over the pattern alone, ρ(A_cov − ½). Its interval is the test the warnings use, and unlike the comparison above it fails when the values are at chance |
| **matched delta** | the head-to-head restricted to variants both scorers reach, with the inflation against the usual quotation |
| **composition** | whether class-dependent reach survives holding variant consequence fixed, or was composition all along |
| **join integrity** | whether two scorers overlap far less than their reaches imply, which is what a merge on mismatched keys looks like |
| **breakdown frontier** | with covered AUROCs, `glmtrust reach` gives each pair's breakdown point λ*, the largest share of its unscored pairs that may resolve arbitrarily with its order intact, and counts the pairs still ordered at each share (`--lambdas`); λ = 1 is the sharp bound |
| **sequence-blind baseline** | `glmtrust baseline` (`sequence_blind`) scores every variant by the out-of-fold rate of its gene, its consequence class or both, reading no sequence and no score: the floor an accuracy on that panel is read against |

### Design notes worth knowing before you rely on it

- **A declared `readout` is mandatory.** How a model's output becomes one number per variant can
  invert a verdict, so scorers that declare different readouts are refused a delta rather than
  quietly differenced.
- **Warnings are gated on effect size, not just significance.** On a million-variant panel a reach
  gap far too small to matter still excludes zero: CADD's +0.0006 on the 1.4-million-variant ClinVar
  panel. Every number is printed regardless; only the warnings are gated.
- **It says so when nothing is wrong.** Two conservation tracks with matching 99.9% reach report an
  inflation of exactly 0.0000 — a tool that only ever reports problems carries no information when
  it stays quiet.
- **It is fast enough for the panels that matter.** Head-to-head intervals use DeLong's closed form
  above 20,000 matched variants and the paired bootstrap below it, validated against each other:
  the full ClinVar audit takes 42 seconds rather than half an hour.
- **Missing means NaN.** A scorer that imputes its own no-calls before reaching this module cannot
  be audited for reach, which is the point.
- **Intervals are variant-level unless you name a cluster.** Where variants share a gene or locus
  that is too narrow: on the dbNSFP panel gene-resampled intervals run several times wider.
  `cluster=` (CLI `--cluster-col`) makes every whole-panel interval a bootstrap over whole groups
  (the within-stratum intervals stay variant-level), and the report states which kind it printed.
- **An AUROC below one half is flagged, not reported as weakness.** It usually means the score is
  stored the other way round; declare that with `higher_is_worse=False` (CLI `--lower-is-worse`).

### What the audit does *not* claim

Restricting a comparison to shared variants is **not new** — it is the pairwise "rank score" of the
deep-mutational-scanning benchmarking literature. That composition heterogeneity inflates apparent
accuracy is likewise established. What is added here is the must-answer accounting, the
test of what a scorer's values add beyond its reach, and the treatment of reach as a reportable property of a *comparison*
rather than a nuisance to impute away.

---

## The trust layer

A genomic language model (Evo 2, Nucleotide Transformer, GPN, Caduceus) returns a variant-effect
score whose *ordering* is informative but whose *scale* and *reliability* are not self-evident,
especially in a non-model species that has no curated labels of its own. `glmtrust` turns such a
score into three things a user can actually act on:

1. a **calibrated probability** that a variant is deleterious;
2. a **conformal prediction set** with a coverage guarantee, including a class-conditional (Mondrian)
   form that targets coverage *within* the rare positive class rather than only on average (valid under exchangeability);
3. a **selective call** that abstains when the model is not confident, with an operating point set by a
   target coverage or a precision guarantee that is distribution-free under exchangeability.

> **Scope of the guarantees: read before deploying.** The conformal coverage and the precision
> guarantee are finite-sample valid *under exchangeability* between the calibration data and the
> variants being scored. The study accompanying this package tested exactly the setting it targets,
> transporting a threshold to a species with no labels of its own, and found that exchangeability
> does **not** hold across species: the achieved positive-class false-negative rate was **0.107–0.321
> against a nominal 0.05** (up to sixfold), holding only in-distribution or at a lenient error target.
> Do **not** rely on these as a distribution-free guarantee in a label-poor target species; treat the
> layer as a calibrated *ordering* with an operating point, not a safety net against missed positive
> variants.

> **On novelty, stated plainly.** Every method here is standard (isotonic/Platt calibration, split and
> Mondrian conformal prediction, confidence-ordered selective prediction, RCPS precision control). The
> contribution is that they are assembled, tested and documented as **one deployable layer for this
> task**, with the honesty rails the accompanying study argues for: out-of-fold thresholds,
> cross-validated reporting, class-conditional coverage, and separate accounting for the
> missed-positive error the selective layer does *not* fix. `glmtrust` is a packaging and application
> contribution, not a new machine-learning method, which is exactly why it is safe to depend on.

## Install

```bash
pip install glmtrust            # from PyPI (when published)
# or, from a checkout:
pip install -e ".[io]"          # [io] adds polars for parquet input in the CLI
```

Requires Python ≥ 3.9 and `numpy`, `scipy`, `scikit-learn`.

## Quickstart

```python
import numpy as np
from glmtrust import TrustLayer

# scores from your genomic LM (higher = more deleterious) and known labels
score = ...   # shape (n,)
label = ...   # shape (n,) in {0, 1}

layer = TrustLayer(calibration="isotonic", conformal="mondrian",
                   alpha=0.10, coverage=0.85).fit(score, label)

# an honest, cross-validated report
print(layer.summary(score, label))

# deploy on new variants
out = layer.predict(new_scores)
out["probability"]          # calibrated P(deleterious)
out["conformal_set"]        # (n, 2) bool: [negative in set, positive in set]
out["conformal_decision"]   # 0 / 1 / ABSTAIN
out["selective_decision"]   # 0 / 1 / ABSTAIN at the target coverage
```

### Cross-species (label-free target)

If your labels come from several species and you want the honest "no labels in the target" estimate,
pass `groups`; calibration is then fitted leave-one-group-out:

```python
layer.evaluate(score, label, groups=species)   # each species scored by a map fitted on the others
```

## Command line

`glmtrust` reads one table (`.parquet`, `.csv`, or `.tsv`) with a score column, a label column, and
optionally a group column:

```bash
glmtrust evaluate variants.parquet --score-col evo2 --label-col label --group-col species
glmtrust calibrate variants.csv    --score-col evo2 --label-col label --out probs.csv
glmtrust transfer  variants.parquet --score-col evo2 --label-col label --group-col species --out probs.csv
```

`audit` takes one `--score-col`/`--readout` pair per scorer, or `--auto` to detect them:

```bash
glmtrust audit variants.parquet \
    --score-col revel --readout "precomputed per-substitution score" \
    --score-col alphamissense --readout "precomputed per-substitution score" \
    --label-col label --strata-col consequence --cluster-col gene --card audit.html
```

Give two scorers the same readout only when their numbers are produced the same way; with different
readouts the audit reports each one's reach and accuracy but no head-to-head delta.

**Bringing your own table.** Every command reads `.csv`, `.tsv` or `.txt` (optionally gzipped,
comma-, tab- or semicolon-separated, UTF-8, UTF-16 or Windows-1252) and `.parquet`; save a
spreadsheet as CSV first. Labels are 0/1 or true/false; for text labels name the values, e.g.
`--positive-label Pathogenic --positive-label Likely_pathogenic --negative-label Benign
--negative-label Likely_benign`. A variant with no label stops the run unless `--drop-unlabelled`
says to leave it out. An empty or NA score cell means "not scored"; an infinite value is refused.
`--lower-is-worse` declares a score where lower means more damaging, for every command, and the
reports say when a score appears to run the other way. `calibrate` and `transfer` write one row per
input row, numbered from 0, and `--id-col` carries an identifier beside each. Any input problem ends
in one line saying what is wrong.

`--lower-is-worse COL` declares a score stored with lower values meaning more damaging. When the
scores themselves cannot be shared, `reach` works from reach indicators alone, and counts identified
pairs once each scorer's covered AUROC is supplied:

```bash
glmtrust reach reach_panel.parquet --label-col label --reach-prefix reach__ \
    --covered covered_aurocs.csv --covered-name-col pred --covered-auroc-col a_cov
```

## The components

| Module | What it does | Key classes / functions |
|---|---|---|
| `glmtrust.audit` | reach, must-answer accounting, what the values add beyond reach, matched head-to-head, join integrity, and reach-only accounting | `Scorer`, `audit`, `reach_audit`, `must_answer_auroc`, `missingness_auroc`, `lexicographic_gain` |
| `glmtrust.card` | an audit report as one self-contained HTML page | `render_card`, `write_card` |
| `glmtrust.delong` | closed-form intervals for AUROC differences, so large panels stay fast | `delong_delta_ci` |
| `glmtrust.calibration` | score → probability (isotonic / Platt), fitted out-of-fold | `IsotonicCalibrator`, `PlattCalibrator`, `cross_conformal_calibrate` |
| `glmtrust.conformal` | prediction sets with coverage guarantees | `MondrianConformal`, `SplitConformal` |
| `glmtrust.selective` | abstain to a target coverage or certified precision | `SelectivePredictor`, `precision_operating_point` |
| `glmtrust.transfer` | cross-group (cross-species) calibration transfer | `leave_one_group_out`, `transfer_calibration` |
| `glmtrust.metrics` | ECE, Brier, AUROC CI, risk-coverage, capture, lift | — |
| `glmtrust.pipeline` | the whole layer as one object | `TrustLayer` |

## What it deliberately does *not* claim

- It does **not** improve discrimination: calibration and conformal prediction re-express a score, they
  do not add signal.
- A **transferred probability** across species is, per the study, no better than a single global
  two-parameter sigmoid; the component that survives transfer is the selective-prediction *ordering*.
- The selective layer does **not** bound the missed-positive (false-negative) rate; its benefit is
  class-asymmetric, so `SelectivePredictor.evaluate` reports capture for over-calls and misses
  separately. It is a precision aid, not a safety net.

## Reproducing the study

The package reproduces the paper's published numbers. From the study repository root:

```bash
python glmtrust/benchmarks/reproduce_paper_trust_layer.py
```

runs leave-one-species-out isotonic calibration and per-species 15% selective refusal through the
public API and checks the result against the deposited numbers. It matches to four decimals: pooled
capture 0.6096, pooled lift 4.07, macro ECE 0.0209, 415 errors. Those figures use `tie_policy="rank"`
in the fixture's row order; because variants tied at the refusal boundary make capture depend on row
order, the benchmark also checks that the published capture lies inside the range that 20 row-order
permutations produce. The package's default, `tie_policy="whole_block"`, refuses whole tie blocks and
never part of one, and gives a pooled capture of 0.542 on this panel.

The dbNSFP pair counts reproduce from Additional file 4, saved as `dbnsfp_reach_panel.parquet` (the
name Additional file 3 expects in its `tables/`), and the covered AUROCs in Additional file 3's
`figures/`:

```bash
python glmtrust/benchmarks/reproduce_paper_reach.py \
    --panel dbnsfp_reach_panel.parquet \
    --covered <Additional file 3>/figures/dbnsfp_49.csv \
    --missense-covered <Additional file 3>/figures/dbnsfp_49_missense.csv
```

It exits 0 when the whole panel gives 640 feasible, 506 sharp and 83 identified pairs and 131 under
monotone coverage, and the missense subset 1,176, 1,174, 429 and 488. The package was tested under
Python 3.12.10 with NumPy 2.5.1, SciPy 1.18.0 and scikit-learn 1.9.0.

## Relationship to the study & citation

`glmtrust` 0.1.1 packages the trust-layer methodology from *"Auditing variant effect predictor comparisons for missing scores: a reach resource for 49 predictors and nine amniote species"* (Asediya, GK, Kadarmideen and Pareek). The analysis
scripts that produced the paper's numbers live in the study repository; this package is the reusable,
tested extraction of those methods. A `CITATION.cff`, carrying the reserved Zenodo DOI, is included beside this README.

## License

MIT; see [LICENSE](LICENSE).
