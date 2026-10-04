# eQTL harness/context ablation - does the regulatory blind spot survive a better harness?

> **Development log.** Both branches of the decision template below survive under `## Read`, two of six harness rows are `TOO_FEW` or `PENDING`, and a baseline is quoted as 0.496 where its own row reads 0.521. Nothing here is a manuscript result; the published eQTL values are 0.488 at the 1,001-bp readout and 0.498 at 8,192 bp.

Question: could the 1001-bp window + single-position readout MANUFACTURE the regulatory blind
spot? Causal (n=1000) vs non-causal (n=1000) pig cis-eQTLs, Evo2-40B,
re-scored under each harness. Baseline = original 1002-bp single-position (AUROC 0.496). AUROC on |score|
magnitude (matches baseline |LLR|); signed oriented >=0.5. 95% CI = 1000x bootstrap on |score|.

| harness | window(bp) | readout | n | AUROC(|score|) | 95% CI | AUROC(signed) | status |
|---|---|---|---|---|---|---|---|
| baseline | 1002 | single-pos | 2000 | 0.521 | 0.495-0.546 | 0.515 | ok |
| sp2048 | 2048 | single-pos | 2000 | 0.520 | 0.495-0.545 | 0.513 | ok |
| sp4096 | 4096 | single-pos | 2000 | 0.523 | 0.498-0.548 | 0.516 | ok |
| ll1002 | 1002 | mean-LL | 2000 | 0.507 | 0.482-0.532 | 0.504 | ok |
| ll2048 | 2048 | mean-LL | 192 | - | - | - | TOO_FEW |
| ll4096 | 4096 | mean-LL | - | - | - | - | PENDING |

## Read
- AUROC stays ~0.5 (CI includes 0.5) across windows & readouts => regulatory blind spot is a MODEL
  property, not a harness artifact.
- AUROC rises materially (CI excludes 0.5) under mean-LL / larger windows => partly a scoring artifact
  => re-scope the regulatory claim.

Caveat (separate): causal cis-eQTLs are common/weakly-selected, so a fitness-trained
model being near chance here may be EXPECTED, not a representational failure. This ablation isolates the
harness confound only; the construct-validity question is addressed in the manuscript text.
