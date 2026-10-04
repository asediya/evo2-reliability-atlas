# glmtrust documentation

Version 0.1.1.

`glmtrust` does two things, useful independently:

1. **Audits a comparison** between variant-effect scorers, asking whether it measures skill or
   bookkeeping. Different scorers reach different variants, so two accuracies quoted side by side are
   routinely computed on different sets.
2. **Wraps a scorer** so it can abstain: a calibrated probability, a conformal prediction set with a
   coverage guarantee (valid under exchangeability), and a selective call.

- **[Tutorial](tutorial.md)** — a hands-on walk through each component and the one-object pipeline.
- **[API reference](api.md)** — the public classes and functions.
- **[README](../README.md)** — install, quickstart, and the honest scope of what the tool does and
  does not claim.

## When to reach for the audit

Whenever a comparison involves a scorer that cannot score everything — which is most of them.
Alignment-based scores are undefined where the alignment fails; protein-level scores are undefined
outside coding sequence; anything needing an MSA inherits its gaps.

Concretely, reach for it when you are:

- **benchmarking two or more scorers** and want to know whether the winner would change on the
  variants both can actually score;
- **choosing a scorer to deploy** and need its accuracy over everything that will arrive, not over
  the subset it happens to cover;
- **checking someone else's published comparison**, including your own before submission;
- **assembling a panel from several annotation sources**, where a merge on mismatched keys produces
  a plausible-looking accuracy computed over a handful of rows.

It is inert, correctly, when every scorer reaches every variant: the penalty is zero, matched equals
naive, and it says so.

## When to reach for the trust layer

Use it when you have variant-effect scores from a genomic language model and need to know
*how far to trust them* — in particular when:

- you are working in a **non-model species** with few or no curated labels, and need a defensible
  operating point rather than a raw ranking;
- the classes are **imbalanced** (deleterious variants are rare), so a single marginal threshold can
  silently under-cover the class you care about — the `MondrianConformal` class-conditional guarantee
  is the reason the package exists;
- you want to **abstain** on the least-confident calls and need to know how many errors that removes,
  reported honestly and separately for over-calls and missed positives.

## Design principles

1. **Standard methods, assembled honestly.** In the trust layer no component is novel; the value is a
   tested, documented layer with out-of-fold thresholds and cross-validated reporting built in. In the
   audit, restricting a comparison to shared variants is likewise established practice — what is added
   is the must-answer accounting and the treatment of reach as a reportable property.
2. **Silence has to mean something.** Warnings are gated on effect size as well as significance,
   because on a million-variant panel even a reach gap far too small to matter excludes zero. A tool
   that flags everything is a tool whose flags are ignored, so the audit states plainly when it finds
   nothing wrong.
3. **No hidden optimism.** Conformal and selective thresholds are fitted on out-of-fold probabilities;
   `TrustLayer.evaluate` scores every variant with a model fitted without it.
4. **State what does not transfer.** A transferred probability is not better than a global sigmoid; the
   selective *ordering* is what survives a cross-species shift. The docs and docstrings say so.
