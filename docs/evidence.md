# Evidence and current status

SharpeArena's paper treats its experiments as calibrations and falsification
attempts against the environment, not as claims about live markets or model
performance. The exact commands, fixed seeds, JSON evidence, figures, and
provenance manifest live under [`paper/`](../paper/).

## What is established

- The corrected deflation units produce a non-vacuous board: Calm drift can
  saturate deflated Sharpe, but no reference baseline clears pass^k on every
  seed, and Hard/Extreme sharply reduce the score.
- The canonical synthetic tape fails the calibrated three-check realism
  conjunction on 23 of 24 seeded panels. The failure is reported rather than
  hidden or used to move the rank key.
- Market-making regret is measured against a fixed closed-form
  Avellaneda–Stoikov reference, whose zero regret is by construction rather than
  evidence of optimality for every reward model.
- Calm-trained references show substantial zero-shot loss on Hard/Extreme while
  the within-Calm generalization control remains near zero.
- Linear-impact manipulation probes are negative on the sampled 45-cell grid;
  concave/asymmetric extensions include profitable positive controls. The result
  is explicitly finite-grid, not a theorem.
- A bounded scan recovers public seeds but not sealed seeds while a high-entropy
  salt is withheld; revealing the salt reproduces the field. This is evidence
  against one enumeration budget, not a cryptographic proof.
- Out-of-band oracle controls show the eligibility region is non-empty and that
  pass^k, rather than a pooled score, binds at the observed crossings.
- Ecology results are reported as distributions across seeds rather than one
  selected trajectory.
- Native, WebAssembly, npm, and Python golden legs execute the shared engine;
  canonical pre-hash JSON and `SPEC_HASH` make drift both readable and
  fail-closed.
- Generated-strategy evidence counts raw proposals before validation or
  deduplication and binds host-derived family identity, exact generator identity,
  earlier-candidate ancestry, and operator-registered idea sources. Family
  summaries are diagnostic and cannot change the trial denominator or rank key.

## What is not established

- No local open-weight or paid frontier-model field is part of the evidence.
  The runners exist, but CI uses deterministic doubles and no performance result
  is claimed.
- No live-capital route exists in the paper arm. Its provider-dependent paper
  evidence is a separate, non-replayable class from deterministic backtests.
- No hostile entrant has been operated as a tenant. Container acceptance
  evidence belongs to SharpeBench and remains narrow.
- Synthetic realism diagnostics do not establish that the generator captures
  every market mechanism, tail, or interaction.
- Candidate lineage records do not establish causal intellectual attribution.
  They establish that a generated candidate declared references to specific
  earlier proposals and exact source digests that were available in its bound
  search plan.

For the canonical evaluation inputs and eligibility contract, see
[`EVALUATION.md`](../EVALUATION.md). For the exact scientific interpretation,
use the paper rather than this product summary.
