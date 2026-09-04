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
- A prospective protocol pilot sealed three exact local model snapshots before
  24 Binance Spot one-minute candle outcomes existed. All three produced a
  complete forecast ledger on the same support. SharpeBench independently
  imported and rescored the field: Brier losses were 0.2548 for `qwen-7b`,
  0.2599 for `phi-4`, and 0.3148 for `qwen-0.5b`. The result has only six
  settlement-clock blocks, every pairwise interval crosses zero, and the frozen
  30-block minimum prohibits a model-comparison claim.

## What is not established

- The admitted local-model field is a fixed two-token forecast scaffold, not an
  autonomous trading agent. It uses one 12-bar observation, one venue, four
  correlated crypto pairs, six future clocks, one hardware stack, and models
  selected from snapshots already available on the workstation. It establishes
  neither general calibration nor trading performance.
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
