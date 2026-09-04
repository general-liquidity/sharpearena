# Prospective forecast field: pre-outcome methodological review

Status: pre-resolution review of the field sealed in commit
[`70f80d6`](https://github.com/general-liquidity/sharpearena/commit/70f80d65e94a69ddc451ef7b1ba57af308070876).

This review was written while every forecast resolution remained `pending`.
GitHub recorded the workflow for the sealed commit at 2026-09-04 15:32:49 UTC.
The frozen submission deadline is 16:11:00 UTC and the last resolving candle
ends at 16:23:00 UTC. The GitHub event is a public receipt, not a cryptographic
proof of wall time. The exchange server clock, frozen plan, commit history, and
receipt together make a later forecast substitution detectable.

## Review decision

The design is admissible as a small descriptive, prospective demonstration if
resolution completes without changing the frozen files. It cannot support a
claim that one model is better than another, that any entrant is a capable
trading agent, or that the observed forecast quality generalizes beyond this
single field.

## What the design establishes

- Three exact local model snapshots produced forecasts before the same public
  deadline from one content-addressed observation.
- Each model covers the same 24 binary contracts: four instruments at six
  one-minute resolution clocks.
- The field was sealed before any resolving candle opened. The seal binds the
  plan, observation, pending ledgers, inference records, prompts, model
  snapshots, and scaffold sources.
- Inference was local-only. The recorded method normalizes the next-token logits
  for the labels `0` and `1`; it neither samples a completion nor repairs model
  output.
- The stopping rule resolves every frozen contract. No score can stop or extend
  the field.
- SharpeBench recomputes every Brier loss from the raw forecast and outcome, uses
  exact common support, resamples whole resolution-clock blocks, adjusts the
  three pairwise tests with Holm's method, and reports the result outside the
  trading rank.

These are protocol and execution claims. The outcomes will determine the
descriptive numbers, not whether the protocol was followed.

## Limits fixed before outcomes

### The field artifacts were not all in paper provenance

SharpeArena's old artifact glob covered only JSON files directly under
`paper/evidence`. The nested plan, forecasts, and inference records were in Git
but absent from the provenance manifest. Commit `73e60ed` expands the closed
scope to the JSON records and plan checksums for the current field and both
aborted pilots. The manifest now binds 47 artifacts instead of 29.

### Pairwise Monte Carlo output depended on input-file order

SharpeBench originally derived a bootstrap seed from the ordered pair of agent
IDs. Reversing the same two input files changed the resamples, interval, and
Monte Carlo p-value. Commit
[`8166b69`](https://github.com/general-liquidity/sharpebench/commit/8166b69)
derives the seed from the sorted pair. A mutation test restores the asymmetric
seed and fails with different p-values. With the fix, reversing the inputs
preserves the p-value and adjusted decision while negating the point estimate
and swapping the interval endpoints.

## Limits that interpretation must retain

### Six blocks are not 24 independent trials

The four assets resolving in the same minute share one resampling block. The
effective comparative sample is six clocks, not 24 contracts. Crypto assets are
also correlated across clocks, and six one-minute blocks cannot diagnose
longer-range dependence. The preregistered minimum for any comparative claim is
30 blocks. Pairwise intervals and p-values are therefore exploratory output only.

### The number is a two-label logit normalization

For each prompt, the runner reads the logits of the single-token labels `0` and
`1`, normalizes only those two values, and clips the result to `[0.01, 0.99]`.
This is a precise operational probability under the frozen scaffold. It is not
an unconstrained elicitation of the model's subjective probability: probability
mass on every other token is discarded. A reliability table can describe these
numbers against outcomes, but it cannot establish general model calibration.

### The entrants are controlled model-scaffold configurations

The scaffold supplies a frozen 12-bar snapshot and one contract at a time. The
models have no tools, research loop, memory, portfolio, order interface, or
ability to select the assets and horizon. This isolates a forecast head. It does
not evaluate an autonomous trading agent or the full SharpeArena episode loop.

### The field is narrow by construction

It covers one Binance Spot source, four crypto pairs, one-minute direction, one
UTC window, one hardware configuration, and three locally available models.
The models were chosen for local availability rather than sampled from a target
population. Their four-bit inference can vary across kernels and hardware; the
sealed outputs are evidence, while rerunning inference is not claimed to be
byte-identical.

### Two pilots changed the method before outcomes

Attempt 1 was stopped before its deadline because free-form generation could
not complete the field in time. Attempt 2 completed inference but failed the
predeclared exact-JSON parser. Neither attempt queried or recorded an outcome.
Both are retained as `aborted_before_outcomes`. The final binary-logit method
was frozen and published before its resolving window. This protects the final
scores from outcome-driven adaptation, but the field remains a protocol pilot,
not a preregistered confirmatory experiment designed before any operational
feedback.

### Forecast quality cannot change trading eligibility

Brier loss, calibration summaries, and pairwise comparisons are reported only.
They do not satisfy, weaken, or replace the Deflated Sharpe, repeated-run,
significance, process, or mandate gates. No PnL is generated here, so no trading
performance claim follows from this field.

## Required post-resolution checks

Before any result enters either paper:

1. Resolve only after the exchange server clock is strictly later than the last
   frozen candle boundary.
2. Verify the sealed file set and every SHA-256 before fetching outcomes.
3. Require each exact candle and its exact open and close timestamps; use no
   substitute venue, interval, or fallback.
4. Preserve close equal to open as `false`, as frozen in every contract.
5. Require all three resolved ledgers to have identical 24-contract support.
6. Recompute Brier loss and the observed pairwise differences in an independent
   implementation.
7. Treat all pairwise inference as exploratory because there are only six
   settlement blocks.
8. Report both aborted pilots, the fixed scaffold, quantization, hardware,
   model revisions, snapshot digests, and the distinction between recorded and
   reproducible inference.
9. State explicitly that the field measures a fixed local forecast scaffold,
   not a tool-using or capital-bearing trading agent.

