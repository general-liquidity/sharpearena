# Prospective forecast field: post-outcome methodological review

Status: completed review of the field sealed in commit
[`70f80d6`](https://github.com/general-liquidity/sharpearena/commit/70f80d65e94a69ddc451ef7b1ba57af308070876)
and resolved in commit
[`c9c7b2b`](https://github.com/general-liquidity/sharpearena/commit/c9c7b2b).

The pre-outcome review remains unchanged in
[`prospective-forecast-methodology-review.md`](prospective-forecast-methodology-review.md).
This document applies its declared checks to the closed result. It is an
adversarial review by a separate implementation path, not an external peer
review.

## Status after model-panel review

This field is retained as a superseded engineering pilot. Its three
convenience-sample checkpoints are too old to represent the model population
the product intends to evaluate. The raw records remain immutable for audit
transparency, but their scores are excluded from the academic paper's empirical
conclusions, any model leaderboard, and every trading-rank decision.

## Original protocol decision

The completed field was admissible as a descriptive prospective protocol
demonstration. It shows that three fixed local model-scaffold configurations
produced complete, sealed forecasts before the outcomes existed and that a
separate product recomputed proper scores from the committed records.

It does not support model superiority, general calibration, trading skill,
profitability, or any change to SharpeBench rank. There are six settlement
blocks, below the 30-block minimum frozen before outcomes. Every pairwise
interval crosses zero. All three Brier skill scores are negative against the
observed base-rate forecast.

## Protocol audit

| Check frozen before outcomes | Result |
|---|---|
| Forecast seal precedes the deadline | Pass. The last model record was submitted at 15:32:06 UTC, the source clock sealed the field at 15:32:28.578 UTC, GitHub received that commit by 15:32:49 UTC, and the deadline was 16:11:00 UTC. |
| Resolution occurs after every target candle | Pass. The last candle boundary was 16:23:00 UTC; the source clock recorded resolution at 16:23:21.444 UTC. |
| Sealed bytes remain unchanged | Pass. Resolution revalidated all nine SHA-256 records before fetching outcomes. |
| Exact source and candle boundaries | Pass. All 24 records came from the frozen Binance Spot endpoint and exact one-minute open times, with no fallback. |
| Complete common support | Pass. Each model has the same 24 eligible forecasts and 24 resolved outcomes. |
| Stopping rule | Pass. Every frozen contract was resolved; no score changed the field length. |
| Producer and scorer remain separate | Pass. SharpeArena exported raw ledgers. SharpeBench imported the committed Arena tree and generated the report. |
| Independent arithmetic | Pass. A standalone Python checker reconstructed support, Brier and skill scores, fixed-bin calibration, block bootstrap, and Holm adjustment from the ledgers. |
| Trading-rank isolation | Pass. The report records `reported_only_never_trading_rank`, and the Lean model proves that attaching a forecast report preserves the trading-rank projection. |

GitHub's receipt and the exchange clock are corroborating records, not a
cryptographic proof of wall time. The result is tamper-evident because changing
a sealed forecast now breaks its public commit and digest chain.

## Descriptive result

Eight of the 24 candles closed above their open, for an observed base rate of
one third. Lower Brier loss is better.

| Model-scaffold ID | Mean forecast | Brier loss | Brier skill vs. observed base rate |
|---|---:|---:|---:|
| `qwen-7b` | 0.4919 | 0.2548 | -0.1467 |
| `phi-4` | 0.5201 | 0.2599 | -0.1695 |
| `qwen-0.5b` | 0.6373 | 0.3148 | -0.4168 |

All predictions for each entrant fall into one of the five preregistered
calibration bins. The resulting Murphy resolution term is zero for every
entrant. That is a consequence of concentrated forecasts and a very small
field, not evidence that the models have no resolution in a larger population.
Likewise, the negative skill values describe this field only. The reference
base rate is estimated from the same 24 outcomes rather than fixed from an
external population.

The exploratory pairwise output is:

| Loss A minus loss B | Difference | 95% block-bootstrap interval | Raw p | Holm p |
|---|---:|---:|---:|---:|
| `phi-4` minus `qwen-0.5b` | -0.0550 | [-0.1241, 0.0158] | 0.1479 | 0.4438 |
| `phi-4` minus `qwen-7b` | 0.0051 | [-0.0157, 0.0227] | 0.6537 | 0.6537 |
| `qwen-0.5b` minus `qwen-7b` | 0.0600 | [-0.0384, 0.1465] | 0.2104 | 0.4438 |

There is no familywise-significant comparison. More importantly, the frozen
minimum prohibits a comparative claim even if a p-value had crossed 0.05.

## Adversarial interpretation audit

### This is a fixed forecast scaffold, not a trading agent

The models receive one frozen market snapshot and one contract at a time. They
have no tools, memory, research loop, portfolio, order state, or capital. The
field measures an operational forecast head inside a controlled scaffold. It
does not exercise SharpeArena's full episode loop or SharpeBench's trading
eligibility predicate.

### These are not rolling next-minute forecasts

The observation was captured at 15:25:44 UTC from bars ending at 15:24:59 UTC.
The six target candles open from 16:12 through 16:22 UTC. The task asks for the
direction inside each one-minute target candle at lead times of roughly 47 to
57 minutes from the last observed close. No observation refresh occurs between
targets. Describing the field only as "one-minute prediction" would conceal the
lead time and the shared information set.

### Twenty-four contracts provide only six resampling units

Four crypto pairs share each resolution clock. They are correlated within a
clock and plausibly across adjacent clocks. The block bootstrap preserves the
first dependence but six blocks cannot diagnose longer dependence or support a
stable asymptotic approximation. The pairwise values are diagnostics, not
inference about a model population.

### The probabilities are conditional on a two-token elicitation

The runner normalizes next-token logits only over the single-token labels `0`
and `1`, then clips to `[0.01, 0.99]`. This makes the operation deterministic
given the recorded logits. It discards probability mass on every other token
and is not an unconstrained statement of each model's subjective probability.
The paper must call these operational probabilities or normalized binary
logits.

### Model selection and execution are convenience samples

The three snapshots were selected because they were already available on one
workstation and could complete before the frozen deadline. Two are code-tuned
Qwen checkpoints; none was selected from a sampling frame of trading agents.
The run uses one RTX A4000, one Transformers and Torch stack, and four-bit local
inference. Snapshot and runtime identities are recorded, but inference is not
claimed byte-reproducible across hardware or kernels.

### A prospective target does not remove every contamination channel

The outcomes did not exist when forecasts were sealed, so model pretraining
cannot contain these exact outcomes. The scaffold and public exchange format
can still resemble training material, and the model can carry general market
priors. This field closes direct target contamination, not all transfer from
pretraining.

### The final protocol followed two outcome-free pilots

Attempt 1 was too slow under free-form generation. Attempt 2 completed
inference but failed exact JSON parsing. Neither observed an outcome, and both
aborted directories remain committed. The binary-logit protocol was therefore
adapted to operational feedback but not to scores. This makes the result a
transparent protocol pilot, not a confirmatory experiment designed before any
implementation feedback.

## Findings addressed before publication

1. Arena provenance originally omitted nested field files. The artifact scope
   now binds the current field and both aborted attempts.
2. Bench pairwise Monte Carlo output originally depended on input-file order.
   Pair seeds are now derived from sorted agent IDs, with a reversal regression
   test.
3. The forecast command originally required shell redirection for a report.
   `--output` now writes the complete machine report directly.
4. Bench now imports only a closed field whose bytes match the committed Arena
   HEAD. The import receipt records repository, commit, source path, plan hash,
   and every copied file digest.
5. Bench provenance now includes root and nested prospective-field JSON and
   checksum files.
6. The independent checker recomputes the Rust report and fails on a planted
   score change. Its committed receipt records that six blocks do not meet the
   30-block comparative threshold.

## Remaining work that cannot be repaired after seeing outcomes

- Repeat the same frozen protocol over many non-overlapping UTC windows and
  predeclare the aggregation before the first new outcome.
- Use more than one venue and asset class, with dependence units chosen before
  data collection.
- Compare elicitation methods, including a constrained probability grammar,
  as separate preregistered treatments rather than changing this field.
- Evaluate complete agents with fixed tool and information budgets separately
  from this forecast-head scaffold.
- Select models from a declared target population rather than local cache
  availability.

Those items are future experiments. They are not qualifications that can be
made true by adding more prose to this result.
