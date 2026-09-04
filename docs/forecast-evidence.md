# Prospective forecast evidence

SharpeArena can commit forecasts before their outcomes exist and export the raw
ledger for independent analysis. The producer never scores its own export.

## Contract

`ForecastContract` freezes the question, instrument, target, observation source,
open and close definitions, submission window, resolution boundary, unit,
missing-data policy, fallback policy, and scoring rule before a prediction is
accepted. Its canonical JSON digest binds every revision to those exact bytes.

The supported forecast forms are:

| Kind | Prediction | Proper score |
|---|---|---|
| `point` | one finite value | absolute and squared error |
| `probability` | one probability | binary Brier or log loss |
| `categorical` | a probability for every frozen category | multiclass Brier or log loss |
| `normal` | mean and positive standard deviation | Normal CRPS |
| `direction` | `-1` or `1` | directional accuracy outside the frozen neutral band |
| `interval` | lower and upper endpoints | interval score at the frozen alpha |

The compatibility helper `DeferredDesk.commit` still accepts a relative horizon.
Use `commit_contract` or `ForecastLedger.submit` for an administered run. Those
paths keep the deadline and resolution boundary fixed across revisions.

## Lifecycle

`ForecastLedger` is append-only. A revision names the prior revision, carries an
idempotency key, and records the exact information exposure at submission time.
Consensus-visible submissions require a digest of the consensus snapshot. Late
and pre-open attempts remain in the ledger but never replace the latest eligible
forecast.

Resolution is one of `pending`, `resolved`, `cancelled`, or `rejected`. A resolved
outcome must become available after the effective revision and no earlier than the
contract's frozen resolution boundary. The exported document covers every claim
exactly once.

## Export

```python
from pathlib import Path

from sharpearena import ForecastLedger, write_forecast_evidence

evidence = ledger.evidence(outcomes, generated_at=logical_clock)
digest = write_forecast_evidence(Path("forecast-evidence.json"), evidence)
```

`write_forecast_evidence` validates the closed
`sharpe.forecast-evidence.v1` envelope, writes through an exclusive temporary
file, fsyncs it, atomically replaces the destination, and returns the SHA-256 of
the stored bytes. On POSIX it also fsyncs the parent directory.

SharpeBench consumes this file without importing SharpeArena. It validates the
contract and revision graph again and recomputes every diagnostic from the raw
predictions and outcomes.

## Executable cross-product example

[`examples/forecast-quality`](../examples/forecast-quality/) generates two
deterministic ledgers and can pass them directly to a sibling SharpeBench
checkout. The committed output covers exact common support, resolution-time
blocks, eligible and late revisions, a pre-open rejection, and blind versus
consensus-visible exposure. It is a compatibility fixture, not an empirical
agent result.

[`examples/prospective-forecast-field`](../examples/prospective-forecast-field/)
is the separate live protocol. It freezes model bytes, public market
observations, target candles, the stopping rule, and analysis before inference;
seals pending ledgers before the outcome boundary; then refuses resolution until
the data source's server clock is past every frozen candle. Model snapshots are
loaded locally and the runner never trades.

## Superseded engineering pilot

[`paper/evidence/prospective-forecast-field`](../paper/evidence/prospective-forecast-field/)
contains one completed lifecycle exercise. Three older model snapshots already
cached on the workstation forecast 24 binary Binance Spot contracts from one
frozen 12-bar observation. Forecasts were sealed at 15:32:28.578 UTC, before the
16:11 UTC deadline and before target candles opening from 16:12 through 16:22
UTC. The resolver wrote outcomes only after the source clock passed 16:23 UTC.

The field contains 24 contracts but only six settlement-clock blocks because
four correlated assets resolve at each clock. Its preregistered minimum for a
comparative claim is 30 blocks. The older checkpoints were a convenience sample
and are no longer accepted as the product's model experiment. Their raw records
and recomputed scores remain committed so the audit trail cannot be rewritten,
but the field is excluded from model comparison, benchmark rank, and the
paper's empirical conclusions. The
[pre-outcome review](prospective-forecast-methodology-review.md) and
[post-outcome review](prospective-forecast-result-review.md) state the complete
chronology, checks, and limits.

## Non-claims

- Logical clocks make ordering testable; they do not prove wall time.
- A contract digest detects internal drift; it is not a signature or a public
  timestamp.
- Exposure hashes record the inputs the producer names. They do not prove that a
  model used no other information.
- Forecast quality is separate from SharpeBench trading-rank eligibility.
