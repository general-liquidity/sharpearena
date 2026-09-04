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

## Non-claims

- Logical clocks make ordering testable; they do not prove wall time.
- A contract digest detects internal drift; it is not a signature or a public
  timestamp.
- Exposure hashes record the inputs the producer names. They do not prove that a
  model used no other information.
- Forecast quality is separate from SharpeBench trading-rank eligibility.
