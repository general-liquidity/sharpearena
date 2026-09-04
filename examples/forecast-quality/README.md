# Forecast quality tutorial

This example crosses the product boundary without a package dependency.
SharpeArena writes two strict `sharpe.forecast-evidence.v1` documents, then
SharpeBench validates and independently scores their raw predictions and
outcomes.

The fixture is synthetic and deterministic. It demonstrates the protocol; it
is not an empirical result for either agent.

From an installed SharpeArena checkout:

```bash
python examples/forecast-quality/tutorial.py \
  --output-dir /tmp/sharpe-forecast-tutorial \
  --sharpebench-dir ../sharpebench
```

The output contains:

- `agent-alpha.json` and `agent-beta.json`, each a complete append-only ledger
- `manifest.json`, which binds the two evidence files by SHA-256
- `report.json`, recomputed by SharpeBench with frozen bootstrap settings

The eight questions form exact common support across two resolution-time
blocks. The ledgers also retain an eligible revision, a late revision that
cannot replace it, a pre-open rejection, and both blind and consensus-visible
information exposures.

To generate only the producer artifacts, omit `--sharpebench-dir`. The files in
`fixtures/` are committed compatibility fixtures and are regenerated in the
SharpeArena test suite.
