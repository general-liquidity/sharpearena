# Agent contract

Use the observation/decision contract when an agent runs outside the Rust process or
when you need one language-neutral decision format across evaluators.

The contract is a protocol, not a standalone command runner. A host must provide the
market environment, send observations, receive decisions, enforce timeouts, and record
transport health. SharpeArena exposes Rust stdio and HTTP clients and ships reference
programs in Rust, Python, and TypeScript.

## One decision turn

The host sends one `MarketObservation`:

```json
{
  "date": "2025-01-02",
  "cash": 1.0,
  "symbols": [
    {
      "symbol": "AAPL",
      "close_history": [187.2, 188.0, 190.4],
      "fundamentals": {},
      "news": []
    }
  ],
  "portfolio": []
}
```

The agent returns one `Decision`:

```json
{
  "orders": [
    {
      "symbol": "AAPL",
      "action": "buy",
      "target_weight": 0.5,
      "confidence": 0.7,
      "rationale": "positive trailing momentum"
    }
  ],
  "reasoning": "one bounded target-weight rebalance"
}
```

`target_weight` is signed for shorts and must lie in `[-1, 1]`. `confidence` lies
in `[0, 1]`; it is optional and defaults to `0.5`.
`rationale` and top-level `reasoning` are optional audit text. An empty `orders` array
is a deliberate hold, not an error value.

The authoritative schemas are
[`observation.schema.json`](../crates/sharpearena/contract/observation.schema.json) and
[`decision.schema.json`](../crates/sharpearena/contract/decision.schema.json). Unknown
fields are rejected at closed caller-input boundaries.

## Stdio and HTTP

The stdio transport uses newline-delimited JSON: one observation line in, one decision
line out. Diagnostics belong on stderr. A reference agent can be exercised from the
repository with:

```bash
python crates/sharpearena/examples/reference-agent.py
npx --yes tsx crates/sharpearena/examples/reference-agent.ts
```

The HTTP transport sends an observation to `POST /decide` and expects a decision JSON
response. The protocol does not prescribe how the agent is implemented, trained, or
hosted.

## Failure semantics

A timeout, malformed response, transport loss, duplicate or unknown symbol, or invalid
action/weight pair is not an investment decision. It must not be converted into a
scoreable hold.

Use `run_backtest_checked` with Rust wire transports. It returns either
`CellOutcome::Scored` or `CellOutcome::Failed(TransportFault)`. Python field runners
record the corresponding typed failed cell.

The low-level `run_backtest` function predates this checked boundary. An external
transport implements the `Agent` trait by returning an empty decision after a fault and
recording the fault in `TransportHealth`; a caller that ignores that health can produce
a full return series from a degraded run. Use the unchecked function only for trusted
in-process policies.

## Capture, replay, and evidence

`run_backtest_capture` records raw decisions with the run's window and seed.
`replay_run` and `replay_submission` then regenerate returns from those decisions, the
frozen dataset, and the cost model.

Replay is recomputation, not a complete evidence envelope. The engine does not consume
`DecisionStep.step` or `observation_id`, so changing those metadata fields does not
change the recomputed result. A verifier should bind all of the following separately:

- dataset and cost-model identity;
- engine and `SPEC_HASH` identity;
- expected windows, seeds, retries, and cell count;
- ordered step and observation metadata;
- the raw decision trajectory and its digest.

SharpeBench owns that scoring and attestation layer. SharpeArena owns the environment
and the captured decisions it produces.

## Compatibility

`CONTRACT_VERSION` governs additive wire evolution. New optional fields require
defaults; removals, renames, type changes, or semantic changes require a new major
contract version. Bidirectional fixtures test Rust serialization against the published
schemas.

`SPEC_HASH` answers a different question. It fingerprints the files and dependency pins
that determine tape semantics across Rust, Python, WASM, and npm. A wrapper built for a
different engine hash refuses to start instead of returning a plausible result under
mismatched rules.

Read [`GOVERNANCE.md`](../crates/sharpearena/GOVERNANCE.md) for the normative evolution
rules.
