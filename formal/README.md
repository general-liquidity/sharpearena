# Formal model

This Lean project verifies selected invariants behind SharpeArena's prospective
forecast protocol:

- the exact eligibility window and late-attempt behavior;
- append-only effective-revision selection;
- immutable claim-to-contract binding;
- exact decomposition, minimization, and uniqueness of fixed-point binary
  Brier scoring.

Build it with:

```console
cd formal
lake build
```

The scoring proof uses integer fixed-point probabilities. Its expected loss is
the real-valued Brier expectation multiplied by a positive scale cubed, so it
has the same unique minimizer without depending on floating-point semantics.

The proofs model the protocol decisions in
`crates/sharpearena-py/python/sharpearena/forecast_evidence.py` and the formulas
independently recomputed by SharpeBench. They are not an extraction of Python or
Rust semantics. Executable conformance tests remain responsible for connecting
the model to those implementations.

## Scope

Every module under `SharpeArenaFormal/` carries a `## Scope` block in its doc
comment with a `Covers:` line (the production symbols it models) and an
`Assumes:` line (the assumptions the proofs rest on).
`scripts/check-lean-scope.py` fails CI when a module lacks the block or its
block references no existing repository path; it runs as one step of the
`Lean model` job.

`Forecast.lean` covers the submission status classification and claim binding
inside `submit`, effective-revision selection in `effective_claims`, and the
fixed-point binary Brier loss, under `Nat` timestamps, an integer probability
scale above zero, and no floating-point semantics. The environment kernel
(`crates/sharpearena/src/market.rs`, `lob_market.rs`, `vec_env.rs`) is not
modelled here. Its reset, step, terminal, fill and accounting transitions are
covered by seeded properties executed against the shipped functions in
`crates/sharpearena/tests/kernel_properties.rs` and, for the Python
limit-order-book accounts, `crates/sharpearena-py/tests/test_lob_accounting_properties.py`.
