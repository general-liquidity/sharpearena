# Formal model

This Lean project models selected invariants of the rules SharpeArena declares
for its prospective forecast protocol, and proves them about the model:

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

The model states the protocol decisions declared in
`crates/sharpearena-py/python/sharpearena/forecast_evidence.py` and the formulas
independently recomputed by SharpeBench. It is not an extraction of Python or
Rust semantics, and there is no mechanical link or refinement proof between the
model and either implementation: no Rust, Python or TOML file references a Lean
declaration. Separate executable tests cover several of the same rules
independently of this model:
`crates/sharpearena-py/tests/test_forecast_evidence.py` exercises the
eligibility window, late and pre-open attempts, contract-byte reuse and the
fixed-point Brier identity against the Python implementation, and SharpeBench's
`fixed_point_brier_model_matches_the_executable_float_rule` test checks the same
Brier identity against its Rust scorer.

## Scope

Every module under `SharpeArenaFormal/` carries a `## Scope` block in its doc
comment with a `Covers:` line (the production symbols it models) and an
`Assumes:` line (the assumptions the proofs rest on).
`scripts/check-lean-scope.py` fails CI when a module lacks the block or its
block references no existing repository path; it runs as one step of the
`Lean model` job. The check proves that a named path exists, not that the model
still corresponds to the code at that path.

`Forecast.lean` covers the submission status classification and claim binding
inside `submit`, effective-revision selection in `effective_claims`, and the
fixed-point binary Brier loss, under `Nat` timestamps, an integer probability
scale above zero, and no floating-point semantics. The environment kernel
(`crates/sharpearena/src/market.rs`, `lob_market.rs`, `vec_env.rs`) is not
modelled here. Its reset, step, terminal, fill and accounting transitions are
covered by seeded properties executed against the shipped functions in
`crates/sharpearena/tests/kernel_properties.rs` and, for the Python
limit-order-book accounts, `crates/sharpearena-py/tests/test_lob_accounting_properties.py`.
