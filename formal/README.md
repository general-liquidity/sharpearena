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
