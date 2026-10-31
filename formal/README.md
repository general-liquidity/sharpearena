# Formal model

This Lean project models selected invariants of the rules SharpeArena declares
for its prospective forecast protocol, and proves them about the model:

- the exact eligibility window and late-attempt behavior;
- append-only effective-revision selection;
- immutable claim-to-contract binding;
- exact decomposition, minimization, and uniqueness of fixed-point binary
  Brier scoring.

The revision model now includes a per-claim left fold over an arbitrary history:
another claim's append and a late append preserve the result; the last appended
eligible record replaces it. Selection is not maximum timestamp or maximum
ordinal. Binding proofs require the submitted digest to equal the accepted
digest and preserve the claim ID; a different digest is refused. These are
model-level statements, not cryptographic collision-resistance or ordinal
allocation proofs. The supplied list is assumed to be append order.

Build it with:

```console
cd formal
lake build
```

The scoring proof uses integer fixed-point probabilities. Its expected loss is
the real-valued Brier expectation multiplied by a positive scale cubed, so it
has the same unique minimizer without depending on floating-point semantics.

The package sets `autoImplicit false`, so a misspelt variable in a statement is
an error rather than a fresh universally bound variable, and `warningAsError
true`, so a `sorry` fails the build.

## What links the model to the code

The model states the protocol decisions declared in
`crates/sharpearena-py/python/sharpearena/forecast_evidence.py` and the formulas
independently recomputed by SharpeBench. It is not an extraction of Python or
Rust semantics and there is no refinement proof. One mechanism links it to the
Python implementation: the shared case table `formal/cases/forecast.json`.
`SharpeArenaFormal/Cases.lean` evaluates every case against `classify`,
`effectiveClaim`, `rebind` and `brierNumerator` while the library builds and
fails the build on a disagreement. `crates/sharpearena-py/tests/test_formal_cases.py`
runs every case through the production code: the status `ForecastLedger.submit`
records against the case's window, the revision `ForecastLedger.effective_claims`
selects after the case's history, whether `submit` refuses to rebind a claim to
another contract, and the expected Brier loss from `score_claim` scaled by
`scale ^ 3`. That links those four rules on the cases in the table only. The
model's binding compares digests where the ledger compares contract ids, so the
binding cases test refusal of a different contract, not digest comparison. The
Rust scorer in SharpeBench is not part of this link. Separate executable tests
cover several of the same rules independently of the model:
`crates/sharpearena-py/tests/test_forecast_evidence.py` exercises the
eligibility window, late and pre-open attempts, contract-byte reuse and the
fixed-point Brier identity against the Python implementation, and SharpeBench's
`fixed_point_brier_model_matches_the_executable_float_rule` test checks the same
Brier identity against its Rust scorer.

## Gates

The `Lean model` CI job fails when any of these fails:

- `lake build`, then `leanchecker` (a kernel re-check of the library) and the
  axiom audit of `leanprover/lean-action`;
- the statement manifest: `formal/scripts/Statements.lean` prints every
  declaration's elaborated type, every definition's body and every theorem's
  axioms, and CI diffs that output against the committed
  `formal/statements.txt`. A weakened statement, a dropped converse or an extra
  hypothesis passes build, leanchecker and the axiom audit, and fails here. A
  deliberate change regenerates the file (from `formal/`,
  `lake env lean --run scripts/Statements.lean SharpeArenaFormal > statements.txt`)
  and the diff is reviewed with the change. The same script refuses any declared
  `axiom` and any dependency outside `propext`, `Classical.choice` and
  `Quot.sound`, a second axiom check the repository owns;
- `scripts/check-lean-scope.py` (below), and its own tests in
  `scripts/test_check_lean_scope.py`;
- the negative controls: `formal/scripts/negative_controls.py` copies the
  package to a scratch directory and plants a `sorry`, a `sorry` with
  warnings-as-errors switched off, an extra axiom, an undeclared variable, a
  weakened statement and a wrong case row, and fails unless each is refused by
  the gate meant to refuse it.

Every theorem has a non-vacuity entry in `SharpeArenaFormal/NonVacuity.lean`: an
instance, checked by `decide`, where its hypotheses hold and its conclusion is
not trivial, and a plausible wrong definition with a decided instance on which
the theorem's property fails. The scope check fails when a theorem has none.

## Scope

Every module in the `SharpeArenaFormal` library, the root module and every
module under `SharpeArenaFormal/` at any depth, carries a `## Scope` block in
its doc comment with a `Covers:` line (the production symbols it models) and an
`Assumes:` line (the assumptions the proofs rest on).
`scripts/check-lean-scope.py` fails CI when a module lacks the block, when its
block names a repository path that does not exist, when it names a production
symbol (a backticked identifier that is not a Lean name the library uses) that
occurs in none of the non-Lean files the block names, when a module's
`## Main results` list names a theorem the module does not declare, or when a
theorem has no entry in the non-vacuity module. The check proves that every name
exists where the documentation says, not that the model still corresponds to the
code at those names.

`Forecast.lean` covers the submission status classification and claim binding
inside `submit`, effective-revision selection in `effective_claims`, and the
fixed-point binary Brier loss, under `Nat` timestamps, an integer probability
scale above zero, and no floating-point semantics. The environment kernel
(`crates/sharpearena/src/market.rs`, `lob_market.rs`, `vec_env.rs`) is not
modelled here. Its reset, step, terminal, fill and accounting transitions are
covered by seeded properties executed against the shipped functions in
`crates/sharpearena/tests/kernel_properties.rs` and, for the Python
limit-order-book accounts, `crates/sharpearena-py/tests/test_lob_accounting_properties.py`.
