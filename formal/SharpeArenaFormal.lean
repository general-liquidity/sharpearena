/-
Copyright (c) 2026 Tiberiu Toca. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE-APACHE.
Authors: Tiberiu Toca
-/
module

public import SharpeArenaFormal.Forecast

/-!
# SharpeArena formal model

Selected mathematical and lifecycle invariants used by the prospective-forecast protocol.

## Scope

Covers: no rule of its own. This is the library root; it re-exports
`formal/SharpeArenaFormal/Forecast.lean`, whose own `## Scope` block names the rules that module
models and the assumptions its proofs rest on.

Assumes: nothing beyond the assumptions the re-exported modules declare.

Check: the CI scope check (scripts/check-lean-scope.py) proves only that every repository path
named in backticks in this block exists. It does not prove that the rules described here still
correspond to the code at that path.
-/
