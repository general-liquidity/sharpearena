import Lake

open Lake DSL

package SharpeArenaFormal where
  version := v!"0.1.0"
  -- Every variable in a statement is declared, so a misspelt name is an error rather than a
  -- fresh universally bound variable; and every warning, `sorry` included, fails the build.
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩,
    ⟨`warningAsError, true⟩
  ]

/-- The shared conformance cases. `SharpeArenaFormal.Cases` reads this file while it elaborates,
so the library depends on it and rebuilds when it changes. -/
input_file formalCases where
  path := "cases/forecast.json"
  text := true

@[default_target]
lean_lib SharpeArenaFormal where
  needs := #[formalCases]
