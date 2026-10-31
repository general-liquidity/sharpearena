/-
Copyright (c) 2026 Tiberiu Toca. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE-APACHE.
Authors: Tiberiu Toca
-/
module

public import SharpeArenaFormal.Forecast
meta import SharpeArenaFormal.Forecast
meta import Lean.Data.Json

/-!
# Shared conformance cases

`formal/cases/forecast.json` holds concrete cases for submission classification, effective-revision
selection, claim-to-contract binding and the fixed-point Brier numerator. This module evaluates
every case against `classify`, `effectiveClaim`, `rebind` and `brierNumerator` while it elaborates
and throws on the first disagreement, so a wrong case or a changed definition fails `lake build`.
The Python test `crates/sharpearena-py/tests/test_formal_cases.py` reads the same file and checks
each case against `ForecastLedger.submit`, `ForecastLedger.effective_claims` and `score_claim`.
That is the mechanical link between the model and the implementation: one table both must agree
with. It is a finite set of cases, not a refinement proof.

## Scope

Covers: the cases in `formal/cases/forecast.json`, checked here against
`formal/SharpeArenaFormal/Forecast.lean` and in Python against
`crates/sharpearena-py/python/sharpearena/forecast_evidence.py` and
`crates/sharpearena-py/python/sharpearena/deferred.py` by
`crates/sharpearena-py/tests/test_formal_cases.py`.

Assumes: integer timestamps; one contract window per effective-revision case; a submission's
ordinal is the number of earlier submissions for its claim, which is how the ledger numbers them.
-/

open Lean

namespace SharpeArenaFormal.Cases

meta def field {α : Type} [FromJson α] (case : Json) (key : String) : IO α :=
  IO.ofExcept ((case.getObjVal? key).bind fromJson?)

meta def statusName : RevisionStatus → String
  | .rejected => "rejected"
  | .eligible => "eligible"
  | .late => "late"

meta def checkClassify (case : Json) : IO Unit := do
  let opensAt : Nat ← field case "opens_at"
  let deadline : Nat ← field case "deadline"
  let submittedAt : Nat ← field case "submitted_at"
  let status : String ← field case "status"
  let got := statusName (classify opensAt deadline submittedAt)
  unless got == status do
    throw <| IO.userError
      s!"classify {opensAt} {deadline} {submittedAt} is {got}, case says {status}"

/-- Number each submission by the earlier submissions for its claim and classify it. -/
meta def history (opensAt deadline : Nat) (entries : Array Json) : IO (List ClaimRevision) := do
  let mut seen : List String := []
  let mut out : Array ClaimRevision := #[]
  for entry in entries do
    let claim : String ← field entry "claim"
    let submittedAt : Nat ← field entry "submitted_at"
    let ordinal := seen.count claim
    seen := claim :: seen
    out := out.push ⟨claim, ⟨ordinal, classify opensAt deadline submittedAt⟩⟩
  return out.toList

meta def checkEffective (case : Json) : IO Unit := do
  let name : String ← field case "name"
  let opensAt : Nat ← field case "opens_at"
  let deadline : Nat ← field case "deadline"
  let entries : Array Json ← field case "history"
  let revisions ← history opensAt deadline entries
  let expected ← IO.ofExcept ((case.getObjVal? "effective").bind Json.getObj?)
  for ⟨claim, value⟩ in expected.toArray do
    let ordinal : Option Nat ← IO.ofExcept (fromJson? value)
    unless effectiveClaim claim revisions == ordinal do
      throw <| IO.userError
        s!"{name}: effectiveClaim {claim} is {effectiveClaim claim revisions}, case says {ordinal}"

meta def checkRebind (case : Json) : IO Unit := do
  let prior : String ← field case "prior"
  let candidate : String ← field case "candidate"
  let accepted : Bool ← field case "accepted"
  unless (rebind ⟨"claim", prior⟩ candidate).isSome == accepted do
    throw <| IO.userError s!"rebind {prior} {candidate} is not accepted = {accepted}"

meta def checkBrier (case : Json) : IO Unit := do
  let probability : Int ← field case "probability"
  let truth : Int ← field case "truth"
  let scale : Int ← field case "scale"
  let numerator : Int ← field case "numerator"
  unless brierNumerator probability truth scale == numerator do
    throw <| IO.userError s!"brierNumerator {probability} {truth} {scale} is not {numerator}"

meta def section_ (cases : Json) (key : String) (check : Json → IO Unit) : IO Unit := do
  let rows : Array Json ← field cases key
  if rows.isEmpty then throw <| IO.userError s!"cases/forecast.json has no {key} cases"
  rows.forM check

#eval show IO Unit from do
  let cases ← IO.ofExcept (Json.parse (← IO.FS.readFile "cases/forecast.json"))
  section_ cases "classify" checkClassify
  section_ cases "effective" checkEffective
  section_ cases "rebind" checkRebind
  section_ cases "brier" checkBrier

end SharpeArenaFormal.Cases
