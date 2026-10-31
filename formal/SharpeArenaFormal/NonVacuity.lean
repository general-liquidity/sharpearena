/-
Copyright (c) 2026 Tiberiu Toca. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE-APACHE.
Authors: Tiberiu Toca
-/
module

public import SharpeArenaFormal.Forecast

/-!
# Non-vacuity of the forecast-protocol theorems

A theorem whose hypotheses cannot be met is true of nothing, and a theorem that holds of every
definition says nothing about the one it names. For every theorem in `Forecast.lean` this module
gives two things, each checked by `decide` on a concrete instance:

- a witness: an instance where the hypotheses hold and the conclusion is not trivially true;
- a mutant: a plausible wrong definition, and an instance on which the theorem's property fails
  for it, so the property separates the definition from that bug.

`scripts/check-lean-scope.py` fails CI when a theorem of the library is not named in this module.

## Scope

Covers: no production rule of its own. It exercises the model in
`formal/SharpeArenaFormal/Forecast.lean`, whose own block names the production code.

Assumes: the definitions of that module, evaluated on small natural-number and integer instances.
-/

namespace SharpeArenaFormal.NonVacuity

open SharpeArenaFormal

/-! ## `classify_eligible_iff`, `classify_late` and `classify_rejected` -/

/-- Witness: the closed window admits both of its ends, and each side of it is classified. -/
example : classify 10 20 10 = .eligible ∧ classify 10 20 20 = .eligible ∧
    (10 ≤ 20 ∧ 20 < 21) ∧ classify 10 20 21 = .late ∧ (9 < 10) ∧ classify 10 20 9 = .rejected := by
  decide

/-- Mutant: a half-open window that treats a submission at the deadline as late. -/
def classifyHalfOpen (opensAt deadline submittedAt : Nat) : RevisionStatus :=
  if submittedAt < opensAt then .rejected else if deadline ≤ submittedAt then .late else .eligible

example : classifyHalfOpen 10 20 20 ≠ .eligible ∧ (10 ≤ 20 ∧ 20 ≤ 20) := by decide

/-- Mutant: no deadline at all. -/
def classifyNoDeadline (opensAt _deadline submittedAt : Nat) : RevisionStatus :=
  if submittedAt < opensAt then .rejected else .eligible

example : 10 ≤ 20 ∧ 20 < 21 ∧ classifyNoDeadline 10 20 21 ≠ .late := by decide

/-- Mutant: no opening time at all. -/
def classifyNoOpening (_opensAt deadline submittedAt : Nat) : RevisionStatus :=
  if deadline < submittedAt then .late else .eligible

example : 9 < 10 ∧ classifyNoOpening 10 20 9 ≠ .rejected := by decide

/-! ## `applyRevision_late`, `applyRevision_rejected` and `applyRevision_eligible` -/

/-- Witness: a late and a rejected attempt keep a non-empty effective ordinal; an eligible one
replaces it. -/
example : applyRevision (some 0) ⟨1, .late⟩ = some 0 ∧
    applyRevision (some 0) ⟨1, .rejected⟩ = some 0 ∧
    applyRevision (some 0) ⟨1, .eligible⟩ = some 1 := by decide

/-- Mutant: every attempt replaces the effective ordinal, whatever its status. -/
def applyRevisionAny (_current : Option Nat) (revision : Revision) : Option Nat :=
  some revision.ordinal

example : applyRevisionAny (some 0) ⟨1, .late⟩ ≠ some 0 ∧
    applyRevisionAny (some 0) ⟨1, .rejected⟩ ≠ some 0 := by decide

/-- Mutant: the first eligible revision is kept and later ones are ignored. -/
def applyRevisionFirst (current : Option Nat) (revision : Revision) : Option Nat :=
  if current.isNone && revision.status = .eligible then some revision.ordinal else current

example : applyRevisionFirst (some 0) ⟨1, .eligible⟩ ≠ some 1 := by decide

/-! ## `effectiveClaim_append_ineligible`, `effectiveClaim_append_other`,
`effectiveClaim_append_late` and `effectiveClaim_append_eligible` -/

/-- A history with an effective revision for claim `"x"`. -/
def sampleHistory : List ClaimRevision := [⟨"x", ⟨0, .eligible⟩⟩, ⟨"y", ⟨0, .eligible⟩⟩]

/-- Witness: the history has an effective revision, and each kind of append is exercised on it. -/
example : effectiveClaim "x" sampleHistory = some 0 ∧
    effectiveClaim "x" (sampleHistory ++ [⟨"x", ⟨1, .late⟩⟩]) = some 0 ∧
    effectiveClaim "x" (sampleHistory ++ [⟨"x", ⟨1, .rejected⟩⟩]) = some 0 ∧
    effectiveClaim "x" (sampleHistory ++ [⟨"y", ⟨1, .eligible⟩⟩]) = some 0 ∧
    effectiveClaim "x" (sampleHistory ++ [⟨"x", ⟨1, .eligible⟩⟩]) = some 1 := by decide

/-- Mutant: a fold that applies every status, so a late or rejected append replaces the claim. -/
def effectiveClaimAnyStatus (claimId : String) (history : List ClaimRevision) : Option Nat :=
  history.foldl (fun current record ↦
    if record.claimId = claimId then applyRevisionAny current record.revision else current) none

example : effectiveClaimAnyStatus "x" (sampleHistory ++ [⟨"x", ⟨1, .late⟩⟩]) ≠ some 0 := by decide

/-- Mutant: a fold that ignores which claim a record belongs to. -/
def effectiveClaimAnyClaim (_claimId : String) (history : List ClaimRevision) : Option Nat :=
  history.foldl (fun current record ↦ applyRevision current record.revision) none

example : effectiveClaimAnyClaim "x" (sampleHistory ++ [⟨"y", ⟨1, .eligible⟩⟩]) ≠
    effectiveClaimAnyClaim "x" sampleHistory := by decide

/-- Mutant: selection by the largest ordinal rather than by append order. -/
def effectiveClaimMaxOrdinal (claimId : String) (history : List ClaimRevision) : Option Nat :=
  history.foldl (fun current record ↦
    if record.claimId = claimId ∧ record.revision.status = .eligible then
      some (max (current.getD 0) record.revision.ordinal)
    else current) none

example : effectiveClaimMaxOrdinal "x" ([⟨"x", ⟨5, .eligible⟩⟩] ++ [⟨"x", ⟨2, .eligible⟩⟩]) ≠
    some 2 := by decide

/-! ## `rebind_accepted_preserves_digest`, `rebind_accepted_matches_candidate`,
`rebind_accepted_preserves_claim` and `rebind_changed_digest_refused` -/

/-- Witness: a repeated binding with the frozen digest is accepted, and returns the prior. -/
example : rebind ⟨"c", "a"⟩ "a" = some ⟨"c", "a"⟩ ∧ rebind ⟨"c", "a"⟩ "b" = none := by decide

/-- Mutant: every candidate is accepted with the prior binding returned. The returned digest is
the prior one, so it would pass a check on the returned value alone, but not the candidate check. -/
def rebindAcceptAll (prior : ClaimBinding) (_candidateDigest : String) : Option ClaimBinding :=
  some prior

example : rebindAcceptAll ⟨"c", "a"⟩ "b" = some ⟨"c", "a"⟩ ∧ "b" ≠ "a" ∧
    rebindAcceptAll ⟨"c", "a"⟩ "b" ≠ none := by decide

/-- Mutant: every candidate is accepted and rebinds the claim to it. -/
def rebindOverwrite (prior : ClaimBinding) (candidateDigest : String) : Option ClaimBinding :=
  some { prior with contractDigest := candidateDigest }

example : rebindOverwrite ⟨"c", "a"⟩ "b" = some ⟨"c", "b"⟩ ∧ "b" ≠ "a" := by decide

/-- Mutant: an accepted binding comes back under another claim's identity. -/
def rebindAnonymous (prior : ClaimBinding) (candidateDigest : String) : Option ClaimBinding :=
  if candidateDigest = prior.contractDigest then some { prior with claimId := "" } else none

example : rebindAnonymous ⟨"c", "a"⟩ "a" = some ⟨"", "a"⟩ ∧ "" ≠ "c" := by decide

/-! ## `intSquare_nonneg`, `brierNumerator_decomposition`, `brierNumerator_minimized` and
`brierNumerator_unique` -/

/-- Witness: a negative value squares to a positive one; on a positive scale the truthful report
scores strictly better than another, and the decomposition holds with a non-zero regret term. -/
example : 0 < intSquare (-3) ∧ 0 ≤ (10 : Int) ∧ 0 < (10 : Int) ∧
    brierNumerator 7 7 10 < brierNumerator 3 7 10 ∧
    brierNumerator 3 7 10 = 7 * (10 - 7) * 10 + 10 * intSquare (3 - 7) := by decide

/-- Mutant: a signed square. -/
def intSquareSigned (value : Int) : Int := value * value.natAbs

example : ¬ 0 ≤ intSquareSigned (-3) := by decide

/-- Mutant: an absolute-error loss in place of the squared one. It is not proper: on scale 10 with
truth 7, reporting 10 scores better than reporting 7, so neither the decomposition, the
minimization nor the uniqueness holds of it. -/
def absoluteNumerator (probability truth scale : Int) : Int :=
  truth * (scale - probability).natAbs + (scale - truth) * probability.natAbs

example : absoluteNumerator 10 7 10 < absoluteNumerator 7 7 10 ∧
    absoluteNumerator 3 7 10 ≠ 7 * (10 - 7) * 10 + 10 * intSquare (3 - 7) ∧
    absoluteNumerator 3 5 10 = absoluteNumerator 5 5 10 ∧ (3 : Int) ≠ 5 := by decide

end SharpeArenaFormal.NonVacuity
