/-
Copyright (c) 2026 Tiberiu Toca. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE-APACHE.
Authors: Tiberiu Toca
-/
module

public import Init.Grind
public import Std

/-!
# Prospective forecast invariants

This module formalizes the deadline classifier, append-only effective-revision rule,
contract binding, and fixed-point Brier identity used by SharpeArena's forecast evidence.

The executable Python implementation is linked to this model by conformance tests. This
module is not a proof-producing extraction of that Python program.

## Main results

- `classify_eligible_iff`: eligibility is exactly the frozen submission window.
- `applyRevision_late`: a late revision cannot replace the effective revision.
- `rebind_accepted_preserves_digest`: an accepted claim binding keeps its contract digest.
- `brierNumerator_decomposition`: expected Brier loss is baseline plus squared regret.
- `brierNumerator_minimized`: truthful fixed-point probability minimizes expected loss.
- `brierNumerator_unique`: the minimizer is unique for a positive scale.
-/

public section

namespace SharpeArenaFormal

/-- The three submission states represented in a forecast evidence revision. -/
inductive RevisionStatus where
  | rejected
  | eligible
  | late
  deriving DecidableEq, Repr

/-- Classify a submission against the contract's closed eligibility window. -/
def classify (opensAt deadline submittedAt : Nat) : RevisionStatus :=
  if submittedAt < opensAt then
    .rejected
  else if deadline < submittedAt then
    .late
  else
    .eligible

/-- A submission is eligible exactly when it lies in the frozen closed window. -/
theorem classify_eligible_iff (opensAt deadline submittedAt : Nat) :
    classify opensAt deadline submittedAt = .eligible ↔
      opensAt ≤ submittedAt ∧ submittedAt ≤ deadline := by
  grind [classify]

/-- A submission after a valid contract's deadline is classified as late. -/
theorem classify_late {opensAt deadline submittedAt : Nat}
    (hwindow : opensAt ≤ deadline) (h : deadline < submittedAt) :
    classify opensAt deadline submittedAt = .late := by
  grind [classify]

/-- A submission before the opening time is classified as rejected. -/
theorem classify_rejected {opensAt deadline submittedAt : Nat}
    (h : submittedAt < opensAt) :
    classify opensAt deadline submittedAt = .rejected := by
  grind [classify]

/-- The state needed to choose the latest eligible revision. -/
structure Revision where
  ordinal : Nat
  status : RevisionStatus
  deriving DecidableEq, Repr

/-- Apply one append-only revision to the currently effective ordinal. -/
def applyRevision (current : Option Nat) (revision : Revision) : Option Nat :=
  if revision.status = .eligible then some revision.ordinal else current

/-- A late attempt remains recorded but cannot replace the effective ordinal. -/
theorem applyRevision_late (current : Option Nat) (ordinal : Nat) :
    applyRevision current { ordinal, status := .late } = current := by
  simp [applyRevision]

/-- A pre-open rejected attempt cannot replace the effective ordinal. -/
theorem applyRevision_rejected (current : Option Nat) (ordinal : Nat) :
    applyRevision current { ordinal, status := .rejected } = current := by
  simp [applyRevision]

/-- An eligible revision becomes the effective ordinal. -/
theorem applyRevision_eligible (current : Option Nat) (ordinal : Nat) :
    applyRevision current { ordinal, status := .eligible } = some ordinal := by
  simp [applyRevision]

/-- The immutable contract digest bound to a claim. -/
structure ClaimBinding where
  claimId : String
  contractDigest : String
  deriving DecidableEq, Repr

/-- Accept a repeated binding only when its digest matches the frozen binding. -/
def rebind (prior : ClaimBinding) (candidateDigest : String) : Option ClaimBinding :=
  if candidateDigest = prior.contractDigest then some prior else none

/-- Every accepted repeated binding preserves the original contract digest. -/
theorem rebind_accepted_preserves_digest {prior result : ClaimBinding}
    {candidateDigest : String} (h : rebind prior candidateDigest = some result) :
    result.contractDigest = prior.contractDigest := by
  simp only [rebind] at h
  split at h
  · simpa using (congrArg ClaimBinding.contractDigest (Option.some.inj h)).symm
  · contradiction

/-- An integer square used by the fixed-point Brier model. -/
def intSquare (value : Int) : Int :=
  value * value

/-- The numerator of expected Brier loss on a fixed-point probability scale.

If `probability / scale` is reported and `truth / scale` is the event probability,
the expected Brier loss is this value divided by `scale ^ 3`. The positive constant
does not change the minimizer.
-/
def brierNumerator (probability truth scale : Int) : Int :=
  truth * intSquare (scale - probability) +
    (scale - truth) * intSquare probability

/-- Every integer square is nonnegative. -/
theorem intSquare_nonnegative (value : Int) : 0 ≤ intSquare value := by
  rcases Int.le_total 0 value with hvalue | hvalue
  · exact Int.mul_nonneg hvalue hvalue
  · exact Int.mul_nonneg_of_nonpos_of_nonpos hvalue hvalue

/-- Fixed-point expected Brier loss is uncertainty plus scaled squared regret. -/
theorem brierNumerator_decomposition (probability truth scale : Int) :
    brierNumerator probability truth scale =
      truth * (scale - truth) * scale +
        scale * intSquare (probability - truth) := by
  grind [brierNumerator, intSquare]

/-- Reporting the true fixed-point probability minimizes expected Brier loss. -/
theorem brierNumerator_minimized (probability truth scale : Int)
    (hscale : 0 ≤ scale) :
    brierNumerator truth truth scale ≤ brierNumerator probability truth scale := by
  rw [brierNumerator_decomposition, brierNumerator_decomposition]
  have hregret : 0 ≤ scale * intSquare (probability - truth) :=
    Int.mul_nonneg hscale (intSquare_nonnegative _)
  grind [intSquare]

/-- On a positive scale, only the true fixed-point probability attains the minimum. -/
theorem brierNumerator_unique {probability truth scale : Int}
    (hscale : 0 < scale)
    (hscore : brierNumerator probability truth scale = brierNumerator truth truth scale) :
    probability = truth := by
  rw [brierNumerator_decomposition, brierNumerator_decomposition] at hscore
  have hregret : scale * intSquare (probability - truth) = 0 := by
    grind [intSquare]
  have hscale0 : scale ≠ 0 := by grind
  have hsquare : intSquare (probability - truth) = 0 :=
    (Int.mul_eq_zero.mp hregret).resolve_left hscale0
  have hdifference : probability - truth = 0 := by
    rcases Int.mul_eq_zero.mp hsquare with hzero | hzero
    · exact hzero
    · exact hzero
  exact Int.sub_eq_zero.mp hdifference

end SharpeArenaFormal
