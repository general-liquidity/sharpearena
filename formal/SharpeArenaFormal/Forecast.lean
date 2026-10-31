/-
Copyright (c) 2026 Tiberiu Toca. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE-APACHE.
Authors: Tiberiu Toca
-/
module

public import Init.Grind

/-!
# Prospective forecast invariants

This module formalizes the deadline classifier, append-only effective-revision rule,
contract binding, and fixed-point Brier identity used by SharpeArena's forecast evidence.

The model is linked to the Python implementation by one mechanism only: the shared case table
`formal/cases/forecast.json`, which `formal/SharpeArenaFormal/Cases.lean` evaluates against
`classify`, `effectiveClaim`, `rebind` and `brierNumerator` while the library builds, and which
`crates/sharpearena-py/tests/test_formal_cases.py` evaluates against `submit`, `effective_claims`
and `score_claim`. There is no extraction or refinement proof relating the two, and agreement on a
finite table is not one. Separate executable tests cover rules this model states, independently of
it:
`crates/sharpearena-py/tests/test_forecast_evidence.py` exercises the eligibility window, late and
pre-open attempts, contract-byte reuse and the fixed-point Brier identity against the Python
implementation.

## Main results

- `classify_eligible_iff`: eligibility is exactly the frozen submission window.
- `applyRevision_late`: a late revision cannot replace the effective revision.
- `effectiveClaim_append_other` and the late/eligible append theorems: a per-claim
  fold selects by append order without cross-claim interference.
- `rebind_accepted_preserves_digest`: an accepted claim binding keeps its contract digest.
- `rebind_accepted_matches_candidate`: acceptance requires the submitted digest
  to match, not merely a return value carrying the old digest.
- `brierNumerator_decomposition`: expected Brier loss is baseline plus squared regret.
- `brierNumerator_minimized`: truthful fixed-point probability minimizes expected loss.
- `brierNumerator_unique`: the minimizer is unique for a positive scale.

## Scope

Covers: the protocol decisions in
`crates/sharpearena-py/python/sharpearena/forecast_evidence.py`: the submission status
classification inside `submit` (`classify`), effective-revision selection in
`effective_claims` (`applyRevision`, `applyClaimRevision`, `effectiveClaim`), the claim-to-contract binding refusal in `submit`
(`rebind`), and the fixed-point binary Brier loss SharpeBench recomputes from the sealed
evidence (`brierNumerator`; Arena's descriptive counterpart is the `brier` entry in
`crates/sharpearena-py/python/sharpearena/deferred.py`).

Assumes: `Nat` timestamps, an integer fixed-point probability `scale > 0`, and no
floating-point semantics. This module is not an extraction of the Python implementation and
carries no refinement proof; the shared case table is the only mechanical link, on its cases.

The list supplied to `effectiveClaim` is append order. Ordinals label records;
this model does not prove their monotonic allocation, hash collision resistance,
deserialization validation, or timestamp ordering.

Not modelled: the environment kernel in `crates/sharpearena/src/market.rs`,
`crates/sharpearena/src/lob_market.rs` and `crates/sharpearena/src/vec_env.rs`. Its reset,
step, terminal, fill and accounting transitions are covered by executable properties
against the shipped functions in `crates/sharpearena/tests/kernel_properties.rs`, not by
this model.

Check: the CI scope check (scripts/check-lean-scope.py) proves that every repository path named in
backticks in this block exists and that every production symbol named in backticks occurs in a file
the block names. It does not prove that the rules described here still correspond to that code.
-/

@[expose] public section

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

/-- A recorded revision carries the claim whose state it may replace. -/
structure ClaimRevision where
  claimId : String
  revision : Revision
  deriving DecidableEq, Repr

/-- Apply an append-order record to one claim, ignoring other claims. -/
def applyClaimRevision (claimId : String) (current : Option Nat)
    (record : ClaimRevision) : Option Nat :=
  if record.claimId = claimId then applyRevision current record.revision else current

/-- Effective state is the per-claim left fold over append order, not timestamp order. -/
def effectiveClaim (claimId : String) (history : List ClaimRevision) : Option Nat :=
  history.foldl (applyClaimRevision claimId) none

/-- Any ineligible append, late or rejected, leaves the per-claim state unchanged. -/
theorem effectiveClaim_append_ineligible (claimId : String) (history : List ClaimRevision)
    (record : ClaimRevision) (h : record.revision.status ≠ .eligible) :
    effectiveClaim claimId (history ++ [record]) = effectiveClaim claimId history := by
  simp [effectiveClaim, List.foldl_append, applyClaimRevision, applyRevision, h]

/-- Appending another claim's revision does not change this claim's result. -/
theorem effectiveClaim_append_other (claimId : String) (history : List ClaimRevision)
    (record : ClaimRevision) (h : record.claimId ≠ claimId) :
    effectiveClaim claimId (history ++ [record]) = effectiveClaim claimId history := by
  simp [effectiveClaim, List.foldl_append, applyClaimRevision, h]

/-- A late append cannot overwrite the effective result even after an arbitrary history. -/
theorem effectiveClaim_append_late (claimId : String) (history : List ClaimRevision)
    (ordinal : Nat) :
    effectiveClaim claimId (history ++ [{ claimId, revision := { ordinal, status := .late } }]) =
      effectiveClaim claimId history := by
  simp [effectiveClaim, List.foldl_append, applyClaimRevision, applyRevision]

/-- The latest appended eligible revision wins, irrespective of its timestamp or ordinal size. -/
theorem effectiveClaim_append_eligible (claimId : String) (history : List ClaimRevision)
    (ordinal : Nat) :
    effectiveClaim claimId (history ++ [{ claimId, revision := { ordinal, status := .eligible } }]) =
      some ordinal := by
  simp [effectiveClaim, List.foldl_append, applyClaimRevision, applyRevision]

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

/-- Acceptance also requires the candidate digest, not only the returned digest, to match. -/
theorem rebind_accepted_matches_candidate {prior result : ClaimBinding}
    {candidateDigest : String} (h : rebind prior candidateDigest = some result) :
    candidateDigest = result.contractDigest := by
  grind [rebind]

/-- An accepted binding preserves the claim identity as well as its contract. -/
theorem rebind_accepted_preserves_claim {prior result : ClaimBinding}
    {candidateDigest : String} (h : rebind prior candidateDigest = some result) :
    result.claimId = prior.claimId := by
  grind [rebind]

/-- A changed digest is refused rather than accepted with the prior value. -/
theorem rebind_changed_digest_refused (prior : ClaimBinding) (candidateDigest : String)
    (h : candidateDigest ≠ prior.contractDigest) :
    rebind prior candidateDigest = none := by
  simp [rebind, h]

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
theorem intSquare_nonneg (value : Int) : 0 ≤ intSquare value := by
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
    Int.mul_nonneg hscale (intSquare_nonneg _)
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
