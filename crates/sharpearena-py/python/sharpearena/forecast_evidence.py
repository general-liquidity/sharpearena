"""Append-only prospective forecast ledger and cross-product evidence export.

The ledger records attempts, not just successful forecasts.  A late or rejected
revision remains visible and can never replace the latest eligible revision.  The
export contains raw predictions and outcomes only; SharpeBench recomputes every
forecast score independently.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .deferred import Claim, ClaimRejected, Outcome, score_claim
from .forecast_contract import (
    FORECAST_EVIDENCE_SCHEMA_VERSION,
    ForecastContract,
    ForecastContractError,
    canonical_json,
)


ELIGIBLE = "eligible"
LATE = "late"
REJECTED = "rejected"
PENDING = "pending"
RESOLVED = "resolved"
CANCELLED = "cancelled"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ForecastEvidenceError(ValueError):
    """The append-only ledger or its exported evidence violates the v1 contract."""


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForecastEvidenceError(f"{field} must be a non-empty string")
    return value.strip()


def _clock(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ForecastEvidenceError(f"{field} must be a non-negative integer")
    return value


def _digest(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if not _SHA256.fullmatch(text):
        raise ForecastEvidenceError(f"{field} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class ForecastRunIdentity:
    """Identity of the deployed system that produced one forecast ledger."""

    agent_id: str
    model_id: str
    model_sha256: str
    scaffold_id: str
    scaffold_sha256: str
    prompt_sha256: str
    operator_id: str
    config_sha256: str

    def __post_init__(self) -> None:
        for field in ("agent_id", "model_id", "scaffold_id", "operator_id"):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        for field in ("model_sha256", "scaffold_sha256", "prompt_sha256", "config_sha256"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    def to_dict(self) -> dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "scaffold_id": self.scaffold_id,
            "scaffold_sha256": self.scaffold_sha256,
            "prompt_sha256": self.prompt_sha256,
            "operator_id": self.operator_id,
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True)
class InformationExposure:
    """Hashed information available when a forecast or revision was submitted."""

    observed_at: int
    market_snapshot_sha256: Optional[str] = None
    consensus_visible: bool = False
    consensus_snapshot_sha256: Optional[str] = None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _clock(self.observed_at, "observed_at"))
        if self.market_snapshot_sha256 is not None:
            object.__setattr__(
                self,
                "market_snapshot_sha256",
                _digest(self.market_snapshot_sha256, "market_snapshot_sha256"),
            )
        if not isinstance(self.consensus_visible, bool):
            raise ForecastEvidenceError("consensus_visible must be boolean")
        if self.consensus_visible:
            if self.consensus_snapshot_sha256 is None:
                raise ForecastEvidenceError(
                    "a visible consensus requires consensus_snapshot_sha256"
                )
            object.__setattr__(
                self,
                "consensus_snapshot_sha256",
                _digest(self.consensus_snapshot_sha256, "consensus_snapshot_sha256"),
            )
        elif self.consensus_snapshot_sha256 is not None:
            raise ForecastEvidenceError(
                "a hidden consensus cannot carry consensus_snapshot_sha256"
            )
        sources = tuple(_nonempty(value, "source_ids[]") for value in self.source_ids)
        if len(set(sources)) != len(sources):
            raise ForecastEvidenceError("source_ids must be unique")
        object.__setattr__(self, "source_ids", sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "market_snapshot_sha256": self.market_snapshot_sha256,
            "consensus_visible": self.consensus_visible,
            "consensus_snapshot_sha256": self.consensus_snapshot_sha256,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class ForecastRevision:
    revision_id: str
    claim_id: str
    ordinal: int
    supersedes: Optional[str]
    contract_sha256: str
    prediction: tuple[float, ...]
    confidence: float
    rationale: str
    submitted_at: int
    status: str
    reason: Optional[str]
    trigger_event_id: Optional[str]
    revision_reason: Optional[str]
    exposure: InformationExposure
    idempotency_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "claim_id": self.claim_id,
            "ordinal": self.ordinal,
            "supersedes": self.supersedes,
            "contract_sha256": self.contract_sha256,
            "prediction": list(self.prediction),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "reason": self.reason,
            "trigger_event_id": self.trigger_event_id,
            "revision_reason": self.revision_reason,
            "exposure": self.exposure.to_dict(),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class ForecastResolution:
    claim_id: str
    status: str
    outcome: float | str | None
    available_at: Optional[int]
    recorded_at: int
    reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "outcome": self.outcome,
            "available_at": self.available_at,
            "recorded_at": self.recorded_at,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ForecastEvidence:
    identity: ForecastRunIdentity
    contracts: tuple[ForecastContract, ...]
    revisions: tuple[ForecastRevision, ...]
    resolutions: tuple[ForecastResolution, ...]
    generated_at: int
    schema_version: str = FORECAST_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": {"name": "sharpearena", "contract": "native"},
            "generated_at": self.generated_at,
            "identity": self.identity.to_dict(),
            "contracts": [contract.to_dict() for contract in self.contracts],
            "revisions": [revision.to_dict() for revision in self.revisions],
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def write_forecast_evidence(
    path: str | os.PathLike[str], evidence: ForecastEvidence
) -> str:
    """Atomically publish a validated v1 evidence document and return its digest.

    The temporary file is created exclusively in the destination directory, then
    flushed, synced, and replaced.  On POSIX the containing directory is synced as
    well, so a reported success survives a crash after the rename.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = evidence.to_json() + "\n"
    forecast_evidence_from_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha256(payload.encode("utf-8")).hexdigest()


class ForecastLedger:
    """Append-only commit, revision, cancellation, and resolution state."""

    def __init__(self, identity: ForecastRunIdentity) -> None:
        self.identity = identity
        self._contracts: dict[str, ForecastContract] = {}
        self._claim_contract: dict[str, str] = {}
        self._revisions: list[ForecastRevision] = []
        self._idempotency: dict[str, ForecastRevision] = {}
        self._cancellations: dict[str, tuple[int, str]] = {}

    def submit(
        self,
        *,
        claim_id: str,
        contract: ForecastContract,
        prediction: Sequence[float] | float,
        confidence: float,
        rationale: str,
        submitted_at: int,
        idempotency_key: str,
        expected_revision: Optional[int] = None,
        trigger_event_id: Optional[str] = None,
        revision_reason: Optional[str] = None,
        exposure: Optional[InformationExposure] = None,
    ) -> ForecastRevision:
        """Append one submission attempt; idempotent retries return the first record."""

        claim_id = _nonempty(claim_id, "claim_id")
        submitted_at = _clock(submitted_at, "submitted_at")
        idempotency_key = _nonempty(idempotency_key, "idempotency_key")
        if idempotency_key in self._idempotency:
            prior = self._idempotency[idempotency_key]
            candidate = self._revision_payload(
                claim_id,
                contract,
                prediction,
                confidence,
                rationale,
                submitted_at,
                prior.ordinal,
                prior.supersedes,
                prior.status,
                prior.reason,
                trigger_event_id,
                revision_reason,
                exposure,
                idempotency_key,
            )
            if candidate.to_dict() != prior.to_dict():
                raise ForecastEvidenceError(
                    f"idempotency key {idempotency_key!r} was reused with different content"
                )
            return prior

        existing_contract = self._contracts.get(contract.contract_id)
        if existing_contract is not None and existing_contract.sha256 != contract.sha256:
            raise ForecastEvidenceError(
                f"contract_id {contract.contract_id!r} was reused with different bytes"
            )
        bound_contract_id = self._claim_contract.get(claim_id)
        if bound_contract_id is not None and bound_contract_id != contract.contract_id:
            raise ForecastEvidenceError(
                f"claim {claim_id!r} is already bound to contract {bound_contract_id!r}"
            )
        prior = [revision for revision in self._revisions if revision.claim_id == claim_id]
        ordinal = len(prior)
        if expected_revision is not None:
            if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
                raise ForecastEvidenceError("expected_revision must be an integer")
            actual = ordinal - 1
            if expected_revision != actual:
                raise ForecastEvidenceError(
                    f"stale revision for {claim_id!r}: expected {expected_revision}, current {actual}"
                )
        if ordinal == 0 and revision_reason is not None:
            raise ForecastEvidenceError("an initial forecast cannot have revision_reason")
        if ordinal > 0 and (revision_reason is None or not revision_reason.strip()):
            raise ForecastEvidenceError("a revision requires revision_reason")
        status = ELIGIBLE
        reason = None
        if submitted_at < contract.opens_at:
            status = REJECTED
            reason = "submitted before the contract opened"
        elif submitted_at > contract.deadline:
            status = LATE
            reason = "submitted after the contract deadline"
        supersedes = prior[-1].revision_id if prior else None
        revision = self._revision_payload(
            claim_id,
            contract,
            prediction,
            confidence,
            rationale,
            submitted_at,
            ordinal,
            supersedes,
            status,
            reason,
            trigger_event_id,
            revision_reason,
            exposure,
            idempotency_key,
        )
        self._contracts[contract.contract_id] = contract
        self._claim_contract[claim_id] = contract.contract_id
        self._revisions.append(revision)
        self._idempotency[idempotency_key] = revision
        return revision

    def _revision_payload(
        self,
        claim_id: str,
        contract: ForecastContract,
        prediction: Sequence[float] | float,
        confidence: float,
        rationale: str,
        submitted_at: int,
        ordinal: int,
        supersedes: Optional[str],
        status: str,
        reason: Optional[str],
        trigger_event_id: Optional[str],
        revision_reason: Optional[str],
        exposure: Optional[InformationExposure],
        idempotency_key: str,
    ) -> ForecastRevision:
        # Claim performs the one canonical validation of prediction and confidence.
        if isinstance(prediction, bool):
            raise ForecastEvidenceError("prediction must not be boolean")
        validation_at = min(max(submitted_at, contract.opens_at), contract.deadline)
        try:
            checked = Claim(
                claim_id=claim_id,
                contract=contract,
                prediction=(
                    tuple(prediction)
                    if not isinstance(prediction, (int, float))
                    else (float(prediction),)
                ),
                committed_at=validation_at,
                confidence=confidence,
                rationale=rationale,
            )
        except (ClaimRejected, TypeError, ValueError) as error:
            raise ForecastEvidenceError(str(error)) from error
        if trigger_event_id is not None:
            trigger_event_id = _nonempty(trigger_event_id, "trigger_event_id")
        if exposure is None:
            exposure = InformationExposure(observed_at=submitted_at)
        if exposure.observed_at != submitted_at:
            raise ForecastEvidenceError("exposure.observed_at must equal submitted_at")
        return ForecastRevision(
            revision_id=f"{claim_id}:r{ordinal}",
            claim_id=claim_id,
            ordinal=ordinal,
            supersedes=supersedes,
            contract_sha256=contract.sha256,
            prediction=checked.prediction,
            confidence=checked.confidence,
            rationale=checked.rationale,
            submitted_at=submitted_at,
            status=status,
            reason=reason,
            trigger_event_id=trigger_event_id,
            revision_reason=revision_reason,
            exposure=exposure,
            idempotency_key=idempotency_key,
        )

    def cancel(self, claim_id: str, *, recorded_at: int, reason: str) -> None:
        claim_id = _nonempty(claim_id, "claim_id")
        if claim_id not in self._claim_contract:
            raise ForecastEvidenceError(f"cannot cancel unknown claim {claim_id!r}")
        if claim_id in self._cancellations:
            raise ForecastEvidenceError(f"claim {claim_id!r} is already cancelled")
        self._cancellations[claim_id] = (
            _clock(recorded_at, "recorded_at"),
            _nonempty(reason, "reason"),
        )

    def revisions(self) -> tuple[ForecastRevision, ...]:
        return tuple(self._revisions)

    def effective_claims(self) -> tuple[Claim, ...]:
        latest: dict[str, ForecastRevision] = {}
        for revision in self._revisions:
            if revision.status == ELIGIBLE:
                latest[revision.claim_id] = revision
        claims: list[Claim] = []
        for claim_id, revision in latest.items():
            contract_id = self._claim_contract[claim_id]
            claims.append(
                Claim(
                    claim_id=claim_id,
                    contract=self._contracts[contract_id],
                    prediction=revision.prediction,
                    committed_at=revision.submitted_at,
                    confidence=revision.confidence,
                    rationale=revision.rationale,
                )
            )
        return tuple(claims)

    def evidence(
        self, outcomes: Iterable[Outcome], *, generated_at: int
    ) -> ForecastEvidence:
        generated_at = _clock(generated_at, "generated_at")
        effective = {claim.claim_id: claim for claim in self.effective_claims()}
        outcome_by_id: dict[str, Outcome] = {}
        unknown: list[str] = []
        for outcome in outcomes:
            if outcome.claim_id in outcome_by_id:
                raise ForecastEvidenceError(f"duplicate outcome for claim {outcome.claim_id!r}")
            if outcome.claim_id not in self._claim_contract:
                unknown.append(outcome.claim_id)
            outcome_by_id[outcome.claim_id] = outcome
        if unknown:
            raise ForecastEvidenceError(f"outcomes name unknown claims: {sorted(unknown)}")

        resolutions: list[ForecastResolution] = []
        for claim_id in self._claim_contract:
            if claim_id in self._cancellations:
                recorded_at, reason = self._cancellations[claim_id]
                resolutions.append(
                    ForecastResolution(claim_id, CANCELLED, None, None, recorded_at, reason)
                )
                continue
            claim = effective.get(claim_id)
            if claim is None:
                resolutions.append(
                    ForecastResolution(
                        claim_id,
                        REJECTED,
                        None,
                        None,
                        generated_at,
                        "claim has no eligible submission",
                    )
                )
                continue
            outcome = outcome_by_id.get(claim_id)
            if outcome is None:
                resolutions.append(
                    ForecastResolution(claim_id, PENDING, None, None, generated_at, None)
                )
                continue
            if outcome.available_at <= claim.committed_at:
                resolutions.append(
                    ForecastResolution(
                        claim_id,
                        REJECTED,
                        None,
                        outcome.available_at,
                        generated_at,
                        "outcome was available at or before the effective revision",
                    )
                )
                continue
            if outcome.available_at < claim.contract.resolves_at:
                resolutions.append(
                    ForecastResolution(
                        claim_id,
                        REJECTED,
                        None,
                        outcome.available_at,
                        generated_at,
                        "outcome predates the frozen resolution boundary",
                    )
                )
                continue
            # Validate the outcome under the contract here, but export no producer score.
            score_claim(claim, outcome.value)
            resolutions.append(
                ForecastResolution(
                    claim_id,
                    RESOLVED,
                    outcome.value,
                    outcome.available_at,
                    generated_at,
                    None,
                )
            )
        return ForecastEvidence(
            identity=self.identity,
            contracts=tuple(self._contracts.values()),
            revisions=tuple(self._revisions),
            resolutions=tuple(resolutions),
            generated_at=generated_at,
        )


def forecast_evidence_from_json(payload: str) -> dict[str, Any]:
    """Strict shape check for a v1 evidence document before cross-product transfer.

    SharpeBench remains the authoritative semantic consumer.  This local check makes
    accidental extension, truncation, and non-finite JSON fail before the file leaves
    SharpeArena.
    """

    import json

    try:
        document = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ForecastEvidenceError(f"forecast evidence is not strict JSON: {error}") from error
    top = {
        "schema_version",
        "producer",
        "generated_at",
        "identity",
        "contracts",
        "revisions",
        "resolutions",
    }
    if not isinstance(document, dict) or set(document) != top:
        raise ForecastEvidenceError("forecast evidence top-level fields do not match v1")
    if document["schema_version"] != FORECAST_EVIDENCE_SCHEMA_VERSION:
        raise ForecastEvidenceError("unsupported forecast evidence schema_version")
    if document["producer"] != {"name": "sharpearena", "contract": "native"}:
        raise ForecastEvidenceError("producer must identify the native SharpeArena contract")
    _clock(document["generated_at"], "generated_at")
    identity_fields = {
        "agent_id",
        "model_id",
        "model_sha256",
        "scaffold_id",
        "scaffold_sha256",
        "prompt_sha256",
        "operator_id",
        "config_sha256",
    }
    identity = document["identity"]
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ForecastEvidenceError("identity fields do not match v1")
    ForecastRunIdentity(**identity)
    if not isinstance(document["contracts"], list) or not document["contracts"]:
        raise ForecastEvidenceError("contracts must be a non-empty array")
    if not isinstance(document["revisions"], list) or not document["revisions"]:
        raise ForecastEvidenceError("revisions must be a non-empty array")
    if not isinstance(document["resolutions"], list) or not document["resolutions"]:
        raise ForecastEvidenceError("resolutions must be a non-empty array")

    contracts: dict[str, ForecastContract] = {}
    contract_by_digest: dict[str, ForecastContract] = {}
    for index, raw in enumerate(document["contracts"]):
        try:
            contract = ForecastContract.from_dict(raw)
        except (ForecastContractError, TypeError, ValueError) as error:
            raise ForecastEvidenceError(f"contracts[{index}]: {error}") from error
        if contract.contract_id in contracts:
            raise ForecastEvidenceError(f"duplicate contract_id {contract.contract_id!r}")
        contracts[contract.contract_id] = contract
        contract_by_digest[contract.sha256] = contract

    revision_fields = {
        "revision_id",
        "claim_id",
        "ordinal",
        "supersedes",
        "contract_sha256",
        "prediction",
        "confidence",
        "rationale",
        "submitted_at",
        "status",
        "reason",
        "trigger_event_id",
        "revision_reason",
        "exposure",
        "idempotency_key",
    }
    exposure_fields = {
        "observed_at",
        "market_snapshot_sha256",
        "consensus_visible",
        "consensus_snapshot_sha256",
        "source_ids",
    }
    revision_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    claim_revisions: dict[str, list[dict[str, Any]]] = {}
    checked_claims: dict[str, Claim] = {}
    claim_contract_digests: dict[str, str] = {}
    for index, raw in enumerate(document["revisions"]):
        if not isinstance(raw, dict) or set(raw) != revision_fields:
            raise ForecastEvidenceError(f"revisions[{index}] fields do not match v1")
        revision_id = _nonempty(raw["revision_id"], f"revisions[{index}].revision_id")
        claim_id = _nonempty(raw["claim_id"], f"revisions[{index}].claim_id")
        ordinal = _clock(raw["ordinal"], f"revisions[{index}].ordinal")
        submitted_at = _clock(raw["submitted_at"], f"revisions[{index}].submitted_at")
        contract_digest = _digest(
            raw["contract_sha256"], f"revisions[{index}].contract_sha256"
        )
        contract = contract_by_digest.get(contract_digest)
        if contract is None:
            raise ForecastEvidenceError(f"revisions[{index}] names an unknown contract digest")
        prior_digest = claim_contract_digests.setdefault(claim_id, contract_digest)
        if prior_digest != contract_digest:
            raise ForecastEvidenceError(f"claim {claim_id!r} changes contract across revisions")
        if revision_id in revision_ids:
            raise ForecastEvidenceError(f"duplicate revision_id {revision_id!r}")
        revision_ids.add(revision_id)
        key = _nonempty(raw["idempotency_key"], f"revisions[{index}].idempotency_key")
        if key in idempotency_keys:
            raise ForecastEvidenceError(f"duplicate idempotency_key {key!r}")
        idempotency_keys.add(key)
        if raw["status"] not in {ELIGIBLE, LATE, REJECTED}:
            raise ForecastEvidenceError(f"revisions[{index}] has unknown status")
        if raw["status"] == ELIGIBLE and raw["reason"] is not None:
            raise ForecastEvidenceError("eligible revision cannot carry a rejection reason")
        if raw["status"] in {LATE, REJECTED} and (
            not isinstance(raw["reason"], str) or not raw["reason"].strip()
        ):
            raise ForecastEvidenceError("late and rejected revisions require a reason")
        if raw["trigger_event_id"] is not None:
            _nonempty(raw["trigger_event_id"], f"revisions[{index}].trigger_event_id")
        if not isinstance(raw["prediction"], list):
            raise ForecastEvidenceError(f"revisions[{index}].prediction must be an array")
        if not isinstance(raw["rationale"], str):
            raise ForecastEvidenceError(f"revisions[{index}].rationale must be a string")
        exposure = raw["exposure"]
        if not isinstance(exposure, dict) or set(exposure) != exposure_fields:
            raise ForecastEvidenceError(f"revisions[{index}].exposure fields do not match v1")
        sources = exposure["source_ids"]
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            raise ForecastEvidenceError(f"revisions[{index}].source_ids must be strings")
        checked_exposure = InformationExposure(
            observed_at=exposure["observed_at"],
            market_snapshot_sha256=exposure["market_snapshot_sha256"],
            consensus_visible=exposure["consensus_visible"],
            consensus_snapshot_sha256=exposure["consensus_snapshot_sha256"],
            source_ids=tuple(sources),
        )
        if checked_exposure.observed_at != submitted_at:
            raise ForecastEvidenceError(
                f"revisions[{index}].exposure.observed_at must equal submitted_at"
            )
        # Reuse Claim to validate prediction, confidence, and contract clock.  For a
        # late/rejected attempt clamp only the validation clock, never the recorded one.
        validation_at = min(max(submitted_at, contract.opens_at), contract.deadline)
        try:
            checked_claim = Claim(
                claim_id=claim_id,
                contract=contract,
                prediction=tuple(raw["prediction"]),
                committed_at=validation_at,
                confidence=raw["confidence"],
                rationale=raw["rationale"],
            )
        except (ClaimRejected, TypeError, ValueError) as error:
            raise ForecastEvidenceError(f"revisions[{index}]: {error}") from error
        checked_claims[revision_id] = checked_claim
        if raw["status"] == ELIGIBLE and not contract.opens_at <= submitted_at <= contract.deadline:
            raise ForecastEvidenceError("eligible revision lies outside the contract window")
        if raw["status"] == LATE and submitted_at <= contract.deadline:
            raise ForecastEvidenceError("late revision does not fall after the contract deadline")
        if raw["status"] == REJECTED and submitted_at >= contract.opens_at:
            raise ForecastEvidenceError("rejected pre-open revision is not before opens_at")
        if ordinal == 0 and raw["supersedes"] is not None:
            raise ForecastEvidenceError("the first revision cannot supersede another revision")
        prior_revisions = claim_revisions.setdefault(claim_id, [])
        if ordinal != len(prior_revisions):
            raise ForecastEvidenceError(
                f"claim {claim_id!r} revisions are not in append order"
            )
        prior_revisions.append(raw)

    for claim_id, revisions in claim_revisions.items():
        revisions.sort(key=lambda raw: raw["ordinal"])
        if [raw["ordinal"] for raw in revisions] != list(range(len(revisions))):
            raise ForecastEvidenceError(f"claim {claim_id!r} revision ordinals are not contiguous")
        for ordinal, raw in enumerate(revisions):
            expected = None if ordinal == 0 else revisions[ordinal - 1]["revision_id"]
            if raw["supersedes"] != expected:
                raise ForecastEvidenceError(
                    f"claim {claim_id!r} revision {ordinal} has the wrong supersedes link"
                )
            if ordinal == 0 and raw["revision_reason"] is not None:
                raise ForecastEvidenceError("initial forecast cannot have revision_reason")
            if ordinal > 0 and (
                not isinstance(raw["revision_reason"], str)
                or not raw["revision_reason"].strip()
            ):
                raise ForecastEvidenceError("a revision requires revision_reason")

    resolution_fields = {
        "claim_id",
        "status",
        "outcome",
        "available_at",
        "recorded_at",
        "reason",
    }
    resolution_claims: set[str] = set()
    resolution_records: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(document["resolutions"]):
        if not isinstance(raw, dict) or set(raw) != resolution_fields:
            raise ForecastEvidenceError(f"resolutions[{index}] fields do not match v1")
        claim_id = _nonempty(raw["claim_id"], f"resolutions[{index}].claim_id")
        if claim_id not in claim_revisions:
            raise ForecastEvidenceError(f"resolution names unknown claim {claim_id!r}")
        if claim_id in resolution_claims:
            raise ForecastEvidenceError(f"duplicate resolution for claim {claim_id!r}")
        resolution_claims.add(claim_id)
        resolution_records[claim_id] = raw
        _clock(raw["recorded_at"], f"resolutions[{index}].recorded_at")
        if raw["status"] not in {PENDING, RESOLVED, CANCELLED, REJECTED}:
            raise ForecastEvidenceError(f"resolutions[{index}] has unknown status")
        if raw["status"] == RESOLVED:
            _clock(raw["available_at"], f"resolutions[{index}].available_at")
            if raw["outcome"] is None or raw["reason"] is not None:
                raise ForecastEvidenceError("resolved records need an outcome and no reason")
        elif raw["outcome"] is not None:
            raise ForecastEvidenceError("only resolved records may carry an outcome")
        if raw["status"] in {PENDING, CANCELLED} and raw["available_at"] is not None:
            raise ForecastEvidenceError(
                "pending and cancelled records cannot carry available_at"
            )
        if raw["status"] == PENDING and raw["reason"] is not None:
            raise ForecastEvidenceError("pending records cannot carry a reason")
        if raw["status"] in {CANCELLED, REJECTED} and (
            not isinstance(raw["reason"], str) or not raw["reason"].strip()
        ):
            raise ForecastEvidenceError("cancelled and rejected records require a reason")
    if resolution_claims != set(claim_revisions):
        raise ForecastEvidenceError("resolutions must cover every claim exactly once")
    for claim_id, revisions in claim_revisions.items():
        eligible = [raw for raw in revisions if raw["status"] == ELIGIBLE]
        resolution = resolution_records[claim_id]
        if not eligible:
            if resolution["status"] not in {REJECTED, CANCELLED}:
                raise ForecastEvidenceError(
                    f"claim {claim_id!r} has no eligible revision but is not rejected or cancelled"
                )
            continue
        latest = max(eligible, key=lambda raw: raw["ordinal"])
        claim = checked_claims[latest["revision_id"]]
        if resolution["status"] == RESOLVED:
            available_at = resolution["available_at"]
            if available_at <= claim.committed_at or available_at < claim.contract.resolves_at:
                raise ForecastEvidenceError(
                    f"claim {claim_id!r} resolved before its frozen information boundary"
                )
            try:
                score_claim(claim, resolution["outcome"])
            except (ClaimRejected, TypeError, ValueError) as error:
                raise ForecastEvidenceError(
                    f"claim {claim_id!r} has an invalid resolved outcome: {error}"
                ) from error
    return document


__all__ = [
    "ELIGIBLE",
    "LATE",
    "REJECTED",
    "PENDING",
    "RESOLVED",
    "CANCELLED",
    "ForecastEvidenceError",
    "ForecastRunIdentity",
    "InformationExposure",
    "ForecastRevision",
    "ForecastResolution",
    "ForecastEvidence",
    "ForecastLedger",
    "forecast_evidence_from_json",
    "write_forecast_evidence",
]
