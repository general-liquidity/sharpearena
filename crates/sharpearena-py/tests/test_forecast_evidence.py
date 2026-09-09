"""Prospective forecast evidence is append-only, closed, and provider-neutral."""

from __future__ import annotations

import json

import pytest

from sharpearena.deferred import Claim, Outcome, PROBABILITY, score_claim
from sharpearena.forecast_contract import BINARY_BRIER, ForecastContract
from sharpearena.forecast_evidence import (
    CANCELLED,
    ELIGIBLE,
    LATE,
    PENDING,
    REJECTED,
    RESOLVED,
    ForecastEvidenceError,
    ForecastLedger,
    ForecastRunIdentity,
    InformationExposure,
    forecast_evidence_from_json,
    write_forecast_evidence,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def identity() -> ForecastRunIdentity:
    return ForecastRunIdentity(
        agent_id="agent-a",
        model_id="model-a",
        model_sha256=SHA_A,
        scaffold_id="scaffold-a",
        scaffold_sha256=SHA_B,
        prompt_sha256=SHA_C,
        operator_id="operator-a",
        config_sha256=SHA_D,
    )


def contract(contract_id: str = "es-up") -> ForecastContract:
    return ForecastContract(
        contract_id=contract_id,
        question="Will ES finish above the frozen threshold?",
        instrument="ES",
        target="ES.close_above_threshold",
        kind=PROBABILITY,
        opens_at=10,
        deadline=20,
        resolves_at=30,
        observation_source="fixture:es-daily-v1",
        open_definition="official close at bar 10",
        close_definition="official close at bar 30",
        unit="binary",
        scoring_rule=BINARY_BRIER,
        neutral_threshold=0.001,
        boundary_ownership="threshold is false",
        missing_data_policy="cancel",
        fallback_policy="retry once, then cancel",
    )


def submit_first(ledger: ForecastLedger, **overrides):
    values = {
        "claim_id": "claim-es",
        "contract": contract(),
        "prediction": 0.6,
        "confidence": 0.6,
        "rationale": "observable evidence",
        "submitted_at": 12,
        "idempotency_key": "request-1",
    }
    values.update(overrides)
    return ledger.submit(**values)


def test_first_submission_and_revision_form_an_append_only_chain():
    ledger = ForecastLedger(identity())
    first = submit_first(ledger)
    second = ledger.submit(
        claim_id="claim-es",
        contract=contract(),
        prediction=0.75,
        confidence=0.75,
        rationale="new public release",
        submitted_at=18,
        idempotency_key="request-2",
        expected_revision=0,
        trigger_event_id="event-cpi",
        revision_reason="CPI changed the posterior",
    )
    assert first.status == second.status == ELIGIBLE
    assert second.ordinal == 1
    assert second.supersedes == first.revision_id
    assert [record.prediction for record in ledger.revisions()] == [(0.6,), (0.75,)]
    assert ledger.effective_claims()[0].prediction == (0.75,)


def test_late_revision_is_retained_but_cannot_replace_the_eligible_forecast():
    ledger = ForecastLedger(identity())
    first = submit_first(ledger)
    late = ledger.submit(
        claim_id="claim-es",
        contract=contract(),
        prediction=0.99,
        confidence=0.99,
        rationale="answer now visible",
        submitted_at=31,
        idempotency_key="request-late",
        expected_revision=0,
        revision_reason="late attempt",
    )
    assert late.status == LATE
    assert "deadline" in (late.reason or "")
    assert ledger.effective_claims()[0].prediction == first.prediction


def test_pre_open_attempt_is_rejected_and_has_no_effective_claim():
    ledger = ForecastLedger(identity())
    attempt = submit_first(ledger, submitted_at=9)
    assert attempt.status == REJECTED
    assert ledger.effective_claims() == ()
    evidence = ledger.evidence([], generated_at=31)
    assert evidence.resolutions[0].status == REJECTED


def test_idempotent_retry_returns_the_same_record_and_payload_reuse_is_refused():
    ledger = ForecastLedger(identity())
    first = submit_first(ledger)
    assert submit_first(ledger) is first
    with pytest.raises(ForecastEvidenceError, match="different content"):
        submit_first(ledger, prediction=0.7)
    assert len(ledger.revisions()) == 1


def test_expected_revision_refuses_a_lost_update():
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    with pytest.raises(ForecastEvidenceError, match="stale revision"):
        ledger.submit(
            claim_id="claim-es",
            contract=contract(),
            prediction=0.7,
            confidence=0.7,
            rationale="new",
            submitted_at=15,
            idempotency_key="request-2",
            expected_revision=4,
            revision_reason="new evidence",
        )


def test_information_exposure_requires_a_hash_when_consensus_was_visible():
    with pytest.raises(ForecastEvidenceError, match="requires consensus_snapshot"):
        InformationExposure(observed_at=12, consensus_visible=True)
    blind = InformationExposure(observed_at=12, market_snapshot_sha256=SHA_A)
    assert blind.consensus_snapshot_sha256 is None
    exposed = InformationExposure(
        observed_at=12,
        market_snapshot_sha256=SHA_A,
        consensus_visible=True,
        consensus_snapshot_sha256=SHA_B,
        source_ids=("market", "peer-consensus"),
    )
    assert exposed.consensus_visible is True


def test_resolved_pending_and_cancelled_states_are_explicit():
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    submit_first(
        ledger,
        claim_id="claim-cancel",
        contract=contract("cancelled-contract"),
        idempotency_key="request-cancel",
    )
    submit_first(
        ledger,
        claim_id="claim-pending",
        contract=contract("pending-contract"),
        idempotency_key="request-pending",
    )
    ledger.cancel("claim-cancel", recorded_at=25, reason="source unavailable")
    evidence = ledger.evidence(
        [Outcome("claim-es", 1.0, available_at=30)], generated_at=31
    )
    states = {record.claim_id: record.status for record in evidence.resolutions}
    assert states == {
        "claim-es": RESOLVED,
        "claim-cancel": CANCELLED,
        "claim-pending": PENDING,
    }


def test_export_contains_raw_outcome_but_no_producer_score():
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    evidence = ledger.evidence(
        [Outcome("claim-es", 1.0, available_at=30)], generated_at=31
    )
    document = forecast_evidence_from_json(evidence.to_json())
    assert document["schema_version"] == "sharpe.forecast-evidence.v2"
    assert document["resolutions"][0]["outcome"] == 1.0
    assert "score" not in evidence.to_json()


def _v2_document() -> dict:
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    submit_first(
        ledger,
        prediction=0.7,
        confidence=0.7,
        submitted_at=15,
        idempotency_key="request-2",
        expected_revision=0,
        revision_reason="public update",
    )
    evidence = ledger.evidence([Outcome("claim-es", 1.0, available_at=30)], generated_at=31)
    return json.loads(evidence.to_json())


def _as_v1(document: dict) -> dict:
    v1 = json.loads(json.dumps(document))
    v1["schema_version"] = "sharpe.forecast-evidence.v1"
    for revision in v1["revisions"]:
        del revision["contract_digest_encoding"]
    return v1


def test_v2_round_trip_declares_the_v1_encoding_on_every_revision(tmp_path):
    document = _v2_document()
    assert document["schema_version"] == "sharpe.forecast-evidence.v2"
    assert len(document["revisions"]) == 2
    for revision in document["revisions"]:
        assert revision["contract_sha256"] == contract().sha256
        assert revision["contract_digest_encoding"] == "sharpebench/canonical-json/v1"
    assert set(document["contracts"][0]) == set(contract().to_dict())
    assert document["contracts"][0]["schema_version"] == "sharpearena.forecast-contract.v1"

    ledger = ForecastLedger(identity())
    submit_first(ledger)
    destination = tmp_path / "evidence.json"
    write_forecast_evidence(
        destination, ledger.evidence([Outcome("claim-es", 1.0, available_at=30)], generated_at=31)
    )
    stored = forecast_evidence_from_json(destination.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "sharpe.forecast-evidence.v2"
    assert stored["revisions"][0]["contract_digest_encoding"] == "sharpebench/canonical-json/v1"


def test_v1_document_without_the_declaration_still_parses():
    v1 = _as_v1(_v2_document())
    parsed = forecast_evidence_from_json(json.dumps(v1))
    assert parsed["schema_version"] == "sharpe.forecast-evidence.v1"
    assert all("contract_digest_encoding" not in revision for revision in parsed["revisions"])

    legacy = json.loads(json.dumps(v1))
    legacy["revisions"][0]["contract_sha256"] = contract().legacy_sha256
    legacy["revisions"][1]["contract_sha256"] = contract().legacy_sha256
    assert forecast_evidence_from_json(json.dumps(legacy))["revisions"][0][
        "contract_sha256"
    ] == contract().legacy_sha256


def test_v1_document_carrying_the_declaration_is_refused():
    v1 = _as_v1(_v2_document())
    v1["revisions"][1]["contract_digest_encoding"] = "sharpebench/canonical-json/v1"
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(v1))
    assert str(error.value) == (
        "contract_digest_encoding is not a field of sharpe.forecast-evidence.v1; "
        "a document that declares it must be sharpe.forecast-evidence.v2"
    )


def test_v2_document_missing_the_declaration_on_any_revision_is_refused():
    document = _v2_document()
    del document["revisions"][1]["contract_digest_encoding"]
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(document))
    assert str(error.value) == (
        "sharpe.forecast-evidence.v2 requires contract_digest_encoding on every revision"
    )


@pytest.mark.parametrize("label", ["sharpebench/canonical-json/v2", "", None, 1, "Legacy"])
def test_v2_document_with_an_unknown_encoding_label_is_refused(label):
    document = _v2_document()
    document["revisions"][0]["contract_digest_encoding"] = label
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(document))
    assert str(error.value) == (
        f"unknown contract_digest_encoding {json.dumps(label)}; expected "
        "sharpebench/canonical-json/v1 or legacy"
    )


def test_v2_declaration_inconsistent_with_the_digest_is_refused_naming_both():
    v1_digest = contract().sha256
    legacy_digest = contract().legacy_sha256
    assert v1_digest != legacy_digest

    # A v1 digest labelled legacy.
    document = _v2_document()
    document["revisions"][1]["contract_digest_encoding"] = "legacy"
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(document))
    assert str(error.value) == (
        f"revision claim-es:r1 declares contract digest {v1_digest} under legacy, "
        "but it recomputes under sharpebench/canonical-json/v1"
    )

    # A legacy digest labelled v1: a v1 document would accept it by inference.
    document = _v2_document()
    for revision in document["revisions"]:
        revision["contract_sha256"] = legacy_digest
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(document))
    assert str(error.value) == (
        f"revision claim-es:r0 declares contract digest {legacy_digest} under "
        "sharpebench/canonical-json/v1, but it recomputes under legacy"
    )

    # The same legacy digests correctly declared are accepted.
    for revision in document["revisions"]:
        revision["contract_digest_encoding"] = "legacy"
    assert forecast_evidence_from_json(json.dumps(document))["revisions"][1][
        "contract_digest_encoding"
    ] == "legacy"

    # A digest matching no contract names neither encoding.
    document["revisions"][0]["contract_sha256"] = "f" * 64
    with pytest.raises(ForecastEvidenceError) as error:
        forecast_evidence_from_json(json.dumps(document))
    assert str(error.value) == (
        f"revision claim-es:r0 declares contract digest {'f' * 64} under legacy, "
        "but it recomputes under neither sharpebench/canonical-json/v1 nor the "
        "legacy encoding of any contract"
    )


def test_file_export_is_atomic_valid_and_content_addressed(tmp_path):
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    evidence = ledger.evidence(
        [Outcome("claim-es", 1.0, available_at=30)], generated_at=31
    )
    destination = tmp_path / "nested" / "forecast-evidence.json"

    digest = write_forecast_evidence(destination, evidence)

    payload = destination.read_text(encoding="utf-8")
    assert forecast_evidence_from_json(payload)["resolutions"][0]["status"] == RESOLVED
    assert payload.endswith("\n")
    assert digest == __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()
    assert list(destination.parent.glob("*.partial")) == []


def test_closed_export_rejects_unknown_fields_broken_links_and_nonfinite_json():
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    document = ledger.evidence([], generated_at=31).to_dict()

    unknown = json.loads(json.dumps(document))
    unknown["revisions"][0]["surprise"] = True
    with pytest.raises(ForecastEvidenceError, match="fields do not match"):
        forecast_evidence_from_json(json.dumps(unknown))

    broken = json.loads(json.dumps(document))
    broken["revisions"][0]["contract_sha256"] = "f" * 64
    with pytest.raises(ForecastEvidenceError, match="recomputes under neither"):
        forecast_evidence_from_json(json.dumps(broken))
    broken["schema_version"] = "sharpe.forecast-evidence.v1"
    del broken["revisions"][0]["contract_digest_encoding"]
    with pytest.raises(ForecastEvidenceError, match="unknown contract digest"):
        forecast_evidence_from_json(json.dumps(broken))

    with pytest.raises(ForecastEvidenceError, match="strict JSON"):
        forecast_evidence_from_json('{"score": NaN}')


def test_contract_id_cannot_be_reused_with_different_settlement_bytes():
    ledger = ForecastLedger(identity())
    submit_first(ledger)
    changed = contract()
    changed = ForecastContract(**{**changed.__dict__, "close_definition": "different close"})
    with pytest.raises(ForecastEvidenceError, match="different bytes"):
        ledger.submit(
            claim_id="other",
            contract=changed,
            prediction=0.5,
            confidence=0.5,
            rationale="x",
            submitted_at=12,
            idempotency_key="other-request",
        )


def test_fixed_point_brier_model_matches_the_executable_float_rule():
    scale = 1_000
    for probability in (0, 100, 500, 900, 1_000):
        for truth in (0, 200, 500, 800, 1_000):
            checked = Claim(
                claim_id="brier-conformance",
                contract=contract(),
                prediction=(probability / scale,),
                committed_at=12,
                confidence=0.5,
                rationale="formal model conformance",
            )
            loss_zero = score_claim(checked, 0.0)["brier"]
            loss_one = score_claim(checked, 1.0)["brier"]
            expected = (truth / scale) * loss_one + (1.0 - truth / scale) * loss_zero
            numerator = truth * (scale - probability) ** 2 + (
                scale - truth
            ) * probability**2
            decomposed = truth * (scale - truth) * scale + scale * (
                probability - truth
            ) ** 2
            assert numerator == decomposed
            assert expected == pytest.approx(numerator / scale**3, abs=1e-15)
