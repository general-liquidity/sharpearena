"""``sharpebench/canonical-json/v1`` conformance, pinned to SharpeBench's bytes (R07).

Every expected string and digest below was printed by the Rust implementation
in ``sharpebench-protocol`` (``canonical_number``, ``canonical_json`` and
``versioned_preimage`` at Bench ``955d7f8``), not by this module, so the test
compares the two languages rather than the module against itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from sharpearena.canonical_json import (
    CANONICAL_JSON_VERSION,
    CanonicalJsonError,
    canonical_json_v1,
    canonical_number,
    canonical_sha256_v1,
    versioned_preimage,
)
from sharpearena.deferred import ClaimRejected, DeferredDesk, Outcome, claims_from_json
from sharpearena.forecast_contract import (
    PROBABILITY,
    ForecastContract,
    canonical_json,
    legacy_canonical_sha256,
)
from sharpearena.forecast_evidence import (
    ForecastEvidenceError,
    ForecastLedger,
    ForecastRunIdentity,
    InformationExposure,
    forecast_evidence_from_json,
)
from sharpearena.prospective_field import verify_field_settlement


REPOSITORY = Path(__file__).resolve().parents[3]

# (value, canonical text) as printed by Bench's canonical_number.
NUMBER_VECTORS = (
    # The R07 case and the exponent window's lower edge.
    (1e-5, "0.00001"),
    (1e-6, "0.000001"),
    (1e-7, "1e-7"),
    (1.5e-7, "1.5e-7"),
    (1e-8, "1e-8"),
    (0.000001234, "0.000001234"),
    # The window's upper edge.
    (1e16, "10000000000000000"),
    (1e20, "100000000000000000000"),
    (123456789012345680000.0, "123456789012345680000"),
    (1e21, "1e+21"),
    (1.5e21, "1.5e+21"),
    (1234567890123456800000.0, "1.2345678901234568e+21"),
    (123456.789, "123456.789"),
    # Extremes keep the shortest round-tripping digits.
    (5e-324, "5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (2.2250738585072014e-308, "2.2250738585072014e-308"),
    (1.0000000000000002, "1.0000000000000002"),
    (9007199254740993.0, "9007199254740992"),
    # Signed zero and integer-valued floats.
    (0.0, "0"),
    (-0.0, "0"),
    (1.0, "1"),
    (-3.0, "-3"),
    (100.0, "100"),
    # One leading sign.
    (-1e-5, "-0.00001"),
    (-1e21, "-1e+21"),
    (-0.5, "-0.5"),
    (0.1, "0.1"),
    (0.82, "0.82"),
)


@pytest.mark.parametrize(("value", "expected"), NUMBER_VECTORS)
def test_numbers_render_as_bench_renders_them(value, expected):
    assert canonical_number(value) == expected
    assert canonical_json_v1(value) == expected


def test_python_json_dumps_is_not_v1():
    # The defect: the same members, a different number text, a different digest.
    assert json.dumps(1e-5) == "1e-05"
    assert json.dumps(1e-7) == "1e-07"
    assert json.dumps(1e16) == "1e+16"
    assert json.dumps([0.0, 1.0, -0.0]) == "[0.0, 1.0, -0.0]"
    assert canonical_json_v1([1e-5, 1e-7, 1e16, 0.0, 1.0, -0.0]) == (
        "[0.00001,1e-7,10000000000000000,0,1,0]"
    )


def test_non_finite_numbers_have_no_canonical_form():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(CanonicalJsonError):
            canonical_number(value)
        with pytest.raises(CanonicalJsonError):
            canonical_json_v1({"x": value})


def _contract_document(neutral_threshold: float) -> dict:
    """The one-contract fixture Bench's R07 regression hashes."""

    return {
        "schema_version": "sharpearena.forecast-contract.v1",
        "contract_id": "c0",
        "question": "question 0",
        "instrument": "ES",
        "target": "close_up",
        "kind": "probability",
        "opens_at": 0,
        "deadline": 1,
        "resolves_at": 10,
        "observation_source": "fixture:v1",
        "open_definition": "close at opens_at",
        "close_definition": "close at resolves_at",
        "unit": "binary",
        "scoring_rule": "binary_brier",
        "neutral_threshold": neutral_threshold,
        "boundary_ownership": "threshold is false",
        "missing_data_policy": "cancel",
        "fallback_policy": "cancel",
        "categories": [],
        "interval_alpha": None,
    }


# (document, canonical text, sha256 of the versioned pre-image) from Bench.
DOCUMENT_VECTORS = (
    (
        _contract_document(1e-5),
        '{"boundary_ownership":"threshold is false","categories":[],"close_definition":'
        '"close at resolves_at","contract_id":"c0","deadline":1,"fallback_policy":"cancel",'
        '"instrument":"ES","interval_alpha":null,"kind":"probability","missing_data_policy":'
        '"cancel","neutral_threshold":0.00001,"observation_source":"fixture:v1",'
        '"open_definition":"close at opens_at","opens_at":0,"question":"question 0",'
        '"resolves_at":10,"schema_version":"sharpearena.forecast-contract.v1",'
        '"scoring_rule":"binary_brier","target":"close_up","unit":"binary"}',
        "60f5c087ed7ad90550b9f556081760675355a6dc7f3a51acab15e483b7df5499",
    ),
    (
        _contract_document(0.0),
        '{"boundary_ownership":"threshold is false","categories":[],"close_definition":'
        '"close at resolves_at","contract_id":"c0","deadline":1,"fallback_policy":"cancel",'
        '"instrument":"ES","interval_alpha":null,"kind":"probability","missing_data_policy":'
        '"cancel","neutral_threshold":0,"observation_source":"fixture:v1",'
        '"open_definition":"close at opens_at","opens_at":0,"question":"question 0",'
        '"resolves_at":10,"schema_version":"sharpearena.forecast-contract.v1",'
        '"scoring_rule":"binary_brier","target":"close_up","unit":"binary"}',
        "213944676b3e82e92c297db2fbe62e4f33ec5b721461c4c2dc549a5825a3d30a",
    ),
    (
        _contract_document(0.001),
        '{"boundary_ownership":"threshold is false","categories":[],"close_definition":'
        '"close at resolves_at","contract_id":"c0","deadline":1,"fallback_policy":"cancel",'
        '"instrument":"ES","interval_alpha":null,"kind":"probability","missing_data_policy":'
        '"cancel","neutral_threshold":0.001,"observation_source":"fixture:v1",'
        '"open_definition":"close at opens_at","opens_at":0,"question":"question 0",'
        '"resolves_at":10,"schema_version":"sharpearena.forecast-contract.v1",'
        '"scoring_rule":"binary_brier","target":"close_up","unit":"binary"}',
        "ef113d70d384c7af18b86d5fc5cbb903b8c6e5d8933fa36a6860ae42c6be2226",
    ),
    (
        # RFC 8785 escaping: short escapes, \u00xx for the other controls, and
        # U+007F, the solidus, non-ASCII and the two line separators literal.
        {
            "é😀": 'a"b\\c/d\x00\x1f\x7f  ',
            "\b\t\n\f\r": "é😀",
        },
        '{"\\b\\t\\n\\f\\r":"é😀","é😀":"a\\"b\\\\c/d\\u0000\\u001f\x7f  "}',
        "0f07d0d120e5258e24fc7d2da97e06808173a920f1f41422dc52131fe2bcea5c",
    ),
    (
        # Members by code point: the astral key sorts after U+FF3A, which is
        # the deliberate divergence from RFC 8785's UTF-16 order.
        {"\U0001f600": 1, "Ｚ": 2, "b": 1, "a": 2, "A": 3},
        '{"A":3,"a":2,"b":1,"Ｚ":2,"\U0001f600":1}',
        "f94fdf0502f955a4a0519454124d224eb5f30611db77df4216b666518661305e",
    ),
    (
        {
            "a": 0.0,
            "b": -0.0,
            "n": 1.0,
            "z": [3, 1, 2],
            "o": {"n": None, "t": True, "f": False},
            "x": 1e-5,
            "y": 1e21,
        },
        '{"a":0,"b":0,"n":1,"o":{"f":false,"n":null,"t":true},"x":0.00001,"y":1e+21,"z":[3,1,2]}',
        "c05e89fc2af55575ca8692fec09db4c140519dca39a823490f8297cab40d5eab",
    ),
    ("plain", '"plain"', "8b2f9564be4c8ff0c6b89679e2dc1d9735995e0f48f5ca5e6e621754e41f31ff"),
    (1e-5, "0.00001", "96a019048e1180021d021f352f7995fa2d12e28180ea42ef011262f438a33bc2"),
)


@pytest.mark.parametrize(("document", "text", "digest"), DOCUMENT_VECTORS)
def test_documents_canonicalize_and_digest_as_bench_does(document, text, digest):
    assert canonical_json_v1(document) == text
    assert canonical_sha256_v1(document) == digest


def test_the_preimage_is_the_versioned_length_framed_body():
    document = {"neutral_threshold": 1e-5}
    body = b'{"neutral_threshold":0.00001}'
    preimage = versioned_preimage(document)
    assert preimage == (
        b"sharpebench/canonical-json/v1" + b"\x00" + len(body).to_bytes(8, "big") + body
    )
    assert CANONICAL_JSON_VERSION == "sharpebench/canonical-json/v1"
    # The frame is injective across body boundaries: a document that contains
    # the frame's own punctuation cannot imitate a different one.
    left = versioned_preimage({"a": "b\x00", "c": "d"})
    right = versioned_preimage({"a": "b", "c\x00": "d"})
    assert left != right
    short = versioned_preimage("ab")
    assert not versioned_preimage("abc").startswith(short)


def test_unsupported_values_are_refused():
    with pytest.raises(CanonicalJsonError):
        canonical_json_v1({1: "non-string key"})
    with pytest.raises(CanonicalJsonError):
        canonical_json_v1({"x": object()})


def _contract(neutral_threshold: float) -> ForecastContract:
    return ForecastContract.from_dict(_contract_document(neutral_threshold))


def test_r07_contract_digests_to_the_value_bench_computes():
    # Bench's `r07_small_threshold_contract_verifies_under_canonical_json_v1`
    # accepts exactly this digest for exactly this contract.
    contract = _contract(1e-5)
    assert canonical_json_v1(contract.to_dict()).count('"neutral_threshold":0.00001,') == 1
    assert contract.sha256 == "60f5c087ed7ad90550b9f556081760675355a6dc7f3a51acab15e483b7df5499"
    assert canonical_json(contract.to_dict()).count('"neutral_threshold":1e-05,') == 1
    assert contract.legacy_sha256 == hashlib.sha256(
        canonical_json(contract.to_dict()).encode("utf-8")
    ).hexdigest()


def test_legacy_and_v1_digests_of_one_contract_differ():
    for threshold in (0.0, 0.001, 1e-5):
        contract = _contract(threshold)
        assert contract.sha256 != contract.legacy_sha256
        assert contract.digests == (contract.sha256, contract.legacy_sha256)
        assert contract.legacy_sha256 == legacy_canonical_sha256(contract.to_dict())


def test_the_tutorial_fixture_pins_the_legacy_digest_bench_labels_legacy():
    # Bench's committed `report.json` at 955d7f8 lists this digest for the first
    # tutorial contract under `contract_digest_versions` as `legacy`.
    contract = ForecastContract(
        contract_id="tutorial-binary-01",
        question="Will synthetic instrument 01 close above its frozen reference?",
        instrument="SYNTH-01",
        target="close_above_frozen_reference",
        kind=PROBABILITY,
        opens_at=10,
        deadline=20,
        resolves_at=30,
        observation_source="tutorial:synthetic-market-v1",
        open_definition="value in the frozen observation at logical clock 10",
        close_definition="value in the frozen resolution at logical clock 30",
        unit="binary",
        scoring_rule="binary_brier",
        boundary_ownership="an equal close resolves false",
        missing_data_policy="cancel the contract",
        fallback_policy="no fallback source",
    )
    assert (
        contract.legacy_sha256
        == "9dfb54637b823966590993de454372d759943ab20e09f5552e32b11192318a42"
    )
    assert contract.sha256 != contract.legacy_sha256


def _identity() -> ForecastRunIdentity:
    return ForecastRunIdentity(
        agent_id="agent",
        model_id="model",
        model_sha256="a" * 64,
        scaffold_id="scaffold",
        scaffold_sha256="b" * 64,
        prompt_sha256="c" * 64,
        operator_id="operator",
        config_sha256="d" * 64,
    )


def _evidence_document() -> dict:
    ledger = ForecastLedger(_identity())
    ledger.submit(
        claim_id="claim-0",
        contract=_contract(0.0),
        prediction=0.7,
        confidence=0.7,
        rationale="evidence",
        submitted_at=0,
        idempotency_key="request-0",
        exposure=InformationExposure(observed_at=0, market_snapshot_sha256="e" * 64),
    )
    evidence = ledger.evidence([Outcome("claim-0", 1.0, available_at=10)], generated_at=1000)
    return json.loads(evidence.to_json())


def test_a_producer_writes_v1_and_a_reader_accepts_either_encoding():
    document = _evidence_document()
    contract = _contract(0.0)
    assert document["revisions"][0]["contract_sha256"] == contract.sha256

    legacy = json.loads(json.dumps(document))
    legacy["revisions"][0]["contract_sha256"] = contract.legacy_sha256
    assert forecast_evidence_from_json(json.dumps(legacy))["revisions"][0][
        "contract_sha256"
    ] == contract.legacy_sha256

    # The Python-convention digest of a 1e-5 contract is neither encoding:
    # unframed text with `1e-05`, which is the R07 input, stays refused.
    neither = json.loads(json.dumps(document))
    neither["revisions"][0]["contract_sha256"] = hashlib.sha256(
        canonical_json(contract.to_dict()).replace('"neutral_threshold":0.0,', '"neutral_threshold":0,').encode("utf-8")
    ).hexdigest()
    with pytest.raises(ForecastEvidenceError, match="unknown contract digest"):
        forecast_evidence_from_json(json.dumps(neither))


def test_claims_documents_accept_either_encoding_and_refuse_others():
    desk = DeferredDesk()
    desk.tick(0)
    claim = desk.commit_contract(_contract(0.0), 0.7, claim_id="prob-1")
    assert claim.to_dict()["contract_sha256"] == claim.contract.sha256
    document = json.loads(desk.to_json())
    document["claims"][0]["contract_sha256"] = claim.contract.legacy_sha256
    restored = claims_from_json(json.dumps(document))
    assert restored[0].contract == claim.contract
    document["claims"][0]["contract_sha256"] = "0" * 64
    with pytest.raises(ClaimRejected, match="contract_sha256"):
        claims_from_json(json.dumps(document))


def test_the_frozen_prospective_field_still_verifies_under_its_legacy_digests():
    field = REPOSITORY / "paper" / "evidence" / "prospective-forecast-field"
    plan = json.loads((field / "field-plan.json").read_text(encoding="utf-8"))
    for raw in plan["contracts"]:
        contract = ForecastContract.from_dict(
            {key: value for key, value in raw.items() if key not in {"sha256", "target_open_ms"}}
        )
        assert raw["sha256"] == contract.legacy_sha256
        assert raw["sha256"] != contract.sha256
    settlement = verify_field_settlement(field)
    assert len(settlement["contracts"]) == 24
    # The settlement record binds the digest the field was frozen under, so the
    # migration does not move the published field's settlement digest.
    assert (
        settlement["settlement_sha256"]
        == "10792f64bed05e58a242834fbcb8f03200b1f8d6ba78f8248e7d0a1a36f1b8d7"
    )
