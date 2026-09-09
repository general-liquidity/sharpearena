"""Prospective field phases preserve ordering, identity, and settlement boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from sharpearena import prospective_field
from sharpearena.forecast_contract import ForecastContract
from sharpearena.forecast_evidence import forecast_evidence_from_json
from sharpearena.prospective_field import (
    LocalModelSpec,
    ProspectiveFieldError,
    _ledger_from_pending,
    _target_candle,
    forecast_agent,
    parse_model_spec,
    parse_prediction_map,
    prepare_field,
    resolve_field,
    seal_forecasts,
    snapshot_digest,
    verify_field_settlement,
)


def test_prediction_parser_requires_exact_complete_probability_support():
    ids = ["a", "b"]
    assert parse_prediction_map('{"forecasts":{"a":0.2,"b":0.8}}', ids) == {
        "a": 0.2,
        "b": 0.8,
    }
    with pytest.raises(ProspectiveFieldError, match="complete forecasts"):
        parse_prediction_map('{"forecasts":{"a":0.2}}', ids)
    with pytest.raises(ProspectiveFieldError, match="complete forecasts"):
        parse_prediction_map('{"forecasts":{"a":0.2,"b":1.2}}', ids)
    with pytest.raises(ProspectiveFieldError, match="complete forecasts"):
        parse_prediction_map('{"forecasts":{"a":0.2,"b":0.8},"note":"x"}', ids)
    with pytest.raises(ProspectiveFieldError, match="exactly one"):
        parse_prediction_map(
            '{"forecasts":{"a":0.2,"b":0.8}} ' '{"forecasts":{"a":0.3,"b":0.7}}',
            ids,
        )


def test_snapshot_digest_binds_paths_lengths_and_bytes(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    nested = tmp_path / "weights"
    nested.mkdir()
    weight = nested / "part.bin"
    weight.write_bytes(b"abc")
    first = snapshot_digest(tmp_path)
    weight.write_bytes(b"abd")
    second = snapshot_digest(tmp_path)
    assert first[1:] == second[1:] == (2, 5)
    assert first[0] != second[0]


def test_model_spec_requires_a_content_revision_and_matching_snapshot(tmp_path):
    with pytest.raises(ProspectiveFieldError, match="content revision"):
        parse_model_spec("agent,owner/model,main,/model")
    revision = "a" * 40
    snapshot = tmp_path / revision
    assert parse_model_spec(
        f"agent,owner/model,{revision},{snapshot}"
    ) == LocalModelSpec("agent", "owner/model", revision, snapshot)
    with pytest.raises(ProspectiveFieldError, match="agent ID"):
        parse_model_spec(f"../agent,owner/model,{revision},{snapshot}")


def test_target_candle_uses_strict_close_above_open_and_exact_clock():
    def fetch(path, params):
        assert path == "/api/v3/klines"
        open_time = params["startTime"]
        return [
            [
                open_time,
                "10.0",
                "11.0",
                "9.0",
                "10.0",
                "2.0",
                open_time + 59_999,
                "20.0",
                4,
                "1.0",
                "10.0",
                "0",
            ]
        ]

    assert _target_candle("BTCUSDT", 60_000, fetch=fetch)["outcome"] == 0.0


def test_pending_reconstruction_preserves_every_revision(tmp_path):
    fixture = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "forecast-quality"
        / "fixtures"
        / "agent-alpha.json"
    )
    document = json.loads(fixture.read_text(encoding="utf-8"))
    ledger = _ledger_from_pending(document)
    assert [item.to_dict() for item in ledger.revisions()] == document["revisions"]


def test_field_phases_refuse_early_resolution_and_publish_complete_evidence(
    tmp_path, monkeypatch
):
    revision_a = "a" * 40
    revision_b = "b" * 40
    model_a = tmp_path / revision_a
    model_b = tmp_path / revision_b
    model_a.mkdir()
    model_b.mkdir()
    (model_a / "weights.bin").write_bytes(b"a")
    (model_b / "weights.bin").write_bytes(b"b")
    specs = [
        LocalModelSpec("agent-a", "fixture/a", revision_a, model_a),
        LocalModelSpec("agent-b", "fixture/b", revision_b, model_b),
    ]
    clock = {"now": 1_000_000}

    def fetch(path, params):
        if path == "/api/v3/time":
            return {"serverTime": clock["now"]}
        if "startTime" in params:
            opened = params["startTime"]
            return [
                [
                    opened,
                    "10.0",
                    "11.0",
                    "9.0",
                    "10.5",
                    "2.0",
                    opened + 59_999,
                    "20.0",
                    4,
                    "1.0",
                    "10.0",
                    "0",
                ]
            ]
        return [
            [
                opened,
                "10.0",
                "11.0",
                "9.0",
                "10.5",
                "2.0",
                opened + 59_999,
                "20.0",
                4,
                "1.0",
                "10.0",
                "0",
            ]
            for opened in (720_000, 780_000, 840_000, 900_000)
        ]

    field = tmp_path / "field"
    prepare_field(
        field,
        specs,
        deadline_delay_minutes=10,
        symbols=("BTCUSDT",),
        target_offsets_minutes=(1,),
        lookback_bars=2,
        fetch=fetch,
    )

    scaffold_sha256 = prospective_field._scaffold_sha256
    monkeypatch.setattr(prospective_field, "_scaffold_sha256", lambda: "0" * 64)
    with pytest.raises(ProspectiveFieldError, match="runner bytes"):
        forecast_agent(
            field,
            specs[0],
            infer=lambda _path, _prompts: ("", {}),
            fetch=fetch,
        )
    monkeypatch.setattr(prospective_field, "_scaffold_sha256", scaffold_sha256)

    def infer(_path, prompts):
        probabilities = {contract_id: 0.5 for contract_id in prompts}
        return (
            json.dumps({"forecasts": probabilities}),
            {
                "method": "binary_next_token_logit",
                "class_token_ids": {"false": 15, "true": 16},
                "logits": {
                    contract_id: {
                        "false_logit": 0.0,
                        "true_logit": 0.0,
                        "unclipped_probability": 0.5,
                        "probability": 0.5,
                    }
                    for contract_id in prompts
                },
                "contract_count": len(prompts),
            },
        )

    clock["now"] = 1_100_000
    for spec in specs:
        forecast_agent(field, spec, infer=infer, fetch=fetch)
    audit_path = field / "inference" / "agent-a.json"
    original_audit = audit_path.read_bytes()
    altered_audit = json.loads(original_audit)
    altered_audit["parsed_predictions"] = {
        key: 0.7 for key in altered_audit["parsed_predictions"]
    }
    audit_path.write_text(json.dumps(altered_audit), encoding="utf-8")
    with pytest.raises(ProspectiveFieldError, match="differs from its ledger"):
        seal_forecasts(field, fetch=fetch)
    audit_path.write_bytes(original_audit)
    clock["now"] = 1_200_000
    seal_forecasts(field, fetch=fetch)
    commit_path = field / "forecast-commit.json"
    original_commit = commit_path.read_bytes()
    altered_commit = json.loads(original_commit)
    altered_commit["files"]["../outside.json"] = "0" * 64
    commit_path.write_text(json.dumps(altered_commit), encoding="utf-8")
    with pytest.raises(ProspectiveFieldError, match="file set"):
        resolve_field(field, fetch=fetch)
    commit_path.write_bytes(original_commit)
    clock["now"] = 1_700_000
    with pytest.raises(ProspectiveFieldError, match="resolution is early"):
        resolve_field(field, fetch=fetch)
    clock["now"] = 1_800_000
    resolve_field(field, fetch=fetch)

    for spec in specs:
        document = json.loads(
            (field / "resolved" / f"{spec.agent_id}.json").read_text()
        )
        assert document["resolutions"][0]["status"] == "resolved"
        assert document["resolutions"][0]["outcome"] == 1.0


def _sealed_two_agent_field(tmp_path):
    """Prepare, forecast and seal a minimal honest field, ready to resolve."""

    revision_a = "a" * 40
    revision_b = "b" * 40
    model_a = tmp_path / revision_a
    model_b = tmp_path / revision_b
    model_a.mkdir()
    model_b.mkdir()
    (model_a / "weights.bin").write_bytes(b"a")
    (model_b / "weights.bin").write_bytes(b"b")
    specs = [
        LocalModelSpec("agent-a", "fixture/a", revision_a, model_a),
        LocalModelSpec("agent-b", "fixture/b", revision_b, model_b),
    ]
    clock = {"now": 1_000_000}

    def fetch(path, params):
        if path == "/api/v3/time":
            return {"serverTime": clock["now"]}
        if "startTime" in params:
            opened = params["startTime"]
            return [_row(opened)]
        return [_row(opened) for opened in (720_000, 780_000, 840_000, 900_000)]

    def infer(_path, prompts):
        return (
            json.dumps({"forecasts": {key: 0.5 for key in prompts}}),
            {
                "method": "binary_next_token_logit",
                "class_token_ids": {"false": 15, "true": 16},
                "logits": {
                    key: {
                        "false_logit": 0.0,
                        "true_logit": 0.0,
                        "unclipped_probability": 0.5,
                        "probability": 0.5,
                    }
                    for key in prompts
                },
                "contract_count": len(prompts),
            },
        )

    field = tmp_path / "field"
    prepare_field(
        field,
        specs,
        deadline_delay_minutes=10,
        symbols=("BTCUSDT", "ETHUSDT"),
        target_offsets_minutes=(1,),
        lookback_bars=2,
        fetch=fetch,
    )
    clock["now"] = 1_100_000
    for spec in specs:
        forecast_agent(field, spec, infer=infer, fetch=fetch)
    return field, specs, clock, fetch


def _row(opened):
    return [
        opened,
        "10.0",
        "11.0",
        "9.0",
        "10.5",
        "2.0",
        opened + 59_999,
        "20.0",
        4,
        "1.0",
        "10.0",
        "0",
    ]


def test_sealing_refuses_a_forecast_bound_outside_the_frozen_contracts(tmp_path):
    """AI1: a committed forecast may not name a frozen ID over other bytes.

    The pending document stays internally valid, keeps the frozen contract ID,
    and rebinds it to a different instrument. Checking contract IDs alone would
    accept it and later attach the BTC outcome to an ETH question.
    """

    field, _specs, clock, fetch = _sealed_two_agent_field(tmp_path)
    pending_path = field / "pending" / "agent-a.json"
    document = json.loads(pending_path.read_text(encoding="utf-8"))
    swapped = dict(document["contracts"][0])
    swapped["instrument"] = "ETHUSDT"
    swapped["question"] = "Will ETHUSDT close above its open?"
    document["contracts"][0] = swapped
    digest = ForecastContract.from_dict(swapped).sha256
    assert digest != document["revisions"][0]["contract_sha256"]
    document["revisions"][0]["contract_sha256"] = digest
    pending_path.write_text(json.dumps(document), encoding="utf-8")
    # The tampered document is still a valid v1 evidence document on its own.
    forecast_evidence_from_json(pending_path.read_text(encoding="utf-8"))
    clock["now"] = 1_200_000
    with pytest.raises(ProspectiveFieldError, match="not the frozen contract"):
        seal_forecasts(field, fetch=fetch)


def test_every_agent_is_settled_from_one_canonical_settlement_record(tmp_path):
    """R06 (Arena half): contract equality has to mean outcome equality.

    Two agents holding the identical frozen contract must carry the identical
    outcome. A resolved document that disagrees is refused even when it is a
    well-formed evidence document and the manifest digests are consistent.
    """

    field, specs, clock, fetch = _sealed_two_agent_field(tmp_path)
    clock["now"] = 1_200_000
    seal_forecasts(field, fetch=fetch)
    clock["now"] = 1_800_000
    resolve_field(field, fetch=fetch)
    report = verify_field_settlement(field)
    assert report["agents"] == ["agent-a", "agent-b"]
    assert len(report["contracts"]) == 2

    resolved_path = field / "resolved" / "agent-b.json"
    document = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolution = document["resolutions"][0]
    assert resolution["outcome"] == 1.0
    resolution["outcome"] = 0.0
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    resolved_path.write_text(payload, encoding="utf-8")
    # Keep the manifest self-consistent so the settlement check, not the digest
    # check, is what refuses the disagreement.
    manifest_path = field / "resolution-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["resolved/agent-b.json"] = hashlib.sha256(
        resolved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ProspectiveFieldError, match="canonical settlement record"):
        verify_field_settlement(field)


@pytest.mark.parametrize("changed", ["prediction", "identity", "rationale", "exposure"])
def test_settlement_refuses_rewriting_the_sealed_forecast(tmp_path, changed):
    field, _specs, clock, fetch = _sealed_two_agent_field(tmp_path)
    clock["now"] = 1_200_000
    seal_forecasts(field, fetch=fetch)
    clock["now"] = 1_800_000
    resolve_field(field, fetch=fetch)
    assert verify_field_settlement(field)["agents"] == ["agent-a", "agent-b"]
    sealed_path = field / "pending" / "agent-a.json"
    sealed_bytes = sealed_path.read_bytes()
    resolved_path = field / "resolved" / "agent-a.json"
    document = json.loads(resolved_path.read_text(encoding="utf-8"))
    if changed == "prediction":
        # A hindsight-perfect forecast must not become verifiable merely by
        # updating the final file's hash. The pending forecast is still 0.5.
        document["revisions"][0]["prediction"] = [1.0]
    elif changed == "identity":
        document["identity"]["model_id"] = "different-model"
    elif changed == "rationale":
        document["revisions"][0]["rationale"] = "Rewritten after settlement"
    else:
        document["revisions"][0]["exposure"]["source_ids"] = ["different-source"]
    payload = json.dumps(document) + "\n"
    # Structural validation and final-file integrity still hold in each attack.
    forecast_evidence_from_json(payload)
    resolved_path.write_text(payload, encoding="utf-8")
    manifest_path = field / "resolution-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["resolved/agent-a.json"] = hashlib.sha256(
        resolved_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    assert sealed_path.read_bytes() == sealed_bytes
    with pytest.raises(ProspectiveFieldError, match="differs from the sealed forecast"):
        verify_field_settlement(field)


class TestLogitRuntimeValidation:
    """Drive the logit validator away from its fixed point.

    At `true_logit == false_logit` the delta is zero, both sigmoid branches
    agree, and the probability is 0.5 whichever way the arithmetic is written.
    A suite that only ever validates that point cannot tell a correct
    implementation from an inverted sigmoid, a swapped class-token order, or a
    changed clip bound. These cases are asymmetric, so each of those mutations
    moves a number the assertions read. Real evidence carries logits of this
    shape: `inference/phi-4.json` records a true_logit of 27.046875.
    """

    @staticmethod
    def _runtime(false_logit: float, true_logit: float) -> tuple[dict, dict]:
        delta = true_logit - false_logit
        if delta >= 0.0:
            unclipped = 1.0 / (1.0 + math.exp(-delta))
        else:
            exp_delta = math.exp(delta)
            unclipped = exp_delta / (1.0 + exp_delta)
        probability = min(0.99, max(0.01, unclipped))
        record = {
            "false_logit": false_logit,
            "true_logit": true_logit,
            "unclipped_probability": unclipped,
            "probability": probability,
        }
        runtime = {
            "method": "binary_next_token_logit",
            "class_token_ids": {"false": 15, "true": 16},
            "logits": {"contract-1": record},
            "contract_count": 1,
        }
        return runtime, {"contract-1": probability}

    @pytest.mark.parametrize(
        "false_logit,true_logit",
        [
            (0.0, 2.0),
            (2.0, 0.0),
            (-1.5, 3.25),
            (0.0, 27.046875),
            (27.046875, 0.0),
        ],
    )
    def test_a_consistent_asymmetric_record_is_accepted(
        self, false_logit: float, true_logit: float
    ) -> None:
        runtime, parsed = self._runtime(false_logit, true_logit)
        prospective_field._validate_logit_runtime(runtime, parsed)

    def test_an_inverted_sigmoid_is_rejected(self) -> None:
        runtime, parsed = self._runtime(0.0, 2.0)
        record = runtime["logits"]["contract-1"]
        flipped = 1.0 - record["unclipped_probability"]
        record["unclipped_probability"] = flipped
        record["probability"] = flipped
        with pytest.raises(ProspectiveFieldError):
            prospective_field._validate_logit_runtime(runtime, {"contract-1": flipped})

    def test_swapped_class_tokens_are_rejected(self) -> None:
        runtime, parsed = self._runtime(0.0, 2.0)
        record = runtime["logits"]["contract-1"]
        record["false_logit"], record["true_logit"] = (
            record["true_logit"],
            record["false_logit"],
        )
        with pytest.raises(ProspectiveFieldError):
            prospective_field._validate_logit_runtime(runtime, parsed)

    def test_a_clip_bound_that_moved_is_rejected(self) -> None:
        runtime, parsed = self._runtime(0.0, 27.046875)
        record = runtime["logits"]["contract-1"]
        assert record["probability"] == 0.99
        record["probability"] = 0.995
        with pytest.raises(ProspectiveFieldError):
            prospective_field._validate_logit_runtime(runtime, {"contract-1": 0.995})
