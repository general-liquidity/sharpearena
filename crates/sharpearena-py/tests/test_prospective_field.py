"""Prospective field phases preserve ordering, identity, and settlement boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sharpearena import prospective_field
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
            infer=lambda _path, _prompt, _contract_ids: ("", {}),
            fetch=fetch,
        )
    monkeypatch.setattr(prospective_field, "_scaffold_sha256", scaffold_sha256)

    def infer(_path, prompt, _contract_ids):
        contract_id = prompt.split('"contract_id":"', 1)[1].split('"', 1)[0]
        return json.dumps({"forecasts": {contract_id: 0.6}}), {"fixture": True}

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
