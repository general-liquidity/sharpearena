"""Finish-reason accounting for the local-model field and the Bench bridge.

A completion the token budget cut off can still parse into a decision. Before this
record existed, the local field kept no stop reason at all, so a heavily truncated
field looked operationally clean in the bridge's rank-neutral profile. No model
server or network access is required.
"""

from __future__ import annotations

import json

import pytest
from sharpearena.bench_bridge import BenchBridgeError, compile_benchmark_evidence
from sharpearena.local_agents import (
    FINISH_REASONS,
    DatasetSpec,
    EvidenceJournal,
    FieldPlan,
    InferenceOutcome,
    InferenceResult,
    LocalAgentError,
    LocalFieldRunner,
    ModelIdentity,
    ModelRunConfig,
    OllamaClient,
    OpenAICompatibleClient,
    PromptRenderer,
    SamplingConfig,
    classify_finish_reason,
)

_OBSERVATION = {
    "date": "2026-01-01",
    "cash": 1000.0,
    "symbols": [{"symbol": "AAA", "close_history": [100.0]}],
    "portfolio": [],
}
_DECISION = json.dumps(
    {
        "orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.2}],
        "reasoning": "fixture",
    }
)


def _result(seed, finish_reason):
    return InferenceResult(
        decision={
            "orders": [],
            "reasoning": "fixture",
            "cost": {
                "cost_usd": 0.0,
                "tokens_in": 10,
                "tokens_out": 5,
                "reasoning_tokens": 0,
            },
        },
        raw_response_sha256=f"response-{seed}",
        prompt_tokens=10,
        output_tokens=5,
        reasoning_tokens=None,
        total_duration_ns=100,
        duration_source="host-monotonic-request",
        raw_response=f"raw-{seed}",
        finish_reason=finish_reason,
    )


class ScriptedModel:
    """Answers every request, cycling through a script of finish reasons.

    After ``fail_after`` rounds of requests it fails every later one, so a cell's
    failed attempt still carries the reasons of the requests it completed.
    """

    def __init__(self, reasons, fail_after=None):
        self._reasons = list(reasons)
        self._served = 0
        self._rounds = 0
        self._fail_after = fail_after

    def identity(self, model):
        return ModelIdentity(
            model=model.model,
            digest="sha256:fixed",
            parameter_size="test",
            quantization="none",
            family="fixture",
            server="fixture",
            server_version="1",
        )

    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        self._rounds += 1
        if self._fail_after is not None and self._rounds > self._fail_after:
            return [
                InferenceOutcome(
                    error_type="DecisionResponseError",
                    error="truncated JSON",
                    attempt_duration_ns=50,
                    finish_reason="length",
                )
                for _ in observations
            ]
        outcomes = []
        for seed in sampling_seeds:
            reason = self._reasons[self._served % len(self._reasons)]
            self._served += 1
            outcomes.append(InferenceOutcome(result=_result(seed, reason)))
        return outcomes


def _plan(repetitions=1):
    return FieldPlan(
        models=(
            ModelRunConfig(
                "test-fixture:synthetic",
                SamplingConfig(seed=40),
                decision_cadence=2,
                entry_class="field",
                source_url="https://example.test/models/test-fixture",
                source_revision="0123456789abcdef",
                license_id="MIT",
            ),
        ),
        datasets=(DatasetSpec("synthetic-calm", tier="calm", n_symbols=2, n_days=12),),
        seeds=(1, 2),
        repetitions=repetitions,
        max_steps=5,
    )


def _records(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()]


def _rewrite(journal, records):
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _profile(result):
    return result["outputs"][0]["models"][0]["operational_profile"]


def test_the_class_vocabulary_is_exact():
    assert FINISH_REASONS == ("stop", "length", "other", "absent")
    assert classify_finish_reason("stop") == "stop"
    assert classify_finish_reason("length") == "length"
    assert classify_finish_reason(None) == "absent"
    # Spellings this module does not normalize are not guessed at.
    for value in ("max_tokens", "STOP", "", "tool_calls", 1, True, ["stop"]):
        assert classify_finish_reason(value) == "other"


def test_an_unknown_class_is_refused_where_a_result_is_built():
    with pytest.raises(LocalAgentError, match="finish_reason"):
        _result(0, "truncated")
    with pytest.raises(LocalAgentError, match="finish_reason"):
        InferenceOutcome(error_type="X", error="y", finish_reason="stopped")
    # A client that does not read the reason records it as absent, never stop.
    unread = InferenceResult(
        decision={"orders": []},
        raw_response_sha256="x",
        prompt_tokens=1,
        output_tokens=1,
        reasoning_tokens=None,
        total_duration_ns=1,
        duration_source="host-monotonic-request",
    )
    assert unread.finish_reason == "absent"


def test_a_length_stopped_request_is_counted_in_the_record_and_the_bridge_profile(
    tmp_path,
):
    """The regression: a truncated completion that still became a decision."""

    journal = tmp_path / "field.jsonl"
    runner = LocalFieldRunner(ScriptedModel(["length", "stop", "absent"]))
    assert runner.run(_plan(), EvidenceJournal(journal))["failed"] == 0
    records = _records(journal)
    # Two cells step in one batch, three requests each, reasons assigned in turn.
    for record in records:
        assert len(record["finish_reason_observations"]) == 3
        assert record["finish_reasons"] == {
            reason: record["finish_reason_observations"].count(reason)
            for reason in FINISH_REASONS
        }
    result = compile_benchmark_evidence([journal], tmp_path / "compiled")
    profile = _profile(result)
    assert profile["rank_input"] is False
    assert profile["finish_reasons"] == {
        "stop": 2,
        "length": 2,
        "other": 0,
        "absent": 2,
        "unrecorded": 0,
    }
    assert profile["attempt_ledger"]["finish_reasons"] == profile["finish_reasons"]
    manifest = json.loads((tmp_path / "compiled" / "benchmark-manifest.json").read_text())
    assert (
        manifest["outputs"][0]["models"][0]["operational_profile"]["finish_reasons"][
            "length"
        ]
        == 2
    )


def _drop(field):
    return lambda record: record.pop(field)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record["finish_reasons"].__setitem__(
                "stop", record["finish_reasons"]["stop"] + 1
            ),
            "does not match",
        ),
        (
            lambda record: record["finish_reasons"].__setitem__(
                "length", bool(record["finish_reasons"]["length"])
            ),
            "does not match",
        ),
        (lambda record: record["finish_reasons"].pop("other"), "does not match"),
        (
            lambda record: record["finish_reasons"].__setitem__("truncated", 0),
            "does not match",
        ),
        (
            lambda record: record.__setitem__(
                "finish_reason_observations",
                record["finish_reason_observations"][:-1],
            ),
            "one entry per model request",
        ),
        (
            lambda record: record["finish_reason_observations"].__setitem__(
                0, "max_tokens"
            ),
            "must be one of",
        ),
        (_drop("finish_reason_observations"), "together"),
        (_drop("finish_reasons"), "together"),
    ],
)
def test_bridge_validation_refuses_inconsistent_finish_reason_evidence(
    tmp_path, mutate, message
):
    journal = tmp_path / "field.jsonl"
    LocalFieldRunner(ScriptedModel(["length"])).run(_plan(), EvidenceJournal(journal))
    records = _records(journal)
    mutate(records[0])
    _rewrite(journal, records)
    with pytest.raises(BenchBridgeError, match=message):
        compile_benchmark_evidence([journal], tmp_path / "compiled")


def test_a_record_written_before_the_field_is_unrecorded_not_absent(tmp_path):
    journal = tmp_path / "field.jsonl"
    LocalFieldRunner(ScriptedModel(["stop"])).run(_plan(), EvidenceJournal(journal))
    records = _records(journal)
    for record in records:
        del record["finish_reason_observations"]
        del record["finish_reasons"]
    _rewrite(journal, records)
    profile = _profile(compile_benchmark_evidence([journal], tmp_path / "compiled"))
    assert profile["finish_reasons"] == {
        "stop": 0,
        "length": 0,
        "other": 0,
        "absent": 0,
        "unrecorded": 6,
    }


def test_the_attempt_ledger_keeps_the_reasons_of_a_failed_attempt(tmp_path):
    """A resumed cell's failed attempt still made truncated requests."""

    journal = tmp_path / "resumed.jsonl"
    plan = _plan()
    assert LocalFieldRunner(ScriptedModel(["length"], fail_after=1)).run(
        plan, EvidenceJournal(journal)
    ) == {"completed": 0, "failed": 2, "skipped": 0}
    failed = _records(journal)
    for record in failed:
        assert record["finish_reason_observations"] == ["length"]
        # The rejected completion's reason is in the failure, not in the counts.
        assert record["failure"]["finish_reason"] == "length"
    LocalFieldRunner(ScriptedModel(["stop"])).run(plan, EvidenceJournal(journal))

    result = compile_benchmark_evidence([journal], tmp_path / "compiled")
    profile = _profile(result)
    assert profile["finish_reasons"]["stop"] == 6
    assert profile["finish_reasons"]["length"] == 0
    assert profile["attempt_ledger"]["finish_reasons"] == {
        "stop": 6,
        "length": 2,
        "other": 0,
        "absent": 0,
        "unrecorded": 0,
    }

    # The ledger applies the same validation to a failed attempt.
    records = _records(journal)
    records[0]["finish_reasons"]["length"] = 0
    _rewrite(journal, records)
    with pytest.raises(BenchBridgeError, match="does not match"):
        compile_benchmark_evidence([journal], tmp_path / "compiled-again")


class _Ollama(OllamaClient):
    def __init__(self, content, extra):
        super().__init__()
        self._content = content
        self._extra = extra

    def _request(self, method, path, payload=None):
        if path == "/api/tags":
            return {
                "models": [
                    {
                        "name": "test-fixture:synthetic",
                        "digest": "sha256:model",
                        "details": {
                            "family": "fixture",
                            "parameter_size": "1B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            }
        if path == "/api/show":
            return {"model_info": {"fixture.context_length": 8192}}
        if path == "/api/version":
            return {"version": "9.9.9"}
        if path == "/api/ps":
            return {"models": []}
        assert path == "/api/chat"
        return {
            "message": {"content": self._content},
            "prompt_eval_count": 12,
            "eval_count": 7,
            "total_duration": 123,
            **self._extra,
        }


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"done_reason": "length"}, "length"),
        ({"done_reason": "stop"}, "stop"),
        ({"done_reason": "unload"}, "other"),
        ({"done_reason": None}, "absent"),
        ({}, "absent"),
    ],
)
def test_ollama_records_its_done_reason(monkeypatch, extra, expected):
    monkeypatch.setattr("sharpearena.local_agents._local_gpu_identity", lambda: {})
    monkeypatch.setattr("sharpearena.local_agents._wrapper_version", lambda: "0.0-test")
    client = _Ollama(_DECISION, extra)
    config = ModelRunConfig("test-fixture:synthetic", SamplingConfig(seed=4))
    client.identity(config)
    assert client.decide(_OBSERVATION, config, PromptRenderer()).finish_reason == expected


def test_a_rejected_truncated_completion_keeps_its_reason(monkeypatch):
    monkeypatch.setattr("sharpearena.local_agents._local_gpu_identity", lambda: {})
    monkeypatch.setattr("sharpearena.local_agents._wrapper_version", lambda: "0.0-test")
    client = _Ollama(_DECISION[:20], {"done_reason": "length"})
    config = ModelRunConfig("test-fixture:synthetic", SamplingConfig(seed=4))
    client.identity(config)
    outcome = client.decide_many(
        [_OBSERVATION], config, PromptRenderer(), max_workers=1
    )[0]
    assert outcome.error_type == "DecisionResponseError"
    assert outcome.finish_reason == "length"


def test_an_openai_compatible_backend_records_its_finish_reason():
    identity = ModelIdentity(
        model="frontier-fixture",
        digest="sha256:" + "c" * 64,
        parameter_size="27B",
        quantization="Q4_K_M",
        offload="full CUDA",
        server="vllm",
        server_version="0.11",
    )

    class Stub(OpenAICompatibleClient):
        def __init__(self, choice):
            super().__init__((identity,))
            self._choice = choice

        def _request(self, method, path, payload=None):
            if path == "/models":
                return {"data": [{"id": "frontier-fixture"}]}
            return {
                "choices": [{"message": {"content": _DECISION}, **self._choice}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            }

    config = ModelRunConfig("frontier-fixture")
    for choice, expected in (
        ({"finish_reason": "length"}, "length"),
        ({"finish_reason": "stop"}, "stop"),
        ({"finish_reason": "content_filter"}, "other"),
        ({}, "absent"),
    ):
        client = Stub(choice)
        client.identity(config)
        result = client.decide(_OBSERVATION, config, PromptRenderer())
        assert result.finish_reason == expected, choice
