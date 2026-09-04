"""Local model harness tests. No model server or network access is required."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
from pathlib import Path

import pytest
from sharpearena.bench_bridge import BenchBridgeError, compile_benchmark_evidence
from sharpearena.local_agents import (
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
    load_identity_manifest,
)
from sharpearena.local_field_cli import load_plan
from sharpearena.local_field_cli import main as field_cli_main
from sharpearena.ollama_shim import run_stdio
from sharpearena.sharpearena_py import decision_schema_json


class FixedModel:
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
        self,
        observations,
        model,
        renderer,
        *,
        max_workers,
        sampling_seeds=None,
    ):
        assert len(sampling_seeds) == len(observations)
        return [
            InferenceOutcome(
                result=InferenceResult(
                    decision={
                        "orders": [
                            {
                                "symbol": item["symbol"],
                                "action": "buy",
                                "target_weight": 0.1,
                                "confidence": 0.5,
                                "rationale": "fixture",
                            }
                            for item in observation["symbols"]
                        ],
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
                    reasoning_tokens=0,
                    total_duration_ns=100,
                    reasoning_tokens_available=False,
                    raw_response=f"raw-{seed}",
                )
            )
            for observation, seed in zip(observations, sampling_seeds)
        ]


class BrokenModel(FixedModel):
    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        return [
            InferenceOutcome(error_type="DecisionParseError", error="bad output")
            for _ in observations
        ]


class NoConfidenceModel(FixedModel):
    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        outcomes = super().decide_many(
            observations,
            model,
            renderer,
            max_workers=max_workers,
            sampling_seeds=sampling_seeds,
        )
        for outcome in outcomes:
            assert outcome.result is not None
            for order in outcome.result.decision["orders"]:
                order.pop("confidence", None)
        return outcomes


def _plan(repetitions=2, shard_index=0, shard_count=1):
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
        shard_index=shard_index,
        shard_count=shard_count,
    )


def test_embedded_decision_schema_is_the_published_closed_contract():
    schema = json.loads(decision_schema_json())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["orders"]
    assert set(schema["properties"]) == {"orders", "reasoning", "cost"}


def test_field_runner_batches_scores_and_records_repetition_seed(tmp_path):
    path = tmp_path / "field.jsonl"
    journal = EvidenceJournal(path)
    counts = LocalFieldRunner(FixedModel()).run(_plan(), journal)
    assert counts == {"completed": 4, "failed": 0, "skipped": 0}

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 4
    assert {record["inference_seed"] for record in records} == {40, 41}
    assert all(
        record["evidence_class"] == "retrospective_local_model" for record in records
    )
    assert all(record["deterministic_environment"] is True for record in records)
    assert all(record["deterministic_agent"] is False for record in records)
    assert all(
        record["n_trials_source"] == "precommitted-model-entry" for record in records
    )
    assert all(len(record["returns"]) == 5 for record in records)
    assert all("deflated_sharpe" in record["score"] for record in records)
    assert all(record["cost"] == 0.0 for record in records)
    assert all(record["cost_unit"] == "usd-self-reported" for record in records)
    assert all(
        record["tokens_in"] > 0 and record["tokens_out"] > 0 for record in records
    )
    assert all(record["reasoning_tokens_source"] == "unavailable" for record in records)
    assert all(record["retry_count"] == 0 for record in records)
    assert all(record["schema_version"] == 2 for record in records)
    assert all(record["inference_duration_ns"] == 300 for record in records)
    assert all(
        record["inference_duration_samples_ns"] == [100, 100, 100] for record in records
    )
    assert all(
        record["inference_duration_source"] == "unspecified" for record in records
    )
    assert all(record["raw_responses"] for record in records)
    assert {record["cell_ordinal"] for record in records} == {0, 1, 2, 3}
    assert all(record["field_shape"]["total_cells"] == 4 for record in records)
    assert all(record["termination"] == "runner_step_budget" for record in records)
    # Cadence two means only steps 0, 2, and 4 carry the model-call cost.
    assert all(
        sum("cost" in decision for decision in record["decisions"]) == 3
        for record in records
    )

    resumed = LocalFieldRunner(FixedModel()).run(_plan(), journal)
    assert resumed == {"completed": 0, "failed": 0, "skipped": 4}
    assert len(path.read_text().splitlines()) == 4


def test_field_runner_preserves_the_native_engine_trace_exactly(tmp_path):
    plan = _plan(repetitions=1)
    path = tmp_path / "field.jsonl"
    LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(path))
    record = json.loads(path.read_text().splitlines()[0])

    env = LocalFieldRunner._build_env(plan.datasets[0], [record["seed"]])
    env.reset_batch()
    expected = []
    for decision in record["decisions"]:
        stepped = json.loads(env.step_batch(json.dumps([decision])))
        expected.extend(stepped["infos"][0]["events"])
    assert record["trace"]["events"] == expected
    assert record["trace"]["events"]


def test_failed_inference_is_evidence_not_a_flat_return_series(tmp_path):
    path = tmp_path / "failed.jsonl"
    counts = LocalFieldRunner(BrokenModel()).run(
        _plan(repetitions=1), EvidenceJournal(path)
    )
    assert counts == {"completed": 0, "failed": 2, "skipped": 0}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert record["status"] == "failed"
        assert record["steps"] == 0
        assert "returns" not in record
        assert record["failure"]["type"] == "DecisionParseError"


@pytest.mark.parametrize(
    ("events", "reward", "expected_type"),
    [
        ("not-a-list", 0.0, "InvalidProcessTrace"),
        ([], float("nan"), "NonFiniteReward"),
    ],
)
def test_native_step_faults_fail_the_cell_without_returns(
    tmp_path, monkeypatch, events, reward, expected_type
):
    class FaultEnv:
        def reset_batch(self):
            return json.dumps(
                {
                    "observations": [
                        {
                            "date": "2026-01-01",
                            "cash": 1.0,
                            "symbols": [
                                {"symbol": "AAA", "close_history": [1.0]}
                            ],
                            "portfolio": [],
                        }
                    ]
                }
            )

        def step_batch(self, decisions):
            return json.dumps(
                {
                    "observations": [{}],
                    "rewards": [reward],
                    "terminated": [False],
                    "truncated": [False],
                    "infos": [{"events": events}],
                }
            )

    monkeypatch.setattr(LocalFieldRunner, "_build_env", staticmethod(lambda *_: FaultEnv()))
    plan = FieldPlan(
        models=_plan(repetitions=1).models,
        datasets=_plan(repetitions=1).datasets,
        seeds=(1,),
        repetitions=1,
        max_steps=2,
    )
    path = tmp_path / f"{expected_type}.jsonl"
    counts = LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(path))
    assert counts == {"completed": 0, "failed": 1, "skipped": 0}
    record = json.loads(path.read_text())
    assert record["failure"]["type"] == expected_type
    assert "returns" not in record


def test_one_scored_bar_is_insufficient_and_not_published_as_a_run(tmp_path):
    source = _plan(repetitions=1)
    plan = FieldPlan(
        models=source.models,
        datasets=source.datasets,
        seeds=(1,),
        repetitions=1,
        max_steps=1,
    )
    path = tmp_path / "short.jsonl"
    counts = LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(path))
    assert counts == {"completed": 0, "failed": 1, "skipped": 0}
    record = json.loads(path.read_text())
    assert record["failure"]["type"] == "InsufficientReturns"
    assert "returns" not in record


def test_unreported_confidence_stays_absent_from_scored_calibration(tmp_path):
    path = tmp_path / "no-confidence.jsonl"
    counts = LocalFieldRunner(NoConfidenceModel()).run(
        _plan(repetitions=1), EvidenceJournal(path)
    )
    assert counts == {"completed": 2, "failed": 0, "skipped": 0}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert record["confidences"] == []
        assert record["outcomes"] == []


def test_shards_are_disjoint_and_cover_the_field():
    full = {cell.key(_plan()) for cell in LocalFieldRunner.cells(_plan())}
    left_plan = _plan(shard_index=0, shard_count=2)
    right_plan = _plan(shard_index=1, shard_count=2)
    left = {cell.key(left_plan) for cell in LocalFieldRunner.cells(left_plan)}
    right = {cell.key(right_plan) for cell in LocalFieldRunner.cells(right_plan)}
    # The plan hash excludes shard_index, so cell identity is stable across shards.
    assert left.isdisjoint(right)
    assert left | right == full


def test_two_executed_shards_recombine_into_one_complete_benchmark_field(tmp_path):
    left_plan = _plan(shard_index=0, shard_count=2)
    right_plan = _plan(shard_index=1, shard_count=2)
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    assert LocalFieldRunner(FixedModel()).run(left_plan, EvidenceJournal(left)) == {
        "completed": 2,
        "failed": 0,
        "skipped": 0,
    }
    assert LocalFieldRunner(FixedModel()).run(right_plan, EvidenceJournal(right)) == {
        "completed": 2,
        "failed": 0,
        "skipped": 0,
    }
    manifest = compile_benchmark_evidence([left, right], tmp_path / "bench")
    assert manifest["field_shape"]["total_cells"] == 4
    assert len(manifest["source_journals"]) == 2
    assert manifest["outputs"][0]["runs_per_agent"] == 4


def test_field_plan_rejects_ambiguous_coordinates_and_invalid_dataset_controls():
    model = ModelRunConfig("test-fixture:synthetic")
    dataset = DatasetSpec("one")
    with pytest.raises(ValueError, match="duplicate model"):
        FieldPlan(models=(model, model), datasets=(dataset,), seeds=(1,))
    with pytest.raises(ValueError, match="dataset_id"):
        FieldPlan(
            models=(model,),
            datasets=(DatasetSpec("one"), DatasetSpec("one", tier="hard")),
            seeds=(1,),
        )
    with pytest.raises(ValueError, match="window_start"):
        DatasetSpec("bad-window", window_start=5, window_end=5)
    with pytest.raises(ValueError, match="fee_bps"):
        DatasetSpec("bad-cost", fee_bps=-1.0)
    with pytest.raises(ValueError, match="max_participation"):
        DatasetSpec("bad-participation", max_participation=1.1)
    with pytest.raises(ValueError, match="sampling seed plus repetition"):
        FieldPlan(
            models=(
                ModelRunConfig(
                    "test-fixture:synthetic", SamplingConfig(seed=2**63 - 1)
                ),
            ),
            datasets=(dataset,),
            seeds=(1,),
            repetitions=2,
        )


def test_prompt_renderer_is_stateless_and_carries_cadence():
    obs = {"date": "2026-01-01", "cash": 1000.0, "symbols": [], "portfolio": []}
    messages = PromptRenderer().messages(obs, cadence=12)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert json.loads(messages[1]["content"])["decision_cadence_bars"] == 12


def test_ollama_client_is_local_only_and_records_artifact_identity(monkeypatch):
    with pytest.raises(ValueError, match="non-loopback"):
        OllamaClient("https://models.example.com")

    class StubOllama(OllamaClient):
        def __init__(self):
            super().__init__()
            self.chat_payload = None

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
                return {
                    "models": [
                        {
                            "name": "test-fixture:synthetic",
                            "size": 1000,
                            "size_vram": 750,
                        }
                    ]
                }
            assert path == "/api/chat"
            self.chat_payload = payload
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "orders": [
                                {
                                    "symbol": "AAA",
                                    "action": "buy",
                                    "target_weight": 0.2,
                                }
                            ],
                            "reasoning": "fixture",
                            "cost": {"cost_usd": 999},
                        }
                    )
                },
                "prompt_eval_count": 12,
                "eval_count": 7,
                "total_duration": 123,
            }

    monkeypatch.setattr("sharpearena.local_agents._local_gpu_identity", lambda: {})
    monkeypatch.setattr("sharpearena.local_agents._wrapper_version", lambda: "0.0-test")
    client = StubOllama()
    config = ModelRunConfig(
        "test-fixture:synthetic", SamplingConfig(seed=4, thinking=True)
    )
    identity = client.identity(config)
    assert asdict(identity) == {
        "model": "test-fixture:synthetic",
        "digest": "sha256:model",
        "parameter_size": "1B",
        "quantization": "Q4_K_M",
        "offload": "size_vram=750; size=1000",
        "family": "fixture",
        "context_length": 8192,
        "server": "ollama",
        "server_version": "9.9.9",
        "size_bytes": None,
        "format": "unknown",
        "capabilities": (),
        "license_sha256": None,
        "modelfile_sha256": None,
        "template_sha256": None,
        "parameters_sha256": None,
        "quantizer": "unresolved",
        "converter_version": "unresolved",
        "quantization_calibration": "unresolved",
        "server_commit": "unresolved",
        "wrapper": "sharpearena",
        "wrapper_version": "0.0-test",
        "chat_template": "unresolved",
        "reasoning_parser": "none",
        "tool_parser": "canonical-decision-json",
        "constrained_decoding_backend": "ollama-json-schema",
        "kv_cache_dtype": "unresolved",
        "tensor_parallelism": 1,
        "batch_size": 1,
        "parallel_slots": 1,
        "prefix_cache": "unresolved",
        "speculative_decoding": "unresolved",
        "gpu_name": "unresolved",
        "gpu_memory_mib": None,
        "gpu_driver_version": "unresolved",
        "gpu_compute_capability": "unresolved",
        "cuda_version": "unresolved",
    }
    observation = {
        "date": "2026-01-01",
        "cash": 1000.0,
        "symbols": [{"symbol": "AAA", "close_history": [100.0]}],
        "portfolio": [],
    }
    result = client.decide(observation, config, PromptRenderer(), sampling_seed=8)
    assert result.decision["cost"] == {
        "cost_usd": 0.0,
        "tokens_in": 12,
        "tokens_out": 7,
        "reasoning_tokens": 0,
    }
    assert client.chat_payload["options"]["seed"] == 8
    assert client.chat_payload["options"]["num_ctx"] == 8192
    assert client.chat_payload["think"] is True
    assert client.chat_payload["format"]["title"] == "Decision"
    assert result.reasoning_tokens_available is False


def test_openai_compatible_client_is_local_strict_and_provenance_complete(tmp_path):
    manifest = tmp_path / "identities.json"
    manifest.write_text(
        json.dumps(
            {
                "model": "frontier-fixture",
                "digest": "sha256:" + "a" * 64,
                "parameter_size": "27B",
                "quantization": "Q4_K_M",
                "offload": "58/65 layers on CUDA; remainder CPU",
                "family": "fixture",
                "context_length": 262144,
                "server": "llama.cpp",
                "server_version": "b9999",
                "format": "gguf",
                "capabilities": ["thinking", "tools"],
            }
        ),
        encoding="utf-8",
    )
    identities = load_identity_manifest(manifest)
    with pytest.raises(ValueError, match="non-loopback"):
        OpenAICompatibleClient(identities, base_url="http://models.example.test:8000")
    with pytest.raises(ValueError, match="explicit provenance"):
        OpenAICompatibleClient((ModelIdentity(model="missing", digest="sha256:x"),))

    class StubClient(OpenAICompatibleClient):
        def __init__(self):
            super().__init__(
                identities,
                supports_thinking=True,
                supports_thinking_budget=True,
            )
            self.chat_payload = None

        def _request(self, method, path, payload=None):
            if path == "/models":
                return {"data": [{"id": "frontier-fixture"}]}
            assert path == "/chat/completions"
            self.chat_payload = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "orders": [
                                        {
                                            "symbol": "AAA",
                                            "action": "buy",
                                            "target_weight": 0.2,
                                        }
                                    ],
                                    "reasoning": "fixture",
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            }

    client = StubClient()
    config = ModelRunConfig(
        "frontier-fixture",
        SamplingConfig(seed=7, thinking=True, thinking_budget_tokens=256),
    )
    identity = client.identity(config)
    assert identity.server == "llama.cpp"
    assert identity.server_version == "b9999"
    assert identity.quantization == "Q4_K_M"
    assert identity.offload == "58/65 layers on CUDA; remainder CPU"
    observation = {
        "date": "2026-01-01",
        "cash": 1000.0,
        "symbols": [{"symbol": "AAA", "close_history": [100.0]}],
        "portfolio": [],
    }
    result = client.decide(observation, config, PromptRenderer(), sampling_seed=11)
    assert result.decision["cost"] == {
        "cost_usd": 0.0,
        "tokens_in": 20,
        "tokens_out": 8,
        "reasoning_tokens": 3,
    }
    assert result.reasoning_tokens_available is True
    assert client.chat_payload["seed"] == 11
    assert client.chat_payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking_budget": 256,
    }
    response_format = client.chat_payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["title"] == "Decision"


def test_openai_compatible_client_rejects_schema_valid_but_semantically_invalid_output():
    identity = ModelIdentity(
        model="frontier-fixture",
        digest="sha256:" + "b" * 64,
        parameter_size="27B",
        quantization="Q4_K_M",
        offload="full CUDA",
        server="sglang",
        server_version="0.5.5",
    )

    class InvalidClient(OpenAICompatibleClient):
        def __init__(self):
            super().__init__((identity,))

        def _request(self, method, path, payload=None):
            if path == "/models":
                return {"data": [{"id": "frontier-fixture"}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "orders": [
                                        {
                                            "symbol": "UNKNOWN",
                                            "action": "buy",
                                            "target_weight": 0.2,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    client = InvalidClient()
    client.identity(ModelRunConfig("frontier-fixture"))
    observation = {
        "date": "2026-01-01",
        "cash": 1000.0,
        "symbols": [{"symbol": "AAA", "close_history": [100.0]}],
        "portfolio": [],
    }
    outcome = client.decide_many(
        [observation],
        ModelRunConfig("frontier-fixture"),
        PromptRenderer(),
        max_workers=1,
    )[0]
    assert outcome.error_type == "DecisionResponseError"
    assert "invalid Decision" in (outcome.error or "")
    assert outcome.raw_response_sha256 == sha256(
        json.dumps(
            {
                "orders": [
                    {
                        "symbol": "UNKNOWN",
                        "action": "buy",
                        "target_weight": 0.2,
                    }
                ]
            }
        ).encode()
    ).hexdigest()


def test_sampling_context_and_thinking_budget_are_explicit_capability_axes():
    with pytest.raises(ValueError, match="context_tokens"):
        SamplingConfig(context_tokens=0)
    with pytest.raises(ValueError, match="requires thinking"):
        SamplingConfig(thinking_budget_tokens=10)
    config = ModelRunConfig(
        "fixture", SamplingConfig(thinking=True, thinking_budget_tokens=10)
    )
    identity = ModelIdentity(
        model="fixture",
        digest="sha256:fixture",
        parameter_size="1B",
        quantization="Q4",
        offload="GPU",
        server="fixture",
        server_version="1",
    )
    client = OpenAICompatibleClient((identity,), supports_thinking=True)
    with pytest.raises(LocalAgentError, match="budget-capable"):
        client.decide(
            {
                "symbols": [{"symbol": "AAA", "close_history": [1.0]}],
                "portfolio": [],
                "cash": 1.0,
                "date": "2026-01-01",
            },
            config,
            PromptRenderer(),
        )


def test_stdio_shim_emits_one_decision_per_observation_and_fails_closed(tmp_path):
    class ShimClient:
        def identity(self, model):
            return FixedModel().identity(model)

        def decide(self, observation, model, renderer, *, sampling_seed=None):
            symbol = observation["symbols"][0]["symbol"]
            return InferenceResult(
                decision={
                    "orders": [
                        {
                            "symbol": symbol,
                            "action": "buy",
                            "target_weight": 0.1,
                        }
                    ],
                    "reasoning": "shim fixture",
                    "cost": {"tokens_in": 1, "tokens_out": 1},
                },
                raw_response_sha256="response",
                prompt_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                total_duration_ns=1,
            )

    observation = json.dumps(
        {
            "date": "2026-01-01",
            "cash": 1000.0,
            "symbols": [{"symbol": "AAA", "close_history": [100.0]}],
            "portfolio": [],
        }
    )
    output, errors = StringIO(), StringIO()
    identity = tmp_path / "identity.json"
    config = ModelRunConfig("test-fixture:synthetic", decision_cadence=2)
    exit_code = run_stdio(
        ShimClient(),
        config,
        StringIO(observation + "\n" + observation + "\n"),
        output,
        errors,
        identity_path=identity,
    )
    assert exit_code == 0 and errors.getvalue() == ""
    emitted = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(emitted) == 2
    assert "cost" in emitted[0] and "cost" not in emitted[1]
    assert json.loads(identity.read_text())["digest"] == "sha256:fixed"

    bad_output, bad_errors = StringIO(), StringIO()
    exit_code = run_stdio(
        ShimClient(), config, StringIO("[]\n"), bad_output, bad_errors
    )
    assert exit_code == 1
    assert bad_output.getvalue() == ""
    assert json.loads(bad_errors.getvalue())["error_type"] == "ValueError"


def test_field_cli_resolves_historical_data_and_inspects_without_inference(
    tmp_path, capsys
):
    dataset = tmp_path / "prices.csv"
    dataset.write_text(
        "date,symbol,close\n2026-01-01,AAA,100\n2026-01-02,AAA,101\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "field.json"
    plan_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "test-fixture:synthetic",
                        "decision_cadence": 5,
                        "sampling": {"seed": 11, "thinking": False},
                    }
                ],
                "datasets": [{"dataset_id": "fixture-daily", "csv_path": "prices.csv"}],
                "seeds": [7, 8],
                "repetitions": 3,
            }
        ),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.datasets[0].kind == "historical"
    assert plan.datasets[0].csv_text == dataset.read_text(encoding="utf-8")
    assert plan.models[0].decision_cadence == 5
    assert (
        field_cli_main(
            [
                "--plan",
                str(plan_path),
                "--evidence",
                str(tmp_path / "unused.jsonl"),
                "--inspect",
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["plan_sha256"] == plan.plan_sha256
    assert inspected["cells_in_shard"] == 6
    assert inspected["datasets"][0]["content_sha256"] == plan.datasets[0].content_sha256


def test_field_cli_rejects_unknown_plan_fields(tmp_path):
    plan_path = tmp_path / "field.json"
    plan_path.write_text(
        json.dumps(
            {
                "models": [{"model": "test-fixture:synthetic"}],
                "datasets": [{"dataset_id": "fixture"}],
                "seeds": [1],
                "typo_parallel": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_plan(plan_path)


def test_field_entries_require_public_source_revision_and_license_provenance():
    with pytest.raises(ValueError, match="source_url, source_revision, and license_id"):
        ModelRunConfig("anonymous:9b", entry_class="field")
    config = ModelRunConfig(
        "published:9b",
        entry_class="field",
        source_url="https://example.test/model-card",
        source_revision="0123456789abcdef",
        license_id="Apache-2.0",
    )
    assert config.entry_class == "field"
    with pytest.raises(ValueError, match="immutable revision"):
        ModelRunConfig(
            "published:9b",
            entry_class="field",
            source_url="https://example.test/model-card",
            source_revision="main",
            license_id="Apache-2.0",
        )


def test_unverified_local_model_cannot_compile_as_independent_benchmark_evidence(
    tmp_path,
):
    plan = FieldPlan(
        models=(ModelRunConfig("anonymous:9b"),),
        datasets=(DatasetSpec("synthetic", n_symbols=1, n_days=12),),
        seeds=(1,),
        repetitions=1,
        max_steps=3,
    )
    journal = tmp_path / "unverified.jsonl"
    LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(journal))
    with pytest.raises(BenchBridgeError, match="entry_class=field"):
        compile_benchmark_evidence([journal], tmp_path / "compiled")


def test_bench_bridge_compiles_complete_shards_and_preserves_frequency(tmp_path):
    plan = _plan()
    journal = tmp_path / "field.jsonl"
    assert (
        LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(journal))["failed"]
        == 0
    )
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    for record_index, record in enumerate(records):
        first = record_index * 3 + 1
        record["inference_duration_samples_ns"] = [first, first + 1, first + 2]
        record["inference_duration_ns"] = sum(
            record["inference_duration_samples_ns"]
        )
        record["inference_duration_source"] = "fixture-monotonic"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = compile_benchmark_evidence([journal], tmp_path / "compiled")
    assert result["field_shape"]["repetitions"] == 2
    output = result["outputs"][0]
    assert output["periods_per_year"] == 252.0
    assert output["execution_seeds_per_window"] == 1
    assert output["score_command"][-5:] == [
        "--periods-per-year",
        "252",
        "--execution-seeds-per-window",
        "1",
        "--json",
    ]
    submissions = json.loads(Path(output["submissions_path"]).read_text())
    assert len(submissions) == 1
    assert len(submissions[0]["runs"]) == 4
    assert submissions[0]["in_sample_trials"] == 1
    assert submissions[0]["candidates"] == []
    assert submissions[0]["runs"][0]["trace"]["events"]
    profile = output["models"][0]["operational_profile"]
    assert profile == {
        "rank_input": False,
        "latency_definition": "one model request, nearest-rank percentile",
        "inference_calls": 12,
        "inference_duration_ns_total": 78,
        "inference_duration_ns_p50": 6,
        "inference_duration_ns_p95": 12,
        "duration_sources": ["fixture-monotonic"],
        "tokens_in_total": 120,
        "tokens_out_total": 60,
        "reasoning_tokens_total": 0,
        "reasoning_token_sources": ["unavailable"],
        "retry_count_total": 0,
        "cells": 4,
    }


def test_bench_bridge_refuses_inconsistent_inference_accounting(tmp_path):
    journal = tmp_path / "field.jsonl"
    LocalFieldRunner(FixedModel()).run(_plan(repetitions=1), EvidenceJournal(journal))
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[0]["inference_duration_samples_ns"][0] += 1
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(BenchBridgeError, match="do not sum"):
        compile_benchmark_evidence([journal], tmp_path / "compiled")


def test_bench_bridge_preserves_disclosed_selection_candidates(tmp_path):
    candidate_returns = ((0.01, -0.005, 0.02), (0.002, 0.003, -0.001))
    plan = FieldPlan(
        models=(
                ModelRunConfig(
                    "test-fixture:synthetic",
                    precommitted_n_trials=2,
                    selection_candidates=candidate_returns,
                    entry_class="field",
                    source_url="https://example.test/models/test-fixture",
                    source_revision="0123456789abcdef",
                    license_id="MIT",
                ),
        ),
        datasets=(DatasetSpec("synthetic-calm", n_symbols=2, n_days=12),),
        seeds=(1,),
        repetitions=1,
        max_steps=5,
    )
    journal = tmp_path / "selection.jsonl"
    LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(journal))
    result = compile_benchmark_evidence([journal], tmp_path / "compiled-selection")
    submissions = json.loads(
        Path(result["outputs"][0]["submissions_path"]).read_text()
    )
    assert submissions[0]["in_sample_trials"] == 2
    assert submissions[0]["candidates"] == [list(series) for series in candidate_returns]


def test_bench_bridge_refuses_incomplete_or_failed_fields(tmp_path):
    complete = tmp_path / "complete.jsonl"
    LocalFieldRunner(FixedModel()).run(_plan(), EvidenceJournal(complete))
    lines = complete.read_text(encoding="utf-8").splitlines()
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(BenchBridgeError, match="incomplete field"):
        compile_benchmark_evidence([incomplete], tmp_path / "compiled")

    failed = tmp_path / "failed-field.jsonl"
    LocalFieldRunner(BrokenModel()).run(_plan(repetitions=1), EvidenceJournal(failed))
    with pytest.raises(BenchBridgeError, match="failed/incomplete"):
        compile_benchmark_evidence([failed], tmp_path / "compiled-failed")


def test_bench_bridge_accepts_a_failed_attempt_followed_by_one_completion(tmp_path):
    journal = tmp_path / "resumed.jsonl"
    plan = _plan(repetitions=1)
    LocalFieldRunner(BrokenModel()).run(plan, EvidenceJournal(journal))
    LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(journal))
    assert len(journal.read_text().splitlines()) == 4
    result = compile_benchmark_evidence([journal], tmp_path / "compiled")
    assert result["field_shape"]["total_cells"] == 2
