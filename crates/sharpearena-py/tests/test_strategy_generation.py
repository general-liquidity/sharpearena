"""Strategy-generation protocol tests; no model server or network required."""

from __future__ import annotations

import json

import pytest
from sharpearena.local_agents import DatasetSpec, ModelIdentity, ModelRunConfig
from sharpearena.strategy_cli import load_strategy_plan
from sharpearena.strategy_cli import main as strategy_cli_main
from sharpearena.strategy_generation import (
    MAX_GENERATED_CANDIDATES,
    STRATEGY_GENERATION_SCHEMA,
    GenerationResult,
    StrategyProtocolError,
    StrategySearchPlan,
    StrategySearchRunner,
    evaluate_condition,
    parse_generated_pool,
    strategy_decision,
)


def _condition(op="gt"):
    return {
        "op": op,
        "left": {"indicator": "momentum", "window": 3},
        "right": {"constant": 0.0},
    }


def _response():
    return json.dumps(
        {
            "strategies": [
                {
                    "id": "trend",
                    "thesis": "follow three-bar momentum",
                    "long_when": _condition("gt"),
                    "short_when": _condition("lt"),
                    "gross_target": 0.8,
                },
                {
                    "id": "contrarian",
                    "thesis": "fade three-bar momentum",
                    "long_when": _condition("lt"),
                    "short_when": _condition("gt"),
                    "gross_target": 0.5,
                },
                {
                    "id": "trend-copy",
                    "thesis": "duplicate must still count as a trial",
                    "long_when": _condition("gt"),
                    "short_when": _condition("lt"),
                    "gross_target": 0.8,
                },
                {
                    "id": "unsafe",
                    "thesis": "attempt an executable escape",
                    "long_when": _condition("gt"),
                    "gross_target": 0.5,
                    "python": "import os",
                },
            ]
        }
    )


class FixtureGenerator:
    def identity(self, model):
        return ModelIdentity(
            model=model.model,
            digest="sha256:generator",
            parameter_size="1B",
            quantization="fixture",
            family="fixture",
            server="fixture",
            server_version="1",
        )

    def generate(self, model, prompt, requested_candidates):
        return GenerationResult(_response(), prompt_tokens=20, output_tokens=40)


def test_generation_schema_has_no_executable_code_surface():
    text = json.dumps(STRATEGY_GENERATION_SCHEMA, sort_keys=True)
    assert '"python"' not in text
    assert '"command"' not in text
    assert '"tool"' not in text
    assert set(STRATEGY_GENERATION_SCHEMA["$defs"]) == {"value", "condition"}


def test_observed_trial_count_precedes_validation_and_deduplication():
    observed, accepted, rejected = parse_generated_pool(_response())
    assert observed == 4
    assert [candidate.candidate_id for candidate in accepted] == ["trend", "contrarian"]
    assert [item.candidate_id for item in rejected] == ["trend-copy", "unsafe"]
    assert "duplicate strategy fingerprint" in rejected[0].reason
    assert "unknown fields" in rejected[1].reason


def test_malformed_generation_fails_instead_of_inventing_a_trial_count():
    with pytest.raises(StrategyProtocolError, match="not JSON"):
        parse_generated_pool("not-json")
    with pytest.raises(StrategyProtocolError, match="must not be empty"):
        parse_generated_pool('{"strategies": []}')
    oversized = {"strategies": [{}] * (MAX_GENERATED_CANDIDATES + 1)}
    with pytest.raises(StrategyProtocolError, match="hard cap"):
        parse_generated_pool(json.dumps(oversized))


def test_dsl_uses_trailing_values_and_normalizes_gross_exposure():
    _, candidates, _ = parse_generated_pool(_response())
    trend = candidates[0]
    assert evaluate_condition(trend.long_when, [100.0, 101.0, 102.0])
    assert not evaluate_condition(trend.long_when, [100.0, 99.0, 98.0])
    observation = {
        "symbols": [
            {"symbol": "AAA", "close_history": [100.0, 101.0, 102.0]},
            {"symbol": "BBB", "close_history": [100.0, 102.0, 103.0]},
        ]
    }
    decision = strategy_decision(trend, observation)
    assert sum(
        abs(order["target_weight"]) for order in decision["orders"]
    ) == pytest.approx(0.8)


def test_search_selects_on_validation_and_tests_only_the_winner(tmp_path):
    plan = StrategySearchPlan(
        model=ModelRunConfig("test-fixture:synthetic"),
        prompt="Generate a small, interpretable family.",
        requested_candidates=4,
        validation_dataset=DatasetSpec(
            "validation", tier="calm", n_symbols=2, n_days=16
        ),
        test_dataset=DatasetSpec("test", tier="hard", n_symbols=2, n_days=16),
        validation_seeds=(1, 2),
        test_seeds=(101, 102),
        max_steps=8,
    )
    path = tmp_path / "strategy-evidence.json"
    evidence = StrategySearchRunner(FixtureGenerator()).run(plan, path)
    assert json.loads(path.read_text()) == evidence
    assert evidence["generation"]["observed_n_trials"] == 4
    assert (
        evidence["generation"]["n_trials_source"] == "host-counted-generation-response"
    )
    assert set(evidence["selection"]["scores"]) == {"trend", "contrarian"}
    assert len(evidence["test"]["scores"]) == 2
    assert evidence["test"]["selected_candidate_only"] is True
    assert evidence["generated_code_executed"] is False
    assert evidence["generation"]["raw_response"] == _response()
    assert evidence["generation"]["prompt"] == plan.prompt


def test_search_refuses_overlapping_validation_and_test_splits():
    dataset = DatasetSpec("same", tier="calm")
    with pytest.raises(ValueError, match="must be explicit and disjoint"):
        StrategySearchPlan(
            model=ModelRunConfig("test-fixture:synthetic"),
            prompt="x",
            requested_candidates=1,
            validation_dataset=dataset,
            test_dataset=dataset,
            validation_seeds=(1,),
            test_seeds=(2,),
        )
    with pytest.raises(ValueError, match="seeds must be disjoint"):
        StrategySearchPlan(
            model=ModelRunConfig("test-fixture:synthetic"),
            prompt="x",
            requested_candidates=1,
            validation_dataset=DatasetSpec("validation", tier="calm"),
            test_dataset=DatasetSpec("test", tier="hard"),
            validation_seeds=(1,),
            test_seeds=(1,),
        )

    csv_text = "date,symbol,close\n2026-01-01,AAA,100\n"
    with pytest.raises(ValueError, match="must be explicit and disjoint"):
        StrategySearchPlan(
            model=ModelRunConfig("test-fixture:synthetic"),
            prompt="x",
            requested_candidates=1,
            validation_dataset=DatasetSpec("validation", csv_text=csv_text),
            test_dataset=DatasetSpec("test", csv_text=csv_text),
            validation_seeds=(1,),
            test_seeds=(2,),
        )
    with pytest.raises(ValueError, match="datasets/windows must be disjoint"):
        StrategySearchPlan(
            model=ModelRunConfig("test-fixture:synthetic"),
            prompt="x",
            requested_candidates=1,
            validation_dataset=DatasetSpec(
                "validation", csv_text=csv_text, window_start=0, window_end=10
            ),
            test_dataset=DatasetSpec(
                "test", csv_text=csv_text, window_start=9, window_end=20
            ),
            validation_seeds=(1,),
            test_seeds=(2,),
        )
    disjoint = StrategySearchPlan(
        model=ModelRunConfig("test-fixture:synthetic"),
        prompt="x",
        requested_candidates=1,
        validation_dataset=DatasetSpec(
            "validation", csv_text=csv_text, window_start=0, window_end=10
        ),
        test_dataset=DatasetSpec(
            "test", csv_text=csv_text, window_start=10, window_end=20
        ),
        validation_seeds=(1,),
        test_seeds=(2,),
    )
    assert disjoint.plan_sha256


def test_search_caps_requested_trials_and_persists_protocol_failure(tmp_path):
    with pytest.raises(ValueError, match="must lie"):
        StrategySearchPlan(
            model=ModelRunConfig("test-fixture:synthetic"),
            prompt="x",
            requested_candidates=MAX_GENERATED_CANDIDATES + 1,
            validation_dataset=DatasetSpec("validation", tier="calm"),
            test_dataset=DatasetSpec("test", tier="hard"),
            validation_seeds=(1,),
            test_seeds=(2,),
        )

    class BadGenerator(FixtureGenerator):
        def generate(self, model, prompt, requested_candidates):
            return GenerationResult("not-json")

    plan = StrategySearchPlan(
        model=ModelRunConfig("test-fixture:synthetic"),
        prompt="x",
        requested_candidates=1,
        validation_dataset=DatasetSpec("validation", tier="calm"),
        test_dataset=DatasetSpec("test", tier="hard"),
        validation_seeds=(1,),
        test_seeds=(2,),
    )
    path = tmp_path / "failed.json"
    with pytest.raises(StrategyProtocolError, match="not JSON"):
        StrategySearchRunner(BadGenerator()).run(plan, path)
    failure = json.loads(path.read_text())
    assert failure["status"] == "failed"
    assert failure["generation"]["raw_response_sha256"]
    assert failure["failure"]["type"] == "StrategyProtocolError"


def test_strategy_plan_rejects_empty_prompt_duplicate_seed_and_bad_step_budget():
    common = {
        "model": ModelRunConfig("test-fixture:synthetic"),
        "requested_candidates": 1,
        "validation_dataset": DatasetSpec("validation", tier="calm"),
        "test_dataset": DatasetSpec("test", tier="hard"),
        "test_seeds": (2,),
    }
    with pytest.raises(ValueError, match="prompt"):
        StrategySearchPlan(prompt="  ", validation_seeds=(1,), **common)
    with pytest.raises(ValueError, match="unique"):
        StrategySearchPlan(prompt="x", validation_seeds=(1, 1), **common)
    with pytest.raises(ValueError, match="max_steps"):
        StrategySearchPlan(prompt="x", validation_seeds=(1,), max_steps=0, **common)


def test_strategy_cli_resolves_paths_and_inspects_without_inference(tmp_path, capsys):
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,symbol,close\n2026-01-01,AAA,100\n2026-01-02,AAA,101\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "strategy.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": {"model": "test-fixture:synthetic"},
                "prompt": "Generate one strategy.",
                "requested_candidates": 1,
                "validation_dataset": {
                    "dataset_id": "validation",
                    "csv_path": "prices.csv",
                    "window_start": 0,
                    "window_end": 1,
                },
                "test_dataset": {
                    "dataset_id": "test",
                    "csv_path": "prices.csv",
                    "window_start": 1,
                    "window_end": 2,
                },
                "validation_seeds": [1],
                "test_seeds": [2],
            }
        ),
        encoding="utf-8",
    )
    plan = load_strategy_plan(plan_path)
    assert plan.validation_dataset.csv_text == csv_path.read_text(encoding="utf-8")
    assert (
        strategy_cli_main(
            [
                "--plan",
                str(plan_path),
                "--evidence",
                str(tmp_path / "unused.json"),
                "--inspect",
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["plan_sha256"] == plan.plan_sha256

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["typo"] = True
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_strategy_plan(plan_path)
