"""Strategy-generation protocol tests; no model server or network required."""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

import pytest
import sharpearena.strategy_generation as strategy_generation
from jsonschema import Draft202012Validator
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


def _canonical_sha256(value) -> str:
    """The canonical-JSON digest, reimplemented here rather than imported.

    Importing the module's own helper would make every digest assertion below compare a
    value against itself.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _condition(op="gt"):
    return {
        "op": op,
        "left": {"indicator": "momentum", "window": 3},
        "right": {"constant": 0.0},
    }


def _edge_manifest(hypothesis="Three-bar momentum persists after costs."):
    return {
        "hypothesis": hypothesis,
        "mechanism": "Short-horizon continuation reflects gradual information diffusion.",
        "regimes": ["trending"],
        "instruments": ["synthetic_panel"],
        "invariants": [
            {
                "condition_id": "net-edge-positive",
                "metric": "net_edge",
                "comparator": "gt",
                "threshold": {"value": 0.0, "unit": "basis_points"},
                "description": "The edge remains positive after costs.",
            }
        ],
        "kill_conditions": [
            {
                "condition_id": "drawdown-breach",
                "metric": "drawdown",
                "comparator": "gt",
                "threshold": {"value": 20.0, "unit": "percent"},
                "description": "Retire after a twenty-percent drawdown.",
            }
        ],
        "verification_plan": {
            "selection_metric": "deflated_sharpe",
            "selection_split": "validation",
            "confirmation_split": "test",
            "minimum_observations": 8,
        },
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
                    "edge_manifest": _edge_manifest(),
                },
                {
                    "id": "contrarian",
                    "thesis": "fade three-bar momentum",
                    "long_when": _condition("lt"),
                    "short_when": _condition("gt"),
                    "gross_target": 0.5,
                    "edge_manifest": _edge_manifest(
                        "Three-bar momentum mean-reverts after crowded moves."
                    ),
                },
                {
                    "id": "trend-copy",
                    "thesis": "duplicate must still count as a trial",
                    "long_when": _condition("gt"),
                    "short_when": _condition("lt"),
                    "gross_target": 0.8,
                    "edge_manifest": _edge_manifest(),
                },
                {
                    "id": "unsafe",
                    "thesis": "attempt an executable escape",
                    "long_when": _condition("gt"),
                    "gross_target": 0.5,
                    "edge_manifest": _edge_manifest(),
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
    assert set(STRATEGY_GENERATION_SCHEMA["$defs"]) == {
        "value",
        "condition",
        "threshold",
        "edge_condition",
        "verification_plan",
        "edge_manifest",
    }
    item = STRATEGY_GENERATION_SCHEMA["properties"]["strategies"]["items"]
    assert "edge_manifest" in item["required"]

    Draft202012Validator.check_schema(STRATEGY_GENERATION_SCHEMA)
    validator = Draft202012Validator(STRATEGY_GENERATION_SCHEMA)
    payload = json.loads(_response())
    accepted = payload["strategies"][:2]
    assert all(
        not list(validator.iter_errors({"strategies": [candidate]}))
        for candidate in accepted
    )
    assert list(
        validator.iter_errors({"strategies": [payload["strategies"][3]]})
    )


def test_observed_trial_count_precedes_validation_and_deduplication():
    observed, accepted, rejected = parse_generated_pool(_response())
    assert observed == 4
    assert [candidate.candidate_id for candidate in accepted] == ["trend", "contrarian"]
    assert [item.candidate_id for item in rejected] == ["trend-copy", "unsafe"]
    assert "duplicate strategy and edge manifest" in rejected[0].reason
    assert "unknown fields" in rejected[1].reason


def test_missing_or_invalid_manifest_is_counted_and_refused_before_selection():
    payload = json.loads(_response())
    payload["strategies"] = payload["strategies"][:2]
    payload["strategies"][0].pop("edge_manifest")
    payload["strategies"][1]["edge_manifest"].pop("mechanism")
    observed, accepted, rejected = parse_generated_pool(json.dumps(payload))
    assert observed == 2
    assert accepted == []
    assert [item.index for item in rejected] == [0, 1]
    assert "required 'edge_manifest'" in rejected[0].reason
    assert "mechanism" in rejected[1].reason


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


def test_every_dsl_indicator_and_boolean_operator_has_numeric_semantics():
    """Each indicator is pinned to its exact value through the public evaluator.

    ``gte`` and ``lte`` both holding at one constant is equality, and ``gt`` failing at
    the same constant is the false side. One-sided inequalities let a wrong
    implementation through: a volatility returning variance, or a population figure where
    a sample one was meant, clears a `> 0.01` threshold just as well as the right answer.
    """

    def comparison(indicator, threshold, *, window=None, op="gt"):
        value = {"indicator": indicator}
        if window is not None:
            value["window"] = window
        return {
            "op": op,
            "left": value,
            "right": {"constant": threshold},
        }

    def pin(indicator, expected, prices, *, window=None):
        assert evaluate_condition(
            comparison(indicator, expected, window=window, op="gte"), prices
        ), f"{indicator} is below {expected!r}"
        assert evaluate_condition(
            comparison(indicator, expected, window=window, op="lte"), prices
        ), f"{indicator} is above {expected!r}"
        assert not evaluate_condition(
            comparison(indicator, expected, window=window, op="gt"), prices
        ), f"{indicator} is strictly above its own value"
        assert not evaluate_condition(
            comparison(indicator, expected, window=window, op="lt"), prices
        ), f"{indicator} is strictly below its own value"

    geometric = [1.0, 2.0, 4.0, 8.0]
    pin("price", 8.0, geometric)
    pin("sma", 14.0 / 3.0, geometric, window=3)
    # alpha = 2/(3+1); seeded at the window's first value: 2 -> 3 -> 5.5.
    pin("ema", 5.5, geometric, window=3)
    pin("momentum", 3.0, geometric, window=3)
    # Every change is a gain, so the loss average is zero and RSI saturates.
    pin("rsi", 100.0, geometric, window=4)
    pin("rsi", 0.0, [8.0, 4.0, 2.0, 1.0], window=4)
    # Equal average gain and loss is the midpoint, not saturation.
    pin("rsi", 50.0, [1.0, 2.0, 1.0], window=3)
    # Population standard deviation of [0.1, 0.0, 0.1], not the variance (0.00222...),
    # not the sample deviation (0.057735...), and not an annualized figure.
    pin("volatility", 0.04714045207910321, [100.0, 110.0, 110.0, 121.0], window=4)

    rising = comparison("momentum", 0.0, window=3)
    falling = comparison("momentum", 0.0, window=3, op="lt")
    prices = [100.0, 101.0, 102.0]
    assert evaluate_condition(
        {"op": "and", "conditions": [rising, {"op": "not", "condition": falling}]},
        prices,
    )
    assert not evaluate_condition(
        {"op": "and", "conditions": [rising, falling]}, prices
    )
    assert evaluate_condition(
        {"op": "or", "conditions": [falling, rising]}, prices
    )
    assert not evaluate_condition(
        {"op": "or", "conditions": [falling, {"op": "not", "condition": rising}]},
        prices,
    )


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
        evidence["generation"]["n_trials_source"]
        == "ledger-counted-before-validation-and-deduplication"
    )
    ledger = evidence["generation"]["edge_manifest_ledger"]
    assert ledger["summary"]["observed_trials"] == 4
    assert [row["trial_ordinal"] for row in ledger["records"]] == [0, 1, 2, 3]
    assert all(row["model_digest"] == "sha256:generator" for row in ledger["records"])
    assert all(row["split_plan_sha256"] == plan.plan_sha256 for row in ledger["records"])
    ledger_rows = {row["trial_ordinal"]: row for row in ledger["records"]}
    for item in evidence["generation"]["accepted"]:
        row = ledger_rows[item["trial_ordinal"]]
        assert item["binding_sha256"] == _canonical_sha256(
            {
                "raw_candidate_sha256": row["raw_candidate_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "model_digest": row["model_digest"],
                "split_plan_sha256": row["split_plan_sha256"],
                "trial_ordinal": row["trial_ordinal"],
            }
        )
    assert set(evidence["selection"]["scores"]) == {"trend", "contrarian"}
    assert len(evidence["test"]["scores"]) == 2
    assert evidence["test"]["selected_candidate_only"] is True
    # The flag itself is a literal in the evidence writer; what makes it true is pinned
    # by test_generated_text_is_never_executed_by_the_dsl_modules below.
    assert evidence["generated_code_executed"] is False
    assert evidence["generation"]["raw_response"] == _response()
    assert evidence["generation"]["prompt"] == plan.prompt

    second = StrategySearchRunner(FixtureGenerator()).run(plan, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["generation"]["raw_response_sha256"] == rows[1]["generation"][
        "raw_response_sha256"
    ]
    assert second["status"] == "completed"


def test_selection_uses_descending_median_and_candidate_id_tie_break(
    tmp_path, monkeypatch
):
    calls = []

    def fixture_scores(candidates, dataset, seeds, n_trials, max_steps):
        calls.append([candidate.candidate_id for candidate in candidates])
        if len(candidates) == 1:
            return {
                candidates[0].candidate_id: [
                    {"score": {"deflated_sharpe": 0.1}, "seed": seeds[0]}
                ]
            }
        return {
            candidate.candidate_id: [
                {"score": {"deflated_sharpe": 0.7}, "seed": seed}
                for seed in seeds
            ]
            for candidate in candidates
        }

    monkeypatch.setattr(strategy_generation, "_evaluate_candidates", fixture_scores)
    plan = StrategySearchPlan(
        model=ModelRunConfig("test-fixture:synthetic"),
        prompt="Generate a small, interpretable family.",
        requested_candidates=4,
        validation_dataset=DatasetSpec("validation", n_days=16),
        test_dataset=DatasetSpec("test", tier="hard", n_days=16),
        validation_seeds=(1, 2),
        test_seeds=(101,),
        max_steps=8,
    )
    evidence = StrategySearchRunner(FixtureGenerator()).run(
        plan, tmp_path / "selection.jsonl"
    )
    assert evidence["selection"]["selected_candidate_id"] == "contrarian"
    assert calls == [["trend", "contrarian"], ["contrarian"]]


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
    # The digest binds the split, so moving the test window has to move it.
    shifted = replace(
        disjoint,
        test_dataset=DatasetSpec("test", csv_text=csv_text, window_start=10, window_end=19),
    )
    assert disjoint.plan_sha256 != shifted.plan_sha256


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
    assert failure["generation"]["raw_response_sha256"] == hashlib.sha256(b"not-json").hexdigest()
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


def test_generated_text_is_never_executed_by_the_dsl_modules():
    """What ``generated_code_executed: False`` claims, checked instead of read.

    The flag is written as a literal, so asserting it verifies a self-declaration. The
    substantive claim is that nothing on the path from model output to a decision can run
    generated text. Two halves: the modules on that path contain no execution primitive,
    and an indicator name that is a Python expression is refused rather than resolved.
    """

    import ast

    package = pathlib.Path(strategy_generation.__file__).parent
    forbidden_names = {"eval", "exec", "compile", "__import__"}
    forbidden_calls = {
        ("os", "system"),
        ("os", "popen"),
        ("os", "execv"),
        ("os", "execvp"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_output"),
        ("pickle", "loads"),
        ("marshal", "loads"),
        ("importlib", "import_module"),
        ("runpy", "run_path"),
    }
    forbidden_modules = {"subprocess", "pickle", "marshal", "runpy"}
    for module in ("strategy_generation.py", "edge_manifest.py", "strategy_cli.py"):
        tree = ast.parse((package / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name.split(".")[0] for alias in node.names}
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split(".")[0])
                assert not names & forbidden_modules, f"{module} imports {names}"
                continue
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                assert function.id not in forbidden_names, f"{module} calls {function.id}"
            if isinstance(function, ast.Attribute):
                owner = getattr(function.value, "id", "")
                # `re.compile` builds a regex; it runs nothing.
                if owner != "re":
                    assert (
                        function.attr not in forbidden_names
                    ), f"{module} calls {owner}.{function.attr}"
                assert (
                    owner,
                    function.attr,
                ) not in forbidden_calls, f"{module} calls {owner}.{function.attr}"

    # An indicator name is a key into a closed table, never something evaluated.
    observed, accepted, rejected = parse_generated_pool(
        json.dumps(
            {
                "strategies": [
                    {
                        "id": "escape",
                        "thesis": "resolve an indicator by executing it",
                        "long_when": {
                            "op": "gt",
                            "left": {
                                "indicator": "__import__('os').getcwd()",
                                "window": 3,
                            },
                            "right": {"constant": 0.0},
                        },
                        "gross_target": 0.5,
                        "edge_manifest": _edge_manifest(),
                    }
                ]
            }
        )
    )
    assert observed == 1
    assert accepted == []
    assert rejected[0].reason == "long_when.left.indicator is unsupported"
