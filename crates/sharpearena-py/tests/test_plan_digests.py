"""What the three ``plan_sha256`` digests actually bind.

Every existing assertion on these digests was either a truthiness check or a comparison
of the property against itself, so dropping a field from the hashed payload would have
passed the whole suite. Each test below changes one field at a time and requires the
digest to move, which is the property the evidence chain depends on: two runs that
differ in any precommitted term cannot share a plan hash.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from sharpearena.local_agents import (
    DatasetSpec,
    FieldPlan,
    ModelRunConfig,
    SamplingConfig,
)
from sharpearena.paper_cli import PaperExecutionPlan
from sharpearena.paper_trading import PaperAccount, PaperRiskConfig
from sharpearena.strategy_generation import StrategySearchPlan


CSV = "date,symbol,close\n" + "\n".join(
    f"2025-01-{day:02d},AAA,{100 + day}" for day in range(1, 25)
)


def _strategy_plan(**overrides) -> StrategySearchPlan:
    values = {
        "model": ModelRunConfig("test-fixture:synthetic"),
        "prompt": "propose momentum candidates",
        "requested_candidates": 2,
        "validation_dataset": DatasetSpec(
            "validation", csv_text=CSV, window_start=0, window_end=10
        ),
        "test_dataset": DatasetSpec("test", csv_text=CSV, window_start=10, window_end=20),
        "validation_seeds": (1, 2),
        "test_seeds": (3, 4),
        "max_steps": 8,
    }
    values.update(overrides)
    return StrategySearchPlan(**values)


def _field_plan(**overrides) -> FieldPlan:
    values = {
        "models": (ModelRunConfig("test-fixture:synthetic"),),
        "datasets": (DatasetSpec("fixture", csv_text=CSV),),
        "seeds": (1, 2),
        "repetitions": 2,
        "max_steps": 8,
        "parallel_requests": 2,
    }
    values.update(overrides)
    return FieldPlan(**values)


def _execution_plan(**overrides) -> PaperExecutionPlan:
    values = {
        "agent_id": "fixture-agent",
        "model_digest": "sha256:fixture",
        "symbols": ("BTCUSDT",),
        "market_data": {"adapter": "in-memory"},
        "broker": {"adapter": "in-memory"},
        "account": PaperAccount(
            cash=10_000.0,
            equity=10_000.0,
            session_start_equity=10_000.0,
            peak_equity=10_000.0,
            positions={},
        ),
        "risk": PaperRiskConfig(
            allowed_symbols=("BTCUSDT",), max_order_notional=1_000.0
        ),
    }
    values.update(overrides)
    return PaperExecutionPlan(**values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", ModelRunConfig("test-fixture:other")),
        ("prompt", "propose mean-reversion candidates"),
        ("requested_candidates", 3),
        ("validation_dataset", DatasetSpec("validation", csv_text=CSV, window_start=1, window_end=10)),
        ("test_dataset", DatasetSpec("test", csv_text=CSV, window_start=10, window_end=19)),
        ("validation_seeds", (1, 5)),
        ("test_seeds", (3, 9)),
        ("max_steps", 9),
    ],
)
def test_strategy_plan_digest_moves_with_every_field_it_binds(field, value) -> None:
    baseline = _strategy_plan()
    assert baseline.plan_sha256 != _strategy_plan(**{field: value}).plan_sha256


def test_strategy_plan_digest_is_stable_across_equal_plans() -> None:
    assert _strategy_plan().plan_sha256 == _strategy_plan().plan_sha256
    assert len(_strategy_plan().plan_sha256) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("models", (ModelRunConfig("test-fixture:synthetic", sampling=SamplingConfig(seed=99)),)),
        ("datasets", (DatasetSpec("fixture", csv_text=CSV + "\n2025-02-01,AAA,150"),)),
        ("seeds", (1, 3)),
        ("repetitions", 3),
        ("max_steps", 9),
        ("parallel_requests", 3),
    ],
)
def test_field_plan_digest_moves_with_every_field_it_binds(field, value) -> None:
    baseline = _field_plan()
    assert baseline.plan_sha256 != _field_plan(**{field: value}).plan_sha256


def test_field_plan_digest_ignores_the_shard_it_was_asked_for() -> None:
    """Sharding splits one field across machines; it is not a different field."""
    baseline = _field_plan(shard_index=0, shard_count=2)
    assert baseline.plan_sha256 == _field_plan(shard_index=1, shard_count=2).plan_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_id", "other-agent"),
        ("model_digest", "sha256:other"),
        ("symbols", ("ETHUSDT",)),
        ("market_data", {"adapter": "alpaca"}),
        ("broker", {"adapter": "alpaca-paper"}),
        (
            "account",
            PaperAccount(
                cash=9_000.0,
                equity=10_000.0,
                session_start_equity=10_000.0,
                peak_equity=10_000.0,
                positions={},
            ),
        ),
        (
            "risk",
            PaperRiskConfig(allowed_symbols=("BTCUSDT",), max_order_notional=999.0),
        ),
    ],
)
def test_execution_plan_digest_moves_with_every_field_it_binds(field, value) -> None:
    baseline = _execution_plan()
    assert baseline.plan_sha256 != _execution_plan(**{field: value}).plan_sha256


def test_execution_plan_digest_survives_a_dataclass_round_trip() -> None:
    plan = _execution_plan()
    assert replace(plan).plan_sha256 == plan.plan_sha256
