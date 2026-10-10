"""MCP decision mapping uses the same fail-closed semantics as every other adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sharpearena.decision_parser import DecisionParseError
from sharpearena.mcp_server import _decision_to_weights


@pytest.fixture
def tools(monkeypatch):
    import sharpearena.mcp_server as module

    class LocalTools:
        def __init__(self, name):
            self.functions = {}

        def tool(self):
            def register(function):
                self.functions[function.__name__] = function
                return function

            return register

    monkeypatch.setattr(module, "FastMCP", LocalTools)
    monkeypatch.setattr(module, "_HAS_MCP", True)
    return lambda **kwargs: module.build_server(kwargs).functions


def _order(weight, action="buy"):
    return json.dumps(
        {"orders": [{"symbol": "SYM00", "action": action, "target_weight": weight}]}
    )


def test_mcp_uses_the_configured_weight_bound_in_both_directions(tools):
    small = tools(n_symbols=1, n_days=12, max_weight=0.5)
    small["reset"]()
    refused = json.loads(small["step"](_order(0.75)))
    assert refused["error"] == "invalid_decision"
    assert refused["environment_advanced"] is False
    large = tools(n_symbols=1, n_days=12, max_weight=2.0)
    large["reset"]()
    assert "error" not in json.loads(large["step"](_order(1.25)))


def test_mcp_shorting_refusal_is_structured_and_does_not_advance(tools):
    session = tools(n_symbols=1, n_days=12, allow_short=False)
    session["reset"]()
    refused = json.loads(session["step"](_order(-0.25, "sell")))
    assert refused["error"] == "invalid_decision"
    assert refused["reason_code"] == "out_of_bounds"
    assert refused["environment_advanced"] is False
    actual = json.loads(session["step"](_order(0.0, "hold")))
    control = tools(n_symbols=1, n_days=12, allow_short=False)
    control["reset"]()
    assert actual == json.loads(control["step"](_order(0.0, "hold")))


def test_mcp_session_contract_and_reset_after_done(tools):
    session = tools(n_symbols=1, n_days=3)
    spec = json.loads(session["spec"]())
    assert spec["session_contract"]["scope"] == "single_episode_per_server"
    assert spec["session_contract"]["shared_by_callers"] is True
    assert json.loads(session["step"](_order(0.0)))["error"] == "episode_not_reset"
    first = session["reset"]()
    while True:
        result = json.loads(session["step"](_order(0.0)))
        if result["terminated"] or result["truncated"]:
            break
    assert json.loads(session["step"](_order(0.0)))["error"] == "episode_done"
    assert session["reset"]() == first


def test_mcp_does_not_relabel_internal_failures_as_action_refusals(tools, monkeypatch):
    from sharpearena.gym import SharpeArenaEnv
    from sharpearena.sharpearena_py import EngineFailure

    session = tools(n_symbols=1, n_days=12)
    session["reset"]()

    def fail(self, weights):
        raise EngineFailure("[ENGINE_FAILURE] synthetic serialization failure")

    monkeypatch.setattr(SharpeArenaEnv, "step", fail)
    with pytest.raises(EngineFailure):
        session["step"](_order(0.0))


@pytest.mark.parametrize("error_type", ["InvalidArgument", "InvalidJson"])
def test_mcp_native_input_refusals_keep_their_type(tools, monkeypatch, error_type):
    from sharpearena import sharpearena_py
    from sharpearena.gym import SharpeArenaEnv

    session = tools(n_symbols=1, n_days=12)
    session["reset"]()

    def fail(self, weights):
        raise getattr(sharpearena_py, error_type)("synthetic input refusal")

    monkeypatch.setattr(SharpeArenaEnv, "step", fail)
    result = json.loads(session["step"](_order(0.0)))
    assert result["error"] == "invalid_decision"
    assert result["reason_code"] == error_type
    assert result["environment_advanced"] is False


@pytest.mark.parametrize("action", [[], {}])
def test_mcp_unhashable_action_is_a_named_parser_refusal(tools, action):
    session = tools(n_symbols=1, n_days=12)
    session["reset"]()
    result = json.loads(session["step"](_order(0.0, action)))
    assert result["error"] == "invalid_decision"
    assert result["environment_advanced"] is False


def test_mcp_unrepresentable_integer_is_a_named_parser_refusal(tools):
    session = tools(n_symbols=1, n_days=12)
    session["reset"]()
    result = json.loads(session["step"](_order(10**400)))
    assert result["error"] == "invalid_decision"
    assert result["environment_advanced"] is False


def _observation():
    return {
        "positions": np.array([0.001, -0.002]),
        "closes": np.array([100.0, 50.0]),
        "cash": np.array([1.0]),
    }


def test_mcp_mapping_accepts_one_canonical_order_per_observed_symbol():
    weights = _decision_to_weights(
        json.dumps(
            {
                "orders": [
                    {"symbol": "AAA", "action": "buy", "target_weight": 0.2},
                    {"symbol": "BBB", "action": "sell", "target_weight": -0.1},
                ]
            }
        ),
        ["AAA", "BBB"],
        _observation(),
    )
    np.testing.assert_array_equal(weights, np.array([0.2, -0.1]))


@pytest.mark.parametrize(
    "orders, message",
    [
        (
            [{"symbol": "ZZZ", "action": "buy", "target_weight": 0.2}],
            "was not observed",
        ),
        (
            [
                {"symbol": "AAA", "action": "buy", "target_weight": 0.2},
                {"symbol": "AAA", "action": "buy", "target_weight": 0.3},
            ],
            "duplicate order",
        ),
        (
            [{"symbol": "AAA", "action": "buy", "target_weight": 1.2}],
            "exceeds 1.0",
        ),
    ],
)
def test_mcp_mapping_refuses_unknown_duplicate_or_out_of_range_orders(orders, message):
    with pytest.raises(DecisionParseError, match=message):
        _decision_to_weights(
            json.dumps({"orders": orders}), ["AAA", "BBB"], _observation()
        )


def test_mcp_empty_orders_and_omitted_symbols_preserve_current_weights():
    current = _observation()
    held = _decision_to_weights('{"orders": []}', ["AAA", "BBB"], current)
    np.testing.assert_allclose(held, np.array([0.1, -0.1]))
    partial = _decision_to_weights(
        '{"orders":[{"symbol":"AAA","action":"buy","target_weight":0.2}]}',
        ["AAA", "BBB"],
        current,
    )
    np.testing.assert_allclose(partial, np.array([0.2, -0.1]))
