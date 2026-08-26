"""MCP decision mapping uses the same fail-closed semantics as every other adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sharpearena.decision_parser import DecisionParseError
from sharpearena.mcp_server import _decision_to_weights


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
