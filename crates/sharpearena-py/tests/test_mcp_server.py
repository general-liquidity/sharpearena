"""MCP decision mapping uses the same fail-closed semantics as every other adapter."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sharpearena.decision_parser import DecisionParseError
from sharpearena.mcp_server import _decision_to_weights


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
        _decision_to_weights(json.dumps({"orders": orders}), ["AAA", "BBB"])
