"""Parse the one authoritative SharpeArena :class:`Decision` wire contract.

Model completions may be either a bare JSON object or the same object wrapped in the
``<action>...</action>`` envelope required by PrimeIntellect ``verifiers``. The JSON
inside that envelope is always the public ``Decision{orders, reasoning, cost}`` shape.
The former ``{"weights": ...}`` / ``{"flat": true}`` dialect is deliberately
rejected: maintaining two action protocols made training and evaluation incomparable.

Malformed output is fail-closed. :func:`parse_decision` raises
:class:`DecisionParseError`; callers must record the protocol failure or abort the
cell. It never fabricates an all-zero action, because doing so turns an agent fault
into an apparently conservative return series.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised only when verifiers is installed
    import verifiers as vf

    _HAS_VERIFIERS = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    vf = None  # type: ignore[assignment]
    _HAS_VERIFIERS = False

_FIELDS = ["reasoning", "action"]
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_ACTIONS = {"buy", "sell", "hold", "close"}
_DECISION_FIELDS = {"orders", "reasoning", "cost"}
_ORDER_FIELDS = {"symbol", "action", "target_weight", "confidence", "rationale"}
_COST_FIELDS = {"cost_usd", "tokens_in", "tokens_out", "reasoning_tokens"}

_PARSER: Any = None


class DecisionParseError(ValueError):
    """The model output is not a valid canonical ``Decision``."""


def build_parser():
    """The ``vf.XMLParser`` over ``<reasoning>``/``<action>`` (answer = action)."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build an XMLParser")
    global _PARSER
    if _PARSER is None:
        _PARSER = vf.XMLParser(fields=_FIELDS, answer_field="action")
    return _PARSER


def _extract_action_json(text: str) -> Optional[dict[str, Any]]:
    """Return the action object from XML or bare JSON, or ``None`` if undecodable."""
    if not text:
        return None
    match = _ACTION_RE.search(text)
    blob = (match.group(1) if match else text).strip()
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionParseError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DecisionParseError(f"{field} must be finite")
    return number


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecisionParseError(f"{field} must be a non-negative integer")
    return int(value)


def parse_decision_payload(text: str) -> dict[str, Any]:
    """Decode and structurally validate a canonical wire ``Decision``.

    Validation mirrors the published closed-object JSON schema without requiring the
    optional ``jsonschema`` package. Environment-specific checks (known symbols and
    target-weight bounds) are applied by :func:`parse_decision`.
    """
    payload = _extract_action_json(text)
    if payload is None:
        raise DecisionParseError("completion does not contain a JSON decision object")

    unknown = set(payload) - _DECISION_FIELDS
    if unknown:
        raise DecisionParseError(
            f"unknown decision field(s): {', '.join(sorted(unknown))}"
        )
    if "orders" not in payload or not isinstance(payload["orders"], list):
        raise DecisionParseError("decision.orders must be an array")
    if "reasoning" in payload and not isinstance(payload["reasoning"], str):
        raise DecisionParseError("decision.reasoning must be a string")

    canonical_orders: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["orders"]):
        path = f"decision.orders[{index}]"
        if not isinstance(raw, dict):
            raise DecisionParseError(f"{path} must be an object")
        unknown = set(raw) - _ORDER_FIELDS
        if unknown:
            raise DecisionParseError(
                f"unknown {path} field(s): {', '.join(sorted(unknown))}"
            )
        missing = {"symbol", "action", "target_weight"} - set(raw)
        if missing:
            raise DecisionParseError(f"{path} missing: {', '.join(sorted(missing))}")
        symbol = raw["symbol"]
        action = raw["action"]
        if not isinstance(symbol, str) or not symbol:
            raise DecisionParseError(f"{path}.symbol must be a non-empty string")
        if action not in _ACTIONS:
            raise DecisionParseError(f"{path}.action must be one of {sorted(_ACTIONS)}")
        target = _finite_number(raw["target_weight"], f"{path}.target_weight")
        # The target is the authoritative desired position; ``action`` is an
        # audit label, not the size or sign.  A sell can reduce a positive long
        # to a smaller positive target, a buy can cover a short while its target
        # remains negative, and a hold can preserve a non-zero position.  Those
        # relations need the current portfolio and cannot be inferred here.
        canonical_order: dict[str, Any] = {
            "symbol": symbol,
            "action": action,
            "target_weight": target,
        }
        if "confidence" in raw:
            confidence = _finite_number(raw["confidence"], f"{path}.confidence")
            if not 0.0 <= confidence <= 1.0:
                raise DecisionParseError(f"{path}.confidence must lie in [0, 1]")
            canonical_order["confidence"] = confidence
        if "rationale" in raw:
            rationale = raw["rationale"]
            if not isinstance(rationale, str):
                raise DecisionParseError(f"{path}.rationale must be a string")
            canonical_order["rationale"] = rationale
        canonical_orders.append(canonical_order)

    canonical: dict[str, Any] = {
        "orders": canonical_orders,
        "reasoning": payload.get("reasoning", ""),
    }
    if "cost" in payload:
        raw_cost = payload["cost"]
        if not isinstance(raw_cost, dict):
            raise DecisionParseError("decision.cost must be an object")
        unknown = set(raw_cost) - _COST_FIELDS
        if unknown:
            raise DecisionParseError(
                f"unknown decision.cost field(s): {', '.join(sorted(unknown))}"
            )
        cost_usd = _finite_number(
            raw_cost.get("cost_usd", 0.0), "decision.cost.cost_usd"
        )
        if cost_usd < 0.0:
            raise DecisionParseError("decision.cost.cost_usd must be non-negative")
        canonical["cost"] = {
            "cost_usd": cost_usd,
            "tokens_in": _nonnegative_int(
                raw_cost.get("tokens_in", 0), "decision.cost.tokens_in"
            ),
            "tokens_out": _nonnegative_int(
                raw_cost.get("tokens_out", 0), "decision.cost.tokens_out"
            ),
            "reasoning_tokens": _nonnegative_int(
                raw_cost.get("reasoning_tokens", 0), "decision.cost.reasoning_tokens"
            ),
        }
    return canonical


def decision_to_weights(
    decision: dict[str, Any],
    symbols: Sequence[str],
    *,
    current_weights: Sequence[float],
    max_abs_weight: float = 1.0,
) -> np.ndarray:
    """Map a validated ``Decision`` to a full target-weight vector.

    The wire contract is sparse: an omitted symbol is unchanged, and therefore
    an empty order list is a true hold.  The Gym action is dense, so it must
    begin from the current portfolio weights rather than an all-zero vector.
    """
    if not math.isfinite(max_abs_weight) or max_abs_weight <= 0.0:
        raise ValueError("max_abs_weight must be finite and positive")
    index = {symbol: i for i, symbol in enumerate(symbols)}
    if len(index) != len(symbols):
        raise ValueError("symbols must be unique")
    vector = np.asarray(current_weights, dtype=np.float64).reshape(-1).copy()
    if vector.shape != (len(symbols),):
        raise ValueError("current_weights must match the symbol axis")
    if not np.all(np.isfinite(vector)):
        raise ValueError("current_weights must be finite")
    seen: set[str] = set()
    for order_index, order in enumerate(decision["orders"]):
        symbol = order["symbol"]
        if symbol not in index:
            raise DecisionParseError(
                f"decision.orders[{order_index}].symbol {symbol!r} was not observed"
            )
        if symbol in seen:
            raise DecisionParseError(
                f"decision contains duplicate order for {symbol!r}"
            )
        seen.add(symbol)
        weight = float(order["target_weight"])
        if abs(weight) > max_abs_weight:
            raise DecisionParseError(
                f"target weight {weight} for {symbol!r} exceeds {max_abs_weight}"
            )
        vector[index[symbol]] = weight
    return vector


def portfolio_weights(
    positions: Sequence[float], closes: Sequence[float], cash: float
) -> np.ndarray:
    """Compute current signed weights from a Gym observation.

    ``positions`` are shares and ``closes`` are the point-in-time marks.  A
    non-positive NAV cannot be represented as portfolio weights and is refused
    rather than converted to a zero vector.
    """

    shares = np.asarray(positions, dtype=np.float64).reshape(-1)
    prices = np.asarray(closes, dtype=np.float64).reshape(-1)
    if shares.shape != prices.shape:
        raise ValueError("positions and closes must have the same shape")
    if not np.all(np.isfinite(shares)) or not np.all(np.isfinite(prices)):
        raise ValueError("positions and closes must be finite")
    cash_value = _finite_number(cash, "cash")
    nav = cash_value + float(np.dot(shares, prices))
    if not math.isfinite(nav) or nav <= 0.0:
        raise DecisionParseError("current portfolio NAV must be positive")
    return shares * prices / nav


def parse_decision(
    text: str,
    symbols: Sequence[str],
    *,
    current_weights: Sequence[float],
    max_abs_weight: float = 1.0,
) -> np.ndarray:
    """Parse one canonical completion into a target-weight vector.

    An empty ``orders`` array is the canonical hold action: it leaves the current
    portfolio unchanged rather than forcibly flattening it. Invalid JSON,
    legacy ``weights`` objects, unknown symbols, duplicate orders, or out-of-range
    weights raise :class:`DecisionParseError`.
    """
    return decision_to_weights(
        parse_decision_payload(text),
        symbols,
        current_weights=current_weights,
        max_abs_weight=max_abs_weight,
    )


def format_reward(completion: Any = None, **kwargs: Any) -> float:
    """Reward well-formed XML only when its action is also a canonical decision."""
    if not _HAS_VERIFIERS or completion is None:
        return 0.0
    try:
        parse_decision_payload(str(completion))
        return float(build_parser().get_format_reward_func()(completion=completion))
    except Exception:  # noqa: BLE001 - a metric probe must not break the evaluator
        return 0.0


__all__ = [
    "DecisionParseError",
    "build_parser",
    "decision_to_weights",
    "parse_decision",
    "parse_decision_payload",
    "portfolio_weights",
    "format_reward",
]
