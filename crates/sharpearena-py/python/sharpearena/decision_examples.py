"""Executable action examples shared by initial and per-turn instructions."""

from __future__ import annotations

import json
from typing import Sequence

from .decision_parser import parse_decision_payload


def render_decision_examples(symbols: Sequence[str]) -> str:
    """Render canonical sparse change, hold and full-close decisions.

    Examples use the actual symbol axis and pass the same structural parser as
    completions. These are syntax examples, not recommendations under a mandate.
    """
    if not symbols or any(not isinstance(s, str) or not s for s in symbols):
        raise ValueError("decision examples require nonempty symbol names")
    if len(set(symbols)) != len(symbols):
        raise ValueError("decision examples require unique symbols")
    examples = [
        (
            "Change one target",
            [{"symbol": symbols[0], "action": "buy", "target_weight": 0.25}],
        ),
        ("Hold existing positions", []),
        (
            "Flatten every position",
            [{"symbol": symbol, "action": "close", "target_weight": 0.0} for symbol in symbols],
        ),
    ]
    rendered = []
    for label, orders in examples:
        payload = {"orders": orders, "reasoning": label}
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        # A symbol containing XML delimiters must stay inside the JSON string.
        encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e")
        parse_decision_payload(encoded)
        rendered.append(f"{label}: <action>{encoded}</action>")
    return (
        "Use canonical Decision JSON. Omitted symbols retain their current weights; "
        "an empty orders array holds the existing portfolio, not an all-zero one. "
        "To flatten, give every observed symbol an explicit zero target. "
        "These examples show syntax; choose targets that satisfy your mandate.\n"
        + "\n".join(rendered)
    )
