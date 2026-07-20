"""The trade-decision protocol for the ``verifiers`` multi-turn rollout.

A turn's model output is XML with two fields: ``<reasoning>`` (free text) and
``<action>`` carrying decision JSON, e.g. ``{"weights": {"SYM00": 0.5, "SYM01": -0.3}}``
or ``{"flat": true}``. :func:`parse_decision` maps that to a target-weight vector over
the env's symbols.

**Error policy:** an unparseable / malformed decision is treated as **flat (hold)** —
all-zero weights — so a bad completion never crashes the episode; the rollout simply
makes no position change that bar. :func:`format_reward` separately (and mildly)
rewards well-formed output so the model is nudged toward valid XML without the episode
being brittle.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only when verifiers is installed
    import verifiers as vf

    _HAS_VERIFIERS = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    vf = None  # type: ignore[assignment]
    _HAS_VERIFIERS = False

_FIELDS = ["reasoning", "action"]
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

_PARSER: Any = None


def build_parser():
    """The ``vf.XMLParser`` over ``<reasoning>``/``<action>`` (answer = ``action``)."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build an XMLParser")
    global _PARSER
    if _PARSER is None:
        _PARSER = vf.XMLParser(fields=_FIELDS, answer_field="action")
    return _PARSER


def _extract_action_json(text: str) -> Optional[dict]:
    """Pull the ``<action>`` JSON object out of a completion; ``None`` if malformed.

    Falls back to parsing the whole string as JSON so callers can pass a bare
    decision blob (used by the unit tests and by env-driven canned messages).
    """
    if not text:
        return None
    m = _ACTION_RE.search(text)
    blob = (m.group(1) if m else text).strip()
    try:
        payload = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_decision(text: str, symbols: list[str]) -> np.ndarray:
    """Target-weight vector over ``symbols`` from a decision completion.

    Weights are read from ``payload["weights"][symbol]``; unknown symbols are ignored,
    missing ones default to 0. ``{"flat": true}`` / ``{"hold": true}`` / any unparseable
    input yields an all-zero (flat) vector. Output is clamped to ``[-1, 1]``.
    """
    n = len(symbols)
    vec = np.zeros(n, dtype=np.float64)
    payload = _extract_action_json(text)
    if payload is None or payload.get("flat") is True or payload.get("hold") is True:
        return vec
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        return vec
    index = {s: i for i, s in enumerate(symbols)}
    for sym, w in weights.items():
        i = index.get(sym)
        if i is None:
            continue
        try:
            vec[i] = float(w)
        except (TypeError, ValueError):
            continue
    return np.clip(vec, -1.0, 1.0)


def format_reward(completion: Any = None, **kwargs: Any) -> float:
    """Mild reward (in ``[0, 1]``) for well-formed ``<reasoning>``/``<action>`` XML.

    Delegates to the parser's own format-reward func; 0.0 when verifiers is
    unavailable or the completion can't be scored. Registered as a low/zero-weight
    metric, not a primary objective.
    """
    if not _HAS_VERIFIERS or completion is None:
        return 0.0
    try:
        return float(build_parser().get_format_reward_func()(completion=completion))
    except Exception:  # noqa: BLE001 - never let a format probe break scoring
        return 0.0


__all__ = ["build_parser", "parse_decision", "format_reward"]
