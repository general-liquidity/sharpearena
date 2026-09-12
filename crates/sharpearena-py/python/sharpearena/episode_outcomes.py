"""Eligibility for training rewards, separate from descriptive return metrics.

A completed episode must cover the planned horizon and have no process failure.
The record is runner-owned accounting, not an attestation from an untrusted agent.
"""

from __future__ import annotations

import json
import math
from functools import wraps
from typing import Any, Callable

from .sharpearena_py import process_event_contract


class UnsupportedProcessEvent(ValueError):
    """An event this boundary cannot classify against the pinned engine's contract.

    Refusing is the whole point. The previous classifier matched the substring
    ``manipulative``, the exact name ``protocol_error`` and an explicit ``severity``
    field, and returned False for everything else. The pinned engine's ``ProcessEvent``
    serializes with neither that token nor a severity field, so four of its five
    block-severity variants read as clean and kept their training reward
    (ARENA-REVIEW A13). An event whose severity is not established is therefore an
    error here, never a pass.
    """


# The engine's own vocabulary and its own severities, obtained from the pinned
# sharpebench-core through the native extension rather than restated. `_DISCRIMINANTS`
# maps each event name to the boolean fields that decide its severity (`order_placed`
# is clean or block depending on `risk_gate_passed`); `_SEVERITY` is keyed by the name
# plus those fields' values.
def _load_engine_contract() -> tuple[dict[str, tuple[str, ...]], dict[tuple, str]]:
    document = json.loads(process_event_contract())
    if document.get("schema_version") != 1:
        raise UnsupportedProcessEvent(
            "the native process-event contract is not schema_version 1"
        )
    discriminants: dict[str, set[str]] = {}
    severity: dict[tuple, str] = {}
    for entry in document["events"]:
        event = entry["event"]
        name = event["event"]
        fields = tuple(
            sorted(
                key
                for key, value in event.items()
                if key != "event" and isinstance(value, bool)
            )
        )
        discriminants.setdefault(name, set()).update(fields)
        severity[(name, tuple((key, event[key]) for key in fields))] = entry["severity"]
    return {name: tuple(sorted(f)) for name, f in discriminants.items()}, severity


_DISCRIMINANTS, _SEVERITY = _load_engine_contract()

# Events this package writes into the same list, which the engine's enum therefore does
# not describe: the rollout layer's own protocol failure and its record of the decided
# target weights (`verifiers_env`), and the liquidation-cascade wrapper's chain
# (`cascade`). Only the protocol failure is a process block; the rest are bookkeeping or
# market-side consequences and must not cost an agent its reward. This list is
# hand-maintained because these names are ours, and `tests/test_process_event_contract.py`
# drives the producers to check it stays complete.
_ARENA_SEVERITY = {
    "protocol_error": "block",
    "target_weights": "none",
    "margin_call": "none",
    "forced_reduce": "none",
    "cascade_impact": "none",
}

if set(_ARENA_SEVERITY) & set(_DISCRIMINANTS):
    raise UnsupportedProcessEvent(
        "an Arena-owned event name collides with the engine's: "
        f"{sorted(set(_ARENA_SEVERITY) & set(_DISCRIMINANTS))}"
    )


def is_process_block(event: dict) -> bool:
    """Whether `event` is a block-severity process violation.

    Raises :class:`UnsupportedProcessEvent` for an event the contract does not define,
    and for an engine event missing the boolean discriminant its severity depends on.
    """
    if not isinstance(event, dict):
        raise UnsupportedProcessEvent(f"a process event must be a dict, got {event!r}")
    name = event.get("event")
    if not isinstance(name, str):
        raise UnsupportedProcessEvent(f"a process event must be named, got {event!r}")
    if name in _DISCRIMINANTS:
        key = []
        for field in _DISCRIMINANTS[name]:
            value = event.get(field)
            if not isinstance(value, bool):
                raise UnsupportedProcessEvent(
                    f"{name} carries no boolean {field}, so its severity is undecided"
                )
            key.append((field, value))
        severity = _SEVERITY.get((name, tuple(key)))
        if severity is None:
            raise UnsupportedProcessEvent(
                f"the engine contract defines no severity for {event!r}"
            )
        return severity == "block"
    if name in _ARENA_SEVERITY:
        return _ARENA_SEVERITY[name] == "block"
    raise UnsupportedProcessEvent(
        f"{name!r} is not an event the pinned engine or this package defines"
    )


def reward_eligible(state: dict | None) -> bool:
    """Refuse missing, failed, nonfinite, or horizon-inconsistent rollout evidence."""
    if (
        not state
        or state.get("error") is not None
        or state.get("protocol_failures", 0)
        or state.get("timed_out", False)
    ):
        return False
    episode = state.get("episode")
    if not isinstance(episode, dict) or episode.get("schema_version") != 1:
        return False
    if episode.get("status") != "completed":
        return False
    counts = [
        episode.get(k)
        for k in ("requested_bars", "available_bars", "planned_bars", "realized_bars")
    ]
    if any(type(n) is not int or n <= 0 for n in counts):
        return False
    requested, available, planned, realized = counts
    if planned != min(requested, available) or realized != planned:
        return False
    returns = state.get("returns")
    if not isinstance(returns, list) or len(returns) != realized:
        return False
    try:
        if not all(math.isfinite(float(r)) for r in returns):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    events = state.get("events")
    if not isinstance(events, list):
        return False
    return not any(is_process_block(e) for e in events)


def eligible_reward(
    func: Callable[..., float], *, floor: float
) -> Callable[..., float]:
    """Gate the actual rubric function while keeping its name/signature for verifiers.

    The primary floor is -1; nonnegative auxiliary rewards use 0. With the rubric's
    fixed weights this makes a failed/incomplete rollout worth -1, never better than
    a valid rollout. Zero alone would instead let an abort beat a valid losing run.
    Raw reward helpers remain usable as descriptive statistics of a partial trace.
    """

    @wraps(func)
    def gated(
        completion: Any = None, state: dict | None = None, **kwargs: Any
    ) -> float:
        if not reward_eligible(state):
            return floor
        value = float(func(completion=completion, state=state, **kwargs))
        if not math.isfinite(value) or not floor <= value <= 1.0:
            raise ValueError(
                f"{func.__name__} produced an out-of-range training reward"
            )
        return value

    return gated
