"""Eligibility for training rewards, separate from descriptive return metrics.

A completed episode must cover the planned horizon and have no process failure.
The record is runner-owned accounting, not an attestation from an untrusted agent.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Callable


def is_process_block(event: dict) -> bool:
    name = str(event.get("event", "")).lower()
    return (
        "manipulative" in name
        or name == "protocol_error"
        or str(event.get("severity", "")).lower() == "block"
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
    return all(isinstance(e, dict) and not is_process_block(e) for e in events)


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
