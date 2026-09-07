"""Refuse figures that would turn missing scoring confidence into a zero error bar."""

from __future__ import annotations

import math


def require_scoring_intervals(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("scoring confidence requires a nonempty field")
    for row in rows:
        interval = row.get("deflated_sharpe_ci")
        if not isinstance(interval, dict):
            raise ValueError(
                f"scoring confidence unavailable for {row.get('policy', '?')}: "
                f"{row.get('confidence_status', 'missing interval')}; "
                "refusing to publish a zero-width substitute"
            )
        for key in ("point", "lo", "hi"):
            value = interval.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"scoring confidence {key} must be finite")
        if interval["lo"] > interval["hi"]:
            raise ValueError("scoring confidence interval endpoints are reversed")
