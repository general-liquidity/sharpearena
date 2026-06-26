"""Cost-adjusted run-metrics for an OpenOutcry rollout.

:class:`RunMetrics` is a **diagnostic** block — separate from the env reward and from the
SharpeBench score, never a scored signal. It tracks the cheap, deterministic facts about a
run (step count, invalid decisions, agent-supplied token / byte / latency budgets, realized
return, max drawdown) so a leaderboard can rank **cost-adjusted, process-checked**
performance rather than raw Sharpe.

:func:`cost_adjusted_score` combines the authoritative SharpeBench composite with a bounded
efficiency penalty. The SharpeBench score stays authoritative; the penalty only ever shrinks
a positive score (and, symmetrically, makes a negative score worse), so a cheaper run with
the same edge ranks above an expensive one — but no amount of frugality can manufacture
edge that the kernel did not credit.

Determinism: this module never reads a wall clock. ``time_to_decision`` durations are
accepted as inputs (seconds), so a replay reproduces the same metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Per-unit fee estimate (fraction of turnover) used only for the diagnostic ``cost_drag``.
_DEFAULT_FEE_RATE = 0.001


class RunMetrics:
    """Per-run diagnostic counters. Update via :meth:`record_step`; read via :meth:`to_dict`."""

    def __init__(self) -> None:
        self.steps = 0
        self.invalid_decisions = 0
        self.decision_durations: list[float] = []  # seconds, agent-supplied
        self.tokens = 0
        self.tool_response_bytes = 0
        self._navs: list[float] = []
        self._returns: list[float] = []
        self._prev_weights: Optional[np.ndarray] = None
        self.realized_return = 0.0
        self.max_drawdown = 0.0
        self.volatility = 0.0
        self.downside_deviation = 0.0
        self.sortino = 0.0
        self.calmar = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.tail_ratio = 0.0
        self.turnover = 0.0

    def record_step(
        self,
        *,
        reward: float = 0.0,
        nav: Optional[float] = None,
        invalid: bool = False,
        duration: Optional[float] = None,
        tokens: int = 0,
        tool_response_bytes: int = 0,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.steps += 1
        if invalid:
            self.invalid_decisions += 1
        if duration is not None:
            self.decision_durations.append(float(duration))
        self.tokens += int(tokens)
        self.tool_response_bytes += int(tool_response_bytes)
        if nav is not None:
            prev = self._navs[-1] if self._navs else 1.0
            self._returns.append(float(nav) / prev - 1.0 if prev else 0.0)
            self._navs.append(float(nav))
        else:
            prev = self._navs[-1] if self._navs else 1.0
            self._navs.append(prev * (1.0 + float(reward)))
            self._returns.append(float(reward))
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            base = (
                self._prev_weights
                if self._prev_weights is not None and self._prev_weights.shape == w.shape
                else np.zeros_like(w)
            )
            self.turnover += float(np.abs(w - base).sum())
            self._prev_weights = w
        self._recompute()

    def _recompute(self) -> None:
        if not self._navs:
            return
        first = self._navs[0] or 1.0
        self.realized_return = self._navs[-1] / first - 1.0
        peak = self._navs[0]
        mdd = 0.0
        for v in self._navs:
            peak = max(peak, v)
            if peak > 0.0:
                mdd = max(mdd, (peak - v) / peak)
        self.max_drawdown = mdd

        r = np.asarray(self._returns, dtype=float)
        if r.size == 0:
            return
        self.volatility = float(np.std(r))
        neg = r[r < 0.0]
        self.downside_deviation = float(np.std(neg)) if neg.size else 0.0
        self.sortino = (
            float(r.mean() / self.downside_deviation)
            if self.downside_deviation > 0.0
            else 0.0
        )
        self.calmar = (
            self.realized_return / self.max_drawdown if self.max_drawdown > 0.0 else 0.0
        )
        p5 = float(np.percentile(r, 5))
        p95 = float(np.percentile(r, 95))
        self.var_95 = p5
        tail = r[r <= p5]
        self.cvar_95 = float(tail.mean()) if tail.size else 0.0
        self.tail_ratio = p95 / abs(p5) if p5 != 0.0 else 0.0

    @property
    def time_to_decision(self) -> float:
        """Mean agent-supplied decision latency (seconds); ``0.0`` if none supplied."""
        return (
            sum(self.decision_durations) / len(self.decision_durations)
            if self.decision_durations
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "invalid_decisions": self.invalid_decisions,
            "time_to_decision": self.time_to_decision,
            "total_decision_seconds": sum(self.decision_durations),
            "tokens": self.tokens,
            "tool_response_bytes": self.tool_response_bytes,
            "realized_return": self.realized_return,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "tail_ratio": self.tail_ratio,
            "turnover": self.turnover,
            "cost_drag": self.turnover * _DEFAULT_FEE_RATE,
        }


# Default per-unit cost weights. ``invalid`` is punished hardest (a malformed decision is a
# process failure); token/byte/latency are gentle so they only break ties between agents of
# comparable edge.
_DEFAULT_WEIGHTS = {
    "invalid": 1.0,
    "token": 1e-4,
    "byte": 1e-6,
    "time": 0.1,
    "turnover": 0.0,  # diagnostic by default; operators may opt turnover into the penalty
}


def cost_adjusted_score(
    composite_score: dict,
    metrics: RunMetrics,
    *,
    base_key: str = "deflated_sharpe",
    weights: Optional[dict] = None,
) -> float:
    """Combine the authoritative SharpeBench composite with a bounded efficiency penalty.

    ``base = composite_score[base_key]`` (the deflated Sharpe the benchmark ranks on).
    ``cost`` is a per-step average of the weighted invalid-decision / token / byte / latency
    budgets; ``penalty = 1 / (1 + cost) ∈ (0, 1]``. The result is ``base * penalty``, so
    ``|result| ≤ |base|`` (bounded) and a higher cost monotonically pulls the magnitude
    toward zero. The SharpeBench score is authoritative — the penalty can only discount it.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    base = float(composite_score.get(base_key, 0.0)) if composite_score else 0.0

    steps = max(metrics.steps, 1)
    raw_cost = (
        w["invalid"] * metrics.invalid_decisions
        + w["token"] * metrics.tokens
        + w["byte"] * metrics.tool_response_bytes
        + w["time"] * sum(metrics.decision_durations)
        + w["turnover"] * metrics.turnover
    )
    cost = max(raw_cost, 0.0) / steps
    penalty = 1.0 / (1.0 + cost)  # ∈ (0, 1]
    return base * penalty


__all__ = [
    "RunMetrics",
    "cost_adjusted_score",
]
