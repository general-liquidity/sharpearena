"""Pluggable reward-scheme registry for SharpeArena ``verifiers`` training.

Reward schemes shape **training only**. The scoring truth stays the Rust ``score_run``
kernel (deflated Sharpe / ``pass^k`` / process checks) — a scheme never feeds the scorer
or the rank key. Every scheme is a pure, bounded function of ``state['returns']`` /
``state['events']`` (already point-in-time, leak-free) so it is GRPO-safe.

``build_scheme_rubric(scheme, ...)`` composes a chosen primary reward (weight 1.0) with the
real deflated Sharpe (0.5) and the per-scenario mandate (0.5) — the same 3-func shape the
hardcoded rubric used. ``"default"`` reproduces the original realized-return scheme exactly.

The flagship scheme is :func:`differential_sharpe` — an online Moody-Saffell differential
Sharpe ratio that aligns the *training* signal with the deflated-Sharpe *scoring* objective
(agents otherwise optimize raw return but are judged on Sharpe).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .decision_parser import format_reward
from .verifiers_env import (
    _HAS_VERIFIERS,
    _returns_from_state,
    deflated_sharpe_reward,
    mandate_reward,
    pass_k_reward,
    process_check_reward,
    realized_return_reward,
    vf,
)

# Moody-Saffell EMA decay. eta=0.04 (~25-bar effective window) keeps the running variance
# estimate responsive without letting a single bar dominate; the 3-bar warm-up suppresses
# the early transient where ``B - A^2`` is near-zero and the per-bar derivative explodes.
_DSR_ETA = 0.04
_DSR_WARMUP = 3
_DSR_CLIP = 1.0


def differential_sharpe(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Online differential Sharpe ratio (Moody-Saffell), tanh-bounded.

    Maintains EMA estimates of returns ``A`` and squared returns ``B``; each bar contributes
    ``Dt = (B*ΔA - 0.5*A*ΔB) / (B - A^2)^1.5`` (the derivative of the Sharpe ratio w.r.t. the
    newest return). Per-bar ``Dt`` is clipped to ``±1`` to tame warm-up spikes, summed, and
    scale-normalized by ``√n`` before ``tanh`` — so the reward tracks the sign and ordering of
    the episode's batch Sharpe while staying in ``[-1, 1]``.
    """
    rets = _returns_from_state(state)
    if len(rets) < _DSR_WARMUP + 2:
        return 0.0
    a = 0.0
    b = 0.0
    total = 0.0
    n = 0
    for i, r in enumerate(rets):
        d_a = r - a
        d_b = r * r - b
        var = b - a * a
        if i >= _DSR_WARMUP and var > 1e-9:
            dt = (b * d_a - 0.5 * a * d_b) / (var ** 1.5)
            total += max(-_DSR_CLIP, min(_DSR_CLIP, dt))
            n += 1
        a += _DSR_ETA * d_a
        b += _DSR_ETA * d_b
    if n == 0:
        return 0.0
    return float(np.tanh(total / np.sqrt(n)))


def sortino(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Downside-deviation-denominated risk-adjusted return, tanh-bounded.

    ``mean(returns) / downside_deviation`` where the denominator is the RMS of the negative
    bars only (upside volatility is not penalized). A series with no losing bar earns the
    mean's sign at full magnitude; an all-flat series scores 0.
    """
    rets = _returns_from_state(state)
    if len(rets) < 2:
        return 0.0
    a = np.asarray(rets, dtype=float)
    downside = a[a < 0.0]
    dd = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
    mean = float(a.mean())
    if dd <= 1e-12:
        return float(np.tanh(np.sign(mean) * a.size))
    return float(np.tanh(mean / dd))


def _max_drawdown(returns: list[float]) -> float:
    """Compounded max drawdown over the equity curve, a fraction in ``[0, 1]``."""
    if not returns:
        return 0.0
    nav = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        nav *= 1.0 + r
        peak = max(peak, nav)
        if peak > 0.0:
            mdd = max(mdd, (peak - nav) / peak)
    return mdd


def drawdown_penalized(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    lam: float = 0.5,
    **kwargs: Any,
) -> float:
    """``tanh(sum(returns)) - lam * max_drawdown``, clipped to ``[-1, 1]``.

    Rewards cumulative return but charges the worst peak-to-trough decline along the path, so
    two paths with the same endpoint are separated by their drawdown.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    val = float(np.tanh(np.sum(rets))) - float(lam) * _max_drawdown(rets)
    return float(max(-1.0, min(1.0, val)))


def _weight_vectors(events: Any) -> list[list[float]]:
    """Per-bar target-weight vectors from ``{"event": "target_weights", "weights": [...]}``."""
    out: list[list[float]] = []
    for e in events or []:
        if isinstance(e, dict) and e.get("event") == "target_weights":
            w = e.get("weights")
            if isinstance(w, (list, tuple)):
                out.append([float(x) for x in w])
    return out


def turnover_penalized(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    lam: float = 0.5,
    **kwargs: Any,
) -> float:
    """Realized return minus a bounded turnover penalty, clipped to ``[-1, 1]``.

    Turnover is the summed L1 change between consecutive target-weight vectors read from the
    rollout's ``{"event": "target_weights", "weights": [...]}`` events (the same shape
    ``mandate_breach`` reads). The penalty is ``lam * tanh(turnover)`` so it is bounded in
    ``[0, lam]`` and churn-heavy paths score below quiet ones with the same return.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    weights = _weight_vectors((state or {}).get("events", []))
    turnover = 0.0
    for prev, cur in zip(weights, weights[1:]):
        m = max(len(prev), len(cur))
        for i in range(m):
            pv = prev[i] if i < len(prev) else 0.0
            cv = cur[i] if i < len(cur) else 0.0
            turnover += abs(cv - pv)
    penalty = float(lam) * float(np.tanh(turnover))
    val = float(np.tanh(np.sum(rets))) - penalty
    return float(max(-1.0, min(1.0, val)))


def loss_averse(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    risk_averse: float = 1.0,
    **kwargs: Any,
) -> float:
    """Asymmetric aggregate: losing bars weighted ``×(1 + risk_averse)``, tanh-bounded.

    Negative per-bar returns are amplified before summation (prospect-theory loss aversion),
    so a path that reaches the same endpoint through deeper losses is penalized.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    k = 1.0 + float(risk_averse)
    agg = sum(r if r >= 0.0 else r * k for r in rets)
    return float(np.tanh(agg))


REWARD_SCHEMES: dict[str, Any] = {
    "default": realized_return_reward,
    "differential_sharpe": differential_sharpe,
    "sortino": sortino,
    "drawdown_penalized": drawdown_penalized,
    "turnover_penalized": turnover_penalized,
    "loss_averse": loss_averse,
}


def list_reward_schemes() -> list[str]:
    """The registered scheme names."""
    return sorted(REWARD_SCHEMES)


def build_scheme_rubric(scheme: str = "default", *, parser: Any = None, mandate: bool = True):
    """A ``vf.Rubric`` composing the chosen primary reward (1.0) + deflated Sharpe (0.5) +
    mandate (0.5 if enabled), matching the original 3-func shape. ``pass^k`` / process / format
    stay zero-weight diagnostics. ``scheme="default"`` is the original realized-return rubric.
    Raises if ``verifiers`` is unavailable or the scheme is unknown."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build a Rubric")
    primary = REWARD_SCHEMES.get(scheme)
    if primary is None:
        raise ValueError(
            f"unknown reward_scheme {scheme!r}; choose from {list_reward_schemes()}"
        )
    funcs = [primary, deflated_sharpe_reward]
    weights = [1.0, 0.5]
    if mandate:
        funcs.append(mandate_reward)
        weights.append(0.5)
    rubric = vf.Rubric(funcs=funcs, weights=weights, parser=parser)
    rubric.add_metric(pass_k_reward)
    rubric.add_metric(process_check_reward)
    rubric.add_metric(format_reward)
    return rubric


__all__ = [
    "REWARD_SCHEMES",
    "list_reward_schemes",
    "build_scheme_rubric",
    "differential_sharpe",
    "sortino",
    "drawdown_penalized",
    "turnover_penalized",
    "loss_averse",
]
