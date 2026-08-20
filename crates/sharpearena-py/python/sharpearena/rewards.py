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

Two schemes price risk per bar rather than at the end of the episode: :func:`risk_aware`
charges a causally-estimated conditional volatility at a fixed aversion, and
:func:`time_inhomogeneous_vol_aversion` lets that aversion vary across the horizon, which is
the one thing no other scheme here does.
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


# Causal-risk EMA decay for the two risk-pricing schemes below. eta=0.06 (~16-bar effective
# window) tracks the local volatility regime fast enough that the charge follows the regime
# rather than the whole episode; the 2-bar warm-up suppresses the transient where the
# variance estimate is still seeded at zero.
_RISK_ETA = 0.06
_RISK_WARMUP = 2


def _causal_risk_path(returns: list[float], eta: float) -> list[float]:
    """Per-bar conditional volatility estimated from bars STRICTLY BEFORE each bar.

    Walks the series once, reading the running EMA variance before folding the current bar
    into it, so the risk charged at bar ``t`` uses only information available at ``t``. That
    causality is what makes the charge a per-step price of risk rather than a post-hoc
    aggregate over the finished path.
    """
    mean_ema = 0.0
    var_ema = 0.0
    out: list[float] = []
    for i, r in enumerate(returns):
        out.append(float(np.sqrt(var_ema)) if i >= _RISK_WARMUP else 0.0)
        dev = r - mean_ema
        mean_ema += eta * dev
        var_ema += eta * (dev * dev - var_ema)
    return out


def risk_aware(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    lam: float = 1.0,
    eta: float = _RISK_ETA,
    **kwargs: Any,
) -> float:
    """Per-bar return net of a causally-estimated risk charge, tanh-bounded.

    Follows "A Risk-Aware Reinforcement Learning Reward for Financial Trading": price risk
    INTO the per-step signal instead of settling up at the end of the episode. Each bar
    contributes ``r_t - lam * sigma_t``, where ``sigma_t`` is the EMA conditional volatility
    built from bars strictly before ``t``. The sum is tanh-bounded to ``[-1, 1]``.

    Why this is not a re-spelling of what is already registered. ``drawdown_penalized``
    charges one episode-terminal scalar (the path extremum) against the aggregate, so it
    cannot separate two paths that share a worst peak-to-trough decline. ``sortino`` divides
    by a single whole-episode downside deviation, so it is a ratio computed once, after the
    fact, and it is blind to WHEN the volatility was carried. This scheme charges at every
    bar against the risk prevailing at that bar, using two-sided conditional volatility
    rather than a downside aggregate: earning a given return while sitting in a turbulent
    stretch scores below earning it in a calm one, even when the two paths share an identical
    max drawdown and an identical episode-wide downside deviation. That per-bar,
    regime-local attribution is genuinely absent from the registry.

    Shapes TRAINING only. The rank key stays the SharpeBench kernel (deflated Sharpe /
    ``pass^k`` / process checks); no scheme ever feeds the scorer.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    risk = _causal_risk_path(rets, float(eta))
    agg = sum(r - float(lam) * s for r, s in zip(rets, risk))
    return float(np.tanh(agg))


_AVERSION_SHAPES = ("linear", "convex", "concave", "exponential")


def time_aversion_schedule(
    n: int,
    *,
    lam_start: float = 0.25,
    lam_end: float = 1.5,
    shape: str = "linear",
    curvature: float = 2.0,
) -> list[float]:
    """The per-bar risk-aversion coefficients ``lam_0 .. lam_{n-1}`` over an episode.

    Time enters through ``u = t / (n - 1)``, the fraction of the horizon already consumed, so
    ``1 - u`` is the fraction of horizon remaining. ``shape`` selects how aversion travels
    from ``lam_start`` to ``lam_end``:

    * ``linear``: constant rate of change in ``u``.
    * ``convex``: ``u ** curvature``, aversion stays low then climbs late, the profile of an
      agent that only starts to fear volatility once the horizon is close.
    * ``concave``: ``1 - (1 - u) ** curvature``, aversion rises early then flattens.
    * ``exponential``: geometric interpolation, requires both endpoints strictly positive.

    Exposed publicly so a caller can inspect the schedule it is training against, and so the
    time dependence is checkable rather than buried inside a loop.
    """
    if shape not in _AVERSION_SHAPES:
        raise ValueError(f"unknown shape {shape!r}; choose from {list(_AVERSION_SHAPES)}")
    count = max(int(n), 1)
    lo = float(lam_start)
    hi = float(lam_end)
    if shape == "exponential" and (lo <= 0.0 or hi <= 0.0):
        raise ValueError("exponential shape requires lam_start > 0 and lam_end > 0")
    k = max(float(curvature), 1e-6)
    out: list[float] = []
    for t in range(count):
        u = 0.0 if count < 2 else t / (count - 1)
        if shape == "linear":
            lam = lo + (hi - lo) * u
        elif shape == "convex":
            lam = lo + (hi - lo) * (u**k)
        elif shape == "concave":
            lam = lo + (hi - lo) * (1.0 - (1.0 - u) ** k)
        else:
            lam = lo * ((hi / lo) ** u)
        out.append(float(lam))
    return out


def time_inhomogeneous_vol_aversion(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    lam_start: float = 0.25,
    lam_end: float = 1.5,
    shape: str = "linear",
    curvature: float = 2.0,
    eta: float = _RISK_ETA,
    horizon: Optional[int] = None,
    **kwargs: Any,
) -> float:
    """Volatility charge whose aversion coefficient varies with time, tanh-bounded.

    Follows "Time-Inhomogeneous Volatility Aversion for Financial Applications of
    Reinforcement Learning". Every other scheme in this registry applies the SAME risk
    treatment at bar 1 and at bar 500: ``drawdown_penalized`` has one ``lam``,
    ``loss_averse`` one asymmetry, ``sortino`` and ``differential_sharpe`` one functional
    form. That is a modelling assumption, not a fact about traders. Aversion to volatility is
    not constant over a horizon, and tolerance for a drawdown at the open of a mandate is not
    tolerance for the same drawdown with a handful of bars left to recover it.

    So the per-bar charge is ``r_t - lam(t) * sigma_t``, with ``lam(t)`` drawn from
    :func:`time_aversion_schedule` (a function of elapsed and remaining horizon) and
    ``sigma_t`` the same causal EMA conditional volatility ``risk_aware`` uses. Pass
    ``horizon`` when the mandate's horizon is longer than the bars actually realized, so a
    truncated episode is priced against the horizon it was meant to run rather than against
    its own length.

    Shapes TRAINING only. The rank key stays the SharpeBench kernel (deflated Sharpe /
    ``pass^k`` / process checks); no scheme ever feeds the scorer.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    n = len(rets) if horizon is None else max(int(horizon), len(rets))
    lam = time_aversion_schedule(
        n, lam_start=lam_start, lam_end=lam_end, shape=shape, curvature=curvature
    )
    risk = _causal_risk_path(rets, float(eta))
    agg = sum(r - lam[i] * risk[i] for i, r in enumerate(rets))
    return float(np.tanh(agg))


REWARD_SCHEMES: dict[str, Any] = {
    "default": realized_return_reward,
    "differential_sharpe": differential_sharpe,
    "sortino": sortino,
    "drawdown_penalized": drawdown_penalized,
    "turnover_penalized": turnover_penalized,
    "loss_averse": loss_averse,
    "risk_aware": risk_aware,
    "time_inhomogeneous_vol_aversion": time_inhomogeneous_vol_aversion,
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
    "risk_aware",
    "time_inhomogeneous_vol_aversion",
    "time_aversion_schedule",
]
