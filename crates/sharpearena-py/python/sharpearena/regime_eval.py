"""Per-regime robustness breakdown + a baseline-anchored two-axis radar score.

Two diagnostics that strengthen the leaderboard's *legibility* without ever touching
its rank key. The rank stays deflated Sharpe + pass^k + process — these are display
and diagnostic only:

* :func:`evaluate_per_regime` labels every bar a deterministic regime (trend-up /
  trend-down / chop, from the realized drift-vs-vol of a trailing market index) and
  reports deflated Sharpe *per regime* — so a policy that only works in trends is
  caught, even when its pooled number looks fine. Labeling is causal: a bar's regime
  is computed from closes observed up to that bar only.

* :func:`radar_score` collapses a metric panel onto two bounded axes — Profitability
  and Risk-Control. Flat profitability maps to ``0``; zero drawdown maps to the
  best risk score, ``scale``. A usable ``EqualWeightLong`` reference maps to
  ``base`` on both axes. The squashing is a deterministic
  ``tanh`` (logistic-family CDF approximation), so the score is reproducible and
  indicative, not a calibrated probability.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Optional, Sequence

import numpy as np

from .kernel_score import kernel_score_or_unavailable
from .sharpearena_py import score_run

MakeEnv = Callable[[int], object]
Policy = Callable[[dict], np.ndarray]

TREND_UP = "trend_up"
TREND_DOWN = "trend_down"
CHOP = "chop"
REGIMES = (TREND_UP, TREND_DOWN, CHOP)


def _market_index(closes: np.ndarray) -> float:
    return float(np.mean(np.asarray(closes, dtype=np.float64).reshape(-1)))


def label_regime(
    index_history: Sequence[float], window: int, trend_frac: float
) -> str:
    """Classify the latest bar from a trailing market-index window.

    Computes the mean and stdev of the window's per-step returns; the bar is a trend
    (up/down by the sign of the drift) when ``|mean| > trend_frac * vol``, else chop.
    Uses only the supplied history, so when called per step with the closes seen so far
    the label is causal. Warms up to ``chop`` until a full window exists.
    """
    hist = np.asarray(index_history, dtype=np.float64)
    if hist.shape[0] < window + 1:
        return CHOP
    tail = hist[-(window + 1) :]
    rets = tail[1:] / np.where(tail[:-1] == 0.0, 1.0, tail[:-1]) - 1.0
    mean = float(rets.mean())
    vol = float(rets.std())
    if abs(mean) <= trend_frac * vol:
        return CHOP
    return TREND_UP if mean > 0.0 else TREND_DOWN


def evaluate_per_regime(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    policy: Policy,
    *,
    max_steps: int = 512,
    window: int = 20,
    trend_frac: float = 0.5,
    n_trials: int = 0,
) -> dict:
    """Roll ``policy`` over ``seeds``, bucket each bar's reward by regime, score each bucket.

    For every seed the rollout accumulates a market index from the observed closes and
    labels each bar (causally) before stepping. Rewards are pooled per regime across all
    seeds and scored with the real SharpeBench kernel. Deterministic given env
    determinism: repeated calls return identical results.

    Returns ``{"per_regime": {regime: {deflated_sharpe, passed_k, n_bars, mean_return}},
    "overall": {...}}``. A bucket the kernel withheld with a typed error carries the
    ``unavailable_scoring_kernel_error: ...`` reason string in ``deflated_sharpe``;
    :func:`radar_score` then refuses that panel rather than anchoring on a floor.
    """
    buckets: dict[str, list[float]] = {r: [] for r in REGIMES}
    pooled: list[float] = []
    for s in seeds:
        env = make_env_for_seed(s)
        obs, _ = env.reset()
        index_hist: list[float] = [_market_index(obs["closes"])]
        for _ in range(max_steps):
            regime = label_regime(index_hist, window, trend_frac)
            obs, reward, terminated, truncated, _info = env.step(policy(obs))
            buckets[regime].append(float(reward))
            pooled.append(float(reward))
            index_hist.append(_market_index(obs["closes"]))
            if bool(terminated) or bool(truncated):
                break
    return {
        "per_regime": {r: _score_bucket(buckets[r], n_trials) for r in REGIMES},
        "overall": _score_bucket(pooled, n_trials),
    }


def _score_bucket(returns: Sequence[float], n_trials: int) -> dict:
    rets = list(returns)
    if len(rets) < 2:
        return {
            "deflated_sharpe": 0.0,
            "passed_k": False,
            "n_bars": len(rets),
            "mean_return": float(np.mean(rets)) if rets else 0.0,
        }
    comp = json.loads(score_run(rets, n_trials))
    return {
        "deflated_sharpe": kernel_score_or_unavailable(comp),
        "passed_k": bool(comp.get("passed_k", False)),
        "n_bars": len(rets),
        "mean_return": float(np.mean(rets)),
    }


# -- radar score ------------------------------------------------------------

_PROFIT_KEY = "deflated_sharpe"
_RISK_KEY = "max_drawdown"


def _profitability_raw(metrics: dict) -> float:
    value = metrics.get(_PROFIT_KEY, metrics.get("mean_return", 0.0))
    if isinstance(value, str):
        raise ValueError(f"radar profitability is unavailable: {value}")
    return float(value)


def _drawdown(metrics: dict) -> float:
    if _RISK_KEY not in metrics:
        raise ValueError("radar requires observed max_drawdown for candidate and base anchor")
    value = float(metrics[_RISK_KEY])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("max_drawdown must be finite and nonnegative")
    return value


def _anchor_axis(
    value: float, zero: float, base_raw: float, *, base: float, scale: float
) -> float:
    """Map ``value`` to ``[0, scale]`` with ``zero -> 0`` and ``base_raw -> base`` exactly.

    Normalizes to ``n = (value - zero) / (base_raw - zero)`` (so flat is ``0``, the base
    baseline is ``1``) then squashes with ``tanh`` scaled so ``tanh(k) == base/scale`` —
    a deterministic logistic-family CDF approximation. Bounded and clamped to
    ``[0, scale]``; indicative, not calibrated.
    """
    denom = base_raw - zero
    if not all(math.isfinite(x) for x in (value, zero, base_raw, denom)) or denom <= 0.0:
        raise ValueError("radar anchor span must be finite and strictly positive")
    n = (value - zero) / denom
    k = math.atanh(min(max(base / scale, 1e-9), 1.0 - 1e-9))
    squashed = math.tanh(k * n)
    return float(min(max(squashed * scale, 0.0), scale))


def radar_score(
    metrics: dict,
    *,
    zero_anchor: dict,
    base_anchor: dict,
    base: float = 50.0,
    scale: float = 100.0,
) -> dict:
    """Two-axis diagnostic with higher scores meaning better performance/control.

    Profitability maps ``zero_anchor`` to zero and ``base_anchor`` to ``base``.
    Risk maps zero drawdown to ``scale`` and the base's positive drawdown to
    ``base``; more drawdown can never improve it. Requiring flat to be zero on
    both axes would contradict that direction. Missing risk observations or a
    nonpositive anchor span raise ``ValueError``, not a fabricated panel.
    ``overall`` is the diagnostic mean of the two axes, each in ``[0, scale]``.
    It is not a rank-eligibility rule or a calibrated probability.
    """
    if not math.isfinite(base) or not math.isfinite(scale) or not 0.0 < base < scale:
        raise ValueError("radar requires finite 0 < base < scale")
    candidate_drawdown = _drawdown(metrics)
    base_drawdown = _drawdown(base_anchor)
    profit = _anchor_axis(
        _profitability_raw(metrics),
        _profitability_raw(zero_anchor),
        _profitability_raw(base_anchor),
        base=base,
        scale=scale,
    )
    risk = scale - _anchor_axis(
        candidate_drawdown,
        0.0,
        base_drawdown,
        base=scale - base,
        scale=scale,
    )
    return {
        "profitability": profit,
        "risk_control": risk,
        "overall": 0.5 * (profit + risk),
    }


__all__ = [
    "REGIMES",
    "TREND_UP",
    "TREND_DOWN",
    "CHOP",
    "label_regime",
    "evaluate_per_regime",
    "radar_score",
]
