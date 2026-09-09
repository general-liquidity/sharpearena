"""Reward-misspecification **negative controls** for SharpeArena.

This module pairs deliberately incomplete reward functions with hand-written policies
and compares their return-series diagnostics under the SharpeBench kernel. It does not
train or optimize agents, establish an in/out-of-sample gap, or guarantee that a proxy
underperforms. A policy comparison is not evidence of the effect of optimizing a reward.

CRITICAL INVARIANT — these rewards are NEGATIVE CONTROLS for research only. They are NOT
valid scoring options. They MUST NEVER be registered into ``rewards.REWARD_SCHEMES`` and
MUST NEVER feed the scorer or the rank key. They re-introduce over-leverage / overfit /
churn / myopia *by design*. Importing this module does not mutate any production registry.

The indicator proxy aligns net exposure with the last three realized portfolio returns,
which are the signal consumed by ``indicator_shaped``. The recency proxy follows the last
market-price move, a different heuristic. Neither is a global optimizer: actions affect
future returns. Full-long and tiny-long books are not maximizers of raw PnL or win rate
without assumptions about the market. In a frictionless positive scaling of returns,
win rate and Sharpe can stay unchanged; shrinking a position need not improve either.
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .kernel_score import kernel_score_difference, kernel_score_or_unavailable
from .sharpearena_py import score_run

Policy = Callable[[dict], np.ndarray]
MakeEnv = Callable[[int], object]


# ---------------------------------------------------------------------------
# Misspecified reward functions — NEGATIVE CONTROLS. Bounded, pure over ``state``.
# Signature mirrors the production rewards so the rubric *could* route them — which is
# exactly why they must never be registered.
# ---------------------------------------------------------------------------

_RAW_PNL_GAIN = 50.0
_INDICATOR_WINDOW = 3
_RECENCY_DECAY = 0.6


def _returns_from_state(state: Optional[dict]) -> list[float]:
    return [float(r) for r in (state or {}).get("returns", []) or []]


def _net_weights_per_bar(state: Optional[dict]) -> list[float]:
    """Net signed weight per ``target_weights`` event (the agent's directional bet)."""
    out: list[float] = []
    for e in (state or {}).get("events", []) or []:
        if isinstance(e, dict) and e.get("event") == "target_weights":
            w = e.get("weights")
            if isinstance(w, (list, tuple)):
                out.append(float(sum(float(x) for x in w)))
    return out


def raw_pnl_unpenalized(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — amplified summed returns with no added risk penalty.

    ``tanh(GAIN * sum(returns))``: a high-gain reward on raw cumulative return that ignores
    volatility, turnover, and drawdown as separate objectives. Input returns can already
    include execution costs, so this is not a gross-PnL reconstruction and churn need not
    improve it. Bounded in ``[-1, 1]``. The historical function name is retained.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    return float(np.tanh(_RAW_PNL_GAIN * float(np.sum(rets))))


def win_rate(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — fraction of positive bars, magnitude-blind.

    Rewards *how often* a bar is green, never *by how much*, so it pays an agent to harvest
    many tiny wins and accept rare catastrophic losses (the classic blow-up reward). Bounded
    in ``[0, 1]``. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    return float(np.mean([1.0 if r > 0.0 else 0.0 for r in rets]))


def indicator_shaped(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    window: int = _INDICATOR_WINDOW,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — alignment with a naive short-window momentum signal.

    For each bar past ``window``, rewards the agent when its net position sign agrees with the
    sign of the trailing ``window``-bar return momentum — a classic in-sample-overfit reward
    that fits the agent to a single indicator instead of risk-adjusted edge. Reads the recorded
    ``target_weights`` events; vacuously ``0.0`` if the agent never declared a direction.
    Bounded in ``[0, 1]``. NOT a valid scoring option.
    """
    if type(window) is not int or window <= 0:
        raise ValueError("indicator window must be a positive integer")
    rets = _returns_from_state(state)
    nets = _net_weights_per_bar(state)
    if len(rets) <= window or not nets:
        return 0.0
    hits = 0.0
    count = 0
    for i in range(window, min(len(rets), len(nets))):
        mom = float(np.sign(np.sum(rets[i - window : i])))
        pos = float(np.sign(nets[i]))
        if mom != 0.0 and pos != 0.0:
            hits += 1.0 if mom == pos else 0.0
            count += 1
    return float(hits / count) if count else 0.0


def recency_biased(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    decay: float = _RECENCY_DECAY,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — exponentially over-weights the most recent bars (myopic).

    Geometric decay places almost all weight on the last few bars, so an agent is rewarded for
    chasing the latest move and discounting the path that produced it. Bounded in ``[-1, 1]``
    via ``tanh`` of the recency-weighted mean return. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    n = len(rets)
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    weighted = float(np.dot(w, np.asarray(rets, dtype=float)) / w.sum())
    return float(np.tanh(_RAW_PNL_GAIN * weighted))


MISSPECIFIED_REWARDS: dict[str, Callable[..., float]] = {
    "raw_pnl_unpenalized": raw_pnl_unpenalized,
    "win_rate": win_rate,
    "indicator_shaped": indicator_shaped,
    "recency_biased": recency_biased,
}


# ---------------------------------------------------------------------------
# Hand-written proxy policies. No training or global reward optimization is performed.
# ---------------------------------------------------------------------------


class MaxLeveragePolicy:
    """Constant full-long exposure, not a PnL optimizer.

    Gross target exposure is ``n * max_weight``. A falling market can make the opposite
    position more profitable; execution costs and constraints also affect realized returns.
    """

    name = "max_leverage"

    def __init__(self, max_weight: float = 1.0) -> None:
        self._w = float(max_weight)

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), self._w, dtype=np.float32)


class TinyPositionPolicy:
    """A tiny constant long, not a win-rate optimizer.

    Positive scaling alone preserves return signs in a frictionless model. Transaction
    costs and execution can change that relation; no win-rate improvement is promised.
    """

    name = "tiny_position"

    def __init__(self, eps: float = 0.02) -> None:
        self._eps = float(eps)

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), self._eps, dtype=np.float32)


class MomentumChasePolicy:
    """Align net weight with the trailing ``window`` realized portfolio returns.

    Call ``observe_return`` after each step, as the diagnostic runner does. A completed
    history selects the same signal as ``indicator_shaped(window=window)``; zero signal
    selects flat. Warm-up is full long. This aligns the current signal, not future reward.
    """

    name = "momentum_chase"

    def __init__(
        self, max_weight: float = 1.0, window: int = _INDICATOR_WINDOW
    ) -> None:
        if type(window) is not int or window <= 0:
            raise ValueError("indicator window must be a positive integer")
        self._w = float(max_weight)
        if not math.isfinite(self._w) or self._w <= 0:
            raise ValueError("indicator max_weight must be finite and positive")
        self._returns: deque[float] = deque(maxlen=window)
        self._window = window

    def observe_return(self, reward: float) -> None:
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("indicator feedback must be a finite realized return")
        self._returns.append(value)

    def __call__(self, obs: dict) -> np.ndarray:
        n = np.asarray(obs["closes"]).size
        sign = (
            1.0 if len(self._returns) < self._window else np.sign(np.sum(self._returns))
        )
        return np.full((n,), sign * self._w, dtype=np.float32)


class RecencyChasePolicy:
    """One-bar market-price chaser, not an optimizer of ``recency_biased``.

    Sizes each symbol by the sign of its last one-bar change at full weight and ignores all
    earlier history — maximally myopic. Warms up full long."""

    name = "recency_chase"

    def __init__(self, max_weight: float = 1.0) -> None:
        self._w = float(max_weight)
        self._prev: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        if self._prev is None:
            self._prev = closes.copy()
            return np.full((n,), self._w, dtype=np.float32)
        sign = np.sign(closes - self._prev)
        self._prev = closes.copy()
        return (sign * self._w).astype(np.float32)


def _clean_reference_policy() -> Policy:
    """An equal-weight-long reference. No reward optimization is implied."""

    def _policy(obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), 1.0 / n, dtype=np.float32)

    return _policy


MISSPECIFIED_PROXY_POLICIES: dict[str, Callable[[], Policy]] = {
    "raw_pnl_unpenalized": MaxLeveragePolicy,
    "win_rate": TinyPositionPolicy,
    "indicator_shaped": MomentumChasePolicy,
    "recency_biased": RecencyChasePolicy,
}


# ---------------------------------------------------------------------------
# Scoring the proxies through the REAL SharpeBench kernel
# ---------------------------------------------------------------------------


def _rollout_returns(env, policy: Policy, max_steps: int) -> list[float]:
    out: list[float] = []
    try:
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs, reward, terminated, truncated, _info = env.step(policy(obs))
            out.append(float(reward))
            observe = getattr(policy, "observe_return", None)
            if callable(observe):
                observe(float(reward))
            if bool(terminated) or bool(truncated):
                break
    finally:
        env.close()
    return out


def _score_policy(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    factory: Callable[[], Policy],
    max_steps: int,
    n_trials: int,
) -> dict:
    """Roll a fresh policy per seed, pool the return series, and score with ``score_run``.

    ``deflated_sharpe`` is the kernel's number or, when the kernel withheld it with a
    typed error, the ``unavailable_scoring_kernel_error: ...`` reason string.
    """
    pooled: list[float] = []
    passed: list[float] = []
    for s in seeds:
        returns = _rollout_returns(make_env_for_seed(s), factory(), max_steps)
        pooled.extend(returns)
        if len(returns) >= 2:
            comp = json.loads(score_run(returns, n_trials))
            passed.append(1.0 if comp.get("passed_k", False) else 0.0)
    composite = json.loads(score_run(pooled, n_trials)) if len(pooled) >= 2 else {}
    return {
        "deflated_sharpe": (
            kernel_score_or_unavailable(composite) if composite else 0.0
        ),
        "passed_k": float(np.mean(passed)) if passed else 0.0,
        "mean_return": float(np.mean(pooled)) if pooled else 0.0,
    }


def misspecification_gap(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    *,
    clean_reward: str = "differential_sharpe",
    flawed_reward: str,
    policy: Optional[Callable[[], Policy]] = None,
    max_steps: int = 512,
    n_trials: int = 2,
) -> dict:
    """Compare a reference and a hand-written proxy over the same supplied seeds.

    ``policy`` is the clean reference policy factory (defaults to an equal-weight-long book
    with no training); ``clean_reward`` is a legacy label only. The other side is the proxy for
    ``flawed_reward`` from :data:`MISSPECIFIED_PROXY_POLICIES`. Both are scored by the real
    ``score_run`` kernel and the deflated-Sharpe / mean-return gaps are reported. No reward
    optimization, train/test comparison, or causal effect of a training objective is measured.
    """
    if flawed_reward not in MISSPECIFIED_PROXY_POLICIES:
        raise ValueError(
            f"unknown flawed_reward {flawed_reward!r}; choose from "
            f"{sorted(MISSPECIFIED_PROXY_POLICIES)}"
        )
    seeds = list(seeds)
    clean_factory = policy or _clean_reference_policy
    flawed_factory = MISSPECIFIED_PROXY_POLICIES[flawed_reward]
    clean = _score_policy(make_env_for_seed, seeds, clean_factory, max_steps, n_trials)
    flawed = _score_policy(
        make_env_for_seed, seeds, flawed_factory, max_steps, n_trials
    )
    return {
        "clean_reward": clean_reward,
        "flawed_reward": flawed_reward,
        "clean": clean,
        "flawed": flawed,
        "gap_deflated_sharpe": kernel_score_difference(
            clean["deflated_sharpe"], flawed["deflated_sharpe"]
        ),
        "gap_mean_return": clean["mean_return"] - flawed["mean_return"],
        "proxy_is_stand_in": True,
        "comparison_kind": "heuristic_policy_comparison",
        "optimization_performed": False,
        "clean_reward_role": "label_only",
    }


def demonstrate_punishment(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    *,
    max_steps: int = 512,
    n_trials: Optional[int] = None,
) -> dict:
    """Run every flawed-reward proxy over ``seeds`` and score it with SharpeBench.

    Returns ``{reward_name: {deflated_sharpe, passed_k, mean_return}}``. The legacy name
    does not guarantee punishment: proxies can score well on supplied paths. These are
    pooled-return diagnostics and per-seed pass fractions, not a complete ranking-eligibility
    protocol. ``n_trials`` defaults to the number of proxy policies as a declared comparison
    count; it does not reconstruct an operator's search history.
    """
    seeds = list(seeds)
    trials = len(MISSPECIFIED_PROXY_POLICIES) if n_trials is None else int(n_trials)
    table: dict[str, dict] = {}
    for reward_name, factory in MISSPECIFIED_PROXY_POLICIES.items():
        table[reward_name] = _score_policy(
            make_env_for_seed, seeds, factory, max_steps, trials
        )
    return table


__all__ = [
    "MISSPECIFIED_REWARDS",
    "MISSPECIFIED_PROXY_POLICIES",
    "raw_pnl_unpenalized",
    "win_rate",
    "indicator_shaped",
    "recency_biased",
    "MaxLeveragePolicy",
    "TinyPositionPolicy",
    "MomentumChasePolicy",
    "RecencyChasePolicy",
    "misspecification_gap",
    "demonstrate_punishment",
]
