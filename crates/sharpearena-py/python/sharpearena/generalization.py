"""Disjoint train/test seed splits + a single-number generalization-gap metric.

Procgen's thesis, ported to a leak-free market: overfitting is measurable when the
train and test distributions are *provably* disjoint. Here the distributions are
seed intervals — a strategy that scores well on its training seeds but collapses on
a held-out, far-separated band of seeds is overfit, and the gap quantifies it.

The seed arithmetic is kept pure-Python and self-contained (mirroring the same
``(start, num, gap)`` model the Rust split uses) so the metric never depends on a
native split to decide what counts as held out.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .sharpearena_py import score_run

MakeEnv = Callable[[int], object]
MakeEnvMode = Callable[[int, str], object]
Policy = Callable[[dict], np.ndarray]


def train_test_seeds(
    n_train: int,
    n_test: int,
    seed_start: int = 0,
    gap: int = 10_000,
) -> tuple[list[int], list[int]]:
    """Two **provably disjoint** seed lists separated by ``gap``.

    Train occupies ``[seed_start, seed_start + n_train)``; test starts at
    ``seed_start + n_train + gap`` so the bands cannot touch even if ``n_train`` is
    later grown by up to ``gap``. Disjointness is asserted, not assumed.
    """
    assert n_train >= 0 and n_test >= 0 and gap >= 0
    train = list(range(seed_start, seed_start + n_train))
    test_start = seed_start + n_train + gap
    test = list(range(test_start, test_start + n_test))
    assert set(train).isdisjoint(test), "train/test seed bands overlap"
    return train, test


def _equal_weight_policy(obs: dict) -> np.ndarray:
    n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
    return np.full((n,), 1.0 / n, dtype=np.float32)


def _rollout_returns(env, policy: Policy, max_steps: int) -> list[float]:
    obs, _ = env.reset()
    out: list[float] = []
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        out.append(float(reward))
        if bool(terminated) or bool(truncated):
            break
    return out


def evaluate_seeds(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    policy: Optional[Policy] = None,
    max_steps: int = 512,
    *,
    n_trials: int = 0,
) -> dict:
    """Run ``policy`` over each seed's env and score the pooled return series.

    ``policy`` defaults to a flat equal-weight baseline. Per seed we record one
    episode's return series, score each with the real SharpeBench kernel
    (``passed_k`` → pass rate), and score the pooled series for an aggregate
    deflated Sharpe. ``n_trials`` deflates for declared in-sample search breadth.
    """
    policy = policy or _equal_weight_policy
    pooled: list[float] = []
    passed: list[float] = []
    for s in seeds:
        returns = _rollout_returns(make_env_for_seed(s), policy, max_steps)
        pooled.extend(returns)
        if len(returns) >= 2:
            comp = json.loads(score_run(returns, n_trials))
            passed.append(1.0 if comp.get("passed_k", False) else 0.0)
    composite = json.loads(score_run(pooled, n_trials)) if len(pooled) >= 2 else {}
    return {
        "n_seeds": len(list(seeds)),
        "deflated_sharpe": float(composite.get("deflated_sharpe", 0.0)),
        "passed_k_rate": float(np.mean(passed)) if passed else 0.0,
        "mean_return": float(np.mean(pooled)) if pooled else 0.0,
    }


def generalization_gap(
    make_env_for_seed: MakeEnv,
    n_train: int,
    n_test: int,
    policy: Optional[Policy] = None,
    seed_start: int = 0,
    gap: int = 10_000,
    max_steps: int = 512,
    *,
    n_trials: int = 0,
) -> dict:
    """Headline anti-overfitting metric: train vs. disjoint-test score, differenced.

    Returns the per-split aggregates plus ``gap_deflated_sharpe`` (train − test) and
    ``gap_mean_return``. A large positive gap is overfit; near zero generalizes.
    """
    train_seeds, test_seeds = train_test_seeds(n_train, n_test, seed_start, gap)
    train = evaluate_seeds(
        make_env_for_seed, train_seeds, policy, max_steps, n_trials=n_trials
    )
    test = evaluate_seeds(
        make_env_for_seed, test_seeds, policy, max_steps, n_trials=n_trials
    )
    return {
        "train": train,
        "test": test,
        "gap_deflated_sharpe": train["deflated_sharpe"] - test["deflated_sharpe"],
        "gap_mean_return": train["mean_return"] - test["mean_return"],
    }


def cross_regime_transfer(
    make_env_for_seed_and_mode: MakeEnvMode,
    train_mode: str,
    test_mode: str,
    seeds: Sequence[int],
    policy: Optional[Policy] = None,
    max_steps: int = 512,
    *,
    n_trials: int = 0,
) -> dict:
    """Zero-shot cross-regime transfer gap: select on regime A, score on regime B.

    :func:`generalization_gap` varies the *seed band* inside one ``distribution_mode``, so
    a policy that only works in (say) calm markets but is scored solely on calm seeds still
    passes. This instead varies the *regime* while holding the seed band fixed: the policy
    is scored in-distribution on ``train_mode`` and zero-shot out-of-distribution on
    ``test_mode`` over the **same** ``seeds``. The transfer gap (in-distribution minus
    out-of-distribution deflated Sharpe) isolates regime-specific overfit, which a
    within-tier seed gap is blind to, so it is a strictly stronger robustness signal.

    ``make_env_for_seed_and_mode(seed, mode)`` must build a fresh env at a given scenario
    seed and ``distribution_mode``. Because the seed band is identical across the two
    evaluations, ``train_mode == test_mode`` reuses byte-identical envs and the transfer
    gap is exactly ``0`` by construction.
    """
    seeds = list(seeds)
    in_dist = evaluate_seeds(
        lambda s: make_env_for_seed_and_mode(s, train_mode),
        seeds,
        policy,
        max_steps,
        n_trials=n_trials,
    )
    out_dist = evaluate_seeds(
        lambda s: make_env_for_seed_and_mode(s, test_mode),
        seeds,
        policy,
        max_steps,
        n_trials=n_trials,
    )
    return {
        "train_mode": train_mode,
        "test_mode": test_mode,
        "in_distribution": in_dist,
        "out_of_distribution": out_dist,
        "transfer_gap_deflated_sharpe": in_dist["deflated_sharpe"]
        - out_dist["deflated_sharpe"],
        "transfer_gap_mean_return": in_dist["mean_return"] - out_dist["mean_return"],
    }


__all__ = [
    "train_test_seeds",
    "evaluate_seeds",
    "generalization_gap",
    "cross_regime_transfer",
]
