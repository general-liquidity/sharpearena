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

from .openoutcry_py import score_run

MakeEnv = Callable[[int], object]
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


__all__ = ["train_test_seeds", "evaluate_seeds", "generalization_gap"]
