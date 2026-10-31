"""The ``martingale_v1`` tier, driven through the installed extension.

The Rust tests pin the generator. These pin what an agent receives through the Python
surface: the tier is reachable by name, the observed tape carries the claimed properties
with Calm as the control that keeps the test able to fail, and a typo is refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sharpearena import SharpeArenaEnv, certify_realism
from sharpearena.sharpearena_py import generate_scenario_json

SEEDS = range(300, 332)
N_SYMBOLS = 4
N_DAYS = 120


def _observed_closes(mode: str, seed: int) -> np.ndarray:
    env = SharpeArenaEnv(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=mode)
    obs, _ = env.reset(seed=seed)
    closes = [np.asarray(obs["closes"], dtype=np.float64).reshape(-1)]
    flat = np.zeros(N_SYMBOLS, dtype=np.float32)
    done = False
    while not done:
        obs, _reward, terminated, truncated, _info = env.step(flat)
        closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
        done = terminated or truncated
    return np.stack(closes)


def _lag1(returns: np.ndarray) -> float:
    x = returns - returns.mean(axis=0, keepdims=True)
    return float(np.mean(np.sum(x[1:] * x[:-1], axis=0) / np.sum(x * x, axis=0)))


def test_observed_tape_has_no_lag1_signal_where_calm_does() -> None:
    def mean_lag1(mode: str) -> float:
        values = []
        for seed in SEEDS:
            closes = _observed_closes(mode, seed)
            values.append(_lag1(closes[1:] / closes[:-1] - 1.0))
        return float(np.mean(values))

    assert abs(mean_lag1("martingale_v1")) < 0.04
    assert mean_lag1("calm") > 0.12


def test_observed_tape_passes_the_shipped_gate_more_often_than_calm() -> None:
    def passes(mode: str) -> int:
        count = 0
        for seed in SEEDS:
            closes = _observed_closes(mode, seed)
            count += certify_realism(closes).passed
        return count

    assert passes("martingale_v1") >= 0.6 * len(SEEDS)
    assert passes("calm") == 0


def test_tier_label_round_trips_and_moves_no_other_tier() -> None:
    panel = json.loads(generate_scenario_json(7, n_symbols=2, n_days=30, distribution_mode="martingale_v1"))
    calm = json.loads(generate_scenario_json(7, n_symbols=2, n_days=30, distribution_mode="calm"))
    assert panel["closes"]["SYM00"][0] == 100.0
    assert panel["closes"] != calm["closes"]


def test_unknown_version_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="martingale_v1"):
        generate_scenario_json(7, n_symbols=2, n_days=30, distribution_mode="martingale_v2")
