"""Seeded accounting properties for the Python LOB market (``lob_env.py``).

The Rust ``OrderBook`` carries no accounts: cash and inventory live only in
``LOBMarketEnv`` (``lob_env.py``, ``_apply_fills`` / ``_equity`` / ``_reward``). These
properties drive the shipped environment with seeded ``random.Random`` order streams and
check the interface-level accounting after every step, exactly as the environment
computes it (integer tick prices, so cash sums are exact in float).

The seeded-reset property pins the repaired defect from
``docs/audits/2026-09-07/arena-reviewer.md`` item 7 (LOB reset retained ``_prev_equity``),
closed under batch G in ``VERIFICATION-LOG.md``; it is a regression guard, not a reopening.
"""

from __future__ import annotations

import json
import random

import pytest

pytest.importorskip("pettingzoo")

from sharpearena.lob_env import LOBMarketEnv  # noqa: E402

CASES = 24


def _actions(rng: random.Random, env: LOBMarketEnv) -> dict:
    return {
        a: [rng.randint(1, env._max_offset), rng.randint(1, env._max_offset)]
        for a in env.agents
    }


def _capture_fills(env: LOBMarketEnv) -> list:
    """Record every fill tape the environment books, without changing what it books."""
    captured: list = []
    original = env._apply_fills

    def wrapped(fills, ladder):
        captured.append(fills)
        original(fills, ladder)

    env._apply_fills = wrapped
    return captured


def test_equity_reward_identity_and_flow_conservation_every_step():
    """Binds ``_equity`` / ``_reward`` (lob_env.py:186-196) and ``_apply_fills`` (:163-184).

    After each step, for every agent: ``cash + inventory * mid`` is the stored equity
    baseline, and ``reward == equity - prev_equity - penalty * inventory**2``. Across
    agents, inventory and cash are conserved against the exogenous noise trader's fills;
    with the noise trader disabled they sum to exactly zero.
    """
    for case in range(CASES):
        rng = random.Random(case)
        n_agents = rng.randint(1, 3)
        noise = rng.choice([-5.0, 2.0, 4.0])  # -5.0 makes the noise trader never fire
        env = LOBMarketEnv(
            n_agents, n_steps=rng.randint(5, 40), seed=case, noise_intensity=noise
        )
        captured = _capture_fills(env)
        env.reset(seed=case)
        noise_inventory = 0
        noise_cash = 0
        while env.agents:
            prev_equity = dict(env._prev_equity)
            _obs, rewards, _terms, _truncs, infos = env.step(_actions(rng, env))
            ladder = json.loads(env._book.ladder())
            mid = ladder["mid"] or float(env._mid)
            for agent in env.possible_agents:
                inventory = env._inventory[agent]
                cash = env._cash[agent]
                equity = cash + inventory * mid
                assert env._prev_equity[agent] == equity
                expected = float(equity - prev_equity[agent] - env._inv_pen * inventory**2)
                assert rewards[agent] == expected
                assert infos[agent] == {"inventory": inventory, "cash": cash}
            for fill in captured[-1]:
                assert fill["maker_agent"] < n_agents, "the noise trader never rests"
                if fill["taker_agent"] == n_agents:
                    sign = 1 if fill["taker_side"] == "buy" else -1
                    noise_inventory += sign * fill["qty"]
                    noise_cash -= sign * fill["price_tick"] * fill["qty"]
            assert sum(env._inventory.values()) + noise_inventory == 0
            assert sum(env._cash.values()) + noise_cash == 0
            if noise == -5.0:
                assert noise_inventory == 0 and noise_cash == 0
                assert sum(env._inventory.values()) == 0


def test_seeded_reset_reproduces_the_first_rewards():
    """Binds ``reset`` (lob_env.py:103-118): ``reset(seed)`` twice on one instance gives
    identical reward sequences, and the equity baseline restarts at zero."""
    for case in range(CASES):
        rng = random.Random(1000 + case)
        env = LOBMarketEnv(rng.randint(1, 3), n_steps=8, seed=case)
        episodes = []
        for _ in range(2):
            env.reset(seed=case)
            assert all(value == 0.0 for value in env._prev_equity.values())
            assert all(value == 0 for value in env._inventory.values())
            assert all(value == 0.0 for value in env._cash.values())
            stream = random.Random(case)
            rewards = [env.step(_actions(stream, env))[1] for _ in range(4)]
            episodes.append(rewards)
        assert episodes[0] == episodes[1]
