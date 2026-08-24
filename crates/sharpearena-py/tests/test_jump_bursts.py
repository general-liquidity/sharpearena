"""Opt-in jump-burst driver: defaults stay stable and a configured tape replays."""

from __future__ import annotations

import json

from sharpearena import TradingEnv


def _closes(env: TradingEnv) -> list[str]:
    frames = [env.reset()]
    flat = json.dumps({"orders": []})
    done = False
    while not done:
        obs, _reward, done, _info = env.step(flat)
        frames.append(obs)
    return frames


def test_zero_burst_arguments_preserve_the_native_default() -> None:
    base = TradingEnv(n_symbols=3, n_days=40, seed=17, distribution_mode="calm")
    explicit = TradingEnv(
        n_symbols=3,
        n_days=40,
        seed=17,
        distribution_mode="calm",
        jump_burst_probability=0.0,
        jump_burst_persistence=0.9,
        jump_burst_size=0.1,
    )
    assert _closes(base) == _closes(explicit)


def test_burst_configuration_is_deterministic_and_changes_the_tape() -> None:
    kwargs = dict(
        n_symbols=3,
        n_days=40,
        seed=17,
        distribution_mode="calm",
        jump_burst_probability=0.05,
        jump_burst_persistence=0.8,
        jump_burst_size=0.08,
    )
    a = _closes(TradingEnv(**kwargs))
    b = _closes(TradingEnv(**kwargs))
    plain = _closes(TradingEnv(n_symbols=3, n_days=40, seed=17, distribution_mode="calm"))
    assert a == b
    assert a != plain
