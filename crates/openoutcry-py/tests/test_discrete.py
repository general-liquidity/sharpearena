"""Tests for the discrete-action adapter (Stream S9).

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/openoutcry-py/tests/test_discrete.py -q
"""

import numpy as np
import pytest

from gymnasium import spaces

from openoutcry import OpenOutcryEnv
from openoutcry.discrete import DiscreteAction


def _env(**kw) -> OpenOutcryEnv:
    return OpenOutcryEnv(n_symbols=kw.pop("n_symbols", 3), n_days=40, seed=1, **kw)


def test_multidiscrete_long_flat_short_space_and_signs():
    base = _env(max_weight=0.5)
    env = DiscreteAction(base, scheme="long_flat_short")
    n = len(base.symbols)
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert list(env.action_space.nvec) == [3] * n

    short = env.action(np.zeros(n, dtype=np.int64))
    flat = env.action(np.ones(n, dtype=np.int64))
    long = env.action(np.full(n, 2, dtype=np.int64))
    assert np.allclose(short, -0.5)
    assert np.allclose(flat, 0.0)
    assert np.allclose(long, 0.5)


def test_single_symbol_is_discrete():
    env = DiscreteAction(_env(n_symbols=1), scheme="long_flat_short")
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 3
    assert env.action(2).shape == (1,)


def test_sample_and_step():
    env = DiscreteAction(_env(), scheme="long_flat_short")
    env.reset(seed=0)
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_allow_short_false_drops_short():
    base = _env(max_weight=0.4, allow_short=False)
    env = DiscreteAction(base, scheme="long_flat_short")
    n = len(base.symbols)
    assert list(env.action_space.nvec) == [2] * n
    flat = env.action(np.zeros(n, dtype=np.int64))
    long = env.action(np.ones(n, dtype=np.int64))
    assert np.allclose(flat, 0.0)
    assert np.allclose(long, 0.4)


def test_binned_monotonic_across_bounds():
    base = _env(max_weight=1.0)
    n_bins = 5
    env = DiscreteAction(base, scheme="binned", n_bins=n_bins)
    n = len(base.symbols)
    assert list(env.action_space.nvec) == [n_bins] * n

    weights = [env.action(np.full(n, b, dtype=np.int64)) for b in range(n_bins)]
    cols = np.stack(weights, axis=0)  # (n_bins, n_symbols)
    for sym in range(n):
        col = cols[:, sym]
        assert np.all(np.diff(col) > 0), "bins must increase monotonically"
        assert np.isclose(col[0], base.action_space.low[sym])
        assert np.isclose(col[-1], base.action_space.high[sym])


def test_binned_requires_n_bins():
    with pytest.raises(ValueError):
        DiscreteAction(_env(), scheme="binned")
    with pytest.raises(ValueError):
        DiscreteAction(_env(), scheme="binned", n_bins=1)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        DiscreteAction(_env(), scheme="bogus")


def test_determinism_same_action_same_weights():
    env = DiscreteAction(_env(), scheme="binned", n_bins=4)
    n = len(env.unwrapped.symbols)
    a = np.array([0, 2, 3][:n], dtype=np.int64)
    assert np.array_equal(env.action(a), env.action(a))
