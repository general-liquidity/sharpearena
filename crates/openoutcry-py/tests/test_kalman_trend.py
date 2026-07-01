"""Tests for :class:`openoutcry.obs_extra.KalmanTrendObservation`.

The velocity / reversal tests run against a fake Dict-obs env replaying a
*constructed* price path (trend then reversal) so the sign of the true velocity is
known; the integration test over a real :class:`OpenOutcryEnv` is skipped when the
compiled binding is unavailable.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_kalman_trend.py -q
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.obs_extra import KalmanTrendObservation


class _FakeDictEnv(gym.Env):
    """Minimal Dict-obs env that replays a fixed per-symbol close series."""

    def __init__(self, closes: np.ndarray) -> None:
        self._closes = np.asarray(closes, dtype=np.float64)  # (T, n_symbols)
        self._t = 0
        n = self._closes.shape[1]
        self.observation_space = spaces.Dict(
            {
                "closes": spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float64),
                "cash": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def _obs(self) -> dict:
        return {
            "closes": self._closes[self._t].copy(),
            "cash": np.array([0.0], dtype=np.float64),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        done = self._t >= self._closes.shape[0] - 1
        return self._obs(), 0.0, False, bool(done), {}


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _run(env, n_steps: int):
    obs, _ = env.reset()
    vel = [obs["kalman_velocity"].copy()]
    sign = [obs["kalman_velocity_sign"].copy()]
    for _ in range(n_steps):
        obs, _r, term, trunc, _i = env.step(_zero_action(env))
        vel.append(obs["kalman_velocity"].copy())
        sign.append(obs["kalman_velocity_sign"].copy())
        if term or trunc:
            break
    return np.asarray(vel), np.asarray(sign)


def test_adds_keys_with_shape_and_space():
    closes = np.column_stack([np.linspace(100, 120, 40), np.linspace(50, 40, 40)])
    env = KalmanTrendObservation(_FakeDictEnv(closes))
    obs, _ = env.reset()
    assert obs["kalman_velocity"].shape == (2,)
    assert obs["kalman_velocity_sign"].shape == (2,)
    assert env.observation_space.contains(obs)


def test_first_obs_is_warmup_zero():
    closes = np.column_stack([np.linspace(100, 120, 30)])
    env = KalmanTrendObservation(_FakeDictEnv(closes))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["kalman_velocity"], np.zeros(1))
    np.testing.assert_allclose(obs["kalman_velocity_sign"], np.zeros(1))


def test_velocity_tracks_a_steady_uptrend():
    # A clean +2/bar ramp: filtered velocity should settle positive near the slope.
    closes = np.column_stack([100.0 + 2.0 * np.arange(80)])
    env = KalmanTrendObservation(_FakeDictEnv(closes), process_var=1e-2)
    vel, sign = _run(env, 79)
    assert vel[-1, 0] > 0.0
    assert sign[-1, 0] == 1.0
    assert vel[-1, 0] == pytest.approx(2.0, abs=0.5)


def test_velocity_flips_sign_on_reversal():
    up = 100.0 + 2.0 * np.arange(60)
    down = up[-1] - 2.0 * np.arange(1, 61)
    closes = np.column_stack([np.concatenate([up, down])])
    env = KalmanTrendObservation(_FakeDictEnv(closes), process_var=1e-2)
    vel, sign = _run(env, closes.shape[0] - 1)
    # Positive while trending up (sampled inside the up-leg, past warmup).
    assert sign[40, 0] == 1.0
    assert vel[40, 0] > 0.0
    # Negative by the end of the down-leg.
    assert sign[-1, 0] == -1.0
    assert vel[-1, 0] < 0.0


def test_per_symbol_independence():
    up = 100.0 + 1.5 * np.arange(70)
    down = 200.0 - 1.5 * np.arange(70)
    closes = np.column_stack([up, down])
    env = KalmanTrendObservation(_FakeDictEnv(closes), process_var=1e-2)
    vel, sign = _run(env, 69)
    assert sign[-1, 0] == 1.0 and vel[-1, 0] > 0.0
    assert sign[-1, 1] == -1.0 and vel[-1, 1] < 0.0


def test_is_leak_free_prefix_reproduces_velocity():
    rng = np.random.default_rng(3)
    path = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120))
    closes = np.column_stack([path])
    full = KalmanTrendObservation(_FakeDictEnv(closes), process_var=1e-2)
    vel, _sign = _run(full, 119)
    for t in (5, 30, 90, vel.shape[0] - 1):
        prefix = KalmanTrendObservation(
            _FakeDictEnv(closes[: t + 1]), process_var=1e-2
        )
        pvel, _ps = _run(prefix, t)
        np.testing.assert_allclose(pvel[-1], vel[t], atol=1e-9)


def test_sign_deadband_zeroes_small_velocity():
    closes = np.column_stack([100.0 + 2.0 * np.arange(60)])
    env = KalmanTrendObservation(
        _FakeDictEnv(closes), process_var=1e-2, sign_eps=1e6
    )
    _vel, sign = _run(env, 59)
    # An enormous dead-band forces every sign to 0 regardless of velocity.
    assert np.all(sign == 0.0)


def test_rejects_bad_hyperparameters():
    closes = np.column_stack([100.0 + np.arange(10)])
    with pytest.raises(ValueError):
        KalmanTrendObservation(_FakeDictEnv(closes), process_var=0.0)
    with pytest.raises(ValueError):
        KalmanTrendObservation(_FakeDictEnv(closes), obs_var=0.0)
    with pytest.raises(ValueError):
        KalmanTrendObservation(_FakeDictEnv(closes), sign_eps=-1.0)


# -- integration over the real env -------------------------------------------

try:
    import openoutcry.openoutcry_py  # noqa: F401

    _HAVE_BINDING = True
except Exception:  # pragma: no cover - environment-dependent
    _HAVE_BINDING = False

_needs_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native binding not built"
)


@_needs_binding
def test_wraps_real_env():
    from openoutcry.gym import OpenOutcryEnv

    base = OpenOutcryEnv(n_symbols=3, n_days=80, seed=1)
    env = KalmanTrendObservation(base)
    obs, _ = env.reset()
    n = len(base.symbols)
    assert obs["kalman_velocity"].shape == (n,)
    assert obs["kalman_velocity_sign"].shape == (n,)
    assert env.observation_space.contains(obs)
    for _ in range(30):
        obs, _r, term, trunc, _i = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        assert np.all(np.isfinite(obs["kalman_velocity"]))
        assert np.all(np.isin(obs["kalman_velocity_sign"], (-1.0, 0.0, 1.0)))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
