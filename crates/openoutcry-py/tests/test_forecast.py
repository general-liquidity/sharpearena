"""Tests for the FinPILOT calibrated forecast-skill axis (``openoutcry.forecast``).

The calibration math (``calibrated_forecast`` / ``oos_r2``) is numpy-only and always
runs. The env-wrapper and curve tests need the native binding via ``OpenOutcryEnv``
and ``score_run`` and are skip-guarded if it is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from openoutcry.forecast import (
    calibrated_forecast,
    oos_r2,
    trailing_mean_baseline,
    ForecastChannelObservation,
    forecast_skill_curve,
)


def _returns(seed: int = 7, n: int = 400) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.001, 0.01, n)


def test_calibration_hits_target_r2():
    r = _returns()
    for target in (0.001, 0.01, 0.05, 0.1, 0.2, 0.4):
        f = calibrated_forecast(r, target, seed=3)
        assert abs(oos_r2(r, f) - target) < 1e-6


def test_zero_skill_is_no_skill():
    r = _returns()
    f = calibrated_forecast(r, 0.0, seed=11)
    assert abs(oos_r2(r, f)) < 1e-9
    # With no noise the no-skill forecast is exactly the trailing-mean baseline.
    f0 = calibrated_forecast(r, 0.0, seed=11, noise_scale=0.0)
    assert np.allclose(f0, trailing_mean_baseline(r))


def test_perfect_skill_is_truth():
    r = _returns()
    f = calibrated_forecast(r, 1.0, seed=5)
    assert np.allclose(f, r)
    assert oos_r2(r, f) > 0.999999


def test_determinism():
    r = _returns()
    a = calibrated_forecast(r, 0.1, seed=42)
    b = calibrated_forecast(r, 0.1, seed=42)
    c = calibrated_forecast(r, 0.1, seed=43)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def _make_env(seed: int):
    from openoutcry import OpenOutcryEnv

    return OpenOutcryEnv(n_symbols=3, n_days=80, seed=seed)


def _binding_ok() -> bool:
    try:
        from openoutcry import OpenOutcryEnv  # noqa: F401
        from openoutcry.openoutcry_py import score_run  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _binding_ok(), reason="native binding unavailable")
def test_wrapper_adds_forecast_channel():
    env = ForecastChannelObservation(_make_env(0), target_r2=0.1, seed=0)
    n = len(env.unwrapped.symbols)
    assert "forecast" in env.observation_space.spaces
    obs, _info = env.reset()
    assert obs["forecast"].shape == (n,)
    assert np.all(np.isfinite(obs["forecast"]))
    obs, _r, _term, _trunc, _i = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert obs["forecast"].shape == (n,)
    # Eval-only leakage surface: the base (leak-free) obs never carries it.
    assert "forecast" not in env.unwrapped.observation_space.spaces


@pytest.mark.skipif(not _binding_ok(), reason="native binding unavailable")
def test_forecast_skill_curve_one_entry_per_grid_point():
    grid = (0.001, 0.1)
    curve = forecast_skill_curve(
        _make_env, seeds=[0, 1], r2_grid=grid, max_steps=64
    )
    assert set(curve.keys()) == {float(g) for g in grid}
    assert all(np.isfinite(v) for v in curve.values())
