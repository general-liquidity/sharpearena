"""Tests for the canonical preprocessing defaults + one-call constructor.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_preprocessing.py -q
"""

import numpy as np
import pytest

from sharpearena.preprocessing import (
    ExecutionNoiseConfig,
    PreprocessingConfig,
    CANONICAL_PREPROCESSING,
    make_preprocessed_env,
    describe_preprocessing,
)
from sharpearena.wrappers import CausalNormalizeObservation

_ENV_KWARGS = dict(n_symbols=3, n_days=40, seed=1)


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def test_canonical_is_frozen_default():
    assert CANONICAL_PREPROCESSING == PreprocessingConfig()
    assert CANONICAL_PREPROCESSING.lookback == 1
    assert CANONICAL_PREPROCESSING.causal_normalize_obs is True
    assert CANONICAL_PREPROCESSING.causal_normalize_reward is False
    assert CANONICAL_PREPROCESSING.reward_clip is None
    assert CANONICAL_PREPROCESSING.flatten is False
    assert CANONICAL_PREPROCESSING.execution_noise.enabled is False


def test_defaults_build_and_step():
    env = make_preprocessed_env(**_ENV_KWARGS)
    obs, info = env.reset()
    assert set(obs) == {"closes", "positions", "cash"}
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(_zero_action(env))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_lookback_stacks_frames():
    cfg = PreprocessingConfig(lookback=4)
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    obs, _info = env.reset()
    # FrameStack adds a leading axis of length `lookback`.
    assert obs["closes"].shape == (4, 3)


def test_custom_flatten_lookback_reward_clip():
    cfg = PreprocessingConfig(flatten=True, lookback=4, reward_clip=1.0)
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    obs, _info = env.reset()
    # flatten => single 1-D Box.
    assert isinstance(obs, np.ndarray) and obs.ndim == 1
    assert env.observation_space.contains(obs)
    # reward clip bounds the per-step reward.
    _obs, reward, _t, _tr, _info = env.step(_zero_action(env))
    assert -1.0 <= reward <= 1.0


def test_near_raw_config_skips_normalization():
    cfg = PreprocessingConfig(causal_normalize_obs=False, lookback=1)
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    raw = make_preprocessed_env(  # the bare base env for comparison
        PreprocessingConfig(causal_normalize_obs=False, lookback=1), **_ENV_KWARGS
    )
    obs_cfg, _ = env.reset()
    obs_raw, _ = raw.reset()
    # No normalization => raw close prices (strictly positive), not z-scores.
    assert np.all(obs_cfg["closes"] > 0.0)
    np.testing.assert_allclose(obs_cfg["closes"], obs_raw["closes"])


def test_normalize_default_changes_first_obs_to_zeros():
    # Causal normalize emits zeros on the first (history-free) observation.
    env = make_preprocessed_env(CANONICAL_PREPROCESSING, **_ENV_KWARGS)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["closes"], np.zeros_like(obs["closes"]))


def test_causal_normalize_is_leak_free_delegation():
    # The pipeline reuses the existing causal wrapper (the leak-free implementation),
    # rather than re-implementing normalization. Light structural check: the wrapper
    # stack contains a CausalNormalizeObservation layer.
    env = make_preprocessed_env(CANONICAL_PREPROCESSING, **_ENV_KWARGS)
    found = False
    cur = env
    while hasattr(cur, "env"):
        if isinstance(cur, CausalNormalizeObservation):
            found = True
            break
        cur = cur.env
    assert found, "canonical pipeline must delegate to CausalNormalizeObservation"


def test_execution_noise_applied_when_enabled():
    cfg = PreprocessingConfig(
        causal_normalize_obs=False,
        execution_noise=ExecutionNoiseConfig(slippage_bps=50.0, seed=7),
    )
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    env.reset()
    # Should step without error; the noise layer is in the stack.
    _obs, reward, _t, _tr, _info = env.step(_zero_action(env))
    assert np.isfinite(reward)


def test_describe_renders_knobs():
    text = describe_preprocessing(CANONICAL_PREPROCESSING)
    for key in (
        "lookback",
        "causal_normalize_obs",
        "causal_normalize_reward",
        "reward_clip",
        "max_episode_steps",
        "flatten",
        "delay_prob",
        "slippage_bps",
    ):
        assert key in text
    assert "CANONICAL" in text

    noisy = describe_preprocessing(PreprocessingConfig(reward_clip=2.0))
    assert "[DISCLOSE]" in noisy
    assert "NON-CANONICAL" in noisy


def test_max_episode_steps_truncates():
    cfg = PreprocessingConfig(max_episode_steps=3)
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    env.reset()
    steps, truncated = 0, False
    while not truncated and steps < 100:
        _obs, _r, terminated, truncated, _info = env.step(_zero_action(env))
        steps += 1
        if terminated:
            break
    assert steps <= 3


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PreprocessingConfig(lookback=0)
    with pytest.raises(ValueError):
        PreprocessingConfig(reward_clip=0.0)
    with pytest.raises(ValueError):
        PreprocessingConfig(max_episode_steps=0)
