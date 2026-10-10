"""Installed vector reset and time-limit contracts, using finite synthetic lanes."""

import json

import gymnasium as gym
import numpy as np
import pytest

from sharpearena.vector import SharpeArenaVectorEnv
from sharpearena.sharpearena_py import VecTradingEnv


def assert_obs_equal(left, right):
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("seed,expected", [(41, [41, 42]), ([71, 19], [71, 19])])
def test_seeded_reset_rebuilds_native_lanes_and_readbacks(seed, expected):
    kwargs = dict(n_symbols=2, n_days=12, distribution_mode="hard",
                  vol_clustering=0.3, env_kwargs={"exec_seed": 88, "fee_bps": 7.0})
    env = SharpeArenaVectorEnv(seeds=[1, 2], **kwargs)
    before, _ = env.reset()
    env.step(np.full((2, 2), 0.2, dtype=np.float32))
    env.step_async(np.zeros((2, 2), dtype=np.float32))
    obs, info = env.reset(seed=seed)
    fresh = SharpeArenaVectorEnv(seeds=expected, **kwargs)
    expected_obs, _ = fresh.reset()
    assert_obs_equal(obs, expected_obs)
    assert not np.array_equal(obs["closes"], before["closes"])
    assert env.scenario_seeds == expected == env._env.scenario_seeds
    assert info["scenario_seed"].tolist() == expected
    assert env._env.effective_configs == fresh._env.effective_configs
    with pytest.raises(RuntimeError, match="without a pending"):
        env.step_wait()
    action = np.full((2, 2), 0.2, dtype=np.float32)
    actual = env.step(action)
    reference = fresh.step(action)
    assert_obs_equal(actual[0], reference[0])
    np.testing.assert_array_equal(actual[1], reference[1])
    assert_obs_equal(env.reset()[0], expected_obs)
    assert_obs_equal(env.reset(seed=seed)[0], expected_obs)


@pytest.mark.parametrize("seed", [-1, 1.5, True, [1], [1, 2, 3], [1, -1],
                                  [1, 2.5], [1, True], 2**63 - 1])
def test_invalid_reset_is_atomic(seed):
    env = SharpeArenaVectorEnv(seeds=[1, 2], n_symbols=1, n_days=12)
    original = env._env
    env.step_async(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        env.reset(seed=seed)
    assert env._env is original
    assert env.scenario_seeds == [1, 2]
    env.step_wait()  # Validation must not discard a queued action.


def test_eval_explicit_reset_seeds_are_literal_not_shifted():
    env = SharpeArenaVectorEnv(num_envs=2, mode="eval", n_symbols=1, n_days=10)
    assert env.scenario_seeds == [1_000_000, 1_000_001]
    env.reset(seed=21)
    assert env._env.scenario_seeds == [21, 22]


@pytest.mark.parametrize("mode", ["next_step", "same_step", "disabled"])
def test_episode_cap_counts_executed_steps_and_preserves_autoreset(mode):
    env = SharpeArenaVectorEnv(seeds=[1, 2], n_symbols=2, n_days=12,
                              max_episode_steps=2, autoreset_mode=mode)
    initial, _ = env.reset(seed=31)
    fresh = SharpeArenaVectorEnv(seeds=[31, 32], n_symbols=2, n_days=12,
                                autoreset_mode="disabled")
    fresh.reset()
    action = np.full((2, 2), 0.2, dtype=np.float32)
    first = env.step(action)
    assert not first[2].any() and not first[3].any()
    assert not first[4]["first"].any()
    fresh.step(action)
    terminal_reference = fresh.step(action)
    ending = env.step(action)
    assert ending[3].all() and not ending[2].any()
    np.testing.assert_array_equal(ending[1], terminal_reference[1])
    assert ending[4]["scenario_seed"].tolist() == [31, 32]
    if mode == "same_step":
        assert ending[4]["first"].all()
        assert_obs_equal(ending[0], initial)
        for lane in range(2):
            assert_obs_equal(ending[4]["final_obs"][lane],
                             {key: value[lane] for key, value in terminal_reference[0].items()})
        assert not env.step(action)[3].any()
        assert env.step(action)[3].all()
    elif mode == "next_step":
        assert_obs_equal(ending[0], terminal_reference[0])
        resetting = env.step(action)
        assert resetting[4]["first"].all()
        assert not resetting[2].any() and not resetting[3].any()
        np.testing.assert_array_equal(resetting[1], [0.0, 0.0])
        assert_obs_equal(resetting[0], initial)
        assert not env.step(action)[3].any()
        assert env.step(action)[3].all()
    else:
        assert_obs_equal(ending[0], terminal_reference[0])
        for _ in range(3):
            frozen = env.step(-action)
            assert frozen[3].all() and not frozen[4]["first"].any()
            np.testing.assert_array_equal(frozen[1], [0.0, 0.0])
            assert_obs_equal(frozen[0], ending[0])
            np.testing.assert_array_equal(frozen[4]["nav"], ending[4]["nav"])
    # Explicit reset also clears pending reset/cap state and elapsed counters.
    assert_obs_equal(env.reset()[0], initial)
    assert not env.step(action)[3].any()


@pytest.mark.parametrize("limit", [None, 100])
def test_native_horizon_precedes_a_longer_cap(limit):
    env = SharpeArenaVectorEnv(seeds=[1], n_symbols=1, n_days=3,
                              max_episode_steps=limit, autoreset_mode="next_step")
    env.reset()
    action = np.zeros((1, 1), dtype=np.float32)
    assert not env.step(action)[3][0]
    assert not env.step(action)[3][0]
    assert env.step(action)[3][0]
    assert env.step(action)[4]["first"][0]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, float("inf")])
def test_invalid_episode_caps_are_refused(limit):
    with pytest.raises(ValueError, match="positive integer"):
        SharpeArenaVectorEnv(seeds=[1], max_episode_steps=limit)


def test_native_binding_applies_cap_and_refuses_zero():
    with pytest.raises(ValueError, match="positive"):
        VecTradingEnv(seeds=[1], max_episode_steps=0)
    env = VecTradingEnv(seeds=[1], n_days=10, max_episode_steps=1)
    env.reset_batch()
    out = json.loads(env.step_batch('[{"orders": [], "reasoning": "test"}]'))
    assert out["truncated"] == [True]
    assert out["terminated"] == [False]


def test_registered_vector_entry_point_honors_custom_cap():
    env = gym.make_vec("SharpeArena/Hard-v1", num_envs=2, max_episode_steps=1)
    try:
        env.reset(seed=41)
        out = env.step(np.zeros((2, 4), dtype=np.float32))
        assert out[3].all()
        assert out[4]["scenario_seed"].tolist() == [41, 42]
    finally:
        env.close()
