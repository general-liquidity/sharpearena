"""Malformed vector actions must not partially advance a native environment."""

import numpy as np
import pytest

from sharpearena.gym import SharpeArenaEnv
from sharpearena.vector import SharpeArenaVectorEnv


def make_env(vector, **kwargs):
    common = dict(n_symbols=2, n_days=40, max_weight=0.25, allow_short=False)
    common.update(kwargs)
    return (SharpeArenaVectorEnv(num_envs=2, **common) if vector
            else SharpeArenaEnv(**common))


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("bad", [
    [0.1], [0.1, 0.1, 0.1], [0.1, 0.1, np.nan], [[0.1], [0.1]],
    [np.nan, 0.0], [np.inf, 0.0], [-0.1, 0.1], [0.3, 0.0],
    [True, False], ["0.1", "0.0"], [0.1j, 0.0],
])
def test_invalid_actions_refused_before_any_lane_advances(vector, bad):
    env, reference = make_env(vector), make_env(vector)
    env.reset()
    reference.reset()
    action = np.array([bad, bad]) if vector else np.array(bad)
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    valid = np.array([[0.25, 0.0]] * 2 if vector else [0.25, 0.0])
    actual, expected = env.step(valid), reference.step(valid)
    for key in actual[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    for index in (1, 2, 3):
        np.testing.assert_array_equal(actual[index], expected[index])


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("limit", [0, -1, np.nan, np.inf, True])
def test_invalid_weight_limit_is_refused(vector, limit):
    with pytest.raises(ValueError, match="max_weight"):
        make_env(vector, max_weight=limit)


@pytest.mark.parametrize("vector", [False, True])
def test_action_space_boundary_is_exact_and_signed_when_enabled(vector):
    env = make_env(vector, allow_short=True, max_weight=0.1)
    env.reset()
    # Float32 Box endpoints must themselves be valid even when the requested
    # decimal limit cannot be exactly represented in float32.
    env.step(env.action_space.low.copy())
    env.step(env.action_space.high.copy())
    outside = env.action_space.high.astype(np.float64)
    outside.flat[0] = np.nextafter(outside.flat[0], np.inf)
    with pytest.raises(ValueError, match="bounds"):
        env.step(outside)


def test_async_action_is_a_validated_snapshot_not_a_borrowed_buffer():
    env, reference = make_env(True), make_env(True)
    env.reset()
    reference.reset()
    actions = np.full((2, 2), 0.1)
    env.step_async(actions)
    expected = reference.step(actions.copy())
    actions.fill(0.25)
    actual = env.step_wait()
    for key in actual[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_reset_discards_a_pending_action():
    env = make_env(True)
    env.step_async(np.full((2, 2), 0.1))
    env.reset()
    with pytest.raises(RuntimeError, match="without a pending"):
        env.step_wait()
