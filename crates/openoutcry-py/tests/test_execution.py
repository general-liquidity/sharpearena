"""Tests for the VWAP/TWAP optimal-execution env (:mod:`openoutcry.execution`)."""

import math

import numpy as np
import pytest

from openoutcry.execution import (
    ExecutionEnv,
    execution_quality,
    twap_policy,
    immediate_policy,
)


def _rollout(env, policy):
    obs, _info = env.reset(seed=7)
    traj = []
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append(info)
        done = terminated or truncated
    return traj


def test_constructs_and_reset_obs_keys():
    env = ExecutionEnv(window=12, seed=1, history_len=5)
    obs, info = env.reset(seed=1)
    assert set(obs) == {"leftover_order", "leftover_time", "prices"}
    assert obs["leftover_order"][0] == 1.0
    assert obs["leftover_time"][0] == 1.0
    assert obs["prices"].shape == (5,)
    assert env.observation_space.contains(obs)
    assert info["scenario_seed"] == 1


def test_step_works_and_window_terminates():
    env = ExecutionEnv(window=8, seed=2)
    env.reset(seed=2)
    steps = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([0.3], dtype=np.float32))
        assert math.isfinite(reward)
        assert not truncated
        done = terminated
        steps += 1
        assert steps <= 100
    assert steps == 8


def test_determinism_same_seed():
    def run(seed):
        env = ExecutionEnv(window=10, seed=seed)
        return [r["reward"] for r in _record_rewards(env)]

    assert run(3) == run(3)
    assert run(3) != run(4)


def _record_rewards(env):
    obs, _ = env.reset()
    out = []
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([0.2], dtype=np.float32))
        out.append({**info, "reward": reward})
        done = terminated or truncated
    return out


def test_running_average_is_causal():
    """The running average at bar t must depend only on prices < t, never the future."""
    base = np.linspace(100.0, 110.0, 10)
    perturbed = base.copy()
    perturbed[6:] += 50.0  # only the future (bars >= 6) changes

    a = ExecutionEnv(window=10, price_path=base)
    b = ExecutionEnv(window=10, price_path=perturbed)
    ra, rb = _record_rewards(a), _record_rewards(b)

    for t in range(6):
        assert ra[t]["running_avg"] == rb[t]["running_avg"]
    # And it equals the explicit causal mean of prior bars.
    for t in range(1, 10):
        assert math.isclose(ra[t]["running_avg"], float(np.mean(base[:t])), rel_tol=1e-12)
    assert ra[0]["running_avg"] == base[0]


def test_twap_completes_order():
    env = ExecutionEnv(window=10, seed=5)
    traj = _rollout(env, lambda o: twap_policy(o, env.window))
    assert traj[-1]["completed"] is True
    assert traj[-1]["forced_remainder"] == pytest.approx(0.0, abs=1e-9)
    assert traj[-1]["leftover_order"] == pytest.approx(0.0, abs=1e-9)
    # Uniform child orders: every bar fills the same fraction of the parent.
    childs = [r["child"] for r in traj]
    assert max(childs) - min(childs) < 1e-6  # float32 action precision
    assert sum(childs) == pytest.approx(1.0, abs=1e-6)


def test_immediate_fills_bar_zero():
    env = ExecutionEnv(window=10, seed=5)
    traj = _rollout(env, immediate_policy)
    assert traj[0]["executed"] == pytest.approx(1.0, abs=1e-9)
    assert sum(r["executed"] for r in traj[1:]) == pytest.approx(0.0, abs=1e-9)
    assert traj[-1]["completed"] is True


def test_completion_check_fires_on_unfilled_order():
    env = ExecutionEnv(window=10, seed=5)
    # A do-nothing policy never fills, so the window closes with the order open.
    traj = _rollout(env, lambda o: np.array([0.0], dtype=np.float32))
    assert traj[-1]["completed"] is False
    assert traj[-1]["forced_remainder"] == pytest.approx(1.0, abs=1e-9)


def test_execution_quality_finite_fields():
    env = ExecutionEnv(window=12, seed=9)
    traj = _rollout(env, lambda o: twap_policy(o, env.window))
    q = execution_quality(traj)
    assert set(q) == {"shortfall_bps", "participation_variance", "completion_fraction"}
    assert all(math.isfinite(v) for v in q.values())
    assert q["completion_fraction"] == pytest.approx(1.0, abs=1e-9)
    # TWAP is a uniform schedule, so participation variance is ~0.
    assert q["participation_variance"] == pytest.approx(0.0, abs=1e-9)


def test_execution_quality_distinguishes_incomplete():
    env = ExecutionEnv(window=10, seed=5)
    traj = _rollout(env, lambda o: np.array([0.0], dtype=np.float32))
    q = execution_quality(traj)
    assert q["completion_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(q["shortfall_bps"])
