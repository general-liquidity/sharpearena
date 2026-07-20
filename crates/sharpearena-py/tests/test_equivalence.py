"""Equivalence gate: the batched SharpeArena path is the same env as the scalar path.

ALE ships its vectorized env alongside a written equivalence statement so callers trust the
fast path is byte-for-byte the env they would get from ``N`` single envs plus the autoreset
wrapper. These tests make that statement executable for SharpeArena:

- ``test_batched_equals_scalar`` drives a ``B``-lane ``SharpeArenaVectorEnv`` and ``B`` separate
  ``SharpeArenaEnv`` over identical per-lane actions to episode end and asserts the
  observations / rewards / terminated / truncated agree lane-for-lane, byte-identical.
- ``test_batched_equals_scalar_under_execution_noise`` repeats the proof one lane at a time
  with the default stochastic execution model active.
- ``test_step_async_step_wait_equals_step`` pins the async send/recv split to ``step``.

Seed note: ``SharpeArenaEnv`` splits each user seed into decorrelated (scenario, execution)
streams via ``SeedSequence``; ``SharpeArenaVectorEnv`` takes the lane (scenario) seeds
directly and exposes a single global execution seed. The tests bridge the two by
constructing the vector over each scalar's resolved scenario seed so both wrap the same
point-in-time price path; ``test_batched_equals_scalar`` zeroes execution noise so the
single global execution stream cannot diverge from the scalars' per-seed ones, while
``test_batched_equals_scalar_under_execution_noise`` reproduces a scalar's exact
(scenario, execution) pair in one lane to keep the default noise model in the proof.
"""

import numpy as np
import pytest

from sharpearena.gym import SharpeArenaEnv
from sharpearena.vector import SharpeArenaVectorEnv


def _obs_equal(a, b) -> bool:
    return (
        np.array_equal(a["closes"], b["closes"])
        and np.array_equal(a["positions"], b["positions"])
        and np.array_equal(a["cash"], b["cash"])
    )


def _lane_obs(batch, i) -> dict:
    return {k: batch[k][i] for k in ("closes", "positions", "cash")}


def test_batched_equals_scalar():
    seeds = [0, 1, 5, 7]
    n_symbols, n_days = 3, 40
    # slippage_bps=0 makes execution deterministic, so the single global execution stream
    # the batched env exposes cannot diverge from the scalars' per-seed ones; the scenario
    # (price-path) seed is then the whole env and is matched lane-for-lane below.
    env_kwargs = {"slippage_bps": 0.0}
    scalars = [
        SharpeArenaEnv(
            seed=s, n_symbols=n_symbols, n_days=n_days, env_kwargs=dict(env_kwargs)
        )
        for s in seeds
    ]
    scenario_seeds = [e._resolved_seeds["scenario"] for e in scalars]
    vec = SharpeArenaVectorEnv(
        seeds=scenario_seeds,
        n_symbols=n_symbols,
        n_days=n_days,
        autoreset_mode="same_step",
        env_kwargs=dict(env_kwargs),
    )

    s_obs = [e.reset()[0] for e in scalars]
    v_obs, _ = vec.reset()
    for i in range(len(seeds)):
        assert _obs_equal(s_obs[i], _lane_obs(v_obs, i))

    b = len(seeds)
    done = [False] * b
    steps = 0
    while not all(done):
        actions = np.full((b, n_symbols), 0.15, dtype=np.float32)
        v_obs, v_rew, v_term, v_trunc, v_info = vec.step(actions)
        for i, env in enumerate(scalars):
            if done[i]:
                continue
            o, r, term, trunc, _ = env.step(actions[i])
            assert r == v_rew[i]
            assert term == v_term[i]
            assert trunc == v_trunc[i]
            ended = bool(term or trunc)
            # same_step recycles a finished lane in place: its terminal obs is in
            # final_obs[i] while obs[i] is already the next episode's t0.
            target = v_info["final_obs"][i] if ended else _lane_obs(v_obs, i)
            assert _obs_equal(o, target)
            if ended:
                done[i] = True
        steps += 1
        assert steps < 1000, "episode never ended"


@pytest.mark.parametrize("seed", [0, 3, 7, 42])
def test_batched_equals_scalar_under_execution_noise(seed):
    n_symbols, n_days = 3, 40
    scalar = SharpeArenaEnv(seed=seed, n_symbols=n_symbols, n_days=n_days)
    resolved = scalar._resolved_seeds
    # Reproduce the scalar's exact (scenario, execution) native construction in one lane,
    # so the default stochastic fill model is part of the equivalence proof.
    vec = SharpeArenaVectorEnv(
        seeds=[resolved["scenario"]],
        n_symbols=n_symbols,
        n_days=n_days,
        autoreset_mode="same_step",
        env_kwargs={"exec_seed": resolved["execution"]},
    )

    s_obs, _ = scalar.reset()
    v_obs, _ = vec.reset()
    assert _obs_equal(s_obs, _lane_obs(v_obs, 0))

    for _ in range(1000):
        actions = np.full(n_symbols, 0.2, dtype=np.float32)
        o, r, term, trunc, _ = scalar.step(actions)
        v_obs, v_rew, v_term, v_trunc, v_info = vec.step(actions.reshape(1, n_symbols))
        assert r == v_rew[0]
        assert term == v_term[0]
        assert trunc == v_trunc[0]
        ended = bool(term or trunc)
        target = v_info["final_obs"][0] if ended else _lane_obs(v_obs, 0)
        assert _obs_equal(o, target)
        if ended:
            break
    else:
        pytest.fail("episode never ended")


def test_step_async_step_wait_equals_step():
    n_symbols = 3
    a = SharpeArenaVectorEnv(seeds=[2, 3, 4], n_symbols=n_symbols, n_days=30)
    b = SharpeArenaVectorEnv(seeds=[2, 3, 4], n_symbols=n_symbols, n_days=30)
    a.reset()
    b.reset()
    for _ in range(20):
        actions = np.full((3, n_symbols), 0.1, dtype=np.float32)
        a_obs, a_rew, a_term, a_trunc, _ = a.step(actions)
        b.step_async(actions)
        b_obs, b_rew, b_term, b_trunc, _ = b.step_wait()
        for i in range(3):
            assert _obs_equal(_lane_obs(a_obs, i), _lane_obs(b_obs, i))
        assert np.array_equal(a_rew, b_rew)
        assert np.array_equal(a_term, b_term)
        assert np.array_equal(a_trunc, b_trunc)


def test_step_wait_without_pending_raises():
    env = SharpeArenaVectorEnv(seeds=[1], n_symbols=2, n_days=20)
    env.reset()
    with pytest.raises(RuntimeError):
        env.step_wait()


def test_step_async_twice_raises():
    env = SharpeArenaVectorEnv(seeds=[1], n_symbols=2, n_days=20)
    env.reset()
    actions = np.zeros((1, 2), dtype=np.float32)
    env.step_async(actions)
    with pytest.raises(RuntimeError):
        env.step_async(actions)
    env.step_wait()
