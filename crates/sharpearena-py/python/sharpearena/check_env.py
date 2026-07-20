"""A conformance harness that *proves* SharpeArena's determinism thesis.

Modeled on :func:`gymnasium.utils.env_checker.check_env`, but specialized to the
one claim that makes a leak-free, point-in-time market worth publishing:
``reset`` is byte-for-byte reproducible. The standard checker tolerates stochastic
resets; here a same-seed reset that differs by a single ULP is a failure.

``check_env(env)`` validates the gymnasium contract (space membership, 5-tuple
shape, finite rewards, a rollout to a terminal/truncated boundary) and the
**determinism core**: two ``reset()`` calls on the same env yield decoded
observation arrays that compare equal under :func:`numpy.array_equal` (exact, not
``allclose``). Pass ``make_env_for_seed`` to additionally prove that same-seed
constructions are byte-identical and different-seed constructions diverge — the
proof that the seed actually drives the market, not just the RNG.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

Obs = Mapping[str, np.ndarray]


def _decoded_equal(a: Obs, b: Obs) -> bool:
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def _assert_in_space(env, obs: Obs, where: str) -> None:
    assert env.observation_space.contains(obs), (
        f"{where}: observation is not contained in env.observation_space "
        f"(keys={sorted(obs)}, space={env.observation_space})"
    )


def _equal_weight_action(env) -> np.ndarray:
    n = int(np.prod(env.action_space.shape))
    a = np.full((n,), 1.0 / n, dtype=env.action_space.dtype)
    return np.clip(a, env.action_space.low, env.action_space.high)


def check_env(
    env,
    *,
    make_env_for_seed: Optional[Callable[[int], object]] = None,
    rollout_cap: int = 1000,
) -> None:
    """Validate ``env`` against the SharpeArena conformance contract.

    Raises :class:`AssertionError` with a descriptive message on any violation.
    """
    # -- reset() contract --------------------------------------------------
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2, (
        "reset() must return a (obs, info) 2-tuple per gymnasium 1.x"
    )
    obs, info = out
    assert isinstance(info, dict), "reset() info must be a dict"
    _assert_in_space(env, obs, "reset")

    # -- determinism core: reset() is byte-identical on repeat -------------
    obs_a, _ = env.reset()
    obs_b, _ = env.reset()
    assert _decoded_equal(obs_a, obs_b), (
        "DETERMINISM VIOLATION: two reset() calls on the same env produced "
        "different decoded observations; compared exactly with np.array_equal"
    )

    # -- step() contract ---------------------------------------------------
    env.reset()
    action = _equal_weight_action(env)
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5, (
        "step() must return a 5-tuple (obs, reward, terminated, truncated, info)"
    )
    obs, reward, terminated, truncated, info = result
    assert isinstance(terminated, (bool, np.bool_)), "terminated must be a bool"
    assert isinstance(truncated, (bool, np.bool_)), "truncated must be a bool"
    assert np.isfinite(reward), f"reward must be finite, got {reward!r}"
    assert isinstance(info, dict), "step() info must be a dict"
    _assert_in_space(env, obs, "step")

    # -- a rollout to a terminal/truncated boundary stays finite -----------
    env.reset()
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"reward went non-finite at step {steps}"
        _assert_in_space(env, obs, f"rollout step {steps}")
        done = bool(terminated) or bool(truncated)
        steps += 1
        assert steps <= rollout_cap, (
            f"episode failed to reach a boundary within {rollout_cap} steps"
        )
    assert steps > 0, "episode terminated before a single step"

    # -- cross-construction determinism (optional, the seed-drives-market proof)
    if make_env_for_seed is not None:
        check_determinism_across_constructors(lambda: make_env_for_seed(0))
        e0, _ = make_env_for_seed(0).reset()
        e1, _ = make_env_for_seed(1).reset()
        assert not _decoded_equal(e0, e1), (
            "different constructor seeds produced byte-identical observations; "
            "the seed is not driving the market"
        )


def check_determinism_across_constructors(make_env: Callable[[], object]) -> None:
    """Build two envs the same way and prove reset+rollout observations match.

    This is the cross-instance reproducibility proof: identical construction must
    yield identical trajectories under a fixed (equal-weight) policy. It later
    underpins cross-binding (WASM vs. py) byte-identity checks — same contract,
    different runtimes.
    """
    env_a = make_env()
    env_b = make_env()

    obs_a, _ = env_a.reset()
    obs_b, _ = env_b.reset()
    assert _decoded_equal(obs_a, obs_b), (
        "reset() diverged across identical constructions (step 0)"
    )

    action = _equal_weight_action(env_a)
    done = False
    step = 0
    while not done:
        ra = env_a.step(action)
        rb = env_b.step(action)
        assert _decoded_equal(ra[0], rb[0]), (
            f"observation diverged across identical constructions at step {step}"
        )
        assert ra[1] == rb[1], (
            f"reward diverged across identical constructions at step {step}: "
            f"{ra[1]!r} != {rb[1]!r}"
        )
        done = bool(ra[2]) or bool(ra[3])
        step += 1
        assert step <= 10_000, "rollout did not terminate"


__all__ = ["check_env", "check_determinism_across_constructors"]
