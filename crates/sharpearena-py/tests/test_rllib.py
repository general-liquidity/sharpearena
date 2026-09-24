"""RLlib single-agent and multi-agent routes (INT-07): guarded import, registration,
the flatten/unflatten parity claim, and config construction under both constraints the
module docstring records as re-verified against the installed Ray build.

Everything here runs with `ray[rllib]` installed and no GPU. It deliberately does not
call `AlgorithmConfig.build_algo()` or train a step: that additionally needs a deep
learning framework (`torch` or `tf2`), which is not part of this package's `ray` extra
or its CI dependency set, so a route that needs it is out of scope for this suite.
Both PPO configs here were built and trained for one iteration by hand against Ray
2.58.0 with `torch==2.14.0+cpu` outside this test run; `docs/integrations/support-status.md`
records that as local, not CI, evidence, which is the distinction the project already
draws for the `verifiers` version gap in `inventory.md`.

The multi-agent route additionally needs `pettingzoo` (`SharpeArenaMultiAgentEnv`
wraps `pettingzoo_env.MultiAgentSharpeArenaEnv`), so those tests carry both guards.
"""

from __future__ import annotations

import importlib.util

import gymnasium as gym
import numpy as np
import pytest

import sharpearena  # noqa: F401  (ensures the package imports without ray)
import sharpearena.rllib_env as rl_env
from sharpearena.integrations.parity import CORE_FIXTURES, check_adapter_parity
from sharpearena.rllib_env import (
    MULTI_AGENT_ENV,
    SINGLE_AGENT_ENV,
    RLlibUnavailable,
    SharpeArenaMultiAgentEnv,
    multi_agent_config,
    register_sharpearena_envs,
    sharpearena_env_creator,
    sharpearena_multi_agent_env_creator,
    single_agent_config,
    worker_seed,
)

_HAS_RLLIB = importlib.util.find_spec("ray") is not None and importlib.util.find_spec(
    "ray.rllib"
) is not None
_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_rllib = pytest.mark.skipif(not _HAS_RLLIB, reason="ray[rllib] not installed")
needs_rllib_and_pettingzoo = pytest.mark.skipif(
    not (_HAS_RLLIB and _HAS_PETTINGZOO), reason="ray[rllib] and pettingzoo both required"
)


# ---------------------------------------------------------------------------
# Optional-import contract (runs when ray[rllib] is NOT installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_RLLIB, reason="ray[rllib] is installed")
def test_import_works_and_construction_raises_without_rllib():
    assert rl_env._HAS_RLLIB is False
    with pytest.raises(RLlibUnavailable, match="ray\\[rllib\\] is not installed"):
        sharpearena_env_creator({"n_symbols": 2})
    with pytest.raises(RLlibUnavailable, match="ray\\[rllib\\] is not installed"):
        register_sharpearena_envs()
    with pytest.raises(RLlibUnavailable, match="ray\\[rllib\\] is not installed"):
        single_agent_config()


# ---------------------------------------------------------------------------
# worker_seed: pure, no ray required
# ---------------------------------------------------------------------------


def test_worker_seed_without_per_worker_seeding_is_the_base_seed():
    assert worker_seed(7, 3, seed_per_worker=False) == 7


def test_worker_seed_with_per_worker_seeding_offsets_by_worker_index():
    assert worker_seed(7, 0, seed_per_worker=True) == 7
    assert worker_seed(7, 3, seed_per_worker=True) == 10


# ---------------------------------------------------------------------------
# Single-agent route
# ---------------------------------------------------------------------------


@needs_rllib
def test_registration_goes_through_tunes_registry_not_gymnasiums():
    """The module docstring's central claim: `SingleAgentEnvRunner.make_env` resolves
    the env through `ray.tune.registry._global_registry`, not through Gymnasium's own
    registry, so registration must use `ray.tune.registry.register_env`."""
    from ray.tune.registry import _global_registry

    single_name, multi_name = register_sharpearena_envs()
    assert single_name == SINGLE_AGENT_ENV
    assert multi_name == MULTI_AGENT_ENV
    assert _global_registry.contains("env_creator", SINGLE_AGENT_ENV)
    assert _global_registry.contains("env_creator", MULTI_AGENT_ENV)


@needs_rllib
def test_creator_default_rlmodule_shape_is_a_flat_box_not_the_dict():
    """Confirms the module docstring's second re-verified claim: the raw env's Dict
    observation is flattened before an RLModule would see it, because 2.58.0's default
    encoder has no config for a Dict observation space."""
    env = sharpearena_env_creator({"n_symbols": 3, "n_days": 20, "seed": 0})
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs, _info = env.reset()
    assert env.observation_space.contains(obs)


@needs_rllib
def test_creator_rejects_an_unknown_config_key():
    with pytest.raises(ValueError, match="unknown env-config keys"):
        sharpearena_env_creator({"not_a_real_key": 1})


@needs_rllib
def test_max_episode_steps_installs_a_time_limit_that_truncates_not_terminates():
    """A horizon is a wrapper on this API stack, not a config field (module
    docstring); `TimeLimit` sets `truncated`, which is what the learner connector
    bootstraps, so running out of the declared horizon must not read as `terminated`."""
    env = sharpearena_env_creator({"n_symbols": 2, "n_days": 100, "max_episode_steps": 3})
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert truncated and not terminated


@needs_rllib
def test_single_agent_config_builds_ppo_over_the_registered_env():
    cfg = single_agent_config(
        env_config={"n_symbols": 2, "n_days": 20},
        num_env_runners=0,
        train_batch_size=32,
        minibatch_size=16,
        num_epochs=1,
        seed=0,
    )
    assert cfg.env == SINGLE_AGENT_ENV
    assert cfg.env_config == {"n_symbols": 2, "n_days": 20}
    assert cfg.num_env_runners == 0


@needs_rllib
@pytest.mark.parametrize("fixture", CORE_FIXTURES, ids=lambda f: f.name)
def test_flatten_round_trip_reproduces_the_engine_bit_for_bit(fixture):
    """The parity claim `integrations.parity` exists to check, applied to the RLlib
    single-agent creator: flatten the Dict observation going out, unflatten it coming
    back in, and the result must match the native engine field for field -- not merely
    produce a tensor of the right shape, which the module docstring specifically
    disclaims as sufficient."""

    class _UnflattenedRLlibAdapter:
        def __init__(self, fixture):
            self._env = sharpearena_env_creator(
                {
                    "n_symbols": fixture.n_symbols,
                    "n_days": fixture.n_days,
                    "seed": fixture.seed,
                    "distribution_mode": fixture.distribution_mode,
                    "max_weight": fixture.max_weight,
                    "allow_short": fixture.allow_short,
                    "mode": fixture.mode,
                    "seed_per_worker": False,
                }
            )

        @property
        def symbols(self):
            return self._env.env.symbols

        def reset(self):
            flat_obs, info = self._env.reset()
            return self._env.unflatten(flat_obs), info

        def step(self, action):
            flat_obs, reward, terminated, truncated, info = self._env.step(action)
            return self._env.unflatten(flat_obs), reward, terminated, truncated, info

    report = check_adapter_parity(
        fixture, _UnflattenedRLlibAdapter, source="rllib_env.sharpearena_env_creator"
    )
    assert report.ok, report.mismatches
    assert report.steps_compared > 0


# ---------------------------------------------------------------------------
# Multi-agent route
# ---------------------------------------------------------------------------


@needs_rllib_and_pettingzoo
def test_multi_agent_env_flattens_every_seats_observation():
    env = sharpearena_multi_agent_env_creator({"n_agents": 2, "n_symbols": 2, "n_days": 20})
    obs, infos = env.reset()
    assert set(obs) == set(env.agents) == {"agent_0", "agent_1"}
    for agent in env.agents:
        assert isinstance(env.observation_space[agent], gym.spaces.Box)
        assert env.observation_space[agent].contains(obs[agent])
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards) == set(actions)
    assert set(terms) - {"__all__"} == set(actions)
    assert set(truncs) - {"__all__"} == set(actions)
    assert "__all__" in terms and "__all__" in truncs
    env.close()


@needs_rllib_and_pettingzoo
def test_multi_agent_env_matches_the_pettingzoo_env_it_wraps():
    """`SharpeArenaMultiAgentEnv` adds only the RLlib surface and flattening on top of
    `MultiAgentSharpeArenaEnv`; it must not change what that environment decides."""
    from sharpearena.pettingzoo_env import MultiAgentSharpeArenaEnv

    rllib_env = SharpeArenaMultiAgentEnv({"n_agents": 2, "n_symbols": 2, "n_days": 16, "seed": 3})
    pz_env = MultiAgentSharpeArenaEnv(n_agents=2, n_symbols=2, n_days=16, seed=3)

    rllib_obs, _ = rllib_env.reset(seed=3)
    pz_obs, _ = pz_env.reset(seed=3)
    for agent in pz_env.agents:
        expected = gym.spaces.flatten(pz_env.observation_space(agent), pz_obs[agent])
        np.testing.assert_array_equal(rllib_obs[agent], expected)

    zero_actions = {a: np.zeros(2, dtype=np.float32) for a in pz_env.agents}
    _, rllib_rewards, rllib_terms, rllib_truncs, _ = rllib_env.step(dict(zero_actions))
    _, pz_rewards, pz_terms, pz_truncs, _ = pz_env.step(dict(zero_actions))
    for agent in pz_env.agents:
        assert rllib_rewards[agent] == pz_rewards[agent]
        assert rllib_terms[agent] == pz_terms[agent]
        assert rllib_truncs[agent] == pz_truncs[agent]
    rllib_env.close()
    pz_env.close()


@needs_rllib_and_pettingzoo
def test_multi_agent_config_shares_one_policy_across_symmetric_seats():
    cfg = multi_agent_config(
        env_config={"n_agents": 2, "n_symbols": 2, "n_days": 20},
        num_env_runners=0,
        train_batch_size=32,
        minibatch_size=16,
        num_epochs=1,
    )
    assert cfg.env == MULTI_AGENT_ENV
    assert set(cfg.policies) == {"shared"}
    assert cfg.policy_mapping_fn("agent_0") == "shared"
    assert cfg.policy_mapping_fn("agent_1") == "shared"


@needs_rllib_and_pettingzoo
def test_multi_agent_config_accepts_vectorised_env_runners():
    """The docstring's fourth re-verified claim: at 2.58.0, `MultiAgentEnvRunner`
    actually vectorises via `gymnasium.make_vec`, unlike what both cited upstream pages
    still state. This only checks the config accepts the setting the module offers for
    it; the vectorised run itself was exercised by hand (see module docstring)."""
    cfg = multi_agent_config(
        env_config={"n_agents": 2, "n_symbols": 2, "n_days": 20},
        num_env_runners=1,
        num_envs_per_env_runner=2,
        train_batch_size=32,
        minibatch_size=16,
        num_epochs=1,
    )
    assert cfg.num_envs_per_env_runner == 2
