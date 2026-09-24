"""RLlib single-agent and multi-agent routes over SharpeArena (INT-07).

RLlib ships in the same wheel as Ray, so this module and
:mod:`sharpearena.ray_executor` share one optional extra and one version pin. The
import is guarded the way ``pettingzoo_env`` and ``verifiers_env`` are: importing this
module with no Ray installed works, and only building a config or an environment raises
the named :class:`RLlibUnavailable`.

Four upstream constraints shape everything here. Each was rechecked against the
installed Ray 2.58.0 rather than taken from a planning note, and the one that turned out
to be stale is recorded as stale rather than designed around.

**Registration goes through Tune's registry, not Gymnasium's.** RLlib states that
"gymnasium's own registry is incompatible with Ray" for remote workers, and the code
agrees: ``SingleAgentEnvRunner.make_env`` looks the environment up in
``_global_registry`` under ``ENV_CREATOR`` and then calls ``gym.register`` *locally,
inside the worker*, from what it found there. So the package's own
``SharpeArena/<Tier>-v1`` Gymnasium ids do not reach an ``EnvRunner`` actor, and
:func:`register_sharpearena_envs` uses ``ray.tune.registry.register_env``.

**The creator must be importable by name and free of captured state.** The creator is
what crosses the boundary, so :func:`sharpearena_env_creator` is a module-level function
taking the ``EnvContext`` RLlib hands it. It holds no environment and no engine handle;
the native environment is built inside the worker, on the worker's own call.

**The default RLModules do not accept a Dict observation.** Confirmed on 2.58.0: a
``PPOConfig`` over the raw environment fails at module build with ``No default encoder
config for obs space=Dict(...)``. The creator therefore wraps the environment in
:class:`~sharpearena.spaces.FlattenObservation`, whose flattening is exactly invertible,
and ``tests/test_rllib.py`` proves through :mod:`sharpearena.integrations.parity` that
the flatten/unflatten round trip reproduces the engine's observation bit for bit rather
than merely producing a tensor of the right shape.

**A horizon is a wrapper, not a config key.** The new API stack has no ``horizon``
setting, and time-limit bootstrapping is structural: the learner connector pipeline
appends a phantom timestep and ``GeneralAdvantageEstimation`` bootstraps it. What
decides whether the value target is bootstrapped or zeroed is only which flag the
environment sets, so ``max_episode_steps`` here installs
``gymnasium.wrappers.TimeLimit``, which sets ``truncated``. That matches the engine's
own reading, where running out of bars is truncation and a blown account is termination.

**Multi-agent vectorisation: the documented restriction is stale at 2.58.0.** Both
``rllib-env`` and ``multi-agent-envs`` state that "multi-agent setups aren't
vectorizable yet". In the installed 2.58.0, ``MultiAgentEnvRunner.make_env`` calls
``gymnasium.make_vec`` with ``num_envs_per_env_runner`` and asserts the result is a
``VectorMultiAgentEnv``. ``tests/test_rllib.py`` trains the multi-agent route with
``num_envs_per_env_runner=2`` and records what actually happens, so the claim in this
tree rests on the run rather than on either page.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np

from .gym import SharpeArenaEnv
from .spaces import FlattenObservation, flatten_obs

try:  # pragma: no cover - exercised only when ray is installed
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.tune.registry import register_env

    _HAS_RLLIB = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    MultiAgentEnv = object  # type: ignore[assignment,misc]
    PPOConfig = None  # type: ignore[assignment]
    register_env = None  # type: ignore[assignment]
    _HAS_RLLIB = False


#: The Tune-registry names this module registers. They are deliberately distinct from
#: the package's Gymnasium ids (``SharpeArena/<Tier>-v1``), because these live in a
#: different registry with different reach and conflating the two is the mistake the
#: registry constraint punishes.
SINGLE_AGENT_ENV = "sharpearena-single-agent"
MULTI_AGENT_ENV = "sharpearena-multi-agent"

#: Scenario keys a creator config may carry through to the environment. Anything else
#: is refused rather than ignored, so a misspelled key cannot silently leave the
#: environment at its default.
_SCENARIO_KEYS = (
    "n_symbols",
    "n_days",
    "seed",
    "distribution_mode",
    "max_weight",
    "allow_short",
    "mode",
)
_CREATOR_KEYS = _SCENARIO_KEYS + ("max_episode_steps", "seed_per_worker", "n_agents")


class RLlibUnavailable(RuntimeError):
    """``ray[rllib]`` was asked for and is not importable."""


def _require_rllib() -> None:
    if not _HAS_RLLIB:
        raise RLlibUnavailable(
            "ray[rllib] is not installed. Install 'sharpearena[rllib]' to use the RLlib "
            "routes; the rest of the sharpearena package works without it. Note that "
            "ray[rllib] 2.58.0 pins gymnasium==1.2.2, which this package's gymnasium>=1.0 "
            "floor admits but a stricter pin elsewhere in your environment may not."
        )


def _scenario_kwargs(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = dict(config or {})
    unknown = sorted(set(payload) - set(_CREATOR_KEYS))
    if unknown:
        raise ValueError(
            f"unknown env-config keys {unknown}; known keys are {sorted(_CREATOR_KEYS)}"
        )
    return {key: payload[key] for key in _SCENARIO_KEYS if key in payload}


def worker_seed(base_seed: int, worker_index: int, *, seed_per_worker: bool) -> int:
    """The scenario seed one worker uses.

    With ``seed_per_worker`` every ``EnvRunner`` trades a different scenario, which is
    what a training run wants; without it every worker trades the same one, which is
    what a comparison against a single-process replay needs. The offset is the worker
    index itself rather than a hash, so the mapping from (base seed, worker) to scenario
    is stated rather than discovered, and a run is reproducible from the two numbers.
    """
    if not seed_per_worker:
        return int(base_seed)
    return int(base_seed) + int(worker_index)


def sharpearena_env_creator(config: Optional[Mapping[str, Any]] = None):
    """Build one single-agent environment inside the worker that will step it.

    RLlib passes an ``EnvContext``, which is a mapping that also carries
    ``worker_index`` and ``num_workers``. Both are read off the object rather than out
    of the mapping, because ``EnvContext`` exposes them as attributes and a plain dict
    passed by a test has neither.
    """
    _require_rllib()
    kwargs = _scenario_kwargs(config)
    payload = dict(config or {})
    base_seed = int(kwargs.pop("seed", 0))
    seed = worker_seed(
        base_seed,
        int(getattr(config, "worker_index", 0) or 0),
        seed_per_worker=bool(payload.get("seed_per_worker", True)),
    )
    env = FlattenObservation(SharpeArenaEnv(seed=seed, **kwargs))
    max_episode_steps = payload.get("max_episode_steps")
    if max_episode_steps:
        # A horizon is truncation, never termination: the learner pipeline bootstraps a
        # truncated episode's value and zeroes a terminated one, and TimeLimit is the
        # only place on the new API stack that distinction can be made.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))
    return env


class SharpeArenaMultiAgentEnv(MultiAgentEnv):
    """RLlib's multi-agent view of the existing PettingZoo competition environment.

    Delegates every scenario and scoring decision to
    :class:`~sharpearena.pettingzoo_env.MultiAgentSharpeArenaEnv`, which already defines
    what a SharpeArena tournament is: every agent trades its own copy of one shared
    frozen scenario and is ranked by realized deflated Sharpe. This class adds exactly
    two things, and deliberately nothing else: the ``MultiAgentEnv`` surface RLlib
    requires, and the per-agent observation flattening the default RLModules require.

    Agents that finish are dropped from the live roster by the wrapped environment. The
    ``"__all__"`` key RLlib reads is set when no live agent is left, and each agent's
    own flag is reported for the step it finished on, so a per-agent truncation stays a
    truncation and is bootstrapped rather than zeroed.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        _require_rllib()
        from .pettingzoo_env import MultiAgentSharpeArenaEnv

        super().__init__()
        payload = dict(config or {})
        kwargs = _scenario_kwargs(config)
        base_seed = int(kwargs.pop("seed", 0))
        seed = worker_seed(
            base_seed,
            int(getattr(config, "worker_index", 0) or 0),
            seed_per_worker=bool(payload.get("seed_per_worker", True)),
        )
        self._par = MultiAgentSharpeArenaEnv(
            n_agents=int(payload.get("n_agents", 2)), seed=seed, **kwargs
        )
        self.possible_agents = list(self._par.possible_agents)
        self.agents = list(self.possible_agents)

        single_obs = self._par.observation_space(self.possible_agents[0])
        single_act = self._par.action_space(self.possible_agents[0])
        self._dict_obs_space = single_obs
        flat_obs = gym.spaces.flatten_space(single_obs)
        self.observation_spaces = {agent: flat_obs for agent in self.possible_agents}
        self.action_spaces = {agent: single_act for agent in self.possible_agents}
        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)

    @property
    def symbols(self) -> list[str]:
        return list(self._par.symbols)

    def _flatten(self, observations: Mapping[str, Any]) -> dict[str, np.ndarray]:
        return {
            agent: flatten_obs(self._dict_obs_space, obs)
            for agent, obs in observations.items()
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, infos = self._par.reset(seed=seed, options=options)
        self.agents = list(self._par.agents)
        return self._flatten(observations), dict(infos)

    def step(self, action_dict: Mapping[str, Any]):
        observations, rewards, terminations, truncations, infos = self._par.step(
            dict(action_dict)
        )
        terminations = dict(terminations)
        truncations = dict(truncations)
        self.agents = list(self._par.agents)
        terminations["__all__"] = not self.agents
        truncations["__all__"] = not self.agents and not any(
            value for key, value in terminations.items() if key != "__all__"
        )
        return (
            self._flatten(observations),
            dict(rewards),
            terminations,
            truncations,
            dict(infos),
        )

    def close(self) -> None:  # pragma: no cover - the wrapped env holds no OS resource
        self._par.close()


def sharpearena_multi_agent_env_creator(config: Optional[Mapping[str, Any]] = None):
    """Build one multi-agent environment inside the worker that will step it."""
    return SharpeArenaMultiAgentEnv(config)


def register_sharpearena_envs() -> tuple[str, str]:
    """Register both creators with Tune's registry and return the two names.

    Idempotent: re-registering a name overwrites the entry with the same callable.
    """
    _require_rllib()
    register_env(SINGLE_AGENT_ENV, sharpearena_env_creator)
    register_env(MULTI_AGENT_ENV, sharpearena_multi_agent_env_creator)
    return SINGLE_AGENT_ENV, MULTI_AGENT_ENV


def single_agent_config(
    *,
    env_config: Optional[Mapping[str, Any]] = None,
    num_env_runners: int = 0,
    num_envs_per_env_runner: int = 1,
    train_batch_size: int = 256,
    minibatch_size: int = 64,
    num_epochs: int = 1,
    seed: int = 0,
):
    """A PPO config over the single-agent route.

    PPO because it is one of the algorithms RLlib's own table marks as supporting
    continuous action spaces on the new API stack; the action here is a target-weight
    vector, so the discrete-only algorithms (DQN, IMPALA) do not apply and are not
    offered rather than being adapted into a different task.

    ``num_envs_per_env_runner`` above 1 vectorises within an ``EnvRunner``. That is
    supported for this route; see the module docstring for what 2.58.0 actually does for
    the multi-agent one.
    """
    _require_rllib()
    register_sharpearena_envs()
    return (
        PPOConfig()
        .environment(SINGLE_AGENT_ENV, env_config=dict(env_config or {}))
        .env_runners(
            num_env_runners=int(num_env_runners),
            num_envs_per_env_runner=int(num_envs_per_env_runner),
        )
        .training(
            train_batch_size_per_learner=int(train_batch_size),
            minibatch_size=int(minibatch_size),
            num_epochs=int(num_epochs),
        )
        .learners(num_learners=0)
        .debugging(seed=int(seed))
    )


def shared_policy_mapping(agent_id, episode=None, **kwargs) -> str:
    """Map every seat onto one shared policy.

    Named and module-level rather than a lambda, because the mapping function is
    serialised out to the runners and a lambda is exactly the shape that fails there.
    """
    return "shared"


def multi_agent_config(
    *,
    env_config: Optional[Mapping[str, Any]] = None,
    num_env_runners: int = 0,
    num_envs_per_env_runner: int = 1,
    train_batch_size: int = 256,
    minibatch_size: int = 64,
    num_epochs: int = 1,
    seed: int = 0,
):
    """A PPO config over the multi-agent route, with every seat on one shared policy.

    One shared policy is the honest default for this environment: the seats are
    symmetric by construction, since every agent trades its own copy of the same frozen
    scenario with identical spaces. A population with distinct opponents is a different
    experiment and belongs in a manifest rather than in a default.
    """
    _require_rllib()
    register_sharpearena_envs()
    return (
        PPOConfig()
        .environment(MULTI_AGENT_ENV, env_config=dict(env_config or {}))
        .multi_agent(
            policies={"shared"},
            policy_mapping_fn=shared_policy_mapping,
        )
        .env_runners(
            num_env_runners=int(num_env_runners),
            num_envs_per_env_runner=int(num_envs_per_env_runner),
        )
        .training(
            train_batch_size_per_learner=int(train_batch_size),
            minibatch_size=int(minibatch_size),
            num_epochs=int(num_epochs),
        )
        .learners(num_learners=0)
        .debugging(seed=int(seed))
    )


__all__ = [
    "MULTI_AGENT_ENV",
    "SINGLE_AGENT_ENV",
    "RLlibUnavailable",
    "SharpeArenaMultiAgentEnv",
    "multi_agent_config",
    "register_sharpearena_envs",
    "shared_policy_mapping",
    "sharpearena_env_creator",
    "sharpearena_multi_agent_env_creator",
    "single_agent_config",
    "worker_seed",
]
