"""RLlib as a learner (INT-07): one PPO iteration, a checkpoint, a restore, and the lanes.

Separate from ``test_rllib.py`` because this needs ``torch`` on top of ``ray[rllib]``, and
because the ``wheel-extras`` Ray cell runs this file with skips refused: nothing here may
pass by not running. That cell resolves its own Gymnasium, since ``ray[rllib]`` pins
``gymnasium==1.2.2`` and the source suite pins a newer one.

Budgets are tiny on purpose. Nothing asserted here is about whether a policy learned;
it is about whether the route trains, persists and resumes.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.rllib_env import multi_agent_config, single_agent_config

_READY = all(
    importlib.util.find_spec(name) is not None for name in ("ray", "torch", "pettingzoo")
)
pytestmark = pytest.mark.skipif(not _READY, reason="needs ray[rllib], torch and pettingzoo")

_SINGLE_SCENARIO = {"n_symbols": 2, "n_days": 12}
_MULTI_SCENARIO = {"n_agents": 2, "n_symbols": 2, "n_days": 12}
_BUDGET = {"train_batch_size": 24, "minibatch_size": 12, "num_epochs": 1}

_ROUTES = {
    "single": (single_agent_config, _SINGLE_SCENARIO),
    "multi": (multi_agent_config, _MULTI_SCENARIO),
}


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    import ray

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _build(route, seed=None, env_seed=None, n_days=None, **overrides):
    make_config, scenario = _ROUTES[route]
    env_config = dict(scenario)
    if n_days is not None:
        env_config["n_days"] = n_days
    if env_seed is not None:
        env_config["seed"] = env_seed
    options = {"num_env_runners": 0, **_BUDGET, **overrides}
    return make_config(env_config=env_config, seed=seed, **options).build_algo()


def _weights(algo):
    return {
        module: {key: np.array(value) for key, value in state.items()}
        for module, state in algo.learner_group.get_weights().items()
    }


def _assert_same_weights(left, right):
    assert left.keys() == right.keys()
    for module in left:
        assert left[module].keys() == right[module].keys()
        for key in left[module]:
            np.testing.assert_array_equal(left[module][key], right[module][key])


def _differs(left, right):
    return any(
        not np.array_equal(left[module][key], right[module][key])
        for module in left
        for key in left[module]
    )


def _lane_seeds(env_runner):
    """Which scenario each lane of one runner is actually trading."""
    seeds = []
    for lane in env_runner.env.unwrapped.envs:
        base = lane.unwrapped
        seeds.append(base._par.scenario_seed if hasattr(base, "_par") else base._seed)
    return seeds


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_one_iteration_checkpoints_and_restores_bit_for_bit_then_keeps_training(
    route, tmp_path
):
    from ray.rllib.algorithms.algorithm import Algorithm

    algo = _build(route, seed=5)
    try:
        before = _weights(algo)
        result = algo.train()
        trained = _weights(algo)
        assert result["env_runners"]["num_env_steps_sampled"] >= _BUDGET["train_batch_size"]
        assert _differs(before, trained), "one PPO iteration left every weight unchanged"

        checkpoint = algo.save_to_path(str(tmp_path / "checkpoint"))
        restored = Algorithm.from_checkpoint(checkpoint)
        try:
            _assert_same_weights(trained, _weights(restored))
            sampled = result["num_env_steps_sampled_lifetime"]
            resumed = restored.train()
            # Restart accounting: the lifetime counter continues rather than restarts,
            # so a resumed run cannot report a budget it did not spend.
            assert resumed["num_env_steps_sampled_lifetime"] > sampled
            assert _differs(trained, _weights(restored))
        finally:
            restored.stop()
    finally:
        algo.stop()


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_declared_scenario_seed_survives_the_runners_first_reset(route):
    """Held-out evaluation regression: `reset(seed=k)` rebuilds the scenario, and an
    EnvRunner's first reset used to replace `env_config["seed"]` with the band around
    `.debugging(seed=0)`, so an evaluation-band config traded the training band."""
    algo = _build(route, env_seed=EVAL_SEED_BASE, num_envs_per_env_runner=2)
    try:
        algo.train()
        assert _lane_seeds(algo.env_runner) == [EVAL_SEED_BASE, EVAL_SEED_BASE + 1]
    finally:
        algo.stop()


@pytest.mark.parametrize(
    "make_config", [single_agent_config, multi_agent_config], ids=["single", "multi"]
)
@pytest.mark.parametrize(
    "env_config,seed,message",
    [
        ({"seed": 3}, 4, "contradicts"),
        ({"seed_per_worker": False}, None, "seed_per_worker"),
    ],
    ids=["contradictory-seed", "shared-seed"],
)
def test_settings_rllib_would_silently_override_are_refused(
    make_config, env_config, seed, message
):
    with pytest.raises(ValueError, match=message):
        make_config(env_config=env_config, seed=seed)


def test_multi_agent_route_vectorises_lanes_on_distinct_scenarios_and_trains(tmp_path):
    """Settles the conflict: upstream's docs and the integration plan say multi-agent
    setups cannot be vectorised; an installed run says they can. Reads the lane count
    and scenarios back from the runner, because a config that merely accepts the setting
    proves nothing, and requires both lanes to have reached the sampled batch."""
    from ray.rllib.env.vector.vector_multi_agent_env import VectorMultiAgentEnv

    # Longer than the batch, so no lane resets away the record of having stepped.
    algo = _build("multi", seed=9, n_days=40, num_envs_per_env_runner=2)
    try:
        before = _weights(algo)
        result = algo.train()
        runner = algo.env_runner
        assert isinstance(runner.env.unwrapped, VectorMultiAgentEnv)
        assert runner.num_envs == 2
        assert _lane_seeds(runner) == [9, 10]
        for lane in runner.env.unwrapped.envs:
            assert all(lane.unwrapped._par._returns[seat] for seat in ("agent_0", "agent_1"))
        # Every seat of every lane feeds the one shared module.
        assert result["env_runners"]["num_module_steps_sampled"]["shared"] >= (
            2 * _BUDGET["train_batch_size"]
        )
        assert _differs(before, _weights(algo))
    finally:
        algo.stop()


def test_multi_agent_vector_lanes_run_on_a_remote_env_runner():
    """The same claim where it is used: inside an EnvRunner actor, not the local runner."""
    from ray import cloudpickle

    # This module is not importable inside the actor, so the helper travels by value.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    algo = _build("multi", seed=9, num_env_runners=1, num_envs_per_env_runner=2)
    try:
        algo.train()
        lanes = algo.env_runner_group.foreach_env_runner(
            lambda runner: (runner.num_envs, _lane_seeds(runner)), local_env_runner=False
        )
        # The runner offsets its seed by worker index; the lanes stay adjacent and distinct.
        assert len(lanes) == 1
        count, seeds = lanes[0]
        assert count == 2
        assert seeds[1] == seeds[0] + 1
        assert seeds[0] >= 9
    finally:
        algo.stop()
        cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
