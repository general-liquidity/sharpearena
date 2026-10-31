"""The binary step path against the canonical JSON path.

``step_array`` is only worth having if it is the same environment, so every comparison
here is exact: observation, reward and flag bytes, plus the engine's own snapshot, whose
trace records the decisions and whose RNG cursor records the execution noise. Weight
paths move through long, short and flat so the action label and the fill logic are both
exercised.
"""

import json

import numpy as np
import pytest

from sharpearena.gym import SharpeArenaEnv
from sharpearena.sharpearena_py import InvalidArgument, TradingEnv, VecTradingEnv
from sharpearena.vector import SharpeArenaVectorEnv

PANEL = dict(n_symbols=3, n_days=24)
MODES = ("next_step", "same_step", "disabled")
SCENARIOS = ("replay", "fresh_v1")


def weights_at(step, shape):
    """A deterministic path through long, short and flat targets."""
    phase = (step % 5) - 2
    return np.full(shape, phase * 0.2, dtype=np.float32) * np.linspace(1.0, 0.25, shape[-1])


def equal_obs(a, b):
    return all(a[k].tobytes() == b[k].tobytes() for k in ("closes", "positions", "cash"))


def test_scalar_paths_produce_identical_trajectories_and_engine_state():
    json_env = SharpeArenaEnv(seed=5, distribution_mode="hard", **PANEL)
    array_env = SharpeArenaEnv(seed=5, distribution_mode="hard", **PANEL)
    json_obs, _ = json_env.reset(seed=5)
    array_obs, _ = array_env.reset(seed=5)
    assert equal_obs(json_obs, array_obs)
    for step in range(PANEL["n_days"]):
        action = weights_at(step, (3,))
        obs_j, reward_j, term_j, trunc_j, info_j = json_env.step(action)
        obs_a, reward_a, term_a, trunc_a, info_a = array_env.step_array(action)
        assert equal_obs(obs_j, obs_a)
        assert np.float64(reward_j).tobytes() == np.float64(reward_a).tobytes()
        assert (term_j, trunc_j) == (term_a, trunc_a)
        assert type(term_a) is bool and type(trunc_a) is bool
        assert info_j["nav"] == info_a["nav"]
        assert info_j["scenario_seed"] == info_a["scenario_seed"]
        assert json_env.clone_state() == array_env.clone_state()
    assert trunc_a


@pytest.mark.parametrize("scenarios", SCENARIOS)
@pytest.mark.parametrize("mode", MODES)
def test_vector_paths_produce_identical_batches(mode, scenarios):
    build = lambda: SharpeArenaVectorEnv(  # noqa: E731
        seeds=[3, 8],
        autoreset_mode=mode,
        autoreset_scenarios=scenarios,
        max_episode_steps=6,
        **PANEL,
    )
    json_env, array_env = build(), build()
    json_env.reset()
    array_env.reset()
    recycled = 0
    for step in range(40):
        actions = weights_at(step, (2, 3))
        obs_j, rewards_j, term_j, trunc_j, infos_j = json_env.step(actions)
        obs_a, rewards_a, term_a, trunc_a, infos_a = array_env.step_array(actions)
        assert equal_obs(obs_j, obs_a)
        assert rewards_j.tobytes() == rewards_a.tobytes()
        assert term_j.tobytes() == term_a.tobytes() and term_a.dtype == bool
        assert trunc_j.tobytes() == trunc_a.tobytes() and trunc_a.dtype == bool
        assert infos_j["first"].tobytes() == infos_a["first"].tobytes()
        assert infos_j["nav"].tobytes() == infos_a["nav"].tobytes()
        assert infos_j["scenario_seed"].tobytes() == infos_a["scenario_seed"].tobytes()
        if scenarios == "fresh_v1":
            assert list(infos_j["drawn_scenario_seed"]) == list(infos_a["drawn_scenario_seed"])
        if mode == "same_step":
            for final_j, final_a in zip(infos_j["final_obs"], infos_a["final_obs"]):
                assert (final_j is None) == (final_a is None)
                if final_j is not None:
                    assert equal_obs(final_j, final_a)
            for final_j, final_a in zip(infos_j["final_info"], infos_a["final_info"]):
                assert (final_j is None) == (final_a is None)
                if final_j is not None:
                    assert final_j["nav"] == final_a["nav"]
                    assert final_j.get("scenario_seed") == final_a.get("scenario_seed")
        recycled += int(infos_a["first"].sum())
    assert (recycled > 0) == (mode != "disabled")
    assert json_env.scenario_seeds == array_env.scenario_seeds


def test_paths_can_interleave_without_a_seam():
    json_env = SharpeArenaVectorEnv(seeds=[3], **PANEL)
    mixed_env = SharpeArenaVectorEnv(seeds=[3], **PANEL)
    json_env.reset()
    mixed_env.reset()
    for step in range(30):
        actions = weights_at(step, (1, 3))
        expected = json_env.step(actions)
        got = (mixed_env.step_array if step % 2 else mixed_env.step)(actions)
        assert equal_obs(expected[0], got[0])
        assert expected[1].tobytes() == got[1].tobytes()


def native_scalar():
    env = TradingEnv(**PANEL)
    env.reset()
    return env


def buffers(n=3):
    return [np.zeros(3), np.zeros(n), np.zeros(n), np.zeros(1)]


def read_only():
    array = np.zeros(3)
    array.flags.writeable = False
    return array


@pytest.mark.parametrize(
    "arrays, match",
    [
        pytest.param([np.zeros(3, np.float32)] + buffers()[1:], "weights", id="float32"),
        pytest.param([np.zeros(2)] + buffers()[1:], "weights", id="short"),
        pytest.param([np.zeros((1, 3))] + buffers()[1:], "weights", id="rank"),
        pytest.param([[0.0, 0.0, 0.0]] + buffers()[1:], "weights", id="list"),
        pytest.param([np.full(3, np.nan)] + buffers()[1:], "finite", id="nan"),
        pytest.param([np.full(3, np.inf)] + buffers()[1:], "finite", id="inf"),
        pytest.param(
            [np.zeros(3), read_only(), np.zeros(3), np.zeros(1)], "writable", id="read-only"
        ),
        pytest.param(
            [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2)], "cash", id="cash-shape"
        ),
    ],
)
def test_a_refused_array_leaves_the_engine_where_it_was(arrays, match):
    env = native_scalar()
    before = env.clone_state()
    with pytest.raises(InvalidArgument, match=match):
        env.step_array(*arrays)
    assert env.clone_state() == before


def test_vector_refusal_happens_before_any_lane_advances():
    env = VecTradingEnv(seeds=[1, 2], n_symbols=3, n_days=24, autoreset_mode="same_step")
    env.reset_batch()
    primary = dict(
        weights=np.zeros((2, 3)),
        closes=np.zeros((2, 3)),
        positions=np.zeros((2, 3)),
        cash=np.zeros((2, 1)),
        rewards=np.zeros(2),
        terminated=np.zeros(2, np.uint8),
        truncated=np.zeros(2, np.uint8),
        first=np.zeros(2, np.uint8),
        navs=np.zeros(2),
    )
    with pytest.raises(InvalidArgument, match="same_step"):
        env.step_batch_array(**primary)
    with pytest.raises(InvalidArgument, match="all four"):
        env.step_batch_array(**primary, final_navs=np.zeros(2))
    with pytest.raises(InvalidArgument, match="terminated"):
        env.step_batch_array(**{**primary, "terminated": np.zeros(2, bool)})
    finals = dict(
        final_closes=np.zeros((2, 3)),
        final_positions=np.zeros((2, 3)),
        final_cash=np.zeros((2, 1)),
        final_navs=np.zeros(2),
    )
    env.step_batch_array(**primary, **finals)
    # The refusals above consumed nothing: this is bar one, not bar four.
    probe = VecTradingEnv(seeds=[1, 2], n_symbols=3, n_days=24, autoreset_mode="same_step")
    probe.reset_batch()
    expected = json.loads(probe.step_batch(json.dumps([{"orders": [], "reasoning": ""}] * 2)))
    assert primary["rewards"].tolist() == expected["rewards"]
    closes = [o["symbols"][0]["close_history"][-1] for o in expected["observations"]]
    assert primary["closes"][:, 0].tolist() == closes


def test_an_unfinished_lane_reads_nan_rather_than_a_flat_terminal():
    env = VecTradingEnv(seeds=[1, 2], n_symbols=2, n_days=3, autoreset_mode="same_step")
    env.reset_batch()
    final_closes = np.zeros((2, 2))
    first = np.zeros(2, np.uint8)
    env.step_batch_array(
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 1)),
        np.zeros(2),
        np.zeros(2, np.uint8),
        np.zeros(2, np.uint8),
        first,
        np.zeros(2),
        final_closes=final_closes,
        final_positions=np.zeros((2, 2)),
        final_cash=np.zeros((2, 1)),
        final_navs=np.zeros(2),
    )
    assert not first.any()
    assert np.isnan(final_closes).all()


def test_step_array_refuses_while_a_step_async_is_pending():
    env = SharpeArenaVectorEnv(seeds=[1], **PANEL)
    env.reset()
    actions = np.zeros((1, 3), dtype=np.float32)
    env.step_async(actions)
    with pytest.raises(RuntimeError, match="pending"):
        env.step_array(actions)


def test_returned_observations_do_not_alias_between_steps():
    env = SharpeArenaEnv(seed=2, **PANEL)
    env.reset()
    first_obs, *_ = env.step_array(np.zeros(3, dtype=np.float32))
    kept = first_obs["closes"].copy()
    env.step_array(np.zeros(3, dtype=np.float32))
    assert first_obs["closes"].tobytes() == kept.tobytes()
