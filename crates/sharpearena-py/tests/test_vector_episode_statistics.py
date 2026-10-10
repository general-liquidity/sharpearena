"""Reward-path diagnostics include opening equity and exclude reset-only steps."""

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from sharpearena.wrappers import RecordEpisodeStatistics
from sharpearena.wrappers_vector import VectorRecordEpisodeStatistics


@pytest.mark.parametrize(
    "wrapper", [RecordEpisodeStatistics, VectorRecordEpisodeStatistics]
)
@pytest.mark.parametrize(
    "returns,expected",
    [([-0.2], 0.2), ([-1.0], 1.0), ([-1.0, 0.0], 1.0), ([0.1, -0.5], 0.5), ([], 0.0)],
)
def test_drawdown_includes_opening_unit_equity_and_finite_ruin(
    wrapper, returns, expected
):
    with np.errstate(divide="raise", invalid="raise"):
        value = wrapper._max_drawdown(np.asarray(returns, dtype=float))
    assert value == pytest.approx(expected)
    assert np.isfinite(value)


class StaggeredEpisodes(VectorEnv):
    def __init__(self):
        self.num_envs = 2
        self.single_observation_space = spaces.Box(-1, 1, (1,), dtype=np.float64)
        self.single_action_space = spaces.Box(-1, 1, (1,), dtype=np.float64)
        self.observation_space = spaces.Box(-1, 1, (2, 1), dtype=np.float64)
        self.action_space = spaces.Box(-1, 1, (2, 1), dtype=np.float64)
        self.frames = iter(
            [
                ([0.2, 0.1], [True, False], [False, False]),
                ([0.0, -0.1], [False, True], [True, False]),
                ([0.3, 0.0], [True, False], [False, True]),
            ]
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros((2, 1)), {"first": np.ones(2, dtype=bool)}

    def step(self, actions):
        rewards, done, first = next(self.frames)
        return (
            np.zeros((2, 1)),
            np.array(rewards),
            np.zeros(2, dtype=bool),
            np.array(done),
            {"first": np.array(first)},
        )


def test_next_step_reset_reward_is_not_a_new_episode_sample():
    env = VectorRecordEpisodeStatistics(StaggeredEpisodes())
    env.reset()
    _, _, _, _, first = env.step(np.zeros((2, 1)))
    assert first["episode"]["l"][0] == 1
    assert first["episode"]["max_drawdown"][0] == 0.0
    _, _, _, _, second = env.step(np.zeros((2, 1)))
    assert second["episode"]["l"][1] == 2
    _, _, _, _, third = env.step(np.zeros((2, 1)))
    assert third["episode"]["l"][0] == 1
    assert third["episode"]["r"][0] == pytest.approx(0.3)
    assert third["episode"]["sharpe"][0] == 0.0


def test_same_step_terminal_reward_counts_even_with_first_flag():
    base = StaggeredEpisodes()
    base.frames = iter([([-0.2, 0.1], [True, False], [True, False])])
    env = VectorRecordEpisodeStatistics(base)
    env.reset()
    _, _, _, _, info = env.step(np.zeros((2, 1)))
    assert info["episode"]["l"][0] == 1
    assert info["episode"]["r"][0] == pytest.approx(-0.2)
    assert info["episode"]["max_drawdown"][0] == pytest.approx(0.2)
