"""Terminal tournament delivery does not invent numeric scores or revive agents."""

from copy import deepcopy

import numpy as np
import pytest

from sharpearena import pettingzoo_env
from sharpearena.kernel_score import is_kernel_score_unavailable


@pytest.mark.parametrize("returns", [[], [0.01]])
def test_insufficient_returns_are_unavailable_not_a_scored_zero(returns, monkeypatch):
    def unexpected_kernel(*args):
        raise AssertionError("insufficient observations must not call the kernel")

    monkeypatch.setattr(pettingzoo_env, "score_run", unexpected_kernel)
    result = pettingzoo_env._deflated_sharpe(returns, 0)
    assert is_kernel_score_unavailable(result)
    assert "fewer than two return observations" in result


class ScriptedLane:
    def __init__(self, frames):
        self.frames = frames

    def reset(self, **kwargs):
        self.pending = iter(self.frames)
        return {}, {}

    def step(self, action):
        reward, done = next(self.pending)
        return {}, reward, done, False, {}

    def close(self):
        pass


def _staggered_tournament():
    pytest.importorskip("pettingzoo")
    env = pettingzoo_env.MultiAgentSharpeArenaEnv(n_agents=2, n_symbols=2, n_days=8)
    for lane in env._envs.values():
        lane.close()
    env._envs = {
        "agent_0": ScriptedLane([(0.01, True)]),
        "agent_1": ScriptedLane([(0.01, False), (-0.02, True)]),
    }
    return env


def test_final_result_includes_early_finished_agent_without_reviving_it():
    env = _staggered_tournament()
    assert env.final_ranking is None
    env.reset()
    first = env.step({agent: np.zeros(2) for agent in env.agents})
    assert env.agents == ["agent_1"]
    assert "ranking" not in first[4]["agent_0"]
    assert env.final_ranking is None
    last = env.step({"agent_1": np.zeros(2)})
    assert all(set(part) == {"agent_1"} for part in last)
    assert env.agents == []
    ranking = env.final_ranking
    assert ranking == last[4]["agent_1"]["ranking"]
    assert [row["agent"] for row in ranking] == ["agent_1", "agent_0"]
    assert isinstance(ranking[0]["deflated_sharpe"], float)
    assert is_kernel_score_unavailable(ranking[1]["deflated_sharpe"])
    assert [row["rank"] for row in ranking] == [0, 1]
    env.close()


def test_result_snapshots_are_detached_and_reset_clears_them():
    env = _staggered_tournament()
    env.reset()
    env.step({agent: np.zeros(2) for agent in env.agents})
    last = env.step({"agent_1": np.zeros(2)})
    expected = deepcopy(env.final_ranking)
    last[4]["agent_1"]["ranking"][0]["agent"] = "caller mutation"
    snapshot = env.final_ranking
    snapshot[0]["agent"] = "another caller mutation"
    assert env.final_ranking == expected
    assert env.step({}) == ({}, {}, {}, {}, {})
    assert env.final_ranking == expected
    env.reset()
    assert env.final_ranking is None
    env.close()
