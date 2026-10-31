"""The CleanRL-style PPO example: bootstrap arithmetic and a tiny CPU run.

Skipped without torch. CI runs the installed-wheel path through
``scripts/check-cleanrl-style-example.py`` in the SB3 cell.
"""

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "cleanrl-style" / "ppo_vector.py"
_spec = importlib.util.spec_from_file_location("ppo_vector", EXAMPLE)
ppo = importlib.util.module_from_spec(_spec)
sys.modules["ppo_vector"] = ppo
_spec.loader.exec_module(ppo)

GAMMA, LAM = 0.5, 0.5
TINY = dataclasses.replace(
    ppo.Args(), n_days=40, total_timesteps=256, num_steps=32, update_epochs=1
)


def _gae(rewards, values, next_values, terminated, truncated):
    columns = [rewards, values, next_values, terminated, truncated]
    tensors = [torch.tensor(column, dtype=torch.float64).unsqueeze(-1) for column in columns]
    return ppo.compute_gae(*tensors, GAMMA, LAM).squeeze(-1).tolist()


def test_termination_drops_the_bootstrap():
    assert _gae([1.0], [0.25], [8.0], [1.0], [0.0]) == [0.75]


def test_truncation_keeps_the_bootstrap():
    assert _gae([1.0], [0.25], [8.0], [0.0], [1.0]) == [4.75]


def test_an_ending_step_stops_advantage_from_leaking_backward():
    # Step 1 ends by truncation and step 2 belongs to the next episode, so the 100 at
    # step 2 reaches neither earlier step. Step 0 still receives step 1's advantage.
    advantages = _gae(
        [0.0, 1.0, 100.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0, 0, 0], [0, 1, 0]
    )
    assert advantages == [0.5, 2.0, 100.0]


def test_true_next_values_read_the_terminal_observation_of_an_ended_lane():
    class FirstClose:
        @staticmethod
        def value(inputs):
            return inputs[:, 0]

    next_obs = {
        "closes": np.array([[1.0, 1.0], [1.0, 1.0]]),
        "positions": np.zeros((2, 2)),
        "cash": np.ones((2, 1)),
    }
    terminal = {
        "closes": np.array([np.e, 1.0]),
        "positions": np.zeros(2),
        "cash": np.ones(1),
    }
    infos = {"final_obs": np.array([terminal, None], dtype=object)}
    values = ppo.true_next_values(FirstClose, next_obs, infos, np.array([True, False]))
    assert values.tolist() == pytest.approx([0.5, 0.0])
    assert next_obs["closes"][0].tolist() == [1.0, 1.0]


def test_tiny_training_run_is_seeded_and_evaluates_on_held_out_lanes():
    torch.set_num_threads(1)
    first, episodes = ppo.train(TINY)
    second, _ = ppo.train(TINY)
    assert episodes
    assert all(math.isfinite(value) for value in episodes)
    assert ppo.parameter_digest(first) == ppo.parameter_digest(second)
    other, _ = ppo.train(dataclasses.replace(TINY, seed=2))
    assert ppo.parameter_digest(first) != ppo.parameter_digest(other)
    held_out = ppo.evaluate(first, TINY)
    assert len(held_out) == TINY.num_envs
    assert all(math.isfinite(value) for value in held_out)
    assert set(ppo.make_env(TINY, ppo.EVAL_SEED_BASE).scenario_seeds).isdisjoint(
        ppo.make_env(TINY, TINY.seed).scenario_seeds
    )
