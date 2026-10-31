"""SB3 as a learner (INT-11): save, load, resume, and evaluation that cannot learn.

Separate from ``test_sb3.py`` so the ``wheel-extras`` SB3 cell can run this file with
skips refused. Budgets are tiny and nothing here says a policy learned anything.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.sb3_env import SharpeArenaSB3VecEnv

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 not installed",
)

_RECIPE = Path(__file__).resolve().parents[3] / "examples" / "sb3" / "train_ppo.py"
_SEED = 7
_STEPS = 64


def _lanes(first_seed=0):
    return SharpeArenaSB3VecEnv(
        seeds=[first_seed, first_seed + 1], n_symbols=2, n_days=12
    )


def _ppo(env):
    from stable_baselines3 import PPO

    return PPO(
        "MultiInputPolicy",
        env,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        seed=_SEED,
        device="cpu",
        verbose=0,
    )


def _continue(model):
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(_SEED)
    model.learn(total_timesteps=_STEPS, reset_num_timesteps=False)


def _assert_same_state(left, right):
    left_state = left.policy.state_dict()
    right_state = right.policy.state_dict()
    assert left_state.keys() == right_state.keys()
    for key, value in left_state.items():
        np.testing.assert_array_equal(value.numpy(), right_state[key].numpy())


def test_resume_from_disk_matches_resuming_in_memory_bit_for_bit(tmp_path):
    """The model archive has to carry everything resumed training reads: weights,
    optimizer moments, the timestep counter behind the schedule. Both arms restart their
    lanes and seed identically, so any difference is what the archive lost."""
    from stable_baselines3 import PPO

    in_memory = _ppo(_lanes())
    in_memory.learn(total_timesteps=_STEPS)
    in_memory.save(tmp_path / "ppo")

    from_disk = PPO.load(tmp_path / "ppo", env=_lanes(), device="cpu")
    _assert_same_state(in_memory, from_disk)
    assert from_disk.num_timesteps == in_memory.num_timesteps

    in_memory.set_env(_lanes(), force_reset=True)
    trained_to = in_memory.policy.state_dict()["log_std"].clone()
    _continue(in_memory)
    _continue(from_disk)

    assert not np.array_equal(
        trained_to.numpy(), in_memory.policy.state_dict()["log_std"].numpy()
    ), "the resumed phase trained nothing, so equality would be vacuous"
    _assert_same_state(in_memory, from_disk)
    assert from_disk.num_timesteps == in_memory.num_timesteps


def test_loaded_policy_acts_identically_on_a_held_out_lane(tmp_path):
    from stable_baselines3 import PPO

    model = _ppo(_lanes())
    model.learn(total_timesteps=_STEPS)
    model.save(tmp_path / "ppo")
    loaded = PPO.load(tmp_path / "ppo", device="cpu")

    observation = _lanes(EVAL_SEED_BASE).reset()
    expected, _ = model.predict(observation, deterministic=True)
    actual, _ = loaded.predict(observation, deterministic=True)
    np.testing.assert_array_equal(actual, expected)


def test_vecnormalize_statistics_round_trip_and_freeze_during_evaluation(tmp_path):
    """`training=False` is what keeps held-out tapes out of the statistics, and it is
    set by hand, so the control arm below shows the same evaluation does move them."""
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    train = VecNormalize(_lanes())
    model = _ppo(train)
    model.learn(total_timesteps=_STEPS)
    stats_path = str(tmp_path / "stats.pkl")
    train.save(stats_path)

    def restored(training):
        env = VecNormalize.load(stats_path, _lanes(EVAL_SEED_BASE))
        env.training = training
        env.norm_reward = False
        return env

    def means(env):
        return {key: rms.mean.copy() for key, rms in env.obs_rms.items()}

    saved = means(train)
    assert train.obs_rms["closes"].count > 1

    def evaluate(env):
        return evaluate_policy(model, env, n_eval_episodes=2, warn=False)

    frozen = restored(training=False)
    np.testing.assert_equal(means(frozen), saved)
    frozen_count = frozen.obs_rms["closes"].count
    frozen_reward, _ = evaluate(frozen)
    assert frozen.obs_rms["closes"].count == frozen_count
    np.testing.assert_equal(means(frozen), saved)

    control = restored(training=True)
    evaluate(control)
    assert control.obs_rms["closes"].count > frozen_count

    # norm_reward=False: evaluation reports the environment's reward, not a scaled one.
    raw = _lanes(EVAL_SEED_BASE)
    frozen_again = restored(training=False)
    observation = frozen_again.reset()
    raw.reset()
    action, _ = model.predict(observation, deterministic=True)
    _, reported, _, _ = frozen_again.step(action)
    _, native, _, _ = raw.step(action)
    # VecNormalize hands rewards back at float32; a scaled reward would differ by far more.
    np.testing.assert_array_equal(reported, native.astype(np.float32))
    assert np.isfinite(frozen_reward)


def test_runnable_recipe_trains_checkpoints_resumes_and_evaluates(tmp_path):
    """The recipe is what `support-status.md` calls workflow-tested, so it runs here,
    in its own process, against the installed package."""
    completed = subprocess.run(
        [
            sys.executable,
            str(_RECIPE),
            "--timesteps=64",
            "--resume-timesteps=64",
            "--num-envs=2",
            "--n-symbols=2",
            "--n-days=12",
            "--eval-episodes=2",
            f"--out-dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"evaluation lanes: [{EVAL_SEED_BASE}, {EVAL_SEED_BASE + 1}]" in completed.stdout
    assert "train lanes:      [0, 1]" in completed.stdout
    assert (tmp_path / "ppo.zip").is_file()
    assert (tmp_path / "vecnormalize.pkl").is_file()
