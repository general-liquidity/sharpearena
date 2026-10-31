"""A consumer of the Minari export: rollout, dataset on disk, fitted learner.

The round-trip tests in ``test_minari.py`` read fields back. This reads the dataset the
way an offline learner does, through ``minari.load_dataset`` and the n+1 observation
convention, and fits behaviour cloning on it by least squares, which needs nothing beyond
NumPy. The behaviour policy is an exact linear function of the observation it saw, so
the fit recovers it only if ``observations[t]`` really is what produced ``actions[t]``:
an off-by-one in the export leaves a large residual rather than a smaller score.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sharpearena import SharpeArenaEnv
from sharpearena.dataset import EVAL_SEED_BASE
from sharpearena.minari_export import to_minari_train_test
from sharpearena.trace import RolloutTraceWriter

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("minari") is None, reason="minari not installed"
)

_N_SYMBOLS = 2
_N_DAYS = 40
_SCALE = 0.2

# Stored actions are float32, so an exact fit lands within a few float32 epsilons of 0.1.
_RECOVERED = 1e-6
# What a one-step misalignment costs on these tapes, two orders above that.
_MISALIGNED = 1e-4


def _relative_closes(closes):
    closes = np.asarray(closes, dtype=np.float64)
    return closes / closes.sum(axis=-1, keepdims=True)


def _behaviour(observation):
    """Target weights that depend on this bar's closes and on nothing else."""
    return (_SCALE * _relative_closes(observation["closes"])).astype(np.float32)


def _record(seed, path):
    env = SharpeArenaEnv(n_symbols=_N_SYMBOLS, n_days=_N_DAYS, seed=seed)
    observation, _ = env.reset(seed=seed)
    with RolloutTraceWriter(path) as writer:
        for step in range(_N_DAYS):
            decision = _behaviour(observation)
            following, reward, terminated, truncated, info = env.step(decision)
            writer.record_step(
                step=step,
                observation=observation,
                decision=decision,
                reward=reward,
                info=info,
            )
            observation = following
            if terminated or truncated:
                break
        writer.finalize()
    return env


def _design_matrix(dataset, *, action_offset=0):
    """(features, targets) from the dataset alone. Actions pair with `observations[t]`;
    the n+1th observation has no action and never enters, unless `action_offset` shifts
    the pairing on purpose."""
    features, targets = [], []
    for episode in dataset.iterate_episodes():
        actions = np.asarray(episode.actions, dtype=np.float64)
        closes = np.asarray(episode.observations["closes"], dtype=np.float64)
        rows = closes[action_offset : len(actions) + action_offset]
        features.append(_relative_closes(rows))
        targets.append(actions)
    return np.concatenate(features), np.concatenate(targets)


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path / "minari"))
    env = _record(3, str(tmp_path / "train.jsonl"))
    _record(EVAL_SEED_BASE + 3, str(tmp_path / "test.jsonl"))
    train, test = to_minari_train_test(
        str(tmp_path / "train.jsonl"),
        str(tmp_path / "test.jsonl"),
        "sharpearena/learner-v0",
        observation_space=env.observation_space,
        action_space=env.action_space,
        author="sharpearena-tests",
    )
    return train.id, test.id


def _held_out_error(train, test, *, action_offset=0):
    train_x, train_y = _design_matrix(train, action_offset=action_offset)
    test_x, test_y = _design_matrix(test, action_offset=action_offset)
    coefficients, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return float(np.abs(test_x @ coefficients - test_y).max())


def test_learner_fitted_on_the_stored_dataset_recovers_the_behaviour_on_held_out_tapes(
    exported,
):
    import minari

    train, test = (minari.load_dataset(dataset_id) for dataset_id in exported)
    assert train.total_steps == test.total_steps == _N_DAYS

    assert _held_out_error(train, test) < _RECOVERED


def test_a_consumer_pairing_actions_with_the_next_observation_is_detected(exported):
    """The control for the test above: the same fit on a shifted pairing must fail,
    otherwise a misaligned export could pass it."""
    import minari

    train, test = (minari.load_dataset(dataset_id) for dataset_id in exported)

    assert _held_out_error(train, test, action_offset=1) > _MISALIGNED


def test_every_stored_episode_ends_on_the_flag_a_bootstrapping_learner_needs(exported):
    """An offline value learner bootstraps a truncation and zeroes a termination, and it
    reads that from the dataset. These tapes ran out of bars with positive NAV."""
    import minari

    for dataset_id in exported:
        for episode in minari.load_dataset(dataset_id).iterate_episodes():
            assert bool(episode.truncations[-1]) and not bool(episode.terminations[-1])
            assert not np.any(episode.truncations[:-1])
            assert len(episode.observations["closes"]) == len(episode.actions) + 1
