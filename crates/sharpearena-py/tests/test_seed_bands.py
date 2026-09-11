"""The held-out band boundary has one definition, and the deliberate restatements agree.

``EVAL_SEED_BASE`` decides whether an eval seed is disjoint from the train band. It used
to exist as four bare literals; :mod:`sharpearena._seed_bands` is now the single source
the Python surface reads. Two restatements survive on purpose and are checked here by
value rather than by matching text: :mod:`sharpearena.effective_config` restates the
split rule so the readback is derived independently of the environment it checks, and the
native constant is the Rust one.

The band tests drive the seed a consumer actually computes, so a module that reads the
shared value and then offsets by something else fails too.
"""

import importlib

import pytest

from sharpearena import _seed_bands, dataset, effective_config, minari_export
from sharpearena.sharpearena_py import EVAL_SEED_BASE as NATIVE_EVAL_SEED_BASE

BASE = _seed_bands.EVAL_SEED_BASE


def test_python_surface_reads_one_band_definition():
    assert dataset.EVAL_SEED_BASE is _seed_bands.EVAL_SEED_BASE
    assert minari_export.EVAL_SEED_BASE is _seed_bands.EVAL_SEED_BASE


@pytest.mark.parametrize("module_name", ["sharpearena.gym", "sharpearena.vector"])
def test_env_modules_read_one_band_definition(module_name):
    module = importlib.import_module(module_name)
    assert module._EVAL_SEED_BASE is _seed_bands.EVAL_SEED_BASE


def test_deliberate_restatements_agree_with_the_definition():
    # Not imports: effective_config restates the rule independently (its module
    # docstring says why) and the native constant is Rust's. Both must agree in value.
    assert effective_config.EVAL_SEED_BASE == BASE
    assert NATIVE_EVAL_SEED_BASE == BASE


def test_eval_mode_seeds_are_offset_by_the_band_base():
    # The behaviour the constant exists for, on the consumers that compute a seed.
    assert dataset._seed_for("eval", 0, 0) - dataset._seed_for("train", 0, 0) == BASE
    assert dataset._seed_for("eval", 7, 3) == BASE + 10

    import numpy as np

    for user_seed in (0, 3):
        expected = int(np.random.SeedSequence(user_seed + BASE).generate_state(2)[0])
        assert effective_config.resolved_scenario_seed(user_seed, "eval") == expected


def test_gym_env_places_an_eval_seed_in_the_held_out_band():
    from sharpearena.gym import SharpeArenaEnv

    env = SharpeArenaEnv(n_symbols=2, n_days=20, mode="eval")
    try:
        assert env._seed_offset == BASE
        # The offset is what reaches the generator: the resolved scenario seed of user
        # seed 3 under eval must equal the train-mode resolution of 3 + BASE.
        env.reset(seed=3)
        eval_seeds = dict(env._resolved_seeds)
    finally:
        env.close()

    train = SharpeArenaEnv(n_symbols=2, n_days=20, mode="train")
    try:
        train.reset(seed=3 + BASE)
        assert dict(train._resolved_seeds) == eval_seeds
    finally:
        train.close()
