"""Effective-config readback must fail on a mislabelled evidence arm."""

from __future__ import annotations

import pytest

pytest.importorskip("sharpearena.sharpearena_py")

from sharpearena import (  # noqa: E402
    EffectiveConfigError,
    PreprocessingConfig,
    SharpeArenaEnv,
    check_env_effective_config,
    env_effective_config,
    make_preprocessed_env,
    merge_effective_configs,
    wrapper_stack,
)
from sharpearena.vector import SharpeArenaVectorEnv  # noqa: E402
from sharpearena.wrappers import (  # noqa: E402
    CausalNormalizeObservation,
    CausalNormalizeReward,
    TimeLimit,
)
from sharpearena.wrappers_vector import (  # noqa: E402
    VectorCausalNormalizeObservation,
    VectorRecordEpisodeStatistics,
)

OBSERVATION_NORMALIZER = "sharpearena.wrappers.CausalNormalizeObservation"
REWARD_NORMALIZER = "sharpearena.wrappers.CausalNormalizeReward"


def test_effective_config_reads_the_panel_that_the_environment_built() -> None:
    env = SharpeArenaEnv(
        n_symbols=3,
        n_days=41,
        seed=17,
        distribution_mode="hard",
        vol_clustering=0.25,
    )

    effective = check_env_effective_config(
        env,
        seed=17,
        n_symbols=3,
        n_days=41,
        distribution_mode="hard",
        vol_clustering=0.25,
    )

    assert effective["n_symbols"] == 3
    assert effective["n_bars"] == 41
    assert effective["verified"] is True
    assert len(effective["dataset_fnv1a64"]) == 16


def test_effective_config_refuses_a_tier_label_the_environment_did_not_run() -> None:
    env = SharpeArenaEnv(seed=23, distribution_mode="calm")

    with pytest.raises(EffectiveConfigError, match="tape fingerprint"):
        check_env_effective_config(env, seed=23, distribution_mode="extreme")


def test_effective_config_refuses_a_shape_label_the_environment_did_not_run() -> None:
    env = SharpeArenaEnv(n_symbols=2, n_days=31, seed=29)

    with pytest.raises(EffectiveConfigError, match="n_symbols requested 4"):
        check_env_effective_config(env, seed=29, n_symbols=4, n_days=31)


def test_merge_refuses_an_empty_or_shape_inconsistent_arm() -> None:
    with pytest.raises(EffectiveConfigError, match="no effective-config readbacks"):
        merge_effective_configs({})

    first = check_env_effective_config(
        SharpeArenaEnv(n_days=30, seed=1), seed=1, n_days=30
    )
    second = check_env_effective_config(
        SharpeArenaEnv(n_days=31, seed=2), seed=2, n_days=31
    )
    with pytest.raises(EffectiveConfigError, match="this is not one arm"):
        merge_effective_configs({1: first, 2: second})


def test_wrapper_order_is_recorded_innermost_first_and_order_is_identity() -> None:
    observation_first = CausalNormalizeReward(CausalNormalizeObservation(SharpeArenaEnv(seed=1)))
    reward_first = CausalNormalizeObservation(CausalNormalizeReward(SharpeArenaEnv(seed=1)))

    assert wrapper_stack(observation_first) == [OBSERVATION_NORMALIZER, REWARD_NORMALIZER]
    assert wrapper_stack(reward_first) == [REWARD_NORMALIZER, OBSERVATION_NORMALIZER]
    assert env_effective_config(observation_first)["wrapper_stack"] == [
        OBSERVATION_NORMALIZER,
        REWARD_NORMALIZER,
    ]


def test_the_preprocessing_pipeline_reports_the_order_it_documents() -> None:
    env = make_preprocessed_env(
        PreprocessingConfig(max_episode_steps=5, causal_normalize_reward=True, flatten=True),
        seed=1,
    )
    assert [name.rsplit(".", 1)[1] for name in wrapper_stack(env)] == [
        "TimeLimit",
        "CausalNormalizeObservation",
        "CausalNormalizeReward",
        "RecordEpisodeStatistics",
        "FlattenObservation",
    ]


def test_a_wrapped_environment_reads_back_the_native_panel_beneath_it() -> None:
    bare = env_effective_config(SharpeArenaEnv(n_days=33, seed=4))
    wrapped = env_effective_config(TimeLimit(SharpeArenaEnv(n_days=33, seed=4), 5))

    assert bare.pop("wrapper_stack") == []
    assert wrapped.pop("wrapper_stack") == ["sharpearena.wrappers.TimeLimit"]
    assert wrapped == bare


def test_vector_wrapper_order_is_recorded_from_the_vector_stack() -> None:
    env = VectorRecordEpisodeStatistics(
        VectorCausalNormalizeObservation(SharpeArenaVectorEnv(num_envs=2, n_symbols=2, n_days=9))
    )
    assert [name.rsplit(".", 1)[1] for name in wrapper_stack(env)] == [
        "VectorCausalNormalizeObservation",
        "VectorRecordEpisodeStatistics",
    ]


def test_check_refuses_a_wrapper_stack_the_caller_did_not_declare() -> None:
    env = CausalNormalizeObservation(SharpeArenaEnv(n_days=31, seed=3))

    with pytest.raises(EffectiveConfigError, match="wrapper stack requested"):
        check_env_effective_config(env, seed=3, n_days=31)
    with pytest.raises(EffectiveConfigError, match="wrapper stack requested"):
        check_env_effective_config(
            SharpeArenaEnv(n_days=31, seed=3),
            seed=3,
            n_days=31,
            wrappers=[OBSERVATION_NORMALIZER],
        )

    effective = check_env_effective_config(
        env, seed=3, n_days=31, wrappers=[OBSERVATION_NORMALIZER]
    )
    assert effective["verified"] is True


def _wrapped_block(seed: int, *stack: type) -> dict:
    env = SharpeArenaEnv(n_days=30, seed=seed)
    for wrapper in stack:
        env = wrapper(env)
    return check_env_effective_config(env, seed=seed, n_days=30, wrappers=wrapper_stack(env))


def test_merge_refuses_seeds_that_ran_different_wrapper_orders() -> None:
    first = _wrapped_block(1, CausalNormalizeObservation, CausalNormalizeReward)
    second = _wrapped_block(2, CausalNormalizeReward, CausalNormalizeObservation)
    with pytest.raises(EffectiveConfigError, match="wrapper_stack.*this is not one arm"):
        merge_effective_configs({1: first, 2: second})


def test_merge_refuses_a_block_with_no_wrapper_readback() -> None:
    block = _wrapped_block(1)
    del block["wrapper_stack"]
    with pytest.raises(EffectiveConfigError, match="seed 2 has no wrapper_stack"):
        merge_effective_configs({1: _wrapped_block(1), 2: block})


def test_merge_carries_a_stack_and_keeps_the_frozen_shape_for_a_bare_arm() -> None:
    merged = merge_effective_configs(
        {
            1: _wrapped_block(1, CausalNormalizeObservation),
            2: _wrapped_block(2, CausalNormalizeObservation),
        }
    )
    assert merged["wrapper_stack"] == [OBSERVATION_NORMALIZER]

    bare = merge_effective_configs({1: _wrapped_block(1), 2: _wrapped_block(2)})
    assert set(bare) == {
        "n_symbols",
        "n_bars",
        "window_start",
        "window_end",
        "verified",
        "readback",
        "dataset_fnv1a64",
    }


@pytest.mark.parametrize("collector", [None, {}, {7: {"old": True}}])
def test_every_constructed_baseline_consumer_is_verified(monkeypatch, collector):
    from sharpearena import baselines

    checked = []
    real_check = baselines.check_env_effective_config

    def check(env, **kwargs):
        checked.append(env)
        return real_check(env, **kwargs)

    monkeypatch.setattr(baselines, "check_env_effective_config", check)
    monkeypatch.setattr(baselines, "BASELINE_POLICIES", [("one", object), ("two", object)])
    monkeypatch.setattr(baselines, "_rollout_returns", lambda *_args: [0.01, -0.005])
    monkeypatch.setattr(baselines, "score_run", lambda *_args: '{"passed_k": false}')
    collector = None if collector is None else dict(collector)
    for _ in range(2):
        baselines.run_baselines(n_symbols=2, n_days=30, seeds=[7], confidence=False, readback=collector)
    assert len(checked) == 4
    assert len({id(env) for env in checked}) == 4


@pytest.mark.parametrize("collector", [None, {}, {7: {"old": True}}])
def test_later_misconfigured_baseline_consumer_is_refused_before_rollout(monkeypatch, collector):
    from sharpearena import baselines

    real_factory = baselines._make_env
    made = []
    rolled = []

    def factory(n_symbols, n_days, seed, mode):
        env = real_factory(n_symbols, n_days, seed, mode if not made else "extreme")
        made.append(env)
        return env

    def rollout(env, *_args):
        rolled.append(env)
        return [0.01, -0.005]

    monkeypatch.setattr(baselines, "_make_env", factory)
    monkeypatch.setattr(baselines, "BASELINE_POLICIES", [("one", object), ("two", object)])
    monkeypatch.setattr(baselines, "_rollout_returns", rollout)
    monkeypatch.setattr(baselines, "score_run", lambda *_args: '{"passed_k": false}')
    collector = None if collector is None else dict(collector)
    with pytest.raises(EffectiveConfigError, match="tape fingerprint"):
        baselines.run_baselines(n_symbols=2, n_days=30, seeds=[7], confidence=False, readback=collector)
    assert len(made) == 2
    assert rolled == made[:1]
