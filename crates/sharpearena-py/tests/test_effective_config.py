"""Effective-config readback must fail on a mislabelled evidence arm."""

from __future__ import annotations

import pytest

pytest.importorskip("sharpearena.sharpearena_py")

from sharpearena import (  # noqa: E402
    EffectiveConfigError,
    SharpeArenaEnv,
    check_env_effective_config,
    merge_effective_configs,
)


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
