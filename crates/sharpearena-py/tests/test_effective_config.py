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
