"""MMParams validates domains through construction, replacement and env overrides."""

from dataclasses import replace

import numpy as np
import pytest
from sharpearena.market_making import (
    MMParams,
    MarketMakingEnv,
    closed_form_reference_policy,
)


REAL_FIELDS = (
    "sigma",
    "gamma",
    "kappa",
    "arrival_rate",
    "dt",
    "s0",
    "phi",
    "terminal_liq_penalty",
    "max_depth",
    "tick_size",
    "initial_cash",
)


@pytest.mark.parametrize("name", REAL_FIELDS)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), True, "1"]
)
def test_real_parameter_requires_finite_numeric_value(name, value):
    with pytest.raises(ValueError, match=name):
        MMParams(**{name: value})


@pytest.mark.parametrize(
    "name", [name for name in REAL_FIELDS if name != "initial_cash"]
)
def test_parameters_with_nonnegative_meaning_reject_negative_values(name):
    with pytest.raises(ValueError, match=name):
        MMParams(**{name: -0.1})


@pytest.mark.parametrize("name", ["gamma", "kappa", "dt", "tick_size"])
def test_positive_parameters_reject_zero(name):
    with pytest.raises(ValueError, match=name):
        MMParams(**{name: 0})


@pytest.mark.parametrize("name", ["n_steps", "inventory_cap"])
@pytest.mark.parametrize("value", [True, 1.5, "1", float("nan"), -1])
def test_integer_parameters_require_exact_integer_domain(name, value):
    with pytest.raises(ValueError, match=name):
        MMParams(**{name: value})


def test_episode_horizon_requires_at_least_one_step():
    with pytest.raises(ValueError, match="n_steps"):
        MMParams(n_steps=0)


@pytest.mark.parametrize("name", ["phi", "terminal_liq_penalty"])
def test_parameter_replacement_and_environment_overrides_validate_penalty_sign(name):
    params = MMParams()
    with pytest.raises(ValueError, match=name):
        replace(params, **{name: -0.1})
    with pytest.raises(ValueError, match=name):
        MarketMakingEnv(params, **{name: -0.1})


def test_zero_limits_remain_valid_and_execute_without_trading():
    params = MMParams(
        sigma=0,
        arrival_rate=0,
        phi=0,
        terminal_liq_penalty=0,
        max_depth=0,
        inventory_cap=0,
        s0=0,
        initial_cash=-10,
        n_steps=1,
    )
    env = MarketMakingEnv(params)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = closed_form_reference_policy(params)(obs)
    assert np.array_equal(action, np.zeros(2))
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert reward == 0
    assert info["value"] == -10
    assert info["inventory"] == 0
    assert terminated and not truncated


def test_numpy_numeric_parameters_preserve_values():
    params = MMParams(
        sigma=np.float64(0), n_steps=np.int64(1), inventory_cap=np.int64(0)
    )
    assert params.sigma == 0
    assert params.n_steps == 1
    assert params.inventory_cap == 0


@pytest.mark.parametrize("name", [*REAL_FIELDS, "n_steps", "inventory_cap"])
def test_unrepresentable_integer_input_is_rejected_at_construction(name):
    with pytest.raises(ValueError, match=name):
        MMParams(**{name: 10**400})


def test_nonfinite_derived_horizon_is_rejected_before_environment_construction():
    with pytest.raises(ValueError, match="finite horizon"):
        MMParams(n_steps=100, dt=1e308)
