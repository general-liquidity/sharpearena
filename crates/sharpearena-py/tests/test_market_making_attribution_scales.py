"""Finite synthetic accounting scales, not an unrestricted arithmetic guarantee.

Balances, prices, quotes and penalties are scaled together from 1e-9 to 1e12.
The diagnostic uses a forward-error budget for the step's straight-line binary64
arithmetic, including the cash/inventory balances that cancel in the reward.
Relative error in reward alone would be inappropriate near zero. Overflow and
nonfinite inputs/results are outside this test's supported numeric domain.
"""

import math
import sys

import numpy as np
import pytest

from sharpearena.market_making import MMParams, MarketMakingEnv

COMPONENTS = (
    "spread_capture",
    "inventory_pnl",
    "inventory_penalty",
    "liquidation_cost",
)


def _accounting_budget(*terms):
    # At most 32 rounded arithmetic operations lie on either reconciliation
    # path (quote, cash, mark, penalty, liquidation and four-component sum).
    # gamma_n = n*u/(1-n*u), u = binary64 unit roundoff. The sum of absolute
    # intermediate magnitudes bounds cancellation as well as relative error.
    assert all(math.isfinite(term) for term in terms), "nonfinite accounting term"
    magnitude = math.fsum(abs(term) for term in terms)
    assert math.isfinite(magnitude), "accounting magnitude overflow"
    unit_roundoff = sys.float_info.epsilon / 2
    gamma = 32 * unit_roundoff / (1 - 32 * unit_roundoff)
    budget = gamma * magnitude + 32 * math.ulp(0.0)
    assert math.isfinite(budget), "accounting budget overflow"
    return budget


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1.0, 1e6, 1e12])
@pytest.mark.parametrize("cash_multiplier", [-100.0, 0.0, 100.0])
def test_attribution_reconciles_across_finite_currency_scales(scale, cash_multiplier):
    params = MMParams(
        n_steps=24,
        s0=100 * scale,
        sigma=scale,
        kappa=1 / scale,
        phi=0.01 * scale,
        terminal_liq_penalty=0.1 * scale,
        max_depth=10 * scale,
        tick_size=0.01 * scale,
        initial_cash=cash_multiplier * scale,
        inventory_cap=5,
    )
    fill_count = liquidation_count = 0
    for seed in range(3):
        env = MarketMakingEnv(params)
        env.reset(seed=seed)
        # Binary64 actions avoid accidentally testing float32 quote quantization.
        action = np.array([0.25 * scale, 0.5 * scale], dtype=np.float64)
        for step in range(params.n_steps):
            cash_before, q_before, mid_before = env._cash, env._q, env._mid
            _, reward, terminated, truncated, info = env.step(action)
            ask, bid = info["ask_fills"], info["bid_fills"]
            q = q_before + bid - ask
            components = [info[key] for key in COMPONENTS]
            terms = (
                cash_before,
                env._cash,
                q_before * mid_before,
                q * info["mid"],
                ask * (mid_before + action[1]),
                bid * (mid_before - action[0]),
                reward,
                *components,
            )
            budget = _accounting_budget(*terms)
            assert abs(math.fsum(components) - reward) <= budget
            assert info["spread_capture"] == ask * action[1] + bid * action[0]
            assert info["inventory_pnl"] == q * (info["mid"] - mid_before)
            assert info["inventory_penalty"] == -params.phi * q**2
            expected_liquidation = (
                -abs(q) * params.terminal_liq_penalty if terminated else 0
            )
            assert abs(info["liquidation_cost"] - expected_liquidation) <= budget
            assert terminated == (step == params.n_steps - 1)
            assert not truncated
            fill_count += ask + bid
            liquidation_count += terminated and q != 0
    assert fill_count > 0 and liquidation_count > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_accounting_budget_does_not_mask_nonfinite_results(bad):
    with pytest.raises(AssertionError, match="nonfinite"):
        _accounting_budget(1.0, bad)


def test_accounting_budget_does_not_mask_magnitude_overflow():
    with pytest.raises(OverflowError):
        _accounting_budget(sys.float_info.max, sys.float_info.max)


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e12])
def test_budget_cannot_hide_a_missing_material_component(scale):
    budget = _accounting_budget(100 * scale, -100 * scale, 3 * scale)
    assert 0.01 * scale > 1000 * budget
