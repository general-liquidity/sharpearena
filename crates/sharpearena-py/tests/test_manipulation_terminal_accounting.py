"""Zero terminal target must not be mistaken for quantity liquidation."""

import math

import numpy as np
import pytest
from sharpearena import manipulation


class MovingPriceMarket:
    """A small deterministic delta-weight market with visibly changing prices."""

    def __init__(self, *, n_symbols, capital, kyle_lambda, **kwargs):
        self.symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
        self.possible_agents = ["agent_0"]
        self.capital = capital
        self.slope = 4.0 * (1.0 + kyle_lambda)

    def _obs(self, price):
        return {
            "agent_0": {
                "positions": self.positions.copy(),
                "closes": np.full(len(self.symbols), price),
                "cash": np.array([self.cash]),
            }
        }

    def reset(self, *, seed):
        self.agents = list(self.possible_agents)
        self.positions = np.zeros(len(self.symbols))
        self.previous = self.positions.copy()
        self.cash = self.capital
        self.bar = 0
        return self._obs(100.0), {}

    def step(self, actions):
        price = 100.0 + self.slope * self.bar
        target = np.asarray(actions["agent_0"], dtype=float)
        delta = (target - self.previous) * self.capital
        self.positions += delta / price
        self.cash -= float(np.sum(delta))
        self.previous = target.copy()
        self.bar += 1
        if self.bar == 20:
            assert np.all(target == 0.0)
            self.agents = []
        nav = self.cash + float(np.sum(self.positions * price))
        return (
            self._obs(price),
            {},
            {},
            {},
            {"agent_0": {"nav": nav, "cleared_mids": [price] * len(self.symbols)}},
        )

    def close(self):
        pass


@pytest.mark.parametrize("asymmetric", [False, True])
def test_target_flat_probes_disclose_residual_shares_and_marked_nav(
    monkeypatch, asymmetric
):
    monkeypatch.setattr(manipulation, "EndogenousMarketEnv", MovingPriceMarket)
    p = manipulation.ManipulationParams(n_symbols=2, n_days=40, n_followers=0)
    if asymmetric:
        result = manipulation.run_asymmetric_probe(
            params=p, schedule=manipulation.AsymmetricSchedule.uniform(5, 3)
        )
    else:
        result = manipulation.run_manipulation_probe(params=p)
    for side in ("live", "reference"):
        positions = getattr(result, f"terminal_positions_{side}")
        closes = getattr(result, f"terminal_closes_{side}")
        cash = getattr(result, f"terminal_cash_{side}")
        marked = getattr(result, f"terminal_marked_position_value_{side}")
        assert positions[0] != 0.0
        assert positions[1] == 0.0
        assert marked == math.fsum(q * price for q, price in zip(positions, closes))
        assert cash == pytest.approx(p.capital)
        pnl = result.live_pnl if side == "live" else result.reference_pnl
        assert pnl != 0.0
        assert cash + marked - p.capital == pytest.approx(pnl)
        assert result.to_dict()[f"terminal_positions_{side}"] == list(positions)
    assert result.to_dict()["pnl_basis"] == "terminal_marked_nav_minus_initial_capital"


def test_existing_rollout_returns_marked_profit_despite_zero_target_and_unchanged_cash(
    monkeypatch,
):
    created = []

    def market(**kwargs):
        instance = MovingPriceMarket(**kwargs)
        created.append(instance)
        return instance

    monkeypatch.setattr(manipulation, "EndogenousMarketEnv", market)
    p = manipulation.ManipulationParams(n_days=40, n_followers=0)
    pnl, _, _ = manipulation._rollout(p, 0)
    env = created[0]
    assert np.all(env.previous == 0.0)
    assert env.positions[0] != 0.0
    assert env.cash == pytest.approx(p.capital)
    assert pnl != 0.0
    price = 100.0 + env.slope * (env.bar - 1)
    assert pnl == pytest.approx(float(np.sum(env.positions * price)))


def test_existing_native_fixture_reconciles_and_preserves_legacy_rollout_tuple():
    p = manipulation.ManipulationParams(
        n_days=40, n_followers=2, push_bars=4, dump_bars=4
    )
    result = manipulation.run_manipulation_probe(params=p, seed=0)
    legacy = manipulation._rollout(p, 0)
    assert len(legacy) == 3
    assert legacy == (result.live_pnl, result.peak_price_move, result.follower_pnl_live)
    for side in ("live", "reference"):
        positions = getattr(result, f"terminal_positions_{side}")
        closes = getattr(result, f"terminal_closes_{side}")
        cash = getattr(result, f"terminal_cash_{side}")
        marked = getattr(result, f"terminal_marked_position_value_{side}")
        assert marked == math.fsum(q * price for q, price in zip(positions, closes))
        nav = p.capital + (result.live_pnl if side == "live" else result.reference_pnl)
        # A few f64 products/additions may use a different reduction order from
        # the native NAV. Scale the rounding allowance by absolute contribution.
        scale = max(
            1.0,
            abs(cash)
            + math.fsum(abs(q * price) for q, price in zip(positions, closes)),
        )
        assert cash + marked == pytest.approx(
            nav, rel=0.0, abs=16 * np.finfo(float).eps * scale
        )


def test_manual_historical_result_does_not_invent_terminal_zeroes():
    result = manipulation.ManipulationResult(
        seed=0,
        profitable=False,
        impact_pnl=0.0,
        live_pnl=0.0,
        reference_pnl=0.0,
        peak_price_move=0.0,
        kyle_lambda=0.1,
        eta=0.05,
        push_weight=0.8,
        follower_gain=0.0,
    )
    record = result.to_dict()
    assert record["terminal_positions_live"] is None
    assert record["terminal_cash_reference"] is None
    assert record["terminal_marked_position_value_live"] is None
