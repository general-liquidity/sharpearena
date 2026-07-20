"""The information-disclosure difficulty axis (observation richness).

These exercise the native ``PyMarketClearing`` directly (no pettingzoo needed): the
``richness`` tier is orthogonal to ``distribution_mode`` and controls how much of the
market each observation surfaces. ``standard`` must reproduce the historical disclosure
byte-for-byte; ``data_poor`` withholds bars and optional fields; ``data_rich`` surfaces
more bars plus fundamentals and news. None of them ever reveal a future bar.
"""

import json

import pytest

from sharpearena.sharpearena_py import PyMarketClearing


def _rollout(market, orders, steps=6):
    """The reset observations plus each step's observations, as parsed JSON."""
    log = [json.loads(market.reset_market())["observations"]]
    for _ in range(steps):
        result = json.loads(market.step_market(json.dumps(orders)))
        log.append(result["observations"])
        if result["done"]:
            break
    return log


def _flat_orders(n_agents, n_symbols):
    return [[0.0] * n_symbols for _ in range(n_agents)]


def test_default_richness_is_standard_and_byte_identical():
    """Omitting ``richness`` and passing ``standard`` clear a byte-identical stream."""
    common = dict(n_symbols=3, n_days=60, seed=4, n_agents=2, capital=1.0)
    orders = [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]
    default_market = PyMarketClearing(**common)
    standard_market = PyMarketClearing(**common, richness="standard")
    assert _rollout(default_market, orders) == _rollout(standard_market, orders)


def test_richness_getter_reports_the_active_disclosure():
    poor = json.loads(PyMarketClearing(n_symbols=2, n_days=40, richness="data_poor").richness)
    std = json.loads(PyMarketClearing(n_symbols=2, n_days=40, richness="standard").richness)
    rich = json.loads(PyMarketClearing(n_symbols=2, n_days=40, richness="data_rich").richness)
    assert poor == {"lookback": 3, "fundamentals": False, "news": False}
    assert std == {"lookback": 20, "fundamentals": False, "news": False}
    assert rich == {"lookback": 50, "fundamentals": True, "news": True}


def test_data_poor_withholds_bars_and_optional_fields():
    market = PyMarketClearing(n_symbols=2, n_days=60, seed=7, n_agents=2, richness="data_poor")
    obs = json.loads(market.reset_market())["observations"]
    for agent_obs in obs:
        for snap in agent_obs["symbols"]:
            assert len(snap["close_history"]) <= 3
            assert snap["fundamentals"] == {}
            assert snap["news"] == []


def test_data_rich_surfaces_more_bars_and_populates_fields():
    rich = PyMarketClearing(n_symbols=2, n_days=120, seed=9, n_agents=2, richness="data_rich")
    standard = PyMarketClearing(n_symbols=2, n_days=120, seed=9, n_agents=2, richness="standard")
    rich_obs = json.loads(rich.reset_market())["observations"]
    std_obs = json.loads(standard.reset_market())["observations"]
    for ro, so in zip(rich_obs, std_obs):
        for rs, ss in zip(ro["symbols"], so["symbols"]):
            assert len(rs["close_history"]) > len(ss["close_history"])
            assert len(rs["close_history"]) <= 50
            assert set(rs["fundamentals"]) == {"trailing_return", "window_high", "window_low"}
            assert len(rs["news"]) == 1
            assert rs["symbol"] in rs["news"][0]


def test_every_tier_is_leak_free_last_close_is_this_bars_cleared_mid():
    """Under every tier the last surfaced close equals this bar's cleared mid (never a
    future bar), and the surfaced window never exceeds the cleared-bar count."""
    for tier in ("data_poor", "standard", "data_rich"):
        market = PyMarketClearing(
            n_symbols=3, n_days=80, seed=2, n_agents=2, richness=tier
        )
        meta = json.loads(market.reset_market())
        cleared_bars = meta["start_bar"]
        orders = _flat_orders(2, 3)
        while True:
            result = json.loads(market.step_market(json.dumps(orders)))
            cleared_bars += 1
            mids = result["cleared_mids"]
            for agent_obs in result["observations"]:
                for s, snap in enumerate(agent_obs["symbols"]):
                    assert snap["close_history"][-1] == mids[s]
                    assert len(snap["close_history"]) <= cleared_bars
            if result["done"]:
                break


def test_unknown_richness_raises():
    with pytest.raises(ValueError, match="unknown richness"):
        PyMarketClearing(n_symbols=2, n_days=40, richness="bogus")
