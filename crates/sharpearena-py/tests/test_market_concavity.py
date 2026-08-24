"""The opt-in concave permanent-impact exponent on the endogenous shared-book market.

`PyMarketClearing` accepts an `impact_exponent` (ctor kwarg or `set_impact_exponent`):
`1.0` (the default) is the frozen linear golden-hash path and must stay byte-identical
to the pre-concavity binding; below `1.0` the permanent (Kyle) flow term becomes
`sign(Q) * |Q|**exponent`, concave in flow, which is the Huberman-Stanzl regime in
which round-trip manipulation can pay. The `EndogenousMarketEnv` and the manipulation
probe reach the same knob through their `impact_exponent` kwarg/field.
"""

import importlib.util
import json

import numpy as np
import pytest

from sharpearena.sharpearena_py import PyMarketClearing

_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_pz = pytest.mark.skipif(not _HAS_PETTINGZOO, reason="pettingzoo not installed")


def _market(**kwargs):
    # capital = 100 on a ~100-priced panel puts a coordinated 0.8-weight buy at
    # |Q| > 1 flow units, the regime where a concave exponent charges LESS than linear.
    return PyMarketClearing(
        n_symbols=1, n_days=40, seed=0, n_agents=2, capital=100.0, **kwargs
    )


def _buy_orders():
    return json.dumps([[0.8], [0.8]])


def _small_market(**kwargs):
    # capital = 1 puts the same buy at 0 < |Q| < 1, where concave charges MORE.
    return PyMarketClearing(
        n_symbols=1, n_days=40, seed=0, n_agents=2, capital=1.0, **kwargs
    )


# ---------------------------------------------------------------------------
# Default path: exponent 1.0 == the exact legacy arithmetic
# ---------------------------------------------------------------------------


def test_default_exponent_is_one_and_byte_identical_to_omitting_it():
    plain = _market()
    explicit = _market(impact_exponent=1.0)
    assert plain.impact_exponent == 1.0
    assert explicit.impact_exponent == 1.0
    for _ in range(5):
        assert plain.step_market(_buy_orders()) == explicit.step_market(_buy_orders())


def test_set_impact_exponent_back_to_one_restores_the_linear_path():
    roundtrip = _market(impact_exponent=0.5)
    roundtrip.set_impact_exponent(1.0)
    linear = _market()
    for _ in range(5):
        assert roundtrip.step_market(_buy_orders()) == linear.step_market(_buy_orders())


def test_exponent_composes_with_the_uncertainty_set_wire_shape():
    m = _market(impact_exponent=0.5, lambda_radius=0.05, eta_radius=0.02)
    r = json.loads(m.step_market(_buy_orders()))
    assert "robust_impact" in r
    plain = json.loads(_market(impact_exponent=0.5).step_market(_buy_orders()))
    assert "robust_impact" not in plain


# ---------------------------------------------------------------------------
# Concavity: sqrt impact undercharges large flow, overcharges small flow
# ---------------------------------------------------------------------------


def _first_fill_impact(market):
    r = json.loads(market.step_market(_buy_orders()))
    fill = r["fills"][0][0]["fill_price"]
    mid = r["cleared_mids"][0]
    return fill - mid, r["net_flow"][0]


def test_concave_exponent_charges_large_trades_less_than_linear():
    linear_impact, q = _first_fill_impact(_market())
    concave_impact, q2 = _first_fill_impact(_market(impact_exponent=0.5))
    assert q == q2  # sizing shares the cleared mid, so the flow is identical
    assert q > 1.0
    assert concave_impact < linear_impact


def test_concave_exponent_charges_small_trades_more_than_linear():
    linear_impact, q = _first_fill_impact(_small_market())
    concave_impact, q2 = _first_fill_impact(_small_market(impact_exponent=0.5))
    assert q == q2
    assert 0.0 < q < 1.0
    assert concave_impact > linear_impact


def test_concave_clearing_is_deterministic():
    runs = []
    for _ in range(2):
        m = _market(impact_exponent=0.5)
        runs.append([m.step_market(_buy_orders()) for _ in range(5)])
    assert runs[0] == runs[1]


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_non_positive_or_non_finite_exponent_raises(bad):
    with pytest.raises(ValueError):
        _market(impact_exponent=bad)
    m = _market()
    with pytest.raises(ValueError):
        m.set_impact_exponent(bad)


# ---------------------------------------------------------------------------
# EndogenousMarketEnv + manipulation-probe threading
# ---------------------------------------------------------------------------


@needs_pz
def test_env_threads_impact_exponent_to_the_native_market():
    from sharpearena.market_env import EndogenousMarketEnv

    env_lin = EndogenousMarketEnv(
        n_agents=2, n_symbols=1, n_days=40, seed=0, capital=100.0
    )
    env_con = EndogenousMarketEnv(
        n_agents=2, n_symbols=1, n_days=40, seed=0, capital=100.0, impact_exponent=0.5
    )
    assert env_lin.impact_exponent == 1.0
    assert env_con.impact_exponent == 0.5
    env_lin.reset(seed=0)
    env_con.reset(seed=0)
    act = {a: np.array([0.8], dtype=np.float32) for a in env_lin.agents}
    _, _, _, _, info_lin = env_lin.step(dict(act))
    _, _, _, _, info_con = env_con.step(dict(act))
    fill_lin = info_lin["agent_0"]["fills"][0]["fill_price"]
    fill_con = info_con["agent_0"]["fills"][0]["fill_price"]
    mid = info_lin["agent_0"]["cleared_mids"][0]
    assert info_con["agent_0"]["cleared_mids"][0] == mid
    assert fill_con - mid < fill_lin - mid  # |Q| > 1: concave charges less


@needs_pz
def test_manipulation_params_carry_the_exponent():
    from sharpearena.manipulation import ManipulationParams, run_manipulation_probe

    with pytest.raises(ValueError):
        ManipulationParams(impact_exponent=0.0)
    p = ManipulationParams(n_days=40, impact_exponent=0.5)
    res = run_manipulation_probe(params=p, seed=0)
    assert res.impact_exponent == 0.5
    assert res.to_dict()["impact_exponent"] == 0.5
    lin = run_manipulation_probe(params=ManipulationParams(n_days=40), seed=0)
    assert lin.impact_exponent == 1.0
    # The exponent changes the cleared dynamics, so the attributable PnL moves.
    assert res.impact_pnl != lin.impact_pnl
