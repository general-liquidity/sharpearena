"""The elliptic uncertainty set on the endogenous shared-book market.

`PyMarketClearing` clears each bar at the worst-case `(lambda, eta)` inside an
`EllipticUncertaintySet` when one is installed (ctor kwargs `lambda_radius` /
`eta_radius` / `uncertainty_correlation`, or `set_uncertainty`); with no set the
point-estimate path must stay byte-identical to the pre-uncertainty binding. The
`EndogenousMarketEnv` reaches the same knob through its opt-in `uncertainty` kwarg.
"""

import importlib.util
import json

import numpy as np
import pytest

from sharpearena.market_env import _normalize_uncertainty
from sharpearena.sharpearena_py import PyMarketClearing

_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_pz = pytest.mark.skipif(not _HAS_PETTINGZOO, reason="pettingzoo not installed")

N_SYMBOLS = 2
N_AGENTS = 2


def _market(**kwargs):
    return PyMarketClearing(
        n_symbols=N_SYMBOLS, n_days=40, seed=7, n_agents=N_AGENTS, **kwargs
    )


def _buy_orders():
    # Both agents buy, so aggregate flow is nonzero and impact uncertainty binds.
    return json.dumps([[0.5] * N_SYMBOLS for _ in range(N_AGENTS)])


# ---------------------------------------------------------------------------
# Point-estimate path: absent set == the exact legacy arithmetic and wire shape
# ---------------------------------------------------------------------------


def test_absent_uncertainty_is_byte_identical_and_omits_robust_field():
    plain = _market()
    explicit_none = _market()
    explicit_none.set_uncertainty()  # both radii None -> removes/keeps no set
    assert plain.uncertainty is None
    assert explicit_none.uncertainty is None

    for _ in range(5):
        a = plain.step_market(_buy_orders())
        b = explicit_none.step_market(_buy_orders())
        assert a == b
        assert "robust_impact" not in json.loads(a)


def test_zero_radius_worst_case_equals_point_estimate_numbers():
    point = _market()
    zero = _market(lambda_radius=0.0, eta_radius=0.0)
    a = json.loads(point.step_market(_buy_orders()))
    b = json.loads(zero.step_market(_buy_orders()))
    # A degenerate (zero-radius) set charges nothing extra; only the provenance field
    # is added to the wire shape.
    assert "robust_impact" in b
    b.pop("robust_impact")
    assert a == b


# ---------------------------------------------------------------------------
# Worst case charged when a set is present
# ---------------------------------------------------------------------------


def test_worst_case_is_charged_for_buyers():
    set_kwargs = dict(lambda_radius=0.05, eta_radius=0.02, uncertainty_correlation=0.0)
    point = _market()
    robust = _market(**set_kwargs)
    assert json.loads(robust.uncertainty) == {
        "lambda_radius": 0.05,
        "eta_radius": 0.02,
        "correlation": 0.0,
    }

    p = json.loads(point.step_market(_buy_orders()))
    r = json.loads(robust.step_market(_buy_orders()))

    coefficients = r["robust_impact"]
    assert len(coefficients) == N_SYMBOLS
    for c in coefficients:
        # Buying flow: the worst case raises both impact coefficients above the point
        # estimate, inside the axis-aligned box that bounds the ellipse.
        assert 0.1 < c["lambda"] <= 0.1 + 0.05 + 1e-12
        assert 0.05 < c["eta"] <= 0.05 + 0.02 + 1e-12
    # Every buyer fills at a worse (higher) price and books a worse bar return.
    for pf_row, rf_row in zip(p["fills"], r["fills"]):
        for pf, rf in zip(pf_row, rf_row):
            assert rf["fill_price"] > pf["fill_price"]
    for p_reward, r_reward in zip(p["rewards"], r["rewards"]):
        assert r_reward < p_reward


def test_invalid_uncertainty_inputs_raise_valueerror():
    with pytest.raises(ValueError):
        _market(lambda_radius=-0.1)
    with pytest.raises(ValueError):
        _market(lambda_radius=0.1, eta_radius=0.1, uncertainty_correlation=1.5)
    m = _market()
    with pytest.raises(ValueError):
        m.set_uncertainty(eta_radius=-1.0)


# ---------------------------------------------------------------------------
# The kwarg normalizer + the EndogenousMarketEnv threading
# ---------------------------------------------------------------------------


def test_normalize_uncertainty_accepts_none_mapping_and_sequences():
    assert _normalize_uncertainty(None) is None
    assert _normalize_uncertainty({"lambda_radius": 0.1}) == (0.1, 0.0, 0.0)
    assert _normalize_uncertainty(
        {"lambda_radius": 0.1, "eta_radius": 0.2, "correlation": -0.5}
    ) == (0.1, 0.2, -0.5)
    assert _normalize_uncertainty((0.1, 0.2)) == (0.1, 0.2, 0.0)
    assert _normalize_uncertainty([0.1, 0.2, 0.3]) == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError):
        _normalize_uncertainty({"lambda_radius": 0.1, "rho": 0.0})
    with pytest.raises(ValueError):
        _normalize_uncertainty((0.1,))


@needs_pz
def test_env_threads_uncertainty_and_default_path_is_identical():
    from sharpearena.market_env import EndogenousMarketEnv

    def rollout(env):
        obs, _ = env.reset(seed=3)
        rows = []
        actions = {a: np.full(N_SYMBOLS, 0.5, dtype=np.float32) for a in env.agents}
        for _ in range(5):
            _, rewards, _, _, _ = env.step(actions)
            rows.append([rewards[a] for a in sorted(rewards)])
        return rows

    base_kwargs = dict(n_agents=N_AGENTS, n_symbols=N_SYMBOLS, n_days=40, seed=3)
    point = rollout(EndogenousMarketEnv(**base_kwargs))
    point_again = rollout(EndogenousMarketEnv(**base_kwargs, uncertainty=None))
    robust = rollout(
        EndogenousMarketEnv(
            **base_kwargs, uncertainty={"lambda_radius": 0.05, "eta_radius": 0.02}
        )
    )
    assert point == point_again
    assert len(robust) == len(point)
    # Only the first bar is comparable state-for-state; later bars diverge (the
    # worst-case lambda carries into the reference price, which can favor a holder).
    for p_reward, r_reward in zip(point[0], robust[0]):
        assert r_reward < p_reward
    assert robust != point
