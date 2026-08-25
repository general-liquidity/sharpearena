"""Tests for the stylized-facts realism diagnostic (sharpearena.realism).

Two angles: (1) real generated panels from the native scenario generator must show the
Hard/Extreme tiers emitting fatter-tailed tapes than the Calm tier, and (2) a constructed
GARCH panel with known volatility clustering must score higher clustering/intermittency
than an i.i.d. Gaussian panel, validating the clustering metrics on data where the effect
genuinely exists.
"""

import json

import numpy as np
import pytest

from sharpearena.realism import (
    stylized_facts,
    certify_realism,
    RealismReport,
    DEFAULT_THRESHOLDS,
)


# -- constructed panels with known properties -------------------------------


def _iid_panel(t=1000, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, n))


def _garch_panel(t=1000, n=4, seed=1, omega=0.05, alpha=0.1, beta=0.85):
    """A GARCH(1,1) return panel: persistent conditional variance -> volatility clustering
    and fat tails, the canonical process the stylized-facts diagnostic should light up on."""
    rng = np.random.default_rng(seed)
    out = np.zeros((t, n))
    for c in range(n):
        var = 1.0
        for i in range(t):
            r = np.sqrt(var) * float(rng.standard_normal())
            out[i, c] = r
            var = omega + alpha * r * r + beta * var
    return out


# -- basic shape / API ------------------------------------------------------


def test_stylized_facts_keys_and_finite():
    facts = stylized_facts(_garch_panel(), kind="return")
    assert set(facts) == {
        "excess_kurtosis",
        "abs_return_autocorr",
        "zumbach_asymmetry",
        "gain_loss_skew",
        "aggregational_gaussianity",
        "fano_factor",
    }
    assert all(np.isfinite(v) for v in facts.values())


def test_stylized_facts_accepts_1d_and_price_kind():
    # A rising price series (1-D) is coerced to a single-column log-return panel.
    prices = np.cumprod(1.0 + 0.01 * np.sin(np.arange(200) / 5.0))
    facts = stylized_facts(prices, kind="price")
    assert np.isfinite(facts["excess_kurtosis"])


def test_price_kind_rejects_nonpositive():
    with pytest.raises(ValueError):
        stylized_facts(np.array([1.0, -2.0, 3.0]), kind="price")


# -- clustering / tails on constructed panels -------------------------------


def test_garch_fatter_tailed_than_iid():
    iid = stylized_facts(_iid_panel(), kind="return")
    garch = stylized_facts(_garch_panel(), kind="return")
    assert garch["excess_kurtosis"] > iid["excess_kurtosis"]


def test_garch_shows_stronger_vol_clustering():
    iid = stylized_facts(_iid_panel(), kind="return")
    garch = stylized_facts(_garch_panel(), kind="return")
    # |return| autocorrelation (clustering) and exceedance Fano (intermittency) both rise.
    assert garch["abs_return_autocorr"] > iid["abs_return_autocorr"]
    assert garch["fano_factor"] > iid["fano_factor"]


def test_iid_baselines_are_neutral():
    facts = stylized_facts(_iid_panel(t=2000), kind="return")
    # A memoryless Gaussian tape: ~mesokurtic, ~no clustering, and a finite-panel-
    # calibrated conditional-IID Fano ratio near one.
    assert abs(facts["excess_kurtosis"]) < 0.5
    assert abs(facts["abs_return_autocorr"]) < 0.05
    assert abs(facts["fano_factor"] - 1.0) < 0.5


def test_aggregational_gaussianity_uses_distance_to_gaussian_not_signed_kurtosis():
    # The prior signed difference called this a failure (-1 -> -0.3).  A platykurtic
    # panel moving toward zero excess kurtosis is aggregational Gaussianity by definition.
    # Repeating each value makes the sample moments deterministic enough for this unit test.
    base = np.tile(np.array([-1.0, -1.0, 1.0, 1.0, -0.2, 0.2]), 40).reshape(-1, 1)
    facts = stylized_facts(base, kind="return", agg_horizon=4)
    # The construction is intentionally only a regression against the old signed formula:
    # an equivalent direct calculation must agree with the public diagnostic.
    r = base[:, 0]
    k1 = ((r - r.mean()) ** 4).mean() / r.std() ** 4 - 3.0
    agg = r[: len(r) // 4 * 4].reshape(-1, 4).sum(axis=1)
    k4 = ((agg - agg.mean()) ** 4).mean() / agg.std() ** 4 - 3.0
    assert facts["aggregational_gaussianity"] == pytest.approx(abs(k1) - abs(k4))


def test_fano_is_neutral_under_its_conditional_finite_panel_null():
    # Exactly one event per ten-bar block has raw Fano 0, hence the calibrated ratio
    # must also be 0; a clustered placement must be above the IID conditional baseline.
    evenly_spaced = np.tile(np.array([10.0] + [0.0] * 9), 12).reshape(-1, 1)
    clustered = np.array([10.0] * 6 + [0.0] * 114).reshape(-1, 1)
    even = stylized_facts(evenly_spaced, kind="return", fano_z=1.0)["fano_factor"]
    bunch = stylized_facts(clustered, kind="return", fano_z=1.0)["fano_factor"]
    assert even == pytest.approx(0.0)
    assert bunch > 1.0


def test_abs_return_acf_gate_has_a_reproducible_iid_false_positive_rate():
    # The gate uses a fixed 20,000-panel Gaussian null for exactly this estimator rather
    # than the invalid raw threshold ``ACF > 0``.  A fresh IID batch should reject near
    # the advertised one-sided 5% rate, not the old roughly-quarter rate.
    rng = np.random.default_rng(20260825)
    rejected = 0
    n = 200
    for _ in range(n):
        report = certify_realism(
            rng.standard_normal((120, 4)), kind="return", thresholds={
                "excess_kurtosis": (None, None),
                "aggregational_gaussianity": (None, None),
            }
        )
        rejected += report.checks["abs_return_autocorr"]
    rate = rejected / n
    assert 0.01 <= rate <= 0.12


# -- certify_realism --------------------------------------------------------


def test_certify_returns_report():
    report = certify_realism(_garch_panel(), kind="return")
    assert isinstance(report, RealismReport)
    assert set(report.checks) == set(DEFAULT_THRESHOLDS)
    assert isinstance(report.passed, bool)


def test_certify_custom_threshold_can_fail():
    # An unreachable kurtosis floor forces the fat-tails check to fail.
    report = certify_realism(
        _iid_panel(), kind="return", thresholds={"excess_kurtosis": (1000.0, None)}
    )
    assert report.checks["excess_kurtosis"] is False
    assert report.passed is False


# -- native scenario generator panels ---------------------------------------

TradingEnv = pytest.importorskip("sharpearena.sharpearena_py").TradingEnv


def _scenario_panel(mode, seed, n_symbols=4, n_days=120):
    """Reconstruct the full close panel for a difficulty tier by stepping the leak-free env
    and reading each bar's public close (a flat HOLD decision, so no trading feedback)."""
    env = TradingEnv(
        n_symbols=n_symbols, n_days=n_days, seed=seed, distribution_mode=mode
    )
    obs = json.loads(env.reset())
    symbols = [s["symbol"] for s in obs["symbols"]]
    hist = {s["symbol"]: [s["close_history"][-1]] for s in obs["symbols"]}
    hold = json.dumps({"orders": [], "reasoning": ""})
    while True:
        obs_json, _reward, done, _info = env.step(hold)
        obs = json.loads(obs_json)
        for s in obs["symbols"]:
            hist[s["symbol"]].append(s["close_history"][-1])
        if done:
            break
    return np.array([hist[s] for s in symbols], dtype=np.float64).T


def _mean_kurtosis(mode, seeds):
    return float(
        np.mean(
            [stylized_facts(_scenario_panel(mode, s))["excess_kurtosis"] for s in seeds]
        )
    )


def test_hard_and_extreme_are_fatter_tailed_than_calm():
    seeds = range(6)
    calm = _mean_kurtosis("calm", seeds)
    hard = _mean_kurtosis("hard", seeds)
    extreme = _mean_kurtosis("extreme", seeds)
    # The Hard/Extreme tiers amplify volatility and inject jumps, so their tapes are
    # markedly leptokurtic while the Calm synthetic panel is not.
    assert hard > calm
    assert extreme > calm
    assert hard > 1.0
    assert extreme > 1.0


def test_hard_tier_kurtosis_decays_on_aggregation():
    # Aggregational Gaussianity: the jump-driven excess kurtosis thins toward Gaussian when
    # returns are summed over longer horizons, so the fact is strongly positive for Hard.
    facts = stylized_facts(_scenario_panel("hard", 7))
    assert facts["aggregational_gaussianity"] > 0.0
