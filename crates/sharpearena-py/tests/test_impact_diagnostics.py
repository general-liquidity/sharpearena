"""Rank-neutral impact diagnostics: the paired impact-misspecification gap (A4) and the
meta-order impact-shape probe (A8).

The gap report runs one policy at the point estimate and against a declared elliptic
uncertainty set on identical seeds, and must prove the two arms traded on the same
exogenous path. The shape probe measures what the kernel's compounding permanent impact
does to a meta-order.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from sharpearena import impact_diagnostics as diag
from sharpearena.impact_diagnostics import (
    CONSTANT_TRACK,
    INSUFFICIENT_SEEDS,
    GapInterval,
    ImpactDiagnosticError,
    MarketSettings,
    Unavailable,
    UnpairedArmsError,
    check_arm_pairing,
    impact_misspecification_gap,
    meta_order_impact_shape,
    run_impact_arm,
    t_critical,
    track_sharpe,
)

SETTINGS = MarketSettings(n_symbols=2, n_days=48)
ONE_SYMBOL = MarketSettings(n_symbols=1, n_days=48)
GENERAL_SET = {"lambda_radius": 0.05, "eta_radius": 0.02, "correlation": -0.5}


def _constant(weight):
    def make():
        def policy(observation, bar):
            return [weight] * len(observation["symbols"])

        return policy

    return make


def _alternating():
    # Swings between fully long and fully short every bar: heavy, open-loop trading.
    def policy(observation, bar):
        return [1.0 if bar % 2 == 0 else -1.0] * len(observation["symbols"])

    return policy


def _chaser():
    # Reads its own observation (the last two closes), so it is closed loop.
    def policy(observation, bar):
        out = []
        for symbol in observation["symbols"]:
            closes = symbol["close_history"]
            out.append(0.5 if closes[-1] >= closes[-2] else -0.5)
        return out

    return policy


# ---------------------------------------------------------------------------
# Degenerate inputs are exact zeros
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_policy", [_alternating, _chaser], ids=["open", "closed"])
def test_zero_radius_set_reports_exactly_zero_gap(make_policy):
    report = impact_misspecification_gap(
        make_policy, [0, 1, 2], (0.0, 0.0, 0.0), settings=SETTINGS
    )
    assert report.n_seeds == 3
    for row in report.per_seed:
        assert row.identical_arms
        assert row.point_return != 0.0  # it did trade
        assert row.return_gap == 0.0
        assert row.sharpe_gap == 0.0
        assert row.point_sharpe == row.robust_sharpe
        assert isinstance(row.point_sharpe, float)
    assert report.mean_return_gap == 0.0
    assert report.mean_sharpe_gap == 0.0
    interval = report.return_gap_interval
    assert isinstance(interval, GapInterval)
    assert (interval.mean, interval.sd, interval.lo, interval.hi) == (0.0, 0.0, 0.0, 0.0)


def test_never_trading_policy_reports_exactly_zero_gap():
    report = impact_misspecification_gap(
        _constant(0.0), [0, 1, 2], GENERAL_SET, settings=SETTINGS
    )
    for row in report.per_seed:
        assert row.identical_arms
        assert row.point_return == 0.0
        assert row.robust_return == 0.0
        assert row.return_gap == 0.0
        assert row.sharpe_gap == 0.0
        assert row.point_sharpe == Unavailable(CONSTANT_TRACK, "every bar returned 0.0")
        assert row.robust_sharpe == row.point_sharpe
    assert report.mean_return_gap == 0.0
    assert report.mean_sharpe_gap == 0.0
    assert isinstance(report.mean_point_sharpe, Unavailable)


def test_single_seed_says_it_has_no_dispersion_estimate():
    report = impact_misspecification_gap(_alternating, [5], GENERAL_SET, settings=SETTINGS)
    assert report.n_seeds == 1
    for interval in (report.return_gap_interval, report.sharpe_gap_interval):
        assert isinstance(interval, Unavailable)
        assert interval.reason == INSUFFICIENT_SEEDS
    assert report.mean_return_gap == report.per_seed[0].return_gap


# ---------------------------------------------------------------------------
# The sign: guaranteed for an eta-only set, not for a lambda set
# ---------------------------------------------------------------------------


def test_heavy_trading_under_eta_only_set_loses_return_on_every_seed():
    # With lambda_radius == 0 the resolved lambda is bitwise the point estimate, so both
    # arms clear at identical mids with identical sizes, and only eta (the fill cost)
    # rises. The policy ignores cash and average price, so the gap cannot be negative,
    # and heavy trading makes it strictly positive.
    eta_only = {"lambda_radius": 0.0, "eta_radius": 0.05}
    seeds = [0, 1, 2, 3]
    for seed in seeds:
        point = run_impact_arm(_alternating, seed, None, SETTINGS)
        robust = run_impact_arm(_alternating, seed, eta_only, SETTINGS)
        assert robust.applied_lambda == point.applied_lambda
        assert robust.cleared_mids == point.cleared_mids
        assert robust.net_flow == point.net_flow
    report = impact_misspecification_gap(_alternating, seeds, eta_only, settings=SETTINGS)
    for row in report.per_seed:
        assert not row.identical_arms
        assert row.robust_return < row.point_return
        assert row.return_gap > 0.0
    assert isinstance(report.return_gap_interval, GapInterval)
    assert report.return_gap_interval.n == len(seeds)
    assert report.return_gap_interval.lo > 0.0


def test_lambda_uncertainty_can_favor_a_holder_and_the_gap_is_not_clamped():
    # A buy of q shares held to the end has d NAV / d lambda = q^2 / V * (exo_T - exo_0):
    # the worst-case lambda also marks the holder's own position up. On a rising path the
    # worst case is therefore better for the holder and the gap must come out negative.
    lambda_only = {"lambda_radius": 0.05, "eta_radius": 0.0}
    seeds = list(range(8))
    report = impact_misspecification_gap(
        _constant(1.0), seeds, lambda_only, settings=ONE_SYMBOL
    )
    signs = set()
    for seed, row in zip(seeds, report.per_seed):
        trace = run_impact_arm(_constant(1.0), seed, None, ONE_SYMBOL)
        move = trace.exogenous_mids[-1][0] - trace.exogenous_mids[0][0]
        assert move != 0.0
        assert math.copysign(1.0, row.return_gap) == -math.copysign(1.0, move)
        signs.add(row.return_gap > 0.0)
    assert signs == {True, False}, "the seed set must contain both path directions"


# ---------------------------------------------------------------------------
# Pairing is a checked property
# ---------------------------------------------------------------------------


def test_matched_arms_pass_the_pairing_check():
    point = run_impact_arm(_chaser, 4, None, SETTINGS)
    robust = run_impact_arm(_chaser, 4, GENERAL_SET, SETTINGS)
    assert point.cleared_mids != robust.cleared_mids  # the arms really diverged
    check_arm_pairing(point, robust)


def test_pairing_refuses_arms_on_different_exogenous_paths():
    point = run_impact_arm(_alternating, 1, None, SETTINGS)
    other = run_impact_arm(_alternating, 2, GENERAL_SET, SETTINGS)
    relabelled = dataclasses.replace(other, seed=1)
    with pytest.raises(UnpairedArmsError) as excinfo:
        check_arm_pairing(point, relabelled)
    assert excinfo.value.check in {"burn-in closes", "exogenous path"}


def test_pairing_refuses_a_replay_that_differs_by_one_step():
    point = run_impact_arm(_alternating, 1, None, SETTINGS)
    robust = run_impact_arm(_alternating, 1, GENERAL_SET, SETTINGS)
    rows = [list(row) for row in robust.exogenous_mids]
    rows[-1][0] = math.nextafter(rows[-1][0], math.inf)
    tampered = dataclasses.replace(robust, exogenous_mids=tuple(tuple(r) for r in rows))
    with pytest.raises(UnpairedArmsError) as excinfo:
        check_arm_pairing(point, tampered)
    assert excinfo.value.check == "exogenous path"


@pytest.mark.parametrize("arm", ["point", "robust"])
def test_pairing_refuses_a_traded_tape_off_its_replay(arm):
    traces = {
        "point": run_impact_arm(_alternating, 1, None, SETTINGS),
        "robust": run_impact_arm(_alternating, 1, GENERAL_SET, SETTINGS),
    }
    rows = [list(row) for row in traces[arm].cleared_mids]
    rows[7][1] = math.nextafter(rows[7][1], -math.inf)
    traces[arm] = dataclasses.replace(traces[arm], cleared_mids=tuple(tuple(r) for r in rows))
    with pytest.raises(UnpairedArmsError) as excinfo:
        check_arm_pairing(traces["point"], traces["robust"])
    assert excinfo.value.check == f"{arm} traded tape"
    assert "bar 7" in str(excinfo.value)


def test_pairing_refuses_swapped_arm_roles():
    point = run_impact_arm(_alternating, 1, None, SETTINGS)
    robust = run_impact_arm(_alternating, 1, GENERAL_SET, SETTINGS)
    with pytest.raises(UnpairedArmsError) as excinfo:
        check_arm_pairing(robust, point)
    assert excinfo.value.check == "arm roles"


def test_report_refuses_when_an_arm_ran_on_another_path(monkeypatch):
    real = diag.run_impact_arm

    def shifted(make_policy, seed, uncertainty=None, settings=None):
        if uncertainty is None:
            return real(make_policy, seed, uncertainty, settings)
        trace = real(make_policy, seed + 100, uncertainty, settings)
        return dataclasses.replace(trace, seed=seed)

    monkeypatch.setattr(diag, "run_impact_arm", shifted)
    with pytest.raises(UnpairedArmsError):
        impact_misspecification_gap(_alternating, [0, 1], GENERAL_SET, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Inputs and statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seeds,uncertainty,match",
    [
        ([0], None, "uncertainty is required"),
        ([], GENERAL_SET, "seeds is empty"),
        ([0, 1, 0], GENERAL_SET, "seeds must be distinct"),
    ],
)
def test_report_rejects_unmeasurable_inputs(seeds, uncertainty, match):
    with pytest.raises(ImpactDiagnosticError, match=match):
        impact_misspecification_gap(_alternating, seeds, uncertainty, settings=SETTINGS)


def test_report_rejects_a_policy_with_the_wrong_width():
    with pytest.raises(ImpactDiagnosticError, match="returned 3 weights at bar 0, expected 2"):
        impact_misspecification_gap(
            lambda: (lambda observation, bar: [0.1, 0.1, 0.1]),
            [0],
            GENERAL_SET,
            settings=SETTINGS,
        )


def test_report_rejects_nonpositive_volume_scale():
    with pytest.raises(ImpactDiagnosticError, match="volume_scale"):
        impact_misspecification_gap(
            _alternating, [0], GENERAL_SET, settings=MarketSettings(volume_scale=0.0)
        )


def test_track_sharpe_types_degenerate_tracks():
    assert track_sharpe([0.01]).reason == "too_few_bars"
    assert track_sharpe([0.002, 0.002, 0.002]).reason == CONSTANT_TRACK
    assert track_sharpe([0.01, -0.01, 0.03]) == pytest.approx(0.01 / 0.02)


@pytest.mark.parametrize(
    "df,expected",
    [
        (1, 12.706204736432095),
        (2, 4.302652729696142),
        (3, 3.182446305284263),
        (5, 2.570581835636314),
        (15, 2.131449545559323),
        (23, 2.0686576104190406),
        (100, 1.9839715184496334),
        (1000, 1.9623390808264074),
    ],
)
def test_t_critical_matches_reference_quantiles(df, expected):
    # Reference values are scipy.stats.t.ppf(0.975, df); SciPy is not a dependency.
    assert t_critical(df) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# A8: the meta-order impact shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [1.0, 0.5])
def test_meta_order_impact_shape_pins_permanent_impact_shape(exponent):
    # Permanent increments compound and never decay, so impact grows about linearly in
    # executed quantity at every exponent, never relaxes after execution, and a fixed
    # total spread over D bars grows as D ** (1 - exponent). None of this is the
    # square-root law (exponent 0.5 in quantity, no duration dependence, decay).
    shape = meta_order_impact_shape(impact_exponent=exponent)
    assert shape.execution_exponent == pytest.approx(1.0, abs=0.05)
    assert shape.max_relaxation_deviation <= 1e-12
    assert len(shape.relaxation_ratios) == 21
    assert shape.duration_exponent == pytest.approx(1.0 - exponent, abs=0.05)
    assert all(later >= earlier for earlier, later in zip(shape.impact, shape.impact[1:]))


def test_meta_order_probe_refuses_an_unmeasurable_setup():
    with pytest.raises(ImpactDiagnosticError, match="kyle_lambda"):
        meta_order_impact_shape(kyle_lambda=0.0)
    with pytest.raises(ImpactDiagnosticError, match="durations"):
        meta_order_impact_shape(durations=(10, 10))
