"""The price-manipulation red-team probe on the endogenous shared-book market.

These tests do not assert that manipulation pays. Whether it pays is an empirical fact
about the impact specification and it is allowed to change when that specification does.
What must hold is that the probe answers the question it exists to answer: it attributes
P&L to having moved the price rather than to having been long a rally, it reports a
*boundary* rather than an unbounded printing press, and it reproduces exactly from a seed.

Behaviour tests skip when ``pettingzoo`` is absent, matching ``test_market.py``: the
endogenous market is an optional-dependency surface. The module-contract tests, including
the one that checks the diagnostic framing is stated rather than implied, always run.
"""

import importlib.util
from dataclasses import replace

import pytest

from sharpearena import manipulation
from sharpearena.manipulation import (
    DISCLAIMER,
    ManipulationParams,
    impact_boundary_sweep,
    pump_and_dump_schedule,
    run_manipulation_probe,
    size_response,
)

_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_pz = pytest.mark.skipif(not _HAS_PETTINGZOO, reason="pettingzoo not installed")

# Small and cheap: the probe runs each configuration twice (live plus zero-impact
# reference) and the sweeps run it once per grid point per seed.
_P = ManipulationParams(n_days=40, n_followers=2, push_bars=4, hold_bars=1, dump_bars=4)
_SEEDS = (0, 1)

# A dense, hard-chasing crowd and a long hold, which is the regime where a pump has
# somebody to unwind into. The shipped default has no such crowd and the round trip loses
# there, so this configuration is what keeps the probe's positive branch under test.
_CROWD = ManipulationParams(n_days=60, n_followers=16, hold_bars=6, follower_gain=120.0)


# ---------------------------------------------------------------------------
# What this module is for
# ---------------------------------------------------------------------------


def test_module_states_it_is_a_diagnostic_not_a_strategy():
    """The framing has to be in the artefact, not in a reviewer's memory.

    This module generates behaviour that looks like a profitable trading idea, and it will
    be read that way unless it says otherwise in the places a reader actually lands: the
    module docstring and every serialised result.
    """
    doc = manipulation.__doc__ or ""
    assert "not a strategy" in doc
    assert "not trading advice" in doc
    assert "red-team" in doc.lower()
    assert "not a strategy" in DISCLAIMER and "not trading advice" in DISCLAIMER


def test_schedule_is_a_round_trip_that_starts_and_ends_flat():
    """A push that is never unwound is a directional bet, not manipulation. The schedule
    must return to flat, or the P&L attribution measures the wrong thing."""
    weight = pump_and_dump_schedule(_P)
    span = _P.start_bar + _P.push_bars + _P.hold_bars + _P.dump_bars
    path = [weight(bar) for bar in range(span + 3)]

    assert path[0] == 0.0
    assert path[-1] == 0.0
    assert max(path) == pytest.approx(_P.push_weight)
    peak = path.index(max(path))
    assert path[:peak] == sorted(path[:peak])          # ramping in
    assert path[peak:] == sorted(path[peak:], reverse=True)  # unwinding


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_followers": -1},
        {"push_bars": 0},
        {"dump_bars": 0},
        {"hold_bars": -1},
        {"start_bar": -1},
        {"n_days": 5},
    ],
)
def test_params_reject_incoherent_probes(overrides):
    with pytest.raises(ValueError):
        replace(_P, **overrides)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@needs_pz
def test_zero_impact_market_leaves_nothing_to_attribute():
    """With ``kyle_lambda = eta = 0`` flow cannot move price, so the live run and its own
    zero-impact reference are the same run and the attributable P&L is exactly zero.

    This is the attribution's null: if it were nonzero, ``impact_pnl`` would be picking up
    exogenous drift and every boundary drawn on it would be drawn on noise.
    """
    result = run_manipulation_probe(
        params=replace(_P, kyle_lambda=0.0, eta=0.0),
        seed=0,
    )
    assert result.impact_pnl == 0.0
    assert result.live_pnl == result.reference_pnl
    assert result.profitable is False


@needs_pz
def test_pushing_moves_the_cleared_price():
    """The probe is only meaningful if the push actually pushes. A market where a large
    round trip leaves no mark is not testing the impact model, it is testing nothing."""
    hard = run_manipulation_probe(
        params=replace(_P, kyle_lambda=0.4), seed=0
    )
    none = run_manipulation_probe(
        params=replace(_P, kyle_lambda=0.0, eta=0.0),
        seed=0,
    )
    assert hard.peak_price_move > none.peak_price_move


# ---------------------------------------------------------------------------
# The boundary, which is the point
# ---------------------------------------------------------------------------


@needs_pz
def test_boundary_sweep_reports_a_boundary_not_unbounded_profit():
    """Sweeping permanent impact must find a region where the pump stops paying.

    ``profitable_everywhere`` is the failure this test exists to catch: it would mean the
    simulator rewards pushing the price no matter how expensive pushing is, which is a
    statement about the impact specification being wrong rather than about the agent being
    clever. Where there is a profitable region, the crossing must be located.
    """
    report = impact_boundary_sweep(
        params=_P, axis="kyle_lambda", values=(0.0, 0.1, 0.4, 1.0), seeds=_SEEDS
    )

    assert not report.profitable_everywhere
    assert len(report.impact_pnl) == len(report.values) == 4
    if any(report.profitable):
        assert report.boundary is not None
        assert min(report.values) <= report.boundary <= max(report.values)


@needs_pz
def test_probe_can_report_profitable_manipulation_when_a_crowd_chases():
    """The probe's non-vacuity check: it must be able to say "yes" as well as "no".

    A diagnostic that reports "not profitable" under every configuration is
    indistinguishable from one that is broken, and the reassuring answer it gives about the
    shipped defaults would be worth nothing. With a dense crowd chasing the move there is
    somebody to unwind into, and the probe has to find that.
    """
    report = impact_boundary_sweep(
        params=_CROWD, axis="follower_gain", values=(0.0, 120.0), seeds=(0, 1, 2)
    )
    assert report.profitable == (False, True)
    assert not report.profitable_everywhere
    assert not report.unprofitable_everywhere


@needs_pz
def test_manipulation_stops_paying_as_impact_gets_expensive():
    """In the regime where the pump does pay, it must still stop paying once pushing the
    price is expensive enough. That crossing is the module's actual output: a statement
    about how much permanent impact the simulator needs before it stops rewarding a pump.
    """
    report = impact_boundary_sweep(
        params=_CROWD, axis="kyle_lambda", values=(0.1, 0.4), seeds=(0, 1, 2)
    )
    assert report.profitable == (True, False)
    assert report.boundary is not None
    assert 0.1 < report.boundary < 0.4


@needs_pz
def test_size_response_turns_over_where_manipulation_pays():
    """Even inside the profitable regime the payoff must peak and fall away with size.

    A profit that keeps rising with size would mean an agent large enough could print
    money, which is a bug in the impact model rather than a finding about agents.
    """
    response = size_response(
        params=_CROWD, push_weights=(0.25, 0.5, 1.0), seeds=(0, 1, 2)
    )
    assert max(response.impact_pnl) > 0.0
    assert response.bounded
    assert response.peak_push_weight < max(response.push_weights)


@needs_pz
def test_boundary_sweep_accepts_the_other_impact_axes():
    """Temporary impact and crowd-chasing are the other two knobs that decide whether a
    pump pays, so the boundary has to be drawable along them too."""
    for axis, values in (("eta", (0.0, 0.2, 0.8)), ("follower_gain", (0.0, 30.0, 90.0))):
        report = impact_boundary_sweep(
            params=_P, axis=axis, values=values, seeds=(0,)
        )
        assert report.axis == axis
        assert len(report.impact_pnl) == len(values)


def test_boundary_sweep_rejects_an_unknown_axis():
    with pytest.raises(ValueError, match="unknown axis"):
        impact_boundary_sweep(params=_P, axis="not_a_parameter", seeds=(0,))


@needs_pz
def test_size_response_is_bounded():
    """Profit must not still be rising at the largest size tested.

    An impact model that pays more for every extra unit of size, without limit, lets a
    large enough agent print money, and any benchmark run on it is scoring the bug. A
    bounded response is the healthy result, and it is also the simulator's implicit
    statement about how much size a book can absorb.
    """
    response = size_response(params=_P, push_weights=(0.1, 0.4, 0.8), seeds=_SEEDS)

    assert response.bounded
    assert not response.monotone_increasing
    assert response.peak_push_weight in response.push_weights
    assert response.peak_impact_pnl == max(response.impact_pnl)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@needs_pz
def test_probe_is_deterministic_under_a_fixed_seed():
    a = run_manipulation_probe(params=_P, seed=0)
    b = run_manipulation_probe(params=_P, seed=0)
    c = run_manipulation_probe(params=_P, seed=1)

    assert a.to_dict() == b.to_dict()
    assert a.impact_pnl != c.impact_pnl


@needs_pz
def test_sweeps_are_deterministic_under_fixed_seeds():
    kwargs = dict(params=_P, values=(0.0, 0.4), seeds=_SEEDS)
    assert impact_boundary_sweep(**kwargs).to_dict() == impact_boundary_sweep(**kwargs).to_dict()

    sized = dict(params=_P, push_weights=(0.2, 0.6), seeds=_SEEDS)
    assert size_response(**sized).to_dict() == size_response(**sized).to_dict()


@needs_pz
def test_serialised_results_carry_the_disclaimer():
    """Every artefact that can leave this process says what it is. A JSON blob of
    profitable manipulation P&L with no framing attached is how this gets misread."""
    assert run_manipulation_probe(params=_P, seed=0).to_dict()["disclaimer"] == DISCLAIMER
    assert (
        impact_boundary_sweep(params=_P, values=(0.0, 0.4), seeds=(0,)).to_dict()[
            "disclaimer"
        ]
        == DISCLAIMER
    )
    assert (
        size_response(params=_P, push_weights=(0.2,), seeds=(0,)).to_dict()["disclaimer"]
        == DISCLAIMER
    )
