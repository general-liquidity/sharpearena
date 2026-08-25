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

import numpy as np
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capital": 0.0}, {"capital": float("nan")},
        {"volume_scale": 0.0}, {"volume_scale": float("inf")},
        {"kyle_lambda": -0.1}, {"eta": -0.1},
        {"max_weight": 0.0}, {"push_weight": -0.1},
        {"push_weight": 1.1}, {"follower_gain": -1.0},
    ],
)
def test_manipulation_params_reject_invalid_physical_scales(kwargs):
    with pytest.raises(ValueError):
        ManipulationParams(**kwargs)


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


# ---------------------------------------------------------------------------
# The asymmetric round trip (positive-control family for the concave ablation)
# ---------------------------------------------------------------------------

from sharpearena.manipulation import (  # noqa: E402
    AsymmetricSchedule,
    asymmetric_round_trip_schedule,
    run_asymmetric_probe,
)


@pytest.mark.parametrize(
    "schedule",
    [
        AsymmetricSchedule.uniform(9, 1),
        AsymmetricSchedule.uniform(1, 9),
        AsymmetricSchedule(up_bars=6, down_bars=3, size_split=0.5),
        AsymmetricSchedule(up_bars=2, down_bars=7, size_split=0.9),
        AsymmetricSchedule(up_bars=1, down_bars=1, size_split=1.0),
    ],
)
def test_asymmetric_schedule_is_still_flat_to_flat(schedule):
    """Whatever its shape, the trip must start flat, peak at push_weight and end flat,
    monotone in on the pump leg and monotone out on the dump leg. A trip that does not
    return to flat is a directional bet and the attribution would measure the wrong thing."""
    p = ManipulationParams(n_days=40)
    weight = asymmetric_round_trip_schedule(p, schedule)
    span = p.start_bar + schedule.up_bars + p.hold_bars + schedule.down_bars
    path = [weight(bar) for bar in range(span + 3)]

    assert path[0] == 0.0
    assert all(w == 0.0 for w in path[: p.start_bar])
    assert all(w == 0.0 for w in path[span:])
    assert max(path) == pytest.approx(p.push_weight)
    peak = path.index(max(path))
    assert path[:peak] == sorted(path[:peak])
    assert path[peak:] == sorted(path[peak:], reverse=True)
    # The pump leg finishes exactly at the peak and the dump leg finishes exactly flat.
    assert path[p.start_bar + schedule.up_bars - 1] == pytest.approx(p.push_weight)
    assert path[span - 1] == 0.0


def test_uniform_asymmetric_schedule_reproduces_the_symmetric_ramp():
    """With equal legs and no block concentration the family collapses onto the published
    symmetric schedule bar for bar, so the two arms share one origin."""
    p = ManipulationParams(n_days=40, push_bars=5, dump_bars=5)
    sym = pump_and_dump_schedule(p)
    asym = asymmetric_round_trip_schedule(p, AsymmetricSchedule.uniform(5, 5))
    for bar in range(p.n_days):
        assert asym(bar) == pytest.approx(sym(bar))


@pytest.mark.parametrize("up, down", [(9, 1), (1, 9), (18, 2), (45, 5)])
def test_uniform_asymmetric_legs_each_execute_one_full_peak_notional(up, down):
    """Unequal ``uniform`` legs use their own 1/n increment, not one shared split.

    This is a contract-level invariant for the F5 duration sweep: the target-weight
    changes on the accumulation and liquidation legs must sum to +peak and -peak.
    """
    p = ManipulationParams(n_days=80)
    schedule = AsymmetricSchedule.uniform(up, down)
    weight = asymmetric_round_trip_schedule(p, schedule)
    a = p.start_bar
    peak = p.push_weight
    targets = [0.0] + [weight(bar) for bar in range(a, a + up + p.hold_bars + down)]
    deltas = np.diff(targets)
    assert schedule.up_split == pytest.approx(1.0 / up)
    assert schedule.down_split == pytest.approx(1.0 / down)
    assert deltas[:up].sum() == pytest.approx(peak)
    assert deltas[up + p.hold_bars :].sum() == pytest.approx(-peak)


def test_size_split_concentrates_the_turn():
    """A larger block fraction leaves more of the pump for its last bar and takes more of
    the unwind on its first bar; the leg totals are unchanged."""
    p = ManipulationParams(n_days=40)
    slow = asymmetric_round_trip_schedule(p, AsymmetricSchedule(6, 6, size_split=1.0 / 6))
    block = asymmetric_round_trip_schedule(p, AsymmetricSchedule(6, 6, size_split=0.9))
    a = p.start_bar
    penultimate_pump = a + 6 - 2
    first_dump = a + 6 + p.hold_bars
    assert block(penultimate_pump) < slow(penultimate_pump)
    assert block(first_dump) < slow(first_dump)
    assert block(a + 5) == pytest.approx(p.push_weight) == pytest.approx(slow(a + 5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"up_bars": 0, "down_bars": 5, "size_split": 0.5},
        {"up_bars": 5, "down_bars": 0, "size_split": 0.5},
        {"up_bars": 5, "down_bars": 5, "size_split": 0.0},
        {"up_bars": 5, "down_bars": 5, "size_split": 1.5},
        {"up_bars": 5, "down_bars": 5, "size_split": -0.1},
    ],
)
def test_asymmetric_schedule_rejects_incoherent_shapes(kwargs):
    with pytest.raises(ValueError):
        AsymmetricSchedule(**kwargs)


def test_asymmetric_schedule_must_fit_inside_the_episode():
    p = ManipulationParams(n_days=20)
    with pytest.raises(ValueError, match="inside the episode"):
        asymmetric_round_trip_schedule(p, AsymmetricSchedule.uniform(10, 10))


@needs_pz
def test_asymmetric_probe_is_deterministic_and_carries_its_shape():
    sc = AsymmetricSchedule.uniform(7, 1)
    a = run_asymmetric_probe(params=_P, schedule=sc, seed=0)
    b = run_asymmetric_probe(params=_P, schedule=sc, seed=0)
    c = run_asymmetric_probe(params=_P, schedule=sc, seed=1)
    d = run_asymmetric_probe(params=_P, schedule=AsymmetricSchedule.uniform(1, 7), seed=0)

    assert a.to_dict() == b.to_dict()
    assert a.impact_pnl != c.impact_pnl
    assert a.impact_pnl != d.impact_pnl
    blob = a.to_dict()
    assert blob["up_bars"] == 7 and blob["down_bars"] == 1
    assert blob["duration_ratio"] == pytest.approx(7.0)
    assert DISCLAIMER in blob["disclaimer"]


@needs_pz
def test_asymmetric_probe_with_equal_uniform_legs_matches_the_symmetric_probe():
    """The additive family must not have perturbed the published path: the uniform
    equal-leg asymmetric probe and the symmetric probe are the same computation."""
    sym = run_manipulation_probe(params=_P, seed=0)
    asym = run_asymmetric_probe(
        params=_P, schedule=AsymmetricSchedule.uniform(_P.push_bars, _P.dump_bars), seed=0
    )
    assert asym.impact_pnl == sym.impact_pnl
    assert asym.live_pnl == sym.live_pnl


@needs_pz
def test_zero_impact_reference_leaves_nothing_to_attribute_on_the_asymmetric_trip():
    result = run_asymmetric_probe(
        params=replace(_P, kyle_lambda=0.0, eta=0.0),
        schedule=AsymmetricSchedule.uniform(6, 2),
        seed=0,
    )
    assert result.impact_pnl == 0.0


# ---------------------------------------------------------------------------
# Extended-sweep schedule variants: long legs, interleaved holds, the short side
# ---------------------------------------------------------------------------


def _round_trip_path(p: ManipulationParams, schedule: AsymmetricSchedule, side: int):
    weight = asymmetric_round_trip_schedule(p, schedule, side)
    span = p.start_bar + schedule.up_bars + p.hold_bars + schedule.down_bars
    return span, [weight(bar) for bar in range(span + 3)]


@pytest.mark.parametrize(
    "hold_bars, schedule",
    [
        (1, AsymmetricSchedule.uniform(45, 5)),   # the longest extended leg pair
        (12, AsymmetricSchedule.uniform(9, 1)),   # the longest extended hold
        (0, AsymmetricSchedule.uniform(9, 1)),    # no hold at all
        (6, AsymmetricSchedule(27, 3, size_split=0.9)),
    ],
)
@pytest.mark.parametrize("side", [1, -1])
def test_extended_schedule_variants_are_flat_to_flat(hold_bars, schedule, side):
    """Long legs, long holds and the mirrored short trip all keep the round-trip
    invariants: flat before the start bar, a single extremum of magnitude push_weight
    held for exactly ``hold_bars`` bars, monotone in and monotone out, flat after."""
    p = ManipulationParams(n_days=80, hold_bars=hold_bars)
    span, path = _round_trip_path(p, schedule, side)
    signed = [side * w for w in path]

    assert all(w == 0.0 for w in path[: p.start_bar])
    assert all(w == 0.0 for w in path[span:])
    assert max(signed) == pytest.approx(p.push_weight)
    assert min(signed) == 0.0
    first_peak = signed.index(max(signed))
    last_peak = len(signed) - 1 - signed[::-1].index(max(signed))
    assert last_peak - first_peak == hold_bars
    assert signed[: first_peak + 1] == sorted(signed[: first_peak + 1])
    assert signed[last_peak:] == sorted(signed[last_peak:], reverse=True)
    assert first_peak == p.start_bar + schedule.up_bars - 1
    assert path[span - 1] == 0.0


@pytest.mark.parametrize(
    "schedule",
    [
        AsymmetricSchedule.uniform(9, 1),
        AsymmetricSchedule.uniform(1, 9),
        AsymmetricSchedule.uniform(45, 5),
        AsymmetricSchedule(8, 2, size_split=0.5),
    ],
)
@pytest.mark.parametrize("hold_bars", [0, 1, 12])
def test_short_side_is_the_exact_sign_mirror_of_the_long_side(schedule, hold_bars):
    """The short round trip is the long one with the sign flipped bar for bar, and the
    default side is the long side. Any asymmetry the probe then measures between the two
    sides is a property of the market, not of the schedule."""
    p = ManipulationParams(n_days=80, hold_bars=hold_bars)
    long_w = asymmetric_round_trip_schedule(p, schedule)
    short_w = asymmetric_round_trip_schedule(p, schedule, -1)
    explicit_long = asymmetric_round_trip_schedule(p, schedule, 1)
    for bar in range(p.n_days):
        assert short_w(bar) == -long_w(bar)
        assert explicit_long(bar) == long_w(bar)
    assert min(short_w(bar) for bar in range(p.n_days)) == pytest.approx(-p.push_weight)


@pytest.mark.parametrize("side", [0, 2, -2])
def test_schedule_and_probe_reject_a_side_that_is_not_a_sign(side):
    p = ManipulationParams(n_days=40)
    with pytest.raises(ValueError, match="side must be"):
        asymmetric_round_trip_schedule(p, AsymmetricSchedule.uniform(4, 2), side)
    with pytest.raises(ValueError, match="side must be"):
        run_asymmetric_probe(params=p, schedule=AsymmetricSchedule.uniform(4, 2), side=side)


def test_extended_schedule_variants_must_still_fit_inside_the_episode():
    p = ManipulationParams(n_days=80, hold_bars=12)
    with pytest.raises(ValueError, match="inside the episode"):
        asymmetric_round_trip_schedule(p, AsymmetricSchedule.uniform(60, 5), -1)


@needs_pz
def test_short_side_probe_is_deterministic_and_carries_its_side():
    sc = AsymmetricSchedule.uniform(7, 1)
    a = run_asymmetric_probe(params=_P, schedule=sc, seed=0, side=-1)
    b = run_asymmetric_probe(params=_P, schedule=sc, seed=0, side=-1)
    long_side = run_asymmetric_probe(params=_P, schedule=sc, seed=0)

    assert a.to_dict() == b.to_dict()
    assert a.side == -1 and a.to_dict()["side"] == -1
    assert long_side.side == 1 and long_side.to_dict()["side"] == 1
    # The market is not sign-symmetric bit for bit (multiplicative compounding), so the
    # two sides are distinct runs; equality here would mean the side was ignored.
    assert a.impact_pnl != long_side.impact_pnl
    assert a.peak_price_move > 0.0


@needs_pz
def test_zero_impact_reference_leaves_nothing_to_attribute_on_the_short_side():
    result = run_asymmetric_probe(
        params=replace(_P, kyle_lambda=0.0, eta=0.0),
        schedule=AsymmetricSchedule.uniform(6, 2),
        seed=0,
        side=-1,
    )
    assert result.impact_pnl == 0.0
    assert result.live_pnl == result.reference_pnl


@needs_pz
def test_long_hold_and_long_legs_run_and_are_deterministic():
    p = ManipulationParams(n_days=80, hold_bars=12, eta=0.0, follower_gain=0.0)
    sc = AsymmetricSchedule.uniform(45, 5)
    a = run_asymmetric_probe(params=p, schedule=sc, seed=3)
    b = run_asymmetric_probe(params=p, schedule=sc, seed=3)
    assert a.to_dict() == b.to_dict()
    assert a.up_bars == 45 and a.down_bars == 5
