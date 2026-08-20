"""Tests for the two per-bar risk-pricing reward schemes.

``risk_aware`` and ``time_inhomogeneous_vol_aversion`` are pure functions of
``state['returns']``, so they need neither the native binding nor ``verifiers``. The
registry import is guarded anyway, because ``rewards`` pulls in ``verifiers_env``.

These check three things the package promises: the schemes are registered and listed,
they return finite values bounded in ``[-1, 1]``, and they reproduce byte-identically.
The time-inhomogeneous scheme additionally has to demonstrate that its risk treatment
actually MOVES across the episode, which is the whole reason it exists.
"""

import pytest

try:
    import numpy as np

    from sharpearena.rewards import (
        REWARD_SCHEMES,
        _causal_risk_path,
        list_reward_schemes,
        risk_aware,
        time_aversion_schedule,
        time_inhomogeneous_vol_aversion,
    )

    _HAVE_REWARDS = True
except Exception:  # pragma: no cover - exercised only without numpy/binding
    _HAVE_REWARDS = False


requires_rewards = pytest.mark.skipif(
    not _HAVE_REWARDS, reason="sharpearena.rewards not importable"
)


def _series(seed: int, n: int = 64, scale: float = 0.01) -> list[float]:
    """A seeded synthetic return series. Seeded RNG only, never a clock."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.normal(0.0005, scale, size=n)]


# -- registration -----------------------------------------------------------


@requires_rewards
def test_new_schemes_are_registered_and_listed():
    assert REWARD_SCHEMES["risk_aware"] is risk_aware
    assert (
        REWARD_SCHEMES["time_inhomogeneous_vol_aversion"]
        is time_inhomogeneous_vol_aversion
    )
    listed = list_reward_schemes()
    assert "risk_aware" in listed
    assert "time_inhomogeneous_vol_aversion" in listed
    # The six pre-existing schemes must survive the addition.
    assert {
        "default",
        "differential_sharpe",
        "sortino",
        "drawdown_penalized",
        "turnover_penalized",
        "loss_averse",
    } <= set(listed)
    assert len(listed) == 8


# -- boundedness and finiteness ---------------------------------------------


@requires_rewards
@pytest.mark.parametrize("scheme", ["risk_aware", "time_inhomogeneous_vol_aversion"])
@pytest.mark.parametrize("scale", [1e-6, 0.01, 0.5, 5.0])
def test_rewards_are_finite_and_bounded(scheme, scale):
    fn = REWARD_SCHEMES[scheme]
    for seed in range(6):
        val = fn(state={"returns": _series(seed, scale=scale)})
        assert np.isfinite(val)
        assert -1.0 <= val <= 1.0


@requires_rewards
@pytest.mark.parametrize("scheme", ["risk_aware", "time_inhomogeneous_vol_aversion"])
def test_degenerate_inputs_score_zero(scheme):
    fn = REWARD_SCHEMES[scheme]
    assert fn(state=None) == 0.0
    assert fn(state={}) == 0.0
    assert fn(state={"returns": []}) == 0.0
    # A single bar is inside the warm-up: no risk is charged yet, so the reward is the
    # bounded return and nothing else.
    assert fn(state={"returns": [0.0]}) == 0.0


# -- determinism ------------------------------------------------------------


@requires_rewards
@pytest.mark.parametrize("scheme", ["risk_aware", "time_inhomogeneous_vol_aversion"])
def test_rewards_are_deterministic_under_fixed_seed(scheme):
    fn = REWARD_SCHEMES[scheme]
    rets = _series(7)
    first = fn(state={"returns": rets})
    for _ in range(4):
        assert fn(state={"returns": list(rets)}) == first


# -- risk_aware behavior ----------------------------------------------------


@requires_rewards
def test_risk_aware_charges_the_prevailing_volatility():
    # Same summed return, same number of bars; one path carries its return through a
    # turbulent stretch, the other through a calm one. The per-bar charge must separate
    # them, which an episode-terminal aggregate would not.
    calm = [0.001] * 40
    turbulent = [0.021, -0.019] * 20
    assert abs(sum(calm) - sum(turbulent)) < 1e-12
    assert risk_aware(state={"returns": calm}) > risk_aware(state={"returns": turbulent})


@requires_rewards
def test_risk_aware_lam_zero_reduces_to_bounded_return():
    rets = _series(3)
    assert risk_aware(state={"returns": rets}, lam=0.0) == pytest.approx(
        float(np.tanh(np.sum(rets)))
    )


@requires_rewards
def test_risk_aware_penalty_is_monotone_in_lam():
    rets = _series(11, scale=0.03)
    vals = [risk_aware(state={"returns": rets}, lam=lam) for lam in (0.0, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals, reverse=True)


@requires_rewards
def test_risk_charge_is_causal():
    # The charge at bar t may read bars < t and nothing else. Replacing the LAST bar with a
    # wildly different value must leave every charge in the path untouched, and the first
    # two bars are inside the warm-up so they are charged nothing at all.
    base = _series(5, n=40)
    perturbed = base[:-1] + [5.0]
    assert _causal_risk_path(base, 0.06) == _causal_risk_path(perturbed, 0.06)
    assert _causal_risk_path(base, 0.06)[:2] == [0.0, 0.0]
    # And the charge is non-negative everywhere, so it can only ever subtract.
    assert all(s >= 0.0 for s in _causal_risk_path(base, 0.06))


# -- time inhomogeneity -----------------------------------------------------


@requires_rewards
def test_aversion_schedule_moves_across_the_episode():
    sched = time_aversion_schedule(10, lam_start=0.25, lam_end=1.5)
    assert len(sched) == 10
    assert sched[0] == pytest.approx(0.25)
    assert sched[-1] == pytest.approx(1.5)
    assert sched == sorted(sched)
    # A constant-aversion scheme would be flat; this one must not be.
    assert sched[-1] > sched[0]


@requires_rewards
@pytest.mark.parametrize("shape", ["linear", "convex", "concave", "exponential"])
def test_aversion_schedule_shapes_share_endpoints(shape):
    sched = time_aversion_schedule(16, lam_start=0.2, lam_end=2.0, shape=shape)
    assert sched[0] == pytest.approx(0.2)
    assert sched[-1] == pytest.approx(2.0)
    assert all(np.isfinite(sched))


@requires_rewards
def test_aversion_schedule_convex_lags_concave():
    convex = time_aversion_schedule(21, shape="convex")
    concave = time_aversion_schedule(21, shape="concave")
    mid = 10
    assert convex[mid] < concave[mid]


@requires_rewards
def test_aversion_schedule_rejects_unknown_shape_and_bad_endpoints():
    with pytest.raises(ValueError):
        time_aversion_schedule(8, shape="nope")
    with pytest.raises(ValueError):
        time_aversion_schedule(8, lam_start=0.0, shape="exponential")


@requires_rewards
def test_risk_treatment_changes_across_the_episode():
    # The load-bearing claim: the SAME bars are charged differently depending on WHERE in
    # the episode they sit. A series whose volatility arrives late is charged more under a
    # rising aversion schedule than under a falling one, and a constant schedule sits
    # between the two. No other scheme in the registry can produce this spread, because
    # they all apply one fixed risk treatment at every bar.
    late_burst = [0.001] * 30 + [0.04, -0.04] * 5
    rising = time_inhomogeneous_vol_aversion(
        state={"returns": late_burst}, lam_start=0.25, lam_end=1.5
    )
    falling = time_inhomogeneous_vol_aversion(
        state={"returns": late_burst}, lam_start=1.5, lam_end=0.25
    )
    constant = time_inhomogeneous_vol_aversion(
        state={"returns": late_burst}, lam_start=0.875, lam_end=0.875
    )
    assert rising < constant < falling


@requires_rewards
def test_aversion_shape_moves_the_charge_within_one_schedule():
    # Same endpoints, different travel between them: a convex climb spends most of the
    # episode at low aversion, so it charges a late-volatility path less than a concave one.
    late_burst = [0.001] * 30 + [0.04, -0.04] * 5
    convex = time_inhomogeneous_vol_aversion(state={"returns": late_burst}, shape="convex")
    concave = time_inhomogeneous_vol_aversion(state={"returns": late_burst}, shape="concave")
    assert convex > concave


@requires_rewards
def test_time_inhomogeneous_collapses_to_constant_aversion_when_endpoints_match():
    # lam_start == lam_end is the degenerate, time-homogeneous case: it must agree with
    # risk_aware at that same lam, proving the only difference is the schedule.
    rets = _series(9, scale=0.02)
    flat = time_inhomogeneous_vol_aversion(
        state={"returns": rets}, lam_start=0.8, lam_end=0.8
    )
    assert flat == pytest.approx(risk_aware(state={"returns": rets}, lam=0.8))


@requires_rewards
def test_time_inhomogeneous_horizon_override_changes_the_schedule():
    # A truncated episode priced against its mandate horizon never reaches lam_end, so it
    # is charged less than the same bars priced as if they were the whole horizon.
    rets = _series(13, n=20, scale=0.03)
    own = time_inhomogeneous_vol_aversion(state={"returns": rets})
    stretched = time_inhomogeneous_vol_aversion(state={"returns": rets}, horizon=200)
    assert stretched > own
    # A horizon shorter than the realized bars is ignored rather than truncating the run.
    assert time_inhomogeneous_vol_aversion(state={"returns": rets}, horizon=1) == own


@requires_rewards
def test_time_inhomogeneous_shapes_produce_different_rewards():
    rets = _series(17, n=48, scale=0.03)
    vals = {
        shape: time_inhomogeneous_vol_aversion(state={"returns": rets}, shape=shape)
        for shape in ("linear", "convex", "concave", "exponential")
    }
    assert len(set(round(v, 12) for v in vals.values())) > 1
    for v in vals.values():
        assert np.isfinite(v)
        assert -1.0 <= v <= 1.0
