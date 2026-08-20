"""Adverse selection of meta-orders, and the markout that measures it.

The load-bearing test here is :func:`test_informed_flow_marks_out_worse_than_uninformed`.
Every other number this module produces is meaningless if the informed trader has no real
informational edge, because then "adverse selection" is just noise with a metric's name on
it. The rest of the file checks that the measurement is exact (the decomposition is an
identity, not an estimate), reproducible from a seed, and honest about what it saw.

Pure numpy and gymnasium: no native extension, no pettingzoo, so nothing here skips.
"""

import json

import pytest

from sharpearena.adverse_selection import (
    AdverseSelectionParams,
    MakerMarkout,
    MetaOrder,
    compare_informed_vs_uninformed,
    fill_markout,
    run_adverse_selection,
)
from sharpearena.market_making import MMParams, fixed_spread_policy

_H = (1, 5, 20)


def _short(**overrides) -> AdverseSelectionParams:
    """A shorter episode so the suite stays quick; the mechanics are unchanged."""
    base = dict(
        mm=MMParams(n_steps=120),
        n_meta_orders=3,
        parent_size=60,
        n_children=10,
        markout_horizons=_H,
    )
    base.update(overrides)
    return AdverseSelectionParams(**base)


# ---------------------------------------------------------------------------
# Non-vacuity: the informed trader must actually be informed
# ---------------------------------------------------------------------------


def test_informed_flow_marks_out_worse_than_uninformed():
    """Informed meta-orders must cost the makers more than the same flow sided on a coin.

    This is the scenario's non-vacuity check. Both legs share the price path, the slice
    schedule and the parent size; only the correlation between the counterparty's direction
    and the next move differs. If the gap is not there, the informed trader has no edge and
    the whole module is measuring nothing.
    """
    result = compare_informed_vs_uninformed(params=_short(), n_episodes=16)

    for h in _H:
        assert result["informed_is_worse"][h], f"no adverse selection at horizon {h}"
        assert result["gap_per_unit"][h] > 0.0

    # The damage must also grow with the markout horizon: a price move that has not
    # happened yet cannot have hurt the maker, so a longer look-forward has to reveal more.
    gaps = [result["gap_per_unit"][h] for h in _H]
    assert gaps == sorted(gaps)


def test_informed_meta_flow_is_the_source_of_the_loss():
    """The per-counterparty split must pin the loss on the meta-order, not on noise flow.

    A maker can lose money for ordinary reasons (it was carrying inventory through a
    volatile patch), and on any single seed that can dominate. So the claim is made where
    it is actually a claim: pooled across seeds, the drift on meta-order fills is adverse
    and their markout per unit is worse than the noise flow's.
    """
    h = max(_H)
    meta_drift = 0.0
    meta_markout, meta_qty = 0.0, 0
    noise_markout, noise_qty = 0.0, 0
    for seed in range(8):
        report = run_adverse_selection(params=_short(), seed=seed)
        for maker in report.makers:
            assert maker.meta_filled_qty > 0
            mine = maker.meta_markout_per_unit[h] * maker.meta_filled_qty
            meta_drift += maker.by_counterparty["informed"][h]
            meta_markout += mine
            meta_qty += maker.meta_filled_qty
            noise_markout += maker.markout[h] - mine
            noise_qty += maker.filled_qty - maker.meta_filled_qty

    assert meta_drift < 0.0
    assert meta_qty > 0 and noise_qty > 0
    assert meta_markout / meta_qty < noise_markout / noise_qty


def test_uninformed_mode_leaves_the_price_path_untouched():
    """The paired control is only paired if the two legs see the same prices.

    The alpha signs are drawn from their own stream in both modes, so flipping ``informed``
    changes who trades which way and nothing about what the price then does.
    """
    informed = run_adverse_selection(params=_short(informed=True), seed=7)
    uninformed = run_adverse_selection(params=_short(informed=False), seed=7)

    assert [m.alpha for m in informed.meta_orders] == [
        m.alpha for m in uninformed.meta_orders
    ]
    assert [m.start_step for m in informed.meta_orders] == [
        m.start_step for m in uninformed.meta_orders
    ]
    assert [m.children for m in informed.meta_orders] == [
        m.children for m in uninformed.meta_orders
    ]
    assert all(m.side == (1 if m.alpha >= 0 else -1) for m in informed.meta_orders)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def test_markout_decomposes_exactly_into_spread_capture_and_drift():
    """``markout = spread_capture + adverse_drift`` must hold to floating-point exactness.

    The decomposition is a partition of the same per-fill arithmetic, not two separately
    estimated quantities, so any drift here is a bug rather than a rounding artefact.
    """
    report = run_adverse_selection(params=_short(), seed=1)
    for maker in report.makers:
        for h in _H:
            assert maker.markout[h] == pytest.approx(
                maker.spread_capture + maker.adverse_drift[h], abs=1e-9
            )


def test_spread_capture_is_positive_and_horizon_independent():
    """Spread capture is booked at the fill, so it is quote depth times quantity: strictly
    positive whenever anything filled, and unchanged by the markout horizon."""
    report = run_adverse_selection(params=_short(), seed=2)
    for maker in report.makers:
        assert maker.filled_qty > 0
        assert maker.spread_capture > 0.0
        assert maker.spread_capture == pytest.approx(
            sum(abs(f.signed_qty) * f.depth for f in report.fills if f.maker == maker.maker)
        )


def test_fills_near_the_episode_end_mark_out_against_the_last_mid():
    """End-of-episode fills are marked against the final mid rather than dropped.

    Dropping them would quietly exclude the fills a maker takes while unwinding, which is
    exactly where its markout tends to be worst.
    """
    report = run_adverse_selection(params=_short(), seed=4)
    tail = report.fills[-5:]
    assert tail, "the scenario should produce fills"
    marks = fill_markout(tail, report.mid_path, horizon=10_000)
    last = report.mid_path[-1]
    assert marks == pytest.approx([f.signed_qty * (last - f.price) for f in tail])


def test_tighter_quotes_absorb_more_meta_flow_and_get_picked_off_harder():
    """Price priority has to bite: the tighter maker takes more of the meta-order and pays
    more of the markout. Without that tension there is no decision for a maker to make."""
    params = _short(n_makers=2)
    report = run_adverse_selection(
        params=params,
        policies=[fixed_spread_policy(0.3), fixed_spread_policy(1.2)],
        seed=5,
    )
    tight, wide = report.makers
    h = max(_H)
    assert tight.meta_filled_qty > wide.meta_filled_qty
    assert tight.meta_markout_per_unit[h] < wide.meta_markout_per_unit[h]


def test_verdict_names_the_subsidised_book():
    """A maker whose aggregate markout is positive only because noise flow pays for its
    meta-order fills must be told so, not left to compare two dictionaries."""
    report = run_adverse_selection(params=_short(), seed=0)
    h = max(_H)
    for maker in report.makers:
        if maker.markout[h] >= 0.0 and maker.meta_markout_per_unit[h] < 0.0:
            assert "subsidised" in maker.verdict
            break
    else:  # pragma: no cover - defensive, the default scenario hits the branch
        pytest.skip("no subsidised maker in this seed")


def test_toxic_rate_and_tail_share_are_bounded_fractions():
    """Both pick-off diagnostics are shares, so they live in ``[0, 1]``."""
    report = run_adverse_selection(params=_short(), seed=6)
    for maker in report.makers:
        for h in _H:
            assert 0.0 <= maker.toxic_fill_rate[h] <= 1.0
            assert 0.0 <= maker.worst_decile_share[h] <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility and shape
# ---------------------------------------------------------------------------


def test_markout_is_deterministic_under_a_fixed_seed():
    """Byte-identical reproduction is load-bearing: the same seed and params must give the
    same report, and a different seed must give a different one."""
    a = run_adverse_selection(params=_short(), seed=11)
    b = run_adverse_selection(params=_short(), seed=11)
    c = run_adverse_selection(params=_short(), seed=12)

    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())
    assert a.mid_path == b.mid_path
    assert a.fills == b.fills
    assert json.dumps(a.to_dict()) != json.dumps(c.to_dict())


def test_comparison_is_deterministic_under_a_fixed_seed_base():
    a = compare_informed_vs_uninformed(params=_short(), n_episodes=4, seed_base=20)
    b = compare_informed_vs_uninformed(params=_short(), n_episodes=4, seed_base=20)
    assert a == b


def test_report_is_plain_json_serialisable():
    report = run_adverse_selection(params=_short(), seed=8)
    blob = json.loads(json.dumps(report.to_dict()))
    assert blob["meta_offered_qty"] == 3 * 60
    assert 0 < blob["meta_filled_qty"] <= blob["meta_offered_qty"]
    assert len(blob["makers"]) == 2
    assert set(blob["makers"][0]["markout"]) == {"1", "5", "20"}


def test_slice_schedule_sums_to_the_parent_order():
    """A meta-order is a parent sliced into children; the children must be the parent."""
    report = run_adverse_selection(params=_short(parent_size=61, n_children=10), seed=9)
    for order in report.meta_orders:
        assert isinstance(order, MetaOrder)
        assert sum(order.children) == 61
        assert max(order.children) - min(order.children) <= 1


def test_maker_summary_shape():
    report = run_adverse_selection(params=_short(n_makers=3), seed=10)
    assert [m.maker for m in report.makers] == ["maker_0", "maker_1", "maker_2"]
    assert all(isinstance(m, MakerMarkout) for m in report.makers)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_makers": 0},
        {"n_meta_orders": 0},
        {"n_children": 0},
        {"parent_size": 0},
        {"markout_horizons": ()},
        {"markout_horizons": (0,)},
        {"meta_start_step": 119, "n_meta_orders": 4, "n_children": 10},
    ],
)
def test_params_reject_incoherent_scenarios(overrides):
    with pytest.raises(ValueError):
        _short(**overrides)


def test_policy_count_must_match_the_roster():
    with pytest.raises(ValueError, match="expected 2 policies"):
        run_adverse_selection(params=_short(), policies=[fixed_spread_policy(0.5)])
