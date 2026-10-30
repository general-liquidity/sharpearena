"""Tests for the study producers. Pure logic, plus the fixture path of the real-panel reader."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

STUDIES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIES))

import common  # noqa: E402
from common import Job, StudyRefused  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


realism = _load("realism_produce", STUDIES / "realism" / "produce.py")
learner = _load("learner_produce", STUDIES / "learner" / "produce.py")
REALISM, _ = common.load_protocol(STUDIES / "realism" / "protocol.json")
LEARNER, _ = common.load_protocol(STUDIES / "learner" / "protocol.json")


# -- common -------------------------------------------------------------------


def test_overlapping_bands_are_refused_by_name():
    with pytest.raises(StudyRefused, match="seen and heldout"):
        common.require_disjoint({"seen": [1, 2], "dev": [3], "heldout": [2]})


def test_every_protocol_band_is_disjoint():
    bands = REALISM["bands"]
    common.require_disjoint({k: common.band(v["start"], v["count"]) for k, v in bands.items()})
    spec = LEARNER["bands"]
    fixed = {k: common.band(v["start"], v["count"]) for k, v in spec.items() if "count" in v}
    pools = {
        "pilot": common.band(spec["pilot_seen"]["start"], 8 * 16),
        "main16": common.band(spec["main_seen_16"]["start"], 32 * 16),
        "main256": common.band(spec["main_seen_256"]["start"], 32 * 256),
    }
    common.require_disjoint({**fixed, **pools})


def _double(payload: dict) -> dict:
    marker = Path(payload["marker"])
    marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
    if payload["value"] < 0:
        raise ValueError("negative")
    return {"value": payload["value"] * 2}


def test_finished_jobs_are_not_recomputed_and_a_failure_stays(tmp_path):
    marker = tmp_path / "calls"
    jobs = [Job("a", {"value": 2, "marker": str(marker)}), Job("b", {"value": -1, "marker": str(marker)})]
    first = common.run_jobs(jobs, _double, tmp_path, 1)
    assert first == {"ok": 1, "failed": 1, "skipped": 0}
    second = common.run_jobs(jobs, _double, tmp_path, 1)
    assert second == {"ok": 0, "failed": 0, "skipped": 2}
    assert marker.read_text() == "xx"
    results, absent = common.collect(tmp_path, ["a", "b", "c"])
    assert results == {"a": {"value": 4}}
    assert absent == [
        {"key": "b", "status": "failed", "error_type": "ValueError"},
        {"key": "c", "status": "missing"},
    ]


def test_retry_failed_reruns_only_the_failure(tmp_path):
    marker = tmp_path / "calls"
    jobs = [Job("a", {"value": 2, "marker": str(marker)}), Job("b", {"value": -1, "marker": str(marker)})]
    common.run_jobs(jobs, _double, tmp_path, 1)
    counts = common.run_jobs(jobs, _double, tmp_path, 1, retry_failed=True)
    assert counts == {"ok": 0, "failed": 1, "skipped": 1}


def test_a_job_output_has_no_clock():
    assert "seconds" not in common.canonical({"status": "ok", "key": "a", "result": {}})


def test_clopper_pearson_known_values():
    assert common.cp_lower(0, 10, 0.95) == 0.0
    assert common.cp_upper(10, 10, 0.95) == 1.0
    assert common.cp_upper(0, 10, 0.95) == pytest.approx(1 - 0.05 ** 0.1)
    assert common.cp_lower(10, 10, 0.95) == pytest.approx(0.05 ** 0.1)


def test_t_interval_is_typed_when_it_cannot_exist():
    assert "unavailable" in common.t_interval([1.0], 0.95)
    assert "unavailable" in common.t_interval([1.0, float("nan")], 0.95)
    row = common.t_interval([1.0, 3.0], 0.95)
    assert row["mean"] == 2.0 and row["half"] == pytest.approx(12.7062 * 2 ** 0.5 / 2 ** 0.5, rel=1e-3)


def test_holm_stops_at_the_first_failure():
    assert common.holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05) == {"a": True, "b": False, "c": False}


# -- realism panels -----------------------------------------------------------


def test_real_windows_refuse_a_changed_file(tmp_path):
    path = tmp_path / "x.csv"
    realism.write_fixture(path, bars=130)
    with pytest.raises(StudyRefused, match="hash"):
        realism.real_windows(path, "0" * 64, 120, 4, 120)


def test_real_windows_do_not_overlap_and_refuse_a_short_stride(tmp_path):
    path = tmp_path / "x.csv"
    realism.write_fixture(path, bars=370)
    windows = realism.real_windows(path, None, 120, 4, 120)
    assert len(windows) == 3 and windows[0].shape == (120, 4)
    assert not np.array_equal(windows[0], windows[1])
    with pytest.raises(StudyRefused, match="overlapping"):
        realism.real_windows(path, None, 120, 4, 60)


def test_real_job_rows_carry_statistics_and_no_observation(tmp_path):
    path = tmp_path / "x.csv"
    realism.write_fixture(path, bars=130)
    geometry = REALISM["geometry"]
    out = realism.run_price_job(
        {"family": "real", "geometry": geometry, "arm": "pos_real:x", "band": "real", "path": str(path),
         "sha256": None, "stride": 120}
    )
    row = out["rows"][0]
    assert set(row) == {"id", "passed", "checks", "facts"}
    assert len(row["facts"]) == 6


def test_shuffling_keeps_each_marginal_and_breaks_time_order():
    rng = np.random.default_rng(1)
    window = np.cumsum(rng.standard_normal((120, 4)), axis=0)
    shuffled = realism.shuffled_window(np.random.default_rng(2), window)
    assert np.array_equal(np.sort(window, axis=0), np.sort(shuffled, axis=0))
    assert not np.array_equal(window, shuffled)
    first_col_order = np.argsort(shuffled[:, 0])
    second_col_order = np.argsort(shuffled[:, 1])
    assert not np.array_equal(first_col_order, second_col_order)


def test_garch_control_refuses_a_non_stationary_spec():
    with pytest.raises(StudyRefused):
        realism.garch_returns(np.random.default_rng(0), 10, 4, {"alpha": 0.5, "beta": 0.5, "nu": 5, "burn": 1, "scale": 0.01})


# -- realism decision rule ----------------------------------------------------


def _rate(lower: float, upper: float, rate: float = 0.5) -> dict:
    return {"lower": lower, "upper": upper, "rate": rate}


def _good_arms() -> dict:
    return {
        "pos_garch": _rate(0.8, 0.9),
        "neg_gauss": _rate(0.0, 0.02),
        "neg_iid_t": _rate(0.0, 0.08),
        "neg_shuffled_real": _rate(0.0, 0.09),
    }


BANDS = ["diagnostic", "confirmation"]


def _price() -> dict:
    return {b: _good_arms() for b in BANDS}


REAL_OK = {"crypto": _rate(0, 0, 0.75), "fx": _rate(0, 0, 0.8)}


def test_gate_is_valid_only_with_every_control_on_every_band():
    assert realism.gate_validity(_price(), REAL_OK, REALISM, BANDS)["valid"]


@pytest.mark.parametrize(
    "arm,row,reason",
    [
        ("pos_garch", _rate(0.74, 0.9), "pos_garch:lower_below_tau"),
        ("neg_iid_t", _rate(0.0, 0.11), "neg_iid_t:upper_above_limit"),
        ("neg_shuffled_real", _rate(0.0, 0.2), "neg_shuffled_real:upper_above_limit"),
    ],
)
def test_one_failed_control_on_one_band_invalidates_the_gate(arm, row, reason):
    price = _price()
    price["confirmation"][arm] = row
    result = realism.gate_validity(price, REAL_OK, REALISM, BANDS)
    assert not result["valid"] and f"confirmation:{reason}" in result["reasons"]


def test_real_tape_below_tau_invalidates_the_gate():
    result = realism.gate_validity(_price(), {"crypto": _rate(0, 0, 0.74), "fx": _rate(0, 0, 0.9)}, REALISM, BANDS)
    assert result["reasons"] == ["pos_real:crypto:share_below_tau"]


def test_missing_controls_are_never_a_pass():
    price = _price()
    del price["diagnostic"]["neg_gauss"]
    assert not realism.gate_validity(price, REAL_OK, REALISM, BANDS)["valid"]
    assert not realism.gate_validity(_price(), {}, REALISM, BANDS)["valid"]


def test_a_failed_gate_gives_no_tier_verdict():
    price = _price()
    for b in BANDS:
        price[b]["hard"] = _rate(0.9, 0.95)
    verdict = realism.tier_verdict("hard", price, {"valid": False, "reasons": ["x"]}, REALISM, BANDS)
    assert verdict == "unavailable:gate_failed_control"


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (_rate(0.8, 0.9), _rate(0.76, 0.9), "market_like_under_gate"),
        (_rate(0.1, 0.2), _rate(0.1, 0.74), "not_market_like_under_gate"),
        (_rate(0.8, 0.9), _rate(0.6, 0.74), "inconclusive"),
    ],
)
def test_tier_verdict_needs_both_bands(first, second, expected):
    price = _price()
    price["diagnostic"]["hard"], price["confirmation"]["hard"] = first, second
    assert realism.tier_verdict("hard", price, {"valid": True, "reasons": []}, REALISM, BANDS) == expected


def test_tier_bound_is_family_adjusted():
    rows = [{"passed": True, "checks": {"a": True}}] * 200 + [{"passed": False, "checks": {"a": False}}] * 56
    plain = realism.rate_row(rows, 0.95)
    adjusted = realism.rate_row(rows, 0.95, 3)
    assert adjusted["lower"] < plain["lower"] and adjusted["upper"] > plain["upper"]
    assert adjusted["failing_facts"] == ["a"]


def test_execution_block_never_carries_a_price_verdict_and_reads_e1_and_e2():
    rows_good = [{"share": 0.5 + d, "fills": [1, 1]} for d in (0.0005, -0.0005, 0.0002, -0.0002)]
    rows_bad = [{"share": 0.506 + d, "fills": [1, 1]} for d in (0.0005, -0.0005, 0.0002, -0.0002)]
    walk = lambda share: [{"reward": 1.0, "value": 1.0, "absolute_pnl": 100.0, "no_fill_pnl": share}]  # noqa: E731
    results = {
        "execution/b/e1_seeded_shuffle": {"rows": rows_good, "mark": "ex_own_mid", "controllers": "per_seat"},
        "execution/b/e1_agent_index": {"rows": rows_bad, "mark": "ex_own_mid", "controllers": "per_seat"},
        "execution/b/e2_ex_own_mid": {"rows": walk(1.0)},
        "execution/b/e2_book_mid": {"rows": walk(60.0)},
    }
    block = realism.execution_block(results, REALISM, "b")
    assert block["e1"]["exchangeable"] and block["e1"]["bad_arm_detected"]
    assert block["e1"]["good"]["priority"] == "seeded_shuffle" and block["e1"]["good"]["mark"] == "ex_own_mid"
    assert block["e2"]["good_within_tolerance"] and block["e2"]["bad_exceeds_tolerance"]
    assert block["e3"]["1.0"] == {"unavailable": "missing"}
    assert "verdict" not in json.dumps(block)


# -- learner ------------------------------------------------------------------


def test_buy_and_hold_drifts_and_never_rebalances():
    policy = learner.BuyAndHold()
    first = policy({"closes": np.array([100.0, 100.0])})
    second = policy({"closes": np.array([200.0, 100.0])})
    assert first.tolist() == [0.5, 0.5]
    assert second == pytest.approx([2 / 3, 1 / 3])


def test_memoriser_uses_the_next_bar():
    closes = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])
    assert learner.memoriser_actions(closes).tolist() == [[0.5, -0.5], [-0.5, 0.5]]


def test_holding_periods_match_bench_cases():
    gross = [0.0, 0.5, 0.4, 0.0, 0.0, 1e-9, 2e-9, 0.2]
    assert learner.holding_periods(gross, 1e-9) == [(1, 2), (6, 2)]
    assert learner.holding_periods([0.3, 0.3], 1e-9) == [(0, 2)]
    assert learner.holding_periods([0.0, 1e-9], 1e-9) == []


def test_random_layout_keeps_lengths_apart():
    lengths = [3, 1, 2]
    for seed in range(200):
        layout = sorted(learner.random_layout(lengths, 12, np.random.default_rng(seed)), key=lambda x: x[1])
        assert sorted(p for p, _ in layout) == [0, 1, 2]
        for (a, start_a), (_, start_b) in zip(layout, layout[1:]):
            assert start_a + lengths[a] < start_b
        last, start = layout[-1]
        assert start + lengths[last] <= 12


def test_moved_actions_preserve_the_entrants_exposure():
    actions = np.zeros((20, 2), dtype=np.float32)
    actions[3:6] = [0.5, 0.25]
    actions[10:12] = [0.1, 0.1]
    placed, kind = learner.moved_actions(actions, 1e-9, np.random.default_rng(3))
    assert kind == "moved_holding_periods"
    assert np.array_equal(np.sort(placed, axis=0), np.sort(actions, axis=0))


def test_an_always_invested_policy_is_labelled_and_shifted_not_silently_passed():
    actions = np.arange(1, 21, dtype=np.float32).reshape(10, 2)
    placed, kind = learner.moved_actions(actions, 1e-9, np.random.default_rng(4))
    assert kind == "always_invested:circular_shift"
    assert not np.array_equal(placed, actions)
    assert np.array_equal(np.sort(placed, axis=0), np.sort(actions, axis=0))


def test_never_invested_is_typed():
    actions = np.zeros((5, 2), dtype=np.float32)
    assert learner.moved_actions(actions, 1e-9, np.random.default_rng(0))[1] == "never_invested"


def test_inactivity_measures_read_the_actions():
    active = learner.behaviour([0.001, -0.001, 0.002], [np.array([0.5, 0.5]), np.array([-0.5, 0.5]), np.array([0.5, -0.5])])
    assert active["gross"] == 1.0 and active["turnover"] == pytest.approx(0.75) and active["action_std"] > 0
    idle = learner.behaviour([0.0, 0.0], [np.zeros(2), np.zeros(2)])
    assert idle["gross"] == 0.0 and idle["turnover"] == 0.0 and idle["action_std"] == 0.0


def test_excess_pairs_by_seed_not_position():
    rows = [{"seed": 1, "mean_bp": 5.0}, {"seed": 2, "mean_bp": 7.0}]
    base = [{"seed": 2, "mean_bp": 1.0}, {"seed": 1, "mean_bp": 2.0}]
    assert learner.excess_by_seed(rows, base) == [3.0, 6.0]


def test_nested_summary_uses_bands_as_the_unit():
    values = {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 3.0, (1, 1): 3.0, (2, 0): 5.0, (2, 1): 5.0}
    row = learner.nested_summary(values, 0.95)
    assert row["interval"]["n"] == 3 and row["n_runs"] == 6
    assert row["between_band_variance"] == pytest.approx(4.0) and row["within_band_run_variance"] == 0.0


def _row(memo, seen_lo, seen_mean, floor, gap):
    return {
        "memoriser_gap": memo,
        "seen_excess": {"interval": {"lo": seen_lo, "mean": seen_mean}},
        "floor_excess_bp": floor,
        "gap_final": {"interval": gap},
    }


MEMO = {"lo": 5.0}


@pytest.mark.parametrize(
    "row,expected",
    [
        (_row({"lo": -1.0}, 5, 9, 1, {"lo": 1.0, "hi": 2.0}), "unavailable:instrument"),
        (_row({"unavailable": "x"}, 5, 9, 1, {"lo": 1.0, "hi": 2.0}), "unavailable:instrument"),
        (_row(MEMO, 0.5, 9, 1, {"lo": 1.0, "hi": 2.0}), "unavailable:no_capacity"),
        (_row(MEMO, 5, 2, 3, {"lo": 1.0, "hi": 2.0}), "unavailable:no_capacity"),
        (_row(MEMO, 5, 9, 1, {"lo": 0.1, "hi": 2.0}), "overfit_detected"),
        (_row(MEMO, 5, 9, 1, {"lo": -0.5, "hi": 0.5}), "within_margin"),
        (_row(MEMO, 5, 9, 1, {"lo": -2.0, "hi": 0.5}), "inconclusive"),
    ],
)
def test_the_decision_rule_order(row, expected):
    assert learner.verdict(row, LEARNER) == expected


def test_lost_controls_leave_the_cell_unavailable_not_crashed():
    run = {"cell": "A", "config": "c0", "band": 0, "run": 0}
    row = learner.analyze_cell({"k": run}, {}, "", "A", "c0", LEARNER)
    assert row["unavailable"] == "controls_missing"
    assert learner.verdict(row, LEARNER) == "unavailable:controls_missing"


def test_pilot_rule_inflates_the_sd_and_respects_the_cap():
    small = learner.required_bands(0.2, 6, LEARNER)
    assert small["sd_upper"] > 0.2 and not small["cap_binds"] and small["half_width"] <= 0.5
    if small["bands"] > 2:
        from scipy import stats

        k = small["bands"] - 1
        assert stats.t.ppf(0.975, k - 1) * small["sd_upper"] / k ** 0.5 > 0.5
    large = learner.required_bands(50.0, 6, LEARNER)
    assert large["cap_binds"] and large["bands"] == 32 and large["half_width"] > 0.5


def test_freeze_interface_is_required_outside_smoke(monkeypatch):
    import sharpearena.wrappers as wrappers

    if hasattr(wrappers.CausalNormalizeObservation, "state_dict"):
        monkeypatch.delattr(wrappers.CausalNormalizeObservation, "state_dict")
    with pytest.raises(StudyRefused, match="feat/normalizer-freeze"):
        learner.resolve_normalizer("main")
    assert learner.resolve_normalizer("smoke")[1] == "smoke_shim"


def test_main_design_refuses_without_a_pilot_decision():
    with pytest.raises(StudyRefused, match="pilot_decision"):
        learner.stage_design(LEARNER, "main", None, 0)


def test_training_jobs_keep_heldout_out_of_the_seen_band_and_derive_distinct_seeds():
    design = learner.stage_design(LEARNER, "main", {"bands": 3, "env_steps": 100, "config": "c0", "cells": ["A", "C"]}, 0)
    _, training = learner.build_jobs(LEARNER, "main", design, 10, "")
    seeds = [j.payload["train_seed"] for j in training]
    assert len(set(seeds)) == len(seeds) == 3 * 2 * 2
    heldout = set(design["unseen"])
    for job in training:
        assert not heldout & set(job.payload["seen"])
    assert {len(j.payload["seen"]) for j in training} == {16, 256}
    assert all(j.payload["n_trials"]["best"] == 10 * LEARNER["budget"]["checkpoints"] for j in training)


def test_pilot_bands_do_not_reuse_search_bands():
    search = learner.pilot_design(LEARNER, "search", None, 100)
    design = learner.pilot_design(LEARNER, "design", "c0", 100)
    assert design["first_band"] == search["n_bands"]


def test_position_env_closes_ignore_own_orders():
    spec = LEARNER["env"]
    assert learner.closes_are_exogenous(5, spec, spec["primary_tier"])


def test_smoke_shim_freezes_statistics():
    from sharpearena import SharpeArenaEnv

    spec = LEARNER["env"]
    make = learner.SmokeFreezeShim
    state = learner.fit_normalizer(make, [79000, 79001], spec, "hard", 1)
    env = learner.frozen_env(make, state, spec, "hard", 79002)
    before = json.dumps(env.state_dict())
    obs, _ = env.reset()
    env.step(np.zeros(4, dtype=np.float32))
    assert json.dumps(env.state_dict()) == before and np.any(obs["closes"] != 0.0)
    assert isinstance(env.env, SharpeArenaEnv)
