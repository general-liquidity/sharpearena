"""Opt-in execution settings: timing, transient impact, settlement and their identity."""

import json
import math

import pytest
from sharpearena import MarketExecution, manipulation
from sharpearena.impact_diagnostics import (
    ImpactDiagnosticError,
    meta_order_impact_shape,
)
from sharpearena.sharpearena_py import PyMarketClearing, TradingEnv


def transient(shape="square_root", decay=0.99, coefficient=0.1, **kwargs):
    return MarketExecution(
        transient_coefficient=coefficient,
        transient_decay=decay,
        transient_shape=shape,
        **kwargs,
    )


def market(execution=None, **kwargs):
    arguments = dict(n_symbols=1, n_days=40, seed=3, n_agents=1)
    arguments.update(kwargs)
    if execution is not None:
        arguments["execution"] = execution.to_json()
    return PyMarketClearing(**arguments)


def drive(native, weights):
    start = json.loads(native.reset_market())
    steps = []
    for weight in weights:
        steps.append(json.loads(native.step_market(json.dumps([[weight]]))))
        if steps[-1]["done"]:
            break
    return start, steps


# -- defaults stay byte-identical ------------------------------------------------------


def test_absent_settings_keep_the_wire_shape_and_the_frozen_defaults_match_it():
    plain_start, plain = drive(market(), [0.5, 0.2, -0.3] * 10)
    opted_start, opted = drive(market(MarketExecution()), [0.5, 0.2, -0.3] * 10)
    assert plain_start == opted_start
    assert all("execution" not in step for step in plain)
    for left, right in zip(plain, opted):
        right.pop("execution")
        assert left == right


def test_trading_env_default_readback_and_info_are_unchanged():
    native = _env()
    assert set(json.loads(native.effective_config)) == {
        "n_symbols",
        "n_bars",
        "window_start",
        "window_end",
        "exec_seed",
        "dataset_fnv1a64",
    }
    native.reset()
    _, _, _, info = native.step('{"orders": []}')
    assert set(json.loads(info)) == {"nav", "events"}


# -- timing ----------------------------------------------------------------------------


@pytest.mark.parametrize("bars", [1, 2, 3])
def test_market_decision_fills_exactly_bars_after_its_last_observed_close(bars):
    native = market(
        MarketExecution(fixed_delay_bars=bars), kyle_lambda=0.0, eta=0.0, capital=1.0
    )
    flat, _ = drive(market(kyle_lambda=0.0, eta=0.0), [])
    _, tape = drive(market(kyle_lambda=0.0, eta=0.0), [0.0] * 20)
    closes = [step["cleared_mids"][0] for step in tape]
    start, steps = drive(native, [0.01 * (i + 1) for i in range(20)])
    first_close = start["observations"][0]["symbols"][0]["close_history"][-1]
    assert first_close == flat["observations"][0]["symbols"][0]["close_history"][-2]
    assert first_close != closes[0]
    for index, step in enumerate(steps):
        eligible = step["execution"]["decision_eligible"]
        assert eligible is (index + 1 >= bars)
        fill = step["fills"][0][0]
        if eligible:
            assert fill["fill_price"] == closes[index]
            assert fill["size"] != 0.0
        else:
            assert fill["size"] == 0.0


def _env(**kwargs):
    return TradingEnv(n_symbols=1, n_days=30, seed=1, **kwargs)


def _decision(env_reset_json, weight=0.5):
    symbol = json.loads(env_reset_json)["symbols"][0]["symbol"]
    return json.dumps(
        {"orders": [{"symbol": symbol, "action": "buy", "target_weight": weight}]}
    )


HOLD = '{"orders": []}'


@pytest.mark.parametrize("bars", [1, 2])
def test_trading_env_delay_equals_immediate_execution_shifted_by_holds(bars):
    weights = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.6, 0.4]
    late = _env(fixed_delay_bars=bars)
    now = _env()
    first = late.reset()
    assert first == now.reset()
    late_rows = []
    for index, weight in enumerate(weights):
        step = late.step(_decision(first, weight))
        info = json.loads(step[3])
        assert info.pop("decision_eligible") is (index >= bars)
        late_rows.append((step[0], step[1], step[2], info))
    shifted = [HOLD] * bars + [_decision(first, w) for w in weights[: len(weights) - bars]]
    now_rows = []
    for decision in shifted:
        step = now.step(decision)
        now_rows.append((step[0], step[1], step[2], json.loads(step[3])))
    assert late_rows == now_rows


def test_trading_env_binds_the_delay_into_its_readback_and_identity():
    late = _env(fixed_delay_bars=1)
    block = json.loads(late.effective_config)
    assert block["fixed_delay"] == {
        "schema_version": "sharpebench.fixed-decision-delay.v1",
        "bars": 1,
    }
    assert block["dataset_fnv1a64"] == json.loads(_env().effective_config)["dataset_fnv1a64"]
    other_timing = json.loads(_env(fixed_delay_bars=2).effective_config)
    other_costs = json.loads(_env(fixed_delay_bars=1, fee_bps=9.0).effective_config)
    assert block["execution_fnv1a64"] != other_timing["execution_fnv1a64"]
    assert block["execution_fnv1a64"] != other_costs["execution_fnv1a64"]


def test_trading_env_refuses_a_zero_delay_and_keeps_the_queue_in_checkpoints():
    with pytest.raises(ValueError, match="fixed_delay_bars"):
        _env(fixed_delay_bars=0)

    def run(checkpoint_at=None):
        native = _env(fixed_delay_bars=2)
        first = native.reset()
        rewards = []
        for step in range(8):
            if step == checkpoint_at:
                snapshot = native.clone_state()
                assert len(json.loads(snapshot)["delayed_decisions"]) == 2
                restored = _env(fixed_delay_bars=2)
                restored.reset()
                restored.restore_state(snapshot)
                native = restored
            rewards.append(native.step(_decision(first, 0.1 * step))[1])
        return rewards

    assert run() == run(checkpoint_at=4)


def test_trading_env_refuses_a_snapshot_that_overfills_the_queue():
    native = _env(fixed_delay_bars=2)
    first = native.reset()
    native.step(_decision(first))
    native.step(_decision(first))
    snapshot = native.clone_state()
    shorter = _env(fixed_delay_bars=1)
    shorter.reset()
    with pytest.raises(ValueError, match="delayed decisions"):
        shorter.restore_state(snapshot)


def test_trading_env_reset_clears_waiting_decisions():
    native = _env(fixed_delay_bars=2)
    first = native.reset()
    native.step(_decision(first))
    native.reset()
    assert json.loads(native.clone_state())["delayed_decisions"] == []


# -- identity --------------------------------------------------------------------------


def test_effective_config_reads_back_every_setting_and_the_identity_moves_with_each():
    execution = transient(fixed_delay_bars=2, settlement="cash_liquidation")
    block = json.loads(market(execution).effective_config)
    identity = block["identity"]
    assert identity["execution"] == {
        "schema_version": "sharpearena.market-execution.v1",
        "fixed_delay": {"schema_version": "sharpebench.fixed-decision-delay.v1", "bars": 2},
        "transient": {"coefficient": 0.1, "decay_per_bar": 0.99, "shape": "square_root"},
        "settlement": "cash_liquidation",
    }
    assert identity["params"]["lambda"] == 0.1
    variants = [
        execution,
        transient(fixed_delay_bars=3, settlement="cash_liquidation"),
        transient(fixed_delay_bars=2),
        transient(fixed_delay_bars=2, settlement="cash_liquidation", decay=0.9),
        transient(fixed_delay_bars=2, settlement="cash_liquidation", coefficient=0.2),
        transient("linear", fixed_delay_bars=2, settlement="cash_liquidation"),
        None,
    ]
    digests = {
        json.loads(market(variant).effective_config)["execution_fnv1a64"] for variant in variants
    }
    assert len(digests) == len(variants)


def test_identity_follows_live_setters_and_is_stable_across_rebuilds():
    native = market()
    before = json.loads(native.effective_config)
    assert before["identity"]["execution"] is None
    assert before == json.loads(market().effective_config)
    native.set_impact_exponent(0.5)
    assert json.loads(native.effective_config)["execution_fnv1a64"] != before["execution_fnv1a64"]
    native.set_impact_exponent(1.0)
    native.set_uncertainty(0.01, 0.01, 0.0)
    assert json.loads(native.effective_config)["execution_fnv1a64"] != before["execution_fnv1a64"]


@pytest.mark.parametrize(
    "mutation",
    [
        ("sharpearena.market-execution.v1", "sharpearena.market-execution.v2"),
        ("sharpebench.fixed-decision-delay.v1", "sharpebench.fixed-decision-delay.v9"),
        ("square_root", "cube_root"),
        ("cash_liquidation", "liquidate"),
    ],
)
def test_an_unknown_version_or_label_is_refused(mutation):
    wire = transient(fixed_delay_bars=1, settlement="cash_liquidation").to_json()
    with pytest.raises(ValueError, match="invalid execution"):
        PyMarketClearing(execution=wire.replace(*mutation))


@pytest.mark.parametrize(
    "execution",
    [
        MarketExecution(fixed_delay_bars=0),
        transient(decay=1.0),
        transient(decay=-0.1),
        transient(coefficient=-0.1),
    ],
)
def test_an_out_of_range_setting_is_refused(execution):
    with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
        PyMarketClearing(execution=execution.to_json())


def test_a_half_declared_kernel_is_refused_before_it_reaches_the_engine():
    with pytest.raises(ValueError, match="set together"):
        MarketExecution(transient_coefficient=0.1)
    with pytest.raises(ValueError, match="finite"):
        MarketExecution(transient_coefficient=math.nan, transient_decay=0.5)
    with pytest.raises(ValueError, match="transient_shape"):
        transient("cube_root")


# -- the shape claim, checked by the probe that makes it -------------------------------


def test_square_root_shape_holds_only_where_decay_is_slow_against_execution():
    slow = meta_order_impact_shape(kyle_lambda=0.0, execution=transient(decay=0.999))
    assert slow.execution_exponent == pytest.approx(0.5, abs=0.01)
    assert slow.duration_exponent == pytest.approx(0.0, abs=0.01)
    assert slow.relaxation_ratios[-1] < 1.0
    fast = meta_order_impact_shape(kyle_lambda=0.0, execution=transient(decay=0.9))
    assert fast.execution_exponent < 0.35
    assert fast.relaxation_ratios[-1] < 0.5


def test_linear_kernel_relaxes_and_is_not_the_square_root_law():
    shape = meta_order_impact_shape(kyle_lambda=0.0, execution=transient("linear", decay=0.99))
    assert shape.execution_exponent > 0.9
    assert shape.relaxation_ratios == pytest.approx(
        [0.99**j for j in range(len(shape.relaxation_ratios))], rel=1e-6
    )


def test_permanent_part_stays_non_relaxing_beside_the_kernel():
    shape = meta_order_impact_shape(execution=transient(decay=0.5, coefficient=0.0))
    assert shape.execution_exponent == pytest.approx(1.0, abs=0.02)
    assert shape.max_relaxation_deviation < 1e-9


def test_the_shape_probe_refuses_a_fixed_delay():
    with pytest.raises(ImpactDiagnosticError, match="fixed delay"):
        meta_order_impact_shape(execution=MarketExecution(fixed_delay_bars=1))


# -- the no-arbitrage probe ------------------------------------------------------------


def probe(execution, schedule, side, **kwargs):
    arguments = dict(n_followers=0, n_days=80, push_weight=0.8, eta=0.0)
    arguments.update(kwargs)
    params = manipulation.ManipulationParams(execution=execution, **arguments)
    return [
        manipulation.run_asymmetric_probe(params=params, schedule=schedule, seed=seed, side=side)
        for seed in range(4)
    ]


SCHEDULES = [(1, 1), (10, 1), (1, 10), (10, 10), (20, 2), (2, 20)]


@pytest.mark.parametrize("shape", ["linear", "square_root"])
@pytest.mark.parametrize("decay", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("up,down", SCHEDULES)
@pytest.mark.parametrize("side", [1, -1])
def test_no_transient_round_trip_pays_on_the_probe_grid(shape, decay, up, down, side):
    # kyle_lambda = eta = 0: the kernel alone, with no other friction to hide behind.
    execution = transient(
        shape, decay=decay, coefficient=0.2, settlement="cash_liquidation"
    )
    schedule = manipulation.AsymmetricSchedule.uniform(up, down)
    for result in probe(execution, schedule, side, kyle_lambda=0.0):
        assert not result.profitable, result.impact_pnl


def test_the_probe_still_fires_on_its_positive_control_under_cash_settlement():
    control = probe(
        MarketExecution(settlement="cash_liquidation"),
        manipulation.AsymmetricSchedule.uniform(20, 2),
        -1,
        kyle_lambda=0.1,
        impact_exponent=0.5,
    )
    assert all(result.profitable for result in control)


def test_cash_settlement_leaves_the_probe_flat_with_nav_equal_to_cash():
    results = probe(
        transient(settlement="cash_liquidation"),
        manipulation.AsymmetricSchedule.uniform(10, 10),
        1,
        kyle_lambda=0.1,
    )
    for result in results:
        assert result.terminal_positions_live == (0.0,)
        assert result.terminal_positions_reference == (0.0,)
        assert result.terminal_marked_position_value_live == 0.0
        assert result.live_pnl == pytest.approx(result.terminal_cash_live - 1.0, abs=1e-12)


def test_the_reader_note_round_trip_issue_is_real_under_the_marked_default():
    results = probe(None, manipulation.AsymmetricSchedule.uniform(5, 5), 1)
    assert all(abs(result.terminal_positions_live[0]) > 1e-7 for result in results)


def test_the_reference_arm_keeps_timing_and_drops_only_the_displacement():
    execution = transient(fixed_delay_bars=1, settlement="cash_liquidation")
    assert execution.without_impact().transient_coefficient == 0.0
    assert execution.without_impact().fixed_delay_bars == 1
    assert execution.without_impact().settlement == "cash_liquidation"
    assert MarketExecution().without_impact() == MarketExecution()
