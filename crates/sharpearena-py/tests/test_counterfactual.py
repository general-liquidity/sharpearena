"""The counterfactual ledger: every decision recorded, acted on or not."""

from __future__ import annotations

import json

import pytest
from sharpearena.counterfactual import (
    CounterfactualError,
    CounterfactualLedger,
    GhostOrder,
    ghost_orders_from_preflight,
)


def _executed(symbol="AAA", side="buy", quantity=10.0, price=100.0):
    return GhostOrder(symbol, side, quantity, quantity, price, "executed", "allowed")


def _dropped(symbol="AAA", side="buy", quantity=10.0, price=100.0, why="risk_refused"):
    return GhostOrder(symbol, side, quantity, 0.0, price, why, "gross-exposure-limit")


def test_a_ghost_order_cannot_claim_a_difference_that_is_not_there():
    with pytest.raises(CounterfactualError, match="claims a difference"):
        GhostOrder("AAA", "buy", 5.0, 5.0, 100.0, "risk_refused", "no")
    with pytest.raises(CounterfactualError, match="requires the full intended"):
        GhostOrder("AAA", "buy", 5.0, 2.0, 100.0, "executed", "no")
    with pytest.raises(CounterfactualError, match="cannot exceed"):
        GhostOrder("AAA", "buy", 5.0, 9.0, 100.0, "resized", "no")


def test_an_unacted_decision_is_still_recorded():
    ledger = CounterfactualLedger()
    entry = ledger.record(
        decision={"orders": [{"symbol": "AAA", "target_weight": 0.1}]},
        observation={"AAA": 100.0},
        orders=[_dropped()],
        settlement_prices={"AAA": 100.0},
    )
    assert entry.acted is False
    assert len(ledger) == 1
    assert entry.executed_notional == 0.0
    assert entry.intended_notional == 1_000.0


def test_a_decision_with_no_orders_still_occupies_a_sequence_number():
    ledger = CounterfactualLedger()
    ledger.record(
        decision={"orders": []},
        observation={},
        orders=[],
        settlement_prices={},
    )
    ledger.record(
        decision={"orders": []},
        observation={},
        orders=[_executed()],
        settlement_prices={"AAA": 100.0},
    )
    assert [entry.sequence for entry in ledger.records] == [0, 1]
    assert ledger.selection_gap().unacted_decisions == 1


def test_a_dropped_buy_into_a_rally_shows_a_positive_foregone_pnl():
    ledger = CounterfactualLedger()
    entry = ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped(quantity=10.0, price=100.0)],
        settlement_prices={"AAA": 105.0},
    )
    assert entry.intended_pnl == pytest.approx(50.0)
    assert entry.executed_pnl == 0.0
    assert entry.foregone_pnl == pytest.approx(50.0)


def test_a_dropped_buy_into_a_selloff_shows_a_loss_the_gate_avoided():
    ledger = CounterfactualLedger()
    entry = ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped(quantity=10.0, price=100.0)],
        settlement_prices={"AAA": 92.0},
    )
    assert entry.foregone_pnl == pytest.approx(-80.0)


def test_a_short_settles_with_the_opposite_sign():
    ledger = CounterfactualLedger()
    entry = ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped(side="sell", quantity=4.0, price=100.0)],
        settlement_prices={"AAA": 110.0},
    )
    assert entry.intended_pnl == pytest.approx(-40.0)


def test_an_unsettled_decision_reports_unavailable_pnl_not_a_zero_return():
    ledger = CounterfactualLedger()
    entry = ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped(quantity=3.0, price=50.0)],
        settlement_prices={},
    )
    assert entry.intended_pnl is None
    assert entry.foregone_pnl is None
    assert entry.executed_pnl == 0.0  # Known zero fills do not need a mark.
    assert ledger.selection_gap().intended_pnl is None
    assert entry.intended_notional == 150.0


def test_selection_gap_measures_what_the_execution_path_removed():
    ledger = CounterfactualLedger()
    ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_executed(quantity=2.0, price=100.0)],
        settlement_prices={"AAA": 101.0},
    )
    ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped(quantity=8.0, price=100.0)],
        settlement_prices={"AAA": 101.0},
    )
    gap = ledger.selection_gap()
    assert gap.decisions == 2
    assert gap.acted_decisions == 1
    assert gap.intended_notional == pytest.approx(1_000.0)
    assert gap.executed_notional == pytest.approx(200.0)
    assert gap.execution_ratio == pytest.approx(0.2)
    assert gap.foregone_pnl == pytest.approx(8.0)
    assert gap.executed_pnl == pytest.approx(2.0)
    assert gap.dispositions["executed"] == 1
    assert gap.dispositions["risk_refused"] == 1
    assert gap.dispositions["not_submitted"] == 0


def test_execution_ratio_is_one_when_nothing_was_dropped():
    ledger = CounterfactualLedger()
    ledger.record(
        decision={"orders": []},
        observation={},
        orders=[_executed()],
        settlement_prices={"AAA": 100.0},
    )
    assert ledger.selection_gap().execution_ratio == 1.0


def test_orders_behind_a_refusal_are_not_submitted_rather_than_refused():
    intended = [
        {"symbol": "AAA", "side": "buy", "quantity": 5.0},
        {"symbol": "BBB", "side": "buy", "quantity": 7.0},
        {"symbol": "AAA", "side": "sell", "quantity": 1.0},
    ]
    verdicts = [
        {"allowed": True, "reason": "allowed"},
        {"allowed": False, "reason": "gross-exposure-limit"},
    ]
    ghosts = ghost_orders_from_preflight(
        intended, verdicts, {"AAA": 100.0, "BBB": 50.0}
    )
    assert [ghost.disposition for ghost in ghosts] == [
        "not_submitted",
        "risk_refused",
        "not_submitted",
    ]
    assert ghosts[1].reason == "gross-exposure-limit"
    assert ghosts[2].reason == "batch halted by an earlier refusal"
    assert ghosts[2].foregone_notional == pytest.approx(100.0)
    assert all(ghost.executed_quantity == 0 for ghost in ghosts)


def test_the_ledger_persists_one_json_line_per_decision(tmp_path):
    path = tmp_path / "ghost.jsonl"
    ledger = CounterfactualLedger(path)
    ledger.record(
        decision={"orders": [{"symbol": "AAA"}]},
        observation={"AAA": 100.0},
        orders=[_dropped()],
        settlement_prices={"AAA": 103.0},
    )
    ledger.record(
        decision={"orders": []},
        observation={"AAA": 103.0},
        orders=[],
        settlement_prices={},
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["evidence_class"] == "counterfactual_decision_ledger"
    assert rows[0]["acted"] is False
    assert rows[0]["foregone_pnl"] == pytest.approx(30.0)
    assert rows[0]["orders"][0]["disposition"] == "risk_refused"


def test_acted_must_agree_with_what_reached_the_broker():
    with pytest.raises(CounterfactualError, match="acted must agree"):
        from sharpearena.counterfactual import CounterfactualRecord

        CounterfactualRecord(
            sequence=0,
            decision_sha256="x",
            observation_sha256="y",
            orders=(_dropped(),),
            settlement_prices={},
            acted=True,
        )


def test_tiny_positive_orders_do_not_turn_zero_fills_into_full_fills():
    assert _dropped(quantity=1e-15).executed_quantity == 0
    with pytest.raises(CounterfactualError, match="full intended"):
        GhostOrder("AAA", "buy", 1e-15, 0, 100, "executed", "invalid")


def test_unknown_fill_prevents_aggregate_availability_without_erasing_known_fills():
    from dataclasses import replace

    ledger = CounterfactualLedger()
    ledger.record(
        decision={},
        observation={},
        orders=[_executed()],
        settlement_prices={"AAA": 101},
    )
    ledger.record(
        decision={},
        observation={},
        orders=[
            replace(
                _dropped(), execution_complete=False, disposition="submission_unknown"
            )
        ],
        settlement_prices={"AAA": 101},
    )
    gap = ledger.selection_gap()
    assert (
        gap.decisions,
        gap.acted_decisions,
        gap.unacted_decisions,
        gap.unknown_decisions,
    ) == (2, 1, 0, 1)
    assert gap.confirmed_notional == 1000
    assert gap.executed_notional is None
    assert gap.executed_pnl is None
    assert gap.foregone_pnl is None
    assert gap.execution_ratio is None


def test_v1_ledger_is_refused_not_silently_upgraded(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"schema_version": 1}) + "\n")
    with pytest.raises(CounterfactualError, match="requires V2 receipts"):
        CounterfactualLedger(path)
    assert json.loads(path.read_text())["schema_version"] == 1


@pytest.mark.parametrize(
    "mutation", ["sequence", "pnl", "quantity", "truncated", "blank"]
)
def test_corrupt_persisted_snapshot_is_rejected_before_append(tmp_path, mutation):
    path = tmp_path / "ledger.jsonl"
    ledger = CounterfactualLedger(path)
    ledger.record(
        decision={},
        observation={},
        orders=[_executed()],
        settlement_prices={"AAA": 101},
    )
    row = json.loads(path.read_text())
    if mutation == "sequence":
        row["sequence"] = 3
    elif mutation == "pnl":
        row["executed_pnl"] = 1234
    elif mutation == "quantity":
        row["orders"][0]["executed_quantity"] = float("inf")
    payload = (
        "{"
        if mutation == "truncated"
        else "\n" if mutation == "blank" else json.dumps(row) + "\n"
    )
    path.write_text(payload)
    with pytest.raises(CounterfactualError, match="invalid counterfactual"):
        CounterfactualLedger(path)
    assert path.read_text() == payload


def test_revision_cannot_replace_intent_or_reduce_confirmed_fills():
    ledger = CounterfactualLedger()
    args = dict(
        decision={},
        observation={},
        decision_key="decision-1",
        settlement_prices={"AAA": 101},
    )
    ledger.record(**args, orders=[_executed()])
    with pytest.raises(CounterfactualError, match="frozen intent"):
        ledger.record(**args, orders=[_executed(quantity=9)])
    with pytest.raises(CounterfactualError, match="fills cannot decrease"):
        ledger.record(**args, orders=[_dropped()])
    assert len(ledger) == 1
    assert ledger.selection_gap().executed_notional == 1000


def test_failed_append_does_not_advance_memory_and_poisoned_writer_refuses_retry(
    tmp_path, monkeypatch
):
    import sharpearena.counterfactual as module

    path = tmp_path / "ledger.jsonl"
    ledger = CounterfactualLedger(path)

    def fail(_):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(module.os, "fsync", fail)
    args = dict(
        decision={},
        observation={},
        orders=[_executed()],
        settlement_prices={"AAA": 101},
    )
    with pytest.raises(OSError, match="injected fsync"):
        ledger.record(**args)
    assert len(ledger) == 0
    with pytest.raises(CounterfactualError, match="reopen and validate"):
        ledger.record(**args)


def test_settlement_snapshot_is_immutable_and_nonfinite_input_cannot_be_serialized():
    ledger = CounterfactualLedger()
    prices = {"AAA": 101}
    entry = ledger.record(
        decision={}, observation={}, orders=[_executed()], settlement_prices=prices
    )
    prices["AAA"] = 999
    assert entry.executed_pnl == 10
    with pytest.raises(TypeError):
        entry.settlement_prices["AAA"] = 999
    with pytest.raises(ValueError):
        ledger.record(
            decision={"x": float("nan")},
            observation={},
            orders=[],
            settlement_prices={},
        )
    assert len(ledger) == 1


@pytest.mark.parametrize("mutation", ["missing-newline", "duplicate-key"])
def test_incomplete_delimiter_and_ambiguous_json_cannot_be_appended_to(
    tmp_path, mutation
):
    path = tmp_path / "ledger.jsonl"
    ledger = CounterfactualLedger(path)
    ledger.record(decision={}, observation={}, orders=[], settlement_prices={})
    payload = path.read_text()
    payload = (
        payload.rstrip("\n")
        if mutation == "missing-newline"
        else payload.replace('"acted":false', '"acted":true,"acted":false')
    )
    path.write_text(payload)
    with pytest.raises(CounterfactualError, match="newline|duplicate JSON key"):
        CounterfactualLedger(path)
    assert path.read_text() == payload
