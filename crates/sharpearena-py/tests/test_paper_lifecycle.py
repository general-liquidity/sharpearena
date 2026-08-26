"""Order lifecycle, reconciliation, account refresh, and the remote-submit gate.

Every test here runs against an injected fixture. No test opens a socket, and no
code path under test can reach a real-capital endpoint.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest
from sharpearena.counterfactual import CounterfactualLedger
from sharpearena.paper_cli import main as paper_cli_main
from sharpearena.paper_trading import (
    STATE_FILLED,
    STATE_PREPARED,
    STATE_RECONCILED_ABSENT,
    STATE_SUBMISSION_UNKNOWN,
    AccountSnapshot,
    AlpacaPaperBroker,
    ForwardEvidenceJournal,
    ForwardWindowIdentity,
    InMemoryPaperBroker,
    LifecycleError,
    LifecycleStore,
    MarketBar,
    OrderLifecycle,
    PaperAccount,
    PaperOrder,
    PaperRiskConfig,
    PaperRiskGuard,
    PaperTradingError,
    PaperTradingSession,
    SubmissionUnknown,
    bind_forward_window,
    prepare_forward_window_commitment,
    reconcile_account,
    verify_forward_evidence_window,
)


def _account(**overrides):
    values = {
        "cash": 10_000.0,
        "equity": 10_000.0,
        "session_start_equity": 10_000.0,
        "peak_equity": 10_000.0,
        "positions": {},
    }
    values.update(overrides)
    return PaperAccount(**values)


def _guard(**overrides):
    values = {
        "allowed_symbols": ("AAA",),
        "max_order_notional": 5_000.0,
        "max_gross_exposure": 0.9,
    }
    values.update(overrides)
    return PaperRiskGuard(PaperRiskConfig(**values))


def _bar(symbol="AAA", close=100.0):
    return MarketBar(
        symbol, "2026-08-26T10:00:00Z", 99.0, 101.0, 98.0, close, 1_000.0, "fixture"
    )


DECISION = {
    "orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.2}],
    "reasoning": "fixture",
}


class ScriptedBroker:
    """A paper broker whose submit outcome and query answer are dictated."""

    broker_id = "scripted-paper"

    def __init__(self, risk, *, outcomes=(), found=None):
        self.risk = risk
        self.outcomes = list(outcomes)
        self.found = found
        self.submits = []
        self.queries = []

    def submit(self, order, *, account, prices):
        self.submits.append(order.client_order_id)
        outcome = self.outcomes.pop(0) if self.outcomes else "filled"
        if outcome == "unknown":
            raise SubmissionUnknown("connection reset before any verdict")
        return {
            "id": f"scripted-{len(self.submits)}",
            "status": "filled",
            "filled_qty": order.quantity,
            "client_order_id": order.client_order_id,
        }

    def find_order(self, client_order_id):
        self.queries.append(client_order_id)
        return self.found


class UnqueryableBroker:
    """A broker with no order-query surface at all."""

    broker_id = "unqueryable-paper"

    def __init__(self, risk):
        self.risk = risk
        self.submits = []

    def submit(self, order, *, account, prices):
        self.submits.append(order.client_order_id)
        raise SubmissionUnknown("connection reset before any verdict")


class StubAccountSource:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.calls = 0

    def account_snapshot(self):
        self.calls += 1
        return self.snapshot


# -- the state machine itself -------------------------------------------------


def test_illegal_transitions_are_refused():
    lifecycle = OrderLifecycle("cid-1")
    with pytest.raises(LifecycleError, match="prepared -> filled"):
        lifecycle.mark_filled(1.0)
    lifecycle.mark_submitted()
    lifecycle.mark_acknowledged({"status": "new"})
    lifecycle.mark_filled(1.0)
    with pytest.raises(LifecycleError, match="filled -> "):
        lifecycle.mark_rejected("too late")


def test_submission_unknown_is_a_state_and_not_a_rejection():
    lifecycle = OrderLifecycle("cid-2")
    lifecycle.mark_submitted()
    lifecycle.mark_submission_unknown("timeout")
    assert lifecycle.state == STATE_SUBMISSION_UNKNOWN
    assert lifecycle.state != "rejected"
    assert lifecycle.awaiting_reconciliation is True
    assert lifecycle.may_submit_replacement is False
    assert lifecycle.is_terminal is False


def test_a_replacement_is_permitted_only_after_the_broker_reports_absence():
    lifecycle = OrderLifecycle("cid-3")
    lifecycle.mark_submitted()
    assert lifecycle.may_submit_replacement is False
    lifecycle.mark_submission_unknown("timeout")
    assert lifecycle.may_submit_replacement is False
    lifecycle.mark_reconciled_absent()
    assert lifecycle.may_submit_replacement is True

    resting = OrderLifecycle("cid-4")
    resting.mark_submitted()
    resting.mark_submission_unknown("timeout")
    resting.mark_reconciled_accepted({"status": "new"})
    assert resting.may_submit_replacement is False


def test_ack_latency_is_measured_and_is_none_when_nothing_came_back():
    unresolved = OrderLifecycle("cid-5")
    unresolved.mark_submitted(at_unix_ns=1_000)
    assert unresolved.ack_latency_ns is None
    unresolved.mark_submission_unknown("timeout", at_unix_ns=2_000)
    assert unresolved.ack_latency_ns is None

    resolved = OrderLifecycle("cid-6")
    resolved.mark_submitted(at_unix_ns=1_000)
    resolved.mark_acknowledged({"status": "new"}, at_unix_ns=5_500)
    assert resolved.ack_latency_ns == 4_500


def test_every_transition_and_broker_ack_hash_is_recorded():
    lifecycle = OrderLifecycle("cid-7")
    lifecycle.mark_submitted(at_unix_ns=1)
    lifecycle.mark_acknowledged({"status": "new", "id": "x"}, at_unix_ns=2)
    lifecycle.mark_partially_filled(3.0, {"status": "partially_filled"}, at_unix_ns=3)
    lifecycle.mark_filled(5.0, {"status": "filled"}, at_unix_ns=4)
    record = lifecycle.as_record()
    assert [item["to_state"] for item in record["transitions"]] == [
        "submitted",
        "acknowledged",
        "partially_filled",
        "filled",
    ]
    assert record["transitions"][0]["broker_ack_sha256"] is None
    hashes = [
        item["broker_ack_sha256"] for item in record["transitions"][1:]
    ]
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes)
    assert len(set(hashes)) == 3
    assert record["filled_quantity"] == 5.0


def test_lifecycle_records_round_trip():
    lifecycle = OrderLifecycle("cid-8")
    lifecycle.mark_submitted(at_unix_ns=10)
    lifecycle.mark_submission_unknown("timeout", at_unix_ns=20)
    restored = OrderLifecycle.from_record(lifecycle.as_record())
    assert restored.as_record() == lifecycle.as_record()
    assert restored.awaiting_reconciliation is True


# -- reconciliation -----------------------------------------------------------


def _session(broker, tmp_path, **overrides):
    kwargs = {
        "agent_id": "local-model",
        "model_digest": "sha256:model",
        "lifecycles": LifecycleStore(tmp_path / "lifecycles.json"),
    }
    kwargs.update(overrides)
    return PaperTradingSession(
        broker, ForwardEvidenceJournal(tmp_path / "forward.jsonl"), **kwargs
    )


def test_an_unknown_submission_queries_before_it_ever_replaces(tmp_path):
    broker = ScriptedBroker(
        _guard(), outcomes=["unknown"], found={"status": "new", "id": "resting"}
    )
    session = _session(broker, tmp_path)
    with pytest.raises(SubmissionUnknown):
        session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    assert len(broker.submits) == 1
    assert broker.queries == broker.submits

    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    assert len(broker.submits) == 1


def test_a_confirmed_absence_authorizes_exactly_one_replacement(tmp_path):
    broker = ScriptedBroker(_guard(), outcomes=["unknown", "filled"], found=None)
    session = _session(broker, tmp_path)
    with pytest.raises(SubmissionUnknown):
        session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    original = broker.submits[0]
    assert broker.queries == [original]

    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    assert broker.submits == [original, f"{original}-r1"]


def test_reconciliation_is_idempotent_across_a_restart(tmp_path):
    state = tmp_path / "lifecycles.json"
    broker = ScriptedBroker(
        _guard(), outcomes=["unknown"], found={"status": "new", "id": "resting"}
    )
    first = _session(broker, tmp_path, lifecycles=LifecycleStore(state))
    with pytest.raises(SubmissionUnknown):
        first.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    queries_after_first = len(broker.queries)

    reloaded = LifecycleStore(state)
    assert [item.client_order_id for item in reloaded.unresolved()] == []
    second = _session(broker, tmp_path, lifecycles=reloaded)
    assert second.reconcile_all() == []
    second.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    assert len(broker.submits) == 1
    assert len(broker.queries) == queries_after_first


def test_an_unresolved_order_survives_a_crash_before_the_query(tmp_path):
    state = tmp_path / "lifecycles.json"
    store = LifecycleStore(state)
    lifecycle = store.open_or_create("cid-crash")
    lifecycle.mark_submitted()
    lifecycle.mark_submission_unknown("process died before the query")
    store.save()

    reloaded = LifecycleStore(state)
    unresolved = reloaded.unresolved()
    assert [item.client_order_id for item in unresolved] == ["cid-crash"]
    assert unresolved[0].may_submit_replacement is False

    broker = ScriptedBroker(_guard(), found=None)
    session = _session(broker, tmp_path, lifecycles=reloaded)
    session.reconcile_all()
    assert broker.queries == ["cid-crash"]
    assert reloaded.get("cid-crash").state == STATE_RECONCILED_ABSENT


def test_an_unqueryable_broker_can_never_resolve_an_unknown_submission(tmp_path):
    broker = UnqueryableBroker(_guard())
    session = _session(broker, tmp_path)
    with pytest.raises(PaperTradingError, match="cannot be queried"):
        session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    assert len(broker.submits) == 1


def test_transitions_and_ack_hashes_reach_the_forward_evidence(tmp_path):
    broker = ScriptedBroker(
        _guard(), outcomes=["unknown"], found={"status": "filled", "filled_qty": 20.0}
    )
    session = _session(broker, tmp_path)
    with pytest.raises(SubmissionUnknown):
        session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    evidence = [
        json.loads(line)
        for line in (tmp_path / "forward.jsonl").read_text().splitlines()
    ]
    kinds = [record["record_type"] for record in evidence]
    assert kinds == ["decision", "order-reconciliation", "order-submission"]
    reconciliation = evidence[1]
    assert reconciliation["queried_by"] == "client_order_id"
    assert reconciliation["broker_answer"] == "present"
    assert len(reconciliation["broker_answer_sha256"]) == 64
    states = [
        item["to_state"] for item in reconciliation["lifecycle"]["transitions"]
    ]
    assert states == [
        "submitted",
        "submission_unknown",
        "reconciled_accepted",
        "filled",
    ]
    assert evidence[2]["submission_status"] == "submission-unknown-then-reconciled"
    assert all(record["deterministic"] is False for record in evidence)


def test_an_http_client_error_is_a_rejection_and_a_server_error_is_unknown():
    class RaisingTransport:
        def __init__(self, error):
            self.error = error

        def request(self, method, url, *, headers=None, payload=None):
            raise self.error

    order = PaperOrder("AAA", "buy", 1.0)
    rejected = AlpacaPaperBroker(
        "key",
        "secret",
        _guard(),
        transport=RaisingTransport(
            HTTPError("https://paper-api.alpaca.markets", 422, "bad", {}, None)
        ),
    )
    with pytest.raises(PaperTradingError, match="rejected the order with HTTP 422"):
        rejected.submit(order, account=_account(), prices={"AAA": 100.0})

    unknown = AlpacaPaperBroker(
        "key",
        "secret",
        _guard(),
        transport=RaisingTransport(TimeoutError("read timed out")),
    )
    with pytest.raises(SubmissionUnknown, match="no response from the paper broker"):
        unknown.submit(order, account=_account(), prices={"AAA": 100.0})


def test_a_404_order_query_means_absent_and_a_500_leaves_it_unknown():
    class QueryTransport:
        def __init__(self, error):
            self.error = error

        def request(self, method, url, *, headers=None, payload=None):
            raise self.error

    absent = AlpacaPaperBroker(
        "key",
        "secret",
        _guard(),
        transport=QueryTransport(
            HTTPError("https://paper-api.alpaca.markets", 404, "nf", {}, None)
        ),
    )
    assert absent.find_order("cid") is None

    unclear = AlpacaPaperBroker(
        "key",
        "secret",
        _guard(),
        transport=QueryTransport(
            HTTPError("https://paper-api.alpaca.markets", 503, "down", {}, None)
        ),
    )
    with pytest.raises(SubmissionUnknown, match="order query failed with HTTP 503"):
        unclear.find_order("cid")


def test_the_in_memory_broker_never_double_fills_one_client_order_id():
    broker = InMemoryPaperBroker(_guard())
    account = _account()
    order = PaperOrder("AAA", "buy", 5.0, client_order_id="cid-dupe")
    first = broker.submit(order, account=account, prices={"AAA": 100.0})
    second = broker.submit(order, account=account, prices={"AAA": 100.0})
    assert first == second
    assert account.positions["AAA"] == 5.0
    assert broker.find_order("cid-dupe") == first
    assert broker.find_order("cid-never") is None


# -- account reconciliation (2.6) --------------------------------------------


def test_reconcile_account_replaces_the_book_and_keeps_the_session_anchor():
    account = _account(equity=10_000.0, cash=10_000.0)
    drift = reconcile_account(
        account,
        AccountSnapshot(
            cash=4_000.0,
            equity=9_400.0,
            positions={"AAA": 54.0, "BBB": 0.0},
            source="alpaca-paper-v2",
        ),
    )
    assert account.cash == 4_000.0
    assert account.equity == 9_400.0
    assert account.positions == {"AAA": 54.0}
    assert account.session_start_equity == 10_000.0
    assert account.peak_equity == 10_000.0
    assert drift["equity_drift"] == pytest.approx(-600.0)
    assert drift["position_drift"] == {"AAA": 54.0}
    assert drift["source"] == "alpaca-paper-v2"


def test_peak_equity_ratchets_upward_on_reconciliation():
    account = _account()
    reconcile_account(
        account,
        AccountSnapshot(
            cash=1_000.0, equity=12_000.0, positions={}, source="alpaca-paper-v2"
        ),
    )
    assert account.peak_equity == 12_000.0
    assert account.session_start_equity == 10_000.0


def test_a_stale_plan_account_would_have_passed_a_limit_the_real_book_fails(tmp_path):
    stale = _account()
    broker = ScriptedBroker(_guard())
    unreconciled = _session(broker, tmp_path)
    unreconciled.execute_decision(DECISION, ["AAA"], stale, {"AAA": _bar()})
    assert len(broker.submits) == 1

    fresh = _account()
    source = StubAccountSource(
        AccountSnapshot(cash=5_000.0, equity=5_000.0, positions={}, source="broker")
    )
    reconciling = _session(
        ScriptedBroker(_guard()),
        tmp_path / "second",
        account_source=source,
    )
    with pytest.raises(PaperTradingError, match="daily-loss-limit"):
        reconciling.execute_decision(DECISION, ["AAA"], fresh, {"AAA": _bar()})
    assert source.calls == 1
    assert fresh.equity == 5_000.0


def test_the_account_reconciliation_is_written_to_the_forward_evidence(tmp_path):
    account = _account()
    source = StubAccountSource(
        AccountSnapshot(
            cash=6_000.0, equity=9_900.0, positions={"AAA": 39.0}, source="broker"
        )
    )
    session = _session(ScriptedBroker(_guard()), tmp_path, account_source=source)
    session.execute_decision(DECISION, ["AAA"], account, {"AAA": _bar()})
    evidence = [
        json.loads(line)
        for line in (tmp_path / "forward.jsonl").read_text().splitlines()
    ]
    assert evidence[0]["record_type"] == "account-reconciliation"
    assert evidence[0]["reconciliation"]["equity_after"] == 9_900.0
    assert evidence[0]["reconciliation"]["session_start_equity_preserved"] == 10_000.0


# -- forward window wiring (2.4 / D5) ----------------------------------------


def _window(tmp_path, agent_id="local-model"):
    prepared = prepare_forward_window_commitment(
        agent_id,
        "window-004",
        {"model_digest": "sha256:model", "scaffold": "minimal-stateless-v1"},
        "private-salt",
        commitment_path=tmp_path / "commitment.json",
        private_preimage_path=tmp_path / "private" / "preimage.json",
    )
    return bind_forward_window(prepared)


def test_the_pre_committed_window_is_stamped_on_every_forward_record(tmp_path):
    window = _window(tmp_path)
    session = _session(ScriptedBroker(_guard()), tmp_path, window=window)
    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    path = tmp_path / "forward.jsonl"
    assert verify_forward_evidence_window(path, window) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(
        record["forward_window"]["commit_hash"] == window.commit_hash
        for record in records
    )


def test_forward_evidence_never_claims_the_determinism_a_commitment_might_imply(
    tmp_path,
):
    window = _window(tmp_path)
    session = _session(ScriptedBroker(_guard()), tmp_path, window=window)
    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    records = [
        json.loads(line)
        for line in (tmp_path / "forward.jsonl").read_text().splitlines()
    ]
    for record in records:
        assert record["evidence_class"] == "forward_paper_trading"
        assert record["deterministic"] is False
        assert record["forward_window"]["deterministic"] is False
        assert record["replay_guarantee"] == "none-live-market-and-paper-broker-state"


def test_evidence_from_another_commitment_is_rejected(tmp_path):
    window = _window(tmp_path)
    other = ForwardWindowIdentity(
        agent_id="local-model",
        target_window="window-004",
        commit_hash="0" * 64,
        artifact_digest=window.artifact_digest,
    )
    session = _session(ScriptedBroker(_guard()), tmp_path, window=window)
    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    with pytest.raises(PaperTradingError, match="different commitment"):
        verify_forward_evidence_window(tmp_path / "forward.jsonl", other)


def test_unstamped_evidence_does_not_pass_as_committed(tmp_path):
    session = _session(ScriptedBroker(_guard()), tmp_path)
    session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    with pytest.raises(PaperTradingError, match="carries no forward-window"):
        verify_forward_evidence_window(tmp_path / "forward.jsonl", _window(tmp_path))


def test_a_session_refuses_a_commitment_made_by_a_different_agent(tmp_path):
    with pytest.raises(ValueError, match="different agent"):
        _session(
            ScriptedBroker(_guard()), tmp_path, window=_window(tmp_path, "someone-else")
        )


# -- counterfactual wiring (3.4) ---------------------------------------------


def test_a_refused_batch_still_produces_a_counterfactual_record(tmp_path):
    ledger = CounterfactualLedger()
    session = _session(
        ScriptedBroker(_guard(max_order_notional=100.0)),
        tmp_path,
        counterfactual=ledger,
    )
    with pytest.raises(PaperTradingError, match="order-notional-limit"):
        session.execute_decision(DECISION, ["AAA"], _account(), {"AAA": _bar()})
    gap = ledger.selection_gap()
    assert gap.decisions == 1
    assert gap.acted_decisions == 0
    assert gap.intended_notional == pytest.approx(2_000.0)
    assert gap.executed_notional == 0.0
    assert gap.dispositions["risk_refused"] == 1


def test_an_executed_batch_closes_the_selection_gap(tmp_path):
    ledger = CounterfactualLedger()
    session = _session(ScriptedBroker(_guard()), tmp_path, counterfactual=ledger)
    session.execute_decision(
        DECISION,
        ["AAA"],
        _account(),
        {"AAA": _bar()},
        settlement_prices={"AAA": 104.0},
    )
    gap = ledger.selection_gap()
    assert gap.execution_ratio == pytest.approx(1.0)
    assert gap.foregone_pnl == pytest.approx(0.0)
    assert gap.executed_pnl == pytest.approx(80.0)


# -- the remote-submit gate ---------------------------------------------------


def _cli_plan(tmp_path, provider):
    payload = {
        "agent_id": "local-model",
        "model_digest": "sha256:model",
        "symbols": ["AAA"],
        "market_data": {"provider": "binance-public", "interval": "1h", "limit": 2},
        "broker": {"provider": provider},
        "account": {
            "cash": 10_000,
            "equity": 10_000,
            "session_start_equity": 10_000,
            "peak_equity": 10_000,
            "positions": {},
        },
        "risk": {"allowed_symbols": ["AAA"], "max_order_notional": 5_000},
    }
    plan = tmp_path / "plan.json"
    decision = tmp_path / "decision.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    decision.write_text(json.dumps(DECISION), encoding="utf-8")
    return plan, decision


class StubMarketData:
    def __init__(self, *args, **kwargs):
        pass

    def recent_bars(self, symbol, *, limit=120):
        return [_bar(symbol)]


def test_remote_paper_submission_is_refused_without_the_explicit_flag(
    tmp_path, monkeypatch
):
    plan, decision = _cli_plan(tmp_path, "alpaca-paper")
    monkeypatch.setattr("sharpearena.paper_cli.BinancePublicData", StubMarketData)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    argv = [
        "execute",
        "--plan",
        str(plan),
        "--decision",
        str(decision),
        "--evidence",
        str(tmp_path / "forward.jsonl"),
    ]
    with pytest.raises(ValueError, match="requires --allow-remote-paper-submit"):
        paper_cli_main(argv)
    assert not (tmp_path / "forward.jsonl").exists()


def test_the_flag_admits_the_paper_broker_and_reconciles_its_account(
    tmp_path, monkeypatch
):
    plan, decision = _cli_plan(tmp_path, "alpaca-paper")
    constructed = {}

    class StubPaperBroker(ScriptedBroker):
        broker_id = "stub-paper"

        def __init__(self, key_id, secret_key, risk, **kwargs):
            super().__init__(risk)
            constructed["credentials_seen"] = (key_id, secret_key)

        def account_snapshot(self):
            constructed["snapshots"] = constructed.get("snapshots", 0) + 1
            return AccountSnapshot(
                cash=8_000.0, equity=9_950.0, positions={}, source=self.broker_id
            )

    monkeypatch.setattr("sharpearena.paper_cli.BinancePublicData", StubMarketData)
    monkeypatch.setattr("sharpearena.paper_cli.AlpacaPaperBroker", StubPaperBroker)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    assert (
        paper_cli_main(
            [
                "execute",
                "--plan",
                str(plan),
                "--decision",
                str(decision),
                "--evidence",
                str(tmp_path / "forward.jsonl"),
                "--allow-remote-paper-submit",
                "--lifecycle-state",
                str(tmp_path / "lifecycles.json"),
            ]
        )
        == 0
    )
    assert constructed["snapshots"] == 1
    evidence = [
        json.loads(line)
        for line in (tmp_path / "forward.jsonl").read_text().splitlines()
    ]
    assert evidence[0]["record_type"] == "account-reconciliation"
    assert evidence[0]["reconciliation"]["equity_after"] == 9_950.0
    assert (tmp_path / "lifecycles.json").exists()


def test_the_in_memory_path_needs_no_flag_and_touches_no_credentials(
    tmp_path, monkeypatch
):
    plan, decision = _cli_plan(tmp_path, "in-memory")
    monkeypatch.setattr("sharpearena.paper_cli.BinancePublicData", StubMarketData)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    assert (
        paper_cli_main(
            [
                "execute",
                "--plan",
                str(plan),
                "--decision",
                str(decision),
                "--evidence",
                str(tmp_path / "forward.jsonl"),
            ]
        )
        == 0
    )
    evidence = [
        json.loads(line)
        for line in (tmp_path / "forward.jsonl").read_text().splitlines()
    ]
    assert [record["record_type"] for record in evidence] == [
        "decision",
        "order-submission",
    ]
    assert evidence[1]["lifecycle"]["state"] == STATE_FILLED


def test_the_cli_stamps_the_window_from_the_private_preimage(tmp_path, monkeypatch):
    plan, decision = _cli_plan(tmp_path, "in-memory")
    prepared = prepare_forward_window_commitment(
        "local-model",
        "window-004",
        {"model_digest": "sha256:model"},
        "private-salt",
        commitment_path=tmp_path / "commitment.json",
        private_preimage_path=tmp_path / "preimage.json",
    )
    monkeypatch.setattr("sharpearena.paper_cli.BinancePublicData", StubMarketData)
    assert (
        paper_cli_main(
            [
                "execute",
                "--plan",
                str(plan),
                "--decision",
                str(decision),
                "--evidence",
                str(tmp_path / "forward.jsonl"),
                "--window-preimage",
                str(tmp_path / "preimage.json"),
            ]
        )
        == 0
    )
    assert (
        verify_forward_evidence_window(
            tmp_path / "forward.jsonl", bind_forward_window(prepared)
        )
        == 2
    )


def test_a_prepared_lifecycle_is_the_only_submittable_state(tmp_path):
    store = LifecycleStore(tmp_path / "lifecycles.json")
    lifecycle = store.open_or_create("cid-9")
    assert lifecycle.state == STATE_PREPARED
    assert store.open_or_create("cid-9") is lifecycle
    assert store.get("cid-absent") is None
