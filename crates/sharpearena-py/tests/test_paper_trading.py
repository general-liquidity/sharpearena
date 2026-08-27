"""Forward paper-trading safety tests; all network calls use an injected fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sharpearena.paper_cli import load_execution_plan
from sharpearena.paper_cli import main as paper_cli_main
from sharpearena.paper_trading import (
    ALPACA_PAPER_ORIGIN,
    AlpacaMarketData,
    AlpacaPaperBroker,
    BinancePublicData,
    ForwardEvidenceJournal,
    InMemoryPaperBroker,
    MarketBar,
    PaperAccount,
    PaperOrder,
    PaperRiskConfig,
    PaperRiskGuard,
    PaperTradingError,
    PaperTradingSession,
    make_forward_commitment,
    prepare_forward_window_commitment,
    prepare_forward_window_reveal,
    target_weights_to_orders,
)


class FixtureTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, *, headers=None, payload=None):
        self.calls.append((method, url, dict(headers or {}), payload))
        return self.response


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
        "allowed_symbols": ("AAA", "BTCUSDT"),
        "max_order_notional": 2_500.0,
        "max_gross_exposure": 0.5,
    }
    values.update(overrides)
    return PaperRiskGuard(PaperRiskConfig(**values))


def test_market_data_adapters_are_read_only_and_parse_bars():
    binance_transport = FixtureTransport(
        [[1_700_000_000_000, "100", "110", "90", "105", "123.4"]]
    )
    bars = BinancePublicData("1h", transport=binance_transport).recent_bars(
        "BTC/USDT", limit=1
    )
    assert bars[0].close == 105.0
    assert bars[0].open == 100.0
    assert bars[0].high == 110.0
    assert bars[0].low == 90.0
    assert bars[0].volume == 123.4
    assert binance_transport.calls[0][0] == "GET"
    assert binance_transport.calls[0][1].startswith(
        "https://api.binance.com/api/v3/klines?"
    )
    assert "symbol=BTCUSDT" in binance_transport.calls[0][1]
    assert "interval=1h" in binance_transport.calls[0][1]
    assert "limit=1" in binance_transport.calls[0][1]
    assert binance_transport.calls[0][3] is None

    alpaca_transport = FixtureTransport(
        {
            "bars": [
                {
                    "t": "2026-08-25T20:00:00Z",
                    "o": 10,
                    "h": 12,
                    "l": 9,
                    "c": 11,
                    "v": 500,
                }
            ]
        }
    )
    bars = AlpacaMarketData("key", "secret", transport=alpaca_transport).recent_bars(
        "AAA", limit=1
    )
    assert bars[0].source == "alpaca-market-data"
    assert (bars[0].open, bars[0].high, bars[0].low, bars[0].close, bars[0].volume) == (
        10.0,
        12.0,
        9.0,
        11.0,
        500.0,
    )
    method, url, headers, payload = alpaca_transport.calls[0]
    assert method == "GET" and url.startswith("https://data.alpaca.markets/")
    assert headers["APCA-API-KEY-ID"] == "key" and payload is None


@pytest.mark.parametrize("payload", [{}, {"bars": "wrong"}, {"bars": [{}]}])
def test_alpaca_market_data_refuses_malformed_provider_payloads(payload):
    source = AlpacaMarketData("key", "secret", transport=FixtureTransport(payload))
    with pytest.raises(PaperTradingError, match="bars array|malformed bar"):
        source.recent_bars("AAA", limit=1)


def test_alpaca_adapter_cannot_be_pointed_at_a_live_broker():
    with pytest.raises(ValueError, match="paper-api"):
        AlpacaPaperBroker(
            "key",
            "secret",
            _guard(),
            base_url="https://api.alpaca.markets",
            transport=FixtureTransport({}),
        )


def test_native_risk_guard_denies_before_broker_submission():
    guard = _guard()
    account = _account()
    assert guard.assess(PaperOrder("AAA", "buy", 10), account, {"AAA": 100}).allowed
    verdict = guard.assess(PaperOrder("AAA", "buy", 30), account, {"AAA": 100})
    assert not verdict.allowed and verdict.reason == "order-notional-limit"
    verdict = guard.assess(PaperOrder("ZZZ", "buy", 1), account, {"ZZZ": 10})
    assert not verdict.allowed and verdict.reason == "symbol-not-allowlisted"
    stopped = _guard(kill_switch=True)
    assert (
        stopped.assess(PaperOrder("AAA", "buy", 1), account, {"AAA": 100}).reason
        == "kill-switch-active"
    )


def test_remote_paper_order_uses_only_the_paper_origin_and_risk_verdict():
    # A broker that echoes the caller's credentials back in its reply. Asserting that a
    # secret is absent from a response built out of six named keys proves nothing on its
    # own, so the fixture puts the secret on the wire and the assertion below is that the
    # returned object's key set is exactly the closed one.
    api_secret = "super-secret-key-material"
    transport = FixtureTransport(
        {
            "id": "paper-1",
            "status": "accepted",
            "symbol": "AAA",
            "side": "buy",
            "qty": "2",
            "echoed_request": {"APCA-API-SECRET-KEY": api_secret},
        }
    )
    broker = AlpacaPaperBroker("key", api_secret, _guard(), transport=transport)
    response = broker.submit(
        PaperOrder("AAA", "buy", 2, client_order_id="fixed"),
        account=_account(),
        prices={"AAA": 100.0},
    )
    method, url, headers, payload = transport.calls[0]
    assert method == "POST" and url == f"{ALPACA_PAPER_ORIGIN}/v2/orders"
    assert payload["client_order_id"] == "fixed"
    assert response["risk"]["allowed"] is True
    assert api_secret in json.dumps(headers), "the secret must be on the wire"
    assert set(response) == {
        "id",
        "status",
        "symbol",
        "side",
        "submitted_qty",
        "filled_qty",
        "client_order_id",
        "risk",
    }
    assert api_secret not in json.dumps(response)


def test_target_weights_and_session_write_nonreplayable_forward_evidence(tmp_path):
    account = _account()
    prices = {"AAA": 100.0}
    decision = {
        "orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.2}],
        "reasoning": "fixture",
    }
    orders = target_weights_to_orders(decision, ["AAA"], account, prices)
    assert len(orders) == 1 and orders[0].quantity == pytest.approx(20.0)

    path = tmp_path / "forward.jsonl"
    session = PaperTradingSession(
        InMemoryPaperBroker(_guard()),
        ForwardEvidenceJournal(path),
        agent_id="local-model",
        model_digest="sha256:model",
    )
    responses = session.execute_decision(
        decision,
        ["AAA"],
        account,
        {
            "AAA": MarketBar(
                "AAA",
                "2026-08-25T20:00:00Z",
                99.0,
                101.0,
                98.0,
                100.0,
                1000.0,
                "fixture",
            )
        },
    )
    assert responses[0]["status"] == "filled"
    evidence = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["record_type"] for record in evidence] == [
        "decision",
        "order-submission",
    ]
    assert all(
        record["evidence_class"] == "forward_paper_trading" for record in evidence
    )
    assert all(record["deterministic"] is False for record in evidence)
    assert all(
        record["replay_guarantee"] == "none-live-market-and-paper-broker-state"
        for record in evidence
    )
    assert evidence[0]["model_digest"] == "sha256:model"
    assert evidence[0]["decision"]["reasoning"] == "fixture"
    assert evidence[0]["decision"]["orders"][0]["target_weight"] == 0.2
    assert evidence[0]["market_snapshot"]["AAA"]["close"] == 100.0


def test_paper_translation_preserves_sparse_hold_semantics():
    account = _account(positions={"AAA": 10.0, "BBB": 5.0})
    prices = {"AAA": 100.0, "BBB": 200.0}
    assert target_weights_to_orders(
        {"orders": []}, ["AAA", "BBB"], account, prices
    ) == []
    orders = target_weights_to_orders(
        {
            "orders": [
                {"symbol": "AAA", "action": "buy", "target_weight": 0.2}
            ]
        },
        ["AAA", "BBB"],
        account,
        prices,
    )
    assert [order.symbol for order in orders] == ["AAA"]


def test_refused_order_never_reaches_transport():
    transport = FixtureTransport({"id": "should-not-exist"})
    broker = AlpacaPaperBroker("key", "secret", _guard(), transport=transport)
    with pytest.raises(PaperTradingError, match="order-notional-limit"):
        broker.submit(
            PaperOrder("AAA", "buy", 100),
            account=_account(),
            prices={"AAA": 100.0},
        )
    assert transport.calls == []


def test_batch_preflight_blocks_aggregate_exposure_before_any_remote_order(tmp_path):
    transport = FixtureTransport({"id": "must-not-submit"})
    guard = _guard(
        allowed_symbols=("AAA", "BBB"),
        max_order_notional=5_000.0,
        max_gross_exposure=0.5,
    )
    broker = AlpacaPaperBroker("key", "secret", guard, transport=transport)
    session = PaperTradingSession(
        broker,
        ForwardEvidenceJournal(tmp_path / "forward.jsonl"),
        agent_id="local-model",
        model_digest="sha256:model",
    )
    decision = {
        "orders": [
            {"symbol": "AAA", "action": "buy", "target_weight": 0.3},
            {"symbol": "BBB", "action": "buy", "target_weight": 0.3},
        ]
    }
    bars = {
        symbol: MarketBar(
            symbol, "2026-08-26T10:00:00Z", 100, 101, 99, 100, 1_000, "fixture"
        )
        for symbol in ("AAA", "BBB")
    }
    with pytest.raises(PaperTradingError, match="gross-exposure-limit"):
        session.execute_decision(decision, ["AAA", "BBB"], _account(), bars)
    assert transport.calls == []
    evidence = json.loads((tmp_path / "forward.jsonl").read_text())
    assert evidence["record_type"] == "decision"
    assert evidence["submission_status"] == "risk-refused-before-submission"
    assert evidence["refusal_reason"] == "gross-exposure-limit"


def test_risk_refuses_unpriced_existing_positions_and_invalid_target_prices():
    account = _account(positions={"BBB": 2.0})
    verdict = _guard(allowed_symbols=("AAA", "BBB")).assess(
        PaperOrder("AAA", "buy", 1), account, {"AAA": 100.0}
    )
    assert verdict.reason == "missing-existing-position-price"
    with pytest.raises(PaperTradingError, match="invalid price"):
        target_weights_to_orders(
            {"orders": [{"symbol": "AAA", "action": "buy", "target_weight": 0.1}]},
            ["AAA"],
            _account(),
            {"AAA": 0.0},
        )


def test_forward_commitment_matches_the_rust_wire_primitive():
    fixture_path = (
        Path(__file__).parents[2]
        / "sharpearena"
        / "contract"
        / "attestation"
        / "forward-commitment.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    commitment = make_forward_commitment(
        fixture["agent_id"],
        fixture["target_window"],
        fixture["artifact_digest"],
        fixture["salt"],
    )
    assert commitment == {
        "agent_id": fixture["agent_id"],
        "target_window": fixture["target_window"],
        "commit_hash": fixture["commit_hash"],
    }
    with pytest.raises(ValueError, match="delimiter"):
        make_forward_commitment("gor|don", "2025-Q4", "deadbeef", "salt")


def test_forward_commitment_bridge_separates_public_hash_from_private_preimage(
    tmp_path,
):
    public = tmp_path / "commitment.json"
    private = tmp_path / "private" / "preimage.json"
    result = prepare_forward_window_commitment(
        "local-qwen",
        "window-003",
        {
            "model_digest": "sha256:model",
            "scaffold": "minimal-stateless-v1",
            "risk_config_sha256": "sha256:risk",
        },
        "private-salt",
        commitment_path=public,
        private_preimage_path=private,
    )
    public_payload = json.loads(public.read_text())
    private_payload = json.loads(private.read_text())
    assert public_payload == result["commitment"]
    assert "salt" not in public_payload and "artifact_digest" not in public_payload
    assert private_payload["salt"] == "private-salt"
    assert private_payload["artifact_digest"] == result["artifact_digest"]
    assert (
        make_forward_commitment(
            private_payload["agent_id"],
            private_payload["target_window"],
            private_payload["artifact_digest"],
            private_payload["salt"],
        )
        == public_payload
    )


def test_forward_commitment_refuses_to_overwrite_the_private_preimage(tmp_path):
    same = tmp_path / "same.json"
    with pytest.raises(ValueError, match="paths must differ"):
        prepare_forward_window_commitment(
            "agent",
            "window",
            {"model_digest": "sha256:model"},
            "salt",
            commitment_path=same,
            private_preimage_path=same,
        )


def test_forward_reveal_verifies_the_preimage_and_emits_sharpebench_shape(tmp_path):
    public = tmp_path / "commitment.json"
    private = tmp_path / "preimage.json"
    prepare_forward_window_commitment(
        "agent",
        "window",
        {"model_digest": "sha256:model"},
        "salt",
        commitment_path=public,
        private_preimage_path=private,
    )
    submission = {"agent_id": "agent", "runs": []}
    entry = prepare_forward_window_reveal(
        submission,
        json.loads(public.read_text(encoding="utf-8")),
        json.loads(private.read_text(encoding="utf-8")),
    )
    assert entry["submission"] == submission
    assert entry["salt"] == "salt"
    assert len(entry["artifact_digest"]) == 64

    tampered = json.loads(private.read_text(encoding="utf-8"))
    tampered["salt"] = "other"
    with pytest.raises(ValueError, match="does not open"):
        prepare_forward_window_reveal(submission, json.loads(public.read_text()), tampered)


def test_paper_cli_reveal_writes_the_array_consumed_by_arena_score(tmp_path, capsys):
    public = tmp_path / "commitment.json"
    private = tmp_path / "preimage.json"
    submission_path = tmp_path / "submission.json"
    output = tmp_path / "entries.json"
    prepare_forward_window_commitment(
        "agent",
        "window",
        {"model_digest": "sha256:model"},
        "salt",
        commitment_path=public,
        private_preimage_path=private,
    )
    submission_path.write_text(
        json.dumps({"agent_id": "agent", "runs": []}), encoding="utf-8"
    )
    assert paper_cli_main(
        [
            "reveal",
            "--submission",
            str(submission_path),
            "--commitment",
            str(public),
            "--private-preimage",
            str(private),
            "--output",
            str(output),
        ]
    ) == 0
    entries = json.loads(output.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["submission"]["agent_id"] == "agent"
    assert json.loads(capsys.readouterr().out)["entries"] == 1


def test_paper_cli_plan_is_closed_and_inspection_never_uses_network(tmp_path, capsys):
    payload = {
        "agent_id": "local-model",
        "model_digest": "sha256:model",
        "symbols": ["BTCUSDT"],
        "market_data": {
            "provider": "binance-public",
            "interval": "1h",
            "limit": 2,
        },
        "broker": {"provider": "in-memory"},
        "account": {
            "cash": 10_000,
            "equity": 10_000,
            "session_start_equity": 10_000,
            "peak_equity": 10_000,
            "positions": {},
        },
        "risk": {
            "allowed_symbols": ["BTCUSDT"],
            "max_order_notional": 1_000,
        },
    }
    plan_path = tmp_path / "paper.json"
    decision_path = tmp_path / "decision.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    decision_path.write_text('{"orders":[]}', encoding="utf-8")
    plan = load_execution_plan(plan_path)
    assert (
        paper_cli_main(
            [
                "execute",
                "--plan",
                str(plan_path),
                "--decision",
                str(decision_path),
                "--evidence",
                str(tmp_path / "unused.jsonl"),
                "--inspect",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["plan_sha256"] == plan.plan_sha256

    payload["credentials"] = {"secret": "must never be accepted"}
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_execution_plan(plan_path)
