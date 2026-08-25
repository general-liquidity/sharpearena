"""Local-only forward paper trading with explicit non-replayable evidence.

This module has no real-capital broker implementation. Public market-data clients
are read-only; the only remote order adapter hard-pins Alpaca's paper endpoint.
Every proposed order passes a native, deny-first risk guard before submission.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .decision_parser import decision_to_weights, parse_decision_payload

FORWARD_EVIDENCE_CLASS = "forward_paper_trading"
FORWARD_SCHEMA_VERSION = 1
ALPACA_PAPER_ORIGIN = "https://paper-api.alpaca.markets"
ALPACA_DATA_ORIGIN = "https://data.alpaca.markets"
BINANCE_DATA_ORIGIN = "https://api.binance.com"
_SYMBOL = re.compile(r"^[A-Z0-9._/-]{1,32}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


class PaperTradingError(RuntimeError):
    """A forward data, risk, or paper-broker operation failed."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse redirects so paper credentials and orders never change origin."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> Any: ...


class UrlLibJsonTransport:
    """Small injectable HTTPS transport used by the read-only/paper adapters."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._opener = build_opener(_NoRedirectHandler())

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> Any:
        body = None if payload is None else _canonical_bytes(payload)
        request = Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", **dict(headers or {})},
        )
        with self._opener.open(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class MarketBar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

    def __post_init__(self) -> None:
        numbers = (self.open, self.high, self.low, self.close, self.volume)
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError("market bar values must be finite")
        if self.close <= 0.0 or self.high < self.low or self.volume < 0.0:
            raise ValueError("market bar has invalid price/volume geometry")


class MarketDataSource(Protocol):
    def recent_bars(self, symbol: str, *, limit: int) -> list[MarketBar]: ...


def _checked_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if not _SYMBOL.fullmatch(symbol):
        raise ValueError("symbol contains unsupported characters")
    return symbol


class BinancePublicData:
    """Unauthenticated, read-only Binance spot klines."""

    _intervals = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
    }

    def __init__(
        self,
        interval: str = "1h",
        *,
        transport: Optional[JsonTransport] = None,
    ) -> None:
        if interval not in self._intervals:
            raise ValueError("unsupported Binance interval")
        self.interval = interval
        self.transport = transport or UrlLibJsonTransport()

    def recent_bars(self, symbol: str, *, limit: int = 120) -> list[MarketBar]:
        if not 1 <= limit <= 1000:
            raise ValueError("Binance limit must lie in [1, 1000]")
        symbol = _checked_symbol(symbol).replace("/", "")
        query = urlencode({"symbol": symbol, "interval": self.interval, "limit": limit})
        rows = self.transport.request(
            "GET", f"{BINANCE_DATA_ORIGIN}/api/v3/klines?{query}"
        )
        if not isinstance(rows, list):
            raise PaperTradingError("Binance returned a non-array kline response")
        return [
            MarketBar(
                symbol=symbol,
                timestamp=str(int(row[0])),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                source="binance-public-spot",
            )
            for row in rows
        ]


class AlpacaMarketData:
    """Read-only Alpaca stock bars; credentials are never placed in evidence."""

    _timeframes = {"1Min", "5Min", "15Min", "30Min", "1Hour", "1Day", "1Week"}

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        timeframe: str = "1Day",
        *,
        transport: Optional[JsonTransport] = None,
    ) -> None:
        if timeframe not in self._timeframes:
            raise ValueError("unsupported Alpaca timeframe")
        self.headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self.timeframe = timeframe
        self.transport = transport or UrlLibJsonTransport()

    def recent_bars(self, symbol: str, *, limit: int = 120) -> list[MarketBar]:
        if not 1 <= limit <= 10_000:
            raise ValueError("Alpaca limit must lie in [1, 10000]")
        symbol = _checked_symbol(symbol)
        query = urlencode(
            {"timeframe": self.timeframe, "limit": limit, "adjustment": "raw"}
        )
        payload = self.transport.request(
            "GET",
            f"{ALPACA_DATA_ORIGIN}/v2/stocks/{symbol}/bars?{query}",
            headers=self.headers,
        )
        rows = payload.get("bars", []) if isinstance(payload, dict) else []
        return [
            MarketBar(
                symbol=symbol,
                timestamp=str(row["t"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
                source="alpaca-market-data",
            )
            for row in rows
        ]


@dataclass(frozen=True)
class PaperOrder:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    time_in_force: str = "day"
    client_order_id: Optional[str] = None

    def __post_init__(self) -> None:
        _checked_symbol(self.symbol)
        if self.side not in {"buy", "sell"}:
            raise ValueError("paper order side must be buy or sell")
        if not math.isfinite(self.quantity) or self.quantity <= 0.0:
            raise ValueError("paper order quantity must be finite and positive")
        if self.order_type != "market":
            raise ValueError(
                "the forward safety profile permits market paper orders only"
            )
        if self.time_in_force not in {"day", "gtc"}:
            raise ValueError("unsupported paper time-in-force")


@dataclass
class PaperAccount:
    cash: float
    equity: float
    session_start_equity: float
    peak_equity: float
    positions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (self.cash, self.equity, self.session_start_equity, self.peak_equity)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paper account values must be finite")
        if (
            self.equity <= 0.0
            or self.session_start_equity <= 0.0
            or self.peak_equity <= 0.0
        ):
            raise ValueError("paper account equity anchors must be positive")
        if any(not math.isfinite(quantity) for quantity in self.positions.values()):
            raise ValueError("paper position quantities must be finite")


@dataclass(frozen=True)
class PaperRiskConfig:
    allowed_symbols: tuple[str, ...]
    max_order_notional: float
    max_gross_exposure: float = 1.0
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    allow_shorting: bool = False
    kill_switch: bool = False

    def __post_init__(self) -> None:
        if not self.allowed_symbols:
            raise ValueError("risk config requires an explicit symbol allowlist")
        if len(set(self.allowed_symbols)) != len(self.allowed_symbols):
            raise ValueError("risk symbol allowlist must not contain duplicates")
        for symbol in self.allowed_symbols:
            if _checked_symbol(symbol) != symbol:
                raise ValueError("risk symbols must use canonical uppercase spelling")
        if (
            not math.isfinite(self.max_order_notional)
            or not math.isfinite(self.max_gross_exposure)
            or self.max_order_notional <= 0.0
            or self.max_gross_exposure <= 0.0
        ):
            raise ValueError("notional and exposure limits must be positive")
        if not 0.0 <= self.max_daily_loss < 1.0 or not 0.0 <= self.max_drawdown < 1.0:
            raise ValueError("loss limits must lie in [0, 1)")


@dataclass(frozen=True)
class RiskVerdict:
    allowed: bool
    reason: str
    projected_gross_exposure: float
    order_notional: float


class PaperRiskGuard:
    """Deny-first order guard independent of Gordon's permission/risk code."""

    def __init__(self, config: PaperRiskConfig):
        self.config = config
        self.allowed_symbols = {
            _checked_symbol(symbol) for symbol in config.allowed_symbols
        }

    def assess(
        self,
        order: PaperOrder,
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> RiskVerdict:
        price = float(prices.get(order.symbol, 0.0))
        notional = order.quantity * price
        all_prices_valid = self._account_prices_are_valid(account, prices)
        gross = (
            self._projected_gross(order, account, prices)
            if price > 0.0 and all_prices_valid
            else math.inf
        )
        refusal = None
        if self.config.kill_switch:
            refusal = "kill-switch-active"
        elif order.symbol not in self.allowed_symbols:
            refusal = "symbol-not-allowlisted"
        elif not math.isfinite(price) or price <= 0.0:
            refusal = "missing-or-invalid-price"
        elif not all_prices_valid:
            refusal = "missing-existing-position-price"
        elif account.equity <= 0.0:
            refusal = "nonpositive-equity"
        elif (
            1.0 - account.equity / account.session_start_equity
            > self.config.max_daily_loss
        ):
            refusal = "daily-loss-limit"
        elif 1.0 - account.equity / account.peak_equity > self.config.max_drawdown:
            refusal = "drawdown-limit"
        elif notional > self.config.max_order_notional:
            refusal = "order-notional-limit"
        elif gross > self.config.max_gross_exposure:
            refusal = "gross-exposure-limit"
        else:
            current = account.positions.get(order.symbol, 0.0)
            projected = current + (
                order.quantity if order.side == "buy" else -order.quantity
            )
            if projected < -1e-12 and not self.config.allow_shorting:
                refusal = "shorting-disabled"
        return RiskVerdict(refusal is None, refusal or "allowed", gross, notional)

    def assess_batch(
        self,
        orders: Sequence[PaperOrder],
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> list[RiskVerdict]:
        """Preflight an order batch against one shadow portfolio.

        A remote paper broker does not update the caller's account between HTTP
        requests. Assessing each order against that unchanged account would let a
        batch exceed the gross cap even though every order passed individually.
        """

        shadow = PaperAccount(
            cash=account.cash,
            equity=account.equity,
            session_start_equity=account.session_start_equity,
            peak_equity=account.peak_equity,
            positions=dict(account.positions),
        )
        verdicts = []
        for order in orders:
            verdict = self.assess(order, shadow, prices)
            verdicts.append(verdict)
            if not verdict.allowed:
                break
            price = float(prices[order.symbol])
            delta = order.quantity if order.side == "buy" else -order.quantity
            shadow.positions[order.symbol] = (
                shadow.positions.get(order.symbol, 0.0) + delta
            )
            shadow.cash -= delta * price
            shadow.equity = shadow.cash + sum(
                quantity * float(prices[symbol])
                for symbol, quantity in shadow.positions.items()
            )
            shadow.peak_equity = max(shadow.peak_equity, shadow.equity)
        return verdicts

    @staticmethod
    def _account_prices_are_valid(
        account: PaperAccount, prices: Mapping[str, float]
    ) -> bool:
        return all(
            quantity == 0.0
            or (
                symbol in prices
                and math.isfinite(float(prices[symbol]))
                and float(prices[symbol]) > 0.0
            )
            for symbol, quantity in account.positions.items()
        )

    @staticmethod
    def _projected_gross(
        order: PaperOrder, account: PaperAccount, prices: Mapping[str, float]
    ) -> float:
        if account.equity <= 0.0:
            return math.inf
        positions = dict(account.positions)
        delta = order.quantity if order.side == "buy" else -order.quantity
        positions[order.symbol] = positions.get(order.symbol, 0.0) + delta
        return (
            sum(
                abs(quantity * float(prices.get(symbol, 0.0)))
                for symbol, quantity in positions.items()
            )
            / account.equity
        )


class PaperBroker(Protocol):
    @property
    def risk(self) -> PaperRiskGuard: ...

    @property
    def broker_id(self) -> str: ...

    def submit(
        self,
        order: PaperOrder,
        *,
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> dict[str, Any]: ...


class InMemoryPaperBroker:
    """Deterministic local fill simulator for dry runs and tests."""

    broker_id = "in-memory-paper-v1"

    def __init__(self, risk: PaperRiskGuard):
        self.risk = risk
        self._sequence = 0

    def submit(
        self,
        order: PaperOrder,
        *,
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> dict[str, Any]:
        verdict = self.risk.assess(order, account, prices)
        if not verdict.allowed:
            raise PaperTradingError(f"paper order refused: {verdict.reason}")
        price = float(prices[order.symbol])
        delta = order.quantity if order.side == "buy" else -order.quantity
        account.positions[order.symbol] = (
            account.positions.get(order.symbol, 0.0) + delta
        )
        account.cash -= delta * price
        account.equity = account.cash + sum(
            quantity * float(prices.get(symbol, 0.0))
            for symbol, quantity in account.positions.items()
        )
        account.peak_equity = max(account.peak_equity, account.equity)
        self._sequence += 1
        return {
            "id": f"local-paper-{self._sequence:08d}",
            "status": "filled",
            "symbol": order.symbol,
            "side": order.side,
            "filled_qty": order.quantity,
            "filled_avg_price": price,
            "risk": asdict(verdict),
        }


class AlpacaPaperBroker:
    """Remote paper-only order adapter; live Alpaca origins are rejected."""

    broker_id = "alpaca-paper-v2"

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        risk: PaperRiskGuard,
        *,
        base_url: str = ALPACA_PAPER_ORIGIN,
        transport: Optional[JsonTransport] = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            f"{parsed.scheme}://{parsed.netloc}" != ALPACA_PAPER_ORIGIN
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("only https://paper-api.alpaca.markets is permitted")
        self.base_url = ALPACA_PAPER_ORIGIN
        self.headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self.risk = risk
        self.transport = transport or UrlLibJsonTransport()

    def submit(
        self,
        order: PaperOrder,
        *,
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> dict[str, Any]:
        verdict = self.risk.assess(order, account, prices)
        if not verdict.allowed:
            raise PaperTradingError(f"paper order refused: {verdict.reason}")
        payload = {
            "symbol": order.symbol,
            "qty": format(order.quantity, ".12g"),
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id
        response = self.transport.request(
            "POST",
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            payload=payload,
        )
        if not isinstance(response, dict):
            raise PaperTradingError("Alpaca paper order returned a non-object response")
        return {
            "id": response.get("id"),
            "status": response.get("status"),
            "symbol": response.get("symbol", order.symbol),
            "side": response.get("side", order.side),
            "submitted_qty": response.get("qty", payload["qty"]),
            "risk": asdict(verdict),
        }


def target_weights_to_orders(
    decision_payload: str | dict[str, Any],
    symbols: Sequence[str],
    account: PaperAccount,
    prices: Mapping[str, float],
    *,
    min_notional: float = 1.0,
) -> list[PaperOrder]:
    if not math.isfinite(min_notional) or min_notional < 0.0:
        raise ValueError("min_notional must be finite and nonnegative")
    decision = (
        parse_decision_payload(decision_payload)
        if isinstance(decision_payload, str)
        else parse_decision_payload(json.dumps(decision_payload))
    )
    weights = decision_to_weights(decision, symbols)
    orders = []
    for symbol, target_weight in zip(symbols, weights):
        if symbol not in prices:
            raise PaperTradingError(f"missing price for {symbol}")
        price = float(prices[symbol])
        if not math.isfinite(price) or price <= 0.0:
            raise PaperTradingError(f"invalid price for {symbol}")
        current = account.positions.get(symbol, 0.0)
        target = target_weight * account.equity / price
        delta = target - current
        if abs(delta * price) < min_notional:
            continue
        orders.append(PaperOrder(symbol, "buy" if delta > 0.0 else "sell", abs(delta)))
    return orders


class ForwardEvidenceJournal:
    """Append-only evidence that explicitly carries no replay guarantee."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        envelope = {
            "schema_version": FORWARD_SCHEMA_VERSION,
            "evidence_class": FORWARD_EVIDENCE_CLASS,
            "deterministic": False,
            "replay_guarantee": "none-live-market-and-paper-broker-state",
            **record,
        }
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(envelope) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def make_forward_commitment(
    agent_id: str, target_window: str, artifact_digest: str, salt: str
) -> dict[str, str]:
    """Match ``sharpebench_attest::make_commitment`` byte for byte."""

    for name, value in (
        ("agent_id", agent_id),
        ("target_window", target_window),
        ("artifact_digest", artifact_digest),
        ("salt", salt),
    ):
        if not value or "|" in value or "\n" in value or "\r" in value:
            raise ValueError(
                f"{name} must be non-empty and contain no commitment delimiter or newline"
            )
    digest = sha256()
    for part in (agent_id, target_window, artifact_digest, salt):
        digest.update(part.encode("utf-8"))
        digest.update(b"|")
    return {
        "agent_id": agent_id,
        "target_window": target_window,
        "commit_hash": digest.hexdigest(),
    }


def prepare_forward_window_commitment(
    agent_id: str,
    target_window: str,
    artifact_manifest: dict[str, Any],
    salt: str,
    *,
    commitment_path: Path,
    private_preimage_path: Path,
) -> dict[str, Any]:
    """Write the public arena commitment and a separate private reveal preimage.

    ``commitment_path`` is accepted directly by ``sharpebench arena commit``.
    The private file must not be committed before the window unlocks.
    """

    if not salt:
        raise ValueError("forward commitment salt must be non-empty")
    if commitment_path.resolve() == private_preimage_path.resolve():
        raise ValueError("public commitment and private preimage paths must differ")
    artifact_digest = _digest(artifact_manifest)
    commitment = make_forward_commitment(agent_id, target_window, artifact_digest, salt)
    preimage = {
        "agent_id": agent_id,
        "target_window": target_window,
        "artifact_digest": artifact_digest,
        "salt": salt,
        "artifact_manifest": artifact_manifest,
    }
    for path, payload in (
        (commitment_path, commitment),
        (private_preimage_path, preimage),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(_canonical_bytes(payload) + b"\n")
        if path == private_preimage_path:
            try:
                temporary.chmod(0o600)
            except OSError:
                pass
        temporary.replace(path)
    return {"commitment": commitment, "artifact_digest": artifact_digest}


class PaperTradingSession:
    """Translate canonical target weights into guarded paper orders and evidence."""

    def __init__(
        self,
        broker: PaperBroker,
        journal: ForwardEvidenceJournal,
        *,
        agent_id: str,
        model_digest: str,
    ) -> None:
        if not agent_id or not model_digest:
            raise ValueError("paper session requires agent_id and model_digest")
        self.broker = broker
        self.journal = journal
        self.agent_id = agent_id
        self.model_digest = model_digest

    def execute_decision(
        self,
        decision: str | dict[str, Any],
        symbols: Sequence[str],
        account: PaperAccount,
        latest_bars: Mapping[str, MarketBar],
    ) -> list[dict[str, Any]]:
        prices = {symbol: latest_bars[symbol].close for symbol in symbols}
        parsed_decision = (
            parse_decision_payload(decision)
            if isinstance(decision, str)
            else parse_decision_payload(json.dumps(decision))
        )
        orders = target_weights_to_orders(parsed_decision, symbols, account, prices)
        preflight = self.broker.risk.assess_batch(orders, account, prices)
        market_snapshot = {symbol: asdict(latest_bars[symbol]) for symbol in symbols}
        decision_record: dict[str, Any] = {
            "record_type": "decision",
            "recorded_at_unix_ns": time.time_ns(),
            "agent_id": self.agent_id,
            "model_digest": self.model_digest,
            "broker": self.broker.broker_id,
            "market_snapshot": market_snapshot,
            "market_snapshot_sha256": _digest(market_snapshot),
            "decision": parsed_decision,
            "decision_sha256": _digest(parsed_decision),
            "proposed_orders": [asdict(order) for order in orders],
            "batch_preflight": [asdict(verdict) for verdict in preflight],
        }
        refusal = next(
            (
                (index, verdict)
                for index, verdict in enumerate(preflight)
                if not verdict.allowed
            ),
            None,
        )
        if refusal is not None:
            index, verdict = refusal
            decision_record.update(
                {
                    "submission_status": "risk-refused-before-submission",
                    "refused_order_index": index,
                    "refusal_reason": verdict.reason,
                }
            )
            self.journal.append(decision_record)
            raise PaperTradingError(
                f"paper batch refused before submission at order {index}: {verdict.reason}"
            )
        decision_record["submission_status"] = (
            "preflight-passed" if orders else "no-orders"
        )
        self.journal.append(decision_record)
        submitted = []
        for index, order in enumerate(orders):
            stable = _digest(
                {
                    "agent_id": self.agent_id,
                    "model_digest": self.model_digest,
                    "decision": parsed_decision,
                    "market": market_snapshot,
                    "index": index,
                }
            )[:32]
            order = PaperOrder(
                order.symbol,
                order.side,
                order.quantity,
                client_order_id=stable,
            )
            record: dict[str, Any] = {
                "record_type": "order-submission",
                "recorded_at_unix_ns": time.time_ns(),
                "agent_id": self.agent_id,
                "model_digest": self.model_digest,
                "broker": self.broker.broker_id,
                "market_snapshot_sha256": _digest(market_snapshot),
                "decision_sha256": _digest(parsed_decision),
                "order": asdict(order),
                "batch_preflight": asdict(preflight[index]),
            }
            try:
                response = self.broker.submit(order, account=account, prices=prices)
            except Exception as error:
                record.update(
                    {
                        "submission_status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                self.journal.append(record)
                raise
            record.update(
                {
                    "submission_status": "accepted-by-paper-broker",
                    "paper_response": response,
                }
            )
            self.journal.append(record)
            submitted.append(response)
        return submitted


__all__ = [
    "ALPACA_PAPER_ORIGIN",
    "AlpacaMarketData",
    "AlpacaPaperBroker",
    "BinancePublicData",
    "ForwardEvidenceJournal",
    "InMemoryPaperBroker",
    "MarketBar",
    "PaperAccount",
    "PaperOrder",
    "PaperRiskConfig",
    "PaperRiskGuard",
    "PaperTradingError",
    "PaperTradingSession",
    "RiskVerdict",
    "make_forward_commitment",
    "prepare_forward_window_commitment",
    "target_weights_to_orders",
]
