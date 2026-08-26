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
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .counterfactual import CounterfactualLedger, ghost_orders_from_preflight
from .decision_parser import decision_to_weights, parse_decision_payload

FORWARD_EVIDENCE_CLASS = "forward_paper_trading"
FORWARD_SCHEMA_VERSION = 1
ALPACA_PAPER_ORIGIN = "https://paper-api.alpaca.markets"
ALPACA_DATA_ORIGIN = "https://data.alpaca.markets"
BINANCE_DATA_ORIGIN = "https://api.binance.com"
_SYMBOL = re.compile(r"^[A-Z0-9._/-]{1,32}$")

#: The order has been minted locally and has not left the process.
STATE_PREPARED = "prepared"
#: The request left the process. Nothing is known about the broker's verdict yet.
STATE_SUBMITTED = "submitted"
#: The broker returned an order object; the order exists on its book.
STATE_ACKNOWLEDGED = "acknowledged"
STATE_PARTIALLY_FILLED = "partially_filled"
STATE_FILLED = "filled"
STATE_REJECTED = "rejected"
#: The submission produced neither an acknowledgment nor a rejection.
#:
#: This is the state the whole module exists for. Collapsing it into
#: ``rejected`` asserts an absence the caller never observed, and a replacement
#: issued on that assertion doubles a position that may already be resting.
STATE_SUBMISSION_UNKNOWN = "submission_unknown"
#: A query by client order id found the order. No replacement is authorized.
STATE_RECONCILED_ACCEPTED = "reconciled_accepted"
#: A query by client order id confirmed the order is absent. One replacement is.
STATE_RECONCILED_ABSENT = "reconciled_absent"

LIFECYCLE_STATES = (
    STATE_PREPARED,
    STATE_SUBMITTED,
    STATE_ACKNOWLEDGED,
    STATE_PARTIALLY_FILLED,
    STATE_FILLED,
    STATE_REJECTED,
    STATE_SUBMISSION_UNKNOWN,
    STATE_RECONCILED_ACCEPTED,
    STATE_RECONCILED_ABSENT,
)

_TERMINAL_STATES = frozenset({STATE_FILLED, STATE_REJECTED})

_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_PREPARED: frozenset({STATE_SUBMITTED}),
    STATE_SUBMITTED: frozenset(
        {STATE_ACKNOWLEDGED, STATE_SUBMISSION_UNKNOWN, STATE_REJECTED}
    ),
    STATE_ACKNOWLEDGED: frozenset(
        {STATE_PARTIALLY_FILLED, STATE_FILLED, STATE_REJECTED}
    ),
    STATE_PARTIALLY_FILLED: frozenset(
        {STATE_PARTIALLY_FILLED, STATE_FILLED, STATE_REJECTED}
    ),
    STATE_SUBMISSION_UNKNOWN: frozenset(
        {STATE_RECONCILED_ACCEPTED, STATE_RECONCILED_ABSENT}
    ),
    STATE_RECONCILED_ACCEPTED: frozenset(
        {
            STATE_ACKNOWLEDGED,
            STATE_PARTIALLY_FILLED,
            STATE_FILLED,
            STATE_REJECTED,
        }
    ),
    STATE_RECONCILED_ABSENT: frozenset(),
    STATE_FILLED: frozenset(),
    STATE_REJECTED: frozenset(),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


class PaperTradingError(RuntimeError):
    """A forward data, risk, or paper-broker operation failed."""


class SubmissionUnknown(PaperTradingError):
    """The submission produced no verdict at all.

    Distinct from a rejection on purpose. A rejection is an observation; this is
    the absence of one, and the only sound response is to ask the broker by the
    deterministic client order id rather than to assume either outcome.
    """


class LifecycleError(PaperTradingError):
    """A transition was requested that the order lifecycle does not permit."""


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
        bars = []
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, list) or len(row) < 6:
                    raise ValueError("kline must contain at least six fields")
                bars.append(
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
                )
            except (TypeError, ValueError, IndexError) as error:
                raise PaperTradingError(
                    f"Binance returned a malformed kline at index {index}: {error}"
                ) from error
        return bars


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
        if not isinstance(payload, dict) or not isinstance(payload.get("bars"), list):
            raise PaperTradingError("Alpaca returned no bars array")
        bars = []
        for index, row in enumerate(payload["bars"]):
            try:
                if not isinstance(row, dict):
                    raise ValueError("bar must be an object")
                bars.append(
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
                )
            except (KeyError, TypeError, ValueError) as error:
                raise PaperTradingError(
                    f"Alpaca returned a malformed bar at index {index}: {error}"
                ) from error
        return bars


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


@dataclass(frozen=True)
class AccountSnapshot:
    """The broker's own view of the account at one instant."""

    cash: float
    equity: float
    positions: dict[str, float]
    source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.cash) or not math.isfinite(self.equity):
            raise ValueError("account snapshot cash and equity must be finite")
        if self.equity <= 0.0:
            raise ValueError("account snapshot equity must be positive")
        if any(not math.isfinite(value) for value in self.positions.values()):
            raise ValueError("account snapshot positions must be finite")
        if not self.source:
            raise ValueError("account snapshot requires a source")


def reconcile_account(
    account: PaperAccount, snapshot: AccountSnapshot
) -> dict[str, Any]:
    """Replace the local book with the broker's, keeping the session anchors.

    The plan file seeds an account once. After the first remote fill that seed is
    a guess, and every limit measured against it is measured against a book that
    no longer exists. This overwrites cash, equity, and positions from the
    broker, preserves ``session_start_equity`` so the daily-loss limit still
    refers to the session that is actually running, and ratchets ``peak_equity``
    upward so drawdown is measured from the true high-water mark.
    """

    equity_before = account.equity
    cash_before = account.cash
    positions_before = dict(account.positions)
    positions_after = {
        _checked_symbol(str(symbol)): float(quantity)
        for symbol, quantity in snapshot.positions.items()
        if float(quantity) != 0.0
    }
    account.cash = float(snapshot.cash)
    account.equity = float(snapshot.equity)
    account.positions = positions_after
    account.peak_equity = max(account.peak_equity, account.equity)
    drift = {
        symbol: positions_after.get(symbol, 0.0) - positions_before.get(symbol, 0.0)
        for symbol in set(positions_after) | set(positions_before)
    }
    return {
        "source": snapshot.source,
        "cash_before": cash_before,
        "cash_after": account.cash,
        "equity_before": equity_before,
        "equity_after": account.equity,
        "equity_drift": account.equity - equity_before,
        "position_drift": {
            symbol: value for symbol, value in sorted(drift.items()) if value != 0.0
        },
        "positions_after": dict(positions_after),
        "session_start_equity_preserved": account.session_start_equity,
        "peak_equity_after": account.peak_equity,
    }


@dataclass(frozen=True)
class ForwardWindowIdentity:
    """The pre-committed window a forward record belongs to."""

    agent_id: str
    target_window: str
    commit_hash: str
    artifact_digest: str

    def __post_init__(self) -> None:
        for name in ("agent_id", "target_window", "commit_hash", "artifact_digest"):
            if not getattr(self, name):
                raise ValueError(f"forward window identity requires {name}")

    def as_record(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "target_window": self.target_window,
            "commit_hash": self.commit_hash,
            "artifact_digest": self.artifact_digest,
            # A commitment binds the artifact. It does not make the run replayable.
            "deterministic": False,
        }


def bind_forward_window(prepared: Mapping[str, Any]) -> ForwardWindowIdentity:
    """Turn the output of :func:`prepare_forward_window_commitment` into a stamp."""

    commitment = prepared["commitment"]
    return ForwardWindowIdentity(
        agent_id=commitment["agent_id"],
        target_window=commitment["target_window"],
        commit_hash=commitment["commit_hash"],
        artifact_digest=prepared["artifact_digest"],
    )


def forward_window_from_preimage(path: Path) -> ForwardWindowIdentity:
    """Rebuild the window stamp from the private reveal preimage."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    commitment = make_forward_commitment(
        payload["agent_id"],
        payload["target_window"],
        payload["artifact_digest"],
        payload["salt"],
    )
    return ForwardWindowIdentity(
        agent_id=commitment["agent_id"],
        target_window=commitment["target_window"],
        commit_hash=commitment["commit_hash"],
        artifact_digest=payload["artifact_digest"],
    )


def verify_forward_evidence_window(
    path: Path, window: ForwardWindowIdentity
) -> int:
    """Check that every forward record carries exactly this commitment."""

    lines = [
        line for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    for index, line in enumerate(lines):
        record = json.loads(line)
        stamp = record.get("forward_window")
        if not isinstance(stamp, dict):
            raise PaperTradingError(
                f"forward record {index} carries no forward-window commitment"
            )
        if (
            stamp.get("commit_hash") != window.commit_hash
            or stamp.get("agent_id") != window.agent_id
            or stamp.get("target_window") != window.target_window
            or stamp.get("artifact_digest") != window.artifact_digest
        ):
            raise PaperTradingError(
                f"forward record {index} belongs to a different commitment"
            )
    return len(lines)


@dataclass(frozen=True)
class LifecycleTransition:
    from_state: str
    to_state: str
    at_unix_ns: int
    reason: Optional[str]
    broker_ack_sha256: Optional[str]

    def as_record(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "at_unix_ns": self.at_unix_ns,
            "reason": self.reason,
            "broker_ack_sha256": self.broker_ack_sha256,
        }


class OrderLifecycle:
    """One client order id and every state it has been observed in.

    ``submission_unknown`` is representable, so the code never has to encode an
    unobserved verdict as a rejection, and ``may_submit_replacement`` is false
    until the broker has confirmed an absence by that same id.
    """

    def __init__(self, client_order_id: str):
        if not client_order_id:
            raise ValueError("an order lifecycle requires a client order id")
        self.client_order_id = client_order_id
        self.state = STATE_PREPARED
        self.filled_quantity = 0.0
        self.submitted_at_unix_ns: Optional[int] = None
        self.acknowledged_at_unix_ns: Optional[int] = None
        self.ack_latency_ns: Optional[int] = None
        self.replaced_by: Optional[str] = None
        self.transitions: list[LifecycleTransition] = []

    # -- queries ------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    @property
    def awaiting_reconciliation(self) -> bool:
        return self.state == STATE_SUBMISSION_UNKNOWN

    @property
    def may_submit_replacement(self) -> bool:
        """Only a confirmed absence authorizes a replacement, and only once."""

        return self.state == STATE_RECONCILED_ABSENT and self.replaced_by is None

    @property
    def is_submittable(self) -> bool:
        return self.state == STATE_PREPARED

    # -- transitions --------------------------------------------------------

    def _transition(
        self,
        to_state: str,
        *,
        at_unix_ns: Optional[int],
        reason: Optional[str] = None,
        broker_ack: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if to_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise LifecycleError(
                f"illegal order lifecycle transition {self.state} -> {to_state}"
            )
        moment = time.time_ns() if at_unix_ns is None else int(at_unix_ns)
        self.transitions.append(
            LifecycleTransition(
                from_state=self.state,
                to_state=to_state,
                at_unix_ns=moment,
                reason=reason,
                broker_ack_sha256=None if broker_ack is None else _digest(broker_ack),
            )
        )
        self.state = to_state

    def mark_submitted(self, *, at_unix_ns: Optional[int] = None) -> None:
        moment = time.time_ns() if at_unix_ns is None else int(at_unix_ns)
        self._transition(STATE_SUBMITTED, at_unix_ns=moment)
        self.submitted_at_unix_ns = moment

    def mark_acknowledged(
        self,
        broker_ack: Mapping[str, Any],
        *,
        at_unix_ns: Optional[int] = None,
    ) -> None:
        moment = time.time_ns() if at_unix_ns is None else int(at_unix_ns)
        self._transition(STATE_ACKNOWLEDGED, at_unix_ns=moment, broker_ack=broker_ack)
        self.acknowledged_at_unix_ns = moment
        if self.submitted_at_unix_ns is not None:
            self.ack_latency_ns = moment - self.submitted_at_unix_ns

    def mark_partially_filled(
        self,
        filled_quantity: float,
        broker_ack: Optional[Mapping[str, Any]] = None,
        *,
        at_unix_ns: Optional[int] = None,
    ) -> None:
        self._transition(
            STATE_PARTIALLY_FILLED, at_unix_ns=at_unix_ns, broker_ack=broker_ack
        )
        self.filled_quantity = float(filled_quantity)

    def mark_filled(
        self,
        filled_quantity: float,
        broker_ack: Optional[Mapping[str, Any]] = None,
        *,
        at_unix_ns: Optional[int] = None,
    ) -> None:
        self._transition(STATE_FILLED, at_unix_ns=at_unix_ns, broker_ack=broker_ack)
        self.filled_quantity = float(filled_quantity)

    def mark_rejected(
        self,
        reason: str,
        broker_ack: Optional[Mapping[str, Any]] = None,
        *,
        at_unix_ns: Optional[int] = None,
    ) -> None:
        self._transition(
            STATE_REJECTED,
            at_unix_ns=at_unix_ns,
            reason=reason,
            broker_ack=broker_ack,
        )

    def mark_submission_unknown(
        self, reason: str, *, at_unix_ns: Optional[int] = None
    ) -> None:
        self._transition(
            STATE_SUBMISSION_UNKNOWN, at_unix_ns=at_unix_ns, reason=reason
        )

    def mark_reconciled_accepted(
        self,
        broker_ack: Mapping[str, Any],
        *,
        at_unix_ns: Optional[int] = None,
    ) -> None:
        self._transition(
            STATE_RECONCILED_ACCEPTED, at_unix_ns=at_unix_ns, broker_ack=broker_ack
        )

    def mark_reconciled_absent(self, *, at_unix_ns: Optional[int] = None) -> None:
        self._transition(
            STATE_RECONCILED_ABSENT,
            at_unix_ns=at_unix_ns,
            reason="broker reported no order with this client order id",
        )

    def mark_replaced(self, replacement_client_order_id: str) -> None:
        if not self.may_submit_replacement:
            raise LifecycleError(
                f"{self.client_order_id} in {self.state} authorizes no replacement"
            )
        self.replaced_by = replacement_client_order_id

    # -- persistence --------------------------------------------------------

    def as_record(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "state": self.state,
            "filled_quantity": self.filled_quantity,
            "submitted_at_unix_ns": self.submitted_at_unix_ns,
            "acknowledged_at_unix_ns": self.acknowledged_at_unix_ns,
            "ack_latency_ns": self.ack_latency_ns,
            "replaced_by": self.replaced_by,
            "transitions": [item.as_record() for item in self.transitions],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "OrderLifecycle":
        lifecycle = cls(record["client_order_id"])
        lifecycle.state = record["state"]
        lifecycle.filled_quantity = float(record["filled_quantity"])
        lifecycle.submitted_at_unix_ns = record["submitted_at_unix_ns"]
        lifecycle.acknowledged_at_unix_ns = record["acknowledged_at_unix_ns"]
        lifecycle.ack_latency_ns = record["ack_latency_ns"]
        lifecycle.replaced_by = record["replaced_by"]
        lifecycle.transitions = [
            LifecycleTransition(
                from_state=item["from_state"],
                to_state=item["to_state"],
                at_unix_ns=item["at_unix_ns"],
                reason=item["reason"],
                broker_ack_sha256=item["broker_ack_sha256"],
            )
            for item in record["transitions"]
        ]
        if lifecycle.state not in _ALLOWED_TRANSITIONS:
            raise LifecycleError(f"unknown persisted lifecycle state {lifecycle.state}")
        return lifecycle


class LifecycleStore:
    """Durable lifecycle table keyed by client order id.

    An unresolved order has to outlive the process that created it: a crash
    between the submission and the query is exactly the moment where a restart
    would otherwise re-submit blind.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = None if path is None else Path(path)
        self._entries: dict[str, OrderLifecycle] = {}
        if self.path is not None and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for record in payload.get("lifecycles", []):
                lifecycle = OrderLifecycle.from_record(record)
                self._entries[lifecycle.client_order_id] = lifecycle

    def __len__(self) -> int:
        return len(self._entries)

    def open_or_create(self, client_order_id: str) -> OrderLifecycle:
        existing = self._entries.get(client_order_id)
        if existing is not None:
            return existing
        lifecycle = OrderLifecycle(client_order_id)
        self._entries[client_order_id] = lifecycle
        return lifecycle

    def get(self, client_order_id: str) -> Optional[OrderLifecycle]:
        return self._entries.get(client_order_id)

    def all(self) -> list[OrderLifecycle]:
        return list(self._entries.values())

    def unresolved(self) -> list[OrderLifecycle]:
        return [item for item in self._entries.values() if item.awaiting_reconciliation]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": FORWARD_SCHEMA_VERSION,
            "lifecycles": [item.as_record() for item in self._entries.values()],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        temporary.write_bytes(_canonical_bytes(payload) + b"\n")
        temporary.replace(self.path)


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
        self._by_client_order_id: dict[str, dict[str, Any]] = {}

    def submit(
        self,
        order: PaperOrder,
        *,
        account: PaperAccount,
        prices: Mapping[str, float],
    ) -> dict[str, Any]:
        existing = (
            self._by_client_order_id.get(order.client_order_id)
            if order.client_order_id
            else None
        )
        if existing is not None:
            return existing
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
        response = {
            "id": f"local-paper-{self._sequence:08d}",
            "status": "filled",
            "symbol": order.symbol,
            "side": order.side,
            "filled_qty": order.quantity,
            "filled_avg_price": price,
            "client_order_id": order.client_order_id,
            "risk": asdict(verdict),
        }
        if order.client_order_id:
            self._by_client_order_id[order.client_order_id] = response
        return response

    def find_order(self, client_order_id: str) -> Optional[dict[str, Any]]:
        return self._by_client_order_id.get(client_order_id)


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
        try:
            response = self.transport.request(
                "POST",
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                payload=payload,
            )
        except HTTPError as error:
            # A 4xx is the broker speaking: it saw the order and refused it. A 5xx
            # is the broker failing to speak, which says nothing about whether the
            # order was accepted first.
            if 400 <= int(error.code) < 500:
                raise PaperTradingError(
                    f"the paper broker rejected the order with HTTP {int(error.code)}"
                ) from error
            raise SubmissionUnknown(
                f"no response from the paper broker: HTTP {int(error.code)}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise SubmissionUnknown(
                f"no response from the paper broker: {type(error).__name__}"
            ) from error
        if not isinstance(response, dict):
            raise PaperTradingError("Alpaca paper order returned a non-object response")
        return {
            "id": response.get("id"),
            "status": response.get("status"),
            "symbol": response.get("symbol", order.symbol),
            "side": response.get("side", order.side),
            "submitted_qty": response.get("qty", payload["qty"]),
            "filled_qty": response.get("filled_qty"),
            "client_order_id": response.get("client_order_id", order.client_order_id),
            "risk": asdict(verdict),
        }

    def find_order(self, client_order_id: str) -> Optional[dict[str, Any]]:
        """Ask the paper broker whether it holds this deterministic order id.

        ``None`` means a confirmed absence, which is the only answer that
        authorizes a replacement. Anything short of a confirmed absence raises,
        so an unanswerable query can never be mistaken for one.
        """

        if not client_order_id:
            raise ValueError("a reconciliation query requires a client order id")
        query = urlencode({"client_order_id": client_order_id}, quote_via=quote)
        try:
            response = self.transport.request(
                "GET",
                f"{self.base_url}/v2/orders:by_client_order_id?{query}",
                headers=self.headers,
            )
        except HTTPError as error:
            if int(error.code) == 404:
                return None
            raise SubmissionUnknown(
                f"order query failed with HTTP {int(error.code)}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise SubmissionUnknown(
                f"order query failed: {type(error).__name__}"
            ) from error
        if response is None:
            return None
        if not isinstance(response, dict):
            raise SubmissionUnknown("order query returned a non-object response")
        return response

    def account_snapshot(self) -> AccountSnapshot:
        """Read the paper account and its positions. No order is placed here."""

        account = self.transport.request(
            "GET", f"{self.base_url}/v2/account", headers=self.headers
        )
        positions = self.transport.request(
            "GET", f"{self.base_url}/v2/positions", headers=self.headers
        )
        if not isinstance(account, dict):
            raise PaperTradingError("Alpaca paper account returned a non-object")
        return AccountSnapshot(
            cash=float(account["cash"]),
            equity=float(account["equity"]),
            positions={
                _checked_symbol(str(row["symbol"])): float(row["qty"])
                for row in (positions if isinstance(positions, list) else [])
            },
            source=self.broker_id,
        )


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
    weights = decision_to_weights(
        decision, symbols, current_weights=[0.0] * len(symbols)
    )
    ordered_symbols = {str(order["symbol"]) for order in decision["orders"]}
    orders = []
    for symbol, target_weight in zip(symbols, weights):
        if symbol not in ordered_symbols:
            continue
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


def prepare_forward_window_reveal(
    submission: Mapping[str, Any],
    public_commitment: Mapping[str, Any],
    private_preimage: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one ``sharpebench arena score`` reveal after checking its preimage."""

    required_commitment = {"agent_id", "target_window", "commit_hash"}
    if set(public_commitment) != required_commitment:
        raise ValueError("public commitment has an unsupported schema")
    required_preimage = {
        "agent_id",
        "target_window",
        "artifact_digest",
        "salt",
        "artifact_manifest",
    }
    if set(private_preimage) != required_preimage:
        raise ValueError("private preimage has an unsupported schema")
    if not isinstance(submission, Mapping):
        raise ValueError("submission must be a JSON object")
    agent_id = submission.get("agent_id")
    if agent_id != private_preimage["agent_id"]:
        raise ValueError("submission agent_id does not match the committed agent")
    artifact_digest = _digest(private_preimage["artifact_manifest"])
    if artifact_digest != private_preimage["artifact_digest"]:
        raise ValueError("private artifact manifest does not match artifact_digest")
    expected = make_forward_commitment(
        str(private_preimage["agent_id"]),
        str(private_preimage["target_window"]),
        str(private_preimage["artifact_digest"]),
        str(private_preimage["salt"]),
    )
    if dict(public_commitment) != expected:
        raise ValueError("private preimage does not open the public commitment")
    return {
        "submission": dict(submission),
        "artifact_digest": artifact_digest,
        "salt": str(private_preimage["salt"]),
    }


class PaperTradingSession:
    """Translate canonical target weights into guarded paper orders and evidence."""

    def __init__(
        self,
        broker: PaperBroker,
        journal: ForwardEvidenceJournal,
        *,
        agent_id: str,
        model_digest: str,
        lifecycles: Optional[LifecycleStore] = None,
        window: Optional[ForwardWindowIdentity] = None,
        account_source: Optional[Any] = None,
        counterfactual: Optional[CounterfactualLedger] = None,
    ) -> None:
        if not agent_id or not model_digest:
            raise ValueError("paper session requires agent_id and model_digest")
        if window is not None and window.agent_id != agent_id:
            raise ValueError(
                "forward window commitment was made by a different agent: "
                f"{window.agent_id} != {agent_id}"
            )
        self.broker = broker
        self.journal = journal
        self.agent_id = agent_id
        self.model_digest = model_digest
        self.lifecycles = lifecycles if lifecycles is not None else LifecycleStore()
        self.window = window
        self.account_source = account_source
        self.counterfactual = counterfactual

    # -- evidence -----------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        if self.window is not None:
            record["forward_window"] = self.window.as_record()
        self.journal.append(record)

    def _envelope(self, record_type: str) -> dict[str, Any]:
        return {
            "record_type": record_type,
            "recorded_at_unix_ns": time.time_ns(),
            "agent_id": self.agent_id,
            "model_digest": self.model_digest,
            "broker": self.broker.broker_id,
        }

    # -- reconciliation -----------------------------------------------------

    def _find_order(self, client_order_id: str) -> Optional[dict[str, Any]]:
        query = getattr(self.broker, "find_order", None)
        if query is None:
            raise PaperTradingError(
                f"{self.broker.broker_id} cannot be queried by client order id, so "
                f"the fate of {client_order_id} can never be established"
            )
        return query(client_order_id)

    def _apply_broker_view(
        self, lifecycle: OrderLifecycle, view: Mapping[str, Any]
    ) -> None:
        status = str(view.get("status", "")).lower()
        quantity = view.get("filled_qty")
        filled = 0.0 if quantity is None else float(quantity)
        if status == "filled":
            lifecycle.mark_filled(filled, view)
        elif status in {"partially_filled", "partial_fill"} and filled > 0.0:
            lifecycle.mark_partially_filled(filled, view)
        elif status in {"rejected", "canceled", "cancelled", "expired"}:
            lifecycle.mark_rejected(status, view)

    def reconcile(self, lifecycle: OrderLifecycle) -> dict[str, Any]:
        """Resolve one unknown submission by asking for its client order id."""

        if not lifecycle.awaiting_reconciliation:
            raise LifecycleError(
                f"{lifecycle.client_order_id} in state {lifecycle.state} needs no query"
            )
        found = self._find_order(lifecycle.client_order_id)
        if found is None:
            lifecycle.mark_reconciled_absent()
        else:
            lifecycle.mark_reconciled_accepted(found)
            self._apply_broker_view(lifecycle, found)
        self.lifecycles.save()
        record = self._envelope("order-reconciliation")
        record.update(
            {
                "client_order_id": lifecycle.client_order_id,
                "queried_by": "client_order_id",
                "broker_answer": "absent" if found is None else "present",
                "broker_answer_sha256": _digest(found),
                "lifecycle": lifecycle.as_record(),
            }
        )
        self._append(record)
        return record

    def reconcile_all(self) -> list[dict[str, Any]]:
        """Resolve every unknown submission carried over from an earlier run."""

        return [self.reconcile(item) for item in self.lifecycles.unresolved()]

    def reconcile_account_now(self, account: PaperAccount) -> Optional[dict[str, Any]]:
        if self.account_source is None:
            return None
        snapshot = self.account_source.account_snapshot()
        drift = reconcile_account(account, snapshot)
        record = self._envelope("account-reconciliation")
        record["reconciliation"] = drift
        self._append(record)
        return drift

    def execute_decision(
        self,
        decision: str | dict[str, Any],
        symbols: Sequence[str],
        account: PaperAccount,
        latest_bars: Mapping[str, MarketBar],
        *,
        settlement_prices: Optional[Mapping[str, float]] = None,
    ) -> list[dict[str, Any]]:
        self.reconcile_account_now(account)
        prices = {symbol: latest_bars[symbol].close for symbol in symbols}
        parsed_decision = (
            parse_decision_payload(decision)
            if isinstance(decision, str)
            else parse_decision_payload(json.dumps(decision))
        )
        orders = target_weights_to_orders(parsed_decision, symbols, account, prices)
        preflight = self.broker.risk.assess_batch(orders, account, prices)
        market_snapshot = {symbol: asdict(latest_bars[symbol]) for symbol in symbols}
        if self.counterfactual is not None:
            self.counterfactual.record(
                decision=parsed_decision,
                observation=market_snapshot,
                orders=ghost_orders_from_preflight(
                    [asdict(order) for order in orders],
                    [asdict(verdict) for verdict in preflight],
                    prices,
                ),
                settlement_prices=settlement_prices or {},
            )
        decision_record: dict[str, Any] = {
            **self._envelope("decision"),
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
            self._append(decision_record)
            raise PaperTradingError(
                f"paper batch refused before submission at order {index}: {verdict.reason}"
            )
        decision_record["submission_status"] = (
            "preflight-passed" if orders else "no-orders"
        )
        self._append(decision_record)
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
            lifecycle = self.lifecycles.open_or_create(stable)
            while lifecycle.replaced_by is not None:
                lifecycle = self.lifecycles.open_or_create(lifecycle.replaced_by)
            if lifecycle.awaiting_reconciliation:
                # An unknown submission carried over from an earlier attempt is
                # asked about before anything else is sent for this order.
                self.reconcile(lifecycle)
            if lifecycle.may_submit_replacement:
                replacement_id = f"{lifecycle.client_order_id}-r1"
                replacement = self.lifecycles.open_or_create(replacement_id)
                lifecycle.mark_replaced(replacement_id)
                lifecycle = replacement
            if not lifecycle.is_submittable:
                # The broker either holds this order or has already resolved it.
                self.lifecycles.save()
                continue
            order = PaperOrder(
                order.symbol,
                order.side,
                order.quantity,
                client_order_id=lifecycle.client_order_id,
            )
            record: dict[str, Any] = {
                **self._envelope("order-submission"),
                "market_snapshot_sha256": _digest(market_snapshot),
                "decision_sha256": _digest(parsed_decision),
                "order": asdict(order),
                "batch_preflight": asdict(preflight[index]),
            }
            lifecycle.mark_submitted()
            self.lifecycles.save()
            try:
                response = self.broker.submit(order, account=account, prices=prices)
            except SubmissionUnknown as error:
                lifecycle.mark_submission_unknown(str(error))
                self.lifecycles.save()
                self.reconcile(lifecycle)
                record.update(
                    {
                        "submission_status": "submission-unknown-then-reconciled",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "lifecycle": lifecycle.as_record(),
                    }
                )
                self._append(record)
                raise
            except Exception as error:
                lifecycle.mark_rejected(f"{type(error).__name__}: {error}")
                self.lifecycles.save()
                record.update(
                    {
                        "submission_status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "lifecycle": lifecycle.as_record(),
                    }
                )
                self._append(record)
                raise
            lifecycle.mark_acknowledged(response)
            self._apply_broker_view(lifecycle, response)
            self.lifecycles.save()
            record.update(
                {
                    "submission_status": "accepted-by-paper-broker",
                    "paper_response": response,
                    "lifecycle": lifecycle.as_record(),
                }
            )
            self._append(record)
            submitted.append(response)
        return submitted


__all__ = [
    "ALPACA_PAPER_ORIGIN",
    "LIFECYCLE_STATES",
    "STATE_ACKNOWLEDGED",
    "STATE_FILLED",
    "STATE_PARTIALLY_FILLED",
    "STATE_PREPARED",
    "STATE_RECONCILED_ABSENT",
    "STATE_RECONCILED_ACCEPTED",
    "STATE_REJECTED",
    "STATE_SUBMISSION_UNKNOWN",
    "STATE_SUBMITTED",
    "AccountSnapshot",
    "AlpacaMarketData",
    "AlpacaPaperBroker",
    "BinancePublicData",
    "ForwardEvidenceJournal",
    "ForwardWindowIdentity",
    "InMemoryPaperBroker",
    "LifecycleError",
    "LifecycleStore",
    "LifecycleTransition",
    "MarketBar",
    "OrderLifecycle",
    "PaperAccount",
    "PaperOrder",
    "PaperRiskConfig",
    "PaperRiskGuard",
    "PaperTradingError",
    "PaperTradingSession",
    "RiskVerdict",
    "SubmissionUnknown",
    "bind_forward_window",
    "forward_window_from_preimage",
    "make_forward_commitment",
    "prepare_forward_window_commitment",
    "prepare_forward_window_reveal",
    "reconcile_account",
    "target_weights_to_orders",
    "verify_forward_evidence_window",
]
