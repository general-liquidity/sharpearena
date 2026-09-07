"""V2 decision snapshots with confirmed fills and explicit unknown outcomes.

P&L is a reference-price mark, not realized fill-price/cost accounting or a
causal estimate. See docs/counterfactual-ledger.md for the versioned contract.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

COUNTERFACTUAL_SCHEMA_VERSION = 2
COUNTERFACTUAL_EVIDENCE_CLASS = "counterfactual_decision_ledger"

#: Observed order disposition; approval or acknowledgement alone is not a fill.
DISPOSITIONS = frozenset(
    {
        "executed",
        "risk_refused",
        "below_min_notional",
        "not_submitted",
        "resized",
        "acknowledged",
        "partially_filled",
        "submission_unknown",
        "broker_rejected",
        "broker_canceled",
        "broker_expired",
    }
)


class CounterfactualError(ValueError):
    """A counterfactual record is internally inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise CounterfactualError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CounterfactualError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError as error:
        raise CounterfactualError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise CounterfactualError(f"{name} must be finite")
    return number


def _sum_available(values: Iterable[Optional[float]]) -> Optional[float]:
    values = list(values)
    if any(value is None for value in values):
        return None
    try:
        return _finite(math.fsum(values), "aggregate")
    except OverflowError as error:
        raise CounterfactualError("aggregate must be finite") from error


def _acted(orders: Sequence[GhostOrder]) -> Optional[bool]:
    if any(order.executed_quantity > 0 for order in orders):
        return True
    return False if all(order.execution_complete for order in orders) else None


@dataclass(frozen=True)
class GhostOrder:
    """One intended order and what became of it.

    ``executed_quantity`` is a confirmed cumulative fill lower bound until
    ``execution_complete``. Acknowledgement is not execution.
    """

    symbol: str
    side: str
    intended_quantity: float
    executed_quantity: float
    reference_price: float
    disposition: str
    reason: str
    execution_complete: bool = True
    client_order_id: Optional[str] = None
    receipt_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise CounterfactualError("symbol must be nonempty")
        if type(self.execution_complete) is not bool:
            raise CounterfactualError("execution_complete must be boolean")
        if (
            self.disposition
            in {"acknowledged", "partially_filled", "submission_unknown"}
            and self.execution_complete
        ):
            raise CounterfactualError("pending or unknown execution cannot be complete")
        if self.disposition == "executed" and not self.execution_complete:
            raise CounterfactualError(
                "executed disposition requires complete execution"
            )
        if self.client_order_id is not None and (
            not isinstance(self.client_order_id, str) or not self.client_order_id
        ):
            raise CounterfactualError("client_order_id must be nonempty")
        if self.receipt_sha256 is not None and (
            not isinstance(self.receipt_sha256, str)
            or len(self.receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.receipt_sha256)
        ):
            raise CounterfactualError(
                "receipt_sha256 must be a lowercase SHA-256 digest"
            )
        if self.side not in {"buy", "sell"}:
            raise CounterfactualError("side must be buy or sell")
        if self.disposition not in DISPOSITIONS:
            raise CounterfactualError(f"unknown disposition {self.disposition!r}")
        for name in ("intended_quantity", "executed_quantity", "reference_price"):
            _finite(getattr(self, name), name)
        if self.intended_quantity <= 0.0:
            raise CounterfactualError("intended_quantity must be positive")
        if self.executed_quantity < 0.0:
            raise CounterfactualError("executed_quantity must not be negative")
        if self.executed_quantity > self.intended_quantity:
            raise CounterfactualError(
                "executed_quantity cannot exceed intended_quantity"
            )
        if self.reference_price <= 0.0:
            raise CounterfactualError("reference_price must be positive")
        _finite(self.intended_quantity * self.reference_price, "intended_notional")
        if (
            self.disposition in {"risk_refused", "below_min_notional", "not_submitted"}
            and self.executed_quantity != 0
        ):
            raise CounterfactualError(
                "unsubmitted disposition claims a difference that is not there: nonzero fill"
            )
        if (
            self.disposition == "executed"
            and self.executed_quantity != self.intended_quantity
        ):
            raise CounterfactualError(
                "disposition 'executed' requires the full intended quantity"
            )
        if (
            self.disposition not in {"executed", "submission_unknown"}
            and self.executed_quantity == self.intended_quantity
        ):
            raise CounterfactualError(
                f"disposition {self.disposition!r} claims a difference that is not there"
            )

    @property
    def signed_intended_quantity(self) -> float:
        return self.intended_quantity if self.side == "buy" else -self.intended_quantity

    @property
    def signed_executed_quantity(self) -> float:
        return self.executed_quantity if self.side == "buy" else -self.executed_quantity

    @property
    def intended_notional(self) -> float:
        return self.intended_quantity * self.reference_price

    @property
    def confirmed_notional(self) -> float:
        return self.executed_quantity * self.reference_price

    @property
    def executed_notional(self) -> Optional[float]:
        return self.confirmed_notional if self.execution_complete else None

    @property
    def foregone_notional(self) -> Optional[float]:
        """Notional the decision wanted and the execution path did not take."""

        return (
            None
            if not self.execution_complete
            else self.intended_notional - self.confirmed_notional
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "intended_quantity": self.intended_quantity,
            "executed_quantity": self.executed_quantity,
            "reference_price": self.reference_price,
            "disposition": self.disposition,
            "reason": self.reason,
            "execution_complete": self.execution_complete,
            "client_order_id": self.client_order_id,
            "receipt_sha256": self.receipt_sha256,
            "confirmed_notional": self.confirmed_notional,
            "intended_notional": self.intended_notional,
            "executed_notional": self.executed_notional,
            "foregone_notional": self.foregone_notional,
        }


@dataclass(frozen=True)
class CounterfactualRecord:
    """One decision, whether or not any part of it was acted on."""

    sequence: int
    decision_sha256: str
    observation_sha256: str
    orders: tuple[GhostOrder, ...]
    #: Caller-supplied marks, applied to both arms at the same reference price.
    settlement_prices: Mapping[str, float]
    acted: Optional[bool]
    decision_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "settlement_prices", MappingProxyType(dict(self.settlement_prices))
        )
        for symbol, price in self.settlement_prices.items():
            if _finite(price, f"settlement_prices[{symbol}]") <= 0.0:
                raise CounterfactualError("settlement prices must be positive")
        if self.acted is not _acted(self.orders):
            raise CounterfactualError(
                "acted must agree with confirmed fills and execution availability"
            )

    def _pnl(self, quantity_of: str) -> Optional[float]:
        total = 0.0
        for order in self.orders:
            if (
                quantity_of == "signed_executed_quantity"
                and not order.execution_complete
            ):
                return None
            signed = getattr(order, quantity_of)
            if signed == 0:
                continue
            settlement = self.settlement_prices.get(order.symbol)
            if settlement is None:
                return None
            total += signed * (settlement - order.reference_price)
        return _finite(total, "reference-price P&L")

    @property
    def intended_pnl(self) -> Optional[float]:
        """Mark-to-market of the decision as proposed."""

        return self._pnl("signed_intended_quantity")

    @property
    def executed_pnl(self) -> Optional[float]:
        """Reference-price mark on confirmed final fills, not realized P&L."""

        return self._pnl("signed_executed_quantity")

    @property
    def foregone_pnl(self) -> Optional[float]:
        """Hypothetical reference-price difference, unavailable for unknown fills."""
        intended, executed = self.intended_pnl, self.executed_pnl
        return (
            None
            if intended is None or executed is None
            else _finite(intended - executed, "foregone_pnl")
        )

    @property
    def intended_notional(self) -> float:
        return _sum_available(order.intended_notional for order in self.orders)

    @property
    def executed_notional(self) -> Optional[float]:
        return _sum_available(order.executed_notional for order in self.orders)

    @property
    def confirmed_notional(self) -> float:
        return _sum_available(order.confirmed_notional for order in self.orders)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "evidence_class": COUNTERFACTUAL_EVIDENCE_CLASS,
            "valuation": "reference_price_mark",
            "sequence": self.sequence,
            "decision_sha256": self.decision_sha256,
            "observation_sha256": self.observation_sha256,
            "acted": self.acted,
            "decision_key": self.decision_key,
            "confirmed_notional": self.confirmed_notional,
            "orders": [order.as_record() for order in self.orders],
            "settlement_prices": dict(self.settlement_prices),
            "intended_notional": self.intended_notional,
            "executed_notional": self.executed_notional,
            "intended_pnl": self.intended_pnl,
            "executed_pnl": self.executed_pnl,
            "foregone_pnl": self.foregone_pnl,
        }


@dataclass(frozen=True)
class SelectionGap:
    """The measurable distance between what was considered and what was done."""

    decisions: int
    acted_decisions: int
    unacted_decisions: int
    unknown_decisions: int
    snapshots: int
    intended_notional: float
    confirmed_notional: float
    executed_notional: Optional[float]
    intended_pnl: Optional[float]
    executed_pnl: Optional[float]
    foregone_pnl: Optional[float]
    dispositions: dict[str, int]

    @property
    def execution_ratio(self) -> Optional[float]:
        """Final executed/intended notional; None if unknown, 1 for zero intent."""

        if self.intended_notional == 0.0:
            return 1.0
        if self.executed_notional is None:
            return None
        return self.executed_notional / self.intended_notional

    def as_record(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "acted_decisions": self.acted_decisions,
            "unacted_decisions": self.unacted_decisions,
            "unknown_decisions": self.unknown_decisions,
            "snapshots": self.snapshots,
            "confirmed_notional": self.confirmed_notional,
            "intended_notional": self.intended_notional,
            "executed_notional": self.executed_notional,
            "execution_ratio": self.execution_ratio,
            "intended_pnl": self.intended_pnl,
            "executed_pnl": self.executed_pnl,
            "foregone_pnl": self.foregone_pnl,
            "dispositions": dict(self.dispositions),
        }


class CounterfactualLedger:
    """Append-only ledger of every decision, acted on or not.

    ``path=None`` keeps the ledger in memory. The paper-session caller supplies
    broker receipts; direct callers are responsible for their own execution facts.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = None if path is None else Path(path)
        self._records: list[CounterfactualRecord] = []
        self._latest: dict[str, CounterfactualRecord] = {}
        self._write_failed = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                content = self.path.read_text(encoding="utf-8")
                if content and not content.endswith("\n"):
                    raise CounterfactualError(
                        "invalid counterfactual log: final record lacks a newline"
                    )
                for line_number, line in enumerate(content.splitlines(), 1):
                    try:
                        row = json.loads(line, object_pairs_hook=_unique_object)
                        if row.get("schema_version") != COUNTERFACTUAL_SCHEMA_VERSION:
                            raise CounterfactualError(
                                "requires V2 receipts; V1 execution is unverified"
                            )
                        fields = GhostOrder.__dataclass_fields__
                        entry = CounterfactualRecord(
                            sequence=row["sequence"],
                            decision_sha256=row["decision_sha256"],
                            observation_sha256=row["observation_sha256"],
                            orders=tuple(
                                GhostOrder(**{k: item[k] for k in fields})
                                for item in row["orders"]
                            ),
                            settlement_prices=row["settlement_prices"],
                            acted=row["acted"],
                            decision_key=row["decision_key"],
                        )
                        self._validate_revision(entry)
                        if _canonical_bytes(entry.as_record()) != _canonical_bytes(row):
                            raise CounterfactualError(
                                "persisted snapshot does not match its derived fields"
                            )
                        self._remember(entry)
                    except (ValueError, TypeError, KeyError, AttributeError) as error:
                        raise CounterfactualError(
                            f"invalid counterfactual line {line_number}: {error}"
                        ) from error

    @property
    def records(self) -> tuple[CounterfactualRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def latest(self, decision_key: str) -> Optional[CounterfactualRecord]:
        return self._latest.get(decision_key)

    def _remember(self, entry: CounterfactualRecord) -> None:
        self._records.append(entry)
        if entry.decision_key is not None:
            self._latest[entry.decision_key] = entry

    @property
    def decision_records(self) -> tuple[CounterfactualRecord, ...]:
        latest = {}
        for entry in self._records:
            latest[
                (
                    ("key", entry.decision_key)
                    if entry.decision_key is not None
                    else ("sequence", entry.sequence)
                )
            ] = entry
        return tuple(latest.values())

    def refresh_receipt(self, evidence: GhostOrder) -> None:
        """Append revisions for decisions containing this cumulative receipt."""
        if evidence.client_order_id is None:
            raise CounterfactualError("refresh requires a client order id")
        for entry in self.decision_records:
            if entry.decision_key is None or not any(
                item.client_order_id == evidence.client_order_id
                for item in entry.orders
            ):
                continue
            orders = tuple(
                evidence if item.client_order_id == evidence.client_order_id else item
                for item in entry.orders
            )
            revision = replace(
                entry, sequence=len(self._records), orders=orders, acted=_acted(orders)
            )
            self._validate_revision(revision)
            self._append(revision)
            self._remember(revision)

    def _validate_revision(self, entry: CounterfactualRecord) -> None:
        if type(entry.sequence) is not int or entry.sequence != len(self._records):
            raise CounterfactualError("snapshot sequence must be contiguous")
        if entry.decision_key is None:
            return
        if not isinstance(entry.decision_key, str) or not entry.decision_key:
            raise CounterfactualError("decision_key must be nonempty")
        previous = self.latest(entry.decision_key)
        if previous is None:
            return

        def intent(item: GhostOrder) -> tuple:
            return (
                item.symbol,
                item.side,
                item.intended_quantity,
                item.reference_price,
            )

        if (
            entry.decision_sha256 != previous.decision_sha256
            or entry.observation_sha256 != previous.observation_sha256
            or tuple(map(intent, entry.orders)) != tuple(map(intent, previous.orders))
        ):
            raise CounterfactualError(
                "a decision revision cannot replace its frozen intent"
            )
        for before, after in zip(previous.orders, entry.orders):
            if after.executed_quantity < before.executed_quantity:
                raise CounterfactualError("confirmed cumulative fills cannot decrease")

    def record(
        self,
        *,
        decision: Mapping[str, Any],
        observation: Any,
        orders: Sequence[GhostOrder],
        settlement_prices: Mapping[str, float],
        decision_key: Optional[str] = None,
    ) -> CounterfactualRecord:
        """Record one decision. A decision with no orders is still recorded."""

        entry = CounterfactualRecord(
            sequence=len(self._records),
            decision_sha256=_digest(json.loads(_canonical_bytes(decision))),
            observation_sha256=_digest(json.loads(_canonical_bytes(observation))),
            orders=tuple(orders),
            settlement_prices={
                str(symbol): _finite(price, "settlement price")
                for symbol, price in settlement_prices.items()
            },
            acted=_acted(orders),
            decision_key=decision_key,
        )
        self._validate_revision(entry)
        self._append(entry)
        self._remember(entry)
        return entry

    def selection_gap(self) -> SelectionGap:
        # Unkeyed direct-call records are independent; paper-session keys identify
        # revisions of the same decision. Retain all snapshots on disk.
        entries = self.decision_records
        dispositions: dict[str, int] = {name: 0 for name in sorted(DISPOSITIONS)}
        for entry in entries:
            for order in entry.orders:
                dispositions[order.disposition] += 1
        return SelectionGap(
            decisions=len(entries),
            snapshots=len(self._records),
            acted_decisions=sum(entry.acted is True for entry in entries),
            unacted_decisions=sum(entry.acted is False for entry in entries),
            unknown_decisions=sum(entry.acted is None for entry in entries),
            intended_notional=_sum_available(
                entry.intended_notional for entry in entries
            ),
            confirmed_notional=_sum_available(
                entry.confirmed_notional for entry in entries
            ),
            executed_notional=_sum_available(
                entry.executed_notional for entry in entries
            ),
            intended_pnl=_sum_available(entry.intended_pnl for entry in entries),
            executed_pnl=_sum_available(entry.executed_pnl for entry in entries),
            foregone_pnl=_sum_available(entry.foregone_pnl for entry in entries),
            dispositions=dispositions,
        )

    def _append(self, entry: CounterfactualRecord) -> None:
        if self._write_failed:
            raise CounterfactualError(
                "ledger write failed previously; reopen and validate the log before continuing"
            )
        payload = _canonical_bytes(entry.as_record()) + b"\n"
        if self.path is None:
            return
        try:
            with self.path.open("ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            self._write_failed = True
            raise


def ghost_orders_from_preflight(
    intended: Iterable[Mapping[str, Any]],
    verdicts: Sequence[Optional[Mapping[str, Any]]],
    prices: Mapping[str, float],
) -> tuple[GhostOrder, ...]:
    """Describe an unsubmitted batch; preflight never proves execution.

    A batch preflight stops at the first refusal, so orders after the refusal
    carry no verdict at all. Those are ``not_submitted``, which is a different
    fact from ``risk_refused`` and is recorded as such.
    """

    out: list[GhostOrder] = []
    for index, order in enumerate(intended):
        symbol = str(order["symbol"])
        price = float(prices[symbol])
        quantity = float(order["quantity"])
        verdict = verdicts[index] if index < len(verdicts) else None
        if verdict is None:
            disposition, reason, executed = (
                "not_submitted",
                "batch halted by an earlier refusal",
                0.0,
            )
        elif verdict.get("allowed"):
            disposition, reason, executed = (
                "not_submitted",
                "preflight approval is not a fill",
                0.0,
            )
        else:
            disposition, reason, executed = (
                "risk_refused",
                str(verdict.get("reason", "refused")),
                0.0,
            )
        out.append(
            GhostOrder(
                symbol=symbol,
                side=str(order["side"]),
                intended_quantity=quantity,
                executed_quantity=executed,
                reference_price=price,
                disposition=disposition,
                reason=reason,
            )
        )
    return tuple(out)


def ghost_order_from_receipt(
    intended: GhostOrder, receipt: Mapping[str, Any]
) -> GhostOrder:
    """Validate cumulative fill facts against an already bound intended order.

    Broker quantities can be decimal strings. Missing quantities and unfamiliar
    statuses never establish finality. Contradictory receipts are refused.
    """
    if not isinstance(receipt, Mapping):
        raise CounterfactualError("broker receipt must be an object")
    for field, expected in (
        ("client_order_id", intended.client_order_id),
        ("symbol", intended.symbol),
        ("side", intended.side),
    ):
        if field in receipt and receipt[field] != expected:
            raise CounterfactualError(
                f"broker receipt {field} does not match the intended order"
            )
    raw = receipt.get("filled_qty")
    if isinstance(raw, bool):
        raise CounterfactualError("filled_qty must be a finite quantity, not boolean")
    try:
        quantity = (
            intended.executed_quantity
            if raw is None
            else _finite(float(raw), "filled_qty")
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise CounterfactualError("filled_qty must be a finite quantity") from error
    if not intended.executed_quantity <= quantity <= intended.intended_quantity:
        raise CounterfactualError(
            "cumulative filled_qty must not decrease or exceed intended quantity"
        )
    status = receipt.get("status")
    if not isinstance(status, str):
        raise CounterfactualError("broker receipt status must be a string")
    status = status.lower()
    terminal = {
        "filled": "executed",
        "rejected": "broker_rejected",
        "canceled": "broker_canceled",
        "cancelled": "broker_canceled",
        "expired": "broker_expired",
    }
    complete = status in terminal and raw is not None
    if status == "filled" and complete and quantity != intended.intended_quantity:
        raise CounterfactualError(
            "filled receipt must confirm the full intended quantity"
        )
    disposition = (
        terminal[status]
        if complete
        else "partially_filled" if quantity else "acknowledged"
    )
    # A terminal cancellation may report all fills in a race with completion.
    if complete and quantity == intended.intended_quantity:
        disposition = "executed"
    return replace(
        intended,
        executed_quantity=quantity,
        execution_complete=complete,
        disposition=disposition,
        reason=f"broker status: {status}",
        receipt_sha256=_digest(receipt),
    )


__all__ = [
    "COUNTERFACTUAL_EVIDENCE_CLASS",
    "COUNTERFACTUAL_SCHEMA_VERSION",
    "DISPOSITIONS",
    "CounterfactualError",
    "CounterfactualLedger",
    "CounterfactualRecord",
    "GhostOrder",
    "SelectionGap",
    "ghost_orders_from_preflight",
    "ghost_order_from_receipt",
]
