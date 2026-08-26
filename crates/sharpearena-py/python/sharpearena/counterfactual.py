"""A counterfactual ledger over every decision, acted on or not.

The generation path already records every raw candidate before validation and
deduplication, because a trial count measured after filtering understates the
search. The execution path has the same hole one layer down: a decision that was
proposed and then dropped, refused by risk, or clipped to a smaller size leaves
no trace, so the difference between what the agent considered and what it did is
invisible. Selection effects then look like skill.

This module closes that gap. Every decision the environment produces is recorded
with its intended orders, the orders that actually reached the broker, and the
reason for any difference. The ghost fill for a dropped order is priced from the
same observation the live order used, so the two arms are comparable, and the
whole thing is arithmetic over already-recorded data: it costs nothing in the
deterministic arm and it never influences what is executed.

:meth:`CounterfactualLedger.selection_gap` is the reason the ledger exists. It
reports intended notional against executed notional, the counterfactual return of
the orders that were dropped, and the realized return of the ones that were not,
so "the risk gate cost us money" and "the risk gate saved us money" become
measurable statements instead of positions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

COUNTERFACTUAL_SCHEMA_VERSION = 1
COUNTERFACTUAL_EVIDENCE_CLASS = "counterfactual_decision_ledger"

#: Why an intended order did not reach the broker verbatim.
DISPOSITIONS = frozenset(
    {
        "executed",
        "risk_refused",
        "below_min_notional",
        "not_submitted",
        "resized",
    }
)


class CounterfactualError(ValueError):
    """A counterfactual record is internally inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CounterfactualError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CounterfactualError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class GhostOrder:
    """One intended order and what became of it.

    ``intended_quantity`` is what the decision asked for; ``executed_quantity``
    is what reached the broker. They are equal only when ``disposition`` is
    ``executed``.
    """

    symbol: str
    side: str
    intended_quantity: float
    executed_quantity: float
    reference_price: float
    disposition: str
    reason: str

    def __post_init__(self) -> None:
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
        if self.executed_quantity > self.intended_quantity + 1e-12:
            raise CounterfactualError(
                "executed_quantity cannot exceed intended_quantity"
            )
        if self.reference_price <= 0.0:
            raise CounterfactualError("reference_price must be positive")
        if self.disposition == "executed" and not math.isclose(
            self.executed_quantity, self.intended_quantity, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CounterfactualError(
                "disposition 'executed' requires the full intended quantity"
            )
        if self.disposition != "executed" and math.isclose(
            self.executed_quantity, self.intended_quantity, rel_tol=1e-12, abs_tol=1e-12
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
    def executed_notional(self) -> float:
        return self.executed_quantity * self.reference_price

    @property
    def foregone_notional(self) -> float:
        """Notional the decision wanted and the execution path did not take."""

        return self.intended_notional - self.executed_notional

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "intended_quantity": self.intended_quantity,
            "executed_quantity": self.executed_quantity,
            "reference_price": self.reference_price,
            "disposition": self.disposition,
            "reason": self.reason,
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
    #: Mark prices one step later, used to settle both arms identically.
    settlement_prices: dict[str, float]
    acted: bool

    def __post_init__(self) -> None:
        for symbol, price in self.settlement_prices.items():
            if _finite(price, f"settlement_prices[{symbol}]") <= 0.0:
                raise CounterfactualError("settlement prices must be positive")
        if self.acted != any(order.executed_quantity > 0.0 for order in self.orders):
            raise CounterfactualError(
                "acted must agree with whether any quantity reached the broker"
            )

    def _pnl(self, quantity_of: str) -> float:
        total = 0.0
        for order in self.orders:
            settlement = self.settlement_prices.get(order.symbol)
            if settlement is None:
                continue
            signed = getattr(order, quantity_of)
            total += signed * (settlement - order.reference_price)
        return total

    @property
    def intended_pnl(self) -> float:
        """Mark-to-market of the decision as proposed."""

        return self._pnl("signed_intended_quantity")

    @property
    def executed_pnl(self) -> float:
        """Mark-to-market of what actually reached the broker."""

        return self._pnl("signed_executed_quantity")

    @property
    def foregone_pnl(self) -> float:
        """What the untaken part of the decision would have earned or lost.

        Positive means the execution path cost the run money; negative means it
        avoided a loss. Both are informative and neither is available without a
        ledger over unexecuted decisions.
        """

        return self.intended_pnl - self.executed_pnl

    @property
    def intended_notional(self) -> float:
        return sum(order.intended_notional for order in self.orders)

    @property
    def executed_notional(self) -> float:
        return sum(order.executed_notional for order in self.orders)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "evidence_class": COUNTERFACTUAL_EVIDENCE_CLASS,
            "sequence": self.sequence,
            "decision_sha256": self.decision_sha256,
            "observation_sha256": self.observation_sha256,
            "acted": self.acted,
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
    intended_notional: float
    executed_notional: float
    intended_pnl: float
    executed_pnl: float
    foregone_pnl: float
    dispositions: dict[str, int]

    @property
    def execution_ratio(self) -> float:
        """Executed notional over intended notional; 1.0 when nothing was dropped."""

        if self.intended_notional == 0.0:
            return 1.0
        return self.executed_notional / self.intended_notional

    def as_record(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "acted_decisions": self.acted_decisions,
            "unacted_decisions": self.unacted_decisions,
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

    ``path=None`` keeps the ledger in memory, which is what the deterministic arm
    uses: the accounting is arithmetic over data the run already has.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = None if path is None else Path(path)
        self._records: list[CounterfactualRecord] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def records(self) -> tuple[CounterfactualRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def record(
        self,
        *,
        decision: Mapping[str, Any],
        observation: Any,
        orders: Sequence[GhostOrder],
        settlement_prices: Mapping[str, float],
    ) -> CounterfactualRecord:
        """Record one decision. A decision with no orders is still recorded."""

        entry = CounterfactualRecord(
            sequence=len(self._records),
            decision_sha256=_digest(json.loads(_canonical_bytes(decision))),
            observation_sha256=_digest(json.loads(_canonical_bytes(observation))),
            orders=tuple(orders),
            settlement_prices={
                str(symbol): float(price)
                for symbol, price in settlement_prices.items()
            },
            acted=any(order.executed_quantity > 0.0 for order in orders),
        )
        self._records.append(entry)
        self._append(entry)
        return entry

    def selection_gap(self) -> SelectionGap:
        dispositions: dict[str, int] = {name: 0 for name in sorted(DISPOSITIONS)}
        for entry in self._records:
            for order in entry.orders:
                dispositions[order.disposition] += 1
        return SelectionGap(
            decisions=len(self._records),
            acted_decisions=sum(entry.acted for entry in self._records),
            unacted_decisions=sum(not entry.acted for entry in self._records),
            intended_notional=sum(entry.intended_notional for entry in self._records),
            executed_notional=sum(entry.executed_notional for entry in self._records),
            intended_pnl=sum(entry.intended_pnl for entry in self._records),
            executed_pnl=sum(entry.executed_pnl for entry in self._records),
            foregone_pnl=sum(entry.foregone_pnl for entry in self._records),
            dispositions=dispositions,
        )

    def _append(self, entry: CounterfactualRecord) -> None:
        if self.path is None:
            return
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(entry.as_record()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def ghost_orders_from_preflight(
    intended: Iterable[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    prices: Mapping[str, float],
) -> tuple[GhostOrder, ...]:
    """Build ghost orders from an intended batch and its risk preflight verdicts.

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
            disposition, reason, executed = "executed", "allowed", quantity
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
]
