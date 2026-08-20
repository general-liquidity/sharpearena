"""Deferred resolution: predictions that are scored after the episode is over.

Every scored surface in SharpeArena resolves inside the episode. That silently restricts
which questions can be asked at all: an earnings outcome, a macro print, or a multi-month
thesis has no answer before the last bar, so today it simply cannot be posed. OpenFinGym
("A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents") makes the
deferred case a first-class task type, and this module is that mechanism: an agent commits
a claim with a resolution horizon, the episode ends, and the claim is scored later, when
the resolving observation exists.

**How resolution stays leak-free.** The rest of the repo makes look-ahead structurally
impossible rather than forbidden, and the same approach is what this module relies on.

* :class:`DeferredDesk` is the only object the agent touches, and its entire state is a
  list of claims and one integer clock. It holds no dataset, no series, no env, and no
  reference to anything that could produce a future value. There is no code path from a
  desk to resolving data, so "the agent cannot see the answer at commit time" is a fact
  about the object's shape rather than a rule that something has to enforce.
* The desk stamps ``committed_at`` from its own monotonic clock, which only the harness
  advances via :meth:`DeferredDesk.tick`. The agent never supplies a timestamp, so a claim
  cannot be backdated onto information it did not have or forward-dated past it.
* The desk exposes no score and computes none. Scoring is :func:`resolve_claims`, a free
  function over claims and outcomes that the desk has no handle on, so an in-episode
  reward cannot read a deferred result: there is nothing on the desk to read.
* An :class:`Outcome` carries ``available_at``, the bar at which that datum first existed.
  :func:`resolve_claims` refuses any outcome available at or before the claim's commit bar
  (:class:`LeakedResolution`) and any outcome that predates the claim's stated horizon
  (:class:`UnresolvedClaim`). This is the single boundary check in the module, and it is a
  precondition on the data rather than an inspection of the agent's behaviour.

Claims survive the episode by design: :meth:`DeferredDesk.to_json` and
:func:`claims_from_json` round-trip them, so the commit and the resolution can be separate
processes, separate machines, or separate days.

Nothing here uses an RNG, a clock, or I/O.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

#: A point forecast of a quantity: ``prediction = (value,)``.
POINT = "point"
#: A probability in ``[0, 1]`` for a binary event: ``prediction = (p,)``.
PROBABILITY = "probability"
#: A signed direction, ``-1`` or ``+1``: ``prediction = (sign,)``.
DIRECTION = "direction"
#: A closed interval: ``prediction = (lo, hi)`` with ``lo <= hi``.
INTERVAL = "interval"

KINDS = (POINT, PROBABILITY, DIRECTION, INTERVAL)


class DeferredError(RuntimeError):
    """Base class for every refusal in this module."""


class LeakedResolution(DeferredError):
    """The resolving datum already existed when the claim was committed.

    Raised when an :class:`Outcome`'s ``available_at`` is at or before the claim's
    ``committed_at``. Such a claim is not a forecast: the answer was on the tape when it
    was written, so scoring it would reward look-ahead.
    """


class UnresolvedClaim(DeferredError):
    """The resolving datum predates the horizon the claim was committed against.

    A claim that says "in 20 bars" is not settled by a value observed at bar 5, even a
    correct one. Scoring it there would let a caller pick the horizon that flatters the
    prediction after the fact.
    """


class ClaimRejected(DeferredError):
    """The claim is malformed: an unknown kind, a bad payload, or a non-positive horizon."""


@dataclass(frozen=True)
class Claim:
    """One committed prediction. Immutable, and carries no resolving data of any kind.

    ``committed_at`` is the bar index of the last observation the agent had when it wrote
    the claim; ``horizon`` is how many bars later the answer exists. Everything in the
    dataclass was knowable at commit time, which is what lets a claim be serialized,
    shipped, and stored without any possibility of it carrying the answer with it.
    """

    claim_id: str
    question: str
    kind: str
    prediction: tuple[float, ...]
    committed_at: int
    horizon: int
    confidence: float = 0.5
    rationale: str = ""

    @property
    def resolves_at(self) -> int:
        """The earliest bar at which this claim can legitimately be settled."""
        return self.committed_at + self.horizon

    def to_dict(self) -> dict:
        """A JSON-friendly dict. Round-trips through :func:`claims_from_json`."""
        return {
            "claim_id": self.claim_id,
            "question": self.question,
            "kind": self.kind,
            "prediction": [float(x) for x in self.prediction],
            "committed_at": int(self.committed_at),
            "horizon": int(self.horizon),
            "resolves_at": int(self.resolves_at),
            "confidence": float(self.confidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Outcome:
    """The resolving observation for one claim, plus when it first existed.

    ``available_at`` is not metadata: it is the field that makes leak-freedom checkable.
    A caller that constructs an :class:`Outcome` is asserting the bar at which the datum
    became observable, and :func:`resolve_claims` refuses to settle a claim against a datum
    that was already observable when the claim was written.
    """

    claim_id: str
    value: float
    available_at: int


class DeferredDesk:
    """The commit side: where an agent writes claims it cannot yet be scored on.

    Constructed by the harness and handed to the agent. The harness advances the clock with
    :meth:`tick` once per bar; the agent calls :meth:`commit`. The desk holds claims and an
    integer, and that is the whole of its state, which is why nothing an agent can do with
    it reaches resolving data.

    ``min_horizon`` is the shortest deferral the desk accepts. It defaults to ``1``: a
    horizon of zero would be a claim about the bar already observed, which is not a
    prediction.
    """

    def __init__(self, *, min_horizon: int = 1, max_open: Optional[int] = None) -> None:
        if int(min_horizon) < 1:
            raise ValueError("min_horizon must be >= 1")
        self._min_horizon = int(min_horizon)
        self._max_open = None if max_open is None else int(max_open)
        self._now = -1
        self._claims: list[Claim] = []

    @property
    def now(self) -> int:
        """The bar the desk's clock currently stands at (``-1`` before the first tick)."""
        return self._now

    def tick(self, bar: int) -> None:
        """Advance the clock to ``bar``. Monotonic: the harness cannot rewind the desk."""
        bar = int(bar)
        if bar < self._now:
            raise ValueError(f"the desk clock cannot move backwards ({bar} < {self._now})")
        self._now = bar

    def commit(
        self,
        question: str,
        kind: str,
        prediction: Sequence[float] | float,
        *,
        horizon: int,
        confidence: float = 0.5,
        rationale: str = "",
        claim_id: Optional[str] = None,
    ) -> Claim:
        """Write one claim, stamped with the desk's own clock.

        There is deliberately no ``committed_at`` parameter. The commit bar is the desk's
        clock, which only the harness moves, so a claim is always attributed to exactly the
        information the agent had, and a late claim cannot be presented as an early one.
        """
        if self._now < 0:
            raise ClaimRejected("the desk clock has not been started (call tick first)")
        if int(horizon) < self._min_horizon:
            raise ClaimRejected(
                f"horizon {horizon} is shorter than the desk minimum {self._min_horizon}"
            )
        if self._max_open is not None and len(self.open_claims()) >= self._max_open:
            raise ClaimRejected(f"the desk already holds {self._max_open} open claims")
        payload = _validate(kind, prediction)
        if not 0.0 <= float(confidence) <= 1.0:
            raise ClaimRejected("confidence must lie in [0, 1]")
        identifier = claim_id or f"claim-{len(self._claims):04d}"
        if any(c.claim_id == identifier for c in self._claims):
            raise ClaimRejected(f"duplicate claim_id {identifier!r}")
        claim = Claim(
            claim_id=identifier,
            question=str(question),
            kind=kind,
            prediction=payload,
            committed_at=self._now,
            horizon=int(horizon),
            confidence=float(confidence),
            rationale=str(rationale),
        )
        self._claims.append(claim)
        return claim

    def claims(self) -> tuple[Claim, ...]:
        """Every claim written, in commit order."""
        return tuple(self._claims)

    def open_claims(self, at: Optional[int] = None) -> tuple[Claim, ...]:
        """Claims whose horizon has not yet elapsed as of ``at`` (default: the clock).

        Note what this does *not* offer: there is no way to ask the desk what an open claim
        will resolve to, because the desk does not know and cannot find out.
        """
        when = self._now if at is None else int(at)
        return tuple(c for c in self._claims if c.resolves_at > when)

    def to_json(self) -> str:
        """Serialize the claims so they outlive the episode."""
        return json.dumps([c.to_dict() for c in self._claims])


def claims_from_json(payload: str) -> list[Claim]:
    """Rebuild claims written by :meth:`DeferredDesk.to_json`."""
    out: list[Claim] = []
    for raw in json.loads(payload):
        out.append(
            Claim(
                claim_id=str(raw["claim_id"]),
                question=str(raw["question"]),
                kind=str(raw["kind"]),
                prediction=_validate(str(raw["kind"]), raw["prediction"]),
                committed_at=int(raw["committed_at"]),
                horizon=int(raw["horizon"]),
                confidence=float(raw.get("confidence", 0.5)),
                rationale=str(raw.get("rationale", "")),
            )
        )
    return out


def outcomes_from_series(
    claims: Iterable[Claim],
    series: Sequence[float],
    *,
    reference: str = "level",
) -> list[Outcome]:
    """Build outcomes for ``claims`` from a full series, **after** the episode.

    ``series[i]`` is the value at bar ``i``, so a claim resolves against
    ``series[resolves_at]`` and its ``available_at`` is that same bar. This helper takes the
    whole path, which is only safe because it runs after the run it scores: it belongs to
    the resolution side, and handing it, or the series it reads, to a live episode would
    defeat the point of the module.

    ``reference`` selects what the resolving value means: ``"level"`` is the raw value at
    the resolution bar, ``"change"`` is the difference from the commit bar, and ``"return"``
    is the fractional change from the commit bar. Direction claims almost always want
    ``"change"`` or ``"return"``, since the sign of a level is not a forecast.

    Claims whose resolution bar lies past the end of the series are skipped rather than
    resolved: an unresolved claim is a real state, and inventing a value for it would be
    the one thing this module exists to prevent.
    """
    values = [float(x) for x in series]
    out: list[Outcome] = []
    for claim in claims:
        end = claim.resolves_at
        if end >= len(values) or claim.committed_at >= len(values):
            continue
        if reference == "level":
            value = values[end]
        elif reference == "change":
            value = values[end] - values[claim.committed_at]
        elif reference == "return":
            base = values[claim.committed_at]
            value = 0.0 if base == 0.0 else values[end] / base - 1.0
        else:
            raise ValueError("reference must be 'level', 'change', or 'return'")
        out.append(Outcome(claim.claim_id, value, end))
    return out


def resolve_claims(
    claims: Iterable[Claim],
    outcomes: Iterable[Outcome],
    *,
    strict: bool = True,
) -> dict:
    """Settle the claims that have resolving data, and score them.

    The integrity boundary lives here, and it is two comparisons on ``available_at``:

    * ``available_at <= committed_at`` raises :class:`LeakedResolution`. The datum was
      already on the tape when the claim was written, so the claim was not a forecast.
    * ``available_at < resolves_at`` raises :class:`UnresolvedClaim`. The claim named a
      horizon and has not reached it, so settling it early would let the horizon be chosen
      after the fact to suit the prediction.

    With ``strict=False`` those claims are recorded in ``rejected`` instead of raising,
    which is what a batch resolution run over a mixed archive wants. They are never scored
    either way.

    Returns ``{"resolved": [...], "pending": [...], "rejected": [...], "summary": {...}}``.
    A claim with no matching outcome is ``pending``, not wrong.
    """
    by_id = {c.claim_id: c for c in claims}
    seen: set[str] = set()
    resolved: list[dict] = []
    rejected: list[dict] = []

    for outcome in outcomes:
        claim = by_id.get(outcome.claim_id)
        if claim is None:
            problem = {
                "claim_id": outcome.claim_id,
                "reason": "no such claim",
            }
            if strict:
                raise DeferredError(f"outcome for unknown claim {outcome.claim_id!r}")
            rejected.append(problem)
            continue
        if outcome.available_at <= claim.committed_at:
            message = (
                f"{claim.claim_id}: the resolving datum was available at bar "
                f"{outcome.available_at}, at or before the commit bar "
                f"{claim.committed_at}"
            )
            if strict:
                raise LeakedResolution(message)
            rejected.append({"claim_id": claim.claim_id, "reason": message})
            continue
        if outcome.available_at < claim.resolves_at:
            message = (
                f"{claim.claim_id}: the resolving datum is from bar "
                f"{outcome.available_at}, before the claim's horizon bar "
                f"{claim.resolves_at}"
            )
            if strict:
                raise UnresolvedClaim(message)
            rejected.append({"claim_id": claim.claim_id, "reason": message})
            continue
        seen.add(claim.claim_id)
        record = claim.to_dict()
        record["outcome"] = float(outcome.value)
        record["available_at"] = int(outcome.available_at)
        record["score"] = score_claim(claim, outcome.value)
        resolved.append(record)

    pending = [c.to_dict() for c in by_id.values() if c.claim_id not in seen]
    return {
        "resolved": resolved,
        "pending": pending,
        "rejected": rejected,
        "summary": summarize(resolved, n_committed=len(by_id)),
    }


def score_claim(claim: Claim, outcome: float) -> dict:
    """Score one settled claim against its realized value.

    Each kind gets the scoring rule that is actually proper for it rather than a shared
    one: squared and absolute error for a point forecast, the Brier score for a
    probability, a hit for a direction, and coverage plus width for an interval. A single
    "accuracy" number across all four would be meaningless, since a wide interval and a
    confident probability fail in different ways and should be visible as different
    failures.
    """
    value = float(outcome)
    if claim.kind == POINT:
        error = value - claim.prediction[0]
        return {
            "error": error,
            "abs_error": abs(error),
            "squared_error": error * error,
        }
    if claim.kind == PROBABILITY:
        if value not in (0.0, 1.0):
            raise ClaimRejected(
                f"{claim.claim_id}: a probability claim resolves to 0 or 1, got {value}"
            )
        residual = claim.prediction[0] - value
        return {"brier": residual * residual, "realized": value}
    if claim.kind == DIRECTION:
        realized = 0.0 if value == 0.0 else math.copysign(1.0, value)
        return {
            "correct": bool(realized != 0.0 and realized == claim.prediction[0]),
            "realized_direction": realized,
            "magnitude": abs(value),
        }
    if claim.kind == INTERVAL:
        lo, hi = claim.prediction
        return {"covered": bool(lo <= value <= hi), "width": hi - lo, "realized": value}
    raise ClaimRejected(f"unknown claim kind {claim.kind!r}")


def summarize(resolved: Sequence[dict], *, n_committed: int) -> dict:
    """Aggregate settled claims by kind, and report how many are still open.

    ``resolution_rate`` is deliberately part of the summary. A deferred benchmark where
    most claims never resolve is not a benchmark with good scores, it is a benchmark that
    barely ran, and reporting only the scored subset would hide that.
    """
    by_kind: dict[str, list[dict]] = {}
    for record in resolved:
        by_kind.setdefault(record["kind"], []).append(record)

    out: dict[str, Any] = {
        "n_committed": int(n_committed),
        "n_resolved": len(resolved),
        "resolution_rate": (len(resolved) / n_committed) if n_committed else 0.0,
        "by_kind": {},
    }
    for kind, records in sorted(by_kind.items()):
        scores = [r["score"] for r in records]
        entry: dict[str, Any] = {"n": len(records)}
        if kind == POINT:
            entry["mae"] = sum(s["abs_error"] for s in scores) / len(scores)
            entry["mse"] = sum(s["squared_error"] for s in scores) / len(scores)
        elif kind == PROBABILITY:
            entry["brier"] = sum(s["brier"] for s in scores) / len(scores)
        elif kind == DIRECTION:
            entry["accuracy"] = sum(1.0 for s in scores if s["correct"]) / len(scores)
        elif kind == INTERVAL:
            entry["coverage"] = sum(1.0 for s in scores if s["covered"]) / len(scores)
            entry["mean_width"] = sum(s["width"] for s in scores) / len(scores)
        out["by_kind"][kind] = entry
    return out


def _validate(kind: str, prediction: Sequence[float] | float) -> tuple[float, ...]:
    """Normalize and check one claim payload against its kind."""
    if kind not in KINDS:
        raise ClaimRejected(f"unknown claim kind {kind!r} (expected one of {KINDS})")
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        payload = (float(prediction),)
    else:
        payload = tuple(float(x) for x in prediction)
    if not all(math.isfinite(x) for x in payload):
        raise ClaimRejected("a prediction must be finite")
    if kind == INTERVAL:
        if len(payload) != 2:
            raise ClaimRejected("an interval claim is (lo, hi)")
        if payload[0] > payload[1]:
            raise ClaimRejected("an interval claim needs lo <= hi")
        return payload
    if len(payload) != 1:
        raise ClaimRejected(f"a {kind} claim takes exactly one value")
    if kind == PROBABILITY and not 0.0 <= payload[0] <= 1.0:
        raise ClaimRejected("a probability claim must lie in [0, 1]")
    if kind == DIRECTION and payload[0] not in (-1.0, 1.0):
        raise ClaimRejected("a direction claim must be -1 or +1")
    return payload


__all__ = [
    "POINT",
    "PROBABILITY",
    "DIRECTION",
    "INTERVAL",
    "KINDS",
    "Claim",
    "Outcome",
    "DeferredDesk",
    "DeferredError",
    "LeakedResolution",
    "UnresolvedClaim",
    "ClaimRejected",
    "claims_from_json",
    "outcomes_from_series",
    "resolve_claims",
    "score_claim",
    "summarize",
]
