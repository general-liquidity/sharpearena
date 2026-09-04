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
from typing import Any, Iterable, Mapping, Optional, Sequence

from .forecast_contract import (
    BINARY_BRIER,
    BINARY_LOG,
    CATEGORICAL,
    CATEGORICAL_BRIER,
    CATEGORICAL_LOG,
    CLAIMS_SCHEMA_VERSION,
    DIRECTION,
    DIRECTION_ACCURACY,
    INTERVAL,
    INTERVAL_SCORE,
    KINDS,
    NORMAL,
    NORMAL_CRPS,
    POINT,
    POINT_ERRORS,
    PROBABILITY,
    ForecastContract,
    ForecastContractError,
    canonical_json,
)


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
    contract: ForecastContract
    prediction: tuple[float, ...]
    committed_at: int
    confidence: float = 0.5
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ClaimRejected("claim_id must be a non-empty string")
        if isinstance(self.committed_at, bool) or not isinstance(self.committed_at, int):
            raise ClaimRejected("committed_at must be an integer")
        if not self.contract.opens_at <= self.committed_at <= self.contract.deadline:
            raise ClaimRejected(
                f"committed_at {self.committed_at} lies outside contract window "
                f"[{self.contract.opens_at}, {self.contract.deadline}]"
            )
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ClaimRejected("confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ClaimRejected("confidence must be finite and lie in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "prediction", _validate(self.contract, self.prediction))
        object.__setattr__(self, "claim_id", self.claim_id.strip())
        if not isinstance(self.rationale, str):
            raise ClaimRejected("rationale must be a string")

    @property
    def question(self) -> str:
        return self.contract.question

    @property
    def kind(self) -> str:
        return self.contract.kind

    @property
    def horizon(self) -> int:
        return self.contract.resolves_at - self.committed_at

    @property
    def resolves_at(self) -> int:
        """The earliest bar at which this claim can legitimately be settled."""
        return self.contract.resolves_at

    def to_dict(self) -> dict:
        """A JSON-friendly dict. Round-trips through :func:`claims_from_json`."""
        return {
            "claim_id": self.claim_id,
            "contract": self.contract.to_dict(),
            "contract_sha256": self.contract.sha256,
            "prediction": [float(x) for x in self.prediction],
            "committed_at": int(self.committed_at),
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
    value: float | str
    available_at: int

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ClaimRejected("outcome claim_id must be a non-empty string")
        if isinstance(self.available_at, bool) or not isinstance(self.available_at, int):
            raise ClaimRejected("outcome available_at must be an integer")
        if isinstance(self.value, bool):
            object.__setattr__(self, "value", float(self.value))
        elif isinstance(self.value, (int, float)):
            value = float(self.value)
            if not math.isfinite(value):
                raise ClaimRejected("outcome value must be finite")
            object.__setattr__(self, "value", value)
        elif not isinstance(self.value, str) or not self.value.strip():
            raise ClaimRejected("outcome value must be a finite number or non-empty category")


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
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise ClaimRejected("horizon must be an integer")
        if horizon < self._min_horizon:
            raise ClaimRejected(
                f"horizon {horizon} is shorter than the desk minimum {self._min_horizon}"
            )
        if self._max_open is not None and len(self.open_claims()) >= self._max_open:
            raise ClaimRejected(f"the desk already holds {self._max_open} open claims")
        identifier = claim_id or f"claim-{len(self._claims):04d}"
        try:
            contract = ForecastContract.legacy(
                contract_id=f"{identifier}-contract",
                question=str(question),
                kind=kind,
                committed_at=self._now,
                horizon=horizon,
            )
        except (KeyError, ForecastContractError, ValueError) as error:
            raise ClaimRejected(str(error)) from error
        return self.commit_contract(
            contract,
            prediction,
            confidence=confidence,
            rationale=rationale,
            claim_id=identifier,
        )

    def commit_contract(
        self,
        contract: ForecastContract,
        prediction: Sequence[float] | float,
        *,
        confidence: float = 0.5,
        rationale: str = "",
        claim_id: Optional[str] = None,
    ) -> Claim:
        """Commit against a contract whose deadline and resolution were fixed first.

        Unlike :meth:`commit`, this is suitable for revisions and independently
        administered prospective runs: the contract exists before the prediction and
        the prediction cannot move its own resolution horizon.
        """

        if self._now < 0:
            raise ClaimRejected("the desk clock has not been started (call tick first)")
        if not contract.opens_at <= self._now <= contract.deadline:
            raise ClaimRejected(
                f"contract {contract.contract_id!r} is not open at bar {self._now}"
            )
        if contract.resolves_at - self._now < self._min_horizon:
            raise ClaimRejected(
                f"contract resolves too soon for the desk minimum {self._min_horizon}"
            )
        if self._max_open is not None and len(self.open_claims()) >= self._max_open:
            raise ClaimRejected(f"the desk already holds {self._max_open} open claims")
        identifier = claim_id or f"claim-{len(self._claims):04d}"
        if any(c.claim_id == identifier for c in self._claims):
            raise ClaimRejected(f"duplicate claim_id {identifier!r}")
        claim = Claim(
            claim_id=identifier,
            contract=contract,
            prediction=_validate(contract, prediction),
            committed_at=self._now,
            confidence=float(confidence),
            rationale=rationale,
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
        return canonical_json(
            {
                "schema_version": CLAIMS_SCHEMA_VERSION,
                "claims": [c.to_dict() for c in self._claims],
            }
        )


def claims_from_json(payload: str) -> list[Claim]:
    """Strictly rebuild the closed v1 document written by :meth:`DeferredDesk.to_json`."""

    try:
        document = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ClaimRejected(f"claims document is not valid JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"schema_version", "claims"}:
        raise ClaimRejected(
            "claims document must contain exactly schema_version and claims"
        )
    if document["schema_version"] != CLAIMS_SCHEMA_VERSION:
        raise ClaimRejected(
            f"unsupported claims schema_version {document['schema_version']!r}"
        )
    if not isinstance(document["claims"], list):
        raise ClaimRejected("claims must be an array")
    out: list[Claim] = []
    seen: set[str] = set()
    expected = {
        "claim_id",
        "contract",
        "contract_sha256",
        "prediction",
        "committed_at",
        "resolves_at",
        "confidence",
        "rationale",
    }
    for index, raw in enumerate(document["claims"]):
        if not isinstance(raw, Mapping):
            raise ClaimRejected(f"claims[{index}] must be an object")
        if set(raw) != expected:
            raise ClaimRejected(
                f"claims[{index}] fields do not match v1; "
                f"missing={sorted(expected - set(raw))}, unknown={sorted(set(raw) - expected)}"
            )
        try:
            contract = ForecastContract.from_dict(raw["contract"])
        except (ForecastContractError, TypeError, ValueError) as error:
            raise ClaimRejected(f"claims[{index}].contract: {error}") from error
        if raw["contract_sha256"] != contract.sha256:
            raise ClaimRejected(f"claims[{index}] contract_sha256 does not match contract")
        if isinstance(raw["committed_at"], bool) or not isinstance(raw["committed_at"], int):
            raise ClaimRejected(f"claims[{index}].committed_at must be an integer")
        if raw["resolves_at"] != contract.resolves_at:
            raise ClaimRejected(f"claims[{index}].resolves_at does not match contract")
        if not isinstance(raw["rationale"], str):
            raise ClaimRejected(f"claims[{index}].rationale must be a string")
        claim = Claim(
            claim_id=raw["claim_id"],
            contract=contract,
            prediction=_validate(contract, raw["prediction"]),
            committed_at=raw["committed_at"],
            confidence=raw["confidence"],
            rationale=raw["rationale"],
        )
        if claim.claim_id in seen:
            raise ClaimRejected(f"duplicate claim_id {claim.claim_id!r} in claims document")
        seen.add(claim.claim_id)
        out.append(claim)
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
    claim_list = list(claims)
    by_id = {c.claim_id: c for c in claim_list}
    if len(by_id) != len(claim_list):
        raise ClaimRejected("claims contain duplicate claim_id values")
    seen: set[str] = set()
    seen_outcomes: set[str] = set()
    resolved: list[dict] = []
    rejected: list[dict] = []

    for outcome in outcomes:
        if outcome.claim_id in seen_outcomes:
            message = f"duplicate outcome for claim {outcome.claim_id!r}"
            if strict:
                raise DeferredError(message)
            rejected.append({"claim_id": outcome.claim_id, "reason": message, "status": "rejected"})
            continue
        seen_outcomes.add(outcome.claim_id)
        claim = by_id.get(outcome.claim_id)
        if claim is None:
            problem = {
                "claim_id": outcome.claim_id,
                "reason": "no such claim",
                "status": "rejected",
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
            rejected.append({"claim_id": claim.claim_id, "reason": message, "status": "rejected"})
            continue
        if outcome.available_at < claim.resolves_at:
            message = (
                f"{claim.claim_id}: the resolving datum is from bar "
                f"{outcome.available_at}, before the claim's horizon bar "
                f"{claim.resolves_at}"
            )
            if strict:
                raise UnresolvedClaim(message)
            rejected.append({"claim_id": claim.claim_id, "reason": message, "status": "rejected"})
            continue
        seen.add(claim.claim_id)
        record = claim.to_dict()
        record["kind"] = claim.kind
        record["status"] = "resolved"
        record["outcome"] = outcome.value
        record["available_at"] = int(outcome.available_at)
        record["score"] = score_claim(claim, outcome.value)
        resolved.append(record)

    pending = []
    for claim in by_id.values():
        if claim.claim_id not in seen:
            record = claim.to_dict()
            record["kind"] = claim.kind
            record["status"] = "pending"
            pending.append(record)
    return {
        "resolved": resolved,
        "pending": pending,
        "rejected": rejected,
        "summary": summarize(resolved, n_committed=len(by_id)),
    }


def score_claim(claim: Claim, outcome: float | str) -> dict:
    """Score one settled claim against its realized value.

    Each kind gets the scoring rule that is actually proper for it rather than a shared
    one: squared and absolute error for a point forecast, the Brier score for a
    probability, a hit for a direction, and coverage plus width for an interval. A single
    "accuracy" number across all four would be meaningless, since a wide interval and a
    confident probability fail in different ways and should be visible as different
    failures.
    """
    rule = claim.contract.scoring_rule
    if claim.kind == POINT:
        value = _numeric_outcome(claim, outcome)
        error = value - claim.prediction[0]
        return {
            "error": error,
            "abs_error": abs(error),
            "squared_error": error * error,
        }
    if claim.kind == PROBABILITY:
        value = _numeric_outcome(claim, outcome)
        if value not in (0.0, 1.0):
            raise ClaimRejected(
                f"{claim.claim_id}: a probability claim resolves to 0 or 1, got {value}"
            )
        residual = claim.prediction[0] - value
        if rule == BINARY_BRIER:
            return {"brier": residual * residual, "realized": value}
        probability = claim.prediction[0] if value == 1.0 else 1.0 - claim.prediction[0]
        return {"log_loss": -math.log(probability), "realized": value}
    if claim.kind == CATEGORICAL:
        if not isinstance(outcome, str) or outcome not in claim.contract.categories:
            raise ClaimRejected(
                f"{claim.claim_id}: categorical outcome must be one of "
                f"{claim.contract.categories}, got {outcome!r}"
            )
        outcome_index = claim.contract.categories.index(outcome)
        if rule == CATEGORICAL_BRIER:
            total = sum(
                (probability - (1.0 if index == outcome_index else 0.0)) ** 2
                for index, probability in enumerate(claim.prediction)
            )
            return {"brier": total, "realized": outcome}
        return {
            "log_loss": -math.log(claim.prediction[outcome_index]),
            "realized": outcome,
        }
    if claim.kind == NORMAL:
        value = _numeric_outcome(claim, outcome)
        mean, sigma = claim.prediction
        z = (value - mean) / sigma
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
        return {"crps": crps, "z": z, "realized": value}
    if claim.kind == DIRECTION:
        value = _numeric_outcome(claim, outcome)
        realized = (
            0.0
            if abs(value) <= claim.contract.neutral_threshold
            else math.copysign(1.0, value)
        )
        return {
            "correct": bool(realized != 0.0 and realized == claim.prediction[0]),
            "realized_direction": realized,
            "magnitude": abs(value),
        }
    if claim.kind == INTERVAL:
        value = _numeric_outcome(claim, outcome)
        lo, hi = claim.prediction
        alpha = claim.contract.interval_alpha
        assert alpha is not None
        penalty = 0.0
        if value < lo:
            penalty = 2.0 * (lo - value) / alpha
        elif value > hi:
            penalty = 2.0 * (value - hi) / alpha
        return {
            "covered": bool(lo <= value <= hi),
            "width": hi - lo,
            "interval_score": hi - lo + penalty,
            "realized": value,
        }
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
            if all("brier" in score for score in scores):
                entry["brier"] = sum(s["brier"] for s in scores) / len(scores)
            if all("log_loss" in score for score in scores):
                entry["log_loss"] = sum(s["log_loss"] for s in scores) / len(scores)
        elif kind == CATEGORICAL:
            if all("brier" in score for score in scores):
                entry["brier"] = sum(s["brier"] for s in scores) / len(scores)
            if all("log_loss" in score for score in scores):
                entry["log_loss"] = sum(s["log_loss"] for s in scores) / len(scores)
        elif kind == NORMAL:
            entry["crps"] = sum(s["crps"] for s in scores) / len(scores)
        elif kind == DIRECTION:
            entry["accuracy"] = sum(1.0 for s in scores if s["correct"]) / len(scores)
        elif kind == INTERVAL:
            entry["coverage"] = sum(1.0 for s in scores if s["covered"]) / len(scores)
            entry["mean_width"] = sum(s["width"] for s in scores) / len(scores)
            entry["interval_score"] = sum(s["interval_score"] for s in scores) / len(scores)
        out["by_kind"][kind] = entry
    return out


def _numeric_outcome(claim: Claim, outcome: float | str) -> float:
    if isinstance(outcome, bool) or not isinstance(outcome, (int, float)):
        raise ClaimRejected(f"{claim.claim_id}: {claim.kind} outcome must be numeric")
    value = float(outcome)
    if not math.isfinite(value):
        raise ClaimRejected(f"{claim.claim_id}: outcome must be finite")
    return value


def _validate(
    contract: ForecastContract, prediction: Sequence[float] | float
) -> tuple[float, ...]:
    """Normalize and check one claim payload against its kind."""
    kind = contract.kind
    if kind not in KINDS:
        raise ClaimRejected(f"unknown claim kind {kind!r} (expected one of {KINDS})")
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        payload = (float(prediction),)
    else:
        if isinstance(prediction, (str, bytes)):
            raise ClaimRejected("prediction must be numeric or an array of numbers")
        try:
            payload = tuple(float(x) for x in prediction)
        except (TypeError, ValueError) as error:
            raise ClaimRejected("prediction must be numeric or an array of numbers") from error
    if not all(math.isfinite(x) for x in payload):
        raise ClaimRejected("a prediction must be finite")
    if kind == INTERVAL:
        if len(payload) != 2:
            raise ClaimRejected("an interval claim is (lo, hi)")
        if payload[0] > payload[1]:
            raise ClaimRejected("an interval claim needs lo <= hi")
        return payload
    if kind in (CATEGORICAL, NORMAL):
        expected = len(contract.categories) if kind == CATEGORICAL else 2
        if len(payload) != expected:
            raise ClaimRejected(f"a {kind} claim takes exactly {expected} values")
        if kind == CATEGORICAL:
            if any(value < 0.0 or value > 1.0 for value in payload):
                raise ClaimRejected("categorical probabilities must lie in [0, 1]")
            if not math.isclose(sum(payload), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ClaimRejected("categorical probabilities must sum to 1")
            if contract.scoring_rule == CATEGORICAL_LOG and any(value <= 0.0 for value in payload):
                raise ClaimRejected(
                    "categorical_log requires every reported probability to be positive"
                )
        elif payload[1] <= 0.0:
            raise ClaimRejected("a normal forecast requires a positive standard deviation")
        return payload
    if len(payload) != 1:
        raise ClaimRejected(f"a {kind} claim takes exactly one value")
    if kind == PROBABILITY and not 0.0 <= payload[0] <= 1.0:
        raise ClaimRejected("a probability claim must lie in [0, 1]")
    if kind == PROBABILITY and contract.scoring_rule == BINARY_LOG and not 0.0 < payload[0] < 1.0:
        raise ClaimRejected("binary_log requires probability strictly inside (0, 1)")
    if kind == DIRECTION and payload[0] not in (-1.0, 1.0):
        raise ClaimRejected("a direction claim must be -1 or +1")
    return payload


__all__ = [
    "POINT",
    "PROBABILITY",
    "CATEGORICAL",
    "NORMAL",
    "DIRECTION",
    "INTERVAL",
    "KINDS",
    "ForecastContract",
    "ForecastContractError",
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
