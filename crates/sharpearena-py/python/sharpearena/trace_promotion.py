"""Strict promotion of a recorded rollout trace into an offline regression case.

:mod:`sharpearena.trace` is a permissive reader on purpose: it skips a malformed
line and reads a missing reward as ``0.0`` so a half-written trace can still be
eyeballed. That behaviour is correct for exploration and disqualifying for
promotion, because a case frozen from a partially-read trace asserts something
about a run that never happened. This module adds a strict path beside it and
does not change it.

The pipeline is deliberately linear and every stage refuses to guess.

``load_trace_strict``
    Rejects, rather than skips, a blank line, a non-object line, a non-``step``
    record with an unexpected kind, a step whose ordinal is out of sequence, a
    missing or non-finite reward, an absent decision, or a meta record that is
    absent, duplicated, out of position, or missing a fingerprint input. Nothing
    is defaulted.

``compute_trace_fingerprint``
    A deterministic identity over six components: environment, model, scaffold,
    contract, data and the process-event sequence. Two runs that agree on all six
    are the same case; a run that differs in any of them is a different case and
    must not silently reuse another's decision.

``run_promotion_checks``
    Deterministic assertions over the loaded trace. These run BEFORE any
    scoring, ranking or judgement, and a block-severity failure is what makes a
    trace a promotion candidate at all.

``SilverStore`` / ``OperatorDecision`` / ``promote_to_gold``
    A silver candidate is immutable once written: re-appending the same id with
    different content raises. A silver candidate becomes gold only through an
    explicitly recorded operator decision carrying an identity and a rationale.
    There is no code path that transitions a candidate on its own.

``GoldCase``
    A gold case freezes a MINIMAL scenario plus the named invariant that must
    hold on it. It does not freeze the transcript. A transcript-shaped case
    passes for the wrong reason: it re-asserts one recorded conversation instead
    of the property the failure was about.

``evaluate_gold_case``
    Runs the frozen invariant against the frozen scenario with sockets disabled
    and no model in the loop, so a gold case is executable in CI on a machine
    with no network and no credentials.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

PROMOTION_SCHEMA_VERSION = "sharpearena.promotion/1.0.0"
SILVER_EVIDENCE_CLASS = "promotion_silver_candidate"
GOLD_EVIDENCE_CLASS = "promotion_gold_case"

#: Meta fields a trace must carry to be promotable. Each one feeds the
#: fingerprint; a trace that cannot say which model or dataset produced it
#: cannot be turned into a regression case.
REQUIRED_META_FIELDS = (
    "schema_version",
    "environment_id",
    "model_digest",
    "scaffold_digest",
    "contract_version",
    "dataset_sha256",
    "n_steps",
    "scenario_seeds",
)

#: Observation keys that would let an offline re-scorer peek ahead.
FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {"future_close", "next_close", "future_returns", "label", "full_series"}
)

MINIMAL_INFO_KEYS = ("scenario_seed", "events", "terminated", "truncated")


class TraceIntegrityError(ValueError):
    """A trace is malformed or incomplete and must not be promoted."""


class PromotionError(RuntimeError):
    """A promotion step was attempted out of order or without authority."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class TraceFingerprint:
    """Six-component deterministic identity for a recorded run."""

    environment: str
    model: str
    scaffold: str
    contract: str
    data: str
    process: str

    @property
    def composite(self) -> str:
        return _digest(
            {
                "environment": self.environment,
                "model": self.model,
                "scaffold": self.scaffold,
                "contract": self.contract,
                "data": self.data,
                "process": self.process,
            }
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "model": self.model,
            "scaffold": self.scaffold,
            "contract": self.contract,
            "data": self.data,
            "process": self.process,
            "composite": self.composite,
        }


@dataclass(frozen=True)
class StrictTrace:
    """A fully validated trace. Every field present, nothing defaulted."""

    steps: tuple[dict[str, Any], ...]
    meta: dict[str, Any]
    source_sha256: str

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(float(step["reward"]) for step in self.steps)

    @property
    def fingerprint(self) -> TraceFingerprint:
        return compute_trace_fingerprint(self.steps, self.meta)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceIntegrityError(message)


def _validate_step(record: Mapping[str, Any], expected_ordinal: int, where: str) -> dict[str, Any]:
    _require(record.get("kind") == "step", f"{where}: expected a step record")
    for name in ("step", "observation", "decision"):
        _require(name in record, f"{where}: step record is missing {name!r}")
    _require(
        "reward" in record,
        f"{where}: step record has no reward; a missing reward is not zero",
    )
    ordinal = record["step"]
    _require(
        not isinstance(ordinal, bool) and isinstance(ordinal, int),
        f"{where}: step ordinal must be an integer",
    )
    _require(
        ordinal == expected_ordinal,
        f"{where}: step ordinal {ordinal} breaks the sequence at {expected_ordinal}",
    )
    reward = record["reward"]
    _require(
        not isinstance(reward, bool) and isinstance(reward, (int, float)),
        f"{where}: reward must be a number; a missing reward is not zero",
    )
    _require(math.isfinite(float(reward)), f"{where}: reward must be finite")
    _require(record["decision"] is not None, f"{where}: decision must not be null")
    info = record.get("info", {})
    _require(isinstance(info, dict), f"{where}: info must be an object")
    return {
        "kind": "step",
        "step": ordinal,
        "observation": record["observation"],
        "decision": record["decision"],
        "reward": float(reward),
        "info": info,
    }


def load_trace_strict(path: str | Path) -> StrictTrace:
    """Load a JSONL trace, rejecting anything malformed or incomplete.

    Unlike :func:`sharpearena.trace.load_trace` this never skips a line and never
    substitutes a value. Any defect raises :class:`TraceIntegrityError`.
    """

    path = Path(path)
    raw = path.read_bytes()
    steps: list[dict[str, Any]] = []
    meta: Optional[dict[str, Any]] = None
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        where = f"{path}:{number}"
        _require(line.strip() != "", f"{where}: blank line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceIntegrityError(f"{where}: not JSON: {error}") from error
        _require(isinstance(record, dict), f"{where}: record must be an object")
        kind = record.get("kind")
        if kind == "meta":
            _require(meta is None, f"{where}: a trace carries exactly one meta record")
            meta = dict(record)
            continue
        _require(meta is None, f"{where}: step record appears after the meta record")
        steps.append(_validate_step(record, len(steps), where))
    _require(meta is not None, f"{path}: trace has no meta record and is incomplete")
    assert meta is not None
    _require(len(steps) >= 2, f"{path}: a promotable trace needs at least two steps")
    for name in REQUIRED_META_FIELDS:
        _require(name in meta, f"{path}: meta is missing required field {name!r}")
    for name in (
        "environment_id",
        "model_digest",
        "scaffold_digest",
        "contract_version",
        "dataset_sha256",
        "schema_version",
    ):
        value = meta[name]
        _require(
            isinstance(value, str) and value.strip() != "",
            f"{path}: meta.{name} must be a non-empty string",
        )
    _require(
        meta["n_steps"] == len(steps),
        f"{path}: meta.n_steps is {meta['n_steps']} but {len(steps)} step records were read",
    )
    seeds = meta["scenario_seeds"]
    _require(
        isinstance(seeds, list)
        and seeds
        and all(not isinstance(s, bool) and isinstance(s, int) for s in seeds),
        f"{path}: meta.scenario_seeds must be a non-empty integer array",
    )
    return StrictTrace(tuple(steps), meta, sha256(raw).hexdigest())


def process_events(steps: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The deterministic process-event sequence a fingerprint is taken over.

    Derived from what the agent did, not from what it earned: the number and
    direction of orders it emitted, the events the environment recorded, and
    whether the lane terminated. Two runs with identical rewards but different
    action sequences are different processes and fingerprint differently.
    """

    events: list[dict[str, Any]] = []
    for step in steps:
        decision = step["decision"]
        orders = decision.get("orders", []) if isinstance(decision, dict) else []
        actions = tuple(
            str(order.get("action", "unknown"))
            for order in orders
            if isinstance(order, dict)
        )
        info = step.get("info", {})
        recorded = info.get("events", []) if isinstance(info, dict) else []
        events.append(
            {
                "step": step["step"],
                "n_orders": len(orders) if isinstance(orders, list) else 0,
                "actions": sorted(actions),
                "env_events": sorted(str(item) for item in recorded)
                if isinstance(recorded, list)
                else [],
                "terminated": bool(info.get("terminated", False))
                if isinstance(info, dict)
                else False,
            }
        )
    return tuple(events)


def compute_trace_fingerprint(
    steps: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]
) -> TraceFingerprint:
    """Fingerprint over environment, model, scaffold, contract, data and process."""

    return TraceFingerprint(
        environment=_digest(
            {"environment_id": meta["environment_id"], "config": meta.get("config", {})}
        ),
        model=_digest({"model_digest": meta["model_digest"]}),
        scaffold=_digest({"scaffold_digest": meta["scaffold_digest"]}),
        contract=_digest(
            {
                "contract_version": meta["contract_version"],
                "schema_version": meta["schema_version"],
            }
        ),
        data=_digest(
            {
                "dataset_sha256": meta["dataset_sha256"],
                "scenario_seeds": sorted(int(seed) for seed in meta["scenario_seeds"]),
            }
        ),
        process=_digest(list(process_events(steps))),
    )


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    #: ``block`` failures make a trace a promotion candidate; ``warn`` do not.
    severity: str
    passed: bool
    detail: str
    #: Step ordinals implicated, used to minimize the frozen scenario.
    implicated_steps: tuple[int, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "passed": self.passed,
            "detail": self.detail,
            "implicated_steps": list(self.implicated_steps),
        }


def _check_no_lookahead(trace: StrictTrace) -> CheckResult:
    implicated = []
    seen: set[str] = set()
    for step in trace.steps:
        observation = step["observation"]
        if isinstance(observation, dict):
            leaked = FORBIDDEN_OBSERVATION_KEYS & set(observation)
            if leaked:
                implicated.append(step["step"])
                seen |= leaked
    return CheckResult(
        "no_lookahead_in_observation",
        "block",
        not implicated,
        "no forbidden observation keys"
        if not implicated
        else f"observations expose future-bearing keys {sorted(seen)}",
        tuple(implicated),
    )


def _check_rewards_finite(trace: StrictTrace) -> CheckResult:
    implicated = tuple(
        step["step"] for step in trace.steps if not math.isfinite(step["reward"])
    )
    return CheckResult(
        "rewards_finite",
        "block",
        not implicated,
        "every reward is finite"
        if not implicated
        else f"non-finite rewards at steps {list(implicated)}",
        implicated,
    )


def _check_decision_shape(trace: StrictTrace) -> CheckResult:
    implicated = tuple(
        step["step"]
        for step in trace.steps
        if not isinstance(step["decision"], dict)
        or not isinstance(step["decision"].get("orders", []), list)
    )
    return CheckResult(
        "decision_is_structured",
        "block",
        not implicated,
        "every decision is a structured order set"
        if not implicated
        else f"unstructured decisions at steps {list(implicated)}",
        implicated,
    )


def _check_seeds_declared(trace: StrictTrace) -> CheckResult:
    declared = {int(seed) for seed in trace.meta["scenario_seeds"]}
    observed = {
        int(step["info"]["scenario_seed"])
        for step in trace.steps
        if isinstance(step.get("info"), dict) and "scenario_seed" in step["info"]
    }
    missing = sorted(observed - declared)
    return CheckResult(
        "seeds_declared_in_meta",
        "block",
        not missing,
        "every observed scenario seed is declared in meta"
        if not missing
        else f"steps used undeclared seeds {missing}",
        tuple(
            step["step"]
            for step in trace.steps
            if isinstance(step.get("info"), dict)
            and step["info"].get("scenario_seed") in missing
        ),
    )


def _check_not_degenerate(trace: StrictTrace) -> CheckResult:
    rewards = trace.rewards
    flat = len(set(rewards)) <= 1
    return CheckResult(
        "reward_series_varies",
        "warn",
        not flat,
        "reward series varies"
        if not flat
        else "every reward is identical; the lane may not have traded",
        tuple(step["step"] for step in trace.steps) if flat else (),
    )


PROMOTION_CHECKS = (
    _check_rewards_finite,
    _check_no_lookahead,
    _check_decision_shape,
    _check_seeds_declared,
    _check_not_degenerate,
)


def run_promotion_checks(trace: StrictTrace) -> tuple[CheckResult, ...]:
    """Deterministic checks, run before any scoring or judgement."""

    return tuple(check(trace) for check in PROMOTION_CHECKS)


def blocking_failures(results: Sequence[CheckResult]) -> tuple[CheckResult, ...]:
    return tuple(
        result
        for result in results
        if result.severity == "block" and not result.passed
    )


def _run_named_check(trace: StrictTrace, check_id: str) -> CheckResult:
    for check in PROMOTION_CHECKS:
        result = check(trace)
        if result.check_id == check_id:
            return result
    raise PromotionError(f"unknown check {check_id!r}")


def _reduced_step(step: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    info = step.get("info", {})
    return {
        "kind": "step",
        "step": ordinal,
        "observation": step["observation"],
        "decision": step["decision"],
        "reward": step["reward"],
        "info": {key: info[key] for key in MINIMAL_INFO_KEYS if key in info},
    }


def _rebuild(trace: StrictTrace, steps: Sequence[Mapping[str, Any]]) -> StrictTrace:
    renumbered = tuple(
        _reduced_step(step, ordinal) for ordinal, step in enumerate(steps)
    )
    meta = dict(trace.meta)
    meta["n_steps"] = len(renumbered)
    return StrictTrace(renumbered, meta, trace.source_sha256)


def minimize_scenario(trace: StrictTrace, check_id: str) -> StrictTrace:
    """Shrink a trace to the smallest window on which ``check_id`` still fails.

    Greedy two-sided shrink, floored at two steps because a strict trace needs
    two. The result is the frozen scenario: small enough to read, and still a
    genuine reproduction rather than a summary of one.
    """

    if _run_named_check(trace, check_id).passed:
        raise PromotionError(
            f"check {check_id!r} passes on the full trace; there is nothing to minimize"
        )
    steps = list(trace.steps)
    changed = True
    while changed and len(steps) > 2:
        changed = False
        for candidate in (steps[1:], steps[:-1]):
            if len(candidate) < 2:
                continue
            if not _run_named_check(_rebuild(trace, candidate), check_id).passed:
                steps = candidate
                changed = True
                break
    return _rebuild(trace, steps)


@dataclass(frozen=True)
class SilverCandidate:
    """An immutable promotion candidate. Not yet a regression case."""

    candidate_id: str
    triggering_check: str
    severity: str
    detail: str
    source_trace_sha256: str
    fingerprint: TraceFingerprint
    scenario: dict[str, Any]
    expected_invariant: dict[str, Any]
    created_at_unix_ns: int

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "evidence_class": SILVER_EVIDENCE_CLASS,
            "candidate_id": self.candidate_id,
            "triggering_check": self.triggering_check,
            "severity": self.severity,
            "detail": self.detail,
            "source_trace_sha256": self.source_trace_sha256,
            "fingerprint": self.fingerprint.as_record(),
            "scenario": self.scenario,
            "expected_invariant": self.expected_invariant,
            "created_at_unix_ns": self.created_at_unix_ns,
        }

    @property
    def content_sha256(self) -> str:
        record = self.as_record()
        record.pop("created_at_unix_ns")
        return _digest(record)


def _scenario_payload(minimal: StrictTrace) -> dict[str, Any]:
    return {
        "steps": list(minimal.steps),
        "meta": {
            key: minimal.meta[key]
            for key in (*REQUIRED_META_FIELDS, "config")
            if key in minimal.meta
        },
    }


def build_silver_candidate(
    trace: StrictTrace, failure: CheckResult, *, now_unix_ns: Optional[int] = None
) -> SilverCandidate:
    """Freeze a MINIMAL reproduction plus the invariant that must hold on it."""

    if failure.passed:
        raise PromotionError("only a failing check produces a promotion candidate")
    minimal = minimize_scenario(trace, failure.check_id)
    scenario = _scenario_payload(minimal)
    fingerprint = trace.fingerprint
    candidate_id = _digest(
        {
            "fingerprint": fingerprint.composite,
            "check": failure.check_id,
            "scenario": scenario,
        }
    )[:32]
    return SilverCandidate(
        candidate_id=candidate_id,
        triggering_check=failure.check_id,
        severity=failure.severity,
        detail=failure.detail,
        source_trace_sha256=trace.source_sha256,
        fingerprint=fingerprint,
        scenario=scenario,
        expected_invariant={
            "check_id": failure.check_id,
            "must_pass": True,
            "rationale": "the frozen scenario reproduces the failure; a fix is "
            "what makes this check pass on it",
        },
        created_at_unix_ns=time.time_ns() if now_unix_ns is None else int(now_unix_ns),
    )


class SilverStore:
    """Append-only, immutable silver queue.

    Re-appending an id whose content differs from the stored row raises. There is
    no ``status`` field: a candidate does not change state in place, it is either
    superseded by a recorded operator decision or it is not.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, candidate: SilverCandidate) -> bool:
        """Write the candidate. Returns False if it was already stored verbatim."""

        for existing in self.read():
            if existing.candidate_id == candidate.candidate_id:
                if existing.content_sha256 == candidate.content_sha256:
                    return False
                raise PromotionError(
                    f"silver candidate {candidate.candidate_id} is already stored with "
                    "different content; silver rows are immutable"
                )
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(candidate.as_record()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def read(self) -> tuple[SilverCandidate, ...]:
        """Strict read. A malformed row raises rather than being skipped."""

        if not self.path.exists():
            return ()
        out: list[SilverCandidate] = []
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            where = f"{self.path}:{number}"
            _require(line.strip() != "", f"{where}: blank line in the silver queue")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TraceIntegrityError(f"{where}: not JSON: {error}") from error
            _require(isinstance(row, dict), f"{where}: row must be an object")
            missing = sorted(
                {
                    "candidate_id",
                    "triggering_check",
                    "severity",
                    "detail",
                    "source_trace_sha256",
                    "fingerprint",
                    "scenario",
                    "expected_invariant",
                    "created_at_unix_ns",
                }
                - set(row)
            )
            _require(not missing, f"{where}: row is missing {missing}")
            fingerprint_row = row["fingerprint"]
            out.append(
                SilverCandidate(
                    candidate_id=row["candidate_id"],
                    triggering_check=row["triggering_check"],
                    severity=row["severity"],
                    detail=row["detail"],
                    source_trace_sha256=row["source_trace_sha256"],
                    fingerprint=TraceFingerprint(
                        environment=fingerprint_row["environment"],
                        model=fingerprint_row["model"],
                        scaffold=fingerprint_row["scaffold"],
                        contract=fingerprint_row["contract"],
                        data=fingerprint_row["data"],
                        process=fingerprint_row["process"],
                    ),
                    scenario=row["scenario"],
                    expected_invariant=row["expected_invariant"],
                    created_at_unix_ns=int(row["created_at_unix_ns"]),
                )
            )
        return tuple(out)


@dataclass(frozen=True)
class OperatorDecision:
    """An explicit, attributable human decision. Nothing else promotes."""

    candidate_id: str
    #: ``promote`` or ``reject``.
    decision: str
    operator: str
    rationale: str
    decided_at_unix_ns: int

    def __post_init__(self) -> None:
        if self.decision not in {"promote", "reject"}:
            raise PromotionError("decision must be 'promote' or 'reject'")
        if not self.operator.strip():
            raise PromotionError("an operator decision must name its operator")
        if len(self.rationale.strip()) < 20:
            raise PromotionError(
                "an operator decision requires a rationale of at least 20 characters"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "operator": self.operator,
            "rationale": self.rationale,
            "decided_at_unix_ns": self.decided_at_unix_ns,
        }


@dataclass(frozen=True)
class GoldCase:
    """A frozen, offline-executable regression case. No transcript."""

    case_id: str
    triggering_check: str
    scenario: dict[str, Any]
    expected_invariant: dict[str, Any]
    source_trace_sha256: str
    fingerprint: TraceFingerprint
    decision: OperatorDecision

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "evidence_class": GOLD_EVIDENCE_CLASS,
            "case_id": self.case_id,
            "triggering_check": self.triggering_check,
            "scenario": self.scenario,
            "expected_invariant": self.expected_invariant,
            "source_trace_sha256": self.source_trace_sha256,
            "fingerprint": self.fingerprint.as_record(),
            "operator_decision": self.decision.as_record(),
            "offline": {
                "network": "forbidden",
                "model_calls": "none",
            },
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(_canonical_bytes(self.as_record()) + b"\n")
        temporary.replace(path)
        return path


def promote_to_gold(
    candidate: SilverCandidate, decision: OperatorDecision
) -> GoldCase:
    """Freeze a silver candidate as a gold case under a recorded human decision."""

    if decision.candidate_id != candidate.candidate_id:
        raise PromotionError(
            "the operator decision names a different candidate than the one supplied"
        )
    if decision.decision != "promote":
        raise PromotionError(
            f"candidate {candidate.candidate_id} was {decision.decision}ed, not promoted"
        )
    return GoldCase(
        case_id=candidate.candidate_id,
        triggering_check=candidate.triggering_check,
        scenario=candidate.scenario,
        expected_invariant=candidate.expected_invariant,
        source_trace_sha256=candidate.source_trace_sha256,
        fingerprint=candidate.fingerprint,
        decision=decision,
    )


def load_gold_case(path: Path) -> GoldCase:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(row, dict), f"{path}: gold case must be an object")
    _require(
        row.get("schema_version") == PROMOTION_SCHEMA_VERSION,
        f"{path}: gold case schema version mismatch",
    )
    fingerprint_row = row["fingerprint"]
    decision = row["operator_decision"]
    return GoldCase(
        case_id=row["case_id"],
        triggering_check=row["triggering_check"],
        scenario=row["scenario"],
        expected_invariant=row["expected_invariant"],
        source_trace_sha256=row["source_trace_sha256"],
        fingerprint=TraceFingerprint(
            environment=fingerprint_row["environment"],
            model=fingerprint_row["model"],
            scaffold=fingerprint_row["scaffold"],
            contract=fingerprint_row["contract"],
            data=fingerprint_row["data"],
            process=fingerprint_row["process"],
        ),
        decision=OperatorDecision(
            candidate_id=decision["candidate_id"],
            decision=decision["decision"],
            operator=decision["operator"],
            rationale=decision["rationale"],
            decided_at_unix_ns=int(decision["decided_at_unix_ns"]),
        ),
    )


@contextmanager
def sockets_disabled() -> Iterator[None]:
    """Make any socket construction raise for the duration of the block."""

    original = socket.socket

    def _refuse(*args: Any, **kwargs: Any):
        raise PromotionError(
            "a gold case attempted a network call; gold cases must run offline"
        )

    socket.socket = _refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment]


@dataclass(frozen=True)
class GoldOutcome:
    case_id: str
    check_id: str
    passed: bool
    detail: str


def evaluate_gold_case(case: GoldCase) -> GoldOutcome:
    """Run the frozen invariant on the frozen scenario, offline.

    Sockets are disabled for the duration and no model is consulted, so a gold
    case is executable in CI without network or credentials.
    """

    scenario = case.scenario
    meta = dict(scenario["meta"])
    meta["n_steps"] = len(scenario["steps"])
    trace = StrictTrace(
        tuple(dict(step) for step in scenario["steps"]),
        meta,
        case.source_trace_sha256,
    )
    with sockets_disabled():
        result = _run_named_check(trace, case.expected_invariant["check_id"])
    expected = bool(case.expected_invariant["must_pass"])
    return GoldOutcome(
        case.case_id,
        result.check_id,
        result.passed is expected,
        result.detail,
    )


__all__ = [
    "FORBIDDEN_OBSERVATION_KEYS",
    "GOLD_EVIDENCE_CLASS",
    "PROMOTION_CHECKS",
    "PROMOTION_SCHEMA_VERSION",
    "REQUIRED_META_FIELDS",
    "SILVER_EVIDENCE_CLASS",
    "CheckResult",
    "GoldCase",
    "GoldOutcome",
    "OperatorDecision",
    "PromotionError",
    "SilverCandidate",
    "SilverStore",
    "StrictTrace",
    "TraceFingerprint",
    "TraceIntegrityError",
    "blocking_failures",
    "build_silver_candidate",
    "compute_trace_fingerprint",
    "evaluate_gold_case",
    "load_gold_case",
    "load_trace_strict",
    "minimize_scenario",
    "process_events",
    "promote_to_gold",
    "run_promotion_checks",
    "sockets_disabled",
]
