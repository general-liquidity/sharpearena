"""Falsifiability manifests bound to generated strategy candidates.

A generated candidate is a strategy plus a claim about why it should work. Prose
written after selection is not a claim: it cannot be checked, and it is authored
with knowledge of the result. An :class:`EdgeManifest` moves that claim into the
candidate artifact itself, before the candidate has been validated, deduplicated
or scored, so the falsifiable content is fixed at proposal time.

Four properties are load-bearing and each one exists because the obvious
implementation gets it wrong.

Closed schema, no synthesized defaults
    Every object rejects unknown keys and every required field must be present.
    A manifest missing ``kill_conditions``, or carrying a misspelt key, makes the
    candidate INVALID. It does not silently become a manifest with an empty
    condition list, because an empty condition list monitors as healthy forever.

Typed thresholds
    A threshold is ``{"value": ..., "unit": ...}`` with the unit drawn from
    :data:`THRESHOLD_UNITS`. A bare number cannot say whether ``2`` means two
    basis points, two dollars or two percent, and a spec that mixes all three in
    one untyped column produces conditions that are wrong rather than merely
    unclear. A string threshold such as ``"2%"`` is a hard parse error here, not
    a NaN that silently disarms the condition.

No vacuous health
    :func:`monitor_edge` never reports health from zero evaluated conditions. A
    manifest must carry at least one invariant to be valid at all, and a metric
    that is absent from the observation resolves to ``unresolved``, which makes
    the verdict ``indeterminate``. "All 0 invariants hold" is not a pass.

Kill conditions are out-of-sample only
    :func:`monitor_edge` requires an explicit sample label and refuses to run on
    the selection sample. Kill conditions describe retirement, and a retirement
    rule that can see the selection sample is a selection criterion wearing a
    different name.

The recorder writes every raw candidate with its trial ordinal BEFORE validation
and BEFORE deduplication, so the denominator for a deflation correction is the
number of things the model actually proposed. Each record links the manifest hash
to the candidate hash, the model digest and the split-plan hash.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

EDGE_MANIFEST_SCHEMA_VERSION = 2
EDGE_MANIFEST_EVIDENCE_CLASS = "edge_manifest_candidate_pool"

IDEA_SOURCE_TYPES = frozenset(
    {
        "dataset",
        "operator_brief",
        "paper",
        "prior_candidate",
        "repository",
        "other",
    }
)

#: Closed unit vocabulary. A threshold whose unit is outside this set is invalid.
THRESHOLD_UNITS = frozenset(
    {
        "basis_points",
        "percent",
        "fraction",
        "usd",
        "count",
        "days",
        "seconds",
        "ratio",
        "z_score",
        "categorical",
    }
)

#: Comparators that take a numeric threshold.
NUMERIC_COMPARATORS = frozenset({"gt", "gte", "lt", "lte", "eq", "neq"})

#: Comparators that take a categorical membership threshold.
SET_COMPARATORS = frozenset({"in", "not_in"})

COMPARATORS = NUMERIC_COMPARATORS | SET_COMPARATORS

#: Samples a manifest may be monitored against. ``selection`` is refused.
OUT_OF_SELECTION_SAMPLES = frozenset({"test", "forward", "holdout"})

MAX_CONDITIONS = 32
MAX_TEXT = 1000


class EdgeManifestError(ValueError):
    """A manifest violates the closed schema. The candidate is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: Any, path: str) -> str:
    text = _text(value, path)
    if not text.startswith("sha256:") or len(text) != 71:
        raise EdgeManifestError(f"{path} must be 'sha256:' followed by 64 hex digits")
    digest = text[7:]
    if digest != digest.lower() or any(ch not in "0123456789abcdef" for ch in digest):
        raise EdgeManifestError(f"{path} must use 64 lowercase hex digits")
    return text


def _closed(
    value: Any, allowed: set[str], required: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EdgeManifestError(f"{path} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown and missing:
        raise EdgeManifestError(
            f"{path} has unknown fields: {unknown} and is missing required fields: "
            f"{missing}"
        )
    if unknown:
        raise EdgeManifestError(f"{path} has unknown fields: {unknown}")
    if missing:
        raise EdgeManifestError(f"{path} is missing required fields: {missing}")
    return value


def _text(value: Any, path: str, *, min_length: int = 1) -> str:
    if not isinstance(value, str):
        raise EdgeManifestError(f"{path} must be a string")
    stripped = value.strip()
    if len(stripped) < min_length:
        raise EdgeManifestError(f"{path} must be at least {min_length} characters")
    if len(stripped) > MAX_TEXT:
        raise EdgeManifestError(f"{path} exceeds {MAX_TEXT} characters")
    return stripped


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EdgeManifestError(f"{path} must be a non-empty array")
    items = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(items)) != len(items):
        raise EdgeManifestError(f"{path} must not contain duplicates")
    return items


def _string_tuple_allow_empty(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EdgeManifestError(f"{path} must be an array")
    items = tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(items)) != len(items):
        raise EdgeManifestError(f"{path} must not contain duplicates")
    return items


@dataclass(frozen=True)
class IdeaProvenance:
    """One operator-bound source a generated candidate may cite.

    The source is registered on the search plan, not invented in model output.
    ``source_digest`` binds the exact bytes the operator supplied. Locators and
    attribution fields explain those bytes, but never substitute for the digest.
    """

    source_type: str
    source_digest: str
    url_or_doi: Optional[str] = None
    commit: Optional[str] = None
    authors: tuple[str, ...] = ()
    license: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source_type not in IDEA_SOURCE_TYPES:
            raise EdgeManifestError(
                f"source_type {self.source_type!r} is outside the closed source set"
            )
        _sha256_text(self.source_digest, "source_digest")
        if self.url_or_doi is not None:
            _text(self.url_or_doi, "url_or_doi")
        if self.commit is not None:
            commit = _text(self.commit, "commit")
            if commit.lower() in {"head", "latest", "main", "master"}:
                raise EdgeManifestError("commit must name an immutable revision")
        if len(set(self.authors)) != len(self.authors):
            raise EdgeManifestError("authors must not contain duplicates")
        for index, author in enumerate(self.authors):
            _text(author, f"authors[{index}]")
        if self.license is not None:
            _text(self.license, "license")

    def as_record(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_digest": self.source_digest,
            "url_or_doi": self.url_or_doi,
            "commit": self.commit,
            "authors": list(self.authors),
            "license": self.license,
        }


def bind_idea_source(
    content: Union[str, bytes],
    *,
    source_type: str,
    url_or_doi: Optional[str] = None,
    commit: Optional[str] = None,
    authors: Iterable[str] = (),
    license: Optional[str] = None,
) -> IdeaProvenance:
    """Bind exact source bytes into a plan-owned provenance record."""

    payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return IdeaProvenance(
        source_type=source_type,
        source_digest=f"sha256:{sha256(payload).hexdigest()}",
        url_or_doi=url_or_doi,
        commit=commit,
        authors=tuple(authors),
        license=license,
    )


def parse_idea_provenance(
    payload: Any, path: str = "idea_provenance"
) -> IdeaProvenance:
    obj = _closed(
        payload,
        {"source_type", "source_digest", "url_or_doi", "commit", "authors", "license"},
        {"source_type", "source_digest"},
        path,
    )
    authors = obj.get("authors", [])
    if not isinstance(authors, list):
        raise EdgeManifestError(f"{path}.authors must be an array")
    return IdeaProvenance(
        source_type=_text(obj["source_type"], f"{path}.source_type"),
        source_digest=_sha256_text(obj["source_digest"], f"{path}.source_digest"),
        url_or_doi=(
            None
            if obj.get("url_or_doi") is None
            else _text(obj["url_or_doi"], f"{path}.url_or_doi")
        ),
        commit=(
            None
            if obj.get("commit") is None
            else _text(obj["commit"], f"{path}.commit")
        ),
        authors=tuple(
            _text(item, f"{path}.authors[{i}]") for i, item in enumerate(authors)
        ),
        license=(
            None
            if obj.get("license") is None
            else _text(obj["license"], f"{path}.license")
        ),
    )


@dataclass(frozen=True)
class DeclaredCandidateLineage:
    """References declared by a candidate, resolved by the host ledger."""

    parent_candidate_ids: tuple[str, ...] = ()
    idea_source_digests: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "parent_candidate_ids": list(self.parent_candidate_ids),
            "idea_source_digests": list(self.idea_source_digests),
        }


def parse_candidate_lineage(
    payload: Any, path: str = "lineage"
) -> DeclaredCandidateLineage:
    obj = _closed(
        payload,
        {"parent_candidate_ids", "idea_source_digests"},
        {"parent_candidate_ids", "idea_source_digests"},
        path,
    )
    parents = _string_tuple_allow_empty(
        obj["parent_candidate_ids"], f"{path}.parent_candidate_ids"
    )
    source_digests = tuple(
        _sha256_text(item, f"{path}.idea_source_digests[{index}]")
        for index, item in enumerate(
            _string_tuple_allow_empty(
                obj["idea_source_digests"], f"{path}.idea_source_digests"
            )
        )
    )
    return DeclaredCandidateLineage(parents, source_digests)


@dataclass(frozen=True)
class CandidateValidation:
    """Host-derived identity returned by a candidate validator."""

    semantic_fingerprint: str
    family_preimage: Any
    candidate_id: str


@dataclass(frozen=True)
class Threshold:
    """A comparison bound that carries its own unit.

    ``value`` is a float for the numeric comparators and a tuple of labels for
    ``in`` / ``not_in``. There is no untyped form: a caller cannot construct a
    threshold without naming a unit from :data:`THRESHOLD_UNITS`.
    """

    value: Union[float, tuple[str, ...]]
    unit: str

    def __post_init__(self) -> None:
        if self.unit not in THRESHOLD_UNITS:
            raise EdgeManifestError(
                f"threshold unit {self.unit!r} is outside the closed unit set"
            )
        if isinstance(self.value, tuple):
            if self.unit != "categorical":
                raise EdgeManifestError(
                    "a membership threshold requires unit 'categorical'"
                )
            if not self.value or any(
                not isinstance(item, str) or not item for item in self.value
            ):
                raise EdgeManifestError(
                    "a membership threshold must list non-empty labels"
                )
            if len(set(self.value)) != len(self.value):
                raise EdgeManifestError("membership threshold labels must be unique")
            return
        if self.unit == "categorical":
            raise EdgeManifestError(
                "unit 'categorical' requires a membership threshold, not a number"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise EdgeManifestError("a numeric threshold value must be a number")
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise EdgeManifestError("a numeric threshold value must be finite")
        object.__setattr__(self, "value", float(self.value))

    def as_record(self) -> dict[str, Any]:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"value": value, "unit": self.unit}


def parse_threshold(payload: Any, path: str) -> Threshold:
    """Parse ``{"value": ..., "unit": ...}``. A bare number is refused."""

    if not isinstance(payload, dict):
        raise EdgeManifestError(
            f"{path} must be an object with 'value' and 'unit'; a bare number or "
            "string carries no unit and cannot be compared"
        )
    obj = _closed(payload, {"value", "unit"}, {"value", "unit"}, path)
    unit = obj["unit"]
    if not isinstance(unit, str):
        raise EdgeManifestError(f"{path}.unit must be a string")
    raw = obj["value"]
    if isinstance(raw, list):
        return Threshold(tuple(raw), unit)
    if isinstance(raw, str):
        raise EdgeManifestError(
            f"{path}.value is the string {raw!r}; write the number and put the unit "
            "in the 'unit' field"
        )
    return Threshold(raw, unit)


@dataclass(frozen=True)
class EdgeCondition:
    """One checkable claim: a named metric compared against a typed threshold."""

    condition_id: str
    metric: str
    comparator: str
    threshold: Threshold
    description: str

    def __post_init__(self) -> None:
        for name in ("condition_id", "metric", "description"):
            _text(getattr(self, name), name)
        if self.comparator not in COMPARATORS:
            raise EdgeManifestError(f"comparator {self.comparator!r} is unsupported")
        if self.comparator in SET_COMPARATORS and not isinstance(
            self.threshold.value, tuple
        ):
            raise EdgeManifestError(
                f"comparator {self.comparator!r} requires a membership threshold"
            )
        if self.comparator in NUMERIC_COMPARATORS and isinstance(
            self.threshold.value, tuple
        ):
            raise EdgeManifestError(
                f"comparator {self.comparator!r} requires a numeric threshold"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "metric": self.metric,
            "comparator": self.comparator,
            "threshold": self.threshold.as_record(),
            "description": self.description,
        }


def parse_condition(payload: Any, path: str) -> EdgeCondition:
    obj = _closed(
        payload,
        {"condition_id", "metric", "comparator", "threshold", "description"},
        {"condition_id", "metric", "comparator", "threshold", "description"},
        path,
    )
    comparator = obj["comparator"]
    if not isinstance(comparator, str):
        raise EdgeManifestError(f"{path}.comparator must be a string")
    return EdgeCondition(
        condition_id=_text(obj["condition_id"], f"{path}.condition_id"),
        metric=_text(obj["metric"], f"{path}.metric"),
        comparator=comparator,
        threshold=parse_threshold(obj["threshold"], f"{path}.threshold"),
        description=_text(obj["description"], f"{path}.description"),
    )


def _parse_conditions(
    payload: Any, path: str, *, minimum: int
) -> tuple[EdgeCondition, ...]:
    if not isinstance(payload, list):
        raise EdgeManifestError(f"{path} must be an array")
    if len(payload) < minimum:
        raise EdgeManifestError(
            f"{path} must contain at least {minimum} condition(s); an empty list "
            "would monitor as healthy forever"
        )
    if len(payload) > MAX_CONDITIONS:
        raise EdgeManifestError(f"{path} exceeds the cap of {MAX_CONDITIONS}")
    conditions = tuple(
        parse_condition(item, f"{path}[{index}]") for index, item in enumerate(payload)
    )
    ids = [condition.condition_id for condition in conditions]
    if len(set(ids)) != len(ids):
        raise EdgeManifestError(f"{path} condition_id values must be unique")
    return conditions


@dataclass(frozen=True)
class VerificationPlan:
    """How the claim is to be tested, named before any result is known."""

    selection_metric: str
    selection_split: str
    confirmation_split: str
    minimum_observations: int

    def __post_init__(self) -> None:
        for name in ("selection_metric", "selection_split", "confirmation_split"):
            _text(getattr(self, name), name)
        if self.selection_split == self.confirmation_split:
            raise EdgeManifestError(
                "selection and confirmation splits must be different"
            )
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 2
        ):
            raise EdgeManifestError("minimum_observations must be an integer >= 2")

    def as_record(self) -> dict[str, Any]:
        return {
            "selection_metric": self.selection_metric,
            "selection_split": self.selection_split,
            "confirmation_split": self.confirmation_split,
            "minimum_observations": self.minimum_observations,
        }


def parse_verification_plan(payload: Any, path: str) -> VerificationPlan:
    obj = _closed(
        payload,
        {
            "selection_metric",
            "selection_split",
            "confirmation_split",
            "minimum_observations",
        },
        {
            "selection_metric",
            "selection_split",
            "confirmation_split",
            "minimum_observations",
        },
        path,
    )
    return VerificationPlan(
        selection_metric=_text(obj["selection_metric"], f"{path}.selection_metric"),
        selection_split=_text(obj["selection_split"], f"{path}.selection_split"),
        confirmation_split=_text(
            obj["confirmation_split"], f"{path}.confirmation_split"
        ),
        minimum_observations=obj["minimum_observations"],
    )


@dataclass(frozen=True)
class EdgeManifest:
    """The falsifiable claim attached to one candidate."""

    hypothesis: str
    mechanism: str
    regimes: tuple[str, ...]
    instruments: tuple[str, ...]
    invariants: tuple[EdgeCondition, ...]
    kill_conditions: tuple[EdgeCondition, ...]
    verification_plan: VerificationPlan

    def __post_init__(self) -> None:
        _text(self.hypothesis, "hypothesis", min_length=8)
        _text(self.mechanism, "mechanism", min_length=8)
        if not self.invariants:
            raise EdgeManifestError(
                "a manifest with no invariants asserts nothing and can never fail"
            )
        overlap = {condition.condition_id for condition in self.invariants} & {
            condition.condition_id for condition in self.kill_conditions
        }
        if overlap:
            raise EdgeManifestError(
                f"condition ids are shared between invariants and kill conditions: {sorted(overlap)}"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "regimes": list(self.regimes),
            "instruments": list(self.instruments),
            "invariants": [condition.as_record() for condition in self.invariants],
            "kill_conditions": [
                condition.as_record() for condition in self.kill_conditions
            ],
            "verification_plan": self.verification_plan.as_record(),
        }

    @property
    def manifest_sha256(self) -> str:
        return _digest(self.as_record())


MANIFEST_FIELDS = {
    "hypothesis",
    "mechanism",
    "regimes",
    "instruments",
    "invariants",
    "kill_conditions",
    "verification_plan",
}


def parse_edge_manifest(payload: Any, path: str = "edge_manifest") -> EdgeManifest:
    """Parse a manifest under the closed schema. Anything missing is an error."""

    obj = _closed(payload, MANIFEST_FIELDS, MANIFEST_FIELDS, path)
    return EdgeManifest(
        hypothesis=_text(obj["hypothesis"], f"{path}.hypothesis", min_length=8),
        mechanism=_text(obj["mechanism"], f"{path}.mechanism", min_length=8),
        regimes=_string_tuple(obj["regimes"], f"{path}.regimes"),
        instruments=_string_tuple(obj["instruments"], f"{path}.instruments"),
        invariants=_parse_conditions(
            obj["invariants"], f"{path}.invariants", minimum=1
        ),
        kill_conditions=_parse_conditions(
            obj["kill_conditions"], f"{path}.kill_conditions", minimum=1
        ),
        verification_plan=parse_verification_plan(
            obj["verification_plan"], f"{path}.verification_plan"
        ),
    )


@dataclass(frozen=True)
class ConditionCheck:
    """The outcome of evaluating one condition against one observation."""

    condition: EdgeCondition
    observed: Union[float, str, None]
    #: ``holds``, ``violated`` or ``unresolved`` (the metric was absent).
    status: str

    def as_record(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition.condition_id,
            "metric": self.condition.metric,
            "comparator": self.condition.comparator,
            "threshold": self.condition.threshold.as_record(),
            "observed": self.observed,
            "status": self.status,
        }


def _observed_unit(metrics: Mapping[str, Any], metric: str) -> Optional[str]:
    entry = metrics.get(metric)
    if isinstance(entry, Mapping):
        unit = entry.get("unit")
        return unit if isinstance(unit, str) else None
    return None


def _observed_value(metrics: Mapping[str, Any], metric: str) -> Any:
    entry = metrics.get(metric)
    if isinstance(entry, Mapping):
        return entry.get("value")
    return entry


def evaluate_condition_against(
    condition: EdgeCondition, metrics: Mapping[str, Any]
) -> ConditionCheck:
    """Evaluate one condition. An absent or mis-united metric is ``unresolved``.

    A metric may be supplied either bare (``{"net_edge": 4.0}``) or with its own
    unit (``{"net_edge": {"value": 4.0, "unit": "basis_points"}}``). When the unit
    is supplied it must match the threshold's unit; a mismatch resolves to
    ``unresolved`` rather than comparing two different quantities.
    """

    if condition.metric not in metrics:
        return ConditionCheck(condition, None, "unresolved")
    observed_unit = _observed_unit(metrics, condition.metric)
    if observed_unit is not None and observed_unit != condition.threshold.unit:
        return ConditionCheck(condition, None, "unresolved")
    raw = _observed_value(metrics, condition.metric)
    if condition.comparator in SET_COMPARATORS:
        if not isinstance(raw, str):
            return ConditionCheck(condition, None, "unresolved")
        labels = condition.threshold.value
        assert isinstance(labels, tuple)
        member = raw in labels
        holds = member if condition.comparator == "in" else not member
        return ConditionCheck(condition, raw, "holds" if holds else "violated")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return ConditionCheck(condition, None, "unresolved")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        return ConditionCheck(condition, None, "unresolved")
    bound = condition.threshold.value
    assert isinstance(bound, float)
    holds = {
        "gt": value > bound,
        "gte": value >= bound,
        "lt": value < bound,
        "lte": value <= bound,
        "eq": value == bound,
        "neq": value != bound,
    }[condition.comparator]
    return ConditionCheck(condition, value, "holds" if holds else "violated")


@dataclass(frozen=True)
class EdgeHealthReport:
    """A verdict that names how many conditions it actually resolved."""

    #: ``healthy``, ``violated``, ``retired`` or ``indeterminate``.
    health: str
    sample: str
    invariant_checks: tuple[ConditionCheck, ...]
    kill_checks: tuple[ConditionCheck, ...]
    reason: str

    @property
    def resolved_invariants(self) -> int:
        return sum(check.status != "unresolved" for check in self.invariant_checks)

    @property
    def unresolved_metrics(self) -> tuple[str, ...]:
        return tuple(
            check.condition.metric
            for check in (*self.invariant_checks, *self.kill_checks)
            if check.status == "unresolved"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "health": self.health,
            "sample": self.sample,
            "resolved_invariants": self.resolved_invariants,
            "total_invariants": len(self.invariant_checks),
            "unresolved_metrics": list(self.unresolved_metrics),
            "invariant_checks": [check.as_record() for check in self.invariant_checks],
            "kill_checks": [check.as_record() for check in self.kill_checks],
            "reason": self.reason,
        }


def monitor_edge(
    manifest: EdgeManifest, metrics: Mapping[str, Any], *, sample: str
) -> EdgeHealthReport:
    """Monitor a manifest against observed metrics from an out-of-selection sample.

    ``sample`` must name one of :data:`OUT_OF_SELECTION_SAMPLES`. Monitoring the
    selection sample is refused: kill conditions must not be able to influence
    which candidate is chosen.

    The verdict is ``indeterminate`` unless every invariant and every kill
    condition resolved. There is no path by which an unmeasured edge reports as
    healthy.
    """

    if sample not in OUT_OF_SELECTION_SAMPLES:
        raise EdgeManifestError(
            f"sample {sample!r} is not an out-of-selection sample; kill conditions "
            f"may only be evaluated on one of {sorted(OUT_OF_SELECTION_SAMPLES)}"
        )
    invariant_checks = tuple(
        evaluate_condition_against(condition, metrics)
        for condition in manifest.invariants
    )
    kill_checks = tuple(
        evaluate_condition_against(condition, metrics)
        for condition in manifest.kill_conditions
    )
    unresolved = [
        check
        for check in (*invariant_checks, *kill_checks)
        if check.status == "unresolved"
    ]
    fired = [check for check in kill_checks if check.status == "holds"]
    violated = [check for check in invariant_checks if check.status == "violated"]
    if fired:
        ids = sorted(check.condition.condition_id for check in fired)
        return EdgeHealthReport(
            "retired",
            sample,
            invariant_checks,
            kill_checks,
            f"kill conditions fired on the {sample} sample: {ids}",
        )
    if unresolved:
        metrics_missing = sorted({check.condition.metric for check in unresolved})
        return EdgeHealthReport(
            "indeterminate",
            sample,
            invariant_checks,
            kill_checks,
            f"{len(unresolved)} condition(s) could not be resolved; metrics missing "
            f"or mis-united: {metrics_missing}",
        )
    if violated:
        ids = sorted(check.condition.condition_id for check in violated)
        return EdgeHealthReport(
            "violated",
            sample,
            invariant_checks,
            kill_checks,
            f"invariants violated on the {sample} sample: {ids}",
        )
    return EdgeHealthReport(
        "healthy",
        sample,
        invariant_checks,
        kill_checks,
        f"{len(invariant_checks)} invariant(s) hold and "
        f"{len(kill_checks)} kill condition(s) did not fire on the {sample} sample",
    )


@dataclass(frozen=True)
class ManifestedCandidate:
    """One raw candidate, its trial ordinal, and its manifest verdict.

    The record exists whether or not the candidate turned out to be valid or
    unique. ``manifest`` is ``None`` when manifest parsing failed; a candidate
    can also be invalid because its strategy or declared lineage failed after
    the manifest parsed.
    """

    trial_ordinal: int
    raw_candidate: dict[str, Any]
    raw_candidate_sha256: str
    manifest: Optional[EdgeManifest]
    manifest_sha256: Optional[str]
    invalid_reason: Optional[str]
    duplicate_of_ordinal: Optional[int]
    model_digest: str
    split_plan_sha256: str
    family_preimage: Any
    family_digest: str
    declared_lineage: Optional[DeclaredCandidateLineage]
    parent_candidate_digests: tuple[str, ...]
    generator_identity: dict[str, Any]
    generator_identity_sha256: str
    idea_provenance: tuple[IdeaProvenance, ...]
    lineage_status: str

    @property
    def is_selectable(self) -> bool:
        return self.invalid_reason is None and self.duplicate_of_ordinal is None

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": EDGE_MANIFEST_SCHEMA_VERSION,
            "evidence_class": EDGE_MANIFEST_EVIDENCE_CLASS,
            "trial_ordinal": self.trial_ordinal,
            "raw_candidate": self.raw_candidate,
            "raw_candidate_sha256": self.raw_candidate_sha256,
            "manifest": None if self.manifest is None else self.manifest.as_record(),
            "manifest_sha256": self.manifest_sha256,
            "invalid_reason": self.invalid_reason,
            "duplicate_of_ordinal": self.duplicate_of_ordinal,
            "model_digest": self.model_digest,
            "split_plan_sha256": self.split_plan_sha256,
            "family_preimage": self.family_preimage,
            "family_digest": self.family_digest,
            "declared_lineage": (
                None
                if self.declared_lineage is None
                else self.declared_lineage.as_record()
            ),
            "parent_candidate_digests": list(self.parent_candidate_digests),
            "generator_identity": self.generator_identity,
            "generator_identity_sha256": self.generator_identity_sha256,
            "idea_provenance": [source.as_record() for source in self.idea_provenance],
            "lineage_status": self.lineage_status,
            "lineage_binding_sha256": self.lineage_binding_sha256,
            "binding_sha256": self.binding_sha256,
        }

    @property
    def lineage_binding_sha256(self) -> str:
        """Hash the host-resolved family, ancestry, generator, and sources."""

        return _digest(
            {
                "family_digest": self.family_digest,
                "generator_identity_sha256": self.generator_identity_sha256,
                "idea_source_digests": [
                    source.source_digest for source in self.idea_provenance
                ],
                "parent_candidate_digests": list(self.parent_candidate_digests),
                "raw_candidate_sha256": self.raw_candidate_sha256,
            }
        )

    @property
    def binding_sha256(self) -> str:
        """Hash tying manifest, candidate, model and split plan into one object."""

        return _digest(
            {
                "raw_candidate_sha256": self.raw_candidate_sha256,
                "manifest_sha256": self.manifest_sha256,
                "model_digest": self.model_digest,
                "split_plan_sha256": self.split_plan_sha256,
                "trial_ordinal": self.trial_ordinal,
            }
        )


class EdgeManifestLedger:
    """Append-only record of every proposed candidate, in proposal order.

    ``record`` assigns the next trial ordinal and writes the row BEFORE deciding
    whether the candidate is valid or a duplicate, so the ledger length is the
    honest trial count even when most of the pool is garbage.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        model_digest: str,
        split_plan_sha256: str,
        generator_identity: Optional[Mapping[str, Any]] = None,
        idea_provenance: Iterable[IdeaProvenance] = (),
    ) -> None:
        if not model_digest or not split_plan_sha256:
            raise EdgeManifestError(
                "an edge-manifest ledger requires a model digest and a split-plan hash"
            )
        self.path = path
        self.model_digest = model_digest
        self.split_plan_sha256 = split_plan_sha256
        identity = dict(generator_identity or {"digest": model_digest})
        if not identity or identity.get("digest") != model_digest:
            raise EdgeManifestError(
                "generator_identity.digest must equal the ledger model digest"
            )
        try:
            normalized_identity = json.loads(_canonical_bytes(identity))
        except (TypeError, ValueError) as error:
            raise EdgeManifestError(
                f"generator_identity must be canonical-JSON serializable: {error}"
            ) from error
        if not isinstance(normalized_identity, dict):
            raise EdgeManifestError("generator_identity must be an object")
        self.generator_identity = normalized_identity
        self.generator_identity_sha256 = _digest(normalized_identity)
        sources = tuple(idea_provenance)
        source_digests = [source.source_digest for source in sources]
        if len(set(source_digests)) != len(source_digests):
            raise EdgeManifestError("idea_provenance source digests must be unique")
        self.idea_provenance = sources
        self._idea_sources = {source.source_digest: source for source in sources}
        self._records: list[ManifestedCandidate] = []
        self._fingerprints: dict[str, int] = {}
        self._candidate_ids: dict[str, str] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def observed_trials(self) -> int:
        """Every candidate the model proposed, valid or not, unique or not."""

        return len(self._records)

    @property
    def records(self) -> tuple[ManifestedCandidate, ...]:
        return tuple(self._records)

    def selectable(self) -> tuple[ManifestedCandidate, ...]:
        return tuple(record for record in self._records if record.is_selectable)

    def record(
        self,
        raw_candidate: Any,
        *,
        candidate_validator: Optional[
            Callable[[Mapping[str, Any]], Union[str, CandidateValidation]]
        ] = None,
    ) -> ManifestedCandidate:
        """Record one proposal before deciding whether it can be selected.

        ``candidate_validator`` validates the strategy half after the manifest
        has parsed and returns its semantic fingerprint.  Keeping that callback
        inside ``record`` preserves the load-bearing ordering: the ordinal is
        assigned first, manifest validation runs second, strategy validation
        third, and deduplication last.  A generated-strategy caller therefore
        cannot accidentally count only the candidates that survived its DSL.
        """

        ordinal = len(self._records)
        normalized = (
            json.loads(_canonical_bytes(raw_candidate))
            if isinstance(raw_candidate, dict)
            else {"__non_object__": repr(raw_candidate)}
        )
        raw_sha = _digest(normalized)
        manifest: Optional[EdgeManifest] = None
        manifest_sha: Optional[str] = None
        declared_lineage: Optional[DeclaredCandidateLineage] = None
        parent_candidate_digests: tuple[str, ...] = ()
        resolved_sources: tuple[IdeaProvenance, ...] = ()
        invalid: Optional[str] = None
        duplicate: Optional[int] = None
        if not isinstance(raw_candidate, dict):
            invalid = "candidate must be an object"
        elif "edge_manifest" not in raw_candidate:
            invalid = "candidate is missing the required 'edge_manifest' field"
        else:
            try:
                manifest = parse_edge_manifest(raw_candidate["edge_manifest"])
                manifest_sha = manifest.manifest_sha256
            except EdgeManifestError as error:
                invalid = str(error)
        if isinstance(raw_candidate, dict) and raw_candidate.get("lineage") is not None:
            try:
                declared_lineage = parse_candidate_lineage(raw_candidate["lineage"])
                missing_parents = [
                    parent
                    for parent in declared_lineage.parent_candidate_ids
                    if parent not in self._candidate_ids
                ]
                if missing_parents:
                    raise EdgeManifestError(
                        "lineage.parent_candidate_ids must name earlier valid proposals; "
                        f"not found: {missing_parents}"
                    )
                parent_candidate_digests = tuple(
                    self._candidate_ids[parent]
                    for parent in declared_lineage.parent_candidate_ids
                )
                missing_sources = [
                    digest
                    for digest in declared_lineage.idea_source_digests
                    if digest not in self._idea_sources
                ]
                if missing_sources:
                    raise EdgeManifestError(
                        "lineage.idea_source_digests must reference plan-bound sources; "
                        f"not found: {missing_sources}"
                    )
                resolved_sources = tuple(
                    self._idea_sources[digest]
                    for digest in declared_lineage.idea_source_digests
                )
            except EdgeManifestError as error:
                if invalid is None:
                    invalid = str(error)
        strategy_fingerprint: Optional[str] = None
        family_preimage: Any = {"unparsed_raw_candidate_sha256": raw_sha}
        candidate_id: Optional[str] = None
        if manifest is not None and invalid is None and candidate_validator is not None:
            try:
                validation = candidate_validator(normalized)
                if isinstance(validation, CandidateValidation):
                    strategy_fingerprint = validation.semantic_fingerprint
                    family_preimage = json.loads(
                        _canonical_bytes(validation.family_preimage)
                    )
                    candidate_id = _text(validation.candidate_id, "candidate_id")
                else:
                    strategy_fingerprint = validation
                    family_preimage = {"semantic_fingerprint": validation}
                if not strategy_fingerprint:
                    raise EdgeManifestError(
                        "candidate validator returned an empty fingerprint"
                    )
            except (EdgeManifestError, ValueError) as error:
                invalid = str(error)
        if candidate_id is not None:
            if candidate_id in self._candidate_ids:
                invalid = invalid or f"candidate id {candidate_id!r} is not unique"
            else:
                self._candidate_ids[candidate_id] = raw_sha
        family_digest = _digest(family_preimage)
        if manifest is not None and invalid is None:
            fingerprint = _digest(
                {
                    "manifest": manifest.as_record(),
                    "strategy": strategy_fingerprint
                    or {
                        key: value
                        for key, value in normalized.items()
                        if key not in {"edge_manifest", "id"}
                    },
                }
            )
            previous = self._fingerprints.get(fingerprint)
            if previous is None:
                self._fingerprints[fingerprint] = ordinal
            else:
                duplicate = previous
        entry = ManifestedCandidate(
            trial_ordinal=ordinal,
            raw_candidate=normalized,
            raw_candidate_sha256=raw_sha,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            invalid_reason=invalid,
            duplicate_of_ordinal=duplicate,
            model_digest=self.model_digest,
            split_plan_sha256=self.split_plan_sha256,
            family_preimage=family_preimage,
            family_digest=family_digest,
            declared_lineage=declared_lineage,
            parent_candidate_digests=parent_candidate_digests,
            generator_identity=self.generator_identity,
            generator_identity_sha256=self.generator_identity_sha256,
            idea_provenance=resolved_sources,
            lineage_status=(
                "declared"
                if declared_lineage is not None and invalid is None
                else "invalid"
                if declared_lineage is not None
                else "host-derived-unreferenced"
            ),
        )
        self._records.append(entry)
        self._append(entry)
        return entry

    def record_pool(
        self,
        raw_candidates: Iterable[Any],
        *,
        candidate_validator: Optional[
            Callable[[Mapping[str, Any]], Union[str, CandidateValidation]]
        ] = None,
    ) -> tuple[ManifestedCandidate, ...]:
        return tuple(
            self.record(candidate, candidate_validator=candidate_validator)
            for candidate in raw_candidates
        )

    def summary(self) -> dict[str, Any]:
        family_counts: dict[str, dict[str, int]] = {}
        for record in self._records:
            counts = family_counts.setdefault(
                record.family_digest, {"observed_trials": 0, "selectable": 0}
            )
            counts["observed_trials"] += 1
            counts["selectable"] += int(record.is_selectable)
        return {
            "schema_version": EDGE_MANIFEST_SCHEMA_VERSION,
            "evidence_class": EDGE_MANIFEST_EVIDENCE_CLASS,
            "model_digest": self.model_digest,
            "split_plan_sha256": self.split_plan_sha256,
            "observed_trials": self.observed_trials,
            "invalid": sum(
                record.invalid_reason is not None for record in self._records
            ),
            "duplicates": sum(
                record.duplicate_of_ordinal is not None for record in self._records
            ),
            "selectable": len(self.selectable()),
            "families": [
                {"family_digest": digest, **family_counts[digest]}
                for digest in sorted(family_counts)
            ],
            "family_count": len(family_counts),
            "generator_identity_sha256": self.generator_identity_sha256,
            "plan_bound_idea_sources": len(self.idea_provenance),
            "family_grouping_role": "diagnostic-only-never-a-trial-deduplicator",
            "n_trials_source": "ledger-counted-before-validation-and-deduplication",
        }

    def _append(self, entry: ManifestedCandidate) -> None:
        if self.path is None:
            return
        with self.path.open("ab") as handle:
            handle.write(_canonical_bytes(entry.as_record()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_manifest_ledger(path: Path) -> list[dict[str, Any]]:
    """Read and verify a ledger. Version-1 rows retain ordinal-only support.

    Version 2 binds the raw candidate, host-derived family, generator identity,
    resolved ancestry, and cited idea sources. Recomputing those bindings here
    means a ledger is not trusted merely because it is parseable JSON.
    """

    rows: list[dict[str, Any]] = []
    earlier_candidate_digests: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise EdgeManifestError(f"{path}:{number} is blank")
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise EdgeManifestError(
                    f"{path}:{number} is not JSON: {error}"
                ) from error
            if not isinstance(row, dict) or row.get("trial_ordinal") != len(rows):
                raise EdgeManifestError(
                    f"{path}:{number} has a broken trial ordinal; the ledger is not intact"
                )
            version = row.get("schema_version", 1)
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise EdgeManifestError(
                    f"{path}:{number} has an invalid schema_version"
                )
            if version >= 2:
                _verify_v2_ledger_row(
                    row,
                    path=path,
                    line_number=number,
                    earlier_candidate_digests=earlier_candidate_digests,
                )
            raw_digest = row.get("raw_candidate_sha256")
            if isinstance(raw_digest, str):
                earlier_candidate_digests.add(raw_digest)
            rows.append(row)
    return rows


def _verify_v2_ledger_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    earlier_candidate_digests: set[str],
) -> None:
    """Verify one v2 row without accepting its self-reported digests."""

    location = f"{path}:{line_number}"
    required = {
        "evidence_class",
        "trial_ordinal",
        "raw_candidate",
        "raw_candidate_sha256",
        "manifest",
        "manifest_sha256",
        "model_digest",
        "split_plan_sha256",
        "family_preimage",
        "family_digest",
        "generator_identity",
        "generator_identity_sha256",
        "parent_candidate_digests",
        "idea_provenance",
        "lineage_binding_sha256",
        "binding_sha256",
    }
    missing = sorted(required - set(row))
    if missing:
        raise EdgeManifestError(f"{location} is missing v2 fields: {missing}")
    if row["evidence_class"] != EDGE_MANIFEST_EVIDENCE_CLASS:
        raise EdgeManifestError(
            f"{location} has evidence_class {row['evidence_class']!r}, expected "
            f"{EDGE_MANIFEST_EVIDENCE_CLASS!r}"
        )

    digest_checks = (
        ("raw_candidate", "raw_candidate_sha256"),
        ("family_preimage", "family_digest"),
        ("generator_identity", "generator_identity_sha256"),
    )
    for value_field, digest_field in digest_checks:
        expected = _digest(row[value_field])
        if row[digest_field] != expected:
            raise EdgeManifestError(
                f"{location} has a stale {digest_field}; expected {expected}"
            )

    expected_manifest_digest = (
        None if row["manifest"] is None else _digest(row["manifest"])
    )
    if row["manifest_sha256"] != expected_manifest_digest:
        raise EdgeManifestError(
            f"{location} has a stale manifest_sha256; expected "
            f"{expected_manifest_digest}"
        )
    identity = row["generator_identity"]
    if not isinstance(identity, dict) or identity.get("digest") != row["model_digest"]:
        raise EdgeManifestError(
            f"{location} generator_identity.digest does not match model_digest"
        )

    parents = row["parent_candidate_digests"]
    if not isinstance(parents, list) or any(
        not isinstance(item, str) for item in parents
    ):
        raise EdgeManifestError(
            f"{location} parent_candidate_digests must be an array of strings"
        )
    if len(set(parents)) != len(parents):
        raise EdgeManifestError(
            f"{location} parent_candidate_digests must not contain duplicates"
        )
    missing_parents = sorted(set(parents) - earlier_candidate_digests)
    if missing_parents:
        raise EdgeManifestError(
            f"{location} references candidate digests that are not earlier rows: "
            f"{missing_parents}"
        )

    raw_sources = row["idea_provenance"]
    if not isinstance(raw_sources, list):
        raise EdgeManifestError(f"{location} idea_provenance must be an array")
    sources = tuple(
        parse_idea_provenance(item, f"{location}.idea_provenance[{index}]")
        for index, item in enumerate(raw_sources)
    )
    source_digests = [source.source_digest for source in sources]
    if len(set(source_digests)) != len(source_digests):
        raise EdgeManifestError(
            f"{location} idea_provenance source digests must not contain duplicates"
        )

    declared = row.get("declared_lineage")
    if declared is None:
        if parents or sources:
            raise EdgeManifestError(
                f"{location} resolves ancestry or sources without declared_lineage"
            )
    else:
        parsed = parse_candidate_lineage(declared, f"{location}.declared_lineage")
        if list(parsed.idea_source_digests) != source_digests:
            raise EdgeManifestError(
                f"{location} idea_provenance does not match the declared source order"
            )
        if len(parsed.parent_candidate_ids) != len(parents):
            raise EdgeManifestError(
                f"{location} resolved parent count does not match the declaration"
            )

    lineage_binding = _digest(
        {
            "family_digest": row["family_digest"],
            "generator_identity_sha256": row["generator_identity_sha256"],
            "idea_source_digests": source_digests,
            "parent_candidate_digests": parents,
            "raw_candidate_sha256": row["raw_candidate_sha256"],
        }
    )
    if row["lineage_binding_sha256"] != lineage_binding:
        raise EdgeManifestError(
            f"{location} has a stale lineage_binding_sha256; expected {lineage_binding}"
        )

    binding = _digest(
        {
            "raw_candidate_sha256": row["raw_candidate_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "model_digest": row["model_digest"],
            "split_plan_sha256": row["split_plan_sha256"],
            "trial_ordinal": row["trial_ordinal"],
        }
    )
    if row["binding_sha256"] != binding:
        raise EdgeManifestError(
            f"{location} has a stale binding_sha256; expected {binding}"
        )


__all__ = [
    "COMPARATORS",
    "EDGE_MANIFEST_EVIDENCE_CLASS",
    "EDGE_MANIFEST_SCHEMA_VERSION",
    "IDEA_SOURCE_TYPES",
    "NUMERIC_COMPARATORS",
    "OUT_OF_SELECTION_SAMPLES",
    "SET_COMPARATORS",
    "THRESHOLD_UNITS",
    "CandidateValidation",
    "ConditionCheck",
    "DeclaredCandidateLineage",
    "EdgeCondition",
    "EdgeHealthReport",
    "EdgeManifest",
    "EdgeManifestError",
    "EdgeManifestLedger",
    "IdeaProvenance",
    "ManifestedCandidate",
    "Threshold",
    "VerificationPlan",
    "bind_idea_source",
    "evaluate_condition_against",
    "monitor_edge",
    "parse_candidate_lineage",
    "parse_condition",
    "parse_edge_manifest",
    "parse_idea_provenance",
    "parse_threshold",
    "parse_verification_plan",
    "read_manifest_ledger",
]
