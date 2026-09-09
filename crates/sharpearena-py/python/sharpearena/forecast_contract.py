"""Closed, versioned contracts for prospective forecasts.

A forecast is not reproducible from a question and an answer alone.  The target,
observation boundary, missing-data rule, and scoring rule must all be fixed before
the forecast is submitted.  ``ForecastContract`` is that frozen pre-commit object.

The module deliberately uses an integer logical clock.  SharpeArena can therefore
exercise the complete lifecycle without claiming that a local process proves wall
time.  A neutral host may bind the same integers to public timestamps later.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .canonical_json import canonical_sha256_v1


FORECAST_CONTRACT_SCHEMA_VERSION = "sharpearena.forecast-contract.v1"
CLAIMS_SCHEMA_VERSION = "sharpearena.deferred-claims.v1"
FORECAST_EVIDENCE_SCHEMA_VERSION = "sharpe.forecast-evidence.v1"

POINT = "point"
PROBABILITY = "probability"
CATEGORICAL = "categorical"
NORMAL = "normal"
DIRECTION = "direction"
INTERVAL = "interval"
KINDS = (POINT, PROBABILITY, CATEGORICAL, NORMAL, DIRECTION, INTERVAL)

POINT_ERRORS = "point_errors"
BINARY_BRIER = "binary_brier"
BINARY_LOG = "binary_log"
CATEGORICAL_BRIER = "categorical_brier"
CATEGORICAL_LOG = "categorical_log"
NORMAL_CRPS = "normal_crps"
DIRECTION_ACCURACY = "direction_accuracy"
INTERVAL_SCORE = "interval_score"

SCORING_RULES_BY_KIND = {
    POINT: (POINT_ERRORS,),
    PROBABILITY: (BINARY_BRIER, BINARY_LOG),
    CATEGORICAL: (CATEGORICAL_BRIER, CATEGORICAL_LOG),
    NORMAL: (NORMAL_CRPS,),
    DIRECTION: (DIRECTION_ACCURACY,),
    INTERVAL: (INTERVAL_SCORE,),
}

DEFAULT_SCORING_RULE = {kind: rules[0] for kind, rules in SCORING_RULES_BY_KIND.items()}

_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "question",
        "instrument",
        "target",
        "kind",
        "opens_at",
        "deadline",
        "resolves_at",
        "observation_source",
        "open_definition",
        "close_definition",
        "unit",
        "scoring_rule",
        "neutral_threshold",
        "boundary_ownership",
        "missing_data_policy",
        "fallback_policy",
        "categories",
        "interval_alpha",
    }
)


class ForecastContractError(ValueError):
    """A frozen forecast contract is incomplete, inconsistent, or unknown."""


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForecastContractError(f"{field} must be an integer")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForecastContractError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastContractError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ForecastContractError(f"{field} must be finite")
    return number


def canonical_json(value: object) -> str:
    """Sorted, whitespace-free ``json.dumps`` text: the document and file encoding.

    This is the Python convention, not ``sharpebench/canonical-json/v1``: it
    renders ``1e-05`` where v1 renders ``0.00001`` and ``0.0`` where v1 renders
    ``0``.  Evidence documents, claims documents, field plans and every
    file-level digest are still written in this text.  The contract digest that
    SharpeBench recomputes is not; see :attr:`ForecastContract.sha256`.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def legacy_canonical_sha256(value: object) -> str:
    """The pre-migration contract digest: SHA-256 over the unframed ``json.dumps`` text.

    Kept because published evidence pins it: the frozen prospective forecast
    field carries 24 of these, and SharpeBench dual-accepts them as ``legacy``.
    New contract digests come from :func:`~sharpearena.canonical_json.canonical_sha256_v1`.
    """

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ForecastContract:
    """Everything needed to settle one forecast, frozen before submissions open."""

    contract_id: str
    question: str
    instrument: str
    target: str
    kind: str
    opens_at: int
    deadline: int
    resolves_at: int
    observation_source: str
    open_definition: str
    close_definition: str
    unit: str
    scoring_rule: str
    neutral_threshold: float = 0.0
    boundary_ownership: str = "neutral"
    missing_data_policy: str = "cancel"
    fallback_policy: str = "cancel"
    categories: tuple[str, ...] = ()
    interval_alpha: Optional[float] = None
    schema_version: str = FORECAST_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "contract_id",
            "question",
            "instrument",
            "target",
            "observation_source",
            "open_definition",
            "close_definition",
            "unit",
            "boundary_ownership",
            "missing_data_policy",
            "fallback_policy",
        ):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        if self.schema_version != FORECAST_CONTRACT_SCHEMA_VERSION:
            raise ForecastContractError(
                f"unsupported contract schema_version {self.schema_version!r}"
            )
        if self.kind not in KINDS:
            raise ForecastContractError(f"unknown forecast kind {self.kind!r}")
        allowed = SCORING_RULES_BY_KIND[self.kind]
        if self.scoring_rule not in allowed:
            raise ForecastContractError(
                f"scoring_rule {self.scoring_rule!r} is not valid for {self.kind!r}; "
                f"expected one of {allowed}"
            )
        opens_at = _plain_int(self.opens_at, "opens_at")
        deadline = _plain_int(self.deadline, "deadline")
        resolves_at = _plain_int(self.resolves_at, "resolves_at")
        if opens_at < 0 or not opens_at <= deadline < resolves_at:
            raise ForecastContractError(
                "contract clock must satisfy 0 <= opens_at <= deadline < resolves_at"
            )
        threshold = _finite_number(self.neutral_threshold, "neutral_threshold")
        if threshold < 0.0:
            raise ForecastContractError("neutral_threshold must be finite and non-negative")
        object.__setattr__(self, "neutral_threshold", threshold)
        categories = tuple(_nonempty(item, "categories[]") for item in self.categories)
        if len(set(categories)) != len(categories):
            raise ForecastContractError("categories must be unique")
        if self.kind == CATEGORICAL:
            if len(categories) < 2:
                raise ForecastContractError("a categorical contract needs at least two categories")
        elif categories:
            raise ForecastContractError("categories are only valid for categorical contracts")
        object.__setattr__(self, "categories", categories)
        if self.kind == INTERVAL:
            if self.interval_alpha is None:
                raise ForecastContractError("an interval contract requires interval_alpha")
            alpha = _finite_number(self.interval_alpha, "interval_alpha")
            if not 0.0 < alpha < 1.0:
                raise ForecastContractError("interval_alpha must lie strictly inside (0, 1)")
            object.__setattr__(self, "interval_alpha", alpha)
        elif self.interval_alpha is not None:
            raise ForecastContractError("interval_alpha is only valid for interval contracts")

    @classmethod
    def legacy(
        cls,
        *,
        contract_id: str,
        question: str,
        kind: str,
        committed_at: int,
        horizon: int,
        categories: Sequence[str] = (),
        interval_alpha: Optional[float] = None,
        scoring_rule: Optional[str] = None,
    ) -> "ForecastContract":
        """Construct the explicit contract behind the historical ``commit`` shorthand."""

        committed_at = _plain_int(committed_at, "committed_at")
        horizon = _plain_int(horizon, "horizon")
        if horizon <= 0:
            raise ForecastContractError("horizon must be positive")
        return cls(
            contract_id=contract_id,
            question=question,
            instrument="unspecified",
            target=question,
            kind=kind,
            opens_at=committed_at,
            deadline=committed_at,
            resolves_at=committed_at + horizon,
            observation_source="provided-series",
            open_definition="series[committed_at]",
            close_definition="series[resolves_at]",
            unit="unspecified",
            scoring_rule=scoring_rule or DEFAULT_SCORING_RULE[kind],
            categories=tuple(categories),
            interval_alpha=(0.1 if kind == INTERVAL and interval_alpha is None else interval_alpha),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "question": self.question,
            "instrument": self.instrument,
            "target": self.target,
            "kind": self.kind,
            "opens_at": self.opens_at,
            "deadline": self.deadline,
            "resolves_at": self.resolves_at,
            "observation_source": self.observation_source,
            "open_definition": self.open_definition,
            "close_definition": self.close_definition,
            "unit": self.unit,
            "scoring_rule": self.scoring_rule,
            "neutral_threshold": self.neutral_threshold,
            "boundary_ownership": self.boundary_ownership,
            "missing_data_policy": self.missing_data_policy,
            "fallback_policy": self.fallback_policy,
            "categories": list(self.categories),
            "interval_alpha": self.interval_alpha,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ForecastContract":
        if not isinstance(raw, Mapping):
            raise ForecastContractError("contract must be an object")
        actual = set(raw)
        if actual != _CONTRACT_FIELDS:
            missing = sorted(_CONTRACT_FIELDS - actual)
            unknown = sorted(actual - _CONTRACT_FIELDS)
            raise ForecastContractError(
                f"contract fields do not match v1; missing={missing}, unknown={unknown}"
            )
        categories = raw["categories"]
        if not isinstance(categories, list) or any(not isinstance(x, str) for x in categories):
            raise ForecastContractError("categories must be an array of strings")
        return cls(
            schema_version=_nonempty(raw["schema_version"], "schema_version"),
            contract_id=_nonempty(raw["contract_id"], "contract_id"),
            question=_nonempty(raw["question"], "question"),
            instrument=_nonempty(raw["instrument"], "instrument"),
            target=_nonempty(raw["target"], "target"),
            kind=_nonempty(raw["kind"], "kind"),
            opens_at=_plain_int(raw["opens_at"], "opens_at"),
            deadline=_plain_int(raw["deadline"], "deadline"),
            resolves_at=_plain_int(raw["resolves_at"], "resolves_at"),
            observation_source=_nonempty(raw["observation_source"], "observation_source"),
            open_definition=_nonempty(raw["open_definition"], "open_definition"),
            close_definition=_nonempty(raw["close_definition"], "close_definition"),
            unit=_nonempty(raw["unit"], "unit"),
            scoring_rule=_nonempty(raw["scoring_rule"], "scoring_rule"),
            neutral_threshold=_finite_number(raw["neutral_threshold"], "neutral_threshold"),
            boundary_ownership=_nonempty(raw["boundary_ownership"], "boundary_ownership"),
            missing_data_policy=_nonempty(raw["missing_data_policy"], "missing_data_policy"),
            fallback_policy=_nonempty(raw["fallback_policy"], "fallback_policy"),
            categories=tuple(categories),
            interval_alpha=(
                None
                if raw["interval_alpha"] is None
                else _finite_number(raw["interval_alpha"], "interval_alpha")
            ),
        )

    @property
    def sha256(self) -> str:
        """The contract digest: SHA-256 over the ``sharpebench/canonical-json/v1`` pre-image.

        These are the bytes SharpeBench recomputes and verifies, so a contract
        written here and one written by a Rust producer digest identically.
        """

        return canonical_sha256_v1(self.to_dict())

    @property
    def legacy_sha256(self) -> str:
        """The pre-migration digest the frozen paper field carries; accepted, never produced."""

        return legacy_canonical_sha256(self.to_dict())

    @property
    def digests(self) -> tuple[str, str]:
        """Every digest this contract answers to, current first.

        A reader indexes contracts by all of these so a document produced before
        the migration still binds its revisions to the contract it names.  A
        producer writes only :attr:`sha256`.
        """

        return (self.sha256, self.legacy_sha256)


__all__ = [
    "FORECAST_CONTRACT_SCHEMA_VERSION",
    "CLAIMS_SCHEMA_VERSION",
    "FORECAST_EVIDENCE_SCHEMA_VERSION",
    "POINT",
    "PROBABILITY",
    "CATEGORICAL",
    "NORMAL",
    "DIRECTION",
    "INTERVAL",
    "KINDS",
    "POINT_ERRORS",
    "BINARY_BRIER",
    "BINARY_LOG",
    "CATEGORICAL_BRIER",
    "CATEGORICAL_LOG",
    "NORMAL_CRPS",
    "DIRECTION_ACCURACY",
    "INTERVAL_SCORE",
    "SCORING_RULES_BY_KIND",
    "DEFAULT_SCORING_RULE",
    "ForecastContract",
    "ForecastContractError",
    "canonical_json",
    "legacy_canonical_sha256",
]
