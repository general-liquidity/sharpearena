"""Observed-trial strategy generation with a non-executable trading DSL.

The model proposes JSON, never Python. Every emitted candidate is counted before
validation or deduplication, so the deflation trial count is measured by the host
rather than declared by the entrant. Candidate selection uses a validation split;
only the selected strategy touches the test split.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Optional, Protocol, Sequence

from .edge_manifest import (
    CandidateValidation,
    DeclaredCandidateLineage,
    EdgeManifest,
    EdgeManifestError,
    EdgeManifestLedger,
    IdeaProvenance,
    parse_candidate_lineage,
    parse_edge_manifest,
)
from .local_agents import (
    DatasetSpec,
    LocalAgentError,
    LocalFieldRunner,
    ModelIdentity,
    ModelRunConfig,
    OllamaClient,
)
from .sharpearena_py import score_run

STRATEGY_EVIDENCE_CLASS = "retrospective_generated_strategy"
STRATEGY_SCHEMA_VERSION = 2
MAX_GENERATED_CANDIDATES = 256
SUPPORTED_INDICATORS = {"price", "sma", "ema", "momentum", "rsi", "volatility"}
COMPARISON_OPS = {"gt", "gte", "lt", "lte"}
BOOLEAN_OPS = {"and", "or"}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


STRATEGY_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["strategies"],
    "properties": {
        "strategies": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_GENERATED_CANDIDATES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "thesis",
                    "long_when",
                    "gross_target",
                    "edge_manifest",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "thesis": {"type": "string", "minLength": 1, "maxLength": 500},
                    "long_when": {"$ref": "#/$defs/condition"},
                    "short_when": {
                        "anyOf": [{"$ref": "#/$defs/condition"}, {"type": "null"}]
                    },
                    "gross_target": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "edge_manifest": {"$ref": "#/$defs/edge_manifest"},
                    "lineage": {"$ref": "#/$defs/lineage"},
                },
            },
        }
    },
    "$defs": {
        "value": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["constant"],
                    "properties": {"constant": {"type": "number"}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["indicator"],
                    "properties": {
                        "indicator": {"enum": sorted(SUPPORTED_INDICATORS)},
                        "window": {"type": "integer", "minimum": 2, "maximum": 252},
                    },
                },
            ]
        },
        "condition": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "left", "right"],
                    "properties": {
                        "op": {"enum": sorted(COMPARISON_OPS)},
                        "left": {"$ref": "#/$defs/value"},
                        "right": {"$ref": "#/$defs/value"},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "conditions"],
                    "properties": {
                        "op": {"enum": sorted(BOOLEAN_OPS)},
                        "conditions": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 8,
                            "items": {"$ref": "#/$defs/condition"},
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "condition"],
                    "properties": {
                        "op": {"const": "not"},
                        "condition": {"$ref": "#/$defs/condition"},
                    },
                },
            ]
        },
        "threshold": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "unit"],
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                    ]
                },
                "unit": {
                    "enum": [
                        "basis_points",
                        "categorical",
                        "count",
                        "days",
                        "fraction",
                        "percent",
                        "ratio",
                        "seconds",
                        "usd",
                        "z_score",
                    ]
                },
            },
        },
        "lineage": {
            "type": "object",
            "additionalProperties": False,
            "required": ["parent_candidate_ids", "idea_source_digests"],
            "properties": {
                "parent_candidate_ids": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "idea_source_digests": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": "^sha256:[0-9a-f]{64}$",
                    },
                },
            },
        },
        "edge_condition": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "condition_id",
                "metric",
                "comparator",
                "threshold",
                "description",
            ],
            "properties": {
                "condition_id": {"type": "string", "minLength": 1},
                "metric": {"type": "string", "minLength": 1},
                "comparator": {
                    "enum": ["eq", "gt", "gte", "in", "lt", "lte", "neq", "not_in"]
                },
                "threshold": {"$ref": "#/$defs/threshold"},
                "description": {"type": "string", "minLength": 1},
            },
        },
        "verification_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "selection_metric",
                "selection_split",
                "confirmation_split",
                "minimum_observations",
            ],
            "properties": {
                "selection_metric": {"type": "string", "minLength": 1},
                "selection_split": {"type": "string", "minLength": 1},
                "confirmation_split": {"type": "string", "minLength": 1},
                "minimum_observations": {"type": "integer", "minimum": 2},
            },
        },
        "edge_manifest": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "hypothesis",
                "mechanism",
                "regimes",
                "instruments",
                "invariants",
                "kill_conditions",
                "verification_plan",
            ],
            "properties": {
                "hypothesis": {"type": "string", "minLength": 8},
                "mechanism": {"type": "string", "minLength": 8},
                "regimes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "instruments": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "invariants": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {"$ref": "#/$defs/edge_condition"},
                },
                "kill_conditions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {"$ref": "#/$defs/edge_condition"},
                },
                "verification_plan": {"$ref": "#/$defs/verification_plan"},
            },
        },
    },
}


class StrategyProtocolError(ValueError):
    """A generation response or DSL candidate violates the closed protocol."""


def _family_value_shape(value: dict[str, Any]) -> dict[str, Any]:
    """Erase tunable values while preserving the host-validated signal shape."""

    if "constant" in value:
        return {"constant": "parameter"}
    shaped = {"indicator": value["indicator"]}
    if value["indicator"] != "price":
        shaped["window"] = "parameter"
    return shaped


def _family_condition_shape(condition: dict[str, Any]) -> dict[str, Any]:
    op = condition["op"]
    if op in COMPARISON_OPS:
        return {
            "op": op,
            "left": _family_value_shape(condition["left"]),
            "right": _family_value_shape(condition["right"]),
        }
    if op in BOOLEAN_OPS:
        return {
            "op": op,
            "conditions": [
                _family_condition_shape(item) for item in condition["conditions"]
            ],
        }
    return {"op": "not", "condition": _family_condition_shape(condition["condition"])}


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    thesis: str
    long_when: dict[str, Any]
    short_when: Optional[dict[str, Any]]
    gross_target: float
    edge_manifest: EdgeManifest
    declared_lineage: Optional[DeclaredCandidateLineage] = None
    trial_ordinal: int = -1
    manifest_sha256: str = ""
    binding_sha256: str = ""
    family_digest: str = ""
    parent_candidate_digests: tuple[str, ...] = ()
    generator_identity_sha256: str = ""
    idea_provenance: tuple[IdeaProvenance, ...] = ()
    lineage_binding_sha256: str = ""

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "long_when": self.long_when,
                "short_when": self.short_when,
                "gross_target": self.gross_target,
            }
        )

    @property
    def family_preimage(self) -> dict[str, Any]:
        """Conceptual family, excluding parameter values and position size.

        This is derived by the host rather than accepted as a model-provided
        family label. Thresholds, indicator windows, and ``gross_target`` are
        tunable variants; signal operators plus declared market scope define the
        conceptual family, mirroring AIUTS's family-versus-candidate split.
        """

        return {
            "long_when": _family_condition_shape(self.long_when),
            "short_when": (
                None
                if self.short_when is None
                else _family_condition_shape(self.short_when)
            ),
            "regimes": sorted(self.edge_manifest.regimes),
            "instruments": sorted(self.edge_manifest.instruments),
        }

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "thesis": self.thesis,
            "long_when": self.long_when,
            "short_when": self.short_when,
            "gross_target": self.gross_target,
            "edge_manifest": self.edge_manifest.as_record(),
            "declared_lineage": (
                None
                if self.declared_lineage is None
                else self.declared_lineage.as_record()
            ),
            "trial_ordinal": self.trial_ordinal,
            "manifest_sha256": self.manifest_sha256,
            "binding_sha256": self.binding_sha256,
            "fingerprint": self.fingerprint,
            "family_preimage": self.family_preimage,
            "family_digest": self.family_digest,
            "parent_candidate_digests": list(self.parent_candidate_digests),
            "generator_identity_sha256": self.generator_identity_sha256,
            "idea_provenance": [source.as_record() for source in self.idea_provenance],
            "lineage_binding_sha256": self.lineage_binding_sha256,
        }


@dataclass(frozen=True)
class CandidateRejection:
    index: int
    candidate_id: Optional[str]
    reason: str


@dataclass(frozen=True)
class GenerationResult:
    raw_response: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_duration_ns: int = 0


class StrategyGenerator(Protocol):
    def identity(self, model: ModelRunConfig) -> ModelIdentity: ...

    def generate(
        self, model: ModelRunConfig, prompt: str, requested_candidates: int
    ) -> GenerationResult: ...


class OllamaStrategyGenerator:
    """Schema-constrained strategy generation through a loopback Ollama server."""

    def __init__(self, client: OllamaClient):
        self.client = client

    def identity(self, model: ModelRunConfig) -> ModelIdentity:
        return self.client.identity(model)

    def generate(
        self, model: ModelRunConfig, prompt: str, requested_candidates: int
    ) -> GenerationResult:
        response = self.client._request(  # noqa: SLF001 - same-package transport reuse
            "POST",
            "/api/chat",
            {
                "model": model.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Generate point-in-time trading strategies in the supplied closed "
                            "JSON DSL. Do not emit code, tools, prose outside JSON, or future data. "
                            "Attach the required edge_manifest to every candidate before seeing "
                            "any validation or test result. When a lineage object is present, its "
                            "parents must name earlier candidates in this response and its source "
                            "digests must come from the operator-bound catalog in the prompt. "
                            f"Return exactly {requested_candidates} candidate objects."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": STRATEGY_GENERATION_SCHEMA,
                "think": model.sampling.thinking,
                "options": {
                    "temperature": model.sampling.temperature,
                    "top_p": model.sampling.top_p,
                    "seed": model.sampling.seed,
                    "num_predict": model.sampling.max_tokens,
                },
            },
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAgentError("Ollama generation response has no message.content")
        return GenerationResult(
            raw_response=message["content"],
            prompt_tokens=int(response.get("prompt_eval_count", 0) or 0),
            output_tokens=int(response.get("eval_count", 0) or 0),
            total_duration_ns=int(response.get("total_duration", 0) or 0),
        )


def _closed_object(
    value: Any, allowed: set[str], required: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrategyProtocolError(f"{path} must be an object")
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise StrategyProtocolError(f"{path} has unknown fields: {sorted(unknown)}")
    if missing:
        raise StrategyProtocolError(f"{path} is missing fields: {sorted(missing)}")
    return value


def _validate_value(value: Any, path: str) -> dict[str, Any]:
    obj = _closed_object(value, {"constant", "indicator", "window"}, set(), path)
    if set(obj) == {"constant"}:
        number = obj["constant"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise StrategyProtocolError(f"{path}.constant must be finite")
        return {"constant": float(number)}
    if "indicator" not in obj or set(obj) - {"indicator", "window"}:
        raise StrategyProtocolError(
            f"{path} must contain one constant or one indicator"
        )
    indicator = obj["indicator"]
    if indicator not in SUPPORTED_INDICATORS:
        raise StrategyProtocolError(f"{path}.indicator is unsupported")
    window = obj.get("window", 2)
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or not 2 <= window <= 252
    ):
        raise StrategyProtocolError(f"{path}.window must be an integer in [2, 252]")
    if indicator == "price" and "window" in obj:
        raise StrategyProtocolError(f"{path}.price does not take a window")
    return {
        "indicator": indicator,
        **({} if indicator == "price" else {"window": window}),
    }


def _validate_condition(
    value: Any, path: str = "condition", depth: int = 0
) -> dict[str, Any]:
    if depth > 8:
        raise StrategyProtocolError(f"{path} exceeds maximum nesting depth 8")
    obj = _closed_object(
        value, {"op", "left", "right", "conditions", "condition"}, {"op"}, path
    )
    op = obj["op"]
    if op in COMPARISON_OPS:
        _closed_object(obj, {"op", "left", "right"}, {"op", "left", "right"}, path)
        return {
            "op": op,
            "left": _validate_value(obj["left"], f"{path}.left"),
            "right": _validate_value(obj["right"], f"{path}.right"),
        }
    if op in BOOLEAN_OPS:
        _closed_object(obj, {"op", "conditions"}, {"op", "conditions"}, path)
        conditions = obj["conditions"]
        if not isinstance(conditions, list) or not 2 <= len(conditions) <= 8:
            raise StrategyProtocolError(f"{path}.conditions must contain 2..8 items")
        return {
            "op": op,
            "conditions": [
                _validate_condition(item, f"{path}.conditions[{index}]", depth + 1)
                for index, item in enumerate(conditions)
            ],
        }
    if op == "not":
        _closed_object(obj, {"op", "condition"}, {"op", "condition"}, path)
        return {
            "op": "not",
            "condition": _validate_condition(
                obj["condition"], f"{path}.condition", depth + 1
            ),
        }
    raise StrategyProtocolError(f"{path}.op is unsupported")


def parse_generated_pool(
    raw_response: str,
    *,
    ledger: Optional[EdgeManifestLedger] = None,
) -> tuple[int, list[StrategyCandidate], list[CandidateRejection]]:
    """Validate a response while counting candidates before any rejection.

    The edge-manifest ledger owns the ordering.  It assigns a trial ordinal,
    validates the manifest, invokes the strategy validator, and only then
    deduplicates.  The returned ``observed`` count therefore includes malformed
    and duplicate proposals, while every accepted strategy carries the binding
    hash that ties it to the model and split plan used by the caller.
    """

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise StrategyProtocolError(
            f"generation response is not JSON: {error}"
        ) from error
    root = _closed_object(payload, {"strategies"}, {"strategies"}, "response")
    raw = root["strategies"]
    if not isinstance(raw, list):
        raise StrategyProtocolError("response.strategies must be an array")
    if not raw:
        raise StrategyProtocolError("response.strategies must not be empty")
    if len(raw) > MAX_GENERATED_CANDIDATES:
        raise StrategyProtocolError(
            f"response.strategies exceeds the hard cap of {MAX_GENERATED_CANDIDATES}"
        )

    manifest_ledger = ledger or EdgeManifestLedger(
        model_digest="unbound-parser-model",
        split_plan_sha256="unbound-parser-split",
    )
    parsed: dict[int, StrategyCandidate] = {}
    seen_ids: set[str] = set()

    def validate_strategy(item: dict[str, Any]) -> CandidateValidation:
        index = manifest_ledger.observed_trials
        obj = _closed_object(
            item,
            {
                "id",
                "thesis",
                "long_when",
                "short_when",
                "gross_target",
                "edge_manifest",
                "lineage",
            },
            {"id", "thesis", "long_when", "gross_target", "edge_manifest"},
            f"strategies[{index}]",
        )
        try:
            if (
                not isinstance(obj["id"], str)
                or not obj["id"].strip()
                or len(obj["id"]) > 80
            ):
                raise StrategyProtocolError(
                    "id must be a non-empty string of at most 80 characters"
                )
            if (
                not isinstance(obj["thesis"], str)
                or not obj["thesis"].strip()
                or len(obj["thesis"]) > 500
            ):
                raise StrategyProtocolError(
                    "thesis must be a non-empty string of at most 500 characters"
                )
            gross = obj["gross_target"]
            if (
                isinstance(gross, bool)
                or not isinstance(gross, (int, float))
                or not math.isfinite(gross)
            ):
                raise StrategyProtocolError("gross_target must be finite")
            if not 0.0 < float(gross) <= 1.0:
                raise StrategyProtocolError("gross_target must lie in (0, 1]")
            candidate = StrategyCandidate(
                candidate_id=obj["id"].strip(),
                thesis=obj["thesis"].strip(),
                long_when=_validate_condition(obj["long_when"], "long_when"),
                short_when=(
                    None
                    if obj.get("short_when") is None
                    else _validate_condition(obj["short_when"], "short_when")
                ),
                gross_target=float(gross),
                # The ledger has already parsed this before invoking the
                # callback. Parsing again produces the typed value carried by
                # the candidate without allowing strategy validation to run
                # ahead of manifest validation.
                edge_manifest=parse_edge_manifest(obj["edge_manifest"]),
                declared_lineage=(
                    None
                    if obj.get("lineage") is None
                    else parse_candidate_lineage(obj["lineage"])
                ),
            )
            if candidate.candidate_id in seen_ids:
                raise StrategyProtocolError("duplicate candidate id")
            seen_ids.add(candidate.candidate_id)
            parsed[index] = candidate
            return CandidateValidation(
                semantic_fingerprint=candidate.fingerprint,
                family_preimage=candidate.family_preimage,
                candidate_id=candidate.candidate_id,
            )
        except EdgeManifestError as error:
            raise StrategyProtocolError(str(error)) from error

    records = manifest_ledger.record_pool(raw, candidate_validator=validate_strategy)
    candidates: list[StrategyCandidate] = []
    rejected: list[CandidateRejection] = []
    for record in records:
        candidate_id = (
            record.raw_candidate.get("id")
            if isinstance(record.raw_candidate.get("id"), str)
            else None
        )
        if not record.is_selectable:
            reason = record.invalid_reason
            if reason is None:
                reason = (
                    "duplicate strategy and edge manifest; first proposed at trial "
                    f"{record.duplicate_of_ordinal}"
                )
            rejected.append(
                CandidateRejection(record.trial_ordinal, candidate_id, reason)
            )
            continue
        candidate = parsed[record.trial_ordinal]
        candidates.append(
            replace(
                candidate,
                trial_ordinal=record.trial_ordinal,
                manifest_sha256=record.manifest_sha256 or "",
                binding_sha256=record.binding_sha256,
                family_digest=record.family_digest,
                parent_candidate_digests=record.parent_candidate_digests,
                generator_identity_sha256=record.generator_identity_sha256,
                idea_provenance=record.idea_provenance,
                lineage_binding_sha256=record.lineage_binding_sha256,
            )
        )
    return manifest_ledger.observed_trials, candidates, rejected


def _indicator(spec: dict[str, Any], prices: Sequence[float]) -> Optional[float]:
    if not prices or any(not math.isfinite(float(value)) for value in prices):
        return None
    name = spec.get("indicator")
    if name == "price":
        return float(prices[-1])
    window = int(spec["window"])
    if len(prices) < window:
        return None
    tail = [float(value) for value in prices[-window:]]
    if name == "sma":
        return sum(tail) / window
    if name == "ema":
        alpha = 2.0 / (window + 1.0)
        value = tail[0]
        for price in tail[1:]:
            value = alpha * price + (1.0 - alpha) * value
        return value
    if name == "momentum":
        return tail[-1] / tail[0] - 1.0 if tail[0] else None
    returns = [b / a - 1.0 for a, b in zip(tail, tail[1:]) if a]
    if len(returns) < 1:
        return None
    if name == "volatility":
        return pstdev(returns) if len(returns) >= 2 else 0.0
    if name == "rsi":
        changes = [b - a for a, b in zip(tail, tail[1:])]
        gains = sum(max(change, 0.0) for change in changes) / len(changes)
        losses = sum(max(-change, 0.0) for change in changes) / len(changes)
        if losses == 0.0:
            return 100.0 if gains > 0.0 else 50.0
        return 100.0 - 100.0 / (1.0 + gains / losses)
    return None


def _value(spec: dict[str, Any], prices: Sequence[float]) -> Optional[float]:
    if "constant" in spec:
        return float(spec["constant"])
    return _indicator(spec, prices)


def evaluate_condition(condition: dict[str, Any], prices: Sequence[float]) -> bool:
    """Evaluate one validated condition using only trailing prices."""

    op = condition["op"]
    if op in BOOLEAN_OPS:
        values = [evaluate_condition(item, prices) for item in condition["conditions"]]
        return all(values) if op == "and" else any(values)
    if op == "not":
        return not evaluate_condition(condition["condition"], prices)
    left = _value(condition["left"], prices)
    right = _value(condition["right"], prices)
    if left is None or right is None:
        return False
    return {
        "gt": left > right,
        "gte": left >= right,
        "lt": left < right,
        "lte": left <= right,
    }[op]


def strategy_decision(
    candidate: StrategyCandidate, observation: dict[str, Any]
) -> dict[str, Any]:
    signals: list[tuple[str, int]] = []
    for item in observation.get("symbols", []):
        symbol = str(item["symbol"])
        prices = [float(value) for value in item.get("close_history", [])]
        long = evaluate_condition(candidate.long_when, prices)
        short = candidate.short_when is not None and evaluate_condition(
            candidate.short_when, prices
        )
        signal = 1 if long and not short else -1 if short and not long else 0
        signals.append((symbol, signal))
    active = sum(signal != 0 for _, signal in signals)
    unit = candidate.gross_target / active if active else 0.0
    return {
        "orders": [
            {
                "symbol": symbol,
                "action": "buy" if signal > 0 else "sell" if signal < 0 else "close",
                "target_weight": signal * unit,
            }
            for symbol, signal in signals
        ],
        "reasoning": f"generated DSL strategy {candidate.candidate_id}",
    }


@dataclass(frozen=True)
class StrategySearchPlan:
    model: ModelRunConfig
    prompt: str
    requested_candidates: int
    validation_dataset: DatasetSpec
    test_dataset: DatasetSpec
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    max_steps: Optional[int] = None
    idea_provenance: tuple[IdeaProvenance, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.requested_candidates <= MAX_GENERATED_CANDIDATES:
            raise ValueError(
                f"requested_candidates must lie in [1, {MAX_GENERATED_CANDIDATES}]"
            )
        if not self.validation_seeds or not self.test_seeds:
            raise ValueError("validation and test seeds must be non-empty")
        if not self.prompt.strip():
            raise ValueError("strategy-generation prompt must be non-empty")
        if len(set(self.validation_seeds)) != len(self.validation_seeds) or len(
            set(self.test_seeds)
        ) != len(self.test_seeds):
            raise ValueError(
                "validation and test seeds must be unique within each split"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
            for seed in (*self.validation_seeds, *self.test_seeds)
        ):
            raise ValueError("strategy search seeds must be unsigned 64-bit integers")
        if set(self.validation_seeds) & set(self.test_seeds):
            raise ValueError("validation and test seeds must be disjoint")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be positive when supplied")
        if self.validation_dataset.content_sha256 == self.test_dataset.content_sha256:
            validation = (
                self.validation_dataset.window_start,
                self.validation_dataset.window_end,
            )
            test = (self.test_dataset.window_start, self.test_dataset.window_end)
            if None in validation or None in test:
                raise ValueError(
                    "validation and test windows over the same data must be explicit and disjoint"
                )
            validation_start, validation_end = validation
            test_start, test_end = test
            assert validation_start is not None and validation_end is not None
            assert test_start is not None and test_end is not None
            if max(validation_start, test_start) < min(validation_end, test_end):
                raise ValueError(
                    "validation and test datasets/windows must be disjoint"
                )
        source_digests = [source.source_digest for source in self.idea_provenance]
        if len(set(source_digests)) != len(source_digests):
            raise ValueError("idea_provenance source digests must be unique")

    @property
    def generation_prompt(self) -> str:
        """Exact model prompt, including the operator-bound source catalog."""

        if not self.idea_provenance:
            return self.prompt
        catalog = [source.as_record() for source in self.idea_provenance]
        return (
            f"{self.prompt.rstrip()}\n\n"
            "Operator-bound idea-source catalog (cite only these source_digest values "
            "in lineage.idea_source_digests):\n"
            f"{json.dumps(catalog, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"
        )

    @property
    def plan_sha256(self) -> str:
        return _digest(
            {
                "model": asdict(self.model),
                "prompt": self.prompt,
                "requested_candidates": self.requested_candidates,
                "validation_dataset": self.validation_dataset.public_record(),
                "test_dataset": self.test_dataset.public_record(),
                "validation_seeds": self.validation_seeds,
                "test_seeds": self.test_seeds,
                "max_steps": self.max_steps,
                "idea_provenance": [
                    source.as_record() for source in self.idea_provenance
                ],
            }
        )


def _evaluate_candidates(
    candidates: Sequence[StrategyCandidate],
    dataset: DatasetSpec,
    seeds: Sequence[int],
    n_trials: int,
    max_steps: Optional[int],
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    for candidate in candidates:
        env = LocalFieldRunner._build_env(dataset, seeds)
        observations = json.loads(env.reset_batch())["observations"]
        active = [True] * len(seeds)
        returns: list[list[float]] = [[] for _ in seeds]
        steps = 0
        while any(active) and steps < (max_steps or 1_000_000):
            decisions = [
                strategy_decision(candidate, observation)
                if active[index]
                else {"orders": [], "reasoning": "inactive lane"}
                for index, observation in enumerate(observations)
            ]
            stepped = json.loads(env.step_batch(json.dumps(decisions)))
            for index, is_active in enumerate(active):
                if not is_active:
                    continue
                reward = float(stepped["rewards"][index])
                if not math.isfinite(reward):
                    raise RuntimeError(
                        "generated strategy produced a non-finite reward"
                    )
                returns[index].append(reward)
                if stepped["terminated"][index] or stepped["truncated"][index]:
                    active[index] = False
            observations = stepped["observations"]
            steps += 1
        for seed, series in zip(seeds, returns):
            if len(series) < 2:
                raise RuntimeError("generated strategy produced fewer than two returns")
            results[candidate.candidate_id].append(
                {
                    "seed": int(seed),
                    "returns_sha256": _digest(series),
                    "n_returns": len(series),
                    "score": json.loads(
                        score_run(series, n_trials, dataset.periods_per_year)
                    ),
                }
            )
    return results


class StrategySearchRunner:
    """Generate, count, validate, select, and test strategies without executing code."""

    def __init__(self, generator: StrategyGenerator):
        self.generator = generator

    def run(self, plan: StrategySearchPlan, evidence_path: Path) -> dict[str, Any]:
        identity = self.generator.identity(plan.model)
        generated: Optional[GenerationResult] = None
        manifest_ledger: Optional[EdgeManifestLedger] = None
        try:
            generated = self.generator.generate(
                plan.model, plan.generation_prompt, plan.requested_candidates
            )
            manifest_ledger = EdgeManifestLedger(
                model_digest=identity.digest,
                split_plan_sha256=plan.plan_sha256,
                generator_identity=asdict(identity),
                idea_provenance=plan.idea_provenance,
            )
            observed_n_trials, candidates, rejected = parse_generated_pool(
                generated.raw_response, ledger=manifest_ledger
            )
            if not candidates:
                raise StrategyProtocolError("the model emitted no valid candidate")
            validation = _evaluate_candidates(
                candidates,
                plan.validation_dataset,
                plan.validation_seeds,
                observed_n_trials,
                plan.max_steps,
            )
            ranking = []
            for candidate in candidates:
                scores = validation[candidate.candidate_id]
                value = median(
                    float(item["score"]["deflated_sharpe"]) for item in scores
                )
                ranking.append((value, candidate.candidate_id, candidate))
            ranking.sort(key=lambda item: (-item[0], item[1]))
            selected = ranking[0][2]
            test = _evaluate_candidates(
                [selected],
                plan.test_dataset,
                plan.test_seeds,
                observed_n_trials,
                plan.max_steps,
            )[selected.candidate_id]
            evidence = {
                "schema_version": STRATEGY_SCHEMA_VERSION,
                "evidence_class": STRATEGY_EVIDENCE_CLASS,
                "status": "completed",
                "deterministic_environment": True,
                "deterministic_generation": False,
                "generated_code_executed": False,
                "plan_sha256": plan.plan_sha256,
                "model": asdict(identity),
                "model_config": asdict(plan.model),
                "generation": {
                    "requested_candidates": plan.requested_candidates,
                    "observed_n_trials": observed_n_trials,
                    "n_trials_source": manifest_ledger.summary()["n_trials_source"],
                    "edge_manifest_ledger": {
                        "summary": manifest_ledger.summary(),
                        "records": [
                            record.as_record() for record in manifest_ledger.records
                        ],
                    },
                    "raw_response_sha256": sha256(
                        generated.raw_response.encode("utf-8")
                    ).hexdigest(),
                    "raw_response": generated.raw_response,
                    "prompt": plan.generation_prompt,
                    "prompt_sha256": sha256(
                        plan.generation_prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_tokens": generated.prompt_tokens,
                    "output_tokens": generated.output_tokens,
                    "duration_ns": generated.total_duration_ns,
                    "accepted": [candidate.as_record() for candidate in candidates],
                    "rejected": [asdict(item) for item in rejected],
                },
                "selection": {
                    "split": plan.validation_dataset.public_record(),
                    "seeds": list(plan.validation_seeds),
                    "metric": "median per-seed deflated_sharpe",
                    "scores": validation,
                    "selected_candidate_id": selected.candidate_id,
                },
                "test": {
                    "split": plan.test_dataset.public_record(),
                    "seeds": list(plan.test_seeds),
                    "selected_candidate_only": True,
                    "scores": test,
                },
                "recorded_at_unix_ns": time.time_ns(),
            }
            normalized = json.loads(_canonical_bytes(evidence))
            _write_evidence(evidence_path, normalized)
            return normalized
        except Exception as error:
            observed = None
            if generated is not None:
                try:
                    payload = json.loads(generated.raw_response)
                    pool = (
                        payload.get("strategies") if isinstance(payload, dict) else None
                    )
                    observed = len(pool) if isinstance(pool, list) else None
                except json.JSONDecodeError:
                    observed = None
            failure = {
                "schema_version": STRATEGY_SCHEMA_VERSION,
                "evidence_class": STRATEGY_EVIDENCE_CLASS,
                "status": "failed",
                "deterministic_environment": True,
                "deterministic_generation": False,
                "generated_code_executed": False,
                "plan_sha256": plan.plan_sha256,
                "model": asdict(identity),
                "model_config": asdict(plan.model),
                "generation": {
                    "requested_candidates": plan.requested_candidates,
                    "observed_n_trials": observed,
                    "n_trials_source": (
                        manifest_ledger.summary()["n_trials_source"]
                        if manifest_ledger is not None
                        else None
                    ),
                    "edge_manifest_ledger": (
                        {
                            "summary": manifest_ledger.summary(),
                            "records": [
                                record.as_record() for record in manifest_ledger.records
                            ],
                        }
                        if manifest_ledger is not None
                        else None
                    ),
                    "raw_response_sha256": (
                        sha256(generated.raw_response.encode("utf-8")).hexdigest()
                        if generated is not None
                        else None
                    ),
                    "raw_response": (
                        generated.raw_response if generated is not None else None
                    ),
                    "prompt": plan.prompt,
                    "prompt_sha256": sha256(plan.prompt.encode("utf-8")).hexdigest(),
                },
                "failure": {"type": type(error).__name__, "detail": str(error)},
                "recorded_at_unix_ns": time.time_ns(),
            }
            _write_evidence(evidence_path, json.loads(_canonical_bytes(failure)))
            raise


def _write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    """Append one complete search record without replacing earlier trials."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(evidence) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "CandidateRejection",
    "GenerationResult",
    "OllamaStrategyGenerator",
    "MAX_GENERATED_CANDIDATES",
    "STRATEGY_GENERATION_SCHEMA",
    "StrategyCandidate",
    "StrategyGenerator",
    "StrategyProtocolError",
    "StrategySearchPlan",
    "StrategySearchRunner",
    "evaluate_condition",
    "parse_generated_pool",
    "strategy_decision",
]
