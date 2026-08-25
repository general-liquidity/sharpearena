"""Local open-weight model agents over the deterministic SharpeArena environment.

This module is the host-side experiment harness. The market engine remains
byte-replayable; model inference is explicitly recorded as nondeterministic evidence.
Ollama is treated as a trusted local inference service. Generated executable code is
*not* run here and belongs behind the separate no-network sandbox boundary.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .decision_parser import (
    DecisionParseError,
    decision_to_weights,
    parse_decision_payload,
)
from .sharpearena_py import VecTradingEnv, decision_schema_json, score_run

EVIDENCE_SCHEMA_VERSION = 1
LOCAL_EVIDENCE_CLASS = "retrospective_local_model"


class LocalAgentError(RuntimeError):
    """A local model request, response, or experiment cell failed."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep a loopback-only inference request on the committed origin."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SamplingConfig:
    """Inference settings committed as part of a model entry."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 512
    thinking: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must lie in [0, 2]")
        if not math.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        if isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class ModelRunConfig:
    """One model/scaffold arm in the field."""

    model: str
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scaffold: str = "minimal-stateless-v1"
    decision_cadence: int = 1
    precommitted_n_trials: int = 1
    entry_class: str = "unverified-local"
    source_url: Optional[str] = None
    license_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.decision_cadence <= 0:
            raise ValueError("decision_cadence must be positive")
        if self.precommitted_n_trials <= 0:
            raise ValueError("precommitted_n_trials must be positive")
        if self.entry_class not in {"field", "host", "unverified-local"}:
            raise ValueError("entry_class must be field, host, or unverified-local")
        if self.entry_class == "field" and (not self.source_url or not self.license_id):
            raise ValueError(
                "field entries require source_url and license_id provenance"
            )
        if self.source_url is not None:
            parsed = urlsplit(self.source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("source_url must be an absolute HTTPS URL")
        if self.license_id is not None and not self.license_id.strip():
            raise ValueError("license_id must be non-empty when supplied")


@dataclass(frozen=True)
class ModelIdentity:
    """Artifact identity returned by the local model server."""

    model: str
    digest: str
    parameter_size: str = "unknown"
    quantization: str = "unknown"
    offload: str = "unknown"
    family: str = "unknown"
    context_length: Optional[int] = None
    server: str = "ollama"
    server_version: str = "unknown"
    size_bytes: Optional[int] = None
    format: str = "unknown"
    capabilities: tuple[str, ...] = ()
    license_sha256: Optional[str] = None
    modelfile_sha256: Optional[str] = None
    template_sha256: Optional[str] = None
    parameters_sha256: Optional[str] = None


def load_identity_manifest(path: os.PathLike[str] | str) -> tuple[ModelIdentity, ...]:
    """Load explicit local-backend/model provenance from a closed JSON manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(
            "identity manifest must be an object or non-empty object array"
        )
    allowed = {item.name for item in fields(ModelIdentity)}
    identities = []
    for index, record in enumerate(records):
        unknown = set(record) - allowed
        if unknown:
            raise ValueError(
                f"identity manifest record {index} has unknown fields: {sorted(unknown)}"
            )
        normalized = dict(record)
        if "capabilities" in normalized:
            capabilities = normalized["capabilities"]
            if not isinstance(capabilities, list) or not all(
                isinstance(value, str) for value in capabilities
            ):
                raise ValueError(
                    f"identity manifest record {index} capabilities must be a string array"
                )
            normalized["capabilities"] = tuple(capabilities)
        identities.append(ModelIdentity(**normalized))
    return tuple(identities)


@dataclass(frozen=True)
class DatasetSpec:
    """Synthetic tier or frozen historical panel used by a field run."""

    dataset_id: str
    tier: str = "calm"
    csv_text: Optional[str] = field(default=None, repr=False)
    n_symbols: int = 4
    n_days: int = 120
    periods_per_year: float = 252.0
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    fee_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    impact_bps: Optional[float] = None
    financing_bps: Optional[float] = None
    max_participation: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if self.csv_text is None and self.tier not in {"calm", "hard", "extreme"}:
            raise ValueError("synthetic tier must be calm, hard, or extreme")
        if self.csv_text is not None and not self.csv_text.strip():
            raise ValueError("csv_text must be non-empty when supplied")
        if self.n_symbols <= 0 or self.n_days <= 1:
            raise ValueError("synthetic panel dimensions must be positive")
        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0.0:
            raise ValueError("periods_per_year must be finite and positive")
        for name, value in (
            ("window_start", self.window_start),
            ("window_end", self.window_end),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer when supplied")
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_start >= self.window_end
        ):
            raise ValueError("window_start must be smaller than window_end")
        if (
            self.csv_text is None
            and self.window_end is not None
            and self.window_end > self.n_days
        ):
            raise ValueError("synthetic window_end cannot exceed n_days")
        for name, value in (
            ("fee_bps", self.fee_bps),
            ("slippage_bps", self.slippage_bps),
            ("impact_bps", self.impact_bps),
            ("financing_bps", self.financing_bps),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_participation is not None and (
            not math.isfinite(self.max_participation)
            or not 0.0 < self.max_participation <= 1.0
        ):
            raise ValueError("max_participation must lie in (0, 1]")

    @property
    def kind(self) -> str:
        return "historical" if self.csv_text is not None else "synthetic"

    @property
    def content_sha256(self) -> str:
        if self.csv_text is not None:
            return sha256(self.csv_text.encode("utf-8")).hexdigest()
        return _digest(
            {
                "kind": "synthetic",
                "tier": self.tier,
                "n_symbols": self.n_symbols,
                "n_days": self.n_days,
            }
        )

    def public_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("csv_text")
        record.update({"kind": self.kind, "content_sha256": self.content_sha256})
        return record


@dataclass(frozen=True)
class FieldPlan:
    """Precommitted Cartesian field and its reproducibility controls."""

    models: tuple[ModelRunConfig, ...]
    datasets: tuple[DatasetSpec, ...]
    seeds: tuple[int, ...]
    repetitions: int = 4
    max_steps: Optional[int] = None
    parallel_requests: int = 4
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        if not self.models or not self.datasets or not self.seeds:
            raise ValueError("models, datasets, and seeds must be non-empty")
        if self.repetitions <= 0 or self.parallel_requests <= 0:
            raise ValueError("repetitions and parallel_requests must be positive")
        if self.shard_count <= 0 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must lie in [0, shard_count)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique; repetitions are the k axis")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
            for seed in self.seeds
        ):
            raise ValueError("seeds must be unsigned 64-bit integers")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be positive when supplied")
        if len(set(self.models)) != len(self.models):
            raise ValueError(
                "duplicate model configurations are not distinct field arms"
            )
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset_id values must be unique within a field")
        if any(
            model.sampling.seed + self.repetitions - 1 >= 2**63 for model in self.models
        ):
            raise ValueError(
                "sampling seed plus repetition exceeds signed 64-bit range"
            )

    @property
    def plan_sha256(self) -> str:
        return _digest(
            {
                "models": [asdict(model) for model in self.models],
                "datasets": [dataset.public_record() for dataset in self.datasets],
                "seeds": self.seeds,
                "repetitions": self.repetitions,
                "max_steps": self.max_steps,
                "parallel_requests": self.parallel_requests,
            }
        )


@dataclass(frozen=True)
class InferenceResult:
    decision: dict[str, Any]
    raw_response_sha256: str
    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_duration_ns: int


@dataclass(frozen=True)
class InferenceOutcome:
    result: Optional[InferenceResult] = None
    error_type: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None


class PromptRenderer:
    """Scaffold-neutral, stateless conversion from observation to model messages."""

    system_prompt = (
        "You are an evaluated trading policy. Use only the supplied point-in-time "
        "observation. Return one JSON object matching the provided Decision schema. "
        "Orders are target portfolio weights; an empty orders array is a hold. Never "
        "invent symbols, future data, fills, returns, or account state."
    )

    def messages(
        self,
        observation: dict[str, Any],
        *,
        mandate: Optional[dict[str, Any]] = None,
        cadence: int = 1,
    ) -> list[dict[str, str]]:
        payload: dict[str, Any] = {
            "observation": observation,
            "decision_cadence_bars": cadence,
        }
        if mandate is not None:
            payload["mandate"] = mandate
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            },
        ]


class DecisionModel(Protocol):
    """Inference surface used by the field scheduler and deterministic test doubles."""

    def identity(self, model: ModelRunConfig) -> ModelIdentity: ...

    def decide_many(
        self,
        observations: Sequence[dict[str, Any]],
        model: ModelRunConfig,
        renderer: PromptRenderer,
        *,
        max_workers: int,
        sampling_seeds: Optional[Sequence[int]] = None,
    ) -> list[InferenceOutcome]: ...


class OllamaClient:
    """Minimal local Ollama client with JSON-Schema-constrained decoding."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 120.0
    ):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Ollama URL must use http or https")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "local model harness refuses a non-loopback Ollama endpoint"
            )
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama URL must be a bare loopback origin")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._opener = build_opener(_NoRedirectHandler())
        self._schema = json.loads(decision_schema_json())
        self._identities: dict[str, ModelIdentity] = {}

    def _request(
        self, method: str, path: str, payload: Optional[dict[str, Any]] = None
    ) -> Any:
        body = None if payload is None else _canonical_bytes(payload)
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise LocalAgentError(
                f"Ollama {path} returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LocalAgentError(f"Ollama {path} failed: {error}") from error

    def identity(self, model: ModelRunConfig) -> ModelIdentity:
        cached = self._identities.get(model.model)
        if cached is not None:
            return cached
        tags = self._request("GET", "/api/tags")
        entry = next(
            (
                item
                for item in tags.get("models", [])
                if item.get("name") == model.model
            ),
            None,
        )
        if entry is None:
            raise LocalAgentError(f"model {model.model!r} is not installed in Ollama")
        shown = self._request("POST", "/api/show", {"model": model.model})
        version = self._request("GET", "/api/version").get("version", "unknown")
        running = self._request("GET", "/api/ps")
        loaded = next(
            (
                item
                for item in running.get("models", [])
                if item.get("name") == model.model or item.get("model") == model.model
            ),
            None,
        )
        details = entry.get("details", {}) | shown.get("details", {})
        model_info = shown.get("model_info", {})
        context = next(
            (
                int(value)
                for key, value in model_info.items()
                if key.endswith(".context_length") and isinstance(value, (int, float))
            ),
            None,
        )
        identity = ModelIdentity(
            model=model.model,
            digest=str(entry.get("digest", "unknown")),
            parameter_size=str(details.get("parameter_size", "unknown")),
            quantization=str(details.get("quantization_level", "unknown")),
            offload=(
                f"size_vram={int(loaded.get('size_vram', 0) or 0)}; "
                f"size={int(loaded.get('size', 0) or 0)}"
                if isinstance(loaded, dict)
                else "not loaded at identity capture; Ollama runtime split unresolved"
            ),
            family=str(details.get("family", "unknown")),
            context_length=context,
            server_version=str(version),
            size_bytes=(
                int(entry["size"])
                if isinstance(entry.get("size"), (int, float))
                else None
            ),
            format=str(details.get("format", "unknown")),
            capabilities=tuple(
                sorted(str(item) for item in shown.get("capabilities", []))
            ),
            license_sha256=(
                _digest(shown["license"]) if shown.get("license") is not None else None
            ),
            modelfile_sha256=(
                sha256(str(shown["modelfile"]).encode("utf-8")).hexdigest()
                if shown.get("modelfile") is not None
                else None
            ),
            template_sha256=(
                sha256(str(shown["template"]).encode("utf-8")).hexdigest()
                if shown.get("template") is not None
                else None
            ),
            parameters_sha256=(
                sha256(str(shown["parameters"]).encode("utf-8")).hexdigest()
                if shown.get("parameters") is not None
                else None
            ),
        )
        self._identities[model.model] = identity
        return identity

    def decide(
        self,
        observation: dict[str, Any],
        model: ModelRunConfig,
        renderer: PromptRenderer,
        *,
        sampling_seed: Optional[int] = None,
    ) -> InferenceResult:
        symbols = [str(item["symbol"]) for item in observation.get("symbols", [])]
        if not symbols:
            raise LocalAgentError("observation contains no symbols")
        payload = {
            "model": model.model,
            "messages": renderer.messages(observation, cadence=model.decision_cadence),
            "stream": False,
            "format": self._schema,
            "think": model.sampling.thinking,
            "options": {
                "temperature": model.sampling.temperature,
                "top_p": model.sampling.top_p,
                "seed": model.sampling.seed if sampling_seed is None else sampling_seed,
                "num_predict": model.sampling.max_tokens,
            },
        }
        response = self._request("POST", "/api/chat", payload)
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAgentError("Ollama response has no message.content")
        raw = message["content"]
        try:
            decision = parse_decision_payload(raw)
            decision_to_weights(decision, symbols)
        except DecisionParseError as error:
            raise LocalAgentError(
                f"model emitted an invalid Decision: {error}"
            ) from error
        prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
        output_tokens = int(response.get("eval_count", 0) or 0)
        thinking = message.get("thinking", "")
        reasoning_tokens = int(response.get("reasoning_count", 0) or 0)
        if reasoning_tokens == 0 and isinstance(thinking, str) and thinking:
            # Ollama does not consistently expose a thinking-token count. Keep the
            # estimate labeled by recording zero here rather than inventing a tokenizer.
            reasoning_tokens = 0
        decision["cost"] = {
            "cost_usd": 0.0,
            "tokens_in": prompt_tokens,
            "tokens_out": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        return InferenceResult(
            decision=decision,
            raw_response_sha256=sha256(raw.encode("utf-8")).hexdigest(),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_duration_ns=int(response.get("total_duration", 0) or 0),
        )

    def decide_many(
        self,
        observations: Sequence[dict[str, Any]],
        model: ModelRunConfig,
        renderer: PromptRenderer,
        *,
        max_workers: int,
        sampling_seeds: Optional[Sequence[int]] = None,
    ) -> list[InferenceOutcome]:
        if sampling_seeds is None:
            sampling_seeds = [model.sampling.seed] * len(observations)
        if len(sampling_seeds) != len(observations):
            raise ValueError("sampling_seeds must match observations")
        outcomes: list[Optional[InferenceOutcome]] = [None] * len(observations)
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(observations) or 1)
        ) as pool:
            futures = {
                pool.submit(
                    self.decide,
                    observation,
                    model,
                    renderer,
                    sampling_seed=int(sampling_seeds[index]),
                ): index
                for index, observation in enumerate(observations)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = InferenceOutcome(result=future.result())
                except Exception as error:  # noqa: BLE001 - cell failure is evidence
                    outcomes[index] = InferenceOutcome(
                        error_type=type(error).__name__, error=str(error)
                    )
        return [outcome for outcome in outcomes if outcome is not None]


class OpenAICompatibleClient:
    """Strict local client for llama.cpp, vLLM, SGLang, or compatible servers.

    OpenAI-compatible discovery does not expose quantization or CPU/GPU offload
    consistently. Those facts are therefore supplied in an explicit identity
    manifest and cross-checked against ``/v1/models`` before a cell can run.
    """

    def __init__(
        self,
        identities: Sequence[ModelIdentity],
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 120.0,
        supports_thinking: bool = False,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("local OpenAI-compatible URL must use http or https")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "local model harness refuses a non-loopback OpenAI-compatible endpoint"
            )
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/", "/v1", "/v1/"}
        ):
            raise ValueError(
                "OpenAI-compatible URL must be a bare loopback origin or /v1 root"
            )
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not identities:
            raise ValueError("at least one explicit model identity is required")
        self._identities: dict[str, ModelIdentity] = {}
        for identity in identities:
            if identity.model in self._identities:
                raise ValueError(f"duplicate identity for model {identity.model!r}")
            required = {
                "model": identity.model,
                "digest": identity.digest,
                "parameter_size": identity.parameter_size,
                "quantization": identity.quantization,
                "offload": identity.offload,
                "server": identity.server,
                "server_version": identity.server_version,
            }
            missing = [
                name
                for name, value in required.items()
                if not str(value).strip() or str(value).strip().lower() == "unknown"
            ]
            if missing:
                raise ValueError(
                    f"identity for {identity.model!r} lacks explicit provenance: {missing}"
                )
            self._identities[identity.model] = identity
        self.supports_thinking = bool(supports_thinking)
        self._opener = build_opener(_NoRedirectHandler())
        self._schema = json.loads(decision_schema_json())
        self._discovered_models: Optional[set[str]] = None

    def _request(
        self, method: str, path: str, payload: Optional[dict[str, Any]] = None
    ) -> Any:
        body = None if payload is None else _canonical_bytes(payload)
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise LocalAgentError(
                f"OpenAI-compatible {path} returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LocalAgentError(
                f"OpenAI-compatible {path} failed: {error}"
            ) from error

    def identity(self, model: ModelRunConfig) -> ModelIdentity:
        identity = self._identities.get(model.model)
        if identity is None:
            raise LocalAgentError(
                f"model {model.model!r} has no explicit identity manifest entry"
            )
        if self._discovered_models is None:
            response = self._request("GET", "/models")
            data = response.get("data")
            if not isinstance(data, list):
                raise LocalAgentError(
                    "OpenAI-compatible /models returned no data array"
                )
            self._discovered_models = {
                str(item["id"])
                for item in data
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
        if model.model not in self._discovered_models:
            raise LocalAgentError(
                f"model {model.model!r} is not served by the local backend"
            )
        return identity

    def decide(
        self,
        observation: dict[str, Any],
        model: ModelRunConfig,
        renderer: PromptRenderer,
        *,
        sampling_seed: Optional[int] = None,
    ) -> InferenceResult:
        symbols = [str(item["symbol"]) for item in observation.get("symbols", [])]
        if not symbols:
            raise LocalAgentError("observation contains no symbols")
        if model.sampling.thinking and not self.supports_thinking:
            raise LocalAgentError(
                "thinking was requested but this backend was not declared thinking-capable"
            )
        payload: dict[str, Any] = {
            "model": model.model,
            "messages": renderer.messages(observation, cadence=model.decision_cadence),
            "stream": False,
            "temperature": model.sampling.temperature,
            "top_p": model.sampling.top_p,
            "seed": model.sampling.seed if sampling_seed is None else sampling_seed,
            "max_tokens": model.sampling.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Decision",
                    "strict": True,
                    "schema": self._schema,
                },
            },
        }
        if model.sampling.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        started = time.perf_counter_ns()
        response = self._request("POST", "/chat/completions", payload)
        elapsed = time.perf_counter_ns() - started
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalAgentError("OpenAI-compatible response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LocalAgentError(
                "OpenAI-compatible response has no choices[0].message.content"
            )
        raw = message["content"]
        try:
            decision = parse_decision_payload(raw)
            decision_to_weights(decision, symbols)
        except DecisionParseError as error:
            raise LocalAgentError(
                f"model emitted an invalid Decision: {error}"
            ) from error
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("completion_tokens_details")
        reasoning_tokens = (
            int(details.get("reasoning_tokens", 0) or 0)
            if isinstance(details, dict)
            else 0
        )
        decision["cost"] = {
            "cost_usd": 0.0,
            "tokens_in": prompt_tokens,
            "tokens_out": output_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        return InferenceResult(
            decision=decision,
            raw_response_sha256=sha256(raw.encode("utf-8")).hexdigest(),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_duration_ns=elapsed,
        )

    def decide_many(
        self,
        observations: Sequence[dict[str, Any]],
        model: ModelRunConfig,
        renderer: PromptRenderer,
        *,
        max_workers: int,
        sampling_seeds: Optional[Sequence[int]] = None,
    ) -> list[InferenceOutcome]:
        if sampling_seeds is None:
            sampling_seeds = [model.sampling.seed] * len(observations)
        if len(sampling_seeds) != len(observations):
            raise ValueError("sampling_seeds must match observations")
        outcomes: list[Optional[InferenceOutcome]] = [None] * len(observations)
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(observations) or 1)
        ) as pool:
            futures = {
                pool.submit(
                    self.decide,
                    observation,
                    model,
                    renderer,
                    sampling_seed=int(sampling_seeds[index]),
                ): index
                for index, observation in enumerate(observations)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = InferenceOutcome(result=future.result())
                except Exception as error:  # noqa: BLE001 - cell failure is evidence
                    outcomes[index] = InferenceOutcome(
                        error_type=type(error).__name__, error=str(error)
                    )
        return [outcome for outcome in outcomes if outcome is not None]


@dataclass(frozen=True)
class FieldCell:
    model_index: int
    dataset_index: int
    seed: int
    repetition: int

    def ordinal(self, plan: FieldPlan) -> int:
        """Stable position in the unsharded Cartesian field."""

        seed_index = plan.seeds.index(self.seed)
        return (
            (self.model_index * len(plan.datasets) + self.dataset_index)
            * len(plan.seeds)
            + seed_index
        ) * plan.repetitions + self.repetition

    def key(self, plan: FieldPlan) -> str:
        model = plan.models[self.model_index]
        dataset = plan.datasets[self.dataset_index]
        return _digest(
            {
                "plan_sha256": plan.plan_sha256,
                "model_index": self.model_index,
                "dataset_index": self.dataset_index,
                "model": model.model,
                "dataset_id": dataset.dataset_id,
                "tier": dataset.tier,
                "seed": self.seed,
                "repetition": self.repetition,
            }
        )


class EvidenceJournal:
    """Append-only JSONL journal used for resume and sharded field runs."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        completed: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LocalAgentError(
                        f"invalid evidence JSONL at {self.path}:{line_number}: {error}"
                    ) from error
                if record.get("status") == "completed":
                    completed.add(str(record["cell_id"]))
        return completed

    def append(self, record: dict[str, Any]) -> None:
        line = _canonical_bytes(record) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


class LocalFieldRunner:
    """Run model × dataset/tier × seed × k cells through native batch stepping."""

    def __init__(
        self,
        model_client: DecisionModel,
        *,
        renderer: Optional[PromptRenderer] = None,
    ) -> None:
        self.model_client = model_client
        self.renderer = renderer or PromptRenderer()

    @staticmethod
    def cells(plan: FieldPlan) -> list[FieldCell]:
        cells = [
            FieldCell(model_index, dataset_index, seed, repetition)
            for model_index in range(len(plan.models))
            for dataset_index in range(len(plan.datasets))
            for seed in plan.seeds
            for repetition in range(plan.repetitions)
        ]
        return [
            cell
            for index, cell in enumerate(cells)
            if index % plan.shard_count == plan.shard_index
        ]

    @staticmethod
    def _build_env(dataset: DatasetSpec, seeds: Sequence[int]) -> VecTradingEnv:
        kwargs = {
            "window_start": dataset.window_start,
            "window_end": dataset.window_end,
            "fee_bps": dataset.fee_bps,
            "slippage_bps": dataset.slippage_bps,
            "impact_bps": dataset.impact_bps,
            "financing_bps": dataset.financing_bps,
            "max_participation": dataset.max_participation,
            "autoreset_mode": "disabled",
        }
        if dataset.csv_text is not None:
            return VecTradingEnv.from_csv(dataset.csv_text, seeds=list(seeds), **kwargs)
        return VecTradingEnv(
            seeds=list(seeds),
            n_symbols=dataset.n_symbols,
            n_days=dataset.n_days,
            distribution_mode=dataset.tier,
            **kwargs,
        )

    def run(self, plan: FieldPlan, journal: EvidenceJournal) -> dict[str, int]:
        completed = journal.completed_keys()
        counts = {"completed": 0, "failed": 0, "skipped": 0}
        cells = self.cells(plan)
        for model_index, model in enumerate(plan.models):
            identity = self.model_client.identity(model)
            for dataset_index, dataset in enumerate(plan.datasets):
                group = [
                    cell
                    for cell in cells
                    if cell.model_index == model_index
                    and cell.dataset_index == dataset_index
                ]
                pending = [cell for cell in group if cell.key(plan) not in completed]
                counts["skipped"] += len(group) - len(pending)
                if not pending:
                    continue
                group_counts = self._run_group(
                    plan, model, identity, dataset, pending, journal
                )
                counts["completed"] += group_counts["completed"]
                counts["failed"] += group_counts["failed"]
        return counts

    def _run_group(
        self,
        plan: FieldPlan,
        model: ModelRunConfig,
        identity: ModelIdentity,
        dataset: DatasetSpec,
        cells: Sequence[FieldCell],
        journal: EvidenceJournal,
    ) -> dict[str, int]:
        env = self._build_env(dataset, [cell.seed for cell in cells])
        observations = json.loads(env.reset_batch())["observations"]
        active = [True] * len(cells)
        failed: list[Optional[dict[str, str]]] = [None] * len(cells)
        returns: list[list[float]] = [[] for _ in cells]
        response_hashes: list[list[str]] = [[] for _ in cells]
        observation_hashes: list[list[str]] = [[] for _ in cells]
        decisions_history: list[list[dict[str, Any]]] = [[] for _ in cells]
        trace_events: list[list[dict[str, Any]]] = [[] for _ in cells]
        confidences: list[list[float]] = [[] for _ in cells]
        realized_outcomes: list[list[bool]] = [[] for _ in cells]
        tokens_in = [0] * len(cells)
        tokens_out = [0] * len(cells)
        reasoning_tokens = [0] * len(cells)
        inference_ns = [0] * len(cells)
        termination: list[Optional[str]] = [None] * len(cells)
        last_decisions: list[dict[str, Any]] = [
            {"orders": [], "reasoning": "initial hold before first model decision"}
            for _ in cells
        ]
        max_steps = plan.max_steps or 1_000_000
        steps = 0

        while any(active) and steps < max_steps:
            infer_indices = [
                index
                for index, is_active in enumerate(active)
                if is_active and steps % model.decision_cadence == 0
            ]
            if infer_indices:
                inference_outcomes = self.model_client.decide_many(
                    [observations[index] for index in infer_indices],
                    model,
                    self.renderer,
                    max_workers=plan.parallel_requests,
                    sampling_seeds=[
                        model.sampling.seed + cells[index].repetition
                        for index in infer_indices
                    ],
                )
                if len(inference_outcomes) != len(infer_indices):
                    raise LocalAgentError(
                        "model client returned the wrong batch length"
                    )
                for lane, outcome in zip(infer_indices, inference_outcomes):
                    if not outcome.ok:
                        failed[lane] = {
                            "type": outcome.error_type or "InferenceError",
                            "detail": outcome.error or "unknown inference failure",
                        }
                        active[lane] = False
                        continue
                    result = outcome.result
                    assert result is not None
                    last_decisions[lane] = result.decision
                    response_hashes[lane].append(result.raw_response_sha256)
                    tokens_in[lane] += result.prompt_tokens
                    tokens_out[lane] += result.output_tokens
                    reasoning_tokens[lane] += result.reasoning_tokens
                    inference_ns[lane] += result.total_duration_ns

            decisions = []
            infer_index_set = set(infer_indices)
            for index in range(len(cells)):
                if not active[index]:
                    decisions.append(
                        {"orders": [], "reasoning": "inactive failed batch lane"}
                    )
                    continue
                applied = json.loads(json.dumps(last_decisions[index]))
                if index not in infer_index_set:
                    # A target allocation may be held between model decisions, but the
                    # inference cost belongs to the call that produced it only.
                    applied.pop("cost", None)
                decisions.append(applied)
            for index, is_active in enumerate(active):
                if not is_active:
                    continue
                observation_hashes[index].append(_digest(observations[index]))
                applied = json.loads(json.dumps(decisions[index]))
                decisions_history[index].append(applied)
                orders = applied.get("orders", [])
                lane_confidences = []
                for order in orders:
                    if not isinstance(order, dict):
                        continue
                    confidence = float(order.get("confidence", 0.5))
                    lane_confidences.append(confidence)
                confidences[index].append(
                    sum(lane_confidences) / len(lane_confidences)
                    if lane_confidences
                    else 0.5
                )
            stepped = json.loads(env.step_batch(json.dumps(decisions)))
            for index in range(len(cells)):
                if not active[index]:
                    continue
                info = stepped["infos"][index]
                events = info.get("events", []) if isinstance(info, dict) else []
                if not isinstance(events, list):
                    failed[index] = {
                        "type": "InvalidProcessTrace",
                        "detail": f"step {steps} returned non-list info.events",
                    }
                    active[index] = False
                    continue
                # Preserve the exact ProcessEvent values emitted by the shared
                # native engine. Synthesizing a clean risk verdict here would let
                # the artifact lie to SharpeBench's process gate.
                trace_events[index].extend(events)
                reward = float(stepped["rewards"][index])
                if not math.isfinite(reward):
                    failed[index] = {
                        "type": "NonFiniteReward",
                        "detail": f"step {steps} returned {reward}",
                    }
                    active[index] = False
                    continue
                returns[index].append(reward)
                realized_outcomes[index].append(reward > 0.0)
                if stepped["terminated"][index] or stepped["truncated"][index]:
                    termination[index] = "environment"
                    active[index] = False
            observations = stepped["observations"]
            steps += 1

        for index, is_active in enumerate(active):
            if is_active:
                termination[index] = "runner_step_budget"
                active[index] = False

        counts = {"completed": 0, "failed": 0}
        for index, cell in enumerate(cells):
            base: dict[str, Any] = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_class": LOCAL_EVIDENCE_CLASS,
                "deterministic_environment": True,
                "deterministic_agent": False,
                "replay_guarantee": "environment-and-recorded-decisions-only",
                "plan_sha256": plan.plan_sha256,
                "cell_id": cell.key(plan),
                "cell_ordinal": cell.ordinal(plan),
                "model_index": cell.model_index,
                "dataset_index": cell.dataset_index,
                "seed_index": plan.seeds.index(cell.seed),
                "field_shape": {
                    "models": len(plan.models),
                    "datasets": len(plan.datasets),
                    "seeds": list(plan.seeds),
                    "repetitions": plan.repetitions,
                    "total_cells": (
                        len(plan.models)
                        * len(plan.datasets)
                        * len(plan.seeds)
                        * plan.repetitions
                    ),
                },
                "model": asdict(identity),
                "model_config": asdict(model),
                "dataset": dataset.public_record(),
                "seed": cell.seed,
                "repetition": cell.repetition,
                "inference_seed": model.sampling.seed + cell.repetition,
                "steps": len(returns[index]),
                "tokens_in": tokens_in[index],
                "tokens_out": tokens_out[index],
                "reasoning_tokens": reasoning_tokens[index],
                "inference_duration_ns": inference_ns[index],
                "response_sha256": response_hashes[index],
                "observation_sha256": observation_hashes[index],
                "decisions": decisions_history[index],
                "trace": {"events": trace_events[index]},
                "confidences": confidences[index],
                "outcomes": realized_outcomes[index],
                # The shared protocol defines run cost in USD. Local inference has
                # no provider invoice, so token counts remain separate provenance
                # and the monetary cost is explicitly unknown/zero rather than
                # being mislabeled as dollars.
                "cost": 0.0,
                "cost_unit": "usd-self-reported",
                "local_compute_cost_included": False,
                "scaffold": {
                    "name": model.scaffold,
                    "system_prompt_sha256": sha256(
                        self.renderer.system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "decision_schema_sha256": _digest(
                        json.loads(decision_schema_json())
                    ),
                },
                "recorded_at_unix_ns": time.time_ns(),
            }
            if failed[index] is not None:
                base.update({"status": "failed", "failure": failed[index]})
                counts["failed"] += 1
            elif len(returns[index]) < 2:
                base.update(
                    {
                        "status": "failed",
                        "failure": {
                            "type": "InsufficientReturns",
                            "detail": "fewer than two scored bars",
                        },
                    }
                )
                counts["failed"] += 1
            else:
                base.update(
                    {
                        "status": "completed",
                        "termination": termination[index] or "environment",
                        "returns": returns[index],
                        "returns_sha256": _digest(returns[index]),
                        "n_trials": model.precommitted_n_trials,
                        "n_trials_source": "precommitted-model-entry",
                        "score": json.loads(
                            score_run(
                                returns[index],
                                model.precommitted_n_trials,
                                dataset.periods_per_year,
                            )
                        ),
                    }
                )
                counts["completed"] += 1
            journal.append(base)
        return counts


__all__ = [
    "DatasetSpec",
    "DecisionModel",
    "EvidenceJournal",
    "FieldCell",
    "FieldPlan",
    "InferenceOutcome",
    "InferenceResult",
    "LocalAgentError",
    "LocalFieldRunner",
    "ModelIdentity",
    "ModelRunConfig",
    "OllamaClient",
    "PromptRenderer",
    "SamplingConfig",
]
