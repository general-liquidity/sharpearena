"""Phased, publicly auditable prospective forecast field.

The runner separates preparation, local-only model inference, public sealing,
and outcome resolution. A forecast commit can therefore be pushed before any
resolving candle exists. The market-data source is fixed to Binance's public
market-data host; model loading is fixed to local files only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .deferred import Outcome
from .forecast_contract import (
    BINARY_BRIER,
    PROBABILITY,
    ForecastContract,
    canonical_json,
)
from .forecast_evidence import (
    ForecastLedger,
    ForecastRunIdentity,
    InformationExposure,
    forecast_evidence_from_json,
    write_forecast_evidence,
)

FIELD_PLAN_SCHEMA = "sharpearena.prospective-forecast-field-plan.v1"
FORECAST_COMMIT_SCHEMA = "sharpearena.prospective-forecast-commit.v1"
RESOLUTION_SCHEMA = "sharpearena.prospective-forecast-resolution.v1"
MARKET_BASE_URL = "https://data-api.binance.vision"
MARKET_SOURCE_ID = "binance-spot-public-klines-v3"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_TARGET_OFFSETS_MINUTES = (1, 3, 5, 7, 9, 11)
DEFAULT_LOOKBACK_BARS = 12
_SCAFFOLD_MODULES = (
    "deferred.py",
    "forecast_contract.py",
    "forecast_evidence.py",
    "prospective_field.py",
)
_AGENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

JsonFetcher = Callable[[str, Mapping[str, object]], Any]
Clock = Callable[[], int]


class ProspectiveFieldError(RuntimeError):
    """A phase could not preserve the prospective field's frozen contract."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


@dataclass(frozen=True)
class LocalModelSpec:
    agent_id: str
    model_id: str
    revision: str
    snapshot_path: Path


def _scaffold_files() -> list[dict[str, str]]:
    root = Path(__file__).parent
    return [
        {
            "path": name,
            "sha256": _sha256_bytes((root / name).read_bytes().replace(b"\r\n", b"\n")),
        }
        for name in _SCAFFOLD_MODULES
    ]


def _scaffold_sha256() -> str:
    return _sha256_json(_scaffold_files())


def _validate_model_spec(spec: LocalModelSpec) -> None:
    if not _AGENT_ID.fullmatch(spec.agent_id):
        raise ProspectiveFieldError(
            "agent ID must use lowercase letters, digits, dots, underscores, or hyphens"
        )
    if not spec.model_id.strip():
        raise ProspectiveFieldError("model ID must be non-empty")
    if not _IMMUTABLE_REVISION.fullmatch(spec.revision):
        raise ProspectiveFieldError(
            "model revision must be a 40-character lowercase content revision"
        )
    if spec.snapshot_path.resolve().name != spec.revision:
        raise ProspectiveFieldError(
            f"snapshot directory for {spec.agent_id!r} does not match its revision"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    _atomic_text(
        path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _http_json(path: str, params: Mapping[str, object]) -> Any:
    if not path.startswith("/") or "?" in path:
        raise ProspectiveFieldError(
            "market-data path must be an absolute query-free path"
        )
    url = (
        f"{MARKET_BASE_URL}{path}?{urlencode(params)}"
        if params
        else f"{MARKET_BASE_URL}{path}"
    )
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "data-api.binance.vision":
        raise ProspectiveFieldError(
            "market-data request escaped the frozen HTTPS origin"
        )
    request = Request(url, headers={"User-Agent": "SharpeArena prospective field/1"})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=15) as response:
            if response.status != 200:
                raise ProspectiveFieldError(
                    f"market-data endpoint returned HTTP {response.status}"
                )
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProspectiveFieldError(f"market-data request failed: {error}") from error


def _server_time_ms(fetch: JsonFetcher = _http_json) -> int:
    value = fetch("/api/v3/time", {})
    if not isinstance(value, dict) or isinstance(value.get("serverTime"), bool):
        raise ProspectiveFieldError("market-data time response has the wrong shape")
    server_time = value.get("serverTime")
    if not isinstance(server_time, int) or server_time < 0:
        raise ProspectiveFieldError(
            "market-data serverTime must be a non-negative integer"
        )
    return server_time


def _closed_klines(
    symbol: str,
    *,
    server_time_ms: int,
    lookback: int,
    fetch: JsonFetcher = _http_json,
) -> list[dict[str, object]]:
    raw = fetch(
        "/api/v3/klines",
        {"symbol": symbol, "interval": "1m", "limit": lookback + 2},
    )
    if not isinstance(raw, list):
        raise ProspectiveFieldError(f"{symbol} kline response must be an array")
    closed = []
    for index, row in enumerate(raw):
        if not isinstance(row, list) or len(row) < 11:
            raise ProspectiveFieldError(f"{symbol} kline {index} has the wrong shape")
        open_time, close_time = row[0], row[6]
        if not isinstance(open_time, int) or not isinstance(close_time, int):
            raise ProspectiveFieldError(f"{symbol} kline {index} has invalid clocks")
        if close_time >= server_time_ms:
            continue
        values = [float(row[position]) for position in (1, 2, 3, 4, 5, 7)]
        if any(not math.isfinite(value) for value in values):
            raise ProspectiveFieldError(f"{symbol} kline {index} has non-finite values")
        closed.append(
            {
                "open_time_ms": open_time,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
                "close_time_ms": close_time,
                "quote_volume": row[7],
                "trade_count": row[8],
                "taker_buy_base_volume": row[9],
                "taker_buy_quote_volume": row[10],
            }
        )
    if len(closed) < lookback:
        raise ProspectiveFieldError(
            f"{symbol} returned only {len(closed)} closed bars; {lookback} required"
        )
    selected = closed[-lookback:]
    for prior, current in pairwise(selected):
        if current["open_time_ms"] != prior["open_time_ms"] + 60_000:
            raise ProspectiveFieldError(
                f"{symbol} lookback bars are not minute-contiguous"
            )
    return selected


def snapshot_digest(path: Path) -> tuple[str, int, int]:
    """Hash every local model file by relative path, length, and bytes."""

    root = path.resolve()
    if not root.is_dir():
        raise ProspectiveFieldError(f"model snapshot is not a directory: {path}")
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        file_count += 1
        byte_count += size
    if file_count == 0:
        raise ProspectiveFieldError(f"model snapshot contains no files: {path}")
    return digest.hexdigest(), file_count, byte_count


def parse_model_spec(value: str) -> LocalModelSpec:
    parts = value.split(",", 3)
    if len(parts) != 4 or any(not part.strip() for part in parts):
        raise ProspectiveFieldError(
            "--model must be AGENT_ID,MODEL_ID,IMMUTABLE_REVISION,SNAPSHOT_PATH"
        )
    agent_id, model_id, revision, path = (part.strip() for part in parts)
    spec = LocalModelSpec(agent_id, model_id, revision, Path(path))
    _validate_model_spec(spec)
    return spec


def prepare_field(
    output_dir: Path,
    models: Sequence[LocalModelSpec],
    *,
    deadline_delay_minutes: int = 30,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    target_offsets_minutes: Sequence[int] = DEFAULT_TARGET_OFFSETS_MINUTES,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    fetch: JsonFetcher = _http_json,
) -> dict[str, object]:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ProspectiveFieldError(
            "field directory must be absent or empty at preparation"
        )
    for model in models:
        _validate_model_spec(model)
    if len(models) < 2 or len({model.agent_id for model in models}) != len(models):
        raise ProspectiveFieldError("the field requires at least two unique agent IDs")
    identities = {(model.model_id, model.revision) for model in models}
    if len(identities) != len(models):
        raise ProspectiveFieldError("every agent must use a distinct model revision")
    if deadline_delay_minutes < 10:
        raise ProspectiveFieldError("deadline delay must be at least ten minutes")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ProspectiveFieldError("symbols must be non-empty and unique")
    if not target_offsets_minutes or tuple(
        sorted(set(target_offsets_minutes))
    ) != tuple(target_offsets_minutes):
        raise ProspectiveFieldError(
            "target offsets must be positive, unique, and sorted"
        )
    if any(offset <= 0 for offset in target_offsets_minutes):
        raise ProspectiveFieldError("target offsets must be positive")

    prepared_at_ms = _server_time_ms(fetch)
    prepared_at = prepared_at_ms // 1_000
    next_minute_ms = ((prepared_at_ms // 60_000) + 1) * 60_000
    deadline_ms = next_minute_ms + deadline_delay_minutes * 60_000
    deadline = deadline_ms // 1_000
    observation = {
        "schema_version": "sharpearena.prospective-market-snapshot.v1",
        "source": {
            "id": MARKET_SOURCE_ID,
            "base_url": MARKET_BASE_URL,
            "endpoint": "/api/v3/klines",
            "interval": "1m",
            "timezone": "UTC",
        },
        "captured_server_time_ms": prepared_at_ms,
        "lookback_bars": lookback_bars,
        "symbols": {
            symbol: _closed_klines(
                symbol,
                server_time_ms=prepared_at_ms,
                lookback=lookback_bars,
                fetch=fetch,
            )
            for symbol in symbols
        },
    }
    observation_digest = _sha256_json(observation)
    _write_json(output_dir / "observation.json", observation)

    contracts = []
    for offset in target_offsets_minutes:
        target_open_ms = deadline_ms + offset * 60_000
        target_close_ms = target_open_ms + 60_000 - 1
        for symbol in symbols:
            contract = ForecastContract(
                contract_id=f"{target_open_ms:x}-{symbol.lower()}",
                question=(
                    f"Will {symbol} close above its open in the Binance Spot one-minute "
                    f"candle opening at {target_open_ms} ms UTC?"
                ),
                instrument=symbol,
                target="one_minute_close_above_open",
                kind=PROBABILITY,
                opens_at=prepared_at,
                deadline=deadline,
                resolves_at=(target_close_ms + 1) // 1_000,
                observation_source=f"{MARKET_SOURCE_ID}:{symbol}:1m",
                open_definition=f"/api/v3/klines open at openTime={target_open_ms}",
                close_definition=f"/api/v3/klines close at openTime={target_open_ms}",
                unit="binary",
                scoring_rule=BINARY_BRIER,
                boundary_ownership="close equal to open resolves false",
                missing_data_policy="cancel if the exact candle is absent",
                fallback_policy="no alternate venue or interval",
            )
            raw = contract.to_dict()
            raw["sha256"] = contract.sha256
            raw["target_open_ms"] = target_open_ms
            contracts.append(raw)

    model_records = []
    for model in models:
        digest, file_count, byte_count = snapshot_digest(model.snapshot_path)
        model_records.append(
            {
                "agent_id": model.agent_id,
                "model_id": model.model_id,
                "revision": model.revision,
                "snapshot_sha256": digest,
                "snapshot_file_count": file_count,
                "snapshot_bytes": byte_count,
            }
        )
    plan = {
        "schema_version": FIELD_PLAN_SCHEMA,
        "prepared_at_server_ms": prepared_at_ms,
        "submission_deadline_server_ms": deadline_ms,
        "observation_sha256": observation_digest,
        "models": model_records,
        "contracts": contracts,
        "inference": {
            "scaffold_id": "local-transformers-binary-logit-v1",
            "scaffold_files": _scaffold_files(),
            "scaffold_sha256": _scaffold_sha256(),
            "local_files_only": True,
            "method": "normalize next-token logits for token 0 versus token 1",
            "class_tokens": {"false": "0", "true": "1"},
            "probability_clip": [0.01, 0.99],
            "quantization": "bitsandbytes-nf4-4bit",
            "one_prompt_per_contract": True,
            "inference_batch_size": 1,
            "logits_to_keep": 1,
        },
        "analysis": {
            "primary": "descriptive Brier score and calibration on all resolved contracts",
            "pairwise": "exploratory exact-common-support block bootstrap",
            "dependence_unit": "whole one-minute resolution clock across symbols",
            "bootstrap_samples": 2_000,
            "bootstrap_seed": 2_609_04,
            "confidence": 0.95,
            "familywise_alpha": 0.05,
            "calibration_bins": 5,
            "minimum_comparative_claim_blocks": 30,
            "stopping_rule": "resolve every frozen contract; do not stop on scores",
        },
        "claims": {
            "confirmatory": [],
            "descriptive": [
                "forecast completion, Brier loss, and fixed-bin calibration by agent"
            ],
            "not_supported": [
                "model superiority",
                "trading profitability",
                "generalization beyond this UTC field",
            ],
        },
    }
    _write_json(output_dir / "field-plan.json", plan)
    _atomic_text(output_dir / "field-plan.sha256", _sha256_json(plan) + "\n")
    return plan


def _contract_from_plan(raw: Mapping[str, object]) -> ForecastContract:
    fields = {
        key: value
        for key, value in raw.items()
        if key not in {"sha256", "target_open_ms"}
    }
    contract = ForecastContract.from_dict(fields)
    if raw.get("sha256") != contract.sha256:
        raise ProspectiveFieldError(
            f"contract {contract.contract_id!r} digest does not match"
        )
    return contract


def _validated_plan(field_dir: Path) -> dict[str, object]:
    plan = _read_json(field_dir / "field-plan.json")
    if not isinstance(plan, dict) or plan.get("schema_version") != FIELD_PLAN_SCHEMA:
        raise ProspectiveFieldError("field plan has an unsupported schema")
    expected = (field_dir / "field-plan.sha256").read_text(encoding="utf-8").strip()
    if expected != _sha256_json(plan):
        raise ProspectiveFieldError("field plan digest does not match its frozen bytes")
    observation = _read_json(field_dir / "observation.json")
    if plan.get("observation_sha256") != _sha256_json(observation):
        raise ProspectiveFieldError("observation digest does not match the field plan")
    contracts = plan.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ProspectiveFieldError("field plan must contain contracts")
    for raw in contracts:
        if not isinstance(raw, dict):
            raise ProspectiveFieldError("field plan contract must be an object")
        _contract_from_plan(raw)
    return plan


def render_prompts(
    plan: Mapping[str, object], observation: Mapping[str, object]
) -> dict[str, str]:
    compact_market = {}
    for symbol, bars in observation["symbols"].items():
        compact_market[symbol] = [
            [
                bar["open_time_ms"],
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
                bar["volume"],
            ]
            for bar in bars
        ]
    common = {
        "bar_fields": ["open_time_ms", "open", "high", "low", "close", "volume"],
        "market_history": compact_market,
    }
    prompts = {}
    for raw in plan["contracts"]:
        payload = {
            **common,
            "question": {
                "contract_id": raw["contract_id"],
                "instrument": raw["instrument"],
                "target_open_ms": raw["target_open_ms"],
                "event": "one-minute close strictly above one-minute open",
            },
        }
        prompts[raw["contract_id"]] = (
            "Make a prospective binary forecast from this frozen market snapshot. "
            "You have no tools and must not invent newer observations. The next answer "
            "token must be 1 if the event is more likely true and 0 if it is more likely "
            "false. Do not explain.\n" + canonical_json(payload)
        )
    return prompts


def parse_prediction_map(
    raw_output: str, contract_ids: Sequence[str]
) -> dict[str, float]:
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"[\[{]", raw_output):
        try:
            value, _ = decoder.raw_decode(raw_output[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    expected = set(contract_ids)
    complete: list[dict[str, float]] = []
    for candidate in candidates:
        forecasts = candidate.get("forecasts")
        if set(candidate) != {"forecasts"} or not isinstance(forecasts, dict):
            continue
        if set(forecasts) != expected:
            continue
        parsed = {}
        valid = True
        for contract_id, value in forecasts.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                valid = False
                break
            probability = float(value)
            if not math.isfinite(probability) or not 0.01 <= probability <= 0.99:
                valid = False
                break
            parsed[contract_id] = probability
        if valid:
            complete.append(parsed)
    if len(complete) == 1:
        return complete[0]
    raise ProspectiveFieldError(
        "model output did not contain exactly one complete forecasts object"
    )


def _run_local_transformer(
    snapshot_path: Path, prompts: Mapping[str, str]
) -> tuple[str, dict[str, object]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise ProspectiveFieldError(
            "local inference requires torch, transformers, accelerate, and bitsandbytes"
        ) from error
    if not torch.cuda.is_available():
        raise ProspectiveFieldError(
            "the preregistered four-bit field requires a CUDA GPU"
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
        device_map="auto",
        quantization_config=quantization,
        dtype=torch.float16,
    )
    false_ids = tokenizer.encode("0", add_special_tokens=False)
    true_ids = tokenizer.encode("1", add_special_tokens=False)
    if len(false_ids) != 1 or len(true_ids) != 1 or false_ids == true_ids:
        raise ProspectiveFieldError(
            "binary-logit inference requires distinct single-token 0 and 1 labels"
        )
    torch.manual_seed(0)
    started = time.perf_counter_ns()
    probabilities = {}
    logit_records = {}
    input_token_counts = []
    for contract_id, prompt in prompts.items():
        messages = [
            {"role": "system", "content": "Answer with exactly one token: 0 or 1."},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        input_token_counts.append(int(inputs["input_ids"].shape[1]))
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        final_logits = output.logits[0, -1, [false_ids[0], true_ids[0]]].float()
        if not bool(torch.isfinite(final_logits).all()):
            raise ProspectiveFieldError(
                f"model returned non-finite binary logits for {contract_id}"
            )
        normalized = torch.softmax(final_logits, dim=0)
        unclipped = float(normalized[1].item())
        probability = min(0.99, max(0.01, unclipped))
        probabilities[contract_id] = probability
        logit_records[contract_id] = {
            "false_logit": float(final_logits[0].item()),
            "true_logit": float(final_logits[1].item()),
            "unclipped_probability": unclipped,
            "probability": probability,
        }
        del inputs, output, final_logits, normalized
    elapsed_ms = (time.perf_counter_ns() - started) // 1_000_000
    raw = canonical_json({"forecasts": probabilities})
    runtime = {
        "method": "binary_next_token_logit",
        "class_token_ids": {"false": false_ids[0], "true": true_ids[0]},
        "logits": logit_records,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_ms": elapsed_ms,
        "input_tokens_min": min(input_token_counts),
        "input_tokens_max": max(input_token_counts),
        "contract_count": len(prompts),
    }
    del model, tokenizer
    torch.cuda.empty_cache()
    return raw, runtime


def forecast_agent(
    field_dir: Path,
    spec: LocalModelSpec,
    *,
    infer: Callable[
        [Path, Mapping[str, str]], tuple[str, dict[str, object]]
    ] = _run_local_transformer,
    fetch: JsonFetcher = _http_json,
) -> Path:
    _validate_model_spec(spec)
    plan = _validated_plan(field_dir)
    matching = [
        record for record in plan["models"] if record["agent_id"] == spec.agent_id
    ]
    if len(matching) != 1:
        raise ProspectiveFieldError(
            f"agent {spec.agent_id!r} is not uniquely preregistered"
        )
    record = matching[0]
    if record["model_id"] != spec.model_id or record["revision"] != spec.revision:
        raise ProspectiveFieldError(
            "runtime model identity differs from the preregistration"
        )
    if plan["inference"].get("scaffold_sha256") != _scaffold_sha256():
        raise ProspectiveFieldError(
            "forecast runner bytes differ from the preregistered scaffold"
        )
    if plan["inference"].get("scaffold_files") != _scaffold_files():
        raise ProspectiveFieldError(
            "forecast runner sources differ from the preregistered scaffold"
        )
    digest, file_count, byte_count = snapshot_digest(spec.snapshot_path)
    if (digest, file_count, byte_count) != (
        record["snapshot_sha256"],
        record["snapshot_file_count"],
        record["snapshot_bytes"],
    ):
        raise ProspectiveFieldError(
            "runtime model snapshot differs from the preregistration"
        )
    observation = _read_json(field_dir / "observation.json")
    prompts = render_prompts(plan, observation)
    contracts = [_contract_from_plan(raw) for raw in plan["contracts"]]
    contract_ids = [item.contract_id for item in contracts]
    evidence_path = field_dir / "pending" / f"{spec.agent_id}.json"
    audit_path = field_dir / "inference" / f"{spec.agent_id}.json"
    if evidence_path.exists() or audit_path.exists():
        raise ProspectiveFieldError(
            f"agent {spec.agent_id!r} already has forecast evidence; it is append-only"
        )
    raw_output, runtime = infer(spec.snapshot_path, prompts)
    submitted_at_ms = _server_time_ms(fetch)
    if submitted_at_ms > plan["submission_deadline_server_ms"]:
        raise ProspectiveFieldError(
            "model completed after the frozen submission deadline"
        )
    predictions = parse_prediction_map(raw_output, contract_ids)
    inference_config = dict(plan["inference"])
    identity = ForecastRunIdentity(
        agent_id=spec.agent_id,
        model_id=spec.model_id,
        model_sha256=digest,
        scaffold_id=plan["inference"]["scaffold_id"],
        scaffold_sha256=plan["inference"]["scaffold_sha256"],
        prompt_sha256=_sha256_json(prompts),
        operator_id="general-liquidity-prospective-field",
        config_sha256=_sha256_json(inference_config),
    )
    ledger = ForecastLedger(identity)
    submitted_at = submitted_at_ms // 1_000
    for contract in contracts:
        probability = predictions[contract.contract_id]
        ledger.submit(
            claim_id=contract.contract_id,
            contract=contract,
            prediction=probability,
            confidence=max(probability, 1.0 - probability),
            rationale="normalized binary next-token logits from the preregistered local model",
            submitted_at=submitted_at,
            idempotency_key=f"{spec.agent_id}:{contract.contract_id}:initial",
            exposure=InformationExposure(
                observed_at=submitted_at,
                market_snapshot_sha256=plan["observation_sha256"],
                consensus_visible=False,
                source_ids=(MARKET_SOURCE_ID,),
            ),
        )
    write_forecast_evidence(
        evidence_path, ledger.evidence([], generated_at=submitted_at)
    )
    audit = {
        "schema_version": "sharpearena.prospective-model-inference.v1",
        "agent_id": spec.agent_id,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "model_snapshot_sha256": digest,
        "field_plan_sha256": _sha256_json(plan),
        "prompt_sha256": _sha256_json(prompts),
        "raw_output_kind": "canonical forecast map derived from recorded binary logits",
        "raw_output": raw_output,
        "raw_output_sha256": _sha256_bytes(raw_output.encode("utf-8")),
        "parsed_predictions": predictions,
        "submitted_at_server_ms": submitted_at_ms,
        "runtime": runtime,
    }
    _write_json(audit_path, audit)
    return evidence_path


def _validate_logit_runtime(
    runtime: object, parsed_predictions: Mapping[str, float]
) -> None:
    if not isinstance(runtime, dict):
        raise ProspectiveFieldError("inference runtime must be an object")
    token_ids = runtime.get("class_token_ids")
    logits = runtime.get("logits")
    if (
        runtime.get("method") != "binary_next_token_logit"
        or runtime.get("contract_count") != len(parsed_predictions)
        or not isinstance(token_ids, dict)
        or set(token_ids) != {"false", "true"}
        or isinstance(token_ids["false"], bool)
        or isinstance(token_ids["true"], bool)
        or not isinstance(token_ids["false"], int)
        or not isinstance(token_ids["true"], int)
        or token_ids["false"] == token_ids["true"]
        or not isinstance(logits, dict)
        or set(logits) != set(parsed_predictions)
    ):
        raise ProspectiveFieldError("binary-logit runtime has the wrong shape")
    for contract_id, probability in parsed_predictions.items():
        record = logits[contract_id]
        if not isinstance(record, dict) or set(record) != {
            "false_logit",
            "true_logit",
            "unclipped_probability",
            "probability",
        }:
            raise ProspectiveFieldError(
                f"binary-logit record has the wrong shape: {contract_id}"
            )
        values = tuple(record.values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ProspectiveFieldError(
                f"binary-logit record is non-finite: {contract_id}"
            )
        delta = float(record["true_logit"]) - float(record["false_logit"])
        if delta >= 0.0:
            expected_unclipped = 1.0 / (1.0 + math.exp(-delta))
        else:
            exp_delta = math.exp(delta)
            expected_unclipped = exp_delta / (1.0 + exp_delta)
        expected_probability = min(0.99, max(0.01, expected_unclipped))
        if (
            not math.isclose(
                float(record["unclipped_probability"]),
                expected_unclipped,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(record["probability"]),
                expected_probability,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                probability,
                expected_probability,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            raise ProspectiveFieldError(
                f"binary-logit probability does not reproduce: {contract_id}"
            )


def _validate_agent_artifacts(
    field_dir: Path, plan: Mapping[str, object], agent_id: str
) -> None:
    evidence_path = field_dir / "pending" / f"{agent_id}.json"
    audit_path = field_dir / "inference" / f"{agent_id}.json"
    document = forecast_evidence_from_json(evidence_path.read_text(encoding="utf-8"))
    audit = _read_json(audit_path)
    records = [record for record in plan["models"] if record["agent_id"] == agent_id]
    if len(records) != 1:
        raise ProspectiveFieldError(f"agent {agent_id!r} is not uniquely preregistered")
    record = records[0]
    identity = document["identity"]
    inference = plan["inference"]
    observation = _read_json(field_dir / "observation.json")
    prompt_sha256 = _sha256_json(render_prompts(plan, observation))
    expected_identity = {
        "agent_id": agent_id,
        "model_id": record["model_id"],
        "model_sha256": record["snapshot_sha256"],
        "scaffold_id": inference["scaffold_id"],
        "scaffold_sha256": inference["scaffold_sha256"],
        "prompt_sha256": prompt_sha256,
        "operator_id": "general-liquidity-prospective-field",
        "config_sha256": _sha256_json(inference),
    }
    if identity != expected_identity:
        raise ProspectiveFieldError(
            f"{agent_id} pending identity differs from the plan"
        )
    expected_contract_ids = [raw["contract_id"] for raw in plan["contracts"]]
    revisions = document["revisions"]
    if len(revisions) != len(expected_contract_ids):
        raise ProspectiveFieldError(
            f"{agent_id} does not contain one revision per frozen contract"
        )
    revision_predictions: dict[str, float] = {}
    if not isinstance(audit, dict):
        raise ProspectiveFieldError(f"{agent_id} inference audit must be an object")
    submitted_at_ms = audit.get("submitted_at_server_ms")
    if isinstance(submitted_at_ms, bool) or not isinstance(submitted_at_ms, int):
        raise ProspectiveFieldError(f"{agent_id} inference clock is invalid")
    if submitted_at_ms > plan["submission_deadline_server_ms"]:
        raise ProspectiveFieldError(f"{agent_id} inference missed the frozen deadline")
    for revision in revisions:
        prediction = revision["prediction"]
        if (
            revision["status"] != "eligible"
            or revision["ordinal"] != 0
            or revision["submitted_at"] != submitted_at_ms // 1_000
            or not isinstance(prediction, list)
            or len(prediction) != 1
        ):
            raise ProspectiveFieldError(
                f"{agent_id} contains a non-initial or ineligible forecast revision"
            )
        revision_predictions[revision["claim_id"]] = prediction[0]
    if list(revision_predictions) != expected_contract_ids:
        raise ProspectiveFieldError(
            f"{agent_id} revision support differs from the frozen contracts"
        )
    if any(record["status"] != "pending" for record in document["resolutions"]):
        raise ProspectiveFieldError(f"{agent_id} is not a pending forecast document")
    raw_output = audit.get("raw_output")
    if not isinstance(raw_output, str) or audit.get(
        "raw_output_sha256"
    ) != _sha256_bytes(raw_output.encode("utf-8")):
        raise ProspectiveFieldError(f"{agent_id} raw inference digest does not match")
    parsed_predictions = parse_prediction_map(raw_output, expected_contract_ids)
    _validate_logit_runtime(audit.get("runtime"), parsed_predictions)
    if (
        audit.get("schema_version") != "sharpearena.prospective-model-inference.v1"
        or audit.get("agent_id") != agent_id
        or audit.get("model_id") != record["model_id"]
        or audit.get("model_revision") != record["revision"]
        or audit.get("model_snapshot_sha256") != record["snapshot_sha256"]
        or audit.get("field_plan_sha256") != _sha256_json(plan)
        or audit.get("prompt_sha256") != prompt_sha256
        or audit.get("raw_output_kind")
        != "canonical forecast map derived from recorded binary logits"
        or audit.get("parsed_predictions") != parsed_predictions
        or revision_predictions != parsed_predictions
    ):
        raise ProspectiveFieldError(
            f"{agent_id} inference audit differs from its ledger"
        )


def seal_forecasts(
    field_dir: Path, *, fetch: JsonFetcher = _http_json
) -> dict[str, object]:
    if (field_dir / "forecast-commit.json").exists():
        raise ProspectiveFieldError(
            "forecast commit already exists and cannot be replaced"
        )
    plan = _validated_plan(field_dir)
    pending = sorted((field_dir / "pending").glob("*.json"))
    inference = sorted((field_dir / "inference").glob("*.json"))
    expected_agents = sorted(record["agent_id"] for record in plan["models"])
    if [path.stem for path in pending] != expected_agents:
        raise ProspectiveFieldError(
            "pending evidence does not cover every preregistered agent"
        )
    if [path.stem for path in inference] != expected_agents:
        raise ProspectiveFieldError(
            "inference audits do not cover every preregistered agent"
        )
    for agent_id in expected_agents:
        _validate_agent_artifacts(field_dir, plan, agent_id)
    sealed_at_ms = _server_time_ms(fetch)
    if sealed_at_ms > plan["submission_deadline_server_ms"]:
        raise ProspectiveFieldError(
            "forecast field was not sealed by its frozen deadline"
        )
    files = [
        field_dir / "field-plan.json",
        field_dir / "field-plan.sha256",
        field_dir / "observation.json",
    ]
    files.extend(pending)
    files.extend(inference)
    commit = {
        "schema_version": FORECAST_COMMIT_SCHEMA,
        "sealed_at_server_ms": sealed_at_ms,
        "submission_deadline_server_ms": plan["submission_deadline_server_ms"],
        "field_plan_sha256": _sha256_json(plan),
        "files": {
            path.relative_to(field_dir).as_posix(): _sha256_bytes(path.read_bytes())
            for path in files
        },
    }
    _write_json(field_dir / "forecast-commit.json", commit)
    return commit


def _target_candle(
    symbol: str,
    open_time_ms: int,
    *,
    fetch: JsonFetcher = _http_json,
) -> dict[str, object]:
    raw = fetch(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": open_time_ms,
            "endTime": open_time_ms + 59_999,
            "limit": 1,
        },
    )
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list):
        raise ProspectiveFieldError(f"exact resolving candle is absent for {symbol}")
    row = raw[0]
    if len(row) < 11 or row[0] != open_time_ms or row[6] != open_time_ms + 59_999:
        raise ProspectiveFieldError(f"resolving candle boundary differs for {symbol}")
    open_value, close_value = float(row[1]), float(row[4])
    if not math.isfinite(open_value) or not math.isfinite(close_value):
        raise ProspectiveFieldError(f"resolving candle is non-finite for {symbol}")
    return {
        "symbol": symbol,
        "open_time_ms": row[0],
        "open": row[1],
        "close": row[4],
        "close_time_ms": row[6],
        "outcome": 1.0 if close_value > open_value else 0.0,
        "raw_sha256": _sha256_json(row),
    }


def _ledger_from_pending(document: Mapping[str, object]) -> ForecastLedger:
    identity = ForecastRunIdentity(**document["identity"])
    contracts = {
        raw["contract_id"]: ForecastContract.from_dict(raw)
        for raw in document["contracts"]
    }
    ledger = ForecastLedger(identity)
    contract_by_digest = {contract.sha256: contract for contract in contracts.values()}
    for raw in document["revisions"]:
        exposure = InformationExposure(
            observed_at=raw["exposure"]["observed_at"],
            market_snapshot_sha256=raw["exposure"]["market_snapshot_sha256"],
            consensus_visible=raw["exposure"]["consensus_visible"],
            consensus_snapshot_sha256=raw["exposure"]["consensus_snapshot_sha256"],
            source_ids=tuple(raw["exposure"]["source_ids"]),
        )
        ledger.submit(
            claim_id=raw["claim_id"],
            contract=contract_by_digest[raw["contract_sha256"]],
            prediction=raw["prediction"],
            confidence=raw["confidence"],
            rationale=raw["rationale"],
            submitted_at=raw["submitted_at"],
            idempotency_key=raw["idempotency_key"],
            expected_revision=(None if raw["ordinal"] == 0 else raw["ordinal"] - 1),
            trigger_event_id=raw["trigger_event_id"],
            revision_reason=raw["revision_reason"],
            exposure=exposure,
        )
    if [item.to_dict() for item in ledger.revisions()] != document["revisions"]:
        raise ProspectiveFieldError("pending revisions changed during reconstruction")
    return ledger


def _validated_forecast_commit(
    field_dir: Path, plan: Mapping[str, object]
) -> dict[str, object]:
    commit = _read_json(field_dir / "forecast-commit.json")
    expected_fields = {
        "schema_version",
        "sealed_at_server_ms",
        "submission_deadline_server_ms",
        "field_plan_sha256",
        "files",
    }
    if not isinstance(commit, dict) or set(commit) != expected_fields:
        raise ProspectiveFieldError("forecast commit has the wrong shape")
    if commit["schema_version"] != FORECAST_COMMIT_SCHEMA:
        raise ProspectiveFieldError("forecast commit has an unsupported schema")
    if (
        commit["submission_deadline_server_ms"] != plan["submission_deadline_server_ms"]
        or commit["field_plan_sha256"] != _sha256_json(plan)
        or isinstance(commit["sealed_at_server_ms"], bool)
        or not isinstance(commit["sealed_at_server_ms"], int)
        or commit["sealed_at_server_ms"] > plan["submission_deadline_server_ms"]
    ):
        raise ProspectiveFieldError("forecast commit differs from the frozen plan")
    agent_ids = sorted(record["agent_id"] for record in plan["models"])
    expected_files = {
        "field-plan.json",
        "field-plan.sha256",
        "observation.json",
        *(f"pending/{agent_id}.json" for agent_id in agent_ids),
        *(f"inference/{agent_id}.json" for agent_id in agent_ids),
    }
    files = commit["files"]
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ProspectiveFieldError("forecast commit file set differs from the plan")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ProspectiveFieldError(
                f"forecast commit digest is invalid: {relative}"
            )
        path = field_dir / relative
        if not path.is_file() or _sha256_bytes(path.read_bytes()) != expected:
            raise ProspectiveFieldError(f"forecast commit file changed: {relative}")
    return commit


def resolve_field(
    field_dir: Path, *, fetch: JsonFetcher = _http_json
) -> dict[str, object]:
    if (field_dir / "resolution-manifest.json").exists():
        raise ProspectiveFieldError("resolution already exists and cannot be replaced")
    plan = _validated_plan(field_dir)
    _validated_forecast_commit(field_dir, plan)
    now_ms = _server_time_ms(fetch)
    last_resolution_ms = max(
        raw["target_open_ms"] + 60_000 for raw in plan["contracts"]
    )
    if now_ms <= last_resolution_ms:
        raise ProspectiveFieldError(
            f"resolution is early: serverTime={now_ms}, required>{last_resolution_ms}"
        )
    candles = []
    outcomes = {}
    for raw in plan["contracts"]:
        candle = _target_candle(raw["instrument"], raw["target_open_ms"], fetch=fetch)
        candles.append(candle)
        outcomes[raw["contract_id"]] = candle["outcome"]
    resolution_snapshot = {
        "schema_version": RESOLUTION_SCHEMA,
        "resolved_at_server_ms": now_ms,
        "field_plan_sha256": _sha256_json(plan),
        "forecast_commit_sha256": _sha256_bytes(
            (field_dir / "forecast-commit.json").read_bytes()
        ),
        "candles": candles,
    }
    _write_json(field_dir / "resolution.json", resolution_snapshot)
    final_paths = []
    for pending_path in sorted((field_dir / "pending").glob("*.json")):
        document = forecast_evidence_from_json(pending_path.read_text(encoding="utf-8"))
        ledger = _ledger_from_pending(document)
        resolved = [
            Outcome(
                claim_id=contract_id,
                value=outcome,
                available_at=(
                    next(
                        raw["target_open_ms"]
                        for raw in plan["contracts"]
                        if raw["contract_id"] == contract_id
                    )
                    + 60_000
                )
                // 1_000,
            )
            for contract_id, outcome in outcomes.items()
        ]
        final_path = field_dir / "resolved" / pending_path.name
        write_forecast_evidence(
            final_path, ledger.evidence(resolved, generated_at=now_ms // 1_000)
        )
        final_paths.append(final_path)
    resolution_manifest = {
        "schema_version": "sharpearena.prospective-forecast-resolution-manifest.v1",
        "files": {
            path.relative_to(field_dir).as_posix(): _sha256_bytes(path.read_bytes())
            for path in [field_dir / "resolution.json", *final_paths]
        },
    }
    _write_json(field_dir / "resolution-manifest.json", resolution_manifest)
    return resolution_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--field-dir", type=Path, required=True)
    prepare.add_argument("--model", action="append", required=True)
    prepare.add_argument("--deadline-delay-minutes", type=int, default=30)
    forecast = subparsers.add_parser("forecast")
    forecast.add_argument("--field-dir", type=Path, required=True)
    forecast.add_argument("--model", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--field-dir", type=Path, required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--field-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_field(
            args.field_dir,
            [parse_model_spec(value) for value in args.model],
            deadline_delay_minutes=args.deadline_delay_minutes,
        )
    elif args.command == "forecast":
        forecast_agent(args.field_dir, parse_model_spec(args.model))
    elif args.command == "seal":
        seal_forecasts(args.field_dir)
    elif args.command == "resolve":
        resolve_field(args.field_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
