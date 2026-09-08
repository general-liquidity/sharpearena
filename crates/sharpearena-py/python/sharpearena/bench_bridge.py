"""Compile completed SharpeArena field journals into SharpeBench submissions.

The bridge is deliberately an artifact boundary rather than a recursive package
dependency. SharpeArena owns observations, decisions, execution, and raw evidence;
SharpeBench owns field-level statistical scoring. One output file is produced per
dataset because ``periods_per_year`` is a property of the dataset-level score config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Sequence

from .local_agents import (
    DURATION_SOURCES,
    DURATION_UNIT_NS,
    EVIDENCE_SCHEMA_VERSION,
    HOST_DURATION_SOURCE,
    LOCAL_EVIDENCE_CLASS,
)

BRIDGE_SCHEMA_VERSION = 3
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class BenchBridgeError(ValueError):
    """Raw field evidence is incomplete, conflicting, or not scoreable."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _nearest_rank(values: Sequence[int], percentile: int) -> int:
    """Return a deterministic nearest-rank percentile over integer samples."""

    if not values:
        raise BenchBridgeError("an operational percentile requires at least one sample")
    if not 0 < percentile <= 100:
        raise BenchBridgeError("percentile must lie in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered) / 100)
    return ordered[max(0, rank - 1)]


def _count(record: dict[str, Any], field: str) -> int:
    """Read one required nonnegative count, refusing an absent or coerced value."""

    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchBridgeError(
            f"attempt {record.get('cell_id')!r} has no usable {field}"
        )
    return value


def _attempt_ledger(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Append-only accounting over every recorded attempt, not the retained one.

    A cell that failed and was resumed spent time and tokens twice. Scoring keeps
    only the terminal completion, so aggregating operational cost from retained
    records alone deletes the cost of failure and makes an error-prone backend
    look cheaper and faster than it was.
    """

    successful: list[int] = []
    failed_observations: list[Optional[int]] = []
    for record in attempts:
        measurements = record.get("inference_durations")
        if not isinstance(measurements, list):
            raise BenchBridgeError(
                f"attempt {record.get('cell_id')!r} has no inference duration list"
            )
        for measurement in measurements:
            if not isinstance(measurement, dict) or set(measurement) != {
                "value",
                "unit",
                "source",
            }:
                raise BenchBridgeError(
                    "each inference duration measurement must carry exactly a value, "
                    "unit, and source"
                )
            value = measurement["value"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchBridgeError(
                    "an inference duration value must be a nonnegative integer"
                )
            if (
                measurement["unit"] != DURATION_UNIT_NS
                or measurement["source"] not in DURATION_SOURCES
            ):
                raise BenchBridgeError(
                    "an attempt declares an unknown duration unit or observing clock"
                )
            successful.append(value)
        observations = record.get("failed_request_duration_observations")
        if not isinstance(observations, list) or len(observations) != _count(
            record, "failed_requests"
        ):
            raise BenchBridgeError(
                "failed request duration observations must carry one entry per "
                "failed request"
            )
        for observation in observations:
            if observation is not None and (
                isinstance(observation, bool)
                or not isinstance(observation, int)
                or observation < 0
            ):
                raise BenchBridgeError(
                    "a failed request duration must be a nonnegative integer or null"
                )
        reported = [value for value in observations if value is not None]
        if sum(reported) != _count(record, "failed_request_duration_ns"):
            raise BenchBridgeError(
                "failed request durations do not sum to failed_request_duration_ns"
            )
        expected_source = (
            HOST_DURATION_SOURCE
            if observations and len(reported) == len(observations)
            else "unavailable"
            if not reported
            else "mixed"
        )
        if record.get("failed_request_duration_source") != expected_source:
            raise BenchBridgeError(
                "failed_request_duration_source disagrees with the recorded "
                "observations"
            )
        failed_observations.extend(observations)
    reported_failures = [value for value in failed_observations if value is not None]
    return {
        "attempts": len(attempts),
        "completed_attempts": sum(
            1 for record in attempts if record.get("status") == "completed"
        ),
        "failed_attempts": sum(
            1 for record in attempts if record.get("status") != "completed"
        ),
        "inference_calls": len(successful),
        "inference_duration_ns_total": sum(successful),
        "failed_requests": len(failed_observations),
        "failed_request_duration_ns_total": sum(reported_failures),
        "failed_request_duration_source": (
            HOST_DURATION_SOURCE
            if failed_observations
            and len(reported_failures) == len(failed_observations)
            else "unavailable"
            if not reported_failures
            else "mixed"
        ),
        "tokens_in_total": sum(_count(record, "tokens_in") for record in attempts),
        "tokens_out_total": sum(_count(record, "tokens_out") for record in attempts),
        "reasoning_tokens_total": sum(
            _count(record, "reasoning_tokens") for record in attempts
        ),
        "retry_count_total": sum(_count(record, "retry_count") for record in attempts),
    }


def _operational_profile(
    records: Sequence[dict[str, Any]], attempts: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate rank-neutral inference accounting from completed field cells."""

    measurements = [
        measurement
        for record in records
        for measurement in record["inference_durations"]
    ]
    durations = [int(measurement["value"]) for measurement in measurements]
    duration_sources = sorted({str(m["source"]) for m in measurements})
    # A pooled percentile over two different clocks is not one comparable
    # quantity, so each observing clock also reports its own profile.
    by_source = {}
    for source in duration_sources:
        source_durations = [
            int(m["value"]) for m in measurements if str(m["source"]) == source
        ]
        by_source[source] = {
            "inference_calls": len(source_durations),
            "inference_duration_ns_total": sum(source_durations),
            "inference_duration_ns_p50": _nearest_rank(source_durations, 50),
            "inference_duration_ns_p95": _nearest_rank(source_durations, 95),
        }
    reasoning_sources = sorted(
        {str(record["reasoning_tokens_source"]) for record in records}
    )
    return {
        "rank_input": False,
        "latency_definition": "one model request, nearest-rank percentile",
        "inference_calls": len(durations),
        "duration_unit": DURATION_UNIT_NS,
        "inference_duration_ns_total": sum(durations),
        "inference_duration_ns_p50": _nearest_rank(durations, 50),
        "inference_duration_ns_p95": _nearest_rank(durations, 95),
        "duration_sources": duration_sources,
        "duration_profile_by_source": by_source,
        "tokens_in_total": sum(int(record["tokens_in"]) for record in records),
        "tokens_out_total": sum(int(record["tokens_out"]) for record in records),
        "reasoning_tokens_total": sum(
            int(record["reasoning_tokens"]) for record in records
        ),
        "reasoning_token_sources": reasoning_sources,
        "retry_count_total": sum(int(record["retry_count"]) for record in records),
        "cells": len(records),
        # The scored cells above are the terminal completions. The ledger below
        # keeps every attempt that reached those cells, including the failed and
        # resumed ones that the terminal record replaces.
        "attempt_ledger": _attempt_ledger(attempts),
    }


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256(payload.encode("utf-8")).hexdigest()


def _read_journals(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the terminal record per cell, every attempt in order, and sources.

    The attempt list is append-only: a completion never erases the failed attempt
    it resumed, because that attempt's duration and tokens are real operational
    cost that the field paid.
    """

    if not paths:
        raise BenchBridgeError("at least one evidence journal is required")
    records_by_id: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    sources = []
    for path in paths:
        raw_bytes = path.read_bytes()
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(raw_bytes).hexdigest(),
                "size_bytes": len(raw_bytes),
            }
        )
        for line_number, raw_line in enumerate(raw_bytes.splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise BenchBridgeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise BenchBridgeError(
                    f"record at {path}:{line_number} is not an object"
                )
            cell_id = record.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise BenchBridgeError(f"record at {path}:{line_number} has no cell_id")
            existing = records_by_id.get(cell_id)
            if existing is not None and _canonical_bytes(existing) != _canonical_bytes(
                record
            ):
                # A resumable journal may carry failed attempts before the one
                # completed attempt. Score only the terminal completion, but keep
                # every attempt in the ledger. Never permit two conflicting
                # completions or a record after completion.
                if existing.get("status") == "failed" and record.get("status") in {
                    "failed",
                    "completed",
                }:
                    records_by_id[cell_id] = record
                    attempts.append(record)
                    continue
                raise BenchBridgeError(f"conflicting duplicate cell_id {cell_id}")
            if existing is None:
                attempts.append(record)
            records_by_id[cell_id] = record
    return list(records_by_id.values()), attempts, sources


def _validate_field(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise BenchBridgeError("the journals contain no records")
    plan_hashes = {record.get("plan_sha256") for record in records}
    if len(plan_hashes) != 1 or not next(iter(plan_hashes)):
        raise BenchBridgeError("all records must share one non-empty plan_sha256")
    shapes = {_canonical_bytes(record.get("field_shape")) for record in records}
    if len(shapes) != 1:
        raise BenchBridgeError("all records must carry the same field_shape")
    shape = json.loads(next(iter(shapes)))
    required_shape = {"models", "datasets", "seeds", "repetitions", "total_cells"}
    if not isinstance(shape, dict) or set(shape) != required_shape:
        raise BenchBridgeError("field_shape has an unsupported schema")
    model_count = int(shape["models"])
    dataset_count = int(shape["datasets"])
    seeds = shape["seeds"]
    repetitions = int(shape["repetitions"])
    expected_total = model_count * dataset_count * len(seeds) * repetitions
    if (
        min(model_count, dataset_count, repetitions) <= 0
        or expected_total != shape["total_cells"]
    ):
        raise BenchBridgeError("field_shape dimensions are invalid")
    if len(records) != expected_total:
        raise BenchBridgeError(
            f"incomplete field: found {len(records)} of {expected_total} planned cells"
        )
    seen_ordinals: set[int] = set()
    for record in records:
        if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise BenchBridgeError("unsupported raw evidence schema_version")
        if record.get("evidence_class") != LOCAL_EVIDENCE_CLASS:
            raise BenchBridgeError("a record has the wrong evidence_class")
        if record.get("status") != "completed":
            failure = record.get("failure", "unknown")
            raise BenchBridgeError(
                f"field contains a failed/incomplete cell: {failure}"
            )
        model_config = record.get("model_config")
        if not isinstance(model_config, dict):
            raise BenchBridgeError("completed cell contains no model_config object")
        if model_config.get("entry_class") != "field":
            raise BenchBridgeError(
                "only a provenance-complete entry_class=field model may be compiled "
                "into independent benchmark evidence"
            )
        if (
            not model_config.get("source_url")
            or not model_config.get("source_revision")
            or not model_config.get("license_id")
        ):
            raise BenchBridgeError(
                "field model_config lacks its public source URL, exact source revision, "
                "or license identifier"
            )
        model_index = record.get("model_index")
        dataset_index = record.get("dataset_index")
        seed_index = record.get("seed_index")
        repetition = record.get("repetition")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                model_index,
                dataset_index,
                seed_index,
                repetition,
            )
        ):
            raise BenchBridgeError("cell coordinates must be integers")
        if not 0 <= model_index < model_count or not 0 <= dataset_index < dataset_count:
            raise BenchBridgeError("model_index or dataset_index is out of range")
        if not 0 <= seed_index < len(seeds) or not 0 <= repetition < repetitions:
            raise BenchBridgeError("seed_index or repetition is out of range")
        if record.get("seed") != seeds[seed_index]:
            raise BenchBridgeError("cell seed disagrees with field_shape")
        expected_ordinal = (
            (model_index * dataset_count + dataset_index) * len(seeds) + seed_index
        ) * repetitions + repetition
        if record.get("cell_ordinal") != expected_ordinal:
            raise BenchBridgeError("cell_ordinal disagrees with its coordinates")
        if expected_ordinal in seen_ordinals:
            raise BenchBridgeError(
                f"duplicate Cartesian cell ordinal {expected_ordinal}"
            )
        seen_ordinals.add(expected_ordinal)
        returns = record.get("returns")
        if (
            not isinstance(returns, list)
            or len(returns) < 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in returns
            )
        ):
            raise BenchBridgeError("completed cell contains invalid returns")
        confidences = record.get("confidences")
        outcomes = record.get("outcomes")
        if not isinstance(confidences, list) or not isinstance(outcomes, list):
            raise BenchBridgeError("confidences and outcomes must be arrays")
        if len(confidences) != len(outcomes) or len(confidences) > len(returns):
            raise BenchBridgeError(
                "reported confidences and outcomes must align with each other and cannot "
                "outnumber returns"
            )
        if record.get("returns_sha256") != _digest(returns):
            raise BenchBridgeError("returns_sha256 does not match returns")
        accounting_fields = (
            "tokens_in",
            "tokens_out",
            "reasoning_tokens",
            "retry_count",
            "inference_duration_ns",
        )
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), int)
            or int(record[field]) < 0
            for field in accounting_fields
        ):
            raise BenchBridgeError(
                "completed cell contains invalid inference accounting"
            )
        measurements = record.get("inference_durations")
        if not isinstance(measurements, list) or not measurements:
            raise BenchBridgeError(
                "completed cell contains no ordered inference duration measurements"
            )
        samples = []
        for measurement in measurements:
            if not isinstance(measurement, dict) or set(measurement) != {
                "value",
                "unit",
                "source",
            }:
                raise BenchBridgeError(
                    "each inference duration measurement must carry exactly a value, "
                    "unit, and source"
                )
            value = measurement["value"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchBridgeError(
                    "an inference duration value must be a nonnegative integer"
                )
            if measurement["unit"] != DURATION_UNIT_NS:
                raise BenchBridgeError(
                    f"inference duration unit must be {DURATION_UNIT_NS!r}"
                )
            if measurement["source"] not in DURATION_SOURCES:
                raise BenchBridgeError(
                    "an inference duration measurement declares an unknown observing "
                    f"clock: {measurement['source']!r}"
                )
            samples.append(value)
        if sum(samples) != record["inference_duration_ns"]:
            raise BenchBridgeError(
                "inference duration samples do not sum to inference_duration_ns"
            )
        cadence = int(model_config.get("decision_cadence", 0))
        steps = record.get("steps")
        if cadence <= 0 or not isinstance(steps, int) or steps <= 0:
            raise BenchBridgeError(
                "completed cell has invalid steps or decision cadence"
            )
        if steps != len(returns):
            raise BenchBridgeError(
                "declared steps disagree with the realized return count"
            )
        expected_calls = math.ceil(steps / cadence)
        if len(samples) != expected_calls:
            raise BenchBridgeError(
                "inference duration sample count disagrees with steps and decision cadence"
            )
        observations = record.get("reasoning_token_observations")
        if not isinstance(observations, list) or len(observations) != expected_calls:
            raise BenchBridgeError(
                "reasoning token observations must carry one entry per model request"
            )
        if any(
            observation is not None
            and (
                isinstance(observation, bool)
                or not isinstance(observation, int)
                or observation < 0
            )
            for observation in observations
        ):
            raise BenchBridgeError(
                "a reasoning token observation must be a nonnegative integer or null"
            )
        reported = [value for value in observations if value is not None]
        if sum(reported) != record["reasoning_tokens"]:
            raise BenchBridgeError(
                "reasoning token observations do not sum to reasoning_tokens"
            )
        expected_reasoning_source = (
            "provider-reported"
            if len(reported) == len(observations)
            else "unavailable"
            if not reported
            else "mixed"
        )
        if record.get("reasoning_tokens_source") != expected_reasoning_source:
            raise BenchBridgeError(
                "reasoning_tokens_source disagrees with the recorded observations"
            )
    if seen_ordinals != set(range(expected_total)):
        raise BenchBridgeError(
            "field ordinals do not cover the planned Cartesian product"
        )
    return shape


def _agent_id(record: dict[str, Any]) -> str:
    config = record["model_config"]
    identity = record["model"]
    model = str(config["model"])
    suffix = _digest({"identity": identity, "config": config})[:12]
    return f"{model}@{suffix}"


def _safe_dataset_name(dataset_id: str, dataset_index: int) -> str:
    cleaned = _SAFE_NAME.sub("-", dataset_id).strip("-.") or f"dataset-{dataset_index}"
    return f"{dataset_index:02d}-{cleaned}"


def compile_benchmark_evidence(
    journal_paths: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    """Validate a complete field and emit dataset-specific SharpeBench inputs."""

    records, attempts, sources = _read_journals(journal_paths)
    shape = _validate_field(records)
    plan_sha256 = str(records[0]["plan_sha256"])
    outputs = []
    for dataset_index in range(int(shape["datasets"])):
        dataset_records = [r for r in records if r["dataset_index"] == dataset_index]
        dataset_variants = {_canonical_bytes(r["dataset"]) for r in dataset_records}
        if len(dataset_variants) != 1:
            raise BenchBridgeError(
                f"dataset index {dataset_index} has conflicting metadata"
            )
        dataset = json.loads(next(iter(dataset_variants)))
        periods_per_year = float(dataset["periods_per_year"])
        submissions = []
        model_entries = []
        seen_agent_ids: set[str] = set()
        for model_index in range(int(shape["models"])):
            model_records = [
                r for r in dataset_records if r["model_index"] == model_index
            ]
            model_records.sort(key=lambda r: (r["seed_index"], r["repetition"]))
            configs = {_canonical_bytes(r["model_config"]) for r in model_records}
            identities = {_canonical_bytes(r["model"]) for r in model_records}
            trials = {r.get("n_trials") for r in model_records}
            if len(configs) != 1 or len(identities) != 1 or len(trials) != 1:
                raise BenchBridgeError(
                    f"model index {model_index} has conflicting provenance"
                )
            representative = model_records[0]
            model_cell_ids = {record["cell_id"] for record in model_records}
            agent_id = _agent_id(representative)
            if agent_id in seen_agent_ids:
                raise BenchBridgeError(f"duplicate compiled agent_id {agent_id}")
            seen_agent_ids.add(agent_id)
            runs = []
            for record in model_records:
                cost = float(record.get("cost", 0.0))
                if not math.isfinite(cost) or cost < 0.0:
                    raise BenchBridgeError("run cost must be finite and nonnegative")
                runs.append(
                    {
                        "returns": record["returns"],
                        "trace": record["trace"],
                        "confidences": record["confidences"],
                        "outcomes": record["outcomes"],
                        "cost": cost,
                    }
                )
            submissions.append(
                {
                    "agent_id": agent_id,
                    "runs": runs,
                    "in_sample_trials": int(next(iter(trials))),
                    "candidates": representative["model_config"].get(
                        "selection_candidates", []
                    ),
                }
            )
            model_entries.append(
                {
                    "agent_id": agent_id,
                    "model_index": model_index,
                    "identity": representative["model"],
                    "config": representative["model_config"],
                    "operational_profile": _operational_profile(
                        model_records,
                        [
                            attempt
                            for attempt in attempts
                            if attempt["cell_id"] in model_cell_ids
                        ],
                    ),
                }
            )
        stem = _safe_dataset_name(str(dataset["dataset_id"]), dataset_index)
        submissions_path = output_dir / f"{stem}.submissions.json"
        submissions_sha256 = _atomic_json(submissions_path, submissions)
        outputs.append(
            {
                "dataset_index": dataset_index,
                "dataset": dataset,
                "submissions_path": str(submissions_path.resolve()),
                "submissions_sha256": submissions_sha256,
                "periods_per_year": periods_per_year,
                # Repetitions vary the model sampling seed on one market path;
                # they are reliability runs, not execution-noise replicates to
                # be averaged by SharpeBench.
                "execution_seeds_per_window": 1,
                "runs_per_agent": len(shape["seeds"]) * int(shape["repetitions"]),
                "models": model_entries,
                "score_command": [
                    "sharpebench",
                    "score",
                    str(submissions_path.resolve()),
                    "--periods-per-year",
                    format(periods_per_year, ".12g"),
                    "--execution-seeds-per-window",
                    "1",
                    "--json",
                ],
            }
        )
    manifest = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "artifact_class": "sharpearena-to-sharpebench-field",
        "plan_sha256": plan_sha256,
        "field_shape": shape,
        "source_journals": sources,
        "outputs": outputs,
    }
    manifest_path = output_dir / "benchmark-manifest.json"
    manifest_sha256 = _atomic_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = compile_benchmark_evidence(args.journals, args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["BenchBridgeError", "compile_benchmark_evidence"]
