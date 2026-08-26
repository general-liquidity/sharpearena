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

from .local_agents import EVIDENCE_SCHEMA_VERSION, LOCAL_EVIDENCE_CLASS

BRIDGE_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class BenchBridgeError(ValueError):
    """Raw field evidence is incomplete, conflicting, or not scoreable."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise BenchBridgeError("at least one evidence journal is required")
    records_by_id: dict[str, dict[str, Any]] = {}
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
                # completed attempt. Preserve those attempts in the source hash,
                # but score only the terminal completion. Never permit two
                # conflicting completions or a record after completion.
                if existing.get("status") == "failed" and record.get("status") in {
                    "failed",
                    "completed",
                }:
                    records_by_id[cell_id] = record
                    continue
                raise BenchBridgeError(f"conflicting duplicate cell_id {cell_id}")
            records_by_id[cell_id] = record
    return list(records_by_id.values()), sources


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

    records, sources = _read_journals(journal_paths)
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
