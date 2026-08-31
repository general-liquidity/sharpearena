"""Resumable local-model field runner.

Example: ``python -m sharpearena.local_field_cli --plan field.json --evidence out.jsonl``.
The plan is hashed before any cell runs. Historical datasets are referenced by path;
their bytes, not the path, determine dataset identity.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .local_agents import (
    DatasetSpec,
    EvidenceJournal,
    FieldPlan,
    LocalFieldRunner,
    ModelRunConfig,
    OllamaClient,
    OpenAICompatibleClient,
    SamplingConfig,
    load_identity_manifest,
)

_PLAN_FIELDS = {
    "models",
    "datasets",
    "seeds",
    "repetitions",
    "max_steps",
    "parallel_requests",
    "shard_index",
    "shard_count",
}
_MODEL_FIELDS = {
    "model",
    "sampling",
    "scaffold",
    "decision_cadence",
    "precommitted_n_trials",
    "selection_candidates",
    "entry_class",
    "source_url",
    "source_revision",
    "license_id",
}
_SAMPLING_FIELDS = {
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "context_tokens",
    "thinking",
    "thinking_budget_tokens",
}
_DATASET_FIELDS = {
    "dataset_id",
    "tier",
    "csv_path",
    "n_symbols",
    "n_days",
    "periods_per_year",
    "window_start",
    "window_end",
    "fee_bps",
    "slippage_bps",
    "impact_bps",
    "financing_bps",
    "max_participation",
}


def _reject_unknown(payload: dict, allowed: set[str], path: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _resolve_csv_path(plan_dir: Path, csv_path: object) -> Path:
    """Resolve a plan-embedded ``csv_path`` strictly inside the plan's directory.

    A field plan is a document that travels (shared, downloaded, committed), so a
    path inside it is not the operator's own keyboard input: an absolute path or a
    ``..`` escape would let a hostile plan read an arbitrary file into ``csv_text``
    and from there into the evidence artifact. ``resolve()`` runs before the
    containment check, so a symlink inside the plan directory pointing outside it
    is refused too.
    """
    candidate = Path(str(csv_path))
    if candidate.is_absolute():
        raise ValueError(
            f"csv_path must be relative to the plan directory (got absolute path {csv_path!r})"
        )
    root = plan_dir.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"csv_path escapes the plan directory: {csv_path!r} resolves to {resolved}"
        )
    return resolved


def load_plan(path: Path) -> FieldPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("field plan must be a JSON object")
    _reject_unknown(payload, _PLAN_FIELDS, "field plan")
    models = []
    for index, item in enumerate(payload.get("models", [])):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be an object")
        item = dict(item)
        _reject_unknown(item, _MODEL_FIELDS, f"models[{index}]")
        sampling_payload = item.pop("sampling", {})
        if not isinstance(sampling_payload, dict):
            raise ValueError(f"models[{index}].sampling must be an object")
        _reject_unknown(sampling_payload, _SAMPLING_FIELDS, f"models[{index}].sampling")
        sampling = SamplingConfig(**sampling_payload)
        if "selection_candidates" in item:
            candidates = item["selection_candidates"]
            if not isinstance(candidates, list) or not all(
                isinstance(series, list) for series in candidates
            ):
                raise ValueError(
                    f"models[{index}].selection_candidates must be an array of return arrays"
                )
            item["selection_candidates"] = tuple(
                tuple(value for value in series) for series in candidates
            )
        models.append(ModelRunConfig(sampling=sampling, **item))
    datasets = []
    for index, item in enumerate(payload.get("datasets", [])):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        item = dict(item)
        _reject_unknown(item, _DATASET_FIELDS, f"datasets[{index}]")
        csv_path = item.pop("csv_path", None)
        if csv_path is not None:
            resolved = _resolve_csv_path(path.parent, csv_path)
            item["csv_text"] = resolved.read_text(encoding="utf-8")
        datasets.append(DatasetSpec(**item))
    return FieldPlan(
        models=tuple(models),
        datasets=tuple(datasets),
        seeds=tuple(int(seed) for seed in payload.get("seeds", [])),
        repetitions=int(payload.get("repetitions", 4)),
        max_steps=payload.get("max_steps"),
        parallel_requests=int(payload.get("parallel_requests", 4)),
        shard_index=int(payload.get("shard_index", 0)),
        shard_count=int(payload.get("shard_count", 1)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("ollama", "openai-compatible"),
        default="ollama",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--openai-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--supports-thinking", action="store_true")
    parser.add_argument("--supports-thinking-budget", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="validate and print the resolved plan without running model inference",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_plan(args.plan)
    if args.inspect:
        print(
            json.dumps(
                {
                    "plan_sha256": plan.plan_sha256,
                    "cells_in_shard": len(LocalFieldRunner.cells(plan)),
                    "models": [asdict(model) for model in plan.models],
                    "datasets": [dataset.public_record() for dataset in plan.datasets],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if args.backend == "ollama":
        client = OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    else:
        if args.identity_manifest is None:
            raise ValueError(
                "--identity-manifest is required for --backend openai-compatible"
            )
        client = OpenAICompatibleClient(
            load_identity_manifest(args.identity_manifest),
            base_url=args.openai_url,
            timeout_seconds=args.timeout_seconds,
            supports_thinking=args.supports_thinking,
            supports_thinking_budget=args.supports_thinking_budget,
        )
    counts = LocalFieldRunner(client).run(plan, EvidenceJournal(args.evidence))
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
