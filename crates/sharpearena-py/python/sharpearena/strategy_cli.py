"""Run or inspect a closed-DSL, observed-trial local strategy search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from .local_agents import DatasetSpec, ModelRunConfig, OllamaClient, SamplingConfig
from .strategy_generation import (
    OllamaStrategyGenerator,
    StrategySearchPlan,
    StrategySearchRunner,
)

_PLAN_FIELDS = {
    "model",
    "prompt",
    "requested_candidates",
    "validation_dataset",
    "test_dataset",
    "validation_seeds",
    "test_seeds",
    "max_steps",
}
_MODEL_FIELDS = {
    "model",
    "sampling",
    "scaffold",
    "decision_cadence",
    "precommitted_n_trials",
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


def _reject_unknown(payload: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")


def _dataset(payload: dict[str, Any], base: Path) -> DatasetSpec:
    item = dict(payload)
    _reject_unknown(item, _DATASET_FIELDS, "dataset")
    csv_path = item.pop("csv_path", None)
    if csv_path is not None:
        item["csv_text"] = (base / str(csv_path)).resolve().read_text(encoding="utf-8")
    return DatasetSpec(**item)


def load_strategy_plan(path: Path) -> StrategySearchPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strategy plan must be a JSON object")
    _reject_unknown(payload, _PLAN_FIELDS, "strategy plan")
    if not isinstance(payload.get("model"), dict):
        raise ValueError("model must be an object")
    model_payload = dict(payload["model"])
    _reject_unknown(model_payload, _MODEL_FIELDS, "model")
    sampling_payload = model_payload.pop("sampling", {})
    if not isinstance(sampling_payload, dict):
        raise ValueError("model.sampling must be an object")
    _reject_unknown(sampling_payload, _SAMPLING_FIELDS, "model.sampling")
    sampling = SamplingConfig(**sampling_payload)
    for name in ("validation_dataset", "test_dataset"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"{name} must be an object")
    return StrategySearchPlan(
        model=ModelRunConfig(sampling=sampling, **model_payload),
        prompt=str(payload["prompt"]),
        requested_candidates=int(payload["requested_candidates"]),
        validation_dataset=_dataset(payload["validation_dataset"], path.parent),
        test_dataset=_dataset(payload["test_dataset"], path.parent),
        validation_seeds=tuple(int(seed) for seed in payload["validation_seeds"]),
        test_seeds=tuple(int(seed) for seed in payload["test_seeds"]),
        max_steps=payload.get("max_steps"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--inspect", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_strategy_plan(args.plan)
    if args.inspect:
        print(
            json.dumps(
                {
                    "plan_sha256": plan.plan_sha256,
                    "model": asdict(plan.model),
                    "requested_candidates": plan.requested_candidates,
                    "validation_dataset": plan.validation_dataset.public_record(),
                    "test_dataset": plan.test_dataset.public_record(),
                    "validation_seeds": plan.validation_seeds,
                    "test_seeds": plan.test_seeds,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    generator = OllamaStrategyGenerator(
        OllamaClient(args.ollama_url, timeout_seconds=args.timeout_seconds)
    )
    evidence = StrategySearchRunner(generator).run(plan, args.evidence)
    print(json.dumps({"status": evidence["status"], "plan_sha256": plan.plan_sha256}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["load_strategy_plan"]
