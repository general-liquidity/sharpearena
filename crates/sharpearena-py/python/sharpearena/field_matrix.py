"""Per-factor effects of a model x harness x condition field matrix.

A field plan that declares a :class:`~sharpearena.local_agents.FieldMatrix`
holds one arm per (model, harness, condition) cell and runs every arm on the
same datasets, seeds and repetitions. :func:`field_matrix_report` reads the
evidence journals of such a run and reports:

* main effects of each factor and every interaction between factors, as
  sum-to-zero deviations from the grand mean, each with a Student-t interval
  over the independent units, so a model's effect is reported averaged over
  harnesses and conditions and apart from its interaction with them;
* the share of the between-cell sum of squares each term carries;
* for the condition (prompt-variant) factor, the contrast of every variant
  against the reference variant, overall and within each model and harness;
* the sensitivity of decisions to the prompt variant: on the steps where the
  reference arm and the variant arm were shown the identical observation, how
  often their orders agree.

The unit of replication is one (dataset, seed) pair. Repetitions of a seed
share its market path, so they are averaged inside the unit rather than
counted as independent replicates. A unit enters the effect estimates only
when every arm has every repetition scored on it; a missing, failed or
unscored cell excludes the unit and the report names why. Nothing is imputed.

The report is diagnostic and rank-neutral. Its intervals are unadjusted for
the number of effects reported.

Example: ``python -m sharpearena.field_matrix --plan plan.json shard-0.jsonl``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .impact_diagnostics import t_critical
from .local_agents import MATRIX_FACTORS, FieldCell, FieldPlan

REPORT_SCHEMA = "sharpearena.field-matrix-report.v1"
#: Per-cell outcomes the report can decompose. ``mean_return`` is the mean
#: per-bar return, which every completed cell has. ``deflated_sharpe`` is the
#: kernel's ranked number, which a cell can record as unavailable.
MATRIX_METRICS = ("mean_return", "deflated_sharpe")
_METHOD = (
    "Effects are sum-to-zero deviations from the grand mean of a balanced "
    "full factorial, estimated per unit and summarised across units with a "
    "two-sided Student-t interval on n - 1 degrees of freedom. A unit is one "
    "(dataset, seed) pair; its repetitions are averaged inside it. Intervals "
    "are unadjusted for multiplicity."
)

Unit = tuple[int, int]
Cell = tuple[str, str, str]


class FieldMatrixError(ValueError):
    """The journals cannot be read as evidence of the declared matrix."""


def _read_journal(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise FieldMatrixError(f"{path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise FieldMatrixError(f"{path}:{line_number}: a record must be an object")
        records.append(record)
    return records


def _index(record: dict[str, Any], key: str, bound: int) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < bound:
        raise FieldMatrixError(f"record {key} {value!r} is outside [0, {bound})")
    return value


def _select_records(
    plan: FieldPlan, records: Iterable[dict[str, Any]]
) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    """One record per cell: a completed record wins over a failed attempt.

    Keys are (model_index, dataset_index, seed_index, repetition). A record
    from another plan, or whose ``cell_id`` does not match its coordinates, is
    refused, as are two completed records of one cell with different returns.
    """

    plan_sha256 = plan.plan_sha256
    chosen: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for record in records:
        if record.get("plan_sha256") != plan_sha256:
            raise FieldMatrixError(
                f"record plan_sha256 {record.get('plan_sha256')!r} is not this plan's "
                f"{plan_sha256}"
            )
        status = record.get("status")
        if status not in ("completed", "failed"):
            raise FieldMatrixError(f"record status {status!r} is not completed or failed")
        key = (
            _index(record, "model_index", len(plan.models)),
            _index(record, "dataset_index", len(plan.datasets)),
            _index(record, "seed_index", len(plan.seeds)),
            _index(record, "repetition", plan.repetitions),
        )
        cell = FieldCell(key[0], key[1], plan.seeds[key[2]], key[3])
        if record.get("cell_id") != cell.key(plan):
            raise FieldMatrixError(
                f"record cell_id {record.get('cell_id')!r} does not match its coordinates"
            )
        held = chosen.get(key)
        if held is None or held["status"] == "failed":
            chosen[key] = record
        elif status == "completed" and record.get("returns") != held.get("returns"):
            raise FieldMatrixError(f"conflicting completed records for cell {key}")
    return chosen


def _metric(record: Optional[dict[str, Any]], metric: str) -> float | str:
    """The cell's outcome, or why it has none."""

    if record is None:
        return "missing"
    if record["status"] != "completed":
        failure = record.get("failure") or {}
        return f"failed: {failure.get('type', 'unknown')}"
    if metric == "mean_return":
        returns = record.get("returns") or []
        value = math.fsum(float(r) for r in returns) / len(returns) if returns else math.nan
    else:
        value = record.get("deflated_sharpe")
        if isinstance(value, str):
            return f"deflated_sharpe unavailable: {value}"
        value = float(value) if isinstance(value, (int, float)) else math.nan
    if not math.isfinite(value):
        return f"{metric} is not a finite number"
    return value


def _summary(values: Sequence[float], confidence: float) -> dict[str, Any]:
    """Mean of per-unit values with a two-sided Student-t interval."""

    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": None,
            "se": None,
            "lo": None,
            "hi": None,
            "interval": "unavailable: a dispersion estimate needs two units",
        }
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    half = t_critical(n - 1, confidence) * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "lo": mean - half,
        "hi": mean + half,
        "interval": "student-t",
    }


def _terms(sizes: Sequence[int]) -> list[tuple[int, ...]]:
    """Every main effect and interaction among the factors with two or more levels."""

    varying = [axis for axis, size in enumerate(sizes) if size >= 2]
    return [
        subset
        for order in range(1, len(varying) + 1)
        for subset in itertools.combinations(varying, order)
    ]


def _effect(y: np.ndarray, term: tuple[int, ...]) -> np.ndarray:
    """The sum-to-zero effect of ``term``, by inclusion-exclusion over its axes.

    effect_S = sum over T subset of S of (-1)^(|S|-|T|) * mean of y over the
    axes outside T. The result keeps every axis, with length one outside S.
    """

    total = np.zeros([y.shape[axis] if axis in term else 1 for axis in range(y.ndim)])
    for order in range(len(term) + 1):
        for kept in itertools.combinations(term, order):
            dropped = tuple(axis for axis in range(y.ndim) if axis not in kept)
            sign = -1.0 if (len(term) - order) % 2 else 1.0
            total = total + sign * y.mean(axis=dropped, keepdims=True)
    return total


def _orders_key(decision: Any) -> list[str]:
    orders = decision.get("orders", []) if isinstance(decision, dict) else []
    return sorted(
        json.dumps(
            {k: order.get(k) for k in ("symbol", "action", "target_weight")}
            if isinstance(order, dict)
            else order,
            sort_keys=True,
        )
        for order in orders
    )


def _agreement(reference: dict[str, Any], variant: dict[str, Any]) -> tuple[int, int]:
    """(steps shown the identical observation, of those, steps with equal orders)."""

    ref_obs = reference.get("observation_sha256") or []
    var_obs = variant.get("observation_sha256") or []
    ref_dec = reference.get("decisions") or []
    var_dec = variant.get("decisions") or []
    matched = agreeing = 0
    # inv: matched and agreeing count the steps before ``step`` only.
    for step in range(min(len(ref_obs), len(var_obs), len(ref_dec), len(var_dec))):
        if ref_obs[step] != var_obs[step]:
            continue
        matched += 1
        if _orders_key(ref_dec[step]) == _orders_key(var_dec[step]):
            agreeing += 1
    return matched, agreeing


def field_matrix_report(
    plan: FieldPlan,
    records: Iterable[dict[str, Any]],
    *,
    metric: str = "mean_return",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Main effects, interactions and prompt-variant sensitivity of a matrix field."""

    matrix = plan.matrix
    if matrix is None:
        raise FieldMatrixError("the plan declares no matrix")
    if metric not in MATRIX_METRICS:
        raise FieldMatrixError(f"metric must be one of {MATRIX_METRICS}")
    if not 0.0 < confidence < 1.0:
        raise FieldMatrixError("confidence must lie in (0, 1)")
    levels = matrix.levels()
    sizes = [len(levels[name]) for name in MATRIX_FACTORS]
    arms = matrix.arm_indices(plan.models)
    chosen = _select_records(plan, records)
    units: list[Unit] = [
        (dataset, seed) for dataset in range(len(plan.datasets)) for seed in range(len(plan.seeds))
    ]

    complete: list[np.ndarray] = []
    excluded: list[dict[str, Any]] = []
    for dataset, seed in units:
        y = np.zeros(sizes)
        reasons = []
        for position in itertools.product(*(range(size) for size in sizes)):
            cell = tuple(levels[name][i] for name, i in zip(MATRIX_FACTORS, position))
            arm = arms[cell]  # type: ignore[index]
            values = []
            for repetition in range(plan.repetitions):
                value = _metric(chosen.get((arm, dataset, seed, repetition)), metric)
                if isinstance(value, str):
                    reasons.append(
                        {"cell": list(cell), "repetition": repetition, "reason": value}
                    )
                else:
                    values.append(value)
            # A cell short of a repetition has a reason above, which
            # excludes the unit, so this mean is read only when complete.
            y[position] = math.fsum(values) / plan.repetitions
        if reasons:
            excluded.append(
                {"dataset_index": dataset, "seed_index": seed, "cells": reasons}
            )
        else:
            complete.append(y)

    effects: dict[str, list[dict[str, Any]]] = {}
    variance_share: dict[str, float] | str
    if not complete:
        variance_share = "unavailable: no complete unit"
    else:
        for term in _terms(sizes):
            name = ":".join(MATRIX_FACTORS[axis] for axis in term)
            per_unit = [_effect(y, term) for y in complete]
            entries = []
            for position in itertools.product(*(range(sizes[axis]) for axis in term)):
                index = tuple(
                    position[term.index(axis)] if axis in term else 0
                    for axis in range(len(sizes))
                )
                entries.append(
                    {
                        "levels": {
                            MATRIX_FACTORS[axis]: levels[MATRIX_FACTORS[axis]][i]
                            for axis, i in zip(term, position)
                        },
                        **_summary([float(e[index]) for e in per_unit], confidence),
                    }
                )
            effects[name] = entries
        mean_y = np.mean(complete, axis=0)
        total_ss = float(np.sum((mean_y - mean_y.mean()) ** 2))
        if total_ss > 0.0:
            variance_share = {
                ":".join(MATRIX_FACTORS[axis] for axis in term): float(
                    np.sum(np.broadcast_to(_effect(mean_y, term), mean_y.shape) ** 2)
                )
                / total_ss
                for term in _terms(sizes)
            }
        else:
            variance_share = "unavailable: every cell mean is equal"

    reference = matrix.conditions[0]
    condition_contrasts = []
    simple_condition_effects = []
    for k, condition in enumerate(matrix.conditions[1:], 1):
        if complete:
            condition_contrasts.append(
                {
                    "condition": condition,
                    "reference": reference,
                    **_summary(
                        [float(np.mean(y[:, :, k] - y[:, :, 0])) for y in complete],
                        confidence,
                    ),
                }
            )
            for i, model in enumerate(matrix.models):
                for j, harness in enumerate(matrix.harnesses):
                    simple_condition_effects.append(
                        {
                            "model": model,
                            "harness": harness,
                            "condition": condition,
                            "reference": reference,
                            **_summary(
                                [float(y[i, j, k] - y[i, j, 0]) for y in complete],
                                confidence,
                            ),
                        }
                    )

    decision_sensitivity = []
    for condition in matrix.conditions[1:]:
        for model in matrix.models:
            for harness in matrix.harnesses:
                ref_arm = arms[(model, harness, reference)]
                var_arm = arms[(model, harness, condition)]
                matched_total = agreeing_total = 0
                unit_rates = []
                for dataset, seed in units:
                    unit_matched = unit_agreeing = 0
                    for repetition in range(plan.repetitions):
                        ref = chosen.get((ref_arm, dataset, seed, repetition))
                        var = chosen.get((var_arm, dataset, seed, repetition))
                        if (
                            ref is None
                            or var is None
                            or ref["status"] != "completed"
                            or var["status"] != "completed"
                        ):
                            continue
                        matched, agreeing = _agreement(ref, var)
                        unit_matched += matched
                        unit_agreeing += agreeing
                    matched_total += unit_matched
                    agreeing_total += unit_agreeing
                    if unit_matched:
                        unit_rates.append(unit_agreeing / unit_matched)
                decision_sensitivity.append(
                    {
                        "model": model,
                        "harness": harness,
                        "condition": condition,
                        "reference": reference,
                        "matched_steps": matched_total,
                        "agreeing_steps": agreeing_total,
                        "agreement": (
                            agreeing_total / matched_total if matched_total else None
                        ),
                        "per_unit_agreement": (
                            _summary(unit_rates, confidence) if unit_rates else None
                        ),
                    }
                )

    return {
        "schema": REPORT_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "design": matrix.record(),
        "metric": metric,
        "confidence": confidence,
        "method": _METHOD,
        "rank_neutral": True,
        "units": {
            "total": len(units),
            "complete": len(complete),
            "excluded": excluded,
        },
        "effects": effects,
        "variance_share": variance_share,
        "condition_contrasts": condition_contrasts,
        "simple_condition_effects": simple_condition_effects,
        "decision_sensitivity": decision_sensitivity,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .local_field_cli import load_plan

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metric", choices=MATRIX_METRICS, default="mean_return")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("journals", type=Path, nargs="+")
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    records = [record for path in args.journals for record in _read_journal(path)]
    report = field_matrix_report(
        plan, records, metric=args.metric, confidence=args.confidence
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


__all__ = ["FieldMatrixError", "MATRIX_METRICS", "REPORT_SCHEMA", "field_matrix_report"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
