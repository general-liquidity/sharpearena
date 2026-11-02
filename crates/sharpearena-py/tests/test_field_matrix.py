"""The model x harness x condition field matrix and its effects report.

Every model here is a deterministic stub: no model server, network access or
download is involved.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from sharpearena.field_matrix import (
    FieldMatrixError,
    _effect,
    field_matrix_report,
    main as matrix_main,
)
from sharpearena.impact_diagnostics import t_critical
from sharpearena.local_agents import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    SCAFFOLD_RENDERERS,
    DatasetSpec,
    EvidenceJournal,
    FieldCell,
    FieldMatrix,
    FieldPlan,
    InferenceOutcome,
    InferenceResult,
    LocalAgentError,
    LocalFieldRunner,
    ModelIdentity,
    ModelRunConfig,
    PromptRenderer,
    SamplingConfig,
    renderer_for_scaffold,
    renderer_identity,
    scaffold_coordinates,
    scaffold_label,
)
from sharpearena.local_field_cli import load_plan, main as field_main

FRAMED = "framed-test-v1"


class FramedRenderer(PromptRenderer):
    """A second harness: the same payload behind a different frame."""

    scaffold = FRAMED

    def messages(self, observation, *, mandate=None, cadence=1):
        messages = super().messages(observation, mandate=mandate, cadence=cadence)
        messages[1]["content"] = "FRAMED\n" + messages[1]["content"]
        return messages


@pytest.fixture(autouse=True)
def framed_harness(monkeypatch):
    monkeypatch.setitem(SCAFFOLD_RENDERERS, FRAMED, FramedRenderer)


class StubModel:
    """Deterministic decisions keyed on the model name, the sampling seed and,
    when ``prompt_sensitive``, the exact messages the renderer produced."""

    def __init__(self, prompt_sensitive: bool):
        self.prompt_sensitive = prompt_sensitive
        self.seen: list[tuple[str, str, str]] = []

    def identity(self, model):
        return ModelIdentity(model=model.model, digest="sha256:" + model.model)

    def decide_many(
        self, observations, model, renderer, *, max_workers, sampling_seeds=None
    ):
        outcomes = []
        for observation, seed in zip(observations, sampling_seeds):
            messages = renderer.messages(observation)
            self.seen.append((model.model, renderer.scaffold, messages[0]["content"]))
            basis = [model.model, seed]
            if self.prompt_sensitive:
                basis.append(messages)
            bucket = int(sha256(json.dumps(basis).encode()).hexdigest(), 16) % 4
            decision = {
                "orders": [
                    {
                        "symbol": item["symbol"],
                        "action": "buy",
                        "target_weight": 0.05 * (1 + bucket),
                        "rationale": "stub",
                    }
                    for item in observation["symbols"]
                ],
                "reasoning": "stub",
            }
            outcomes.append(
                InferenceOutcome(
                    result=InferenceResult(
                        decision=decision,
                        raw_response_sha256=f"{model.model}-{seed}-{bucket}",
                        prompt_tokens=1,
                        output_tokens=1,
                        reasoning_tokens=None,
                        total_duration_ns=1,
                        duration_source="host-monotonic-request",
                    )
                )
            )
        return outcomes


MODELS = ("model-a", "model-b")
HARNESSES = ("minimal-stateless-v1", FRAMED)
CONDITIONS = (DEFAULT_PROMPT_VARIANT, "explicit-v1")


def _arms(models=MODELS, harnesses=HARNESSES, conditions=CONDITIONS, **overrides):
    return tuple(
        ModelRunConfig(
            model,
            SamplingConfig(seed=10 * (index + 1)),
            scaffold=scaffold_label(harness, condition),
            **overrides,
        )
        for index, model in enumerate(models)
        for harness in harnesses
        for condition in conditions
    )


def _matrix(models=MODELS, harnesses=HARNESSES, conditions=CONDITIONS):
    return FieldMatrix(models=models, harnesses=harnesses, conditions=conditions)


def _plan(arms=None, matrix=None, **overrides):
    settings = dict(
        models=arms if arms is not None else _arms(),
        datasets=(DatasetSpec("calm", tier="calm", n_symbols=2, n_days=8),),
        seeds=(1, 2, 3),
        repetitions=2,
        max_steps=5,
        order_seed=11,
        matrix=matrix if matrix is not None else _matrix(),
    )
    settings.update(overrides)
    return FieldPlan(**settings)


# --- prompt variants and scaffold labels -------------------------------------


def test_prompt_variant_texts_are_pinned_to_their_versioned_labels():
    # Editing a published variant in place must fail here: a changed text
    # needs a new -vN label so old evidence keeps naming what it ran.
    assert {
        label: sha256(text.encode("utf-8")).hexdigest()
        for label, text in PROMPT_VARIANTS.items()
    } == {
        "terse-v1": "d4ec1eaf8843319854bac9d058e24288ae52ac791124098a3aa998cef762b0f8",
        "explicit-v1": "4a203bd228eb7b7e7475cbf5f80b2505185513b5044d572a44a437382a2ba2f7",
    }
    assert DEFAULT_PROMPT_VARIANT not in PROMPT_VARIANTS


def test_a_variant_label_runs_the_harness_code_with_only_the_prompt_replaced():
    default = renderer_for_scaffold("minimal-stateless-v1")
    terse = renderer_for_scaffold("minimal-stateless-v1+terse-v1")
    framed_terse = renderer_for_scaffold(f"{FRAMED}+terse-v1")
    assert type(terse) is PromptRenderer and type(framed_terse) is FramedRenderer
    assert terse.system_prompt == PROMPT_VARIANTS["terse-v1"]
    assert (terse.scaffold, terse.prompt_variant) == (
        "minimal-stateless-v1+terse-v1",
        "terse-v1",
    )
    # The variant leaves the class default alone for every later renderer.
    assert renderer_for_scaffold("minimal-stateless-v1").system_prompt == (
        PromptRenderer.system_prompt
    )
    observation = {"symbols": [], "cash": 1.0}
    assert terse.messages(observation)[1] == default.messages(observation)[1]
    identity = renderer_identity(terse)
    assert identity["name"] == "minimal-stateless-v1+terse-v1"
    assert identity["render_probe_sha256"] != renderer_identity(default)["render_probe_sha256"]
    # The harness default keeps the exact record it had before variants existed.
    assert renderer_identity(default) == {
        "name": "minimal-stateless-v1",
        "renderer": "sharpearena.local_agents.PromptRenderer",
        "system_prompt_sha256": "9ba3ed5b9a76fe1f2b1e4aa034c0e13f0499411cfab01b92f3627484e57339b4",
        "render_probe_sha256": "8bdd61a9ef84ed49f97dae206ea6ddb571a9e93cd10e89bdd139af6a778f3089",
    }


@pytest.mark.parametrize(
    "label, match",
    [
        ("minimal-stateless-v1+terse-v9", "no prompt variant is registered"),
        ("minimal-stateless-v1+default", "no prompt variant is registered"),
        ("minimal-stateless-v1+", "no prompt variant is registered"),
        ("react-v2+terse-v1", "no renderer is registered"),
    ],
)
def test_an_unresolvable_scaffold_label_is_refused(label, match):
    with pytest.raises(LocalAgentError, match=match):
        scaffold_coordinates(label)
    with pytest.raises(ValueError, match="unknown scaffold"):
        ModelRunConfig("m", scaffold=label)


def test_scaffold_label_and_coordinates_round_trip():
    for harness in HARNESSES:
        for condition in (DEFAULT_PROMPT_VARIANT, *PROMPT_VARIANTS):
            label = scaffold_label(harness, condition)
            assert scaffold_coordinates(label) == (harness, condition)
    assert scaffold_label("minimal-stateless-v1", DEFAULT_PROMPT_VARIANT) == (
        "minimal-stateless-v1"
    )


# --- the declared design -----------------------------------------------------


def test_a_plain_plan_digests_exactly_as_before_matrices_existed():
    plan = FieldPlan(
        models=(
            ModelRunConfig("m", SamplingConfig(seed=1)),
            ModelRunConfig("n", SamplingConfig(seed=2), decision_cadence=2),
        ),
        datasets=(DatasetSpec("calm", n_symbols=2, n_days=10),),
        seeds=(1, 2),
        repetitions=2,
        max_steps=4,
        order_seed=7,
    )
    # Computed from origin/main e7c9ed1 before this change.
    assert plan.plan_sha256 == (
        "62031eecf9dffe326425318c37a99ea0609d483a0a370c0e3313996f4b36d4c1"
    )


def test_the_matrix_design_is_bound_into_the_plan_digest():
    plan = _plan()
    assert plan.matrix.record() == {
        "factors": {
            "model": list(MODELS),
            "harness": list(HARNESSES),
            "condition": list(CONDITIONS),
        },
        "condition_factor": "prompt-variant",
        "reference_condition": DEFAULT_PROMPT_VARIANT,
    }
    reordered = _plan(matrix=_matrix(conditions=tuple(reversed(CONDITIONS))))
    assert reordered.plan_sha256 != plan.plan_sha256
    # Shards of one matrix field share its digest, so their journals combine.
    assert _plan(shard_index=1, shard_count=2).plan_sha256 == plan.plan_sha256


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: _plan(arms=_arms()[:-1]), "has no arm for cells"),
        (
            lambda: _plan(arms=_arms() + (ModelRunConfig("model-c"),)),
            "outside the declared matrix",
        ),
        (
            lambda: _plan(
                arms=_arms()[:-1]
                + (replace(_arms()[-2], sampling=SamplingConfig(seed=99)),)
            ),
            "both occupy matrix cell",
        ),
        (
            lambda: _plan(
                arms=(replace(_arms()[0], sampling=SamplingConfig(seed=5, temperature=1.0)),)
                + _arms()[1:]
            ),
            "differ in more than their scaffold label",
        ),
        (
            lambda: _plan(arms=_arms(decision_cadence=2)[:4] + _arms()[4:]),
            "share one decision_cadence",
        ),
        (lambda: _matrix(harnesses=("react-v2",)), "not a registered scaffold"),
        (lambda: _matrix(conditions=("default", "loud-v1")), "not a registered prompt"),
        (lambda: _matrix(models=("a", "a")), "must be unique"),
        (lambda: _matrix(models=()), "non-empty tuple"),
        (lambda: _matrix(models=["a"]), "non-empty tuple"),
        (lambda: _matrix(conditions=(" ",)), "non-empty strings"),
    ],
)
def test_a_design_that_is_not_the_full_unconfounded_product_is_refused(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_condition_that_renders_like_another_is_refused(monkeypatch):
    monkeypatch.setitem(PROMPT_VARIANTS, "same-v1", PromptRenderer.system_prompt)
    with pytest.raises(ValueError, match="render identically under harness"):
        _matrix(conditions=(DEFAULT_PROMPT_VARIANT, "same-v1"))


def test_matrix_cells_interleave_across_every_factor():
    plan = _plan()
    order = LocalFieldRunner.execution_order(plan)
    arms = plan.matrix.arm_indices(plan.models)
    cell_of = {index: cell for cell, index in arms.items()}
    blocks = []
    for cell in order:
        if not blocks or blocks[-1] != (cell.model_index, cell.repetition):
            blocks.append((cell.model_index, cell.repetition))
    sequence = [cell_of[model_index] for model_index, _ in blocks]
    rounds = [sequence[i : i + len(arms)] for i in range(0, len(sequence), len(arms))]
    assert len(rounds) == plan.repetitions
    for round_cells in rounds:
        assert sorted(round_cells) == sorted(arms)
    # Within a round no factor level occupies one contiguous run of blocks.
    for axis in range(3):
        levels = [cell[axis] for cell in rounds[0]]
        runs = [level for i, level in enumerate(levels) if i == 0 or levels[i - 1] != level]
        assert len(runs) > len(set(levels)), (axis, levels)


def test_a_matrix_field_runs_each_arm_under_its_own_harness_and_prompt(tmp_path):
    plan = _plan()
    client = StubModel(prompt_sensitive=True)
    path = tmp_path / "field.jsonl"
    counts = LocalFieldRunner(client).run(plan, EvidenceJournal(path))
    assert counts == {"completed": 8 * 3 * 2, "failed": 0, "skipped": 0}
    shown = {(model, scaffold): prompt for model, scaffold, prompt in client.seen}
    assert shown[("model-a", "minimal-stateless-v1+explicit-v1")] == PROMPT_VARIANTS[
        "explicit-v1"
    ]
    assert shown[("model-b", FRAMED)] == PromptRenderer.system_prompt
    for line in path.read_text().splitlines():
        record = json.loads(line)
        arm = plan.models[record["model_index"]]
        assert record["scaffold"]["name"] == arm.scaffold
        assert record["scaffold"]["render_probe_sha256"] == renderer_identity(
            renderer_for_scaffold(arm.scaffold)
        )["render_probe_sha256"]


# --- the effects report on planted outcomes ----------------------------------

MODEL_EFFECT = (0.3, -0.3)
HARNESS_EFFECT = (0.1, -0.1)
MODEL_HARNESS = ((0.05, -0.05), (-0.05, 0.05))
CONDITION_SPREAD = {1: 0.02, 2: 0.04, 3: 0.06}  # condition effect +-e per seed


def _planted(i, j, k, seed):
    unit_shift = 0.7 * seed  # shared by every cell of a unit, so it cancels
    condition = CONDITION_SPREAD[seed] * (1 if k == 0 else -1)
    return 1.0 + unit_shift + MODEL_EFFECT[i] + HARNESS_EFFECT[j] + MODEL_HARNESS[i][j] + condition


def _record(plan, arm, seed_index, repetition, *, value, status="completed", **extra):
    cell = FieldCell(arm, 0, plan.seeds[seed_index], repetition)
    record = {
        "plan_sha256": plan.plan_sha256,
        "cell_id": cell.key(plan),
        "model_index": arm,
        "dataset_index": 0,
        "seed_index": seed_index,
        "repetition": repetition,
        "status": status,
        "deflated_sharpe": 0.5,
        "observation_sha256": [],
        "decisions": [],
    }
    if status == "completed":
        # Repetitions straddle the cell value, so only their mean is planted.
        offset = 0.01 if repetition == 0 else -0.01
        record["returns"] = [value + offset, value + offset]
    else:
        record["failure"] = {"type": "ModelTransportError"}
    record.update(extra)
    return record


def _planted_records(plan):
    arms = plan.matrix.arm_indices(plan.models)
    records = []
    for (model, harness, condition), arm in arms.items():
        i, j, k = (
            MODELS.index(model),
            HARNESSES.index(harness),
            CONDITIONS.index(condition),
        )
        for seed_index, seed in enumerate(plan.seeds):
            for repetition in range(plan.repetitions):
                records.append(
                    _record(
                        plan, arm, seed_index, repetition, value=_planted(i, j, k, seed)
                    )
                )
    return records


def _effects(report, term):
    return {
        tuple(entry["levels"].values()): entry for entry in report["effects"][term]
    }


def test_planted_main_effects_and_interactions_are_recovered():
    plan = _plan()
    report = field_matrix_report(plan, _planted_records(plan))
    assert report["units"] == {"total": 3, "complete": 3, "excluded": []}
    assert list(report["effects"]) == [
        "model",
        "harness",
        "condition",
        "model:harness",
        "model:condition",
        "harness:condition",
        "model:harness:condition",
    ]
    model = _effects(report, "model")
    assert model[("model-a",)]["mean"] == pytest.approx(0.3)
    assert model[("model-b",)]["mean"] == pytest.approx(-0.3)
    harness = _effects(report, "harness")
    assert harness[(FRAMED,)]["mean"] == pytest.approx(-0.1)
    # The model effect is the same in every unit, so its interval is a point.
    assert model[("model-a",)]["sd"] == pytest.approx(0.0, abs=1e-12)
    interaction = _effects(report, "model:harness")
    assert interaction[("model-a", FRAMED)]["mean"] == pytest.approx(-0.05)
    for term in ("model:condition", "harness:condition", "model:harness:condition"):
        for entry in report["effects"][term]:
            assert entry["mean"] == pytest.approx(0.0, abs=1e-12)
    condition = _effects(report, "condition")[(DEFAULT_PROMPT_VARIANT,)]
    # Per unit the default sits e above the mean: e = 0.02, 0.04, 0.06.
    assert condition["n"] == 3
    assert condition["mean"] == pytest.approx(0.04)
    assert condition["sd"] == pytest.approx(0.02)
    half = t_critical(2, 0.95) * 0.02 / math.sqrt(3)
    assert (condition["lo"], condition["hi"]) == (
        pytest.approx(0.04 - half),
        pytest.approx(0.04 + half),
    )
    shares = report["variance_share"]
    assert sum(shares.values()) == pytest.approx(1.0)
    total = 4 * (0.3**2 + 0.1**2 + 0.05**2 + 0.04**2)
    assert shares["model"] == pytest.approx(4 * 0.3**2 / total)
    assert shares["condition"] == pytest.approx(4 * 0.04**2 / total)


def test_the_prompt_contrast_is_reported_overall_and_within_each_arm_pair():
    plan = _plan()
    report = field_matrix_report(plan, _planted_records(plan))
    (contrast,) = report["condition_contrasts"]
    assert (contrast["condition"], contrast["reference"]) == (
        "explicit-v1",
        DEFAULT_PROMPT_VARIANT,
    )
    assert contrast["mean"] == pytest.approx(-0.08)
    assert contrast["sd"] == pytest.approx(0.04)
    simple = {
        (entry["model"], entry["harness"]): entry
        for entry in report["simple_condition_effects"]
    }
    assert set(simple) == {(m, h) for m in MODELS for h in HARNESSES}
    assert all(entry["mean"] == pytest.approx(-0.08) for entry in simple.values())


def test_a_single_level_factor_contributes_no_terms():
    arms = _arms(conditions=(DEFAULT_PROMPT_VARIANT,))
    plan = _plan(arms=arms, matrix=_matrix(conditions=(DEFAULT_PROMPT_VARIANT,)))
    records = [
        _record(plan, arm, s, r, value=float(arm))
        for arm in range(len(arms))
        for s in range(3)
        for r in range(2)
    ]
    report = field_matrix_report(plan, records)
    assert list(report["effects"]) == ["model", "harness", "model:harness"]
    assert report["condition_contrasts"] == []
    assert report["decision_sensitivity"] == []


def test_an_unscored_cell_excludes_its_unit_and_names_the_reason():
    plan = _plan()
    records = _planted_records(plan)
    victim = next(r for r in records if r["seed_index"] == 1 and r["repetition"] == 1)
    failed = dict(victim, status="failed", failure={"type": "ModelTransportError"})
    failed.pop("returns")
    records = [failed if r is victim else r for r in records]
    records = [r for r in records if not (r["seed_index"] == 2 and r["model_index"] == 0)]
    report = field_matrix_report(plan, records)
    assert report["units"]["complete"] == 1
    excluded = {u["seed_index"]: u["cells"] for u in report["units"]["excluded"]}
    assert [c["reason"] for c in excluded[1]] == ["failed: ModelTransportError"]
    assert {c["reason"] for c in excluded[2]} == {"missing"}
    assert len(excluded[2]) == plan.repetitions
    # One complete unit: effects are point estimates with no interval.
    entry = _effects(report, "model")[("model-a",)]
    assert entry["n"] == 1 and entry["lo"] is None
    assert entry["interval"].startswith("unavailable")


def test_a_retried_cell_is_read_from_its_completed_attempt():
    plan = _plan()
    records = _planted_records(plan)
    first = records[0]
    failed = dict(first, status="failed", failure={"type": "ModelHttpError"})
    failed.pop("returns")
    report = field_matrix_report(plan, [failed, *records, failed])
    assert report["units"]["complete"] == 3


def test_deflated_sharpe_unavailability_excludes_rather_than_scores_the_floor():
    plan = _plan()
    records = _planted_records(plan)
    records[0] = dict(records[0], deflated_sharpe="insufficient_bars: 2")
    report = field_matrix_report(plan, records, metric="deflated_sharpe")
    (unit,) = report["units"]["excluded"]
    assert unit["cells"][0]["reason"] == "deflated_sharpe unavailable: insufficient_bars: 2"
    # Every remaining cell scores 0.5, so no cell mean differs.
    assert report["variance_share"] == "unavailable: every cell mean is equal"
    records[1] = dict(records[1], deflated_sharpe=float("nan"))
    report = field_matrix_report(plan, records, metric="deflated_sharpe")
    reasons = [c["reason"] for u in report["units"]["excluded"] for c in u["cells"]]
    assert "deflated_sharpe is not a finite number" in reasons


@pytest.mark.parametrize(
    "tamper, match",
    [
        (lambda r: r.update(plan_sha256="0" * 64), "is not this plan's"),
        (lambda r: r.update(cell_id="0" * 64), "does not match its coordinates"),
        # One past the last arm: the bound is exclusive.
        (lambda r: r.update(model_index=8), "outside"),
        (lambda r: r.update(repetition=True), "outside"),
        (lambda r: r.update(status="running"), "not completed or failed"),
    ],
)
def test_journals_that_are_not_this_matrix_are_refused(tamper, match):
    plan = _plan()
    records = _planted_records(plan)
    tamper(records[3])
    with pytest.raises(FieldMatrixError, match=match):
        field_matrix_report(plan, records)


def test_conflicting_completed_records_of_one_cell_are_refused():
    plan = _plan()
    records = _planted_records(plan)
    other = dict(records[0], returns=[9.0, 9.0])
    with pytest.raises(FieldMatrixError, match="conflicting completed records"):
        field_matrix_report(plan, [*records, other])
    # An identical duplicate (the same journal passed twice) is harmless.
    assert field_matrix_report(plan, [*records, dict(records[0])])["units"]["complete"] == 3


def test_a_plain_plan_or_a_bad_argument_has_no_matrix_report():
    plan = _plan()
    with pytest.raises(FieldMatrixError, match="declares no matrix"):
        field_matrix_report(replace(plan, matrix=None), [])
    with pytest.raises(FieldMatrixError, match="metric"):
        field_matrix_report(plan, [], metric="sharpe")
    with pytest.raises(FieldMatrixError, match="confidence"):
        field_matrix_report(plan, [], confidence=1.0)


def test_no_records_reports_every_unit_missing():
    plan = _plan()
    report = field_matrix_report(plan, [])
    assert report["units"]["complete"] == 0
    assert report["effects"] == {}
    assert report["variance_share"] == "unavailable: no complete unit"
    assert all(entry["agreement"] is None for entry in report["decision_sensitivity"])


def test_the_effect_decomposition_reconstructs_every_cell():
    y = np.random.default_rng(3).normal(size=(2, 3, 2))
    terms = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    rebuilt = y.mean() + sum(_effect(y, term) for term in terms)
    np.testing.assert_allclose(rebuilt, y, atol=1e-12)
    for term in terms:
        for axis in term:
            # Sum-to-zero coding: every effect averages to zero along its axes.
            np.testing.assert_allclose(_effect(y, term).mean(axis=axis), 0.0, atol=1e-12)


# --- decision sensitivity to the prompt variant ------------------------------


def _run(tmp_path, prompt_sensitive):
    plan = _plan()
    path = tmp_path / f"field-{prompt_sensitive}.jsonl"
    LocalFieldRunner(StubModel(prompt_sensitive)).run(plan, EvidenceJournal(path))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return plan, records


def test_a_prompt_blind_model_agrees_with_itself_on_every_shared_observation(tmp_path):
    plan, records = _run(tmp_path, prompt_sensitive=False)
    report = field_matrix_report(plan, records)
    assert len(report["decision_sensitivity"]) == len(MODELS) * len(HARNESSES)
    for entry in report["decision_sensitivity"]:
        # Identical decisions keep the observations identical at every step.
        assert entry["matched_steps"] == 3 * plan.repetitions * plan.max_steps
        assert entry["agreement"] == 1.0
    (contrast,) = report["condition_contrasts"]
    assert contrast["mean"] == 0.0 and contrast["sd"] == 0.0
    # A failed attempt keeps its partial decisions, but it is not a completed
    # arm, so its steps leave the comparison.
    arms = plan.matrix.arm_indices(plan.models)
    variant_arm = arms[(MODELS[0], HARNESSES[0], CONDITIONS[1])]
    for record in records:
        if record["model_index"] == variant_arm and record["seed_index"] == 0:
            record["status"] = "failed"
            record["failure"] = {"type": "ModelTransportError"}
    entry = field_matrix_report(plan, records)["decision_sensitivity"][0]
    assert (entry["model"], entry["harness"]) == (MODELS[0], HARNESSES[0])
    assert entry["matched_steps"] == 2 * plan.repetitions * plan.max_steps


def test_a_prompt_sensitive_model_shows_its_sensitivity(tmp_path):
    plan, records = _run(tmp_path, prompt_sensitive=True)
    report = field_matrix_report(plan, records)
    agreements = [entry["agreement"] for entry in report["decision_sensitivity"]]
    assert all(entry["matched_steps"] >= 3 * plan.repetitions for entry in report["decision_sensitivity"])
    assert min(agreements) < 1.0
    # Divergent decisions move the portfolio, so later observations stop
    # matching and are not compared as if the model had seen the same input.
    assert any(
        entry["matched_steps"] < 3 * plan.repetitions * plan.max_steps
        for entry in report["decision_sensitivity"]
    )
    per_unit = report["decision_sensitivity"][0]["per_unit_agreement"]
    assert per_unit["n"] == 3


# --- plan files and the command line -----------------------------------------


def _plan_payload():
    return {
        "models": [
            {
                "model": model,
                "sampling": {"seed": 10 * (index + 1)},
                "scaffold": scaffold_label(harness, condition),
            }
            for index, model in enumerate(MODELS)
            for harness in ("minimal-stateless-v1",)
            for condition in ("default", "terse-v1", "explicit-v1")
        ],
        "datasets": [{"dataset_id": "calm", "n_symbols": 2, "n_days": 8}],
        "seeds": [1, 2],
        "repetitions": 1,
        "max_steps": 3,
        "matrix": {
            "models": list(MODELS),
            "harnesses": ["minimal-stateless-v1"],
            "conditions": ["default", "terse-v1", "explicit-v1"],
        },
    }


def test_a_plan_file_declares_the_matrix_and_the_report_reads_its_journals(
    tmp_path, capsys
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan_payload()))
    plan = load_plan(plan_path)
    assert plan.matrix == FieldMatrix(
        models=MODELS,
        harnesses=("minimal-stateless-v1",),
        conditions=("default", "terse-v1", "explicit-v1"),
    )
    assert field_main(["--plan", str(plan_path), "--evidence", "unused", "--inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["matrix"] == plan.matrix.record()
    journal = tmp_path / "field.jsonl"
    LocalFieldRunner(StubModel(prompt_sensitive=True)).run(plan, EvidenceJournal(journal))
    assert matrix_main(["--plan", str(plan_path), str(journal)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["plan_sha256"] == plan.plan_sha256
    assert [c["condition"] for c in report["condition_contrasts"]] == [
        "terse-v1",
        "explicit-v1",
    ]


@pytest.mark.parametrize(
    "matrix, match",
    [
        ("models", "matrix must be an object"),
        ({"models": ["a"], "harnesses": ["minimal-stateless-v1"]}, "matrix.conditions"),
        (
            {"models": [], "harnesses": [], "conditions": [], "factors": []},
            "unknown fields",
        ),
    ],
)
def test_a_malformed_matrix_block_is_refused(tmp_path, matrix, match):
    payload = _plan_payload()
    payload["matrix"] = matrix
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_plan(plan_path)


def test_an_unreadable_journal_line_is_refused(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan_payload()))
    journal = tmp_path / "bad.jsonl"
    journal.write_text("\n[1]\n")
    with pytest.raises(FieldMatrixError, match="must be an object"):
        matrix_main(["--plan", str(plan_path), str(journal)])
    journal.write_text("{not json\n")
    with pytest.raises(FieldMatrixError, match="bad.jsonl:1"):
        matrix_main(["--plan", str(plan_path), str(journal)])
