"""Typed kernel unavailability propagates through every ``score_run`` consumer.

The helper unit tests need no binding. Each consumer test monkeypatches the module's
``score_run`` to return a composite carrying a typed error beside the kernel's
no-skill floor (the exact shape the pinned SharpeBench 0.21.0 serializes for a
non-finite observation) and asserts the reason surfaces where the consumer routes it:
a recorded reason string in row-shaped outputs, ``KernelScoreUnavailable`` from
selection, ranking-input and producer paths.
"""

from __future__ import annotations

import importlib
import json
import math

import numpy as np
import pytest

from sharpearena.kernel_score import (
    UNAVAILABLE_KERNEL_ERROR,
    UNAVAILABLE_KERNEL_VALUE,
    KernelScoreUnavailable,
    is_kernel_score_unavailable,
    kernel_deflated_sharpe,
    kernel_errors,
    kernel_psr,
    kernel_score_difference,
    kernel_score_or_unavailable,
)

_HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="sharpearena_py native module not built"
)

REASON = "observation 1 must be finite"
VALID = {"deflated_sharpe": 0.61, "psr": 0.73, "passed_k": True, "bootstrap_p": 0.02}
ERROR_KEYS = (
    "deflation_error",
    "bootstrap_error",
    "selection_error",
    "statistics_error",
    "snooping_error",
    "pbo_error",
    "inference_error",
)


def _floored(*keys: str) -> dict:
    """The 0.21.0 shape: floor values beside the named typed error(s)."""
    comp = {"deflated_sharpe": 0.0, "psr": None, "passed_k": False, "bootstrap_p": 1.0}
    for key in keys:
        comp[key] = REASON
    return comp


def _erring_score_run(real):
    """A ``score_run`` stand-in that scores for real, then stamps a typed error."""

    def erring(returns, n_trials=0, *args):
        comp = json.loads(real(returns, n_trials, *args))
        comp["deflation_error"] = REASON
        comp["bootstrap_error"] = REASON
        comp["deflated_sharpe"] = 0.0
        comp["psr"] = None
        return json.dumps(comp)

    return erring


# -- helper -------------------------------------------------------------------


def test_valid_composite_returns_the_kernel_numbers():
    assert kernel_deflated_sharpe(VALID) == 0.61
    assert kernel_psr(VALID) == 0.73
    assert kernel_score_or_unavailable(VALID) == 0.61
    assert kernel_score_or_unavailable(VALID, "psr") == 0.73
    assert kernel_errors(VALID) == {}


@pytest.mark.parametrize("key", ERROR_KEYS)
def test_each_typed_error_key_withholds_the_score(key):
    comp = _floored(key)
    assert kernel_errors(comp) == {key: REASON}
    with pytest.raises(KernelScoreUnavailable) as info:
        kernel_deflated_sharpe(comp)
    assert info.value.errors == {key: REASON}
    assert info.value.reason == f"{UNAVAILABLE_KERNEL_ERROR}: {key}: {REASON}"
    with pytest.raises(KernelScoreUnavailable):
        kernel_psr(comp)
    recorded = kernel_score_or_unavailable(comp)
    assert is_kernel_score_unavailable(recorded)
    assert recorded == info.value.reason


def test_every_present_error_key_is_named_in_the_reason():
    comp = _floored("deflation_error", "bootstrap_error")
    with pytest.raises(KernelScoreUnavailable) as info:
        kernel_deflated_sharpe(comp)
    assert info.value.errors == {"deflation_error": REASON, "bootstrap_error": REASON}
    assert "bootstrap_error" in info.value.reason
    assert "deflation_error" in info.value.reason


def test_error_beats_a_finite_score_and_a_null_field_is_unavailable():
    # A typed error with a finite-looking number is still withheld.
    comp = {**VALID, "deflation_error": REASON}
    with pytest.raises(KernelScoreUnavailable):
        kernel_deflated_sharpe(comp)
    # No typed error but the requested field is null, missing or non-finite.
    for bad in ({"deflated_sharpe": None}, {}, {"deflated_sharpe": float("nan")}):
        with pytest.raises(KernelScoreUnavailable) as info:
            kernel_deflated_sharpe(bad)
        assert info.value.errors == {}
        assert info.value.reason.startswith(UNAVAILABLE_KERNEL_VALUE)
    assert is_kernel_score_unavailable(kernel_score_or_unavailable({"psr": True}, "psr"))


def test_empty_or_null_error_values_do_not_withhold():
    comp = {**VALID, "deflation_error": None, "selection_error": ""}
    assert kernel_errors(comp) == {}
    assert kernel_deflated_sharpe(comp) == 0.61


def test_difference_is_unavailable_when_either_side_is():
    assert kernel_score_difference(0.5, 0.2) == pytest.approx(0.3)
    left = f"{UNAVAILABLE_KERNEL_ERROR}: deflation_error: {REASON}"
    assert kernel_score_difference(left, 0.2) == left
    assert kernel_score_difference(0.5, left) == left
    assert kernel_score_difference(left, "other") == left


def test_unavailable_reason_is_never_a_number():
    reason = kernel_score_or_unavailable(_floored("deflation_error"))
    with pytest.raises(TypeError):
        reason - 0.0  # noqa: B018 - the arithmetic itself is the assertion
    assert not isinstance(reason, (int, float))


# -- consumers ----------------------------------------------------------------


@requires_binding
def test_eval_seeds_row_records_the_reason_and_gate_compares_it(monkeypatch):
    from sharpearena import eval_seeds

    monkeypatch.setattr(eval_seeds, "score_run", _erring_score_run(eval_seeds.score_run))
    out = eval_seeds.evaluate_eval_set(n_symbols=2, n_days=12, max_steps=4, n_trials=1)
    for row in out.values():
        assert is_kernel_score_unavailable(row["deflated_sharpe"])
        assert row["deflated_sharpe"].startswith(UNAVAILABLE_KERNEL_ERROR)
        assert REASON in row["deflated_sharpe"]
        assert row["passed_k"] is False
        assert math.isfinite(row["mean_return"])
    eval_seeds.assert_no_regression(out, out)
    numeric = {name: {**row, "deflated_sharpe": 0.0} for name, row in out.items()}
    with pytest.raises(AssertionError, match="regression at"):
        eval_seeds.assert_no_regression(numeric, out)
    with pytest.raises(AssertionError, match="regression at"):
        eval_seeds.assert_no_regression(out, numeric)


@requires_binding
def test_generalization_rows_and_gaps_carry_the_reason(monkeypatch):
    from sharpearena import generalization
    from sharpearena.gym import SharpeArenaEnv

    monkeypatch.setattr(
        generalization, "score_run", _erring_score_run(generalization.score_run)
    )
    make = lambda s: SharpeArenaEnv(n_symbols=2, n_days=12, seed=s)  # noqa: E731
    gap = generalization.generalization_gap(make, 1, 1, max_steps=4)
    for side in ("train", "test"):
        assert is_kernel_score_unavailable(gap[side]["deflated_sharpe"])
        assert REASON in gap[side]["deflated_sharpe"]
    assert gap["gap_deflated_sharpe"] == gap["train"]["deflated_sharpe"]
    assert math.isfinite(gap["gap_mean_return"])

    make_mode = lambda s, m: SharpeArenaEnv(  # noqa: E731
        n_symbols=2, n_days=12, seed=s, distribution_mode=m
    )
    transfer = generalization.cross_regime_transfer(
        make_mode, "calm", "hard", [0], max_steps=4
    )
    assert is_kernel_score_unavailable(transfer["transfer_gap_deflated_sharpe"])
    assert REASON in transfer["transfer_gap_deflated_sharpe"]


@requires_binding
def test_regime_eval_buckets_record_the_reason_and_radar_refuses_them(monkeypatch):
    from sharpearena import regime_eval
    from sharpearena.gym import SharpeArenaEnv

    monkeypatch.setattr(regime_eval, "score_run", _erring_score_run(regime_eval.score_run))
    policy = lambda obs: np.full(2, 0.5, dtype=np.float32)  # noqa: E731
    out = regime_eval.evaluate_per_regime(
        lambda s: SharpeArenaEnv(n_symbols=2, n_days=12, seed=s), [0], policy, max_steps=6
    )
    assert is_kernel_score_unavailable(out["overall"]["deflated_sharpe"])
    assert REASON in out["overall"]["deflated_sharpe"]
    for bucket in out["per_regime"].values():
        if bucket["n_bars"] >= 2:
            assert is_kernel_score_unavailable(bucket["deflated_sharpe"])
    zero = {"deflated_sharpe": 0.0, "max_drawdown": 0.0}
    base = {"deflated_sharpe": 0.5, "max_drawdown": 0.1}
    panel = {**out["overall"], "max_drawdown": 0.05}
    with pytest.raises(ValueError, match="unavailable"):
        regime_eval.radar_score(panel, zero_anchor=zero, base_anchor=base)


@requires_binding
def test_reward_misspecification_rows_and_gap_carry_the_reason(monkeypatch):
    from sharpearena import reward_misspecification as rm
    from sharpearena.gym import SharpeArenaEnv

    monkeypatch.setattr(rm, "score_run", _erring_score_run(rm.score_run))
    make = lambda s: SharpeArenaEnv(n_symbols=2, n_days=12, seed=s)  # noqa: E731
    gap = rm.misspecification_gap(make, [0], flawed_reward="win_rate", max_steps=4)
    for side in ("clean", "flawed"):
        assert is_kernel_score_unavailable(gap[side]["deflated_sharpe"])
        assert REASON in gap[side]["deflated_sharpe"]
    assert gap["gap_deflated_sharpe"] == gap["clean"]["deflated_sharpe"]
    table = rm.demonstrate_punishment(make, [0], max_steps=4)
    assert all(is_kernel_score_unavailable(r["deflated_sharpe"]) for r in table.values())


@requires_binding
def test_ecology_fitness_refuses_an_unscorable_seat(monkeypatch):
    from sharpearena import ecology

    fake = lambda series, n_trials: json.dumps(_floored("deflation_error"))  # noqa: E731
    monkeypatch.setattr("sharpearena.sharpearena_py.score_run", fake)
    with pytest.raises(KernelScoreUnavailable, match="deflation_error"):
        ecology._episode_fitness([0.01, -0.02, 0.03], "deflated_sharpe", 0)
    assert ecology._episode_fitness([0.01, -0.02, 0.03], "mean_return", 0) != 0.0


@requires_binding
def test_pettingzoo_ranking_places_unscorable_agents_last_with_the_reason(monkeypatch):
    pytest.importorskip("pettingzoo")
    from sharpearena import pettingzoo_env

    real = pettingzoo_env.score_run
    calls = {"n": 0}

    def one_agent_errs(returns, n_trials):
        calls["n"] += 1
        if calls["n"] == 1:
            return _erring_score_run(real)(returns, n_trials)
        return real(returns, n_trials)

    monkeypatch.setattr(pettingzoo_env, "score_run", one_agent_errs)
    env = pettingzoo_env.MultiAgentSharpeArenaEnv(n_agents=3, n_symbols=2, n_days=16, seed=1)
    env.reset(seed=1)
    last = {}
    for _ in range(64):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, _, terms, truncs, last = env.step(actions)
        if all(terms[a] or truncs[a] for a in terms):
            break
    ranking = last[next(iter(last))]["ranking"]
    assert ranking[-1]["agent"] == "agent_0"
    assert is_kernel_score_unavailable(ranking[-1]["deflated_sharpe"])
    assert REASON in ranking[-1]["deflated_sharpe"]
    scored = [r["deflated_sharpe"] for r in ranking[:-1]]
    assert all(isinstance(s, float) for s in scored)
    assert scored == sorted(scored, reverse=True)
    assert [r["rank"] for r in ranking] == [0, 1, 2]
    assert is_kernel_score_unavailable(last["agent_0"]["deflated_sharpe"])
    env.close()


@requires_binding
def test_trace_meta_records_the_reason_beside_the_full_composite(monkeypatch):
    from sharpearena import trace

    fake = lambda rets, n_trials: json.dumps(_floored("bootstrap_error"))  # noqa: E731
    monkeypatch.setattr("sharpearena.sharpearena_py.score_run", fake)
    writer = trace.RolloutTraceWriter(None, n_trials=0)
    for t in range(3):
        writer.record_step(step=t, observation={"x": t}, decision=[0.5], reward=0.01)
    meta = writer.finalize()
    assert meta["scores"]["bootstrap_error"] == REASON
    assert is_kernel_score_unavailable(meta["deflated_sharpe"])
    assert "bootstrap_error" in meta["deflated_sharpe"]
    empty = trace.RolloutTraceWriter(None).finalize()
    assert empty["scores"] == {} and empty["deflated_sharpe"] is None


@requires_binding
def test_verifiers_deflated_sharpe_reward_raises_on_a_withheld_composite(monkeypatch):
    from sharpearena import verifiers_env

    monkeypatch.setattr(
        verifiers_env, "score_run", lambda rets, n: json.dumps(_floored("deflation_error"))
    )
    with pytest.raises(KernelScoreUnavailable, match="deflation_error"):
        verifiers_env.deflated_sharpe_reward(state={"returns": [0.01, 0.02, -0.01]})
    assert verifiers_env.deflated_sharpe_reward(state={"returns": [0.01]}) == 0.0


@requires_binding
def test_local_field_journal_records_the_reason_beside_the_score(tmp_path, monkeypatch):
    from sharpearena import local_agents
    from sharpearena.local_agents import EvidenceJournal, LocalFieldRunner

    from test_local_agents import FixedModel, _plan

    monkeypatch.setattr(
        local_agents, "score_run", _erring_score_run(local_agents.score_run)
    )
    path = tmp_path / "field.jsonl"
    counts = LocalFieldRunner(FixedModel()).run(_plan(repetitions=1), EvidenceJournal(path))
    assert counts["completed"] == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        assert record["score"]["deflation_error"] == REASON
        assert is_kernel_score_unavailable(record["deflated_sharpe"])
        assert REASON in record["deflated_sharpe"]


@requires_binding
def test_strategy_selection_refuses_a_withheld_validation_score(tmp_path, monkeypatch):
    from sharpearena import strategy_generation
    from sharpearena.local_agents import DatasetSpec, ModelRunConfig
    from sharpearena.strategy_generation import StrategySearchPlan, StrategySearchRunner

    from test_strategy_generation import FixtureGenerator

    def fixture_scores(candidates, dataset, seeds, n_trials, max_steps):
        return {
            c.candidate_id: [{"score": _floored("deflation_error"), "seed": s} for s in seeds]
            for c in candidates
        }

    monkeypatch.setattr(strategy_generation, "_evaluate_candidates", fixture_scores)
    plan = StrategySearchPlan(
        model=ModelRunConfig("test-fixture:synthetic"),
        prompt="Generate a small, interpretable family.",
        requested_candidates=4,
        validation_dataset=DatasetSpec("validation", n_days=16),
        test_dataset=DatasetSpec("test", tier="hard", n_days=16),
        validation_seeds=(1, 2),
        test_seeds=(101,),
        max_steps=8,
    )
    path = tmp_path / "selection.jsonl"
    with pytest.raises(KernelScoreUnavailable, match="deflation_error"):
        StrategySearchRunner(FixtureGenerator()).run(plan, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["failure"]["type"] == "KernelScoreUnavailable"
    assert "deflation_error" in rows[-1]["failure"]["detail"]


def test_forecast_curve_refuses_a_withheld_kernel_score(monkeypatch):
    from sharpearena import forecast as forecast_mod
    from sharpearena.kernel_score import KernelScoreUnavailable

    reason = "unavailable_scoring_kernel_error: deflation_error: observation 1 must be finite"
    monkeypatch.setattr(
        forecast_mod,
        "evaluate_seeds",
        lambda *a, **k: {"deflated_sharpe": reason, "passed_k": False, "mean_return": 0.0},
    )
    import pytest

    with pytest.raises(KernelScoreUnavailable) as caught:
        forecast_mod.forecast_skill_curve(lambda s: None, [1], None, r2_grid=(0.5,))
    assert caught.value.reason == reason
