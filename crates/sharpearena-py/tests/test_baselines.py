"""Tests for the baseline policies + leaderboard runner.

When the native binding and numpy are importable these exercise a real (small)
baseline sweep against the SharpeBench kernel. When the binding is absent the file
still imports and the dependency-light logic (markdown rendering, policy shapes) is
checked directly.
"""

import importlib

import pytest

try:
    import numpy as np  # noqa: F401

    from sharpearena import baselines
    from sharpearena.baselines import (
        BASELINE_POLICIES,
        EqualWeightLongPolicy,
        FlatPolicy,
        MomentumPolicy,
        leaderboard_markdown,
        run_baselines,
    )

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)


# -- logic that needs no native binding -------------------------------------


@requires_binding
def test_policies_produce_correct_shape():
    obs = {"closes": np.array([100.0, 101.0, 99.0])}
    for _name, factory in BASELINE_POLICIES:
        action = factory()(obs)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))


@requires_binding
def test_flat_is_zero_and_equal_weight_sums_to_one():
    obs = {"closes": np.array([10.0, 20.0, 30.0, 40.0])}
    assert np.allclose(FlatPolicy()(obs), 0.0)
    ew = EqualWeightLongPolicy()(obs)
    assert np.isclose(ew.sum(), 1.0)
    assert np.all(ew > 0.0)


@requires_binding
def test_momentum_warms_up_then_signs():
    pol = MomentumPolicy()
    # First call has no prior close: warms up to equal weight.
    first = pol({"closes": np.array([100.0, 100.0])})
    assert np.allclose(first, 0.5)
    # Second call differences against the first: one up, one down.
    second = pol({"closes": np.array([101.0, 99.0])})
    assert second[0] > 0.0 and second[1] < 0.0


@requires_binding
def test_leaderboard_renders_sorted_by_deflated_sharpe():
    rows = [
        {"policy": "a", "deflated_sharpe": 0.1, "passed_k_rate": 0.5, "mean_return": 0.01},
        {"policy": "b", "deflated_sharpe": 0.9, "passed_k_rate": 0.8, "mean_return": 0.02},
        {"policy": "c", "deflated_sharpe": 0.4, "passed_k_rate": 0.2, "mean_return": -0.01},
    ]
    md = leaderboard_markdown(rows)
    assert "Deflated Sharpe" in md
    # Highest deflated Sharpe ranks first.
    body = md.splitlines()[2:]
    assert body[0].split("|")[2].strip() == "b"
    assert body[-1].split("|")[2].strip() == "a"


def test_leaderboard_rank_separates_exact_ties_from_rounded_equality():
    """Three causes of "these two look equal" get three different rank cells.

    The scores are the committed Calm row values (``paper/evidence/f1-baselines.json``,
    version 0.9.0): ``kelly_vol_target`` and ``equal_weight_long`` differ in the seventh
    decimal and both print ``1.0000``, while ``momentum`` and ``max_sharpe`` are an exact
    ``0.0`` tie whose order is decided by declaration order alone. ``min_variance`` is
    separated on the printed number and carries no marker, so a renderer that marked
    everything would fail here too.
    """
    rows = [
        {"policy": "kelly_vol_target", "deflated_sharpe": 0.9999999999259135},
        {"policy": "equal_weight_long", "deflated_sharpe": 0.9999998622925645},
        {"policy": "min_variance", "deflated_sharpe": 0.9849005537856994},
        {"policy": "momentum", "deflated_sharpe": 0.0},
        {"policy": "max_sharpe", "deflated_sharpe": 0.0},
    ]
    table = leaderboard_markdown(rows)
    ranks = [
        line.split("|")[1].strip()
        for line in table.splitlines()
        if line.startswith("| ") and not line.startswith("| Rank")
    ]
    assert ranks == ["1~", "2~", "3", "4=", "4="]
    assert "exact tie" in table

    # No marked row, no legend: an unambiguous table renders as it always did.
    plain = leaderboard_markdown(
        [
            {"policy": "a", "deflated_sharpe": 0.9},
            {"policy": "b", "deflated_sharpe": 0.4},
        ]
    )
    assert [line.split("|")[1].strip() for line in plain.splitlines()[2:]] == ["1", "2"]
    assert "exact tie" not in plain


def test_leaderboard_markdown_importable_without_binding():
    # Pure rendering must not require the native kernel.
    from sharpearena.baselines import leaderboard_markdown as render

    md = render([{"policy": "x", "deflated_sharpe": 0.0, "passed_k_rate": 0.0, "mean_return": 0.0}])
    assert md.startswith("| Rank | Policy |")
    assert "| x |" in md


# -- end-to-end sweep against the real kernel -------------------------------


@requires_binding
def test_run_baselines_returns_scored_rows():
    # confidence=False keeps the lean historical row shape.
    rows = run_baselines(n_symbols=3, n_days=40, seeds=range(4), confidence=False)
    assert len(rows) == len(BASELINE_POLICIES) > 1
    names = {r["policy"] for r in rows}
    assert {"flat", "equal_weight_long", "momentum"} <= names
    for r in rows:
        assert set(r) == {"policy", "deflated_sharpe", "passed_k_rate", "mean_return"}
        assert np.isfinite(r["deflated_sharpe"])
        assert 0.0 <= r["passed_k_rate"] <= 1.0
        assert np.isfinite(r["mean_return"])


@requires_binding
def test_run_baselines_attaches_confidence_the_kernel_reproduces_bit_for_bit():
    rows = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    for r in rows:
        assert "per_seed_returns" in r
        ci = r["arena_deflated_sharpe_ci"]
        from sharpearena.confidence import deflated_sharpe_ci
        assert ci == deflated_sharpe_ci(r["per_seed_returns"], len(BASELINE_POLICIES))
        # The pinned SharpeBench 0.21.0 and the Arena estimator share the n-normalized
        # moment convention: exact equality, not a tolerance, is the parity witness.
        assert ci["point"] == r["deflated_sharpe"]
        assert r["deflated_sharpe_ci"] == ci
        assert r["confidence_status"] == "scoring_kernel_reproduced"
        assert ci["lo"] - 1e-9 <= ci["point"] <= ci["hi"] + 1e-9
        assert ci["width"] >= 0.0
        # One return series per seed was retained for the paired test.
        assert len(r["per_seed_returns"]) == 4
    rendered = leaderboard_markdown(rows, show_ci=True)
    assert "unavailable" not in rendered
    for r in rows:
        assert "[{:.4f}, {:.4f}]".format(r["deflated_sharpe_ci"]["lo"], r["deflated_sharpe_ci"]["hi"]) in rendered


@requires_binding
def test_run_baselines_withholds_confidence_when_the_kernel_disagrees(monkeypatch):
    import json

    real = baselines.score_run

    def shifted(returns, n_trials):
        comp = json.loads(real(returns, n_trials))
        comp["deflated_sharpe"] = float(comp["deflated_sharpe"]) + 1e-12
        return json.dumps(comp)

    monkeypatch.setattr(baselines, "score_run", shifted)
    rows = baselines.run_baselines(n_symbols=3, n_days=40, seeds=range(2))
    for r in rows:
        assert r["deflated_sharpe_ci"] is None
        assert r["confidence_status"] == "unavailable_scoring_kernel_mismatch"
        assert r["arena_deflated_sharpe_ci"]["width"] >= 0.0
    rendered = leaderboard_markdown(rows, show_ci=True)
    assert rendered.count("unavailable") == len(rows)
    assert "[0.0000, 0.0000]" not in rendered


@requires_binding
def test_run_baselines_withholds_confidence_when_the_kernel_reports_a_typed_error(monkeypatch):
    import json

    real = baselines.score_run

    def erring(returns, n_trials):
        comp = json.loads(real(returns, n_trials))
        comp["deflation_error"] = "observation 1 must be finite"
        comp["deflated_sharpe"] = 0.0
        return json.dumps(comp)

    monkeypatch.setattr(baselines, "score_run", erring)
    rows = baselines.run_baselines(n_symbols=3, n_days=40, seeds=range(2))
    for r in rows:
        assert r["deflated_sharpe_ci"] is None
        assert r["confidence_status"] == (
            "unavailable_scoring_kernel_error: deflation_error: observation 1 must be finite"
        )
    assert all(line.startswith("| - |") for line in leaderboard_markdown(rows, show_ci=True).splitlines()[2:])


@pytest.mark.parametrize("confidence", [False, True])
@pytest.mark.parametrize("error", ["deflation_error", "bootstrap_error", "selection_error"])
def test_unavailable_baselines_keep_the_reason_without_a_numeric_rank(monkeypatch, confidence, error):
    import json
    from sharpearena.confidence import pairwise_significance

    real = baselines.score_run

    def unavailable(returns, n_trials):
        composite = json.loads(real(returns, n_trials))
        composite[error] = "test computation unavailable"
        composite["deflated_sharpe"] = 0.0
        return json.dumps(composite)

    monkeypatch.setattr(baselines, "score_run", unavailable)
    rows = run_baselines(n_symbols=2, n_days=12, seeds=[0, 1], confidence=confidence)
    for row in rows:
        assert error in row["deflated_sharpe"]
        assert error in row["passed_k_rate"]
    scored = {"policy": "measured", "deflated_sharpe": 0.0}
    table = leaderboard_markdown([*rows, scored], show_ci=confidence)
    assert table.splitlines()[2].startswith("| 1 | measured | 0.0000 |")
    assert all(line.startswith("| - |") for line in table.splitlines()[3:])
    assert error in table
    assert pairwise_significance(rows) == []


def test_absent_confidence_is_not_rendered_as_a_zero_width_interval():
    rows = [{"policy": "x", "deflated_sharpe": 0.5}]
    assert "unavailable" in leaderboard_markdown(rows, show_ci=True)
    rows[0]["deflated_sharpe_ci"] = {"lo": 0.3, "hi": 0.8, "confidence": 0.9}
    assert "[0.3000, 0.8000]" in leaderboard_markdown(rows, show_ci=True)
    assert "90%" in leaderboard_markdown(rows, show_ci=True)
    assert "95%" not in leaderboard_markdown(rows, show_ci=True)


def test_a_single_seed_keeps_its_point_but_withholds_between_seed_confidence():
    rows = run_baselines(n_symbols=2, n_days=12, seeds=[0])
    for row in rows:
        assert isinstance(row["deflated_sharpe"], float)
        assert row["deflated_sharpe_ci"] is None
        assert row["arena_deflated_sharpe_ci"] is None
        assert "two independent seed units" in row["confidence_status"]


def test_duplicate_seed_ids_do_not_supply_independent_units():
    with pytest.raises(ValueError, match="unique"):
        run_baselines(n_symbols=2, n_days=12, seeds=[0, 0])


@pytest.mark.parametrize("interval", [
    {"lo": 0.3, "hi": 0.8},
    {"lo": 0.8, "hi": 0.3, "confidence": 0.9},
    {"lo": 0.3, "hi": float("nan"), "confidence": 0.9},
    {"lo": 0.3, "hi": 0.8, "confidence": True},
    {"lo": -0.3, "hi": 0.8, "confidence": 0.9},
])
def test_baseline_renderer_refuses_invalid_confidence(interval):
    with pytest.raises(ValueError):
        leaderboard_markdown([{"policy": "x", "deflated_sharpe": 0.5,
                               "deflated_sharpe_ci": interval}], show_ci=True)


@requires_binding
def test_run_baselines_is_deterministic():
    a = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    b = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    assert a == b


@requires_binding
def test_run_baselines_distribution_mode_degrades_gracefully():
    # The distribution_mode kwarg is added by a sibling stream; the runner must not
    # crash on any tier whether or not the binding wires it yet.
    for mode in ("calm", "hard", "extreme"):
        rows = run_baselines(n_symbols=3, n_days=30, seeds=range(2), distribution_mode=mode)
        assert len(rows) == len(BASELINE_POLICIES)
