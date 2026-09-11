"""Tests for the leaderboard statistical-confidence layer (bootstrap CI + paired A/B).

These exercise the real native core: the seed-paired bootstrap CI uses Arena's corrected
moment convention and widens for a noisier/shorter track, and the
paired-difference test must flag two close entries as tied while separating a clear skill
gap. When the binding is absent the file still imports and the pure-Python rendering is
checked directly.
"""

import importlib
import json
import math

import pytest

try:
    import numpy as np  # noqa: F401

    from sharpearena.confidence import (
        deflated_sharpe_ci,
        paired_dsr_diff,
        pairwise_significance,
        significance_markdown,
    )
    from sharpearena.sharpearena_py import score_run

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)


def _steady_seed(offset: int, n: int) -> list[float]:
    """A low-vol, positive-drift deterministic track (stable, high deflated Sharpe)."""
    return [0.001 + 0.0002 * (((i + offset) % 7) - 3) for i in range(n)]


def _sharpe_series(target: float, n: int, phase: float) -> list[float]:
    """A deterministic track with a prescribed per-period Sharpe (mean/std == target)."""
    base = [math.sin(i + phase) for i in range(n)]
    m = sum(base) / n
    sd = (sum((x - m) ** 2 for x in base) / (n - 1)) ** 0.5
    scale = 0.01
    return [target * scale + ((x - m) / sd) * scale for x in base]


# -- bootstrap CI -----------------------------------------------------------

@pytest.mark.parametrize("n_boot", [0, 1])
def test_seed_bootstrap_requires_a_resampling_distribution(n_boot):
    rows = [[0.01, 0.02], [-0.01, 0.01]]
    with pytest.raises(ValueError):
        deflated_sharpe_ci(rows, n_boot=n_boot)
    with pytest.raises(ValueError):
        paired_dsr_diff(rows, rows, n_boot=n_boot)


@pytest.mark.parametrize("rows", [[], [[0.01, 0.02]], [[], [0.01, 0.02]],
                                  [[float("nan"), 0.01], [0.01, 0.02]],
                                  [[1e308, 1e308], [0.01, 0.02]]])
def test_seed_bootstrap_requires_usable_independent_units(rows):
    with pytest.raises(ValueError):
        deflated_sharpe_ci(rows, n_boot=10)
    with pytest.raises(ValueError):
        paired_dsr_diff(rows, rows, n_boot=10)


def test_paired_seed_bootstrap_refuses_prefix_truncation():
    rows = [[0.01, 0.02], [-0.01, 0.01]]
    with pytest.raises(ValueError):
        paired_dsr_diff(rows, rows + [[0.02, 0.03]], n_boot=10)


@pytest.mark.parametrize("kwargs", [{"n_boot": 2.9}, {"n_trials": True},
                                    {"resample_seed": 1.5}, {"alpha": True},
                                    {"alpha": 0.0}, {"alpha": 1.0},
                                    {"alpha": float("nan")},
                                    {"periods_per_year": 0.0},
                                    {"periods_per_year": -252.0},
                                    {"periods_per_year": float("nan")},
                                    {"periods_per_year": float("inf")},
                                    {"periods_per_year": True}])
def test_confidence_parameters_are_validated_without_coercion(kwargs):
    rows = [[0.01, 0.02], [-0.01, 0.01]]
    with pytest.raises(ValueError):
        deflated_sharpe_ci(rows, **kwargs)
    with pytest.raises(ValueError):
        paired_dsr_diff(rows, rows, **kwargs)


@requires_binding
@pytest.mark.parametrize("periods_per_year", [0.0, -252.0, float("nan"), float("inf")])
def test_native_binding_refuses_invalid_periods_per_year(periods_per_year):
    from sharpearena import sharpearena_py as native

    rows = [[0.01, 0.02], [-0.01, 0.01]]
    with pytest.raises(ValueError, match="periods_per_year"):
        native.bootstrap_dsr_ci(rows, 0, 10, 1, 0.05, periods_per_year)
    with pytest.raises(ValueError, match="periods_per_year"):
        native.paired_dsr_diff(rows, rows, 0, 10, 1, 0.05, periods_per_year)


@requires_binding
def test_periods_per_year_moves_the_deflation_bar_with_the_kernel():
    # Per-period Sharpe 0.08 over 480 bars sits between the daily bar (about 0.073 per
    # period at 56 trials) and the weekly one (about 0.161), so the DSR is unsaturated.
    per_seed = [_sharpe_series(0.08, 40, s) for s in range(12)]
    pooled = [x for s in per_seed for x in s]
    daily = deflated_sharpe_ci(per_seed, 6)
    assert daily == deflated_sharpe_ci(per_seed, 6, periods_per_year=252.0)
    weekly = deflated_sharpe_ci(per_seed, 6, periods_per_year=52.0)
    # The annualized prior carries more dispersion into each weekly period, so the same
    # per-period track clears a higher bar and deflates harder.
    assert 0.0 < weekly["point"] < daily["point"] < 1.0
    for rate, ci in ((252.0, daily), (52.0, weekly)):
        assert ci["point"] == json.loads(score_run(pooled, 6, rate))["deflated_sharpe"]


@requires_binding
def test_python_bootstrap_defaults_are_the_native_defaults():
    """The four ``confidence`` defaults restate pyo3 signature defaults; drive the pair.

    ``DEFAULT_N_BOOT``, ``DEFAULT_RESAMPLE_SEED``, ``DEFAULT_ALPHA`` and
    ``DEFAULT_PERIODS_PER_YEAR`` are written once in ``confidence.py`` and once more in
    the ``#[pyo3(signature = ...)]`` defaults of ``bootstrap_dsr_ci`` / ``paired_dsr_diff``.
    Asserting one literal against the other passes for any value written on both sides.
    This calls the native function with the arguments omitted, so the engine supplies its
    own defaults, and compares against the Python constants passed explicitly. The
    perturbation leg establishes that each argument moves this fixture, so agreement is
    evidence of a shared value rather than of an inert parameter.
    """
    from sharpearena.confidence import (
        DEFAULT_ALPHA,
        DEFAULT_N_BOOT,
        DEFAULT_PERIODS_PER_YEAR,
        DEFAULT_RESAMPLE_SEED,
    )
    from sharpearena.sharpearena_py import bootstrap_dsr_ci as native_bootstrap_dsr_ci
    from sharpearena.sharpearena_py import paired_dsr_diff as native_paired_dsr_diff

    per_seed = [_sharpe_series(0.08, 40, s) for s in range(12)]
    other = [_sharpe_series(0.05, 40, s + 0.5) for s in range(12)]
    stated = (DEFAULT_N_BOOT, DEFAULT_RESAMPLE_SEED, DEFAULT_ALPHA, DEFAULT_PERIODS_PER_YEAR)

    assert json.loads(native_bootstrap_dsr_ci(per_seed, 6)) == json.loads(
        native_bootstrap_dsr_ci(per_seed, 6, *stated)
    )
    assert json.loads(native_paired_dsr_diff(per_seed, other, 6)) == json.loads(
        native_paired_dsr_diff(per_seed, other, 6, *stated)
    )

    # Every one of the four moves this fixture, so the agreement above is not vacuous.
    for index, moved in enumerate((DEFAULT_N_BOOT + 500, DEFAULT_RESAMPLE_SEED + 1,
                                   DEFAULT_ALPHA * 2, DEFAULT_PERIODS_PER_YEAR / 2)):
        perturbed = list(stated)
        perturbed[index] = moved
        assert json.loads(native_bootstrap_dsr_ci(per_seed, 6, *perturbed)) != json.loads(
            native_bootstrap_dsr_ci(per_seed, 6, *stated)
        ), f"argument {index} does not move the fixture, so it cannot witness its default"


@requires_binding
def test_strong_positive_fixture_saturates_both_estimators():
    per_seed = [_steady_seed(s, 120) for s in range(8)]
    pooled = [x for s in per_seed for x in s]
    declared = 6
    point = json.loads(score_run(pooled, declared))["deflated_sharpe"]
    ci = deflated_sharpe_ci(per_seed, declared)
    # Saturation is not cross-kernel parity evidence. The baseline test uses
    # an unsaturated fixture to expose the known old/new moment mismatch.
    assert ci["point"] == point == 1.0
    assert ci["lo"] - 1e-9 <= ci["point"] <= ci["hi"] + 1e-9
    assert ci["width"] >= 0.0
    assert ci["confidence"] == pytest.approx(0.95)


@requires_binding
def test_ci_is_wider_for_a_noisier_shorter_track():
    # Both tracks sit in the DSR's sensitive band. The stable entry is many long seeds with
    # tightly clustered per-seed Sharpe; the noisy entry is a few short seeds with widely
    # dispersed Sharpe, so the resample composition swings the number.
    stable = [_sharpe_series(1.16 + 0.01 * (s % 3), 40, s) for s in range(12)]
    noisy = [_sharpe_series(t, 16, i) for i, t in enumerate([0.6, 1.16, 1.8])]
    stable_ci = deflated_sharpe_ci(stable, 6)
    noisy_ci = deflated_sharpe_ci(noisy, 6)
    assert noisy_ci["width"] > stable_ci["width"]


@requires_binding
def test_ci_is_deterministic_in_resample_seed():
    per_seed = [_steady_seed(s, 100) for s in range(8)]
    a = deflated_sharpe_ci(per_seed, 6, resample_seed=42)
    b = deflated_sharpe_ci(per_seed, 6, resample_seed=42)
    assert a == b


# -- paired significance ----------------------------------------------------


@requires_binding
def test_paired_flags_close_entries_as_tied():
    a = [_steady_seed(s, 120) for s in range(8)]
    # b differs from a by a tiny, sign-alternating per-seed margin: within seed noise.
    b = [
        [r + (0.00003 if s % 2 == 0 else -0.00003) for r in _steady_seed(s, 120)]
        for s in range(8)
    ]
    d = paired_dsr_diff(a, b)
    assert not d["significant"]
    assert d["verdict"] == "tied"
    assert d["lo"] <= 0.0 <= d["hi"]


@requires_binding
def test_paired_separates_clearly_different_skill():
    a = [_steady_seed(s, 120) for s in range(8)]
    b = [[-r for r in _steady_seed(s, 120)] for s in range(8)]
    d = paired_dsr_diff(a, b)
    assert d["significant"]
    assert d["verdict"] == "a_better"
    assert d["lo"] > 0.0
    assert d["p_value"] < 0.05


@requires_binding
def test_pairwise_significance_over_leaderboard_rows():
    winner = [_steady_seed(s, 120) for s in range(8)]
    loser = [[-r for r in _steady_seed(s, 120)] for s in range(8)]
    rows = [
        {"policy": "winner", "deflated_sharpe": 1.0, "per_seed_returns": winner},
        {"policy": "loser", "deflated_sharpe": 0.0, "per_seed_returns": loser},
    ]
    comps = pairwise_significance(rows)
    assert len(comps) == 1
    assert comps[0]["a"] == "winner" and comps[0]["b"] == "loser"
    assert comps[0]["significant"]
    md = significance_markdown(comps)
    assert "winner > loser (CI excludes zero)" in md
    assert "Verdict" in md


def test_significance_markdown_importable_without_binding():
    from sharpearena.confidence import significance_markdown as render

    md = render(
        [
            {
                "a": "x",
                "b": "y",
                "point_diff": 0.0,
                "lo": -0.1,
                "hi": 0.1,
                "p_value": 1.0,
                "significant": False,
                "verdict": "tied",
                "confidence": 0.9,
            }
        ]
    )
    assert md.startswith("| A | B |")
    assert "difference not established" in md
    assert "statistically tied" not in md
    assert "90%" in md and "95%" not in md


def _comparison(**changes):
    return dict(a="alpha", b="beta", point_diff=-0.2, lo=-0.3, hi=-0.1,
                p_value=0.01, significant=True, verdict="b_better",
                confidence=0.9, **changes)


def test_renderer_names_actual_winner_and_each_confidence_level():
    a = _comparison()
    b = {**a, "point_diff": 0.2, "lo": 0.1, "hi": 0.3,
         "verdict": "a_better", "confidence": 0.99}
    md = significance_markdown([a, b])
    lines = md.splitlines()
    assert "beta > alpha (CI excludes zero)" in lines[2]
    assert "90%" in lines[2]
    assert "alpha > beta (CI excludes zero)" in lines[3]
    assert "99%" in lines[3]
    assert "95%" not in md


@pytest.mark.parametrize("changes", [
    {"verdict": "a_better"}, {"significant": False}, {"point_diff": 0.2},
    {"verdict": "unknown"}, {"significant": "false"}, {"confidence": 1.0},
    {"confidence": 0.0}, {"confidence": float("nan")}, {"p_value": -0.1},
    {"p_value": 1.1}, {"point_diff": float("inf")}, {"lo": 0.3},
    {"hi": float("nan")}, {"hi": True},
])
def test_renderer_refuses_inconsistent_or_nonfinite_comparisons(changes):
    with pytest.raises(ValueError):
        significance_markdown([{**_comparison(), **changes}])


@pytest.mark.parametrize("key", ["confidence", "verdict", "point_diff", "lo", "hi",
                                "significant", "p_value", "a", "b"])
def test_renderer_does_not_default_missing_evidence(key):
    comparison = _comparison()
    del comparison[key]
    with pytest.raises(ValueError):
        significance_markdown([comparison])
