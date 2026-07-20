"""Tests for the leaderboard statistical-confidence layer (bootstrap CI + paired A/B).

These exercise the real native core: the seed-paired bootstrap CI must bracket the same
point deflated Sharpe ``score_run`` reports and widen for a noisier/shorter track, and the
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


@requires_binding
def test_ci_point_matches_score_run_and_brackets():
    per_seed = [_steady_seed(s, 120) for s in range(8)]
    pooled = [x for s in per_seed for x in s]
    declared = 6
    point = json.loads(score_run(pooled, declared))["deflated_sharpe"]
    ci = deflated_sharpe_ci(per_seed, declared)
    # The interval is built around the very number the leaderboard ranks on.
    assert np.isclose(ci["point"], point)
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
    assert "beyond seed noise" in md
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
            }
        ]
    )
    assert md.startswith("| A (ranked above)")
    assert "statistically tied" in md
