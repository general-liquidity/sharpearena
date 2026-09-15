"""A constant return track has no Sharpe ratio, so nothing reports one for it.

Paper audit 2026-09-14. The ``flat`` reference policy never trades, so its return
track is identically zero and its Sharpe ratio is 0/0. The pinned SharpeBench 0.21.0
kernel scored it anyway (PSR 0.5000000005, a deflated Sharpe that moves with the
deflation bar), and scored a constant nonzero track, whose rounded mean leaves a
computed standard deviation near 1e-18, at PSR and deflated Sharpe 1.0 with a per-run
pass and rank eligibility. Arena's own seed bootstrap did the same. ``score_run`` now
withholds such a track with a typed ``deflation_error`` and the seed bootstrap refuses
it, both with the reason SharpeBench uses for the same track, and every consumer below
reports that reason instead of a number.

Each regression asserts the exact reason, because a floor, a missing interval or an
ineligible row also follow from other causes. The controls pin that a dispersed track,
however low its volatility, scores as it did before.
"""

from __future__ import annotations

import importlib
import json
import math

import pytest

from sharpearena.kernel_score import (
    KernelScoreUnavailable,
    kernel_deflated_sharpe,
    kernel_psr,
    kernel_score_or_unavailable,
)

_HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
pytestmark = pytest.mark.skipif(not _HAVE_BINDING, reason="native sharpearena binding not built")

CONSTANT = "returns must not be constant: a constant series has no Sharpe ratio"
NON_FINITE = "Sharpe ratio is not finite"
WITHHELD = f"unavailable_scoring_kernel_error: deflation_error: {CONSTANT}"


def _sine(n: int, drift: float, amp: float, phase: float = 0.0) -> list[float]:
    return [drift + amp * math.sin(i * 0.9 + phase) for i in range(n)]


def _score(returns, n_trials=6):
    from sharpearena.sharpearena_py import score_run

    return json.loads(score_run(returns, n_trials))


@pytest.mark.parametrize("value", [0.0, 0.001, -0.002])
@pytest.mark.parametrize("length", [2, 120, 408])
def test_score_run_withholds_a_constant_track_with_the_constant_track_reason(value, length):
    composite = _score([value] * length)
    assert composite["deflation_error"] == CONSTANT
    assert "bootstrap_error" not in composite
    assert kernel_score_or_unavailable(composite) == WITHHELD
    assert kernel_score_or_unavailable(composite, "psr") == WITHHELD
    for read in (kernel_deflated_sharpe, kernel_psr):
        with pytest.raises(KernelScoreUnavailable, match=CONSTANT):
            read(composite)
    assert composite["passed_k"] is False and composite["rank_eligible"] is False
    assert (composite["deflated_sharpe"], composite["psr"], composite["composite"]) == (0.0, 0.0, 0.0)
    for absent in ("dsr_ci_low", "dsr_ci_high", "dsr_se"):
        assert absent not in composite
    assert composite["rolling_min_sharpe"] is None and composite["rolling_frac_positive"] is None


def test_score_run_withholds_the_committed_f1_flat_track():
    # The pooled F1 `flat` track: 16 seeds of 120 zero returns, six declared trials.
    # The committed evidence prints 0.000686601376451601 for it.
    composite = _score([0.0] * 1920)
    assert composite["deflation_error"] == CONSTANT
    assert kernel_score_or_unavailable(composite) == WITHHELD


def test_score_run_withholds_an_underflowing_track_as_a_non_finite_sharpe():
    composite = _score([0.0, 1e-170] * 30)
    assert composite["deflation_error"] == NON_FINITE
    assert kernel_score_or_unavailable(composite) == (
        f"unavailable_scoring_kernel_error: deflation_error: {NON_FINITE}"
    )


def test_a_track_with_a_sharpe_ratio_keeps_its_recorded_score():
    # Recorded from SharpeArena 0.29.0 before this change, on Windows; compared within a
    # few ULPs, as platform libm results differ in the last bits.
    for track, dsr, psr in (
        (_sine(120, 0.001, 0.02, 1.0), 0.5469037468423354, 0.8195866610339282),
        ([0.0] * 119 + [1e-9], 0.6527889836061004, 0.9754070893697193),
    ):
        composite = _score(track)
        assert "deflation_error" not in composite
        assert math.isclose(kernel_deflated_sharpe(composite), dsr, rel_tol=1e-14, abs_tol=0.0)
        assert math.isclose(kernel_psr(composite), psr, rel_tol=1e-14, abs_tol=0.0)
    # Low volatility is not constancy: the same track scaled by 1e-9 is still scored.
    scaled = _score([r * 1e-9 for r in _sine(120, 0.001, 0.02, 1.0)])
    assert "deflation_error" not in scaled
    assert math.isclose(kernel_deflated_sharpe(scaled), 0.5469037468423354, rel_tol=1e-9)
    tiny = _score([0.001 + 1e-12 * ((i % 5) - 2) for i in range(250)])
    assert "deflation_error" not in tiny and kernel_deflated_sharpe(tiny) == 1.0


def test_run_baselines_withholds_flat_and_leaves_every_other_row():
    from sharpearena.baselines import leaderboard_markdown, run_baselines
    from sharpearena.confidence import pairwise_significance

    rows = {r["policy"]: r for r in run_baselines(seeds=range(4), n_boot=200)}
    flat = rows["flat"]
    assert flat["deflated_sharpe"] == WITHHELD
    assert flat["passed_k_rate"] == WITHHELD
    assert flat["confidence_status"] == WITHHELD
    assert flat["deflated_sharpe_ci"] is None and flat["arena_deflated_sharpe_ci"] is None
    assert flat["mean_return"] == 0.0
    # Recorded from SharpeArena 0.29.0 before this change (same seeds, n_boot=200).
    recorded = {
        "equal_weight_long": (0.9999999999999993, 1.0),
        "momentum": (0.0, 0.0),
        "min_variance": (0.9999999999971866, 1.0),
        "max_sharpe": (0.011337902704167457, 0.25),
        "kelly_vol_target": (0.9998911010023319, 0.75),
    }
    for policy, (dsr, rate) in recorded.items():
        row = rows[policy]
        assert row["confidence_status"] == "scoring_kernel_reproduced", policy
        assert math.isclose(row["deflated_sharpe"], dsr, rel_tol=1e-12, abs_tol=1e-300), policy
        assert row["passed_k_rate"] == rate, policy
    table = leaderboard_markdown(list(rows.values()), show_ci=True)
    body = [line for line in table.splitlines()[2:] if line.startswith("| ")]
    flat_line = body[-1]
    assert "| flat |" in flat_line, table
    assert flat_line.count(WITHHELD) == 2 and "| unavailable |" in flat_line
    comparisons = pairwise_significance(list(rows.values()), n_boot=50)
    assert comparisons and all("flat" not in (c["a"], c["b"]) for c in comparisons)


def test_confidence_refuses_a_constant_band_with_the_constant_track_reason():
    from sharpearena.confidence import deflated_sharpe_ci, paired_dsr_diff

    dispersed = [_sine(60, 0.0005 * (k + 1), 0.01, k) for k in range(6)]
    for value in (0.0, 0.001):
        band = [[value] * 120 for _ in range(16)]
        with pytest.raises(ValueError, match=CONSTANT):
            deflated_sharpe_ci(band, 6)
        with pytest.raises(ValueError, match=CONSTANT):
            paired_dsr_diff(dispersed, [row[:60] for row in band[:6]], 6, n_boot=50)
    # Control, recorded before this change: a band that disperses keeps its interval to
    # the bit on this platform even though two of its seeds are constant.
    mixed = [[0.0] * 60, [0.0] * 60] + [_sine(60, 0.0005, 0.01, k) for k in range(4)]
    ci = deflated_sharpe_ci(mixed, 6, n_boot=500, resample_seed=0x00C1)
    for key, value in (("point", 0.4178323955371508), ("lo", 0.1690915940448645),
                       ("hi", 0.6700303951247865)):
        assert math.isclose(ci[key], value, rel_tol=1e-13, abs_tol=0.0), key


def test_generalization_rows_record_the_constant_track_reason():
    from sharpearena.baselines import FlatPolicy
    from sharpearena.generalization import evaluate_seeds
    from sharpearena.gym import SharpeArenaEnv

    result = evaluate_seeds(
        lambda seed: SharpeArenaEnv(n_symbols=3, n_days=60, seed=seed), [0, 1], FlatPolicy()
    )
    assert result["deflated_sharpe"] == WITHHELD
    # A plain tally of the kernel's per-run gate, which a run with no Sharpe ratio fails.
    assert result["passed_k_rate"] == 0.0


def test_verifiers_rewards_and_trace_meta_withhold_a_constant_rollout():
    from sharpearena import trace, verifiers_env

    state = {"returns": [0.0] * 12}
    with pytest.raises(KernelScoreUnavailable, match=CONSTANT):
        verifiers_env.deflated_sharpe_reward(state=state)
    assert verifiers_env.pass_k_reward(state=state) == 0.0

    writer = trace.RolloutTraceWriter(None, n_trials=0)
    for t in range(5):
        writer.record_step(step=t, observation={"x": t}, decision=[0.0], reward=0.001)
    meta = writer.finalize()
    assert meta["scores"]["deflation_error"] == CONSTANT
    assert meta["deflated_sharpe"] == WITHHELD
