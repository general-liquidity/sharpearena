"""Tests for the diagnostic risk/profit panel on :class:`sharpearena.metrics.RunMetrics`.

Pure-series math — no native binding required. The panel keys are diagnostic only: they
land in ``to_dict`` and never feed the rank key, and ``cost_adjusted_score`` keeps its
prior behavior because the turnover penalty weight defaults to 0.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_PKGDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python", "sharpearena")
)

try:
    from sharpearena.metrics import RunMetrics, cost_adjusted_score
except Exception:  # noqa: BLE001 - binding not built; load the module standalone
    _pkg = types.ModuleType("oo_standalone")
    _pkg.__path__ = [_PKGDIR]  # type: ignore[attr-defined]
    sys.modules.setdefault("oo_standalone", _pkg)
    _spec = importlib.util.spec_from_file_location(
        "oo_standalone.metrics", os.path.join(_PKGDIR, "metrics.py")
    )
    _metrics = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _metrics
    _spec.loader.exec_module(_metrics)
    RunMetrics = _metrics.RunMetrics
    cost_adjusted_score = _metrics.cost_adjusted_score


def test_panel_matches_hand_computed_values():
    m = RunMetrics()
    for r in (0.1, -0.1, 0.2, -0.05):
        m.record_step(reward=r)
    d = m.to_dict()

    assert d["volatility"] == pytest.approx(0.11924240017711822)
    assert d["downside_deviation"] == pytest.approx(0.025)
    assert d["sortino"] == pytest.approx(1.5)
    assert d["var_95"] == pytest.approx(-0.0925)
    assert d["cvar_95"] == pytest.approx(-0.1)
    assert d["tail_ratio"] == pytest.approx(2.0)
    assert d["realized_return"] == pytest.approx(0.026)
    assert d["max_drawdown"] == pytest.approx(0.1)
    assert d["calmar"] == pytest.approx(0.26)


def test_empty_series_panel_is_zero_and_finite():
    d = RunMetrics().to_dict()
    for k in (
        "volatility",
        "downside_deviation",
        "sortino",
        "calmar",
        "var_95",
        "cvar_95",
        "tail_ratio",
        "turnover",
        "cost_drag",
    ):
        assert d[k] == 0.0


def test_guards_zero_drawdown_and_no_downside():
    m = RunMetrics()
    for r in (0.05, 0.1, 0.02):  # monotone up: no negative returns, no drawdown
        m.record_step(reward=r)
    d = m.to_dict()
    assert d["downside_deviation"] == 0.0
    assert d["sortino"] == 0.0  # zero downside deviation → sentinel 0.0
    assert d["max_drawdown"] == 0.0
    assert d["calmar"] == 0.0  # zero drawdown → sentinel 0.0
    assert all(np.isfinite(v) for v in d.values() if isinstance(v, float))


def test_turnover_accumulates_from_weight_sequence():
    m = RunMetrics()
    m.record_step(reward=0.0, weights=[0.5, 0.5])  # vs zeros → 1.0
    m.record_step(reward=0.0, weights=[0.3, 0.7])  # |Δ| = 0.4
    m.record_step(reward=0.0, weights=[0.3, 0.7])  # no change → 0.0
    d = m.to_dict()
    assert d["turnover"] == pytest.approx(1.4)
    assert d["cost_drag"] == pytest.approx(1.4 * 0.001)


def test_weights_optional_does_not_break_existing_callers():
    m = RunMetrics()
    m.record_step(reward=0.1)  # no weights= kwarg
    assert m.to_dict()["turnover"] == 0.0


def test_cost_adjusted_score_unchanged_by_turnover_default():
    composite = {"deflated_sharpe": 2.0}

    no_trade = RunMetrics()
    for _ in range(5):
        no_trade.record_step(reward=0.0, tokens=10)

    churny = RunMetrics()
    for _ in range(5):
        churny.record_step(reward=0.0, tokens=10, weights=[1.0, -1.0])

    # Turnover differs wildly but default weight 0 ⇒ identical cost-adjusted score.
    assert churny.turnover > no_trade.turnover
    assert cost_adjusted_score(composite, churny) == pytest.approx(
        cost_adjusted_score(composite, no_trade)
    )

    # Operator may opt turnover into the penalty; it only ever discounts.
    penalized = cost_adjusted_score(composite, churny, weights={"turnover": 0.5})
    baseline = cost_adjusted_score(composite, churny)
    assert 0.0 < penalized < baseline <= composite["deflated_sharpe"]
