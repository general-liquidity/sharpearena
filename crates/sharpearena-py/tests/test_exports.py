"""Tests that the top-level package re-exports its full public surface.

Guards against the drift where a symbol exists in a submodule but was never
added to ``sharpearena/__init__.py`` (import + ``__all__``).
"""

import importlib

import pytest

try:
    import sharpearena

    _HAVE_BINDING = importlib.util.find_spec("sharpearena.sharpearena_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native sharpearena binding not built"
)


NEWLY_EXPORTED = [
    # ecology
    "classify_outcomes",
    # deferred
    "ClaimRejected",
    "claims_from_json",
    "score_claim",
    # rewards
    "risk_aware",
    "time_inhomogeneous_vol_aversion",
    "time_aversion_schedule",
    # baselines
    "BEHAVIORAL_POLICIES",
    "DispositionEffectPolicy",
    "OverconfidentPolicy",
    # repaired realism / endogenous adverse selection
    "CALM_CALIBRATION_CANDIDATE_KNOBS",
    "EndogenousImpact",
    "compare_endogenous_arms",
    "informed_displacement",
    # market_making
    "closed_form_reference_policy",
    # local-model field, observed-trial strategy search, and paper-only forward arm
    "LocalFieldRunner",
    "OllamaClient",
    "StrategySearchRunner",
    "PaperRiskGuard",
    "PaperTradingSession",
    "prepare_forward_window_commitment",
]


@requires_binding
@pytest.mark.parametrize("name", NEWLY_EXPORTED)
def test_symbol_importable_from_top_level(name):
    assert hasattr(sharpearena, name)


@requires_binding
@pytest.mark.parametrize("name", NEWLY_EXPORTED)
def test_symbol_listed_in_all(name):
    assert name in sharpearena.__all__


@requires_binding
def test_all_has_no_duplicates():
    dupes = [n for n in set(sharpearena.__all__) if sharpearena.__all__.count(n) > 1]
    assert dupes == []
