"""Pure publication-boundary checks; no experiment or figure generation."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "confidence_support", Path(__file__).resolve().parents[3] / "paper/src/confidence_support.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("interval", [None, {}, {"point": 0.2, "lo": 0.1, "hi": float("nan")}])
def test_missing_or_invalid_interval_is_not_a_zero_width_bar(interval):
    with pytest.raises(ValueError, match="scoring confidence"):
        module.require_scoring_intervals([{"policy": "x", "deflated_sharpe_ci": interval}])


def test_empty_field_refused_and_real_interval_accepted():
    with pytest.raises(ValueError, match="nonempty"):
        module.require_scoring_intervals([])
    module.require_scoring_intervals([
        {"policy": "x", "deflated_sharpe_ci": {"point": 0.2, "lo": 0.1, "hi": 0.3}}])
