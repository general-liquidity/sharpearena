"""Run every paper producer regression from the binding crate's canonical suite.

The existing producer tests use synthetic fixtures or frozen-input renders redirected to
temporary directories. This discovery entry point never calls a producer's empirical main
function to regenerate authoritative evidence. Missing imports and skipped tests fail this
entry point instead of silently removing producer coverage from CI.
"""

import importlib.util
import io
from pathlib import Path
import sys
import unittest

import pytest
import sharpearena  # Import installed binding before historical tests alter sys.path.


PAPER_SRC = Path(__file__).resolve().parents[3] / "paper" / "src"
PRODUCER_TESTS = sorted(PAPER_SRC.glob("test_producer_*.py"))


def test_producer_regression_inventory_is_not_empty():
    assert PRODUCER_TESTS, "paper producer regression discovery found no files"
    assert sharpearena.__file__


@pytest.mark.parametrize("path", PRODUCER_TESTS, ids=lambda path: path.stem)
def test_paper_producer_regression_suite_executes(path):
    previous_path = list(sys.path)
    name = "arena_paper_regression_" + path.stem
    try:
        sys.path.insert(0, str(PAPER_SRC))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        expected_cases = suite.countTestCases()
        assert expected_cases > 0, f"{path.name} has no discovered cases"
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        assert result.testsRun == expected_cases, (
            "producer cases were imported but not executed"
        )
        assert not result.skipped, (
            f"producer coverage was skipped: {result.skipped}\n{output.getvalue()}"
        )
        assert result.wasSuccessful(), output.getvalue()
    finally:
        sys.path[:] = previous_path
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "body, message",
    [
        ("self.fail('sentinel producer failure')", "sentinel producer failure"),
        ("self.skipTest('sentinel missing dependency')", "coverage was skipped"),
    ],
)
def test_discovery_entry_point_propagates_failures_and_skips(tmp_path, body, message):
    path = tmp_path / "test_producer_sentinel.py"
    path.write_text(
        "import unittest\nclass Sentinel(unittest.TestCase):\n"
        f"    def test_sentinel(self):\n        {body}\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match=message):
        test_paper_producer_regression_suite_executes(path)
