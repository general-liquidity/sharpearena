#!/usr/bin/env python3
"""Run a learner test file against the installed wheel, refusing to pass by skipping.

The learner tests skip when their extra is absent, which is right in the source suite and
wrong in the cell that installed that extra on purpose: a broken import would turn the
whole file into a green run of nothing. This runs each named file from
`crates/sharpearena-py/tests` and fails on any skip, or when nothing passed.

    python scripts/check-wheel-install.py --wheel "$WHEEL" --extra sb3 --cpu-torch --pytest \
        --run "scripts/run-learner-tests.py test_sb3_learner.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1] / "crates" / "sharpearena-py" / "tests"


class Outcomes:
    def __init__(self) -> None:
        self.passed = 0
        self.skipped: list = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.passed and report.when == "call":
            self.passed += 1
        if report.skipped:
            self.skipped.append(report.nodeid)


def main(names: list) -> int:
    if not names:
        raise SystemExit("name at least one test file")
    import sharpearena

    location = Path(sharpearena.__file__).resolve()
    if not {"site-packages", "dist-packages"} & set(location.parts):
        raise SystemExit(f"imported {location}, which is not an installed distribution")

    outcomes = Outcomes()
    paths = [str(TESTS / name) for name in names]
    status = pytest.main(
        ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *paths],
        plugins=[outcomes],
    )
    if status != 0:
        return int(status)
    if outcomes.skipped:
        raise SystemExit(f"skipped in a cell that installed the extra: {outcomes.skipped}")
    if outcomes.passed == 0:
        raise SystemExit(f"nothing passed in {names}")
    print(f"{outcomes.passed} learner tests passed with no skips: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
