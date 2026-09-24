#!/usr/bin/env python3
"""Import an installed SharpeArena wheel and step one bar through the native binding.

The source suite says nothing about the artifact: the native extension, the type stub
and the package data all have to survive the build and the install before a consumer
sees them. This runs against whatever `sharpearena` the interpreter resolves, so it
belongs in a virtual environment holding the wheel and its declared dependencies only,
started from outside the checkout::

    python -m venv consumer
    ./consumer/bin/pip install sharpearena-*.whl
    cd "$(mktemp -d)" && ./consumer/bin/python .../scripts/check-installed-wheel.py

`--expect-version` compares what the installed distribution reports against the version
the tree declares, so a stale wheel left in a build directory cannot pass for the one
this commit would publish.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version as installed_version
from importlib.resources import files
from pathlib import Path


def assert_import_is_the_installed_package(module) -> None:
    location = Path(module.__file__).resolve()
    if not {"site-packages", "dist-packages"} & set(location.parts):
        raise SystemExit(f"imported {location}, which is not an installed distribution")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        metavar="VERSION",
        default=os.environ.get("SHARPEARENA_EXPECTED_VERSION"),
    )
    arguments = parser.parse_args()

    import sharpearena

    assert_import_is_the_installed_package(sharpearena)

    if arguments.expect_version:
        found = installed_version("sharpearena")
        if found != arguments.expect_version:
            raise SystemExit(
                f"the installed distribution reports {found}, not the "
                f"{arguments.expect_version} this tree declares"
            )

    env = sharpearena.TradingEnv()
    env.reset()
    observation, reward, done, _info = env.step('{"orders": []}')
    if not isinstance(reward, float):
        raise SystemExit(f"step returned a {type(reward).__name__} reward: {reward!r}")
    if done:
        raise SystemExit("the first bar of a fresh episode reported the episode over")
    if not observation:
        raise SystemExit("step returned an empty observation")

    for shipped in ("py.typed", "sharpearena_py.pyi"):
        if not files("sharpearena").joinpath(shipped).is_file():
            raise SystemExit(f"{shipped} did not ship in the wheel")

    print(
        f"imported sharpearena {installed_version('sharpearena')} on "
        f"{'.'.join(str(part) for part in sys.version_info[:3])} and stepped one bar"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
