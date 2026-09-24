#!/usr/bin/env python3
"""Check the declared optional extras against an installed wheel.

Three modes, all driven by the registry in `scripts/optional_extras.py`, which is
validated against `crates/sharpearena-py/pyproject.toml` on every run:

``--emit-matrix``
    Print the declared extras and the import names they bring, as JSON, and write them
    to ``$GITHUB_OUTPUT`` when CI set it. The extras leg of the compatibility matrix is
    generated from this, so a newly declared extra gets a cell without anyone editing
    the workflow.

``--extra NAME``
    Assert the extra's dependencies are importable, then run the real adapter behind it
    against the installed package: a PettingZoo tournament, a Minari export, the MCP
    tool list, the verifiers environment build. No model call, no network, no window.

``--none``
    Assert that nothing any extra would install is importable, that the base package
    imports and steps, that every guarded adapter module still imports, and that each
    guard refuses by name rather than dying on an ``ImportError``. This is the check
    that makes "every adapter is guarded" a tested claim.

``--extra`` and ``--none`` must run from outside the checkout, in a virtual environment
holding the installed wheel, or the import resolves to the source tree and proves
nothing about the artifact::

    python -m venv consumer
    ./consumer/bin/pip install "sharpearena-0.31.0-cp312-cp312-linux_x86_64.whl[minari]"
    ./consumer/bin/python scripts/check-optional-extras.py --extra minari
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optional_extras import (  # noqa: E402
    ACCELERATOR_IMPORTS,
    EXERCISES,
    GUARDED_MODULES,
    GUARDS,
    extra_import_names,
    optional_import_names,
    validate_coverage,
)


def assert_import_is_the_installed_package() -> None:
    """A source checkout on `sys.path`, or a `maturin develop` `.pth` entry, would leave
    the packaged surface unchecked. An installed distribution lives under the
    interpreter's own package directory."""
    import sharpearena

    location = Path(sharpearena.__file__).resolve()
    if not {"site-packages", "dist-packages"} & set(location.parts):
        raise SystemExit(f"imported {location}, which is not an installed distribution")


def importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run_extra(extra: str) -> int:
    if extra not in EXERCISES:
        raise SystemExit(f"{extra!r} is not a declared extra")
    missing = [name for name in extra_import_names(extra) if not importable(name)]
    if missing:
        raise SystemExit(
            f"the {extra} extra is supposed to be installed here, but "
            f"{', '.join(missing)} is not importable"
        )
    assert_import_is_the_installed_package()
    print(f"[{extra}] {EXERCISES[extra]()}")
    return 0


def run_none() -> int:
    present = [
        name
        for name in list(optional_import_names()) + list(ACCELERATOR_IMPORTS)
        if importable(name)
    ]
    if present:
        raise SystemExit(
            f"{', '.join(present)} is importable; this check has to run in an "
            "environment holding the wheel and its declared dependencies only"
        )
    assert_import_is_the_installed_package()

    for module in GUARDED_MODULES:
        importlib.import_module(module)
    print(f"[none] every guarded adapter module imports: {', '.join(GUARDED_MODULES)}")

    import sharpearena

    env = sharpearena.TradingEnv()
    env.reset()
    _observation, reward, done, _info = env.step('{"orders": []}')
    if not isinstance(reward, float) or done:
        raise SystemExit(f"the base environment did not step: reward={reward}, done={done}")
    print("[none] the base environment reset and stepped with no extra installed")

    for extra in sorted(GUARDS):
        for line in GUARDS[extra]():
            print(f"[none] {line}")
    return 0


def emit_matrix() -> int:
    extras = validate_coverage()
    payload = {"extras": extras, "optional_imports": optional_import_names()}
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"extras={json.dumps(payload['extras'])}\n")
            handle.write(f"optional_imports={' '.join(payload['optional_imports'])}\n")
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-matrix", action="store_true")
    group.add_argument("--extra", metavar="NAME")
    group.add_argument("--none", action="store_true")
    arguments = parser.parse_args()

    validate_coverage()
    if arguments.emit_matrix:
        return emit_matrix()
    if arguments.extra:
        return run_extra(arguments.extra)
    return run_none()


if __name__ == "__main__":
    sys.exit(main())
