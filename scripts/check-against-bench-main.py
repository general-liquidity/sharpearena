#!/usr/bin/env python3
"""Build and test SharpeArena against an unreleased SharpeBench checkout.

Arena pins exact published SharpeBench crates, so its own CI cannot see a change
on Bench main until a release. This overlays a Bench checkout with Cargo
`[patch.crates-io]` config on the command line and edits no tracked file.

    python scripts/check-against-bench-main.py --bench ../sharpebench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def published_crates(bench: Path) -> dict[str, Path]:
    """Every SharpeBench library crate a patch can stand in for."""
    crates: dict[str, Path] = {}
    for manifest in sorted((bench / "crates").glob("*/Cargo.toml")):
        package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
        name = package.get("name", "")
        if name.startswith("sharpebench") and name != "sharpebench-py":
            crates[name] = manifest.parent.resolve()
    if "sharpebench-sim" not in crates:
        raise SystemExit(f"{bench} does not look like a SharpeBench checkout")
    return crates


def patch_flags(crates: dict[str, Path]) -> list[str]:
    flags: list[str] = []
    for name, path in crates.items():
        flags += ["--config", f'patch.crates-io.{name}.path="{path.as_posix()}"']
    return flags


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true", help="compile only")
    args = parser.parse_args()
    flags = patch_flags(published_crates(args.bench.resolve()))
    python_crate = ROOT / "crates" / "sharpearena-py"
    run(["cargo", "check", *flags, "--workspace", "--all-targets"], ROOT)
    run(["cargo", "check", *flags, "--all-targets"], python_crate)
    if not args.skip_tests:
        run(["cargo", "test", *flags, "--workspace"], ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
