#!/usr/bin/env python3
"""Build and test SharpeArena against an unreleased SharpeBench checkout.

Arena pins exact published SharpeBench crates, so its own CI cannot see a change
on Bench main until a release. This overlays a Bench checkout with Cargo
`[patch.crates-io]` config on the command line. The patched build rewrites both
lockfiles, so they are restored byte for byte when the run ends, pass or fail.
Before building, it checks that every top-level `CompositeScore` key of the
checkout that ends in `_error` or `_refusal` is classified by Arena's kernel
reader, so a new kernel field is named here rather than at the pin bump.

    python scripts/check-against-bench-main.py --bench ../sharpebench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_compat import (  # noqa: E402
    kernel_key_problems,
    lockfiles_preserved,
    patch_flags,
    published_crates,
)

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true", help="compile only")
    args = parser.parse_args()
    bench = args.bench.resolve()
    crates = published_crates(bench)
    problems = kernel_key_problems(crates["sharpebench-core"] / "src" / "composite.rs")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    flags = patch_flags(crates)
    python_crate = ROOT / "crates" / "sharpearena-py"
    with lockfiles_preserved(ROOT):
        run(["cargo", "check", *flags, "--workspace", "--all-targets"], ROOT)
        run(["cargo", "check", *flags, "--all-targets"], python_crate)
        if not args.skip_tests:
            run(["cargo", "test", *flags, "--workspace"], ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
