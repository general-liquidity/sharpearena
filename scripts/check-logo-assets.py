#!/usr/bin/env python3
"""Check the vendored third-party logos against the hashes their record claims.

`docs/assets/logos/README.md` states that every file in that directory is byte-identical
to what its project publishes, records a SHA-256 for each, and tells a reader to replace
the file and update the hash rather than edit it. Until this script existed the hashes
were a table nothing read: a logo could be recoloured, cropped or rescaled and the page
would go on asserting it had not been.

Three properties, all offline:

  1. Every asset in `docs/assets/logos/` has a recorded hash.
  2. Every recorded hash has an asset.
  3. Every asset hashes to what the record says.

The record is the pinned reference and this check never fetches upstream. That is a
deliberate trade and worth being exact about what it buys. It proves the committed bytes
have not drifted since they were vendored, which is the property the README asserts and
the one a supply-chain gate can enforce in CI. It does not prove the bytes match what
upstream serves today: upstream can change its logo, and the README already says a
mismatch found by hand against upstream means the project changed its mark, not that
this copy is wrong. Putting a network fetch in CI would make the gate fail on someone
else's rebrand, on their outage and on their rate limit, which is how a gate teaches
people to ignore it.

Usage:

    python scripts/check-logo-assets.py [--root PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

LOGO_DIR = "docs/assets/logos"
RECORD = f"{LOGO_DIR}/README.md"

# The hashes live in a fenced block of `<sha256>  <filename>` lines, the shape
# `sha256sum` prints, so the record stays copy-pasteable against the command the README
# tells a reader to run.
HASH_LINE = re.compile(r"^([0-9a-f]{64})\s+(\S+)$")
FENCE = re.compile(r"^\s*```")


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", LOGO_DIR],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [entry for entry in out.split("\0") if entry]


def parse_record(text: str) -> dict[str, str]:
    """Read filename -> sha256 from the fenced hash blocks of the record."""
    recorded: dict[str, str] = {}
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        found = HASH_LINE.match(line.strip())
        if found:
            digest, name = found.group(1), found.group(2)
            if name in recorded and recorded[name] != digest:
                raise ValueError(f"{RECORD} records two different hashes for {name}")
            recorded[name] = digest
    return recorded


def check(root: Path) -> list[str]:
    record_path = root / RECORD
    if not record_path.is_file():
        return [f"{RECORD} is missing, so no hash record exists to check against"]

    recorded = parse_record(record_path.read_text(encoding="utf-8"))
    assets = sorted(
        path.rsplit("/", 1)[-1] for path in tracked_files(root) if not path.endswith(".md")
    )

    problems: list[str] = []
    for name in assets:
        if name not in recorded:
            problems.append(
                f"{LOGO_DIR}/{name}: vendored but has no recorded SHA-256 in {RECORD}"
            )
    for name in sorted(recorded):
        if name not in assets:
            problems.append(
                f"{RECORD}: records a SHA-256 for {name}, which is not a tracked file in "
                f"{LOGO_DIR}/"
            )
    for name in assets:
        if name not in recorded:
            continue
        actual = hashlib.sha256((root / LOGO_DIR / name).read_bytes()).hexdigest()
        if actual != recorded[name]:
            problems.append(
                f"{LOGO_DIR}/{name}: hashes to {actual}, but {RECORD} records "
                f"{recorded[name]}. The file has changed since it was vendored, which the "
                f"record says never happens. Restore it from the source path named in the "
                f"table, or replace it from upstream and update the hash."
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendored-logo integrity gate.")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    if root is None:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    root = root.resolve()

    problems = check(root)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} logo integrity problem(s).", file=sys.stderr)
        return 1

    count = len([p for p in tracked_files(root) if not p.endswith(".md")])
    print(f"logo assets: {count} vendored files match the SHA-256 recorded for each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
