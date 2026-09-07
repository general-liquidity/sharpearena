#!/usr/bin/env python3
"""Exercise the SPEC_HASH assertion in Cargo's actual normalized package tree.

No registry publication, model calls or empirical result generation. A source-tree
test alone cannot catch a hash that changes when Cargo rewrites Cargo.toml.
"""

import argparse
from pathlib import Path
import re
import subprocess
import tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="verify uncommitted development changes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]
    command = ["cargo", "package", "-p", "sharpearena", "--target-dir", str(root / "target")]
    if args.allow_dirty:
        command.append("--allow-dirty")
    subprocess.run(command, cwd=root, check=True)
    packaged = root / "target" / "package" / f"sharpearena-{version}" / "Cargo.toml"
    result = subprocess.run(
        ["cargo", "test", "--manifest-path", str(packaged), "--lib", "--",
         "--exact", "spec_hash::tests::committed_spec_hash_record_is_current"],
        cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout, end="")
    if re.search(r"test result: ok\. 1 passed; 0 failed;", result.stdout) is None:
        raise SystemExit("packaged SPEC_HASH test did not execute exactly once")


if __name__ == "__main__":
    main()
