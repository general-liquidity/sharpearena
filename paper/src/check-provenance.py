"""Validate paper/evidence/provenance.json against the working tree.

Recomputes the SHA-256 of every source file and every artifact the manifest
records, recomputes the source-snapshot digest, and re-expands the recorded
scope to catch files that entered or left the tree since the manifest was
written.  Exits nonzero on any mismatch, so a stale manifest is a detectable
failure rather than a silent one.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "paper" / "evidence" / "provenance.json"

# Fallback for a schema-2 manifest, which recorded no exclusion list.
DEFAULT_EXCLUDES = ("target", ".venv", "__pycache__", "node_modules", ".git")
DEFAULT_ARTIFACT_SCOPE = ("paper/evidence/*.json", "paper/figures/*.pdf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_excluded(path: Path, excludes: frozenset[str]) -> bool:
    return any(part in excludes for part in path.relative_to(ROOT).parts)


def expand(patterns: list[str], excludes: frozenset[str]) -> list[str]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(
            path
            for path in ROOT.glob(pattern)
            if path.is_file() and not is_excluded(path, excludes)
        )
    return sorted(path.relative_to(ROOT).as_posix() for path in found)


def check_group(name: str, records: list[dict]) -> list[str]:
    problems: list[str] = []
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            problems.append(f"{name}: MISSING {record['path']}")
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            problems.append(
                f"{name}: DIGEST {record['path']}\n"
                f"    recorded {record['sha256']}\n"
                f"    actual   {actual}"
            )
    return problems


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    problems: list[str] = []
    problems += check_group("source", manifest["source_files"])
    problems += check_group("artifact", manifest["artifacts"])

    excludes = frozenset(manifest.get("source_snapshot_excludes", DEFAULT_EXCLUDES))

    recorded_sources = [item["path"] for item in manifest["source_files"]]
    for path in expand(manifest["source_snapshot_scope"], excludes):
        if path not in recorded_sources:
            problems.append(f"source: UNRECORDED {path}")

    artifact_scope = list(manifest.get("artifact_scope", DEFAULT_ARTIFACT_SCOPE))
    recorded_artifacts = [item["path"] for item in manifest["artifacts"]]
    manifest_rel = MANIFEST.relative_to(ROOT).as_posix()
    for path in expand(artifact_scope, excludes):
        if path != manifest_rel and path not in recorded_artifacts:
            problems.append(f"artifact: UNRECORDED {path}")

    snapshot = hashlib.sha256(
        "".join(
            f"{item['sha256']}  {item['path']}\n" for item in manifest["source_files"]
        ).encode()
    ).hexdigest()
    if snapshot != manifest["source_snapshot_sha256"]:
        problems.append(
            "snapshot: DIGEST source_snapshot_sha256\n"
            f"    recorded {manifest['source_snapshot_sha256']}\n"
            f"    actual   {snapshot}"
        )

    n_sources = len(manifest["source_files"])
    n_artifacts = len(manifest["artifacts"])
    if problems:
        for problem in problems:
            print(problem)
        print(
            f"\nFAIL: {len(problems)} problem(s) over "
            f"{n_sources} sources and {n_artifacts} artifacts"
        )
        return 1
    print(f"OK: {n_sources} sources and {n_artifacts} artifacts match the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
