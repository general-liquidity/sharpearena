"""Validate paper/evidence/provenance.json against the working tree.

Recomputes the SHA-256 of every source file and every artifact the manifest
records, recomputes the source-snapshot digest, and re-expands the recorded
scope to catch files that entered or left the tree since the manifest was
written.  Exits nonzero on any mismatch, so a stale manifest is a detectable
failure rather than a silent one.

Three properties matter beyond recomputing digests. The validator shares
`provenance_common.py` with the writer, so a model identity file the writer would
have refused to record cannot pass here. `generated_at_head_dirty` is a claim
the manifest makes about itself, so it is recomputed rather than believed: a
manifest asserting a clean generation against a dirty tree is a failure. And a
manifest that claims a clean generation is held to what that claims: the recorded
digests are compared against the bytes committed at `generated_at_head`, so flipping
the flag by hand no longer looks like regenerating on a clean checkout.

What none of this can establish is the state of a tree at generation time when the
manifest records a dirty generation. The writer records that state honestly for
inspection, but validation refuses it: a manifest accepted by CI must name a clean,
committed generation that can be read back and checked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance_common import (
    ARTIFACT_SCOPE,
    EXCLUDED_DIR_NAMES,
    MISSING_BLOB,
    MODEL_ARTIFACT_SCOPE,
    MODEL_IDENTITY_FIELDS,
    SOURCE_SCOPE,
    blobs_at_commit,
    digest_bytes,
    expand,
    identity_summaries,
    manifest_rule_problems,
    repo_root,
    sha256,
    snapshot_digest,
    working_tree_dirty,
)

ROOT = repo_root()
MANIFEST = ROOT / "paper" / "evidence" / "provenance.json"


def expand_relative(patterns: list[str], excludes: frozenset[str]) -> list[str]:
    return [
        path.relative_to(ROOT).as_posix() for path in expand(patterns, ROOT, excludes)
    ]


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


def check_clean_generation(manifest: dict, records: list[dict]) -> list[str]:
    """Hold a `generated_at_head_dirty: false` manifest to what that claim means.

    A clean generation says the recorded bytes were the committed bytes of
    ``generated_at_head``, so they are readable out of that commit and must match. A
    manifest that records a dirty generation makes no such claim and is not checked here:
    its bytes existed in no commit when it was written, which is why the flag is true.
    """
    commit = manifest.get("generated_at_head")
    if manifest.get("generated_at_head_dirty") is not False or not isinstance(
        commit, str
    ):
        return []
    blobs = blobs_at_commit(ROOT, commit, [record["path"] for record in records])
    if blobs is None:
        return [
            (
                f"generation: COMMIT {commit} is not present; "
                "the clean-generation claim cannot be verified"
            )
        ]
    problems: list[str] = []
    for record in records:
        blob = blobs[record["path"]]
        if blob is MISSING_BLOB:
            problems.append(
                f"generation: ABSENT {record['path']} is not in {commit[:12]}, "
                "which the manifest names as a clean generation"
            )
            continue
        assert isinstance(blob, bytes)
        committed = digest_bytes(blob)
        if committed != record["sha256"]:
            problems.append(
                f"generation: DIGEST {record['path']} does not match {commit[:12]}\n"
                f"    recorded  {record['sha256']}\n"
                f"    committed {committed}"
            )
    return problems


def model_summaries(path: Path) -> list[dict[str, str]]:
    return identity_summaries(
        json.loads(path.read_text(encoding="utf-8")), MODEL_IDENTITY_FIELDS
    )


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    problems = manifest_rule_problems(manifest)
    if problems:
        for problem in problems:
            print(problem)
        print(f"\nFAIL: {len(problems)} manifest rule problem(s)")
        return 1
    if manifest.get("generated_at_head_dirty") is not False:
        problems.append(
            "generation: DIRTY generated_at_head_dirty must be false; "
            "commit the candidate, regenerate on the clean tree, and commit only "
            "the manifest"
        )
    problems += check_group("source", manifest["source_files"])
    problems += check_group("artifact", manifest["artifacts"])
    problems += check_group("model artifact", manifest.get("model_artifacts", []))

    excludes = frozenset(EXCLUDED_DIR_NAMES)

    recorded_sources = [item["path"] for item in manifest["source_files"]]
    expected_sources = expand_relative(list(SOURCE_SCOPE), excludes)
    if recorded_sources != expected_sources:
        problems.append("source scope does not exactly match the working tree")
        problems.extend(
            f"source: UNRECORDED {path}"
            for path in expected_sources
            if path not in recorded_sources
        )
        problems.extend(
            f"source: EXTRA {path}"
            for path in recorded_sources
            if path not in expected_sources
        )

    recorded_artifacts = [item["path"] for item in manifest["artifacts"]]
    manifest_rel = MANIFEST.relative_to(ROOT).as_posix()
    expected_artifacts = [
        path
        for path in expand_relative(list(ARTIFACT_SCOPE), excludes)
        if path != manifest_rel
    ]
    if recorded_artifacts != expected_artifacts:
        problems.append("artifact scope does not exactly match the working tree")
        problems.extend(
            f"artifact: UNRECORDED {path}"
            for path in expected_artifacts
            if path not in recorded_artifacts
        )
        problems.extend(
            f"artifact: EXTRA {path}"
            for path in recorded_artifacts
            if path not in expected_artifacts
        )

    model_records = manifest.get("model_artifacts", [])
    recorded_model_paths = [item["path"] for item in model_records]
    expected_model_paths = expand_relative(list(MODEL_ARTIFACT_SCOPE), excludes)
    if recorded_model_paths != expected_model_paths:
        problems.append("model-artifact scope does not exactly match the working tree")
        problems.extend(
            f"model artifact: UNRECORDED {path}"
            for path in expected_model_paths
            if path not in recorded_model_paths
        )
        problems.extend(
            f"model artifact: EXTRA {path}"
            for path in recorded_model_paths
            if path not in expected_model_paths
        )
    seen_models: set[str] = set()
    for record in model_records:
        path = record["path"]
        try:
            actual = model_summaries(ROOT / path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"model artifact: INVALID {path}: {error}")
            continue
        if actual != record["identities"]:
            problems.append(f"model artifact: IDENTITY SUMMARY {path}")
        for identity in actual:
            model = identity["model"]
            if model in seen_models:
                problems.append(f"model artifact: DUPLICATE MODEL {model}")
            seen_models.add(model)

    snapshot = snapshot_digest(manifest["source_files"])
    if snapshot != manifest["source_snapshot_sha256"]:
        problems.append(
            "snapshot: DIGEST source_snapshot_sha256\n"
            f"    recorded {manifest['source_snapshot_sha256']}\n"
            f"    actual   {snapshot}"
        )

    problems += check_clean_generation(
        manifest,
        manifest["source_files"]
        + manifest["artifacts"]
        + manifest.get("model_artifacts", []),
    )

    dirty = working_tree_dirty(ROOT)
    if dirty is None:
        print("note: not a git checkout, so generated_at_head_dirty was not re-derived")
    elif dirty and manifest.get("generated_at_head_dirty") is False:
        problems.append(
            "dirty: generated_at_head_dirty records a clean generation, "
            "but this tree has uncommitted work"
        )

    n_sources = len(manifest["source_files"])
    n_artifacts = len(manifest["artifacts"])
    n_model_artifacts = len(manifest.get("model_artifacts", []))
    if problems:
        for problem in problems:
            print(problem)
        print(
            f"\nFAIL: {len(problems)} problem(s) over "
            f"{n_sources} sources, {n_artifacts} artifacts and "
            f"{n_model_artifacts} model artifact manifests"
        )
        return 1
    print(
        f"OK: {n_sources} sources, {n_artifacts} artifacts and "
        f"{n_model_artifacts} model artifact manifests match the tree"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
