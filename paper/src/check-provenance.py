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

What none of this establishes is the state of a tree at generation time when the
manifest records a dirty generation. That case is unobservable after the fact, and
the manifest says so rather than being trusted about it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance_common import (  # noqa: E402
    EXCLUDED_DIR_NAMES,
    MISSING_BLOB,
    blobs_at_commit,
    digest_bytes,
    expand,
    identity_summaries,
    repo_root,
    sha256,
    snapshot_digest,
    working_tree_dirty,
)


ROOT = repo_root()
MANIFEST = ROOT / "paper" / "evidence" / "provenance.json"

# Fallback for a schema-2 manifest, which recorded no exclusion list.
DEFAULT_EXCLUDES = EXCLUDED_DIR_NAMES
DEFAULT_ARTIFACT_SCOPE = ("paper/evidence/*.json", "paper/figures/*.pdf")
DEFAULT_MODEL_ARTIFACT_SCOPE = ("paper/evidence/model-artifacts/*.json",)


def expand_relative(patterns: list[str], excludes: frozenset[str]) -> list[str]:
    return [path.relative_to(ROOT).as_posix() for path in expand(patterns, ROOT, excludes)]


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
    if manifest.get("generated_at_head_dirty") is not False or not isinstance(commit, str):
        return []
    blobs = blobs_at_commit(ROOT, commit, [record["path"] for record in records])
    if blobs is None:
        print(
            f"note: {commit[:12]} is not in this checkout, so the clean-generation "
            "claim was not checked against it"
        )
        return []
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


def model_summaries(path: Path, fields: list[str]) -> list[dict[str, str]]:
    return identity_summaries(json.loads(path.read_text(encoding="utf-8")), fields)


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    problems: list[str] = []
    problems += check_group("source", manifest["source_files"])
    problems += check_group("artifact", manifest["artifacts"])
    problems += check_group("model artifact", manifest.get("model_artifacts", []))

    excludes = frozenset(manifest.get("source_snapshot_excludes", DEFAULT_EXCLUDES))

    recorded_sources = [item["path"] for item in manifest["source_files"]]
    for path in expand_relative(manifest["source_snapshot_scope"], excludes):
        if path not in recorded_sources:
            problems.append(f"source: UNRECORDED {path}")

    artifact_scope = list(manifest.get("artifact_scope", DEFAULT_ARTIFACT_SCOPE))
    recorded_artifacts = [item["path"] for item in manifest["artifacts"]]
    manifest_rel = MANIFEST.relative_to(ROOT).as_posix()
    for path in expand_relative(artifact_scope, excludes):
        if path != manifest_rel and path not in recorded_artifacts:
            problems.append(f"artifact: UNRECORDED {path}")

    model_artifact_scope = list(
        manifest.get("model_artifact_scope", DEFAULT_MODEL_ARTIFACT_SCOPE)
    )
    recorded_model_artifacts = {
        item["path"]: item for item in manifest.get("model_artifacts", [])
    }
    for path in expand_relative(model_artifact_scope, excludes):
        if path not in recorded_model_artifacts:
            problems.append(f"model artifact: UNRECORDED {path}")
            continue
        try:
            actual = model_summaries(
                ROOT / path,
                list(manifest.get("model_identity_fields", [])),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"model artifact: INVALID {path}: {error}")
            continue
        if actual != recorded_model_artifacts[path].get("identities"):
            problems.append(f"model artifact: IDENTITY SUMMARY {path}")

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
