"""Integration tests for the checked SharpeArena release/tag process."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RELEASE = REPO / "scripts" / "release.py"
MANIFEST = Path("paper/evidence/provenance.json")


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "release_test_provenance_common", REPO / "paper/src/provenance_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = _load_common()
MODEL_IDENTITY_FIELDS = common.MODEL_IDENTITY_FIELDS
COMPLETE_IDENTITY = {
    "model": "qwen3-8b",
    "digest": "sha256:0123456789abcdef",
    "quantization": "q4_K_M",
    "server": "ollama",
    "server_version": "0.12.0",
}
CANONICAL_METADATA = {
    "schema_version": 5,
    "digest_convention": (
        "sha256 over file bytes with CRLF collapsed to LF; files containing "
        "a NUL byte are hashed verbatim"
    ),
    "source_snapshot_scope_note": (
        "Globs are expanded from the repository root; any path with a component "
        "in source_snapshot_excludes is skipped, so build outputs and virtual "
        "environments are not part of the source snapshot."
    ),
    "reproduction_entrypoint": "commands in paper/sections/A-commands.tex",
    "validator": "paper/src/check-provenance.py",
}


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE), *args, "--repo", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            GIT_AUTHOR_NAME="release test",
            GIT_AUTHOR_EMAIL="release@example.com",
            GIT_COMMITTER_NAME="release test",
            GIT_COMMITTER_EMAIL="release@example.com",
        ),
    )
    return completed.stdout.strip()


def _digest(data: bytes) -> str:
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _write_versions(root: Path, version: str) -> None:
    files = {
        "Cargo.toml": f'[workspace]\n[workspace.package]\nversion = "{version}"\n',
        "crates/sharpearena-wasm/Cargo.toml": (
            '[package]\nname = "sharpearena-wasm"\nversion.workspace = true\n'
            f'sharpearena = {{ path = "../sharpearena", version = "{version}" }}\n'
        ),
        "crates/sharpearena-py/Cargo.toml": (
            '[package]\nname = "sharpearena-py"\n'
            f'version = "{version}"\n'
            f'sharpearena = {{ path = "../sharpearena", version = "{version}" }}\n'
        ),
        "crates/sharpearena-py/pyproject.toml": (
            f'[project]\nname = "sharpearena"\nversion = "{version}"\n'
        ),
        "crates/sharpearena-py/python/sharpearena/__init__.py": (
            f'__version__ = "{version}"\n'
        ),
        "npm/sharpearena/package.json": json.dumps(
            {"name": "@general-liquidity/sharpearena", "version": version}
        )
        + "\n",
        "npm/sharpearena/pkg/package.json": json.dumps(
            {"name": "sharpearena", "version": version}
        )
        + "\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _write_remaining_source_scope(root: Path) -> None:
    """One file for every canonical source pattern ``_write_versions`` does not cover.

    ``expand`` refuses a scope glob that matches nothing, so a release rehearsal over a
    tree missing most of the scope would exercise a tree the real gate rejects.
    """

    files = {
        ".gitattributes": "* text=auto eol=lf\n",
        ".github/workflows/ci.yml": "name: ci\n",
        "Cargo.lock": "version = 3\n",
        "release.toml": "sign-tag = true\n",
        "scripts/release.py": "print(1)\n",
        "paper/src/make-demo.py": "print(2)\n",
        "paper/main.tex": "\\documentclass{article}\n",
        "paper/sections/intro.tex": "intro\n",
        "paper/refs.bib": "@misc{a}\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    figure = root / "paper/figures/f1.pdf"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"%PDF-1.4\x00binary\n")


def _write_manifest(root: Path, generation: str, *, dirty: bool = False) -> None:
    excludes = frozenset(common.EXCLUDED_DIR_NAMES)
    source_paths = [
        path.relative_to(root).as_posix()
        for path in common.expand(common.SOURCE_SCOPE, root, excludes)
    ]
    artifact_paths = [
        path.relative_to(root).as_posix()
        for path in common.expand(common.ARTIFACT_SCOPE, root, excludes)
        if path.relative_to(root) != MANIFEST
    ]
    source_records = [
        {"path": path, "sha256": _digest((root / path).read_bytes())}
        for path in source_paths
    ]
    artifact_records = [
        {"path": path, "sha256": _digest((root / path).read_bytes())}
        for path in artifact_paths
    ]
    model_records = []
    for path in sorted((root / "paper/evidence/model-artifacts").glob("*.json")):
        identity = json.loads(path.read_text(encoding="utf-8"))
        identities = identity if isinstance(identity, list) else [identity]
        model_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _digest(path.read_bytes()),
                "identities": [
                    {field: item[field] for field in MODEL_IDENTITY_FIELDS}
                    for item in identities
                ],
            }
        )
    snapshot = hashlib.sha256(
        "".join(
            f"{record['sha256']}  {record['path']}\n" for record in source_records
        ).encode()
    ).hexdigest()
    payload = {
        **CANONICAL_METADATA,
        "generated_at_head": generation,
        "generated_at_head_dirty": dirty,
        "source_snapshot_sha256": snapshot,
        "source_snapshot_scope": list(common.SOURCE_SCOPE),
        "source_snapshot_excludes": list(common.EXCLUDED_DIR_NAMES),
        "artifact_scope": list(common.ARTIFACT_SCOPE),
        "model_artifact_scope": list(common.MODEL_ARTIFACT_SCOPE),
        "model_identity_fields": list(MODEL_IDENTITY_FIELDS),
        "source_files": source_records,
        "artifacts": artifact_records,
        "model_artifacts": model_records,
    }
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def release_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write_versions(root, "0.19.0")
    _write_remaining_source_scope(root)
    source = root / "crates/demo/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn value() -> u8 { 19 }\n", encoding="utf-8")
    artifact = root / "paper/evidence/f1.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"value": 19}\n', encoding="utf-8")
    _git(root, "init", "--quiet", "--initial-branch=main")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "initial sources")
    initial = _git(root, "rev-parse", "HEAD")
    _write_manifest(root, initial)
    _git(root, "add", str(MANIFEST))
    _git(root, "commit", "--quiet", "-m", "initial provenance bind")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root


def _release_commit(root: Path) -> str:
    _write_versions(root, "0.20.0")
    (root / "crates/demo/src/lib.rs").write_text(
        "pub fn value() -> u8 { 20 }\n", encoding="utf-8"
    )
    (root / "paper/evidence/f1.json").write_text('{"value": 20}\n', encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "release: v0.20.0")
    return _git(root, "rev-parse", "HEAD")


def _bind_and_tag(
    root: Path,
    release_commit: str,
    *,
    dirty: bool = False,
    annotated: bool = True,
) -> None:
    _write_manifest(root, release_commit, dirty=dirty)
    _git(root, "add", str(MANIFEST))
    _git(root, "commit", "--quiet", "-m", "bind release provenance")
    if annotated:
        _git(root, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")
    else:
        _git(root, "tag", "v0.20.0")


def test_verify_tag_accepts_a_clean_provenance_only_rebind(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    _git(release_tree, "update-ref", "refs/remotes/origin/main", "HEAD")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "release tag v0.20.0 is bound to its exact in-scope tree" in checked.stdout


def test_verify_tag_rejects_a_tag_only_child_not_merged_to_origin_main(
    release_tree: Path,
) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "not an ancestor of origin/main" in checked.stdout


def test_prospective_validation_and_post_release_validation_use_distinct_lineage(
    release_tree: Path,
) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    tag_commit = _git(release_tree, "rev-parse", "v0.20.0^{commit}")

    spec = importlib.util.spec_from_file_location("release_driver", RELEASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    prospective, validated = module.verify_tag(
        release_tree, "v0.20.0", _allow_prospective=True
    )
    assert prospective == []
    assert validated == tag_commit

    _git(release_tree, "update-ref", "refs/remotes/origin/main", tag_commit)
    published, validated = module.verify_tag(release_tree, "v0.20.0")
    assert published == []
    assert validated == tag_commit


def test_rehearsal_clone_checks_out_the_exact_detached_source_head(
    release_tree: Path, tmp_path: Path
) -> None:
    reviewed_commit = _release_commit(release_tree)
    origin_main = _git(release_tree, "rev-parse", "refs/remotes/origin/main^{commit}")
    _git(release_tree, "checkout", "--quiet", "--detach", reviewed_commit)
    _git(release_tree, "branch", "--force", "main", origin_main)

    spec = importlib.util.spec_from_file_location("release_driver_clone", RELEASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    clone = tmp_path / "rehearsal-clone"
    module.clone_rehearsal_tree(release_tree, clone)

    assert _git(clone, "rev-parse", "HEAD") == reviewed_commit
    assert _git(clone, "branch", "--show-current") == "main"
    assert _git(clone, "rev-parse", "refs/remotes/origin/main^{commit}") == origin_main


def test_verify_tag_writes_the_exact_validated_commit(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    tag_commit = _git(release_tree, "rev-parse", "v0.20.0^{commit}")
    _git(release_tree, "update-ref", "refs/remotes/origin/main", tag_commit)
    output = release_tree / "github-output"

    checked = _run(
        release_tree,
        "verify-tag",
        "v0.20.0",
        "--github-output",
        str(output),
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert output.read_text(encoding="utf-8") == f"validated_commit={tag_commit}\n"


def test_verify_tag_rejects_a_stale_by_one_release_tag(release_tree: Path) -> None:
    _release_commit(release_tree)
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "provenance-only rebind" in checked.stdout


def test_verify_tag_rejects_a_manifest_naming_a_commit_other_than_its_parent(
    release_tree: Path,
) -> None:
    """The rebind is well formed and the tree matches, so only rule two can object.

    The tag commit changes nothing but the manifest, and every in-scope byte it records
    is present in the tagged tree, because the commit in between touches only a path no
    scope covers. What is wrong is the binding itself: the manifest names an earlier
    commit rather than the tag commit's parent, which is exactly what a rebind carried
    forward across an unrelated commit looks like.
    """
    release_commit = _release_commit(release_tree)
    notes = release_tree / "docs/notes.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("out of every recorded scope\n", encoding="utf-8")
    _git(release_tree, "add", notes.relative_to(release_tree).as_posix())
    _git(release_tree, "commit", "--quiet", "-m", "note an out-of-scope change")
    parent = _git(release_tree, "rev-parse", "HEAD")
    _write_manifest(release_tree, release_commit)
    _git(release_tree, "add", str(MANIFEST))
    _git(release_tree, "commit", "--quiet", "-m", "bind release provenance")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")
    _git(release_tree, "update-ref", "refs/remotes/origin/main", "HEAD")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "must name the tag commit's parent" in checked.stdout
    assert parent in checked.stdout
    assert release_commit in checked.stdout
    assert "provenance-only rebind" not in checked.stdout


def test_verify_tag_rejects_a_lightweight_release_tag(release_tree: Path) -> None:
    """A lightweight tag carries no tagger, message or signature to audit."""
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit, annotated=False)
    _git(release_tree, "update-ref", "refs/remotes/origin/main", "HEAD")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "release tag v0.20.0 must be annotated" in checked.stdout


def test_verify_tag_rejects_a_dirty_generation_manifest(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit, dirty=True)

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "records a dirty generation" in checked.stdout


def test_verify_tag_rejects_an_extra_edit_in_the_rebind_commit(
    release_tree: Path,
) -> None:
    release_commit = _release_commit(release_tree)
    _write_manifest(release_tree, release_commit)
    (release_tree / "crates/demo/src/lib.rs").write_text(
        "pub fn value() -> u8 { 21 }\n", encoding="utf-8"
    )
    _git(release_tree, "add", "-A")
    _git(release_tree, "commit", "--quiet", "-m", "impure provenance bind")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "also changes crates/demo/src/lib.rs" in checked.stdout


def test_verify_tag_rejects_a_surface_version_drift(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    package = release_tree / "npm/sharpearena/package.json"
    payload = json.loads(package.read_text(encoding="utf-8"))
    payload["version"] = "0.19.0"
    package.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _git(release_tree, "add", package.relative_to(release_tree).as_posix())
    _git(release_tree, "commit", "--quiet", "-m", "drift npm version")
    _git(release_tree, "tag", "-f", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "npm/sharpearena/package.json reports 0.19.0" in checked.stdout


def test_verify_tag_rejects_an_unrecorded_nested_source(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    nested = release_tree / "crates/demo/src/nested/extra.rs"
    nested.parent.mkdir(parents=True)
    nested.write_text("pub fn extra() {}\n", encoding="utf-8")
    _git(release_tree, "add", nested.relative_to(release_tree).as_posix())
    _git(release_tree, "commit", "--quiet", "-m", "add unrecorded source")
    release_commit = _git(release_tree, "rev-parse", "HEAD")
    _write_manifest(release_tree, release_commit)
    manifest_path = release_tree / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_files"] = [
        record
        for record in manifest["source_files"]
        if record["path"] != nested.relative_to(release_tree).as_posix()
    ]
    manifest["source_snapshot_sha256"] = hashlib.sha256(
        "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in manifest["source_files"]
        ).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(release_tree, "add", str(MANIFEST))
    _git(release_tree, "commit", "--quiet", "-m", "bind incomplete provenance")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "source scope does not exactly match" in checked.stdout


def test_verify_tag_recomputes_model_identity_summaries(release_tree: Path) -> None:
    identity_path = release_tree / "paper/evidence/model-artifacts/runner.json"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _git(release_tree, "add", identity_path.relative_to(release_tree).as_posix())
    _git(release_tree, "commit", "--quiet", "-m", "add model identity")
    release_commit = _release_commit(release_tree)
    _write_manifest(release_tree, release_commit)
    manifest_path = release_tree / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_artifacts"][0]["identities"][0]["server_version"] = "forged"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(release_tree, "add", str(MANIFEST))
    _git(release_tree, "commit", "--quiet", "-m", "bind forged model summary")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "model artifact: IDENTITY SUMMARY" in checked.stdout


def test_verify_tag_rejects_an_unknown_model_identity_field(
    release_tree: Path,
) -> None:
    identity_path = release_tree / "paper/evidence/model-artifacts/runner.json"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(
        json.dumps(dict(COMPLETE_IDENTITY, servre_version="typo")), encoding="utf-8"
    )
    _git(release_tree, "add", identity_path.relative_to(release_tree).as_posix())
    _git(release_tree, "commit", "--quiet", "-m", "add misspelled model identity")
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    _git(release_tree, "update-ref", "refs/remotes/origin/main", "HEAD")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "model artifact: INVALID" in checked.stdout
    assert "unknown fields: ['servre_version']" in checked.stdout


def test_verify_tag_rejects_manifest_rule_tampering(release_tree: Path) -> None:
    release_commit = _release_commit(release_tree)
    _write_manifest(release_tree, release_commit)
    manifest_path = release_tree / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_snapshot_excludes"].append("crates")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(release_tree, "add", str(MANIFEST))
    _git(release_tree, "commit", "--quiet", "-m", "bind narrowed release scope")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "manifest rule: source_snapshot_excludes" in checked.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 4),
        ("digest_convention", "sha256 over whatever bytes happen to exist"),
        ("source_snapshot_scope_note", "build outputs are included"),
        ("reproduction_entrypoint", "README.md"),
        ("validator", "scripts/accept-anything.py"),
    ],
)
def test_verify_tag_rejects_semantic_metadata_tampering(
    release_tree: Path, field: str, value: object
) -> None:
    release_commit = _release_commit(release_tree)
    _write_manifest(release_tree, release_commit)
    manifest_path = release_tree / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(release_tree, "add", str(MANIFEST))
    _git(release_tree, "commit", "--quiet", "-m", "tamper with semantic metadata")
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")
    _git(release_tree, "update-ref", "refs/remotes/origin/main", "HEAD")

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert f"manifest rule: {field}" in checked.stdout


def test_verify_tag_rejects_a_release_on_a_diverged_detached_line(
    release_tree: Path,
) -> None:
    release_commit = _release_commit(release_tree)
    _bind_and_tag(release_tree, release_commit)
    origin_main = _git(release_tree, "rev-parse", "refs/remotes/origin/main")
    tree = _git(release_tree, "rev-parse", f"{origin_main}^{{tree}}")
    advanced = subprocess.run(
        ["git", "commit-tree", tree, "-p", origin_main],
        cwd=release_tree,
        input="advance origin main on another line\n",
        text=True,
        capture_output=True,
        check=True,
        env=dict(
            os.environ,
            GIT_AUTHOR_NAME="release test",
            GIT_AUTHOR_EMAIL="release@example.com",
            GIT_COMMITTER_NAME="release test",
            GIT_COMMITTER_EMAIL="release@example.com",
        ),
    ).stdout.strip()
    _git(release_tree, "update-ref", "refs/remotes/origin/main", advanced)

    checked = _run(release_tree, "verify-tag", "v0.20.0")

    assert checked.returncode == 1
    assert "origin/main" in checked.stdout


def _run_blocks(text: str) -> list[str]:
    """Extract inline and block `run` scalars without interpreting expressions."""
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        matched = re.match(r"^(?P<indent>\s*)run:\s*(?P<value>.*)$", line)
        if matched is None:
            continue
        value = matched.group("value")
        if value != "|":
            blocks.append(value)
            continue
        indent = len(matched.group("indent"))
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append("\n".join(body))
    return blocks


def test_release_workflow_never_interpolates_context_directly_into_shell() -> None:
    workflow = (REPO / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert _run_blocks(workflow)
    assert all("${{" not in block for block in _run_blocks(workflow))


def test_release_tag_with_shell_metacharacters_is_data_not_code(
    release_tree: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "workflow-injection"
    malicious = f'v0.20.0"; touch "{marker}"; #'

    checked = _run(release_tree, "verify-tag", malicious)

    assert checked.returncode == 1
    assert not marker.exists()
    assert "vMAJOR.MINOR.PATCH" in checked.stdout


def test_release_workflow_actions_and_installers_are_immutable() -> None:
    text = (REPO / ".github/workflows/release.yml").read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", text)

    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in uses)
    assert "@master" not in text
    assert "@latest" not in text
    assert "@release/v1" not in text
    assert "installer/init.sh" not in text
    assert "tool: wasm-pack@0.15.0" in text
    assert "maturin-version: v1.15.0" in text
    assert "npm@12.0.2" in text
    assert 'python-version: "3.12.14"' in text
    assert "node-version: 24.18.0" in text
    assert 'toolchain: "1.96.0"' in text
    assert (
        "container: quay.io/pypa/manylinux2014_x86_64@sha256:"
        "edb6edbd84c2fa9d40ee83abb160e302ebce82eb93570d43343942a1fb10b962" in text
    )


def test_publish_jobs_checkout_only_the_validated_commit() -> None:
    text = (REPO / ".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "validated_commit: ${{ steps.verify.outputs.validated_commit }}" in text
    for job in ("crates", "npm", "pypi"):
        job_text = text.split(f"  {job}:\n", 1)[1]
        next_job = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\s*$", job_text)
        if next_job is not None:
            job_text = job_text[: next_job.start()]
        assert "ref: ${{ needs.validate_tag.outputs.validated_commit }}" in job_text
        assert "ref: ${{ inputs.release_tag || github.ref }}" not in job_text


def _release_module(name: str):
    spec = importlib.util.spec_from_file_location(name, RELEASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_execute_cuts_the_release_in_a_worktree_off_the_base_commit(
    release_tree: Path,
) -> None:
    """The cut must read the base commit, not whatever the checkout happens to hold.

    The checkout is put in the two states that used to be caught by an assertion at the
    top of ``execute_release``: local ``main`` is ahead of ``origin/main`` with a bogus
    version, and the working tree is dirty. Neither can reach the release now, because
    the release never runs in this checkout.

    The weaker assertion deliberately not written is "execute_release returned v0.20.0".
    It passes just as well when the cut runs in the caller's own checkout off the stale
    local ``main``, which is the entire failure mode this closes. What is asserted is
    where the cut ran and which version it read.
    """
    _write_versions(release_tree, "9.9.9")
    _git(release_tree, "add", "-A")
    _git(release_tree, "commit", "--quiet", "-m", "local main runs ahead")
    (release_tree / "crates/demo/src/lib.rs").write_text(
        "pub fn value() -> u8 { 99 }\n", encoding="utf-8"
    )
    module = _release_module("release_driver_worktree")
    seen: dict[str, object] = {}

    def _record(root, bump, current, target, release_branch, *, push):
        seen.update(
            root=root,
            bump=bump,
            current=current,
            target=target,
            release_branch=release_branch,
            push=push,
            cargo_toml=(root / "Cargo.toml").read_text(encoding="utf-8"),
            status=_git(root, "status", "--porcelain"),
        )
        return f"v{target}"

    module.cut_release = _record

    tag = module.execute_release(release_tree, "minor", push=False, fetch=False)

    assert tag == "v0.20.0"
    assert seen["current"] == "0.19.0"
    assert seen["target"] == "0.20.0"
    assert seen["release_branch"] == "release-v0.20.0"
    assert 'version = "0.19.0"' in seen["cargo_toml"]
    assert seen["status"] == ""
    cut_root = seen["root"]
    assert isinstance(cut_root, Path)
    assert cut_root != release_tree
    assert release_tree not in cut_root.parents
    assert cut_root.name == "v0.20.0"
    assert not cut_root.exists()
    assert str(cut_root) not in _git(release_tree, "worktree", "list")
    assert "release-v0.20.0" not in _git(release_tree, "branch", "--list")
    assert 'version = "9.9.9"' in (release_tree / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    assert "crates/demo/src/lib.rs" in _git(release_tree, "status", "--porcelain")


def test_execute_removes_the_worktree_and_branch_when_the_cut_fails(
    release_tree: Path,
) -> None:
    module = _release_module("release_driver_worktree_failure")
    seen: dict[str, Path] = {}

    def _explode(root, bump, current, target, release_branch, *, push):
        seen["root"] = root
        raise module.ReleaseError("the cut failed part way")

    module.cut_release = _explode

    with pytest.raises(module.ReleaseError):
        module.execute_release(release_tree, "patch", push=False, fetch=False)

    assert not seen["root"].exists()
    assert str(seen["root"]) not in _git(release_tree, "worktree", "list")
    assert "release-v0.19.1" not in _git(release_tree, "branch", "--list")


def test_execute_refuses_a_base_ref_that_does_not_resolve(release_tree: Path) -> None:
    _git(release_tree, "update-ref", "-d", "refs/remotes/origin/main")
    module = _release_module("release_driver_missing_base")

    with pytest.raises(module.ReleaseError, match="does not resolve to a commit"):
        module.execute_release(release_tree, "patch", push=False, fetch=False)


def test_execute_refuses_an_existing_tag_before_creating_any_worktree(
    release_tree: Path,
) -> None:
    _git(release_tree, "tag", "-a", "v0.20.0", "-m", "SharpeArena v0.20.0")
    module = _release_module("release_driver_existing_tag")
    module.cut_release = lambda *args, **kwargs: pytest.fail("the cut must not start")

    with pytest.raises(module.ReleaseError, match="tag v0.20.0 already exists"):
        module.execute_release(release_tree, "minor", push=False, fetch=False)

    assert _git(release_tree, "worktree", "list").count("\n") == 0


def test_execute_fetches_the_base_before_resolving_it(release_tree: Path) -> None:
    """Freshness is the point: the fetch must precede the read of origin/main."""
    module = _release_module("release_driver_fetch_order")
    calls: list[tuple[str, ...]] = []
    real_run = module.run

    def _spy(root, *args, **kwargs):
        calls.append(args)
        if args[:2] == ("git", "fetch"):
            return subprocess.CompletedProcess(list(args), 0, "", "")
        return real_run(root, *args, **kwargs)

    module.run = _spy
    module.cut_release = lambda *args, **kwargs: "v0.20.0"

    module.execute_release(release_tree, "minor", push=False)

    fetched = next(
        index for index, args in enumerate(calls) if args[:2] == ("git", "fetch")
    )
    resolved = next(
        index
        for index, args in enumerate(calls)
        if args[:2] == ("git", "rev-parse")
        and "refs/remotes/origin/main^{commit}" in args
    )
    assert calls[fetched] == ("git", "fetch", "--quiet", "origin", "main", "--tags")
    assert fetched < resolved


def test_rehearsal_drives_execute_off_the_reviewed_commit_without_fetching(
    release_tree: Path,
) -> None:
    """The rehearsal's base is the clone's own main, which is the reviewed commit.

    A fetch inside the rehearsal would replace the commit under review with whatever
    the source repository's ``main`` points at, which is exactly the substitution
    ``clone_rehearsal_tree`` exists to prevent.
    """
    module = _release_module("release_driver_rehearsal_args")
    invocations: list[tuple[str, ...]] = []
    real_run = module.run

    def _spy(root, *args, **kwargs):
        if args[:1] == (sys.executable,):
            invocations.append(args)
            return subprocess.CompletedProcess(list(args), 0, "created v0.20.0\n", "")
        return real_run(root, *args, **kwargs)

    module.run = _spy

    assert module.rehearse(release_tree, "minor") == "v0.20.0"

    (invoked,) = invocations
    assert "--no-fetch" in invoked
    assert invoked[invoked.index("--base-ref") + 1] == "refs/heads/main"


class _Teardown:
    """Records the teardown effects as calls instead of performing them.

    No process, no filesystem, no daemon: the ordering is the only thing under test,
    so the three effects the real teardown injects are replaced by recorders.
    """

    def __init__(self, *, present: bool, removes: bool = True, raises: bool = False):
        self.present = present
        self.removes = removes
        self.raises = raises
        self.calls: list[str] = []

    def is_present(self) -> bool:
        return self.present

    def remove(self) -> None:
        self.calls.append("remove")
        if self.raises:
            raise OSError(32, "the checkout is locked by another process")
        if self.removes:
            self.present = False

    def deregister(self) -> None:
        self.calls.append("deregister")


def _cleanup(teardown: _Teardown) -> tuple[str, ...]:
    module = _release_module("release_driver_cleanup")
    return module.cleanup_worktree(
        is_present=teardown.is_present,
        remove=teardown.remove,
        deregister=teardown.deregister,
    )


def test_cleanup_removes_the_checkout_before_dropping_its_registration() -> None:
    """The weaker assertion deliberately not written is "both effects ran".

    Both effects run in the failure orderings too, and it is the order that decides
    whether a crash between them leaves a sweepable worktree or an orphaned directory.
    So the call sequence is asserted, not the call set.
    """
    teardown = _Teardown(present=True)

    steps = _cleanup(teardown)

    assert teardown.calls == ["remove", "deregister"]
    assert steps == ("remove", "deregister")


def test_cleanup_deregisters_a_checkout_that_is_already_gone() -> None:
    """A crashed earlier attempt must not leak its registration forever."""
    teardown = _Teardown(present=False)

    steps = _cleanup(teardown)

    assert teardown.calls == ["deregister"]
    assert steps == ("deregister",)


def test_cleanup_leaves_the_registration_when_removal_raises() -> None:
    teardown = _Teardown(present=True, raises=True)

    steps = _cleanup(teardown)

    assert teardown.calls == ["remove"]
    assert "deregister" not in teardown.calls
    assert steps == ("remove-failed",)
    assert teardown.present


def test_cleanup_leaves_the_registration_when_removal_only_half_succeeds() -> None:
    """Removal that returns without raising still has to be checked.

    A partially deleted checkout that is no longer registered is unreachable: nothing
    lists it, so no later sweep finds it. Registered and partial is recoverable.
    """
    teardown = _Teardown(present=True, removes=False)

    steps = _cleanup(teardown)

    assert teardown.calls == ["remove"]
    assert steps == ("remove-failed",)
    assert teardown.present


def _safe_to_delete(path: Path, root: Path) -> bool:
    return _release_module("release_driver_delete_guard").safe_to_delete(path, root)


def test_delete_guard_accepts_a_path_strictly_inside_the_root(tmp_path: Path) -> None:
    """Paired with the rejections below: a guard that refuses everything is not correct.

    The weaker assertion deliberately not written is "the guard rejects `..`". On its
    own that passes for a predicate hardcoded to False, which would make every delete
    in the release driver unreachable and every teardown a leak.
    """
    assert _safe_to_delete(tmp_path / "v0.20.0", tmp_path)
    assert _safe_to_delete(tmp_path / "v0.20.0" / "crates", tmp_path)


def test_delete_guard_rejects_an_empty_path(tmp_path: Path) -> None:
    """Pinned, but not independently observable: ``Path("")`` normalizes to ``.``, which
    the containment leg already refuses. Deleting the explicit empty-path leg from the
    predicate does not fail this test. It is pinned as behaviour, not as coverage of
    that leg."""
    assert not _safe_to_delete(Path(""), tmp_path)
    assert not _safe_to_delete(Path("."), tmp_path)


def test_delete_guard_rejects_the_root_itself(tmp_path: Path) -> None:
    assert not _safe_to_delete(tmp_path, tmp_path)


def test_delete_guard_rejects_a_path_that_escapes_the_root(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()

    assert not _safe_to_delete(tmp_path, root)
    assert not _safe_to_delete((root / ".." / "elsewhere").resolve(), root)
    assert not _safe_to_delete(Path(tmp_path.anchor), root)


def test_delete_guard_rejects_a_symlink_inside_the_root(tmp_path: Path) -> None:
    """A link is inside the root; what it names is not, and rmtree follows the name."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "release"
    root.mkdir()
    link = root / "v0.20.0"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not allow creating symlinks unprivileged")

    assert not _safe_to_delete(link, root)


def test_worktree_teardown_refuses_to_delete_its_own_parent(tmp_path: Path) -> None:
    module = _release_module("release_driver_delete_guard_wiring")

    with pytest.raises(module.ReleaseError, match="refusing to delete"):
        module.remove_release_worktree(tmp_path, tmp_path, tmp_path)

    assert tmp_path.exists()
