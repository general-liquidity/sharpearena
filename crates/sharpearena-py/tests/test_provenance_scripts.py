"""Tests for the evidence-provenance writer and validator.

Neither script had a test, and the model-artifact scope they both implement holds no
data in this repository, so the code was unexercised in both directions. These tests
run the real scripts against a throwaway git checkout via
``SHARPEARENA_PROVENANCE_ROOT``.

Two cases are load-bearing. ``test_checker_rejects_an_unresolved_digest_the_writer_refused``:
the validator used to accept a model identity whose checkpoint digest read
``"unresolved"`` while the writer raised on it, which is backwards for a
tamper-evidence artifact. And
``test_checker_rejects_a_hand_flipped_clean_generation_flag``: a manifest's claim to have
been generated on a clean tree used to be the one field nothing could contradict, so
editing it by hand and regenerating on a clean checkout produced identical files.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
from sharpearena.local_agents import ModelIdentity

REPO = Path(__file__).resolve().parents[3]
PAPER_SRC = REPO / "paper" / "src"


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "provenance_common", PAPER_SRC / "provenance_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = _load_common()

FIELDS = ("model", "digest", "quantization", "server", "server_version")
COMPLETE_IDENTITY = {
    "model": "qwen3-8b",
    "digest": "sha256:0123456789abcdef",
    "quantization": "q4_K_M",
    "server": "ollama",
    "server_version": "0.12.0",
}


def _run(script: str, root: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, SHARPEARENA_PROVENANCE_ROOT=str(root))
    return subprocess.run(
        [sys.executable, str(PAPER_SRC / script)],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env=dict(
            os.environ,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.com",
        ),
    )


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A checkout carrying at least one file for every canonical scope pattern.

    Every pattern is populated on purpose: ``expand`` refuses a scope glob that matches
    nothing, so a fixture covering only part of the scope would exercise the scripts
    against a tree the real gate would reject.
    """

    root = tmp_path / "repo"
    for directory in (
        root / ".github" / "workflows",
        root / "crates" / "demo" / "src",
        root / "crates" / "sharpearena-py" / "python" / "sharpearena",
        root / "scripts",
        root / "paper" / "src",
        root / "paper" / "sections",
        root / "paper" / "evidence" / "model-artifacts",
        root / "paper" / "evidence" / "prospective-forecast-field",
        root / "paper" / "figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    # write_bytes throughout: write_text would translate newlines on Windows, and the
    # line-ending cases below need the on-disk bytes to be exactly what they say.
    for relative, content in {
        ".gitattributes": b"* text=auto eol=lf\n",
        ".github/workflows/ci.yml": b"name: ci\n",
        "Cargo.toml": b'[workspace]\nversion = "0.1.0"\n',
        "Cargo.lock": b"version = 3\n",
        "release.toml": b"sign-tag = true\n",
        "crates/demo/Cargo.toml": b'[package]\nname = "demo"\n',
        "crates/demo/src/lib.rs": b"pub fn one() -> u8 { 1 }\n",
        "crates/sharpearena-py/python/sharpearena/__init__.py": b"VERSION = 1\n",
        "scripts/release.py": b"print(1)\n",
        "paper/src/make-demo.py": b"print(2)\n",
        "paper/main.tex": b"\\documentclass{article}\n",
        "paper/sections/intro.tex": b"intro\n",
        "paper/refs.bib": b"@misc{a}\n",
    }.items():
        (root / relative).write_bytes(content)
    (root / "paper" / "evidence" / "f1.json").write_bytes(b'{"result": 1}\n')
    (
        root / "paper" / "evidence" / "prospective-forecast-field" / "field.json"
    ).write_bytes(b'{"field": 1}\n')
    (
        root / "paper" / "evidence" / "prospective-forecast-field" / "field-plan.sha256"
    ).write_bytes(b"0123456789abcdef  field-plan.json\n")
    (root / "paper" / "figures" / "f1.pdf").write_bytes(
        b"%PDF-1.4\x00binary\r\nbytes\n"
    )
    _git(root, "init", "--quiet")
    # The checkout convention this repository is developed on, and the reason every
    # digest in the committed manifest read as a mismatch before they were normalized.
    _git(root, "config", "core.autocrlf", "true")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "-m", "initial")
    return root


def _make_and_commit(tree: Path) -> None:
    """Write the manifest and commit it, which is the release procedure: the manifest is
    generated against a clean tree and the commit carrying it is that commit's child."""
    made = _run("make-provenance.py", tree)
    assert made.returncode == 0, made.stderr
    _git(tree, "add", "-A")
    _git(tree, "commit", "--quiet", "-m", "bind provenance")


# --- shared rules -------------------------------------------------------------------


def test_identity_summaries_accepts_a_fully_resolved_identity() -> None:
    assert common.identity_summaries(COMPLETE_IDENTITY, FIELDS) == [COMPLETE_IDENTITY]
    assert common.identity_summaries([COMPLETE_IDENTITY], FIELDS) == [COMPLETE_IDENTITY]


def test_provenance_identity_keys_match_the_runtime_model_identity_schema() -> None:
    runtime_fields = frozenset(field.name for field in dataclass_fields(ModelIdentity))

    assert common.MODEL_IDENTITY_ALLOWED_FIELDS == runtime_fields


def test_identity_summaries_accepts_optional_runtime_fields() -> None:
    identity = dict(
        COMPLETE_IDENTITY,
        context_length=32_768,
        capabilities=["chat", "tools"],
        gpu_memory_mib=16_384,
    )

    assert common.identity_summaries(identity, FIELDS) == [COMPLETE_IDENTITY]


def test_identity_summaries_rejects_an_unknown_source_field() -> None:
    identity = dict(COMPLETE_IDENTITY, servre_version="typo")

    with pytest.raises(ValueError, match=r"unknown fields: \['servre_version'\]"):
        common.identity_summaries(identity, FIELDS)


@pytest.mark.parametrize("value", ["unresolved", "UNRESOLVED", "unknown", " ", ""])
def test_identity_summaries_rejects_a_placeholder_digest(value: str) -> None:
    identity = dict(COMPLETE_IDENTITY, digest=value)
    with pytest.raises(ValueError, match="digest"):
        common.identity_summaries(identity, FIELDS)


def test_identity_summaries_rejects_a_missing_or_non_string_field() -> None:
    with pytest.raises(ValueError, match="server_version"):
        common.identity_summaries(
            {k: v for k, v in COMPLETE_IDENTITY.items() if k != "server_version"},
            FIELDS,
        )
    with pytest.raises(ValueError, match="quantization"):
        common.identity_summaries(dict(COMPLETE_IDENTITY, quantization=4), FIELDS)


def test_identity_summaries_rejects_a_non_identity_payload() -> None:
    for payload in ([], "text", [COMPLETE_IDENTITY, "text"]):
        with pytest.raises(ValueError, match="identity object or array"):
            common.identity_summaries(payload, FIELDS)


def test_text_digests_ignore_line_endings_but_binary_digests_do_not(
    tmp_path: Path,
) -> None:
    lf = tmp_path / "lf.rs"
    crlf = tmp_path / "crlf.rs"
    lf.write_bytes(b"fn main() {}\nfn other() {}\n")
    crlf.write_bytes(b"fn main() {}\r\nfn other() {}\r\n")
    assert common.sha256(lf) == common.sha256(crlf)

    binary_lf = tmp_path / "a.pdf"
    binary_crlf = tmp_path / "b.pdf"
    binary_lf.write_bytes(b"%PDF\x00stream\n")
    binary_crlf.write_bytes(b"%PDF\x00stream\r\n")
    assert common.sha256(binary_lf) != common.sha256(binary_crlf)


# --- the scripts, end to end --------------------------------------------------------


def test_writer_output_validates_and_records_the_tree(tree: Path) -> None:
    _make_and_commit(tree)
    manifest = json.loads((tree / "paper" / "evidence" / "provenance.json").read_text())
    recorded = {item["path"] for item in manifest["source_files"]}
    # Every canonical source pattern contributes, which is also what makes the fixture
    # a tree the real gate would accept.
    assert recorded == {
        ".gitattributes",
        ".github/workflows/ci.yml",
        "Cargo.lock",
        "Cargo.toml",
        "crates/demo/Cargo.toml",
        "crates/demo/src/lib.rs",
        "crates/sharpearena-py/python/sharpearena/__init__.py",
        "paper/main.tex",
        "paper/refs.bib",
        "paper/sections/intro.tex",
        "paper/src/make-demo.py",
        "release.toml",
        "scripts/release.py",
    }
    assert {item["path"] for item in manifest["artifacts"]} == {
        "paper/evidence/f1.json",
        "paper/evidence/prospective-forecast-field/field-plan.sha256",
        "paper/evidence/prospective-forecast-field/field.json",
        "paper/figures/f1.pdf",
    }
    assert manifest["source_snapshot_scope"] == list(common.SOURCE_SCOPE)
    assert manifest["source_snapshot_excludes"] == list(common.EXCLUDED_DIR_NAMES)
    assert manifest["artifact_scope"] == list(common.ARTIFACT_SCOPE)
    assert manifest["model_artifact_scope"] == list(common.MODEL_ARTIFACT_SCOPE)
    assert manifest["model_identity_fields"] == list(common.MODEL_IDENTITY_FIELDS)
    assert manifest["generated_at_head_dirty"] is False

    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 0, checked.stdout


def test_checker_catches_an_edited_source_and_an_unrecorded_one(tree: Path) -> None:
    _make_and_commit(tree)

    (tree / "crates" / "demo" / "src" / "lib.rs").write_bytes(
        b"pub fn one() -> u8 { 2 }\n"
    )
    edited = _run("check-provenance.py", tree)
    assert edited.returncode == 1
    assert "source: DIGEST crates/demo/src/lib.rs" in edited.stdout

    (tree / "crates" / "demo" / "src" / "lib.rs").write_bytes(
        b"pub fn one() -> u8 { 1 }\n"
    )
    (tree / "crates" / "demo" / "src" / "extra.rs").write_bytes(b"pub fn two() {}\n")
    added = _run("check-provenance.py", tree)
    assert added.returncode == 1
    assert "source: UNRECORDED crates/demo/src/extra.rs" in added.stdout


def test_checker_ignores_a_line_ending_only_change(tree: Path) -> None:
    _make_and_commit(tree)
    source = tree / "crates" / "demo" / "src" / "lib.rs"
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    rewritten = _run("check-provenance.py", tree)
    assert rewritten.returncode == 0, rewritten.stdout


def test_writer_refuses_an_unresolved_model_identity(tree: Path) -> None:
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(
        json.dumps(dict(COMPLETE_IDENTITY, digest="unresolved")), encoding="utf-8"
    )
    made = _run("make-provenance.py", tree)
    assert made.returncode != 0
    assert "lacks explicit fields" in made.stderr


def test_writer_refuses_an_unknown_model_identity_field(tree: Path) -> None:
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(
        json.dumps(dict(COMPLETE_IDENTITY, servre_version="typo")), encoding="utf-8"
    )

    made = _run("make-provenance.py", tree)

    assert made.returncode != 0
    assert "unknown fields: ['servre_version']" in made.stderr


def test_checker_rejects_an_unresolved_digest_the_writer_refused(tree: Path) -> None:
    """A hand-edited manifest carrying an unresolved checkpoint digest must fail.

    This is the asymmetry the audit found: the writer raised on it, the validator
    returned the summary cleanly, so the untrusted half was the strict one.
    """
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _git(tree, "add", artifact.relative_to(tree).as_posix())
    _git(tree, "commit", "--quiet", "-m", "add model identity")
    _make_and_commit(tree)
    assert _run("check-provenance.py", tree).returncode == 0

    unresolved = dict(COMPLETE_IDENTITY, digest="unresolved")
    artifact.write_text(json.dumps(unresolved), encoding="utf-8")
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["model_artifacts"][0]
    entry["sha256"] = common.sha256(artifact)
    entry["identities"] = [unresolved]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "model artifact: INVALID" in checked.stdout
    assert "digest" in checked.stdout


def test_checker_rejects_an_unknown_model_identity_field(tree: Path) -> None:
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _git(tree, "add", artifact.relative_to(tree).as_posix())
    _git(tree, "commit", "--quiet", "-m", "add model identity")
    _make_and_commit(tree)

    artifact.write_text(
        json.dumps(dict(COMPLETE_IDENTITY, servre_version="typo")), encoding="utf-8"
    )
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_artifacts"][0]["sha256"] = common.sha256(artifact)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checked = _run("check-provenance.py", tree)

    assert checked.returncode == 1
    assert "model artifact: INVALID" in checked.stdout
    assert "unknown fields: ['servre_version']" in checked.stdout


def test_checker_rejects_a_manifest_claiming_a_clean_generation_on_a_dirty_tree(
    tree: Path,
) -> None:
    _make_and_commit(tree)
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    assert json.loads(manifest_path.read_text())["generated_at_head_dirty"] is False

    # Out of every recorded scope, so only the recomputed dirty flag can catch it.
    (tree / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "dirty: generated_at_head_dirty records a clean generation" in checked.stdout


def _edit_manifest_and_commit(tree: Path, **fields: object) -> None:
    """Hand-edit the manifest's own provenance fields and commit, so the recomputed
    dirty flag stays satisfied and only the new check can object."""
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(fields)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(tree, "add", "-A")
    _git(tree, "commit", "--quiet", "-m", "hand-edited manifest")


def test_checker_rejects_a_hand_flipped_clean_generation_flag(tree: Path) -> None:
    """A clean generation is a claim about a commit, so it is checked against that commit.

    The manifest here was generated over an uncommitted edit and says so. Committing it
    and flipping the flag is the one edit that leaves every recorded digest matching the
    working tree, so only reading `generated_at_head` back out of git catches it.
    """
    source = tree / "crates" / "demo" / "src" / "lib.rs"
    source.write_bytes(b"pub fn one() -> u8 { 2 }\n")
    assert _run("make-provenance.py", tree).returncode == 0
    _git(tree, "add", "-A")
    _git(tree, "commit", "--quiet", "-m", "edit and bind together")
    honest = _run("check-provenance.py", tree)
    assert honest.returncode == 1
    assert "generation: DIRTY" in honest.stdout

    _edit_manifest_and_commit(tree, generated_at_head_dirty=False)
    flipped = _run("check-provenance.py", tree)
    assert flipped.returncode == 1
    assert "generation: DIGEST crates/demo/src/lib.rs" in flipped.stdout


def test_checker_rejects_a_clean_generation_naming_a_commit_without_the_file(
    tree: Path,
) -> None:
    (tree / "crates" / "demo" / "src" / "extra.rs").write_bytes(b"pub fn two() {}\n")
    assert _run("make-provenance.py", tree).returncode == 0
    _git(tree, "add", "-A")
    _git(tree, "commit", "--quiet", "-m", "add and bind together")
    honest = _run("check-provenance.py", tree)
    assert honest.returncode == 1
    assert "generation: DIRTY" in honest.stdout

    _edit_manifest_and_commit(tree, generated_at_head_dirty=False)
    flipped = _run("check-provenance.py", tree)
    assert flipped.returncode == 1
    assert "generation: ABSENT crates/demo/src/extra.rs" in flipped.stdout


def test_checker_rejects_a_clean_generation_when_its_commit_is_absent(
    tree: Path,
) -> None:
    """Accepted provenance must name a commit CI can read and verify."""
    _make_and_commit(tree)
    _edit_manifest_and_commit(tree, generated_at_head="0" * 40)

    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "generation: COMMIT" in checked.stdout


def _commit_manifest_edit(tree: Path, edit) -> subprocess.CompletedProcess[str]:
    _make_and_commit(tree)
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    edit(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(tree, "add", str(manifest_path.relative_to(tree)))
    _git(tree, "commit", "--quiet", "-m", "tamper with manifest rules")
    return _run("check-provenance.py", tree)


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (
            lambda manifest: manifest["source_snapshot_scope"].pop(),
            "source_snapshot_scope",
        ),
        (
            lambda manifest: manifest["artifact_scope"].append("paper/other/*.json"),
            "artifact_scope",
        ),
        (
            lambda manifest: manifest["source_snapshot_excludes"].append("crates"),
            "source_snapshot_excludes",
        ),
    ],
)
def test_checker_rejects_self_declared_coverage_rules(
    tree: Path, edit, field: str
) -> None:
    checked = _commit_manifest_edit(tree, edit)

    assert checked.returncode == 1
    assert f"manifest rule: {field}" in checked.stdout


def test_checker_rejects_reduced_model_identity_fields(tree: Path) -> None:
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _git(tree, "add", str(artifact.relative_to(tree)))
    _git(tree, "commit", "--quiet", "-m", "add model identity")

    def reduce_fields(manifest: dict) -> None:
        manifest["model_identity_fields"].remove("server_version")
        manifest["model_artifacts"][0]["identities"][0].pop("server_version")

    checked = _commit_manifest_edit(tree, reduce_fields)

    assert checked.returncode == 1
    assert "manifest rule: model_identity_fields" in checked.stdout


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 4),
        ("digest_convention", "platform-dependent sha256"),
        ("source_snapshot_scope_note", "exclude whatever the manifest asks"),
        ("reproduction_entrypoint", "README.md"),
        ("validator", "paper/src/make-provenance.py"),
    ],
)
def test_checker_rejects_semantic_metadata_tampering(
    tree: Path, field: str, value: object
) -> None:
    checked = _commit_manifest_edit(
        tree, lambda manifest: manifest.__setitem__(field, value)
    )

    assert checked.returncode == 1
    assert f"manifest rule: {field}" in checked.stdout


def test_checker_recomputes_the_source_snapshot_digest(tree: Path) -> None:
    """The snapshot digest is recomputed from the records, not just shape-checked.

    Every per-file digest still matches the tree and the replacement is a well-formed
    sha256, so the manifest rules pass and only recomputing the digest over the source
    records can object. The snapshot hash is the single value the paper quotes for
    "the same tree yields the same snapshot hash", so a hand-edited one that nothing
    recomputes would let a manifest advertise a tree it does not describe.
    """
    forged = "f" * 64

    def forge_snapshot(manifest: dict) -> None:
        assert manifest["source_snapshot_sha256"] != forged
        manifest["source_snapshot_sha256"] = forged

    checked = _commit_manifest_edit(tree, forge_snapshot)

    assert checked.returncode == 1
    assert "snapshot: DIGEST source_snapshot_sha256" in checked.stdout
    assert forged in checked.stdout


def test_checker_rejects_an_extra_source_record_outside_canonical_scope(
    tree: Path,
) -> None:
    extra = tree / "README.md"
    extra.write_text("not provenance source scope\n", encoding="utf-8")
    _git(tree, "add", "README.md")
    _git(tree, "commit", "--quiet", "-m", "add out-of-scope file")

    def append_extra(manifest: dict) -> None:
        manifest["source_files"].append(
            {"path": "README.md", "sha256": common.sha256(extra)}
        )
        manifest["source_files"].sort(key=lambda item: item["path"])
        manifest["source_snapshot_sha256"] = common.snapshot_digest(
            manifest["source_files"]
        )

    checked = _commit_manifest_edit(tree, append_extra)

    assert checked.returncode == 1
    assert "source scope does not exactly match" in checked.stdout


def test_checker_rejects_extra_artifact_and_model_records_outside_canonical_scope(
    tree: Path,
) -> None:
    extra = tree / "README.md"
    extra.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _git(tree, "add", "README.md")
    _git(tree, "commit", "--quiet", "-m", "add out-of-scope identity")

    def append_extra(manifest: dict) -> None:
        digest = common.sha256(extra)
        manifest["artifacts"].append({"path": "README.md", "sha256": digest})
        manifest["model_artifacts"].append(
            {
                "path": "README.md",
                "sha256": digest,
                "identities": [COMPLETE_IDENTITY],
            }
        )

    checked = _commit_manifest_edit(tree, append_extra)

    assert checked.returncode == 1
    assert "artifact scope does not exactly match" in checked.stdout
    assert "model-artifact scope does not exactly match" in checked.stdout


def test_writer_records_a_dirty_generation_honestly_but_checker_refuses_it(
    tree: Path,
) -> None:
    (tree / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
    assert _run("make-provenance.py", tree).returncode == 0
    manifest = json.loads((tree / "paper" / "evidence" / "provenance.json").read_text())
    assert manifest["generated_at_head_dirty"] is True
    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "generation: DIRTY" in checked.stdout


# --- the atomic manifest write ------------------------------------------------------------


def test_write_atomic_writes_lf_only_bytes(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    common.write_atomic(target, '{\n  "a": 1\n}\n')
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert raw == b'{\n  "a": 1\n}\n'


def test_write_atomic_fsyncs_the_file_before_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durability is the entire point: a manifest whose bytes are still in the page cache
    when the machine dies is exactly the truncated-manifest case this replaced."""

    order: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        order.append("fsync")
        return real_fsync(fd)

    def recording_replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    common.write_atomic(tmp_path / "manifest.json", "{}\n")

    assert "replace" in order, "nothing was renamed into place"
    assert order.index("fsync") < order.index("replace"), (
        f"the file was renamed into place before it was fsynced: {order}"
    )


def test_write_atomic_keeps_the_old_file_and_no_debris_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "manifest.json"
    target.write_bytes(b'{"old": true}\n')

    def exploding_replace(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(OSError):
        common.write_atomic(target, '{"new": true}\n')

    # The reader still sees a whole old manifest, not a truncated new one.
    assert target.read_bytes() == b'{"old": true}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"], (
        "the temp file was left behind"
    )


def test_write_atomic_does_not_collide_with_an_existing_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers in one directory share a pid namespace at most by accident, but a
    stale temp from a killed run is ordinary. O_EXCL plus the sequence number means the
    writer steps past it instead of truncating whatever is there."""

    target = tmp_path / "manifest.json"
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    squatter = tmp_path / ".manifest.json.tmp.4242.0"
    squatter.write_bytes(b"another writer's half-manifest")

    common.write_atomic(target, '{"new": true}\n')

    assert target.read_bytes() == b'{"new": true}\n'
    assert squatter.read_bytes() == b"another writer's half-manifest"


# --- the empty-scope floor ----------------------------------------------------------------


def test_expand_refuses_a_pattern_that_matches_nothing(tmp_path: Path) -> None:
    """A scope glob that stops matching is the silent failure this exists to catch: both
    scripts share ``expand``, so they would agree on the smaller set and the gate would
    keep printing OK over a scope that no longer covers what it claims to."""

    (tmp_path / "kept.rs").write_bytes(b"fn main() {}\n")
    excludes = frozenset(common.EXCLUDED_DIR_NAMES)

    assert common.expand(("*.rs",), tmp_path, excludes) == [tmp_path / "kept.rs"]
    with pytest.raises(common.EmptyScopePattern) as excinfo:
        common.expand(("*.rs", "renamed/**/*.rs"), tmp_path, excludes)
    assert "renamed/**/*.rs" in str(excinfo.value)


def test_expand_refuses_a_pattern_whose_only_matches_are_excluded(
    tmp_path: Path,
) -> None:
    """Matching only build output is the same vacuity with a file on disk to point at."""

    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "generated.rs").write_bytes(b"fn generated() {}\n")
    excludes = frozenset(common.EXCLUDED_DIR_NAMES)

    with pytest.raises(common.EmptyScopePattern):
        common.expand(("**/*.rs",), tmp_path, excludes)


def test_model_artifact_scope_is_the_only_pattern_allowed_to_be_empty(
    tmp_path: Path,
) -> None:
    """This repository genuinely ships no model identity files. The exemption is named,
    not inferred from a scope happening to be empty on the day it is run."""

    assert common.OPTIONALLY_EMPTY_SCOPE == frozenset(common.MODEL_ARTIFACT_SCOPE)
    assert not set(common.SOURCE_SCOPE) & common.OPTIONALLY_EMPTY_SCOPE
    assert not set(common.ARTIFACT_SCOPE) & common.OPTIONALLY_EMPTY_SCOPE
    assert (
        common.expand(
            common.MODEL_ARTIFACT_SCOPE, tmp_path, frozenset(common.EXCLUDED_DIR_NAMES)
        )
        == []
    )


def test_checker_refuses_a_manifest_that_binds_no_sources() -> None:
    """A manifest with an empty scope must not validate: it records that nothing was
    checked, which is the shape a vacuous gate takes once the writer has been fixed."""

    manifest = {
        "schema_version": common.SCHEMA_VERSION,
        "generated_at_head": "0" * 40,
        "generated_at_head_dirty": False,
        "digest_convention": common.DIGEST_CONVENTION,
        "source_snapshot_sha256": "0" * 64,
        "source_snapshot_scope": list(common.SOURCE_SCOPE),
        "source_snapshot_excludes": list(common.EXCLUDED_DIR_NAMES),
        "source_snapshot_scope_note": common.SOURCE_SCOPE_NOTE,
        "artifact_scope": list(common.ARTIFACT_SCOPE),
        "model_artifact_scope": list(common.MODEL_ARTIFACT_SCOPE),
        "model_identity_fields": list(common.MODEL_IDENTITY_FIELDS),
        "reproduction_entrypoint": common.REPRODUCTION_ENTRYPOINT,
        "validator": common.VALIDATOR,
        "source_files": [],
        "artifacts": [],
        "model_artifacts": [],
    }

    problems = common.manifest_rule_problems(manifest)

    assert any("source_files is empty" in problem for problem in problems), problems
    assert any("artifacts is empty" in problem for problem in problems), problems
