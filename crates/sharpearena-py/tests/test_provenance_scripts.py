"""Tests for the evidence-provenance writer and validator.

Neither script had a test, and the model-artifact scope they both implement holds no
data in this repository, so the code was unexercised in both directions. These tests
run the real scripts against a throwaway git checkout via
``SHARPEARENA_PROVENANCE_ROOT``.

The load-bearing case is ``test_checker_rejects_an_unresolved_digest_the_writer_refused``:
the validator used to accept a model identity whose checkpoint digest read
``"unresolved"`` while the writer raised on it, which is backwards for a
tamper-evidence artifact.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
    """A minimal checkout with one source file, one artifact and one figure."""
    root = tmp_path / "repo"
    (root / "crates" / "demo" / "src").mkdir(parents=True)
    (root / "paper" / "evidence" / "model-artifacts").mkdir(parents=True)
    (root / "paper" / "figures").mkdir(parents=True)
    # write_bytes throughout: write_text would translate newlines on Windows, and the
    # line-ending cases below need the on-disk bytes to be exactly what they say.
    (root / "Cargo.toml").write_bytes(b'[workspace]\nversion = "0.1.0"\n')
    (root / "crates" / "demo" / "src" / "lib.rs").write_bytes(b"pub fn one() -> u8 { 1 }\n")
    (root / "paper" / "evidence" / "f1.json").write_bytes(b'{"result": 1}\n')
    (root / "paper" / "figures" / "f1.pdf").write_bytes(b"%PDF-1.4\x00binary\r\nbytes\n")
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


@pytest.mark.parametrize("value", ["unresolved", "UNRESOLVED", "unknown", " ", ""])
def test_identity_summaries_rejects_a_placeholder_digest(value: str) -> None:
    identity = dict(COMPLETE_IDENTITY, digest=value)
    with pytest.raises(ValueError, match="digest"):
        common.identity_summaries(identity, FIELDS)


def test_identity_summaries_rejects_a_missing_or_non_string_field() -> None:
    with pytest.raises(ValueError, match="server_version"):
        common.identity_summaries(
            {k: v for k, v in COMPLETE_IDENTITY.items() if k != "server_version"}, FIELDS
        )
    with pytest.raises(ValueError, match="quantization"):
        common.identity_summaries(dict(COMPLETE_IDENTITY, quantization=4), FIELDS)


def test_identity_summaries_rejects_a_non_identity_payload() -> None:
    for payload in ([], "text", [COMPLETE_IDENTITY, "text"]):
        with pytest.raises(ValueError, match="identity object or array"):
            common.identity_summaries(payload, FIELDS)


def test_text_digests_ignore_line_endings_but_binary_digests_do_not(tmp_path: Path) -> None:
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
    assert recorded == {"Cargo.toml", "crates/demo/src/lib.rs"}
    assert {item["path"] for item in manifest["artifacts"]} == {
        "paper/evidence/f1.json",
        "paper/figures/f1.pdf",
    }
    assert manifest["generated_at_head_dirty"] is False

    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 0, checked.stdout


def test_checker_catches_an_edited_source_and_an_unrecorded_one(tree: Path) -> None:
    _make_and_commit(tree)

    (tree / "crates" / "demo" / "src" / "lib.rs").write_bytes(b"pub fn one() -> u8 { 2 }\n")
    edited = _run("check-provenance.py", tree)
    assert edited.returncode == 1
    assert "source: DIGEST crates/demo/src/lib.rs" in edited.stdout

    (tree / "crates" / "demo" / "src" / "lib.rs").write_bytes(b"pub fn one() -> u8 { 1 }\n")
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
    artifact.write_text(json.dumps(dict(COMPLETE_IDENTITY, digest="unresolved")), encoding="utf-8")
    made = _run("make-provenance.py", tree)
    assert made.returncode != 0
    assert "lacks explicit fields" in made.stderr


def test_checker_rejects_an_unresolved_digest_the_writer_refused(tree: Path) -> None:
    """A hand-edited manifest carrying an unresolved checkpoint digest must fail.

    This is the asymmetry the audit found: the writer raised on it, the validator
    returned the summary cleanly, so the untrusted half was the strict one.
    """
    artifact = tree / "paper" / "evidence" / "model-artifacts" / "runner.json"
    artifact.write_text(json.dumps(COMPLETE_IDENTITY), encoding="utf-8")
    _make_and_commit(tree)
    assert _run("check-provenance.py", tree).returncode == 0

    unresolved = dict(COMPLETE_IDENTITY, digest="unresolved")
    artifact.write_text(json.dumps(unresolved), encoding="utf-8")
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["model_artifacts"][0]
    entry["sha256"] = common.sha256(artifact)
    entry["identities"] = [unresolved]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "model artifact: INVALID" in checked.stdout
    assert "digest" in checked.stdout


def test_checker_rejects_a_manifest_claiming_a_clean_generation_on_a_dirty_tree(tree: Path) -> None:
    _make_and_commit(tree)
    manifest_path = tree / "paper" / "evidence" / "provenance.json"
    assert json.loads(manifest_path.read_text())["generated_at_head_dirty"] is False

    # Out of every recorded scope, so only the recomputed dirty flag can catch it.
    (tree / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
    checked = _run("check-provenance.py", tree)
    assert checked.returncode == 1
    assert "dirty: generated_at_head_dirty records a clean generation" in checked.stdout


def test_writer_records_a_dirty_generation_honestly(tree: Path) -> None:
    (tree / "notes.txt").write_text("uncommitted\n", encoding="utf-8")
    assert _run("make-provenance.py", tree).returncode == 0
    manifest = json.loads((tree / "paper" / "evidence" / "provenance.json").read_text())
    assert manifest["generated_at_head_dirty"] is True
    assert _run("check-provenance.py", tree).returncode == 0
