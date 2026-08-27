"""Shared rules for the evidence provenance manifest.

The writer (``make-provenance.py``) and the validator (``check-provenance.py``) are
two halves of one tamper-evidence artifact, so anything either of them decides about
a file lives here and is used by both. When the two drifted apart the validator was
the laxer half, which is the wrong direction: the validator is the trusted one.

Three rules are worth stating explicitly.

*Digests are line-ending normalized.* A digest is taken over the file's content with
CRLF collapsed to LF, so the manifest records the same value for a checkout made on
Windows as for one made on Linux, which is what the appendix's "the same tree yields
the same snapshot hash on a different machine" requires. Files carrying a NUL byte
are treated as binary and hashed verbatim; that is git's own text/binary heuristic.

*Dirtiness is recomputed, not read.* ``generated_at_head_dirty`` is a claim the
manifest makes about itself, so the validator recomputes it against the tree it is
checking rather than believing the recorded value.

*A clean generation is a falsifiable claim.* ``generated_at_head_dirty: false`` says the
recorded bytes were the committed bytes of ``generated_at_head``, which
:func:`blobs_at_commit` lets the validator check against that commit rather than take on
trust. Before that check, flipping the flag by hand was indistinguishable from
regenerating on a clean checkout.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


MODEL_IDENTITY_FIELDS = (
    "model",
    "digest",
    "quantization",
    "server",
    "server_version",
)

# A backend fact the local-model recorder could not observe is written as one of these
# rather than invented. That is honest inside a model identity file and inadmissible in
# a provenance manifest, which exists to say what the checkpoint actually was.
UNRESOLVED_VALUES = frozenset({"unknown", "unresolved"})

# `crates/**/*.rs` otherwise matches machine-local build outputs under
# `crates/*/target/**/build/*/out/*.rs`, which are neither source nor
# reproducible across machines.
EXCLUDED_DIR_NAMES = ("target", ".venv", "__pycache__", "node_modules", ".git")


def repo_root() -> Path:
    """The tree to read. ``SHARPEARENA_PROVENANCE_ROOT`` points both scripts at another
    checkout, which is how the test suite exercises them without writing to this one."""
    override = os.environ.get("SHARPEARENA_PROVENANCE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def digest_bytes(data: bytes) -> str:
    """SHA-256 over `data`, with CRLF collapsed to LF unless it carries a NUL byte."""
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    """SHA-256 over the file, with CRLF collapsed to LF for text files."""
    return digest_bytes(path.read_bytes())


def is_excluded(path: Path, root: Path, excludes: frozenset[str]) -> bool:
    return any(part in excludes for part in path.relative_to(root).parts)


def expand(patterns: tuple[str, ...] | list[str], root: Path, excludes: frozenset[str]) -> list[Path]:
    """Every file matching any glob, excluding build and environment directories."""
    found: set[Path] = set()
    for pattern in patterns:
        found.update(
            path
            for path in root.glob(pattern)
            if path.is_file() and not is_excluded(path, root, excludes)
        )
    return sorted(found, key=lambda path: path.as_posix())


def snapshot_digest(records: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in records).encode()
    ).hexdigest()


def identity_summaries(payload: object, fields: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    """The load-bearing fields of a model identity file, or a ``ValueError``.

    A field that is missing, non-string, blank or one of :data:`UNRESOLVED_VALUES`
    rejects the file. Both the writer and the validator go through here, so a manifest
    the writer would have refused to produce cannot pass the validator.
    """
    identities = payload if isinstance(payload, list) else [payload]
    if not identities or not all(isinstance(item, dict) for item in identities):
        raise ValueError("must contain an identity object or array")
    summaries: list[dict[str, str]] = []
    for index, identity in enumerate(identities):
        missing = [
            field
            for field in fields
            if not isinstance(identity.get(field), str)
            or not identity[field].strip()
            or identity[field].strip().lower() in UNRESOLVED_VALUES
        ]
        if missing:
            raise ValueError(f"identity {index} lacks explicit fields: {missing}")
        summaries.append({field: identity[field] for field in fields})
    return summaries


def working_tree_dirty(root: Path) -> bool | None:
    """Whether `root` has uncommitted work, or ``None`` when that cannot be determined.

    ``git diff`` applies git's own checkout filters, so a working tree that differs from
    ``HEAD`` only in line endings reads as clean here. ``git status --porcelain`` does
    not, which is why it is not used.
    """
    try:
        tracked = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=root, capture_output=True
        )
        if tracked.returncode not in (0, 1):
            return None
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return tracked.returncode == 1 or bool(untracked.strip())


MISSING_BLOB = object()


def blobs_at_commit(
    root: Path, commit: str, paths: list[str]
) -> dict[str, bytes | object] | None:
    """The committed bytes of `paths` at `commit`, or ``None`` if that commit is not here.

    A path absent from that commit maps to :data:`MISSING_BLOB`. ``None`` is returned when
    the object cannot be read at all, which is what a shallow clone that does not contain
    the commit looks like; the caller reports that rather than treating it as a mismatch.
    """
    try:
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, capture_output=True
        )
    except OSError:
        return None
    if present.returncode != 0:
        return None
    if not paths:
        return {}
    request = "".join(f"{commit}:{path}\n" for path in paths).encode()
    try:
        process = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=root, input=request, capture_output=True
        )
    except OSError:
        return None
    if process.returncode != 0:
        return None
    out = process.stdout
    blobs: dict[str, bytes | object] = {}
    offset = 0
    for path in paths:
        end = out.find(b"\n", offset)
        if end == -1:
            return None
        header = out[offset:end].split(b" ")
        if header[-1] in (b"missing", b"ambiguous"):
            blobs[path] = MISSING_BLOB
            offset = end + 1
            continue
        if len(header) != 3:
            return None
        size = int(header[2])
        blobs[path] = out[end + 1 : end + 1 + size]
        offset = end + 1 + size + 1
    return blobs


def head_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
