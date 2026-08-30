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
import re
import subprocess
from pathlib import Path, PurePosixPath

MODEL_IDENTITY_FIELDS = (
    "model",
    "digest",
    "quantization",
    "server",
    "server_version",
)

# The source identity files use the same closed object schema as the Python runtime's
# ``ModelIdentity`` dataclass. Keep this module stdlib-only so the provenance writer can
# run before the package and its scientific dependencies are installed; the test suite
# compares this set directly with ``dataclasses.fields(ModelIdentity)`` so either side
# changing alone fails. Optional fields remain admissible even though only the five
# load-bearing fields above are copied into the compact provenance summary.
MODEL_IDENTITY_ALLOWED_FIELDS = frozenset(
    {
        "model",
        "digest",
        "parameter_size",
        "quantization",
        "offload",
        "family",
        "context_length",
        "server",
        "server_version",
        "size_bytes",
        "format",
        "capabilities",
        "license_sha256",
        "modelfile_sha256",
        "template_sha256",
        "parameters_sha256",
        "quantizer",
        "converter_version",
        "quantization_calibration",
        "server_commit",
        "wrapper",
        "wrapper_version",
        "chat_template",
        "reasoning_parser",
        "tool_parser",
        "constrained_decoding_backend",
        "kv_cache_dtype",
        "tensor_parallelism",
        "batch_size",
        "parallel_slots",
        "prefix_cache",
        "speculative_decoding",
        "gpu_name",
        "gpu_memory_mib",
        "gpu_driver_version",
        "gpu_compute_capability",
        "cuda_version",
    }
)

# These are trust policy, not manifest input.  The manifest repeats them so a reader
# can see what was bound, but the writer, working-tree checker, and release-tag
# checker all require exact equality with these values.  Otherwise deleting a glob,
# adding an exclusion, or reducing the identity fields in the manifest would also
# redefine what the validator considers complete.
SOURCE_SCOPE = (
    ".gitattributes",
    ".github/workflows/*.yml",
    "Cargo.toml",
    "Cargo.lock",
    "release.toml",
    "crates/**/*.toml",
    "crates/**/*.rs",
    "crates/sharpearena-py/python/**/*.py",
    "scripts/*.py",
    "paper/src/*.py",
    "paper/main.tex",
    "paper/sections/*.tex",
    "paper/refs.bib",
)

ARTIFACT_SCOPE = ("paper/evidence/*.json", "paper/figures/*.pdf")
MODEL_ARTIFACT_SCOPE = ("paper/evidence/model-artifacts/*.json",)

# A backend fact the local-model recorder could not observe is written as one of these
# rather than invented. That is honest inside a model identity file and inadmissible in
# a provenance manifest, which exists to say what the checkpoint actually was.
UNRESOLVED_VALUES = frozenset({"unknown", "unresolved"})

# `crates/**/*.rs` otherwise matches machine-local build outputs under
# `crates/*/target/**/build/*/out/*.rs`, which are neither source nor
# reproducible across machines.
EXCLUDED_DIR_NAMES = ("target", ".venv", "__pycache__", "node_modules", ".git")

SCHEMA_VERSION = 5
DIGEST_CONVENTION = (
    "sha256 over file bytes with CRLF collapsed to LF; files containing "
    "a NUL byte are hashed verbatim"
)
SOURCE_SCOPE_NOTE = (
    "Globs are expanded from the repository root; any path with a component "
    "in source_snapshot_excludes is skipped, so build outputs and virtual "
    "environments are not part of the source snapshot."
)
REPRODUCTION_ENTRYPOINT = "commands in paper/sections/A-commands.tex"
VALIDATOR = "paper/src/check-provenance.py"

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "generated_at_head",
        "generated_at_head_dirty",
        "digest_convention",
        "source_snapshot_sha256",
        "source_snapshot_scope",
        "source_snapshot_excludes",
        "source_snapshot_scope_note",
        "artifact_scope",
        "model_artifact_scope",
        "model_identity_fields",
        "reproduction_entrypoint",
        "validator",
        "source_files",
        "artifacts",
        "model_artifacts",
    }
)
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _record_problems(name: str, records: object, *, model: bool = False) -> list[str]:
    if not isinstance(records, list):
        return [f"manifest rule: {name} must be a list"]
    expected_keys = {"path", "sha256", "identities"} if model else {"path", "sha256"}
    problems: list[str] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_keys:
            problems.append(
                f"manifest rule: {name}[{index}] must have exactly {sorted(expected_keys)}"
            )
            continue
        path = record["path"]
        digest = record["sha256"]
        posix = PurePosixPath(path) if isinstance(path, str) else None
        if (
            not isinstance(path, str)
            or not path
            or posix is None
            or posix.is_absolute()
            or posix.as_posix() != path
            or "\\" in path
            or any(char in path for char in "\x00\r\n")
            or ".." in posix.parts
        ):
            problems.append(f"manifest rule: {name}[{index}] has an invalid path")
        elif path in seen_paths:
            problems.append(f"manifest rule: {name} repeats path {path}")
        else:
            seen_paths.add(path)
        if not isinstance(digest, str) or _HEX_DIGEST.fullmatch(digest) is None:
            problems.append(f"manifest rule: {name}[{index}] has an invalid sha256")
        if model:
            identities = record["identities"]
            try:
                summaries = identity_summaries(identities, MODEL_IDENTITY_FIELDS)
            except ValueError as error:
                problems.append(
                    f"model artifact: INVALID {name}[{index}] identities: {error}"
                )
                continue
            if identities != summaries:
                problems.append(
                    f"manifest rule: {name}[{index}] identities have non-canonical fields"
                )
    return problems


def manifest_rule_problems(manifest: dict) -> list[str]:
    """Reject a manifest that attempts to redefine the validator's trust policy."""
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "digest_convention": DIGEST_CONVENTION,
        "source_snapshot_scope": list(SOURCE_SCOPE),
        "source_snapshot_excludes": list(EXCLUDED_DIR_NAMES),
        "source_snapshot_scope_note": SOURCE_SCOPE_NOTE,
        "artifact_scope": list(ARTIFACT_SCOPE),
        "model_artifact_scope": list(MODEL_ARTIFACT_SCOPE),
        "model_identity_fields": list(MODEL_IDENTITY_FIELDS),
        "reproduction_entrypoint": REPRODUCTION_ENTRYPOINT,
        "validator": VALIDATOR,
    }
    if not isinstance(manifest, dict):
        return ["manifest rule: top level must be an object"]
    problems = [
        f"manifest rule: {field} does not equal the canonical validator rule"
        for field, expected in canonical.items()
        if manifest.get(field) != expected
    ]
    if set(manifest) != MANIFEST_FIELDS:
        missing = sorted(MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - MANIFEST_FIELDS)
        problems.append(
            f"manifest rule: top-level fields differ; missing={missing}, extra={extra}"
        )
    head = manifest.get("generated_at_head")
    if not isinstance(head, str) or _GIT_COMMIT.fullmatch(head) is None:
        problems.append("manifest rule: generated_at_head is not a full commit id")
    if not isinstance(manifest.get("generated_at_head_dirty"), bool):
        problems.append("manifest rule: generated_at_head_dirty must be boolean")
    snapshot = manifest.get("source_snapshot_sha256")
    if not isinstance(snapshot, str) or _HEX_DIGEST.fullmatch(snapshot) is None:
        problems.append("manifest rule: source_snapshot_sha256 is not a sha256")
    problems += _record_problems("source_files", manifest.get("source_files"))
    problems += _record_problems("artifacts", manifest.get("artifacts"))
    problems += _record_problems(
        "model_artifacts", manifest.get("model_artifacts"), model=True
    )
    return problems


def repo_root() -> Path:
    """The tree to read. ``SHARPEARENA_PROVENANCE_ROOT`` points both scripts at another
    checkout, which is how the test suite exercises them without writing to this one."""
    override = os.environ.get("SHARPEARENA_PROVENANCE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def _fsync_directory(directory: Path) -> None:
    """Persist the directory entry ``os.replace`` just rewrote.

    Without it the rename can still be lost on a crash even though the file's own
    contents were fsynced. Windows cannot open a directory handle for fsync at all
    (there is no ``O_DIRECTORY``), so this is a POSIX-only durability step; the rename
    itself is atomic on both.
    """

    if not hasattr(os, "O_DIRECTORY"):
        return
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so a reader sees the old file or the new one.

    The manifest is the artifact the whole tamper-evidence argument rests on, and it was
    written with a plain truncate-and-write: a crash, a full disk or a killed process
    mid-write left a half-manifest where the validator expects one of two whole files,
    and the failure looks like tampering rather than an interrupted write.

    Written to an ``O_EXCL`` temp file in the destination directory (so the rename stays
    on one filesystem), fsynced, then ``os.replace``d over the target and the parent
    directory fsynced. The temp name carries the pid and a sequence number, so two
    writers in the same directory cannot collide on it. Bytes are written in binary with
    LF line endings on every platform, which is what ``.gitattributes`` pins for the
    repository anyway. The temp file is removed on every failure path.
    """

    payload = text.encode("utf-8")
    directory = path.parent
    handle = None
    for sequence in range(1024):
        temporary = directory / f".{path.name}.tmp.{os.getpid()}.{sequence}"
        try:
            handle = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except FileExistsError:
            continue
        break
    if handle is None:
        raise OSError(f"could not create a temp file next to {path}")

    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    _fsync_directory(directory)


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


def expand(
    patterns: tuple[str, ...] | list[str], root: Path, excludes: frozenset[str]
) -> list[Path]:
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


def identity_summaries(
    payload: object, fields: list[str] | tuple[str, ...]
) -> list[dict[str, str]]:
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
        unknown = set(identity) - MODEL_IDENTITY_ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"identity {index} has unknown fields: {sorted(unknown)}")
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
            ["git", "diff", "--quiet", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
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
    the commit looks like. That is a failure, not a skip: the caller turns ``None`` into a
    problem saying the clean-generation claim cannot be verified, because a manifest whose
    named generation nobody can read back is exactly what an unverifiable claim looks like.
    """
    try:
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            capture_output=True,
            check=False,
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
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=request,
            capture_output=True,
            check=False,
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
