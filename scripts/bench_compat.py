"""Shared checks for building SharpeArena against a SharpeBench other than the pin.

Used by ``check-against-bench-main.py`` (the nightly overlay of Bench main) and
``bump-bench-pin.py`` (the release-round pin bump). Both need the same two answers:
which Bench crates a Cargo patch can stand in for, and whether every top-level
``CompositeScore`` key that looks like a withholding field is one Arena's kernel
reader has been taught.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
#: The two lockfiles a patched build rewrites: the workspace's and the standalone
#: pyo3 crate's.
LOCKFILES = ("Cargo.lock", "crates/sharpearena-py/Cargo.lock")
KERNEL_SCORE_PY = ROOT / "crates" / "sharpearena-py" / "python" / "sharpearena" / "kernel_score.py"

_STRUCT_START = re.compile(r"^pub struct CompositeScore \{\s*$", re.MULTILINE)
_FIELD = re.compile(r"^\s*pub\s+(?:r#)?(?P<name>[a-z_][a-z0-9_]*)\s*:", re.MULTILINE)
_SERDE_RENAMING = re.compile(r"#\[serde\([^\]]*\b(?:rename|flatten|alias)\b")


class BenchCompatError(RuntimeError):
    """A Bench checkout or source tree could not be read the way this check needs."""


def published_crates(bench: Path) -> dict[str, Path]:
    """Every SharpeBench library crate in a checkout that a patch can stand in for."""
    crates: dict[str, Path] = {}
    for manifest in sorted((bench / "crates").glob("*/Cargo.toml")):
        package = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {})
        name = package.get("name", "")
        if name.startswith("sharpebench") and name != "sharpebench-py":
            crates[name] = manifest.parent.resolve()
    if "sharpebench-sim" not in crates:
        raise BenchCompatError(f"{bench} does not look like a SharpeBench checkout")
    return crates


def workspace_version(bench: Path) -> str:
    """The ``[workspace.package] version`` of a SharpeBench checkout."""
    manifest = tomllib.loads((bench / "Cargo.toml").read_text(encoding="utf-8"))
    version = manifest.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str):
        raise BenchCompatError(f"{bench}/Cargo.toml has no workspace package version")
    return version


def patch_flags(crates: dict[str, Path]) -> list[str]:
    """Cargo ``--config`` flags that overlay the crates without editing a tracked file."""
    flags: list[str] = []
    for name, path in crates.items():
        flags += ["--config", f'patch.crates-io.{name}.path="{path.as_posix()}"']
    return flags


@contextlib.contextmanager
def lockfiles_preserved(root: Path = ROOT) -> Iterator[None]:
    """Put both lockfiles back byte for byte however the block exits.

    A build with a ``[patch.crates-io]`` path override rewrites the lockfile: the
    patched crates lose their crates.io source and checksum and the unused patches
    are appended. Left in place, that is a path dependency one ``git add`` away from
    a commit.
    """
    saved = {name: (root / name).read_bytes() for name in LOCKFILES}
    try:
        yield
    finally:
        for name, data in saved.items():
            path = root / name
            if path.read_bytes() != data:
                path.write_bytes(data)


def composite_score_fields(source: str) -> list[str]:
    """The serialized top-level field names of ``CompositeScore`` in ``composite.rs``.

    Refuses a struct that renames, flattens or aliases a field, because then the Rust
    field name is not the JSON key and this listing would audit the wrong names.
    """
    start = _STRUCT_START.search(source)
    if start is None:
        raise BenchCompatError("composite.rs has no `pub struct CompositeScore {`")
    end = source.find("\n}", start.end())
    if end < 0:
        raise BenchCompatError("CompositeScore has no closing brace at column 0")
    body = source[start.end() : end]
    if _SERDE_RENAMING.search(body):
        raise BenchCompatError(
            "CompositeScore renames, flattens or aliases a field; audit its keys by hand"
        )
    fields = [match.group("name") for match in _FIELD.finditer(body)]
    if not fields:
        raise BenchCompatError("CompositeScore declares no public fields")
    return fields


def load_kernel_score(path: Path = KERNEL_SCORE_PY) -> ModuleType:
    """Load ``sharpearena.kernel_score`` by path, without the native extension."""
    spec = importlib.util.spec_from_file_location("_arena_kernel_score", path)
    if spec is None or spec.loader is None:
        raise BenchCompatError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unclassified_kernel_keys(fields: list[str], kernel_score: ModuleType) -> list[str]:
    """Suffixed ``CompositeScore`` keys in none of the reader's three lists, sorted.

    Any such key makes every read of a composite that carries it raise
    ``KernelFieldUnrecognized``, so it has to be classified before Arena builds
    against that kernel.
    """
    known = (
        set(kernel_score.KERNEL_UNAVAILABILITY_KEYS)
        | set(kernel_score.KERNEL_REFUSAL_KEYS)
        | set(kernel_score.KERNEL_NON_WITHHOLDING_KEYS)
    )
    suffixes = tuple(kernel_score.KERNEL_WITHHOLDING_SUFFIXES)
    return sorted(name for name in fields if name.endswith(suffixes) and name not in known)


def kernel_key_problems(composite_rs: Path, kernel_score_py: Path = KERNEL_SCORE_PY) -> list[str]:
    """One message per unclassified key in ``composite_rs``; empty when all are known."""
    fields = composite_score_fields(composite_rs.read_text(encoding="utf-8"))
    unknown = unclassified_kernel_keys(fields, load_kernel_score(kernel_score_py))
    return [
        f"CompositeScore key {name!r} ({composite_rs}) is in no kernel_score list; "
        "classify it in crates/sharpearena-py/python/sharpearena/kernel_score.py "
        "and docs/kernel-field-convention.md"
        for name in unknown
    ]
