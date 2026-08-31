"""The cross-surface SPEC_HASH handshake for the Python package.

The Rust core's ``build.rs`` fingerprints the tape-semantics-defining sources
(FNV-1a/64 over CRLF-normalized bytes, seeded with a manual epoch) and compiles the
value into the engine; the pyo3 extension reports it as ``spec_hash()``. This module
pins the hash the pure-Python package was generated against
(``crates/sharpearena/contract/attestation/spec-hash.json``) and
:func:`check_spec_hash` refuses at package import when the compiled extension
disagrees, so a stale ``.pyd``/``.so`` in a dev venv driven by newer Python sources
(or a stale wheel layered under newer sources) is a named error, not a silently
wrong number.

The engine's ``spec_hash`` attribute is the ONE thing read leniently
(:func:`engine_spec_hash` returns ``None`` instead of raising when the extension
predates the handshake), precisely so a stale surface is diagnosed by name rather
than dying with an ``AttributeError`` that says nothing about versions.
"""

from __future__ import annotations

from typing import Optional

# The pin. Rebind together with contract/attestation/spec-hash.json (a Rust test and
# tests/test_spec_hash.py keep the three copies bound).
EXPECTED_SPEC_HASH = "2d912913eedd333e"


class SpecHashMismatch(RuntimeError):
    """The compiled engine and the pure-Python package disagree on tape semantics."""


def engine_spec_hash(engine: object) -> Optional[str]:
    """The engine's reported spec hash, or ``None`` when it predates the handshake."""
    fn = getattr(engine, "spec_hash", None)
    return None if fn is None else str(fn())


def check_spec_hash(engine_hash: Optional[str], expected: str = EXPECTED_SPEC_HASH) -> None:
    """Refuse a stale engine/wrapper pairing with a named diagnosis."""
    if engine_hash == expected:
        return
    if engine_hash is None:
        raise SpecHashMismatch(
            "the compiled sharpearena_py extension predates the SPEC_HASH handshake "
            f"(no spec_hash export); wrapper built against 0x{expected}. "
            "Rebuild the extension (maturin develop) from the same commit as the "
            "Python sources."
        )
    raise SpecHashMismatch(
        f"engine spec 0x{engine_hash}, wrapper built against 0x{expected}. "
        "The compiled sharpearena_py extension and the Python package come from "
        "different tape-semantics revisions; rebuild the extension (maturin develop) "
        "from the same commit as the Python sources."
    )
