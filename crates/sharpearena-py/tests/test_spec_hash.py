"""The cross-surface SPEC_HASH handshake, on the Python surface.

Three pins must agree: the committed attestation record
(``crates/sharpearena/contract/attestation/spec-hash.json``), the pure-Python
package's pin (``sharpearena._spec_hash.EXPECTED_SPEC_HASH``), and what the compiled
pyo3 extension actually reports. And a stale pairing must refuse with the named
diagnosis, never a wrong number or a bare ``AttributeError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sharpearena
from sharpearena import EXPECTED_SPEC_HASH, SpecHashMismatch, check_spec_hash
from sharpearena._spec_hash import engine_spec_hash
from sharpearena.sharpearena_py import spec_hash

_RECORD = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "sharpearena"
        / "contract"
        / "attestation"
        / "spec-hash.json"
    ).read_text(encoding="utf-8")
)


def test_engine_wrapper_and_attestation_record_agree() -> None:
    assert spec_hash() == _RECORD["spec_hash"], (
        "the compiled extension reports a different spec hash than the attestation "
        "record; rebuild it (maturin develop) or rebind the record"
    )
    assert EXPECTED_SPEC_HASH == _RECORD["spec_hash"], (
        "_spec_hash.py pin drifted from the attestation record"
    )


def test_package_import_ran_the_handshake() -> None:
    # The handshake lives at the top of __init__; a matched pairing imported fine.
    assert engine_spec_hash(sharpearena.sharpearena_py) == EXPECTED_SPEC_HASH


def test_stale_wrapper_is_refused_by_name() -> None:
    stale = "00000000deadbeef"
    with pytest.raises(SpecHashMismatch) as exc:
        check_spec_hash(spec_hash(), expected=stale)
    message = str(exc.value)
    assert f"engine spec 0x{spec_hash()}" in message
    assert f"wrapper built against 0x{stale}" in message


def test_engine_predating_the_handshake_is_refused_by_name() -> None:
    # The lenient leg: an engine module with no spec_hash export reads as None and is
    # diagnosed by name instead of raising AttributeError.
    class _AncientEngine:
        pass

    assert engine_spec_hash(_AncientEngine()) is None
    with pytest.raises(SpecHashMismatch, match="predates the SPEC_HASH handshake"):
        check_spec_hash(None)


def test_matched_pairing_passes() -> None:
    check_spec_hash(EXPECTED_SPEC_HASH)
