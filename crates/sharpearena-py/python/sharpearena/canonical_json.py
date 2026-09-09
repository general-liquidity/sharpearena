"""``sharpebench/canonical-json/v1``: the bytes a forecast-contract digest hashes.

A contract digest is only an identity if every producer, in every language,
serializes the same document to the same bytes.  Language defaults do not: for
``1e-5`` Python's ``repr`` emits ``1e-05`` while Rust's shortest-round-trip
``Display`` emits ``0.00001``, so a contract hashed here was refused by the Rust
consumer as an unknown digest (audit row R07).  The normative definition is
``crates/sharpebench-protocol/src/canonical.rs`` in SharpeBench; this module is
its Python restatement and is pinned to the same bytes by the vectors in
``tests/test_canonical_json.py``.

The form:

1. Numbers use the ECMAScript ``Number::toString`` rendering, the numeric form
   RFC 8785 adopts: shortest round-tripping digits, fixed point while the
   decimal exponent ``n`` satisfies ``-6 < n <= 21``, exponential outside it
   with an explicit sign and no zero padding.  ``-0.0``, ``0.0`` and
   integer-valued floats render as integers.  Non-finite values have no form.
2. Strings use RFC 8785 escaping: only ``"``, ``\\`` and the C0 controls are
   escaped, the short escapes ``\\b \\t \\n \\f \\r`` are preferred and
   ``\\u00xx`` covers the rest; everything else passes through as UTF-8.
3. Object members are ordered by Unicode code point (a documented divergence
   from RFC 8785's UTF-16 order; it is what Python and Rust sort by).
4. Arrays keep their order; ``null``, ``true`` and ``false`` are literal.

:func:`versioned_preimage` frames the text as
``version | 0x00 | big-endian u64 byte length | body`` so a digest taken under
one version can never equal a digest of the same document under another.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence

CANONICAL_JSON_VERSION = "sharpebench/canonical-json/v1"


class CanonicalJsonError(ValueError):
    """The value has no canonical form under :data:`CANONICAL_JSON_VERSION`."""


_REPR = re.compile(r"(?P<integer>\d+)(?:\.(?P<fraction>\d+))?(?:e(?P<exponent>[+-]\d+))?")


def _signed_exponent(exponent: int) -> str:
    return f"-{-exponent}" if exponent < 0 else f"+{exponent}"


def canonical_number(value: float) -> str:
    """The canonical text of one finite float.

    Python's ``repr`` already yields the shortest round-tripping digits; only
    their layout differs from the specified form, so the digits are lifted out
    of ``repr`` and re-laid per the ECMAScript algorithm instead of trusting
    ``repr``'s own choice between fixed point and exponential.
    """

    if not math.isfinite(value):
        raise CanonicalJsonError(f"canonical JSON has no form for the non-finite {value!r}")
    if value == 0.0:
        return "0"
    match = _REPR.fullmatch(repr(abs(value)))
    if match is None:  # pragma: no cover - repr of a finite float always matches
        raise CanonicalJsonError(f"unrecognised float repr {value!r}")
    integer = match.group("integer")
    fraction = match.group("fraction") or ""
    exponent = int(match.group("exponent") or 0)
    all_digits = integer + fraction
    leading_zeros = len(all_digits) - len(all_digits.lstrip("0"))
    # The value is 0.<digits> * 10**n.
    n = len(integer) + exponent - leading_zeros
    digits = all_digits[leading_zeros:].rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    elif k == 1:
        body = digits + "e" + _signed_exponent(n - 1)
    else:
        body = digits[0] + "." + digits[1:] + "e" + _signed_exponent(n - 1)
    return "-" + body if value < 0 else body


_SHORT_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _write_string(text: str, out: list[str]) -> None:
    out.append('"')
    for character in text:
        escaped = _SHORT_ESCAPES.get(character)
        if escaped is not None:
            out.append(escaped)
        elif character < " ":
            out.append(f"\\u{ord(character):04x}")
        else:
            out.append(character)
    out.append('"')


def _write(value: object, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, float):
        out.append(canonical_number(value))
    elif isinstance(value, str):
        _write_string(value, out)
    elif isinstance(value, Mapping):
        out.append("{")
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise CanonicalJsonError("object members must have string keys")
        for index, key in enumerate(sorted(keys)):
            if index:
                out.append(",")
            _write_string(key, out)
            out.append(":")
            _write(value[key], out)
        out.append("}")
    elif isinstance(value, Sequence):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out)
        out.append("]")
    else:
        raise CanonicalJsonError(f"no canonical JSON form for {type(value).__name__}")


def canonical_json_v1(value: object) -> str:
    """The canonical text of a whole document under :data:`CANONICAL_JSON_VERSION`."""

    out: list[str] = []
    _write(value, out)
    return "".join(out)


def versioned_preimage(value: object) -> bytes:
    """The bytes to hash: ``version | 0x00 | big-endian u64 body length | body``."""

    body = canonical_json_v1(value).encode("utf-8")
    return (
        CANONICAL_JSON_VERSION.encode("ascii")
        + b"\x00"
        + len(body).to_bytes(8, "big")
        + body
    )


def canonical_sha256_v1(value: object) -> str:
    """SHA-256 over :func:`versioned_preimage`, the current contract digest."""

    return hashlib.sha256(versioned_preimage(value)).hexdigest()


__all__ = [
    "CANONICAL_JSON_VERSION",
    "CanonicalJsonError",
    "canonical_json_v1",
    "canonical_number",
    "canonical_sha256_v1",
    "versioned_preimage",
]
