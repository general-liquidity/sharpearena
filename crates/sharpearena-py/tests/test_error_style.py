"""Machine-checked error-message style guide for the engine exception family.

The register, derived from the existing messages: ``[UPPER_SNAKE] `` code prefix,
a body starting lowercase (an all-caps acronym such as ``CSV`` is allowed), no
trailing period, and the body names the observable the code saw (the offending
value, bound, or position), never the inference. One REAL error is triggered per
reachable code; ``ENGINE_FAILURE`` is unreachable without an engine defect (its
type mapping is covered by ``test_engine_errors``). The Rust suite applies the
same register to the core crate's own typed errors (``error_style.rs``).
"""

from __future__ import annotations

import re

import pytest

from sharpearena.sharpearena_py import (
    DataUnavailable,
    InvalidArgument,
    InvalidJson,
    InvalidSalt,
    TradingEnv,
    generate_scenario_json,
    sealed_seed,
)

_PREFIX = re.compile(r"^\[([A-Z][A-Z0-9_]*)\] (.+)$", re.DOTALL)


def assert_register(message: str, *, observables: tuple[str, ...]) -> None:
    match = _PREFIX.match(message)
    assert match, f"missing or malformed [CODE] prefix: {message!r}"
    body = match.group(2)
    first_word = re.split(r"[ :]", body, maxsplit=1)[0]
    acronym = len(first_word) > 1 and first_word.isupper()
    assert body[0].islower() or acronym, (
        f"body must start lowercase (or an acronym): {message!r}"
    )
    assert not body.endswith("."), f"no trailing period: {message!r}"
    for observable in observables:
        assert observable in body, f"body must name the observable {observable!r}: {message!r}"


def _stepped_env() -> TradingEnv:
    env = TradingEnv.from_csv(
        "date,symbol,close\n2026-01-01,AAA,100\n2026-01-02,AAA,101\n2026-01-03,AAA,102\n",
        window_start=1,
        window_end=3,
    )
    env.reset()
    return env


# One real trigger per reachable code: (exception type, trigger, observables the
# message must carry).
_CASES = [
    (
        InvalidArgument,
        lambda: generate_scenario_json(1, distribution_mode="nope"),
        ('"nope"', "expected"),
    ),
    (InvalidJson, lambda: _stepped_env().step("not json"), ("line 1",)),
    (InvalidSalt, lambda: sealed_seed(b"short", 0), ("16", "5")),
    (DataUnavailable, lambda: TradingEnv.from_csv("not,a,header"), ("date",)),
]


@pytest.mark.parametrize(
    "exc_type,trigger,observables",
    _CASES,
    ids=[case[0].__name__ for case in _CASES],
)
def test_every_reachable_engine_error_obeys_the_register(exc_type, trigger, observables):
    with pytest.raises(exc_type) as excinfo:
        trigger()
    assert_register(str(excinfo.value), observables=observables)
