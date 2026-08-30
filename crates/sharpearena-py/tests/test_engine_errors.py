"""The typed-error taxonomy at the pyo3 boundary.

Every engine error carries a stable ``[CODE] `` prefix and maps to a class under
``SharpeArenaError``. The codes are a compatibility surface: a wheel and a wrapper can be
different versions, so an unknown code must reach the caller as a generic exception that
still *carries* the code rather than being relabelled as something this build knows.
"""

import pytest

import sharpearena as sa
from sharpearena.sharpearena_py import _raise_coded

CSV_GARBAGE = "not,a,dataset\n"


def _raises(exc_type, code, fn):
    with pytest.raises(exc_type) as excinfo:
        fn()
    message = str(excinfo.value)
    assert message.startswith(f"[{code}] "), message
    # The taxonomy sits under ValueError, which is what the boundary raised before it
    # existed, so no consumer catching ValueError is broken by the change.
    assert isinstance(excinfo.value, sa.SharpeArenaError)
    assert isinstance(excinfo.value, ValueError)
    return message


def test_unknown_enum_label_is_invalid_argument():
    _raises(
        sa.InvalidArgument,
        "INVALID_ARGUMENT",
        lambda: sa.TradingEnv(distribution_mode="no-such-mode"),
    )


def test_malformed_decision_is_invalid_json():
    env = sa.TradingEnv()
    env.reset()
    _raises(sa.InvalidJson, "INVALID_JSON", lambda: env.step("{not json"))


def test_short_salt_is_invalid_salt():
    # The code is stamped by the Rust core's SealedSaltError and relayed, not re-derived
    # at the boundary, so this also pins the relay path on a real core error.
    message = _raises(
        sa.InvalidSalt,
        "INVALID_SALT",
        lambda: sa.sharpearena_py.sealed_seed(b"tooshort", 0),
    )
    assert "at least 16 bytes" in message


def test_unparseable_csv_is_data_unavailable():
    _raises(
        sa.DataUnavailable,
        "DATA_UNAVAILABLE",
        lambda: sa.TradingEnv.from_csv(CSV_GARBAGE),
    )


def test_unknown_code_degrades_to_the_base_class_and_keeps_the_code():
    # A wrapper older than the wheel: the code exists, this build does not know it.
    with pytest.raises(sa.SharpeArenaError) as excinfo:
        _raise_coded("A_CODE_FROM_THE_FUTURE", "something the engine knows about")
    message = str(excinfo.value)
    assert message == "[A_CODE_FROM_THE_FUTURE] something the engine knows about"
    # Generic, not mislabelled as one of the known classes.
    assert type(excinfo.value) is sa.SharpeArenaError


def test_known_codes_still_dispatch_through_the_relay():
    for code, exc_type in [
        ("INVALID_ARGUMENT", sa.InvalidArgument),
        ("INVALID_JSON", sa.InvalidJson),
        ("INVALID_SALT", sa.InvalidSalt),
        ("DATA_UNAVAILABLE", sa.DataUnavailable),
        ("ENGINE_FAILURE", sa.EngineFailure),
    ]:
        with pytest.raises(exc_type) as excinfo:
            _raise_coded(code, "body")
        assert str(excinfo.value) == f"[{code}] body"


def test_uncoded_message_falls_back_to_the_declared_code():
    with pytest.raises(sa.EngineFailure) as excinfo:
        _raise_coded("ENGINE_FAILURE", "a message with no bracketed prefix")
    assert str(excinfo.value) == "[ENGINE_FAILURE] a message with no bracketed prefix"
