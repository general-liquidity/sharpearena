"""The Python style table against the native enum's own labels.

ARENA-REVIEW A7. ``sharpearena.mandate.STYLES`` was a hand-copied tuple of the wire labels
``sharpearena::mandate::MandateStyle`` serializes to, and the pyo3 module exported no style
list, so nothing could derive it and nothing cross-checked it. The labels agree today, and
since the A6 repair an unsupported style raises :class:`MandateError` at the reward boundary
rather than being awarded full credit, so **no present numerical mismatch is demonstrated
here**. What is demonstrated is that the drift is invisible: the table and the enum were two
independent lists, and the next style added upstream would land in only one of them.

The repair follows PR #62: the vocabulary is emitted by the native extension through a
wildcard-free exhaustive match, so a variant added by an upstream change stops the crate
compiling instead of reappearing as a silent gap, and the Python tuple is derived from that
contract rather than restating it.
"""

from __future__ import annotations

import json

import pytest

from sharpearena.mandate import (
    STYLES,
    Mandate,
    MandateError,
    require_mandate,
    sample_mandate,
    validate_mandate,
)
from sharpearena.sharpearena_py import mandate_style_contract


def _contract() -> list[str]:
    document = json.loads(mandate_style_contract())
    assert document["schema_version"] == 1
    return list(document["styles"])


def test_the_style_table_is_the_native_enums_own_labels():
    """Not a restatement: the same object the native extension emitted, in its own order.

    Order is part of the contract, not an accident. ``sample_mandate`` draws a style by
    indexing the enum's canonical order, so a tuple that agreed as a *set* but not as a
    sequence would still describe a different draw.
    """
    assert list(STYLES) == _contract()


def test_every_label_the_contract_names_is_a_style_the_package_accepts():
    """Round trip, both directions, so neither list can be a superset of the other."""
    for label in _contract():
        assert label in STYLES
        assert validate_mandate(Mandate(style=label)) is True
        require_mandate(Mandate(style=label))


def test_a_style_the_contract_does_not_name_is_still_refused():
    """Control. The table is derived, not widened: an unknown style stays a refusal (A6)."""
    unknown = "long_short_equity"
    assert unknown not in _contract()
    assert validate_mandate(Mandate(style=unknown)) is False
    with pytest.raises(MandateError):
        require_mandate(Mandate(style=unknown))


def test_a_style_added_upstream_but_not_to_the_table_is_caught():
    """The hazard itself, demonstrable only against a synthetic future variant.

    A6 already refuses an unrecognized style, so there is no wrong score to reproduce on
    today's tree. What A7 names is drift: before the repair, ``STYLES`` was a literal that a
    new native label could not disturb, and this assertion would have passed against a
    mutilated table. It is written against the contract so it cannot.
    """
    future = _contract() + ["volatility_targeting"]
    assert set(future) - set(STYLES) == {"volatility_targeting"}, (
        "STYLES must be exactly the contract, so a label the contract does not yet name is "
        "the only difference"
    )


def test_sampled_mandates_draw_only_contract_styles():
    """Control over the real generator, so the contract is not checked against itself alone."""
    drawn = {sample_mandate(seed).style for seed in range(256)}
    assert drawn, "the generator drew nothing"
    assert drawn <= set(_contract())
