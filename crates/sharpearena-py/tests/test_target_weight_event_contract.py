"""One shared contract for the target-weight event, read the same way everywhere.

ARENA-REVIEW A20. Four places in this package read per-bar target weights off the same
``state["events"]`` stream, and before this contract existed they read it by three
different rules: ``mandate._weights_per_step`` accepted *any* dict carrying a ``weights``
key whatever its ``event`` name, ``rewards._weight_vectors`` and
``reward_misspecification._net_weights_per_bar`` required ``event == "target_weights"``,
and ``failure_taxonomy._valid_events`` carried a fourth, stricter validity rule of its own.
Canonical events written by :mod:`sharpearena.verifiers_env` satisfy all of them, so no
disagreement on a canonical stream was ever demonstrated; the defect is that nothing made
the rules one rule, so a non-canonical producer is graded by the mandate and ignored by the
reward, silently and in opposite directions.

These tests pin the contract itself: the name, the shape rule, and that every reader is
that one rule. They carry controls throughout so a contract that simply refused everything
could not pass them.
"""

from __future__ import annotations

import pytest

from sharpearena.event_contract import (
    TARGET_WEIGHTS_EVENT,
    MalformedTargetWeights,
    finite_target_weights,
    target_weight_vectors,
)
from sharpearena.mandate import Mandate, mandate_breach
from sharpearena.rewards import turnover_penalized

# A short leg. Under a `long_only` mandate the structural rule scores it 1.0 when it is
# read, and 0.0 when it is not, so it separates "this reader saw the bar" from "it did not"
# without depending on any threshold.
_SHORT = [-0.5, 0.2]
_FLAT = [0.5, 0.2]


def _named(weights: list[float]) -> dict:
    return {"event": TARGET_WEIGHTS_EVENT, "weights": list(weights)}


def _misnamed(weights: list[float]) -> dict:
    """The same payload under a name the contract does not define."""
    return {"event": "weights_update", "weights": list(weights)}


def test_the_canonical_event_is_read_by_every_path():
    """Control. The contract must not be a refusal engine: canonical events still grade."""
    long_only = Mandate(style="long_only")
    assert mandate_breach(long_only, [], [_named(_SHORT)]) == 1.0
    assert mandate_breach(long_only, [], [_named(_FLAT)]) == 0.0
    assert target_weight_vectors([_named(_FLAT)]) == [[0.5, 0.2]]


def test_a_weights_payload_under_another_name_is_refused_by_the_shared_reader():
    """The exact divergence A20 names: one reader graded this bar, the other never saw it.

    Before the contract, ``mandate_breach`` scored this event 1.0 (it accepted any dict
    carrying ``weights``) while ``turnover_penalized`` read no weight vectors from the same
    list at all (it required the name). The contract makes the stream single-valued: an
    unnamed weights payload is a producer bug and is refused, not read two ways.
    """
    with pytest.raises(MalformedTargetWeights):
        target_weight_vectors([_misnamed(_SHORT)])
    with pytest.raises(MalformedTargetWeights):
        mandate_breach(Mandate(style="long_only"), [], [_misnamed(_SHORT)])
    with pytest.raises(MalformedTargetWeights):
        turnover_penalized(state={"returns": [0.01], "events": [_misnamed(_SHORT)]})


def test_events_carrying_no_weights_are_market_records_and_pass_through():
    """Control. The stream also carries market-side records; those are not target weights."""
    market = [
        {"event": "margin_call", "nav": 1.0, "deficit": 0.5},
        {"event": "cascade_impact", "step": 1, "mark_drop": 0.1},
    ]
    assert target_weight_vectors(market) == []
    assert mandate_breach(Mandate(style="long_only"), [], market + [_named(_SHORT)]) == 1.0


def test_a_named_event_whose_weights_are_unusable_is_refused_not_skipped():
    """A ``target_weights`` record with no readable vector is a producer bug, not a bar."""
    for broken in ({"event": TARGET_WEIGHTS_EVENT}, {"event": TARGET_WEIGHTS_EVENT, "weights": 0.5}):
        with pytest.raises(MalformedTargetWeights):
            target_weight_vectors([broken])


def test_finiteness_stays_the_consumers_rule_not_the_readers():
    """The reader owns the name and the shape; the kernel owns the numbers.

    ``mandate_breach`` refuses a non-finite weight *by index* (A5), and that message is the
    evidence the fail-open repair rests on. The shared reader must therefore hand the vector
    through rather than pre-empting it with a shape error, while the stricter predicate
    :func:`finite_target_weights` — which :mod:`sharpearena.failure_taxonomy` needs — still
    answers the separate question.
    """
    from sharpearena.sharpearena_py import InvalidArgument

    nan_bar = _named([0.5, float("nan")])
    assert target_weight_vectors([nan_bar])[0][1] != target_weight_vectors([nan_bar])[0][1]
    with pytest.raises(InvalidArgument) as excinfo:
        mandate_breach(Mandate(style="long_only"), [], [nan_bar])
    assert "weights[0][1]" in str(excinfo.value)

    assert finite_target_weights(_named(_FLAT)) is True
    assert finite_target_weights(_named([float("inf")])) is False
    assert finite_target_weights({"event": TARGET_WEIGHTS_EVENT, "weights": []}) is False
    assert finite_target_weights({"event": "margin_call", "nav": 1.0}) is False


def test_every_reader_in_the_package_is_the_shared_reader():
    """No second implementation survives: the three private readers are the contract.

    A restatement would drift the way the four rules already had, so this asserts on the
    modules rather than on behavior alone.
    """
    from sharpearena import failure_taxonomy, mandate, reward_misspecification, rewards

    # The two verbatim readers are now the same function object, not two copies of it.
    assert mandate._weights_per_step is target_weight_vectors
    assert rewards._weight_vectors is target_weight_vectors

    # The third reads the same vectors and then reduces them, so it is pinned by behavior:
    # it reads the canonical bar and refuses the misnamed one the same way.
    assert reward_misspecification._net_weights_per_bar({"events": [_named(_FLAT)]}) == [0.7]
    with pytest.raises(MalformedTargetWeights):
        reward_misspecification._net_weights_per_bar({"events": [_misnamed(_FLAT)]})

    # The fourth asks the stricter question and answers it through the shared predicate.
    assert failure_taxonomy._valid_events([_named(_FLAT)]) is True
    assert failure_taxonomy._valid_events([_named([float("inf")])]) is False
