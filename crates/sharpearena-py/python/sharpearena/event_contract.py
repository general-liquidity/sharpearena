"""The shared contract for the per-bar target-weight event.

A rollout's ``state["events"]`` list is one stream with several readers: the mandate breach
checker reads structural constraints off the per-bar weight vectors
(:mod:`sharpearena.mandate`), the turnover reward reads the same vectors to price churn
(:mod:`sharpearena.rewards`), the reward-misspecification controls reduce them to a net
directional bet (:mod:`sharpearena.reward_misspecification`), and the failure taxonomy asks
whether the stream is well-formed enough to classify (:mod:`sharpearena.failure_taxonomy`).

Before this module those four readers each carried their own rule, and two of them
disagreed: the mandate adapter accepted *any* dict carrying a ``weights`` key whatever its
``event`` name, while the reward adapters required ``event == "target_weights"``. Canonical
events, written by :mod:`sharpearena.verifiers_env`, satisfy both, so no disagreement on a
canonical stream was ever demonstrated. What was missing is the contract itself: nothing
made the rules one rule, so a non-canonical producer would be graded by the mandate and
ignored by the reward, in opposite directions and silently. See
``docs/audits/2026-09-09/ARENA-REVIEW.md`` A20.

The contract is deliberately narrow:

* :data:`TARGET_WEIGHTS_EVENT` is the one name a per-bar weight vector travels under.
* :func:`target_weight_vectors` is the one reader. It refuses a ``weights`` payload sent
  under any other name, and refuses a ``target_weights`` record whose payload is not a
  vector, rather than skipping either. Skipping is what let the two rules diverge without
  anyone noticing.
* Finiteness is **not** the reader's rule. ``mandate_breach`` refuses a non-finite weight in
  the Rust kernel and names the offending index (ARENA-REVIEW A5), and that message is the
  evidence the fail-open repair rests on; a shape error raised here first would replace it.
  :func:`finite_target_weights` answers that separate, stricter question for the caller that
  needs it as a predicate rather than as a refusal.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

#: The event name a per-bar target-weight vector travels under, everywhere.
TARGET_WEIGHTS_EVENT = "target_weights"


class MalformedTargetWeights(ValueError):
    """A weight payload no reader can agree on, refused instead of read two ways.

    Raised for a ``weights`` payload sent under a name the contract does not define, and
    for a :data:`TARGET_WEIGHTS_EVENT` record whose ``weights`` is not a vector. Both are
    producer bugs: the stream they describe would be graded by one consumer and invisible
    to another.
    """


def is_target_weights(event: Any) -> bool:
    """Whether ``event`` is a per-bar target-weight record by name."""
    return isinstance(event, dict) and event.get("event") == TARGET_WEIGHTS_EVENT


def finite_target_weights(event: Any) -> bool:
    """Whether ``event`` is a target-weight record carrying a usable, finite vector.

    The stricter question, kept separate from :func:`target_weight_vectors` so the kernel
    stays the thing that refuses a non-finite number and says which one.
    """
    if not is_target_weights(event):
        return False
    weights = event.get("weights")
    if not isinstance(weights, (list, tuple)) or not weights:
        return False
    return all(isinstance(w, Real) and math.isfinite(float(w)) for w in weights)


def target_weight_vectors(events: Any) -> list[list[float]]:
    """The per-bar target-weight vectors in ``events``, in order.

    Records carrying no ``weights`` payload are market-side facts (``margin_call``,
    ``cascade_impact``, …) and pass through. A ``weights`` payload under any other name, or
    a :data:`TARGET_WEIGHTS_EVENT` record whose payload is not a vector, raises
    :class:`MalformedTargetWeights`.
    """
    out: list[list[float]] = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        named = event.get("event") == TARGET_WEIGHTS_EVENT
        if "weights" in event and not named:
            raise MalformedTargetWeights(
                f"[INVALID_ARGUMENT] {event.get('event')!r} carries a weights payload; a "
                f"per-bar weight vector travels only as {TARGET_WEIGHTS_EVENT!r}, and an "
                "event read by one consumer and ignored by another is refused here"
            )
        if not named:
            continue
        weights = event.get("weights")
        if not isinstance(weights, (list, tuple)):
            raise MalformedTargetWeights(
                f"[INVALID_ARGUMENT] a {TARGET_WEIGHTS_EVENT!r} record must carry a weights "
                f"vector, got {weights!r}"
            )
        out.append([float(x) for x in weights])
    return out


__all__ = [
    "TARGET_WEIGHTS_EVENT",
    "MalformedTargetWeights",
    "finite_target_weights",
    "is_target_weights",
    "target_weight_vectors",
]
