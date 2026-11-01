"""Typed unavailability from the SharpeBench scoring kernel.

``score_run`` returns the kernel's ``CompositeScore`` as JSON. Since SharpeBench
0.19.0 a composite that the kernel could not score carries a typed error beside a
no-skill floor instead of a coerced estimate: ``deflation_error`` (the deflated
Sharpe, its bar and its interval are the floor), ``bootstrap_error`` (the bootstrap
p-value is the conservative 1.0 sentinel) and ``selection_error`` (the selection
diagnostic is absent). A non-finite observation sets both ``deflation_error`` and
``bootstrap_error`` and nulls ``psr``; fewer than two observations set
``bootstrap_error`` alone while ``deflated_sharpe`` reads ``0.0``. A track with no
Sharpe ratio, which the pinned kernel would score, is withheld by Arena's ``score_run``
in the same shape: ``deflation_error`` reads ``returns must not be constant: a constant
series has no Sharpe ratio`` for a constant track (or ``Sharpe ratio is not finite``
when the computed standard deviation underflows), ``deflated_sharpe`` and ``psr`` read
``0.0``, the interval is absent and ``passed_k`` is false. Reading
``composite["deflated_sharpe"]`` past such a key publishes the floor as a score,
which is the flattering substitution the kernel refused to make.

Keys are classified by explicit allowlists, never by suffix alone (the convention is
written up in ``docs/kernel-field-convention.md``). :data:`KERNEL_UNAVAILABILITY_KEYS`
withhold the score as unavailable. :data:`KERNEL_REFUSAL_KEYS`
(``standard_error_refusal`` and ``exposure_path_error``, which SharpeBench main emits
after 0.28.0) are gate refusals: the kernel computed the track and declined to rank
it, so the score is withheld as :class:`KernelScoreRefused` and recorded as
``refused_scoring_kernel_gate: <key>: <message>``. Any other top-level key ending in
``_error`` or ``_refusal`` raises :class:`KernelFieldUnrecognized`, which is never
recorded as a reason: the reader must be taught the key before it reads the composite.

Every consumer of ``score_run`` reads the ranked numbers through this module. A
consumer that must stop (a paper producer, a selection step) calls
:func:`kernel_deflated_sharpe` or :func:`kernel_psr` and lets
:class:`KernelScoreUnavailable` propagate; a consumer that records rows
(a snapshot, a leaderboard, a journal) calls :func:`kernel_score_or_unavailable`
and stores the reason string in place of the number, in the
``unavailable_scoring_kernel_error: <reason>`` form ``run_baselines`` already uses
for a withheld confidence interval. No path in this module returns a default.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Union

UNAVAILABLE_KERNEL_ERROR = "unavailable_scoring_kernel_error"
UNAVAILABLE_KERNEL_VALUE = "unavailable_scoring_kernel_value"
REFUSED_KERNEL_GATE = "refused_scoring_kernel_gate"

#: Top-level ``CompositeScore`` keys that say the kernel could not compute a figure
#: (SharpeBench main ``crates/sharpebench-core/src/composite.rs``). When one is
#: present the ranked numbers beside it are the no-skill floor or a sentinel.
KERNEL_UNAVAILABILITY_KEYS = frozenset({"bootstrap_error", "deflation_error", "selection_error"})

#: Top-level ``CompositeScore`` keys that say the kernel computed the track and a gate
#: refused it: ``standard_error_refusal`` (the standard-error certification, with the
#: ``NOT_CERTIFIED_UNDER_SERIAL_CORRELATION`` and ``NOT_CERTIFIED_WITHOUT_MOMENT_CREDIT``
#: messages) and ``exposure_path_error`` (the exposure-path gate,
#: ``EXPOSURE_PATH_UNPROVEN``). The deflated Sharpe beside one is an estimate, but the
#: kernel did not rank it, so Arena does not publish it as a score either.
KERNEL_REFUSAL_KEYS = frozenset({"exposure_path_error", "standard_error_refusal"})

#: Suffixes that mark a key as part of the kernel's withholding vocabulary. A key with
#: one of these suffixes that neither allowlist names is a field this reader does not
#: know, and the composite is not read at all.
KERNEL_WITHHOLDING_SUFFIXES = ("_error", "_refusal")

KernelScore = Union[float, str]


class KernelScoreWithheld(ValueError):
    """The kernel did not give the requested score.

    ``reason`` is the full recorded string. ``errors`` maps each unavailability key
    present to its message and ``refusals`` maps each refusal key present to its
    message; both are empty for the value case. Catch this base class where any
    non-scored outcome is handled the same way.
    """

    def __init__(
        self,
        reason: str,
        errors: Mapping[str, str],
        refusals: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.errors = dict(errors)
        self.refusals = dict(refusals or {})


class KernelScoreUnavailable(KernelScoreWithheld):
    """The kernel withheld the requested score because it could not compute it.

    ``reason`` is ``unavailable_scoring_kernel_error: ...`` naming every typed key the
    composite carries (unavailability keys first decide the class; any refusal key
    present is named too and kept in ``refusals``), or
    ``unavailable_scoring_kernel_value: ...`` when the composite carries no typed key
    but the requested field is absent or non-finite.
    """


class KernelScoreRefused(KernelScoreWithheld):
    """A kernel gate refused the track: a typed, non-scored outcome.

    ``reason`` is ``refused_scoring_kernel_gate: ...`` naming every refusal key
    present; ``refusals`` maps them to their messages and ``errors`` is empty.
    """


class KernelFieldUnrecognized(ValueError):
    """The composite carries a withholding key this reader does not know.

    Raised, never recorded: an unknown ``*_error`` or ``*_refusal`` key means the
    kernel's vocabulary moved past this module, so guessing whether it withholds the
    score would be the silent substitution the typed keys exist to prevent. ``keys``
    lists the unrecognized keys in sorted order.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = sorted(keys)
        super().__init__(
            "unrecognized_scoring_kernel_field: "
            + ", ".join(self.keys)
            + " is not a known unavailability or refusal key; classify it in"
            " sharpearena.kernel_score before reading this composite"
        )


def _is_present(value: Any) -> bool:
    return value not in (None, "")


def kernel_fields(composite: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """``(errors, refusals)``: the unavailability and refusal keys the composite carries.

    A key with an empty or null value carries no claim and is left out. Raises
    :class:`KernelFieldUnrecognized` when any top-level key ending in ``_error`` or
    ``_refusal`` is in neither allowlist, whatever its value.
    """
    unknown = [
        str(key)
        for key in composite
        if str(key).endswith(KERNEL_WITHHOLDING_SUFFIXES)
        and str(key) not in KERNEL_UNAVAILABILITY_KEYS
        and str(key) not in KERNEL_REFUSAL_KEYS
    ]
    if unknown:
        raise KernelFieldUnrecognized(unknown)
    errors = {
        str(key): str(value)
        for key, value in composite.items()
        if str(key) in KERNEL_UNAVAILABILITY_KEYS and _is_present(value)
    }
    refusals = {
        str(key): str(value)
        for key, value in composite.items()
        if str(key) in KERNEL_REFUSAL_KEYS and _is_present(value)
    }
    return errors, refusals


def kernel_errors(composite: Mapping[str, Any]) -> dict[str, str]:
    """The unavailability keys the composite carries, keyed by field name.

    Only :data:`KERNEL_UNAVAILABILITY_KEYS` count; refusals are in
    :func:`kernel_refusals`. Raises :class:`KernelFieldUnrecognized` on an unknown
    ``*_error`` or ``*_refusal`` key.
    """
    return kernel_fields(composite)[0]


def kernel_refusals(composite: Mapping[str, Any]) -> dict[str, str]:
    """The refusal keys the composite carries, keyed by field name.

    Raises :class:`KernelFieldUnrecognized` on an unknown key, as :func:`kernel_errors`.
    """
    return kernel_fields(composite)[1]


def _detail(fields: Mapping[str, str]) -> str:
    return "; ".join(f"{name}: {message}" for name, message in sorted(fields.items()))


def _kernel_value(composite: Mapping[str, Any], key: str) -> float:
    errors, refusals = kernel_fields(composite)
    if errors:
        raise KernelScoreUnavailable(
            f"{UNAVAILABLE_KERNEL_ERROR}: {_detail({**errors, **refusals})}", errors, refusals
        )
    if refusals:
        raise KernelScoreRefused(f"{REFUSED_KERNEL_GATE}: {_detail(refusals)}", {}, refusals)
    value = composite.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KernelScoreUnavailable(
            f"{UNAVAILABLE_KERNEL_VALUE}: {key} is {value!r}, not a number", {}
        )
    number = float(value)
    if not math.isfinite(number):
        raise KernelScoreUnavailable(
            f"{UNAVAILABLE_KERNEL_VALUE}: {key} is {number!r}, not finite", {}
        )
    return number


def kernel_deflated_sharpe(composite: Mapping[str, Any]) -> float:
    """The kernel's ``deflated_sharpe``, or :class:`KernelScoreUnavailable`."""
    return _kernel_value(composite, "deflated_sharpe")


def kernel_psr(composite: Mapping[str, Any]) -> float:
    """The kernel's ``psr``, or :class:`KernelScoreUnavailable` under the same check."""
    return _kernel_value(composite, "psr")


def kernel_score_or_unavailable(
    composite: Mapping[str, Any], key: str = "deflated_sharpe"
) -> KernelScore:
    """The kernel's ``key`` as a float, or the unavailability reason as a string.

    For rows that must be recorded rather than abandoned. The string form is never a
    number, so arithmetic or a sort over it fails loudly instead of ranking the floor.

    A refusal is recorded as its ``refused_scoring_kernel_gate: ...`` reason, so it
    is a non-score like an unavailability but stays distinguishable from one. A
    :class:`KernelFieldUnrecognized` is not recorded: it propagates.
    """
    try:
        return _kernel_value(composite, key)
    except KernelScoreWithheld as withheld:
        return withheld.reason


def is_kernel_score_unavailable(value: Any) -> bool:
    """True when ``value`` is a recorded reason (unavailable or refused), not a score."""
    return isinstance(value, str)


def is_kernel_score_refused(value: Any) -> bool:
    """True when ``value`` is a recorded gate refusal rather than a score."""
    return isinstance(value, str) and value.startswith(f"{REFUSED_KERNEL_GATE}:")


def kernel_withheld_from_reason(reason: str) -> KernelScoreWithheld:
    """The exception a recorded ``reason`` stands for, to re-raise it at a rank input.

    A refusal reason gives :class:`KernelScoreRefused`; every other reason gives
    :class:`KernelScoreUnavailable`. The per-key maps are not recoverable from the
    string and are empty.
    """
    if is_kernel_score_refused(reason):
        return KernelScoreRefused(reason, {})
    return KernelScoreUnavailable(reason, {})


def kernel_score_difference(left: KernelScore, right: KernelScore) -> KernelScore:
    """``left - right`` when both are scores, else the reason that makes it unavailable.

    A gap with an unavailable side is unavailable: ``0.0 - x`` would present the
    withheld side as a scored zero.
    """
    if is_kernel_score_unavailable(left):
        return left
    if is_kernel_score_unavailable(right):
        return right
    return float(left) - float(right)


NO_SEED_OBSERVATIONS = "unavailable: no seed observations"


def seed_pass_tally(composites: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pass^k rate over every evaluated seed, with the counts behind it.

    ``composites`` holds one ``score_run`` composite per seed, in seed order, for every
    seed that was evaluated. A seed whose composite the kernel withheld (an
    unavailability or refusal key, or no finite ``deflated_sharpe``) is *unscorable*:
    it stays in the
    denominator and counts as not passed. Dropping it would let a policy that breaks a
    seed's evaluation raise its own pass rate, because the rate would then be computed
    over the seeds it did not break.

    Returns ``passed_k_rate`` (``n_seeds_passed / n_seeds``) beside ``n_seeds_passed``,
    ``n_seeds_scored`` and ``n_seeds_unscorable``, where
    ``n_seeds_scored + n_seeds_unscorable == len(composites)``. When no seed is
    scorable there is no evidence for a rate at all, so ``passed_k_rate`` is the first
    seed's unavailability reason (or :data:`NO_SEED_OBSERVATIONS` for an empty list),
    never ``0.0``.
    """
    passed = 0
    scored = 0
    first_reason: Optional[str] = None
    # inv: passed <= scored <= composites read so far, and first_reason is the reason
    # of the first unscorable composite read so far, if any.
    for composite in composites:
        score = kernel_score_or_unavailable(composite)
        if is_kernel_score_unavailable(score):
            if first_reason is None:
                first_reason = str(score)
            continue
        scored += 1
        if bool(composite.get("passed_k", False)):
            passed += 1
    n_seeds = len(composites)
    rate: KernelScore
    if scored == 0:
        rate = first_reason if first_reason is not None else NO_SEED_OBSERVATIONS
    else:
        rate = passed / n_seeds
    return {
        "passed_k_rate": rate,
        "n_seeds_passed": passed,
        "n_seeds_scored": scored,
        "n_seeds_unscorable": n_seeds - scored,
    }


__all__ = [
    "KERNEL_REFUSAL_KEYS",
    "KERNEL_UNAVAILABILITY_KEYS",
    "KERNEL_WITHHOLDING_SUFFIXES",
    "KernelFieldUnrecognized",
    "KernelScore",
    "KernelScoreRefused",
    "KernelScoreUnavailable",
    "KernelScoreWithheld",
    "REFUSED_KERNEL_GATE",
    "UNAVAILABLE_KERNEL_ERROR",
    "UNAVAILABLE_KERNEL_VALUE",
    "NO_SEED_OBSERVATIONS",
    "is_kernel_score_refused",
    "is_kernel_score_unavailable",
    "kernel_deflated_sharpe",
    "kernel_errors",
    "kernel_fields",
    "kernel_psr",
    "kernel_refusals",
    "kernel_withheld_from_reason",
    "kernel_score_difference",
    "kernel_score_or_unavailable",
    "seed_pass_tally",
]
