"""Typed unavailability from the SharpeBench scoring kernel.

``score_run`` returns the kernel's ``CompositeScore`` as JSON. Since SharpeBench
0.19.0 a composite that the kernel could not score carries a typed error beside a
no-skill floor instead of a coerced estimate: ``deflation_error`` (the deflated
Sharpe, its bar and its interval are the floor), ``bootstrap_error`` (the bootstrap
p-value is the conservative 1.0 sentinel) and ``selection_error`` (the selection
diagnostic is absent). A non-finite observation sets both ``deflation_error`` and
``bootstrap_error`` and nulls ``psr``; fewer than two observations set
``bootstrap_error`` alone while ``deflated_sharpe`` reads ``0.0``. Reading
``composite["deflated_sharpe"]`` past such a key publishes the floor as a score,
which is the flattering substitution the kernel refused to make.

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
from typing import Any, Mapping, Union

UNAVAILABLE_KERNEL_ERROR = "unavailable_scoring_kernel_error"
UNAVAILABLE_KERNEL_VALUE = "unavailable_scoring_kernel_value"

KernelScore = Union[float, str]


class KernelScoreUnavailable(ValueError):
    """The kernel withheld the requested score.

    ``reason`` is the full unavailability string (``unavailable_scoring_kernel_error:
    ...`` naming every typed error key the composite carries, or
    ``unavailable_scoring_kernel_value: ...`` when the composite carries no typed error
    but the requested field is absent or non-finite). ``errors`` maps each typed error
    key present to its message; it is empty for the value case.
    """

    def __init__(self, reason: str, errors: Mapping[str, str]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.errors = dict(errors)


def kernel_errors(composite: Mapping[str, Any]) -> dict[str, str]:
    """Every typed error the composite carries, keyed by field name.

    Matches any top-level ``*_error`` key with a non-empty value, so the three keys a
    0.20.0 ``CompositeScore`` can carry (``deflation_error``, ``bootstrap_error``,
    ``selection_error``) and the keys nested reports use elsewhere in SharpeBench
    (``statistics_error``, ``snooping_error``, ``pbo_error``, ``inference_error``)
    are all recognized if a future kernel surfaces them at the top level.
    """
    return {
        str(key): str(value)
        for key, value in composite.items()
        if str(key).endswith("_error") and value not in (None, "")
    }


def _kernel_value(composite: Mapping[str, Any], key: str) -> float:
    errors = kernel_errors(composite)
    if errors:
        detail = "; ".join(f"{name}: {message}" for name, message in sorted(errors.items()))
        raise KernelScoreUnavailable(f"{UNAVAILABLE_KERNEL_ERROR}: {detail}", errors)
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
    """
    try:
        return _kernel_value(composite, key)
    except KernelScoreUnavailable as unavailable:
        return unavailable.reason


def is_kernel_score_unavailable(value: Any) -> bool:
    """True when ``value`` is a recorded unavailability reason rather than a score."""
    return isinstance(value, str)


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


__all__ = [
    "KernelScore",
    "KernelScoreUnavailable",
    "UNAVAILABLE_KERNEL_ERROR",
    "UNAVAILABLE_KERNEL_VALUE",
    "is_kernel_score_unavailable",
    "kernel_deflated_sharpe",
    "kernel_errors",
    "kernel_psr",
    "kernel_score_difference",
    "kernel_score_or_unavailable",
]
