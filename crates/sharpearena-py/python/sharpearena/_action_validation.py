"""Shared target-vector validation before crossing a stateful native boundary."""

import math
from numbers import Real

import numpy as np


def validate_weight_limit(max_weight, allow_short) -> None:
    if (isinstance(max_weight, (bool, np.bool_)) or not isinstance(max_weight, Real)
            or not math.isfinite(max_weight)
            or not 0.0 < max_weight <= np.finfo(np.float32).max):
        raise ValueError("max_weight must be positive, finite and representable in the action space")
    if not isinstance(allow_short, (bool, np.bool_)):
        raise ValueError("allow_short must be a boolean")


def validated_action(action, action_space) -> np.ndarray:
    """Validate exact shape, numeric kind, finiteness and advertised Box bounds.

    Return an owned float64 snapshot. Do not reshape, clip, discard symbols or
    implicitly coerce booleans/strings into trading instructions.
    """
    try:
        values = np.asarray(action)
    except (ValueError, TypeError) as exc:
        raise ValueError("action must be a rectangular numeric array") from exc
    if values.shape != action_space.shape:
        raise ValueError(f"action shape must be {action_space.shape}, got {values.shape}")
    if values.dtype.kind not in "fiu":
        raise ValueError("action entries must be real numbers, not booleans or strings")
    values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("action entries must be finite")
    if ((values < action_space.low) | (values > action_space.high)).any():
        raise ValueError("action exceeds the configured action-space bounds")
    return values
