"""Exact SplitMix64 output inversion primitives and their observation boundary.

This module is a research diagnostic for the generator-inversion limitation. It is
not an attack on a deployed evaluation by itself: the scenario interface exposes
rounded prices, not an unrounded SplitMix64 word. The distinction matters.

The SplitMix64 finalizer is a bijection on 64-bit words. If an adversary obtains
one *full* finalizer output, :func:`unmix64` recovers its preimage exactly. The
generator's ``next_unit`` publishes only the upper 53 bits as an IEEE-754 float,
so one exact unit observation leaves 2**11 possible full words/states. Price
generation adds further arithmetic and rounding before an observation crosses the
public interface; matching those candidates through that transform remains the
open analytic inversion problem. This module makes the narrowed 2**11-state
boundary reproducible instead of hand-waving about either impossibility or a
completed break.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MUL1 = 0xBF58476D1CE4E5B9
MUL2 = 0x94D049BB133111EB


def mix64(word: int) -> int:
    """SplitMix64's public 64-bit finalizer."""
    z = int(word) & MASK64
    z = ((z ^ (z >> 30)) * MUL1) & MASK64
    z = ((z ^ (z >> 27)) * MUL2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def _undo_xor_right(value: int, shift: int) -> int:
    """Invert ``x ^ (x >> shift)`` over 64-bit words."""
    out = int(value) & MASK64
    # Each iteration recovers another `shift` high bits. Six is enough for the
    # smallest shift used by SplitMix64 (27), and the loop states the invariant.
    for _ in range((64 + shift - 1) // shift):
        out = int(value) ^ (out >> shift)
    return out & MASK64


_MUL1_INV = pow(MUL1, -1, 1 << 64)
_MUL2_INV = pow(MUL2, -1, 1 << 64)


def unmix64(output: int) -> int:
    """Invert :func:`mix64` exactly for one full 64-bit output word."""
    z = _undo_xor_right(output, 31)
    z = (z * _MUL2_INV) & MASK64
    z = _undo_xor_right(z, 27)
    z = (z * _MUL1_INV) & MASK64
    return _undo_xor_right(z, 30)


def next_word(state: int) -> tuple[int, int]:
    """One SplitMix64 step as ``(new_state, full_output_word)``."""
    new_state = (int(state) + GAMMA) & MASK64
    return new_state, mix64(new_state)


def candidate_states_from_unit(unit: float) -> tuple[int, ...]:
    """Return every state consistent with one published 53-bit ``next_unit``.

    ``next_unit`` serializes ``output >> 11`` as a floating value in ``[0, 1)``.
    Recovering that numerator yields 2**11 possible lower-bit completions. The
    true post-increment state is necessarily one of the returned candidates;
    deciding which one from prices requires modelling the full quantized scenario
    transform and is intentionally outside this narrow primitive.
    """
    if not 0.0 <= float(unit) < 1.0:
        raise ValueError("unit must lie in [0, 1)")
    top53 = int(float(unit) * (1 << 53))
    if not 0 <= top53 < (1 << 53):
        raise ValueError("unit is not a representable SplitMix64 next_unit value")
    return tuple(unmix64((top53 << 11) | low) for low in range(1 << 11))


__all__ = [
    "GAMMA",
    "MASK64",
    "candidate_states_from_unit",
    "mix64",
    "next_word",
    "unmix64",
]
