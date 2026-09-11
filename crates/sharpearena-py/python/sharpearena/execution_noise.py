"""Execution-noise wrapper — the trading analog of ALE sticky actions.

Open-loop trajectory memorization is the vector-env analog of an Atari agent learning a
fixed button sequence: a policy can overfit a deterministic point-in-time market by
replaying a memorized action path. :class:`ExecutionNoiseWrapper` perturbs the *realized*
action relative to the *requested* one — repeating the previous action one bar late (the
delay/"sticky" knob) and/or jittering the target weights (the slippage knob) — which breaks
that open-loop replay while leaving a closed-loop policy that reacts to observations intact.

The determinism-sensitive perturbation math lives in Rust (``sharpearena::exec_noise``) and
is reached here via the :func:`perturb_action` binding, so a given ``(seed, step_index,
delay_prob, slippage_bps)`` produces a **byte-reproducible** perturbation from any surface —
not just this Python wrapper. Both knobs default to zero (pass-through, no draws).

The Rust core draws a **bounded uniform** jitter in ``[-1, 1)`` rather than the Gaussian this
wrapper used previously: Gaussian sampling routes through ``ln``/``sqrt`` transforms whose
last bits differ across libm implementations, so a Gaussian stream is not byte-identical
across Rust / WASM / Python. The uniform draw uses only mul/add and stays cross-runtime
reproducible; the jitter magnitude is capped at ``|weight| * slippage_bps / 10_000``.

This is a **reportable benchmark-integrity knob**: any eval run that enables a non-zero
``delay_prob`` or ``slippage_bps`` MUST disclose those values alongside its scores, since
they change the difficulty of the task.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np
import gymnasium as gym

from .sharpearena_py import perturb_action

_U64_MASK = (1 << 64) - 1


def validate_execution_noise(delay_prob: float, slippage_bps: float) -> None:
    """Refuse an out-of-range execution-noise knob at the surface that accepted it.

    These are disclosed benchmark-integrity settings, so each out-of-range shape is a run
    whose disclosure is false rather than a run that fails: a negative knob takes the Rust
    core's "no knobs configured" pass-through, so a run reporting ``delay_prob=-0.1`` was
    noise-free; ``delay_prob > 1`` makes every step replay the previous action, so the
    agent's decisions never reach the market; and NaN passes every guard, so the realized
    action is NaN and :func:`numpy.clip` propagates it into ``env.step``. NaN also reads as
    *enabled* by an ``!= 0.0`` test.

    The Rust core refuses the same ranges, so this is a second copy of one rule rather
    than the only copy; it exists because these values are accepted here, in a dataclass
    and a wrapper constructor, long before any action reaches the core. Written as an
    ``if`` and a ``raise`` so the refusal survives ``python -O``.
    """
    if not math.isfinite(delay_prob) or not 0.0 <= delay_prob <= 1.0:
        raise ValueError(
            f"[INVALID_ARGUMENT] delay_prob {delay_prob} is outside the finite range [0, 1]"
        )
    if not math.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise ValueError(
            f"[INVALID_ARGUMENT] slippage_bps {slippage_bps} is negative or not finite"
        )


class ExecutionNoiseWrapper(gym.Wrapper):
    """Perturb the realized action vs. the requested one (seeded, default-off).

    With probability ``delay_prob`` the previous step's realized action is applied instead
    of the requested one (the order lands one bar late). Independently, bounded
    multiplicative jitter with scale ``slippage_bps`` basis points is applied to the
    (post-delay) target weights. The result is clipped back into the env's action space.
    With both knobs at their ``0.0`` default the requested action passes through unchanged
    and no random draws are taken.

    The perturbation math is the Rust ``sharpearena::exec_noise`` core (via
    :func:`perturb_action`); the jitter is a bounded uniform in ``[-1, 1)`` (see the module
    docstring for why this replaces the historical Gaussian draw).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        delay_prob: float = 0.0,
        slippage_bps: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self._delay_prob = float(delay_prob)
        self._slippage_bps = float(slippage_bps)
        validate_execution_noise(self._delay_prob, self._slippage_bps)
        # A concrete u64 stream seed: explicit when given, otherwise a fixed-per-instance
        # draw (mirrors ``np.random.default_rng(None)`` — nondeterministic across runs but
        # stable within this wrapper instance).
        if seed is None:
            self._seed = int.from_bytes(os.urandom(8), "little")
        else:
            self._seed = int(seed) & _U64_MASK
        self._step_index = 0
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any):
        self._step_index = 0
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        requested = np.asarray(action, dtype=np.float64)
        shape = requested.shape
        # First step has no realized predecessor; passing the request itself makes the
        # delay/sticky branch a no-op (the Rust core returns `previous` when delay fires).
        previous = self._prev_action if self._prev_action is not None else requested

        realized = np.asarray(
            perturb_action(
                self._seed,
                self._step_index,
                requested.reshape(-1).tolist(),
                previous.reshape(-1).tolist(),
                self._delay_prob,
                self._slippage_bps,
            ),
            dtype=np.float64,
        ).reshape(shape)

        space = self.env.action_space
        realized = np.clip(realized, space.low, space.high).astype(space.dtype)
        self._prev_action = realized.astype(np.float64).copy()
        self._step_index += 1
        return self.env.step(realized)


__all__ = ["ExecutionNoiseWrapper", "validate_execution_noise"]
